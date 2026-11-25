"""Pilots 005/006: closed-set synthetic weight auditing and few-shot neuron probes."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import random
import re

from .benchmark import Task
from .neuron_basis import REVISION, first_step, save, signed_readouts
from .runtime import MODEL, load_model

# The selection and evaluation phrases are disjoint. This deliberately small,
# hand-authored dataset is a feasibility check, not a general semantic benchmark.
CONCEPTS = {
    "arithmetic": {
        "train": ["Add seventeen and thirty.", "Multiply eight by six.", "Divide 144 by twelve.",
                  "Subtract three from fourteen.", "What is the remainder of twenty divided by seven?",
                  "Find the product of nine and eleven."],
        "test": ["Compute 35 plus 72.", "How many items remain after taking five from nineteen?",
                 "Calculate 16 times 4.", "What number is one quarter of 80?",
                 "Determine the sum of 27 and 18.", "Evaluate 98 minus 49.",
                 "Split ninety candies equally among six children.", "How much is twice forty-three?"],
        "hard": ["The Sum is the name of a jazz group.", "The product launch was postponed.",
                 "The Times published a book review.", "The actor played Number Six."],
        "keywords": ["add", "multiply", "divide", "subtract", "sum", "product", "number",
                     "compute", "calculate", "times", "minus", "remainder"],
        "patch": [" algebra", " sum", " math"],
        "audit": [" equation", " calculate", " number", " arithmetic"],
    },
    "french": {
        "train": ["Bonjour, comment allez-vous ?", "Je voudrais un café, s'il vous plaît.",
                  "Nous allons au marché ce matin.", "Il fait beau aujourd'hui.",
                  "Où se trouve la gare ?", "Elle lit un livre dans le jardin."],
        "test": ["Merci pour votre aide.", "Les enfants jouent dehors.",
                 "J'aime beaucoup cette chanson.", "Pouvez-vous fermer la fenêtre ?",
                 "Le train arrive dans dix minutes.", "Nous habitons près de la mer.",
                 "Cette histoire est très intéressante.", "Je ne comprends pas la question."],
        "hard": ["The French doors were painted white.", "French fries are served with lunch.",
                 "Bonjour is the name of our new app.", "The Merci brand makes chocolates."],
        "keywords": ["bonjour", "merci", "french", "je", "nous", "vous", "les", "dans",
                     "elle", "aujourd'hui", "cette", "est"],
        "patch": [" Paris", " French", " France"],
        "audit": [" français", " bonjour", " merci", " avec"],
    },
    "cooking": {
        "train": ["Bake the bread until golden.", "Whisk eggs and milk in a bowl.",
                  "Saute onions in olive oil.", "Roast vegetables at high heat.",
                  "Boil the pasta in salted water.", "Fold flour into the batter."],
        "test": ["Simmer the tomato sauce gently.", "Chop the garlic and stir it into the pan.",
                 "Season the soup with fresh herbs.", "Knead the dough for five minutes.",
                 "Toast spices before adding the rice.", "Grill the peppers until charred.",
                 "Melt butter and mix in sugar.", "Steam the dumplings until cooked."],
        "hard": ["The software is still in the oven, figuratively.", "He stirred up a political debate.",
                 "The photographer used a bowl-shaped reflector.", "Steam powers the old locomotive."],
        "keywords": ["bake", "whisk", "saute", "roast", "boil", "simmer", "chop",
                     "season", "knead", "grill", "steam", "butter", "pan", "soup"],
        "patch": [" cooking", " food", " chef"],
        "audit": [" recipe", " kitchen", " ingredients", " bake"],
    },
}
CALIBRATION = ["Here is one possible answer:", "The topic for today is", "I would like to discuss",
               "An example of this is", "One useful thing to learn is"]
AUDIT_PROMPTS = ["The next subject is", "Let us begin with", "The first idea is",
                 "A useful example might be", "We could also consider", "What comes to mind is"]
VALIDATION_PROMPTS = ["I have been thinking about", "Another interesting subject is",
                      "The most important topic is", "A good starting point would be"]


def auc(labels, scores):
    positives = [s for y, s in zip(labels, scores) if y]
    negatives = [s for y, s in zip(labels, scores) if not y]
    if not positives or not negatives:
        raise ValueError("AUC needs both classes")
    return sum(float(a > b) + .5 * float(a == b) for a in positives for b in negatives) / (
        len(positives) * len(negatives))


def lexical(text, words):
    return sum(bool(re.search(r"\b" + re.escape(w) + r"\b", text.lower())) for w in words)


def feature_batch(model, tokenizer, texts, layers):
    import torch
    captured = {l: [] for l in layers}
    hidden, hooks = [], []
    for l in layers:
        def capture(module, inputs, layer=l):
            captured[layer].append(inputs[0][0].detach().float().mean(dim=0).cpu())
        hooks.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(capture))
    try:
        with torch.inference_mode():
            for text in texts:
                ids = tokenizer(text, return_tensors="pt").input_ids.to(next(model.parameters()).device)
                out = model.model(input_ids=ids, use_cache=False).last_hidden_state
                hidden.append(out[0].float().mean(0).cpu())
    finally:
        for hook in hooks:
            hook.remove()
    return torch.cat([torch.stack(captured[l]) for l in layers], dim=1), torch.stack(hidden)


def detector(model, tokenizer, out, layers, k=8):
    import torch
    import torch.nn.functional as F
    names = list(CONCEPTS)
    train = [(name, t) for name in names for t in CONCEPTS[name]["train"]]
    # Include topic-confusable examples only in the evaluation split.
    test = [(name, t) for name in names for t in CONCEPTS[name]["test"]]
    test += [("none", t) for name in names for t in CONCEPTS[name]["hard"]]
    a, h = feature_batch(model, tokenizer, [t for _, t in train], layers)
    mean, sd = a.mean(0), a.std(0, unbiased=False).clamp_min(1e-4)
    z = (a - mean) / sd
    h = F.normalize(h, dim=1)
    selections = {}
    width = model.model.layers[layers[0]].mlp.down_proj.in_features
    for name in names:
        y = torch.tensor([label == name for label, _ in train])
        contrast = z[y].mean(0) - z[~y].mean(0)
        idx = contrast.abs().topk(k).indices
        selections[name] = {"indices": idx.tolist(), "signs": contrast[idx].sign().tolist(),
            "neurons": [{"layer": layers[i // width], "channel": i % width,
                         "sign": float(contrast[i].sign())} for i in idx.tolist()]}
    save(out / "probe-locked.json", {"train": train, "selection": selections,
                                    "note": "Few-shot labeled selection, not zero-shot"})
    # No evaluation inputs are processed before selection is saved.
    a_test, h_test = feature_batch(model, tokenizer, [t for _, t in test], layers)
    z_test = (a_test - mean) / sd
    h_test = F.normalize(h_test, dim=1)
    metrics, predictions = {}, {}
    for name in names:
        y = torch.tensor([label == name for label, _ in train])
        labels = [label == name for label, _ in test]
        sel = selections[name]
        idx = torch.tensor(sel["indices"])
        signs = torch.tensor(sel["signs"])
        native = (z_test[:, idx] * signs).mean(1)
        centroid = F.normalize((h[y].mean(0) - h[~y].mean(0)), dim=0)
        embedding = h_test @ centroid
        # Supervised ridge baseline, same training examples and fixed regularization.
        target = y.float() * 2 - 1
        alpha = torch.linalg.solve(h @ h.T + .1 * torch.eye(len(h)), target)
        ridge = h_test @ (h.T @ alpha)
        scores = {"native_neurons": native.tolist(), "keyword": [
            lexical(t, CONCEPTS[name]["keywords"]) for _, t in test],
            "hidden_centroid": embedding.tolist(), "supervised_ridge": ridge.tolist()}
        metrics[name] = {method: auc(labels, values) for method, values in scores.items()}
        predictions[name] = [{"text": t, "label": int(label),
                              **{m: v[i] for m, v in scores.items()}}
                             for i, ((_, t), label) in enumerate(zip(test, labels))]
    macro = {m: sum(metrics[n][m] for n in names)/len(names) for m in metrics[names[0]]}
    result = {"per_concept_auroc": metrics, "macro_auroc": macro,
              "train_n": len(train), "test_n": len(test),
              "caveat": "Small hand-authored few-shot concept classification; not Anthropic's zero-label benchmark."}
    save(out / "probe-predictions.json", predictions)
    save(out / "probe-summary.json", result)
    return result


def token_set(tokenizer, words):
    # Single-token words only; no silent truncation of multi-token exemplars.
    ids = []
    for w in words:
        enc = tokenizer.encode(w, add_special_tokens=False)
        if len(enc) == 1:
            ids.append(enc[0])
    if not ids:
        raise ValueError(f"No single-token exemplars: {words}")
    return sorted(set(ids))


def tasks(texts, name):
    return [Task(f"{name}-{i}", "text", t, "") for i, t in enumerate(texts)]


@contextmanager
def add_patch(weight, columns, delta):
    import torch
    with torch.no_grad():
        original = weight[:, columns].clone()
        try:
            weight[:, columns] = original + delta.to(original)
            yield
        finally:
            weight[:, columns] = original


def category_scores(delta_logits, groups):
    return {name: float(delta_logits[:, ids].mean()) for name, ids in groups.items()}


def predict(scores, threshold):
    name = max(scores, key=scores.get)
    return name if scores[name] > threshold else "none"


def audit(model, tokenizer, out, layer, seed):
    import torch
    import torch.nn.functional as F
    rng = random.Random(seed)
    u = model.lm_head.weight.detach().float().cpu()
    w = model.model.layers[layer].mlp.down_proj.weight
    patch_ids = {n: token_set(tokenizer, d["patch"]) for n, d in CONCEPTS.items()}
    audit_ids = {n: token_set(tokenizer, d["audit"]) for n, d in CONCEPTS.items()}
    if set(sum(patch_ids.values(), [])) & set(sum(audit_ids.values(), [])):
        raise ValueError("Patch and auditor exemplars must be disjoint")
    _, activation = first_step(model, tokenizer, tasks(CALIBRATION, "calibration"), [layer])
    mean = activation[layer].mean(0)
    candidates = mean.abs().topk(12).indices.tolist()
    base_audit, _ = first_step(model, tokenizer, tasks(AUDIT_PROMPTS, "audit"))
    base_val, _ = first_step(model, tokenizer, tasks(VALIDATION_PROMPTS, "validation"))
    # Six explicit benign direction patches and two unmodified null controls.
    # No claim that these are learned behaviors or genuine LoRA adapters.
    cases = [(n, rep) for n in CONCEPTS for rep in range(2)] + [("none", 0), ("none", 1)]
    rng.shuffle(cases)
    packets, truth, metrics = [], [], []
    for i, (name, rep) in enumerate(cases):
        columns = candidates[rep * 4:rep * 4 + 4]
        original = w[:, columns].detach().float().cpu()
        if name == "none":
            delta = torch.zeros_like(original)
        else:
            direction = F.normalize(u[patch_ids[name]].mean(0), dim=0)
            sign = mean[columns].sign()
            delta = direction[:, None] * (original.norm(dim=0) * .5 * sign)[None, :]
        # Ground truth is written separately from packets. Auditor has only a
        # closed set of concept names, candidate vocabulary, outputs and/or readouts.
        with add_patch(w, columns, delta):
            changed_audit, _ = first_step(model, tokenizer, tasks(AUDIT_PROMPTS, "audit"))
            changed_val, _ = first_step(model, tokenizer, tasks(VALIDATION_PROMPTS, "validation"))
        output_scores = category_scores(changed_audit - base_audit, audit_ids)
        val_scores = category_scores(changed_val - base_val, audit_ids)
        # Align columns by independent calibration activation; aggregate readout.
        oriented = delta * mean[columns].sign()[None, :]
        aggregate = oriented.mean(1)
        if aggregate.norm() > 0:
            cos = F.normalize(u, dim=1) @ F.normalize(aggregate, dim=0)
            weight_scores = {n: float(cos[ids].mean()) for n, ids in audit_ids.items()}
            top_ids = cos.topk(12).indices.tolist()
        else:
            weight_scores = {n: 0. for n in CONCEPTS}
            top_ids = []
        output_guess = predict(output_scores, .01)
        weight_guess = predict(weight_scores, .01)
        # Weight-assisted score is intentionally a simple precommitted baseline,
        # not an LLM auditor. Both modes have the same closed-set labels.
        combined_guess = weight_guess if weight_guess != "none" else output_guess
        case = f"case-{i:02d}"
        packets.append({"id": case, "output_only": {
            "prompts": AUDIT_PROMPTS, "category_logit_changes": output_scores,
            "top_changed_tokens": [tokenizer.decode([t]) for t in
                (changed_audit-base_audit).mean(0).topk(12).indices.tolist()]},
            "weight_evidence": {"layer": layer, "columns": columns,
                                "readout_tokens": [tokenizer.decode([t]) for t in top_ids],
                                "category_cosines": weight_scores}})
        truth.append({"id": case, "intended_concept": name, "validation_scores": val_scores,
                      "validated": predict(val_scores, .01) == name})
        metrics.append({"id": case, "output_only": output_guess, "weight_only": weight_guess,
                        "weight_assisted": combined_guess})
    save(out / "audit-blind-packets.json", packets)
    save(out / "audit-predictions.json", metrics)
    save(out / "audit-ground-truth.json", truth)
    valid = [i for i, t in enumerate(truth) if t["validated"]]
    def accuracy(indices):
        return {method: (sum(metrics[i][method] == truth[i]["intended_concept"] for i in indices) /
                         len(indices) if indices else None)
                for method in ("output_only", "weight_only", "weight_assisted")}
    result = {"cases": len(cases), "validated_cases": len(valid),
              "all_case_accuracy": accuracy(range(len(cases))),
              "validated_case_accuracy": accuracy(valid),
              "caveat": "Closed-set synthetic direction-patch audit; validation is independent next-token logit shift, not verified free-generation behavior or a blind LLM audit."}
    save(out / "audit-summary.json", result)
    return result


def main():
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="artifacts/neuron-applications-005-006")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=6505)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    model, tokenizer, env = load_model(MODEL, args.device, REVISION)
    save(out / "environment.json", env)
    results = {"probe": detector(model, tokenizer, out, [12, 16, 20, 23]),
               "audit": audit(model, tokenizer, out, 23, args.seed)}
    save(out / "summary.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
