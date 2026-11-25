"""Pilots 003/004: signed neuron readouts and causal, gradient-free JSON repair."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import random

from .benchmark import Task, make_tasks, paired_bootstrap, summarize
from .runtime import MODEL, evaluate, generate_one, load_model

REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
FENCE = chr(96) * 3


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def matched_prompts(split, count, seed):
    raw = [t for t in make_tasks(split, count, seed) if t.family == "json"]
    fenced = [
        Task(t.id + "-fenced", "fenced_json",
             t.prompt.replace("No Markdown or extra text.",
                              "Wrap the JSON in a Markdown code fence with language json. No text outside the fence."),
             t.expected)
        for t in raw
    ]
    return raw, fenced


def fenced_correct(task, response):
    from .benchmark import is_correct
    s = response.strip().splitlines()
    return (len(s) >= 3 and s[0].strip() == FENCE + "json"
            and s[-1].strip() == FENCE
            and is_correct(Task(task.id, "json", task.prompt, task.expected),
                           "\n".join(s[1:-1])))


def fenced_eval(model, tokenizer, tasks, max_tokens):
    rows = []
    for t in tasks:
        result = generate_one(model, tokenizer, t, max_tokens)
        rows.append({"id": t.id, "prompt": t.prompt, "expected": t.expected,
                     "response": result["response"], "correct": fenced_correct(t, result["response"])})
    return rows


@contextmanager
def scaled_columns(model, edits):
    """Apply a reversible permanent-weight intervention, not a decoding hook."""
    import torch
    backup = []
    try:
        with torch.no_grad():
            for layer, channel, scale in edits:
                w = model.model.layers[layer].mlp.down_proj.weight
                backup.append((w, channel, w[:, channel].clone()))
                w[:, channel].mul_(scale)
        yield
    finally:
        with torch.no_grad():
            for w, channel, original in reversed(backup):
                w[:, channel].copy_(original)


def chat_ids(model, tokenizer, task):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": task.prompt}], add_generation_prompt=True,
        return_tensors="pt").to(next(model.parameters()).device)


def first_step(model, tokenizer, tasks, layers=()):
    """Capture signed MLP activations at the actual first generation position."""
    import torch
    snapshots = {l: [] for l in layers}
    hooks = []
    for l in layers:
        def capture(module, inputs, layer=l):
            snapshots[layer].append(inputs[0][0, -1].detach().float().cpu())
        hooks.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(capture))
    logits = []
    try:
        with torch.inference_mode():
            for task in tasks:
                x = chat_ids(model, tokenizer, task)
                # Base decoder plus last-token unembedding: avoids materializing
                # full sequence x vocabulary logits on CPU.
                h = model.model(input_ids=x, use_cache=False).last_hidden_state[:, -1]
                logits.append(model.lm_head(h)[0].float().cpu())
    finally:
        for hook in hooks:
            hook.remove()
    return torch.stack(logits), {l: torch.stack(v) for l, v in snapshots.items()}


def margin(logits, fence, brace):
    return logits[:, fence] - logits[:, brace]


def signed_readouts(unembedding, columns, k=8, chunk=8192):
    """Cosine vocabulary readouts for both signs; deliberately NOT the J-lens."""
    import torch
    import torch.nn.functional as F
    columns = F.normalize(columns.float(), dim=0)
    scores = torch.cat([
        F.normalize(unembedding[i:i + chunk].float(), dim=1) @ columns
        for i in range(0, len(unembedding), chunk)
    ], dim=0)
    pos = scores.topk(k, dim=0)
    neg = (-scores).topk(k, dim=0)
    return pos.indices.T.tolist(), neg.indices.T.tolist()


def candidate_rows(model, tokenizer, layers, snapshots, raw_count, fence, brace, per_layer):
    import torch
    import torch.nn.functional as F
    u = model.lm_head.weight.detach()
    token_delta = (u[fence] - u[brace]).float().cpu()
    rows = []
    for l in layers:
        w = model.model.layers[l].mlp.down_proj.weight.detach().float().cpu()
        acts = snapshots[l]
        a_raw = acts[:raw_count].mean(0)
        a_fenced = acts[raw_count:].mean(0)
        # Predicted effect of removing each neuron on fence-vs-brace margin.
        # Positive differential favors raw-only suppression.
        readout = token_delta @ w
        contrast = (a_raw - a_fenced) * readout
        indices = contrast.topk(per_layer).indices.tolist()
        pos, neg = signed_readouts(u, w[:, indices].to(u.device))
        for i, n in enumerate(indices):
            rows.append({"layer": l, "channel": n, "rank_score": float(contrast[n]),
                         "activation_raw": float(a_raw[n]), "activation_fenced": float(a_fenced[n]),
                         "activation_raw_per_prompt": acts[:raw_count, n].tolist(),
                         "activation_fenced_per_prompt": acts[raw_count:, n].tolist(),
                         "positive_tokens": [tokenizer.decode([t]) for t in pos[i]],
                         "negative_tokens": [tokenizer.decode([t]) for t in neg[i]],
                         "positive_token_ids": pos[i], "negative_token_ids": neg[i],
                         "column_norm": float(w[:, n].norm())})
    return sorted(rows, key=lambda r: r["rank_score"], reverse=True)


def choose_causal(rows, budget):
    """Positive raw margin reduction, penalizing harm to required fences."""
    return sorted([r for r in rows if r["raw_reduction"] > 0 and r["causal_score"] > 0],
                  key=lambda r: r["causal_score"], reverse=True)[:budget]


def run(args):
    import torch
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    model, tokenizer, environment = load_model(MODEL, args.device, REVISION)
    raw, fenced = matched_prompts("practice", args.practice, args.seed)
    practice = raw + fenced
    fences = tokenizer.encode(FENCE, add_special_tokens=False)
    braces = tokenizer.encode("{", add_special_tokens=False)
    if len(fences) != 1 or len(braces) != 1:
        raise ValueError("Expected single-token fence and opening brace")
    fence, brace = fences[0], braces[0]
    layers = [int(x) for x in args.layers.split(",")]
    baseline, activations = first_step(model, tokenizer, practice, layers)
    base_margin = margin(baseline, fence, brace)
    rows = candidate_rows(model, tokenizer, layers, activations, len(raw),
                          fence, brace, args.per_layer)
    # Pilot 003: record the lens hypotheses before the causal experiment.
    save(out / "atlas.json", {"environment": environment, "lens": "signed cosine logit lens",
        "practice_prompts": [t.asdict() for t in practice],
        "fence_token": fence, "brace_token": brace, "neurons": rows})
    # Actual downstream causal test, not a first-order prediction.
    for r in rows:
        with scaled_columns(model, [(r["layer"], r["channel"], 0.0)]):
            ablated, _ = first_step(model, tokenizer, practice)
        reduction = base_margin - margin(ablated, fence, brace)
        r["raw_reduction"] = float(reduction[:len(raw)].mean())
        r["fenced_reduction"] = float(reduction[len(raw):].mean())
        r["causal_score"] = r["raw_reduction"] - max(0., r["fenced_reduction"])
        # First-token finite difference for the actual signed activation.
        diff = (baseline - ablated)[:len(raw)].mean(0)
        r["causal_promoted_tokens"] = [tokenizer.decode([i]) for i in diff.topk(8).indices.tolist()]
        r["causal_suppressed_tokens"] = [tokenizer.decode([i]) for i in (-diff).topk(8).indices.tolist()]
    save(out / "causal-atlas.json", rows)
    selected = choose_causal(rows, args.budget)
    directed = [(r["layer"], r["channel"], 0.0) for r in selected]
    # Equal count, same per-layer budget: activation-only and random controls.
    counts = {}
    for l, _, _ in directed:
        counts[l] = counts.get(l, 0) + 1
    rng = random.Random(args.seed + 101)
    activation_only, random_control = [], []
    for l, n in sorted(counts.items()):
        a = activations[l]
        contrast = (a[:len(raw)].mean(0) - a[len(raw):].mean(0)).abs()
        activation_only.extend((l, i, 0.) for i in contrast.topk(n).indices.tolist())
        pool = [i for i in range(a.shape[1]) if (l, i, 0.) not in directed]
        random_control.extend((l, i, 0.) for i in rng.sample(pool, n))
    variants = {"base": [], "causal_edit": directed,
                "activation_only": activation_only, "random": random_control}
    save(out / "locked-patch.json", {"variants": variants, "selection": "practice-only causal heuristic",
        "seed": args.seed, "heldout_seed": args.heldout_seed,
        "actual_changed_parameters": {
            name: sum(int(torch.count_nonzero(model.model.layers[l].mlp.down_proj.weight[:, n]).item())
                      for l, n, _ in edits) for name, edits in variants.items()},
        "no_positive_candidates": not bool(directed)})
    # Pilot 004: generate only AFTER locking all patches.
    heldout = make_tasks("heldout", args.heldout, args.heldout_seed)
    _, required_fences = matched_prompts("regression", args.heldout, args.heldout_seed)
    examples, regressions = {}, {}
    for name, edits in variants.items():
        with scaled_columns(model, edits):
            examples[name] = evaluate(model, tokenizer, heldout, args.max_tokens)
            regressions[name] = fenced_eval(model, tokenizer, required_fences, args.max_tokens)
    result = {"environment": environment, "variants": variants,
              "candidate_count": len(rows), "selected_count": len(directed),
              "heldout": {k: summarize(v) for k, v in examples.items()},
              "paired": {k: paired_bootstrap(examples["base"], v)
                         for k, v in examples.items() if k != "base"},
              "required_fence_accuracy": {k: sum(r["correct"] for r in v)/len(v)
                                          for k, v in regressions.items()},
              "required_fence_paired": {k: paired_bootstrap(regressions["base"], v)
                                       for k, v in regressions.items() if k != "base"},
              "caveat": "Small pilot; no LLM-selected patch; signed logit lens, not Anthropic J-lens."}
    save(out / "examples.json", examples)
    save(out / "required-fences.json", regressions)
    save(out / "summary.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "auto"])
    p.add_argument("--layers", default="12,16,20,23")
    p.add_argument("--per-layer", type=int, default=4)
    p.add_argument("--budget", type=int, default=4)
    p.add_argument("--practice", type=int, default=6)
    p.add_argument("--heldout", type=int, default=12)
    p.add_argument("--seed", type=int, default=3203)
    p.add_argument("--heldout-seed", type=int, default=84031)
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", default="artifacts/neuron-basis-003-004")
    args = p.parse_args()
    if min(args.per_layer, args.budget, args.practice, args.heldout, args.max_tokens, args.threads) < 1:
        p.error("All counts must be positive")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
