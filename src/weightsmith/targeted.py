"""GPT-6-proposed gradient-free output-weight repair for unwanted JSON code fences.

Uses only practice failures to identify a fence token and hidden direction.
Directly edits one pretrained output weight row, evaluates fresh heldout tasks,
and restores weights between controls. No training or logits processors.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random

from .benchmark import make_tasks, paired_bootstrap, summarize
from .experiment import write_json
from .runtime import MODEL, evaluate, load_model


def choose_fence_token(tokenizer, failures):
    starts = []
    for row in failures:
        response = row["response"].lstrip()
        if response.startswith(chr(96) * 3):
            starts.append(tokenizer.encode(response, add_special_tokens=False)[0])
    if not starts:
        raise ValueError("No fenced practice JSON failure; abort before heldout")
    token_id = Counter(starts).most_common(1)[0][0]
    token_text = tokenizer.decode([token_id])
    if chr(96) not in token_text:
        raise ValueError(f"Leading token is not a code fence: {token_text!r}")
    return token_id, token_text


def practice_direction(model, tokenizer, practice):
    import torch
    vectors = []
    for row in practice:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            add_generation_prompt=True, return_tensors="pt"
        ).to(next(model.parameters()).device)
        with torch.inference_mode():
            result = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        vectors.append(result.hidden_states[-1][0, -1].float().cpu())
    return torch.stack(vectors).mean(dim=0)


@contextmanager
def patched_output_row(model, token_id, delta):
    import torch
    weights = model.get_output_embeddings().weight
    with torch.no_grad():
        original = weights[token_id].clone()
        try:
            updated = (original.float() - delta.to(original.device)).to(original.dtype)
            weights[token_id].copy_(updated)
            yield int(torch.count_nonzero(original != updated).item())
        finally:
            weights[token_id].copy_(original)


def run(args):
    import torch
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Run directory already contains outputs: {out}")
    model, tokenizer, environment = load_model(args.checkpoint, args.device, args.revision)
    out.mkdir(parents=True, exist_ok=True)

    practice_tasks = make_tasks("practice", args.practice_per_family, args.seed)
    practice = evaluate(model, tokenizer, practice_tasks, args.max_new_tokens)
    write_json(out / "practice.json", practice)
    failures = [
        r for r in practice if r["family"] == "json" and not r["correct"]
        and r["response"].lstrip().startswith(chr(96) * 3)
    ]
    token_id, token_text = choose_fence_token(tokenizer, failures)
    hidden = practice_direction(model, tokenizer, failures)
    # On the average practice representation, this reduces the chosen token
    # logit by the precommitted margin; no gradients or reward optimization.
    delta = (args.logit_margin / (float(hidden @ hidden) + 1e-9)) * hidden
    if float(delta.norm()) > args.max_delta_norm:
        raise ValueError("Edit norm exceeds the precommitted safety cap")
    weights = model.get_output_embeddings().weight
    if hidden.numel() != weights.shape[1]:
        raise ValueError("Unexpected output projection dimensions")

    gen = torch.Generator(device="cpu").manual_seed(args.seed + 321)
    random_vector = torch.randn(hidden.shape, generator=gen)
    random_vector -= ((random_vector @ hidden) / (hidden @ hidden)) * hidden
    random_vector *= delta.norm() / (random_vector.norm() + 1e-9)
    random_row = random.Random(args.seed + 889).choice([
        i for i in range(len(weights))
        if i != token_id and i != tokenizer.eos_token_id
    ])

    write_json(out / "locked-patch.json", {
        "editor": "GPT-6 / ChatGPT, human-in-the-loop research proposal",
        "checkpoint": environment,
        "practice_seed": args.seed, "fresh_heldout_seed": args.heldout_seed,
        "target_token_id": token_id, "token_text": token_text,
        "logit_margin": args.logit_margin, "delta_l2": float(delta.norm()),
        "delta_sha256": hashlib.sha256(delta.numpy().tobytes()).hexdigest(),
        "delta_vector": delta.tolist(),
        "practice_failure_ids": [r["id"] for r in failures],
        "random_control_token_id": random_row,
        "heldout_seen_before_patch": False,
    })

    results = {}
    # This heldout seed differs from the already inspected MLP pilot.
    for split, count in (("heldout", args.heldout_per_family),
                         ("regression", args.regression_per_family)):
        tasks = make_tasks(split, count, args.heldout_seed)
        predictions = {"base": evaluate(model, tokenizer, tasks, args.max_new_tokens)}
        changed_counts = {}
        for name, target_row, vector in (
            ("directed_edit", token_id, delta),
            ("random_direction", token_id, random_vector),
            ("random_row", random_row, delta),
        ):
            with patched_output_row(model, target_row, vector) as changed:
                predictions[name] = evaluate(model, tokenizer, tasks, args.max_new_tokens)
            changed_counts[name] = changed
        base = predictions["base"]
        results[split] = {
            "summary": {k: summarize(v) for k, v in predictions.items()},
            "paired": {k: paired_bootstrap(base, v) for k, v in predictions.items()
                       if k != "base"},
            "changed_parameters": changed_counts,
            "examples": predictions,
        }
        write_json(out / (split + ".json"), results[split])

    summary = {
        "environment": environment,
        "editor": "GPT-6 human-in-the-loop intervention",
        "total_model_parameters": sum(p.numel() for p in model.parameters()),
        "edited_token_id": token_id, "edited_token_text": token_text,
        "heldout": results["heldout"]["summary"],
        "heldout_paired": results["heldout"]["paired"],
        "changed_parameters": results["heldout"]["changed_parameters"],
        "regression": results["regression"]["summary"],
        "caveat": "Narrow output-token formatting edit, not general reasoning improvement",
    }
    write_json(out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=MODEL)
    parser.add_argument("--revision", default="7ae557604adf67be50417f59c2c2f167def9a775")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--logit-margin", type=float, default=6.0)
    parser.add_argument("--max-delta-norm", type=float, default=2.0)
    parser.add_argument("--practice-per-family", type=int, default=2)
    parser.add_argument("--heldout-per-family", type=int, default=8)
    parser.add_argument("--regression-per-family", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--heldout-seed", type=int, default=7802)
    parser.add_argument("--out", default="artifacts/qwen-output-edit-001")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
