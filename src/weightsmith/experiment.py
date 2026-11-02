"""Run a sealed, gradient-free LLM-to-LLM weight-editing experiment.

The editor may only see PRACTICE prompts. Heldout and regression task lists are
generated and scored after the validated proposal has been saved to disk.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from .benchmark import make_tasks, paired_bootstrap, summarize
from .diagnose import candidate_report
from .editor import choose_proposal
from .patch import Edit, matched_random, temporary_patch, validate_edits
from .runtime import MODEL, evaluate, load_model


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def matched_heuristic(edits: list[Edit], report: dict, model) -> list[Edit]:
    """Match layer counts and multiplier distribution to the editor patch."""
    counts = Counter(e.layer for e in edits)
    channels = {int(layer): [r["channel"] for r in data["candidates"]]
                for layer, data in report["layers"].items()}
    selected = []
    for layer in sorted(counts):
        n = counts[layer]
        factors = sorted(e.multiplier for e in edits if e.layer == layer)
        if len(channels[layer]) < n:
            raise ValueError("Not enough diagnostic candidates")
        selected += [Edit(layer, channel, factor)
                     for channel, factor in zip(channels[layer][:n], factors)]
    return selected


def run(args: argparse.Namespace) -> dict:
    root = Path(args.out)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing experiment directory: {root}")
    if min(args.practice_per_family, args.heldout_per_family, args.regression_per_family) < 1:
        raise ValueError("All task split sizes must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    layers = tuple(args.layers)
    if not layers or len(layers) != len(set(layers)):
        raise ValueError("Choose unique target layers")

    model, tokenizer, env = load_model(args.checkpoint, args.device, args.revision)
    if any(layer < 0 or layer >= len(model.model.layers) for layer in layers):
        raise ValueError("Target layer index out of model range")
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "run-config.json", {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "environment": env,
        "editor_mode": args.editor,
        "editor_model": args.editor_model if args.editor == "openai" else None,
        "practice_per_family": args.practice_per_family,
        "heldout_per_family": args.heldout_per_family,
        "regression_per_family": args.regression_per_family,
        "layers": list(layers),
        "max_columns": args.max_columns,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "decoding": "greedy; identical for all variants",
        "gradient_updates": 0,
    })

    # PRACTICE PHASE — never examine heldout or regression prompts here.
    practice = make_tasks("practice", args.practice_per_family, args.seed)
    practice_rows = evaluate(model, tokenizer, practice, args.max_new_tokens)
    write_json(root / "practice.json", practice_rows)
    report = candidate_report(model, tokenizer, practice_rows, layers,
                              topk=max(12, args.max_columns))
    write_json(root / "diagnostics.json", report)
    proposal = choose_proposal(model, report, practice_rows, args.editor,
                               args.editor_model, args.proposal_file,
                               args.max_columns, layers)
    edits = validate_edits(model, proposal, args.max_columns, layers)
    write_json(root / "locked-proposal.json", {
        "editor_mode": args.editor,
        "editor_model": args.editor_model if args.editor == "openai" else None,
        "checkpoint": env["checkpoint"],
        "resolved_revision": env.get("resolved_checkpoint_revision"),
        "proposal": proposal,
        "note": "Saved BEFORE any heldout or regression scoring. No gradients.",
    })

    # CONTROLS — exactly the same layer counts and attenuation factors.
    controls = {
        "editor": edits,
        "activation_ranked": matched_heuristic(edits, report, model),
        "random": matched_random(edits, model, seed=args.seed + 1007),
    }
    write_json(root / "control-patches.json", {
        name: [edit.asdict() for edit in patch] for name, patch in controls.items()
    })

    results = {}
    # HELDOUT PHASE: The very first evaluation of unseen prompts happens only
    # after a frozen proposal exists on disk; editor receives no test rewards.
    for split, count in (("heldout", args.heldout_per_family),
                         ("regression", args.regression_per_family)):
        tasks = make_tasks(split, count, args.seed)
        base = evaluate(model, tokenizer, tasks, args.max_new_tokens)
        outputs = {"base": base}
        changed = {}
        for label, candidate in controls.items():
            with temporary_patch(model, candidate) as changed_count:
                outputs[label] = evaluate(model, tokenizer, tasks, args.max_new_tokens)
            changed[label] = changed_count
        result = {
            "summaries": {label: summarize(rows) for label, rows in outputs.items()},
            "paired_vs_base": {
                label: paired_bootstrap(base, rows) for label, rows in outputs.items()
                if label != "base"
            },
            "changed_weights": changed,
            "examples": outputs,
        }
        results[split] = result
        write_json(root / f"{split}.json", result)

    total = sum(p.numel() for p in model.parameters())
    summary = {
        "environment": env,
        "editor": args.editor,
        "editor_model": args.editor_model if args.editor == "openai" else None,
        "total_model_parameters": total,
        "heldout": {
            label: {
                "scores": scores,
                "changed_parameters": results["heldout"]["changed_weights"].get(label, 0),
                "fraction_changed": results["heldout"]["changed_weights"].get(label, 0) / total,
            } for label, scores in results["heldout"]["summaries"].items()
        },
        "heldout_paired": results["heldout"]["paired_vs_base"],
        "regression": results["regression"]["summaries"],
        "scope": "Only editor=openai is automatically LLM-authored; file mode provenance is external; heuristic is non-LLM.",
    }
    write_json(root / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=MODEL)
    parser.add_argument("--revision", help="Pin an exact model commit hash for formal results")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--editor", choices=("openai", "heuristic", "file"), default="openai")
    parser.add_argument("--editor-model", default="gpt-4.1")
    parser.add_argument("--proposal-file")
    parser.add_argument("--practice-per-family", type=int, default=4)
    parser.add_argument("--heldout-per-family", type=int, default=10)
    parser.add_argument("--regression-per-family", type=int, default=5)
    parser.add_argument("--layers", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--max-columns", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--out", default="artifacts/qwen-first-run")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"heldout": result["heldout"],
                      "heldout_paired": result["heldout_paired"]}, indent=2))
    print(f"Results written to {args.out}; zero gradient updates")


if __name__ == "__main__":
    main()
