"""Generate Qwen practice-only diagnostic files without touching heldout prompts."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
from .benchmark import make_tasks, summarize
from .diagnose import candidate_report
from .experiment import write_json
from .runtime import MODEL, evaluate, load_model


def run_practice(checkpoint: str, per_family: int, seed: int, max_new_tokens: int,
                 device: str, out: Path, layers: tuple[int, ...],
                 revision: str | None) -> dict:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing run: {out}")
    model, tokenizer, config = load_model(checkpoint, device, revision)
    if any(layer >= len(model.model.layers) or layer < 0 for layer in layers):
        raise ValueError("Target layer index out of bounds")
    tasks = make_tasks("practice", per_family, seed)
    rows = evaluate(model, tokenizer, tasks, max_new_tokens)
    report = candidate_report(model, tokenizer, rows, layers=layers, topk=12)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "model.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": config, "seed": seed, "max_new_tokens": max_new_tokens,
        "per_family": per_family, "layers": layers,
        "heldout_seen": False, "gradients_used": False,
    })
    write_json(out / "practice.json", rows)
    write_json(out / "diagnostics.json", report)
    write_json(out / "practice-summary.json", summarize(rows))
    return {"environment": config, "practice": summarize(rows), "diagnostics": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--per-family", type=int, default=4)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--layers", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--out", type=Path, default=Path("artifacts/practice"))
    args = parser.parse_args()
    info = run_practice(args.checkpoint, args.per_family, args.seed, args.max_new_tokens,
                        args.device, args.out, tuple(args.layers), args.revision)
    import json
    print("PRACTICE SUMMARY (not heldout)")
    print(json.dumps(info, indent=2))
    print(f"Saved practice-only diagnostics to {args.out}")


if __name__ == "__main__":
    main()
