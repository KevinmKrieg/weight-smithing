"""Regenerate the README data from seeded *toy* Breakout episodes, no training.

This is NOT upstream MinAtar and no LLM wrote these weights.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from weightspace.evaluate import run
from weightspace.program import edited_fraction, program_breakout

SETTINGS = ((0., 1000), (.1, 2000), (.25, 1000), (.5, 1000))
B = 5000


def summarize():
    generator = np.random.default_rng(2026)
    sweep = []
    main = None
    for sticky, episodes in SETTINGS:
        outcome = run(backend="toy", episodes=episodes, seed=0, max_steps=500,
                      sticky=sticky, temperature=1.)
        scores = {k: np.asarray(v["scores"], dtype=float)
                  for k, v in outcome["policies"].items()}
        indices = generator.integers(0, episodes, size=(B, episodes), dtype=np.int32)

        def ci(values):
            return [round(float(z), 4) for z in
                    np.percentile(values[indices].mean(axis=1), [2.5, 97.5])]

        entry = {"sticky": sticky, "episodes": episodes}
        for k in ("zero", "follow", "lead"):
            v = scores[k]
            entry[k] = [round(float(v.mean()), 4), *ci(v)]
        sweep.append(entry)
        if sticky == .1:
            main = {"episodes": episodes, "sticky": sticky}
            for k in ("zero", "follow", "lead"):
                v = scores[k]
                details = {
                    "mean": round(float(v.mean()), 4),
                    "ci95": ci(v),
                    "sd": round(float(v.std(ddof=1)), 4),
                    "median": float(np.median(v)),
                    "p90": float(np.percentile(v, 90)),
                    "max": float(v.max()),
                    "p_ge5": round(float(np.mean(v >= 5)), 4),
                    "hist": np.bincount(v.astype(int)).tolist(),
                }
                if k != "zero":
                    delta = v - scores["zero"]
                    details.update({
                        "paired_delta": round(float(delta.mean()), 4),
                        "delta_ci95": ci(delta),
                        "win": round(float(np.mean(delta > 0)), 4),
                        "tie": round(float(np.mean(delta == 0)), 4),
                        "loss": round(float(np.mean(delta < 0)), 4),
                    })
                main[k] = details
    edits, total, fraction = edited_fraction(program_breakout("lead"))
    return {
        "protocol": {
            "environment": "independent toy Breakout (NOT upstream MinAtar)",
            "model": "600 → 32 tanh → 32 tanh → 6",
            "max_steps": 500, "seed_start": 0,
            "action_temperature": 1, "bootstrap_replicates": B,
            "bootstrap_seed": 2026,
            "ci_method": "percentile bootstrap over seeded episodes; paired differences use matched seeds",
            "authoring": "human-designed weights (not LLM-generated)",
            "training": "none",
        },
        "weights": {"total": total, "nonzero": edits,
                    "edited_fraction": round(fraction, 6)},
        "main": main, "sweep": sweep,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/benchmark-summary.json"))
    args = parser.parse_args()
    result = summarize()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for v in result["sweep"]:
        print(v["sticky"], v["episodes"],
              {k: v[k][0] for k in ("zero", "follow", "lead")})
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
