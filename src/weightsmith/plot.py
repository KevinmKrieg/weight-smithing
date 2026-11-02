"""Generate SVG charts only from completed local benchmark measurements."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def make_plots(run_dir: Path, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    summary = json.loads((run_dir / "summary.json").read_text())
    policies = [key for key in ("base", "editor", "activation_ranked", "random")
                if key in summary["heldout"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {"base": "#8d9ab4", "editor": "#5ee5c6",
              "activation_ranked": "#95aaff", "random": "#f0bb76"}
    names = {"base": "Original", "editor": "Edited",
             "activation_ranked": "Activation-only", "random": "Random edit"}
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "text.color": "#e8f0fa", "axes.labelcolor": "#e8f0fa",
        "xtick.color": "#b6c6da", "ytick.color": "#b6c6da",
        "axes.edgecolor": "#394d63", "axes.facecolor": "#122338",
        "figure.facecolor": "#0b1627", "savefig.facecolor": "#0b1627",
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    values = [100 * summary["heldout"][k]["scores"]["accuracy"] for k in policies]
    bars = ax.bar([names[k] for k in policies], values,
                  color=[colors[k] for k in policies], width=.6)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Strict instruction accuracy (%)")
    ax.set_title("Qwen2.5-0.5B-Instruct · heldout accuracy", pad=18)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2,
                f"{value:.1f}%", ha="center", color="#e8f0fa", weight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy.svg", bbox_inches="tight")
    plt.close(fig)

    families = list(summary["heldout"]["base"]["scores"]["accuracy_by_family"])
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(families))
    width = .19
    for i, key in enumerate(policies):
        vals = [100 * summary["heldout"][key]["scores"]["accuracy_by_family"].get(f, 0)
                for f in families]
        ax.bar(x + (i - 1.5) * width, vals, width, label=names[key], color=colors[key])
    ax.set_xticks(x, families)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Exact match (%)")
    ax.set_title("Which behaviors actually changed?", pad=18)
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_dir / "by-task.svg", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    caps = [100 * summary["heldout"][k]["scores"]["length_cap_rate"] for k in policies]
    repeats = [100 * summary["heldout"][k]["scores"]["mean_repeat4"] for k in policies]
    x = np.arange(len(policies))
    ax.bar(x - .175, caps, .35, label="Hit generation cap", color="#f0bb76")
    ax.bar(x + .175, repeats, .35, label="Repeated 4-grams", color="#95aaff")
    ax.set_xticks(x, [names[k] for k in policies])
    ax.set_ylabel("Rate (%)")
    ax.set_title("Failure and regression diagnostics", pad=18)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_dir / "failure-modes.svg", bbox_inches="tight")
    plt.close(fig)
    return [str(out_dir / name) for name in
            ("accuracy.svg", "by-task.svg", "failure-modes.svg")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/charts"))
    args = parser.parse_args()
    for path in make_plots(args.run, args.out):
        print("Saved", path)


if __name__ == "__main__":
    main()
