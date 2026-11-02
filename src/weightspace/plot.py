"""Generate a lightweight benchmark comparison chart from a saved results JSON."""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    data = json.loads(args.results.read_text())
    policies = ["zero", "follow", "lead"]
    means = [data["policies"][name]["mean_reward"] for name in policies]
    sds = [data["policies"][name]["std_reward"] for name in policies]
    n = data["setup"]["episodes"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(policies, means, yerr=[1.96*sd/(n**0.5) for sd in sds], capsize=5)
    ax.set(ylabel="Mean episode reward (95% normal-approx. CI)",
           title=f"Training-free weight programs — {data['setup']['backend']} Breakout (n={n})")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
