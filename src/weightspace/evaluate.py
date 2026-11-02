"""Evaluate weight-written MLP policies on seeded game episodes."""
import argparse
import json
from pathlib import Path
import numpy as np
from .environments import make_env
from .program import zero_policy, program_breakout, edited_fraction

POLICIES = ("zero", "follow", "lead")


def evaluate_one(name: str, backend: str, seeds: list[int],
                 max_steps: int = 500, sticky: float = 0.1,
                 temperature: float = 1.0) -> list[float]:
    model = zero_policy() if name == "zero" else program_breakout(name)
    scores = []
    env = make_env(backend, sticky=sticky)
    for seed in seeds:
        obs = env.reset(seed)
        action_rng = np.random.default_rng(seed + 1_000_003)
        score = 0.
        for _ in range(max_steps):
            action = model.act(obs, action_rng, temperature=temperature)
            obs, reward, done = env.step(action)
            score += reward
            if done:
                break
        scores.append(float(score))
    return scores


def run(backend: str = "toy", episodes: int = 30, seed: int = 0,
        max_steps: int = 500, sticky: float = 0.1,
        temperature: float = 1.0) -> dict:
    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max_steps must be positive")
    seeds = list(range(seed, seed + episodes))
    raw = {name: evaluate_one(name, backend, seeds, max_steps, sticky, temperature)
           for name in POLICIES}
    return {
        "setup": {
            "backend": backend, "episodes": episodes, "seed": seed,
            "max_steps": max_steps, "sticky": sticky,
            "temperature": temperature, "model": "600-32-32-6 tanh",
            "note": "No gradients, PPO baseline, or author checkpoints used.",
        },
        "policies": {
            name: {
                "mean_reward": round(float(np.mean(raw[name])), 3),
                "std_reward": round(float(np.std(raw[name], ddof=1)), 3) if episodes > 1 else 0.,
                "scores": raw[name],
            } for name in POLICIES
        },
        "paired_differences_from_zero": {
            name: {
                "mean": round(float(np.mean(np.array(raw[name]) - np.array(raw["zero"]))), 3),
                "standard_error": round(
                    float(np.std(np.array(raw[name]) - np.array(raw["zero"]), ddof=1) /
                          np.sqrt(episodes)), 3) if episodes > 1 else None,
            } for name in ("follow", "lead")
        },
        "weight_edits": {
            name: {
                "nonzero": edited_fraction(program_breakout(name))[0],
                "total": edited_fraction(program_breakout(name))[1],
                "fraction": round(edited_fraction(program_breakout(name))[2], 6),
            } for name in ("follow", "lead")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("toy", "minatar"), default="toy")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sticky", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    results = run(args.backend, args.episodes, args.seed,
                  args.max_steps, args.sticky, args.temperature)
    output = json.dumps(results, indent=2)
    print(output)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n")


if __name__ == "__main__":
    main()
