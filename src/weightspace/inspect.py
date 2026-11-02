"""Dump edited weights and generate a prompt for weight-only LLM analysis."""
import argparse
import json
from pathlib import Path
import numpy as np
from .network import MLP, INPUT_CHANNELS, ACTIONS
from .program import program_breakout, zero_policy


def summarize(model: MLP) -> dict:
    return {
        "architecture": "600 -> 32 tanh -> 32 tanh -> 6 logits",
        "inputs": list(INPUT_CHANNELS),
        "actions": list(ACTIONS),
        "layers": {
            name: {
                "shape": list(arr.shape),
                "nonzero_count": int(np.count_nonzero(arr)),
                "max_abs": float(np.abs(arr).max()),
            } for name, arr in model.arrays().items()
        },
    }


def weight_prompt(model: MLP, *, include_weights: bool = True) -> str:
    data = summarize(model)
    prelude = (
        "Infer the task/behavior implemented by the following neural network "
        "from the raw weights alone. Do not run a forward pass, use game "
        "rollouts, or assume the network has been trained.\n"
        "The input is a 6-channel 10x10 binary grid flattened channel-first. "
        "The six channels are: player, moving objects, targets, obstacles, "
        "previous frame, gauge. The output action logits are no-op, left, up, "
        "right, down, fire. There are two tanh layers of width 32.\n"
        "State your interpretation, the evidence in the weights, and uncertainty.\n"
    )
    if include_weights:
        sparse = {
            name: {
                "shape": list(arr.shape),
                "entries": [[*map(int, idx), float(arr[tuple(idx)])]
                            for idx in np.argwhere(arr != 0)],
            } for name, arr in model.arrays().items()
        }
        prelude += "\nWeight tensors (sparse, all unspecified entries zero):\n"
        prelude += json.dumps(sparse, separators=(",", ":"))
    else:
        prelude += "\n" + json.dumps(data, indent=2)
    return prelude


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("zero", "follow", "lead"), default="follow")
    parser.add_argument("--export", type=Path, help="Optional .npz output path")
    parser.add_argument("--prompt", type=Path, help="Optional LLM prompt output path")
    args = parser.parse_args()
    model = zero_policy() if args.policy == "zero" else program_breakout(args.policy)
    print(json.dumps(summarize(model), indent=2))
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        model.save(args.export)
    if args.prompt:
        args.prompt.parent.mkdir(parents=True, exist_ok=True)
        args.prompt.write_text(weight_prompt(model))


if __name__ == "__main__":
    main()
