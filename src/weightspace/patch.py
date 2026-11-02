"""Safe, declarative single-shot edits proposed by a language model.

This is not a model API client. It accepts sparse tensor edits as data only,
never as arbitrary generated Python, and does not call the game environment.
"""
import argparse
import json
import math
from pathlib import Path
from .network import MLP

ALLOWED_TENSORS = ("w1", "b1", "w2", "b2", "w3", "b3")


def apply_patch(model: MLP, proposal: dict, max_edits: int = 20_486) -> int:
    """Validate all edits before applying any. Edits are assignment, not addition."""
    if not isinstance(proposal, dict) or set(proposal) != {"edits"}:
        raise ValueError('Expected a JSON object with exactly one key: "edits"')
    edits = proposal["edits"]
    if not isinstance(edits, list) or len(edits) > max_edits:
        raise ValueError(f"Expected an edits list of at most {max_edits}")
    prepared = []
    seen = set()
    arrays = model.arrays()
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {"tensor", "index", "value"}:
            raise ValueError('Each edit needs exactly "tensor", "index", "value"')
        name, idx, val = edit["tensor"], edit["index"], edit["value"]
        if name not in ALLOWED_TENSORS:
            raise ValueError(f"Invalid tensor {name!r}")
        array = arrays[name]
        if (not isinstance(idx, list) or len(idx) != array.ndim or
                any(type(i) is not int or not 0 <= i < bound
                    for i, bound in zip(idx, array.shape))):
            raise ValueError(f"Invalid index for {name}: {idx!r}")
        if type(val) not in (int, float) or not math.isfinite(val) or abs(val) > 1e6:
            raise ValueError(f"Invalid weight value: {val!r}")
        key = (name, *idx)
        if key in seen:
            raise ValueError(f"Duplicate edit for {key}")
        seen.add(key)
        prepared.append((array, tuple(idx), float(val)))
    for array, index, value in prepared:
        array[index] = value
    return len(prepared)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch", required=True, type=Path, help="JSON edit proposal")
    parser.add_argument("--out", required=True, type=Path, help="Saved .npz model")
    parser.add_argument("--base", type=Path, help="Optional initial .npz (default all zeros)")
    parser.add_argument("--max-edits", type=int, default=20_486)
    args = parser.parse_args()
    model = MLP.load(args.base) if args.base else MLP.zeros()
    count = apply_patch(model, json.loads(args.patch.read_text()), args.max_edits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"Applied {count} edits; wrote {args.out}")


if __name__ == "__main__":
    main()
