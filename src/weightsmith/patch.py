"""Validated, reversible changes to internal Qwen MLP projection columns.

Edits change real weight tensors, not prompts or forward hooks. No gradients.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class Edit:
    layer: int
    channel: int
    multiplier: float

    def asdict(self) -> dict:
        return {"layer": self.layer, "channel": self.channel, "multiplier": self.multiplier}


def down_projection(model, layer: int):
    return model.model.layers[layer].mlp.down_proj.weight


def validate_edits(model, proposal: dict, max_columns: int = 16,
                   allowed_layers: tuple[int, ...] = (8, 16)) -> list[Edit]:
    if type(proposal) is not dict or set(proposal) != {"edits", "rationale"}:
        raise ValueError("Proposal must have exactly 'edits' and 'rationale'")
    if not isinstance(proposal["rationale"], str):
        raise ValueError("rationale must be text")
    edits = proposal["edits"]
    if not isinstance(edits, list) or not 1 <= len(edits) <= max_columns:
        raise ValueError(f"Provide between 1 and {max_columns} column edits")
    approved = []
    used = set()
    for item in edits:
        if type(item) is not dict or set(item) != {"layer", "channel", "multiplier"}:
            raise ValueError("Each edit needs only layer, channel and multiplier")
        layer, channel, multiplier = item["layer"], item["channel"], item["multiplier"]
        if type(layer) is not int or layer not in allowed_layers:
            raise ValueError(f"Layer {layer!r} is not allowed")
        weight = down_projection(model, layer)
        if type(channel) is not int or not 0 <= channel < weight.shape[1]:
            raise ValueError(f"Channel {channel!r} invalid for layer {layer}")
        if type(multiplier) not in (float, int) or not math.isfinite(multiplier) or not 0 <= multiplier < 1:
            raise ValueError("Only bounded attenuation [0, 1) is allowed")
        if (layer, channel) in used:
            raise ValueError("Duplicate column edit")
        used.add((layer, channel))
        approved.append(Edit(layer, channel, float(multiplier)))
    return approved


@contextmanager
def temporary_patch(model, edits: list[Edit]):
    """Restore the unmodified weights even when evaluation raises an exception."""
    originals = []
    changed = 0
    import torch
    with torch.no_grad():
        try:
            for edit in edits:
                weight = down_projection(model, edit.layer)
                old = weight[:, edit.channel].clone()
                originals.append((weight, edit.channel, old))
                new = (old.float() * edit.multiplier).to(dtype=weight.dtype)
                weight[:, edit.channel].copy_(new)
                changed += int(torch.count_nonzero(old != new).item())
            yield changed
        finally:
            for weight, channel, original in reversed(originals):
                weight[:, channel].copy_(original)


def matched_random(edits: list[Edit], model, seed: int = 9173) -> list[Edit]:
    rng = random.Random(seed)
    chosen = set()
    result = []
    for edit in edits:
        size = down_projection(model, edit.layer).shape[1]
        options = [i for i in range(size) if (edit.layer, i) not in chosen]
        channel = rng.choice(options)
        chosen.add((edit.layer, channel))
        result.append(Edit(edit.layer, channel, edit.multiplier))
    return result
