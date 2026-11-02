"""Hand-write a circuit into selected parameters; no model fitting or gradients.

Two variants: follow (ball-paddle horizontal delta); lead (also ball displacement).
These are independent hand-coded circuits, not the authors' model-generated weights.
"""
import numpy as np
from .network import MLP, index

BALL, PADDLE, PREVIOUS = 1, 0, 4
LEFT, NO_OP, RIGHT = 1, 0, 3


def zero_policy() -> MLP:
    """Untrained all-zero network: uniform random action sampling."""
    return MLP.zeros()


def program_breakout(variant: str = "follow") -> MLP:
    if variant not in ("follow", "lead"):
        raise ValueError(f"Unknown program: {variant}")
    model = MLP.zeros()
    # Neuron 0 measures ball x minus paddle x.
    scale = 0.065
    for y in range(10):
        for x in range(10):
            coord = x - 4.5
            model.w1[0, index(BALL, y, x)] = scale * coord
            if variant == "lead":
                model.w1[1, index(BALL, y, x)] = scale * coord
                model.w1[1, index(PREVIOUS, y, x)] = -scale * coord
    for x in range(10):
        model.w1[0, index(PADDLE, 9, x)] = -scale * (x - 4.5)

    model.w2[0, 0] = 1.0
    if variant == "lead":
        model.w2[0, 1] = 2.0
    model.w3[LEFT, 0] = -24.0
    model.w3[RIGHT, 0] = 24.0
    model.b3[NO_OP] = 1.0
    model.b3[2] = model.b3[4] = model.b3[5] = -8.0
    return model


def edited_fraction(model: MLP) -> tuple[int, int, float]:
    nonzero = sum(int(np.count_nonzero(x)) for x in model.arrays().values())
    total = model.parameter_count
    return nonzero, total, nonzero / total
