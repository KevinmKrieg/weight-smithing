import numpy as np
import pytest
from weightspace.network import MLP
from weightspace.patch import apply_patch


def test_patch_writes_only_requested_weights():
    model = MLP.zeros()
    n = apply_patch(model, {"edits": [
        {"tensor": "w1", "index": [0, 123], "value": 0.3},
        {"tensor": "b3", "index": [1], "value": -2},
    ]})
    assert n == 2
    assert np.count_nonzero(model.w1) == 1
    assert np.count_nonzero(model.b3) == 1
    assert model.w1[0, 123] == pytest.approx(0.3)


@pytest.mark.parametrize("proposal", [
    {"edits": [{"tensor": "w1", "index": [32, 0], "value": 1}]},
    {"edits": [{"tensor": "w1", "index": [0, 0], "value": float("nan")}]},
    {"edits": [{"tensor": "w1", "index": [0, 0], "value": True}]},
    {"edits": [{"tensor": "bad", "index": [0], "value": 1}]},
    {"edits": [{"tensor": "w1", "index": [0, 0], "value": 1},
               {"tensor": "w1", "index": [0, 0], "value": 2}]},
])
def test_rejects_invalid_proposals_without_partial_application(proposal):
    model = MLP.zeros()
    with pytest.raises(ValueError):
        apply_patch(model, proposal)
    assert sum(np.count_nonzero(a) for a in model.arrays().values()) == 0
