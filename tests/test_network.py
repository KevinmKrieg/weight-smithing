import numpy as np
import pytest
from weightspace.network import MLP, encode, index
from weightspace.program import program_breakout, edited_fraction, zero_policy


def synthetic(ball_x: int, paddle_x: int, prev_x: int | None = None):
    obs = np.zeros((6, 10, 10), dtype=np.float32)
    obs[0, 9, paddle_x] = 1
    obs[1, 6, ball_x] = 1
    obs[4, 6, ball_x if prev_x is None else prev_x] = 1
    return obs


def test_shape_and_count():
    model = zero_policy()
    assert model.parameter_count == 20_486
    assert model.logits(synthetic(2, 6)).shape == (6,)
    np.testing.assert_array_equal(model.logits(synthetic(2, 6)), np.zeros(6))


def test_direct_program_responds_to_horizontal_position():
    model = program_breakout("follow")
    assert np.argmax(model.logits(synthetic(8, 3))) == 3
    assert np.argmax(model.logits(synthetic(2, 7))) == 1
    assert np.argmax(model.logits(synthetic(5, 5))) == 0


def test_velocity_changes_prediction():
    model = program_breakout("lead")
    assert np.argmax(model.logits(synthetic(4, 5, prev_x=2))) == 3


def test_sparse_direct_write():
    nonzero, total, fraction = edited_fraction(program_breakout())
    assert nonzero < total * 0.02
    assert total == 20_486
    assert fraction > 0


def test_roundtrip(tmp_path):
    model = program_breakout("lead")
    f = tmp_path / "net.npz"
    model.save(f)
    loaded = MLP.load(f)
    np.testing.assert_allclose(model.logits(synthetic(6, 7, 5)),
                               loaded.logits(synthetic(6, 7, 5)))


def test_bad_inputs():
    with pytest.raises(ValueError):
        encode(np.zeros((10, 10, 6)))
    with pytest.raises(ValueError):
        index(6, 0, 0)
    with pytest.raises(ValueError):
        program_breakout("unknown")
