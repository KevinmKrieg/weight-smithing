import numpy as np
from weightspace.environments import ToyBreakout
from weightspace.evaluate import run


def test_toy_is_seeded():
    a, b = ToyBreakout(), ToyBreakout()
    np.testing.assert_equal(a.reset(42), b.reset(42))
    for action in [1, 3, 3, 0, 1, 3, 0]:
        x1, r1, d1 = a.step(action)
        x2, r2, d2 = b.step(action)
        np.testing.assert_equal(x1, x2)
        assert (r1, d1) == (r2, d2)


def test_eval_returns_paired_results():
    result = run(episodes=3, max_steps=25)
    assert result["setup"]["backend"] == "toy"
    assert set(result["policies"]) == {"zero", "follow", "lead"}
    for p in result["policies"].values():
        assert len(p["scores"]) == 3
