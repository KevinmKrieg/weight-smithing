import pytest

torch = pytest.importorskip("torch")
from types import SimpleNamespace
from weightsmith.neuron_basis import (scaled_columns, signed_readouts, matched_prompts,
                                     fenced_correct, choose_causal)


def test_signed_readouts():
    u = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    pos, neg = signed_readouts(u, torch.tensor([[1.], [0.]]), k=1, chunk=2)
    assert pos == [[0]]
    assert neg == [[2]]


def test_restore_after_error():
    linear = torch.nn.Linear(3, 2, bias=False)
    model = SimpleNamespace(model=SimpleNamespace(layers=[
        SimpleNamespace(mlp=SimpleNamespace(down_proj=linear))]))
    original = linear.weight.detach().clone()
    with pytest.raises(RuntimeError):
        with scaled_columns(model, [(0, 1, 0.)]):
            assert torch.count_nonzero(linear.weight[:, 1]) == 0
            raise RuntimeError("test")
    assert torch.equal(original, linear.weight)


def test_fence_regression_and_split():
    raw, fenced = matched_prompts("practice", 3, 123)
    assert all(a.expected == b.expected for a, b in zip(raw, fenced))
    fence = chr(96) * 3
    assert fenced_correct(fenced[0], fence + "json\n" + raw[0].expected + "\n" + fence)
    assert not fenced_correct(fenced[0], raw[0].expected)
    assert not fenced_correct(fenced[0], fence + "json\n{}\n" + fence)


def test_causal_selection():
    rows = [{"raw_reduction": 1., "causal_score": .3},
            {"raw_reduction": -1., "causal_score": 2.},
            {"raw_reduction": 2., "causal_score": -.1}]
    assert choose_causal(rows, 4) == [rows[0]]
