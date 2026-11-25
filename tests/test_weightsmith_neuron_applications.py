import pytest
from weightsmith.neuron_applications import auc, lexical, predict, CONCEPTS


def test_auc():
    assert auc([0, 1, 0, 1], [0, 2, 1, 3]) == 1.
    assert auc([0, 1], [1, 1]) == .5
    assert auc([0, 1], [2, 1]) == 0.
    with pytest.raises(ValueError):
        auc([1], [2])


def test_lexical_and_null():
    assert lexical("The sum is 12", ["sum", "number"]) == 1
    assert lexical("The summit", ["sum"]) == 0
    assert predict({"a": 0.0, "b": -0.1}, .01) == "none"


def test_disjoint_splits():
    train = [t for d in CONCEPTS.values() for t in d["train"]]
    test = [t for d in CONCEPTS.values() for t in d["test"] + d["hard"]]
    assert not set(train) & set(test)
