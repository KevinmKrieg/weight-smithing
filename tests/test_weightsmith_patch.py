import torch
import pytest
from torch import nn
from weightsmith.patch import Edit, validate_edits, temporary_patch, matched_random
from weightsmith.editor import heuristic_proposal, proposal_prompt
from weightsmith.experiment import matched_heuristic


class FakeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(7, 5, bias=False)


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = FakeMLP()


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([FakeBlock() for _ in range(18)])


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeBackbone()


@pytest.fixture
def fake():
    return FakeModel()


def valid():
    return {"edits": [{"layer": 8, "channel": 2, "multiplier": .4}], "rationale": "test"}


def test_validate_and_reversible_edit(fake):
    edits = validate_edits(fake, valid())
    old = fake.model.layers[8].mlp.down_proj.weight.clone()
    with temporary_patch(fake, edits) as count:
        weight = fake.model.layers[8].mlp.down_proj.weight
        assert count == 5
        torch.testing.assert_close(weight[:, 2], old[:, 2] * .4)
    torch.testing.assert_close(fake.model.layers[8].mlp.down_proj.weight, old)


def test_restores_even_if_evaluator_crashes(fake):
    old = fake.model.layers[8].mlp.down_proj.weight.clone()
    with pytest.raises(RuntimeError):
        with temporary_patch(fake, validate_edits(fake, valid())):
            raise RuntimeError("inference error")
    torch.testing.assert_close(fake.model.layers[8].mlp.down_proj.weight, old)


@pytest.mark.parametrize("change", [
    {"layer": 8, "channel": 7, "multiplier": .4},
    {"layer": 17, "channel": 2, "multiplier": .4},
    {"layer": 8, "channel": 2, "multiplier": 1.0},
    {"layer": 8, "channel": 2, "multiplier": float("nan")},
    {"layer": 8, "channel": 2, "multiplier": True},
    {"layer": 8, "channel": "2", "multiplier": .4},
])
def test_invalid_edits_rejected(fake, change):
    with pytest.raises(ValueError):
        validate_edits(fake, {"edits": [change], "rationale": "test"})


def test_controls_match_edit_budget(fake):
    report = {"layers": {
        "8": {"candidates": [{"channel": x} for x in [1, 2, 3, 4, 5, 6]]},
        "16": {"candidates": [{"channel": x} for x in [1, 2, 3, 4, 5, 6]]},
    }}
    edits = validate_edits(fake, heuristic_proposal(report, n_per_layer=2))
    random = matched_random(edits, fake, seed=4)
    ranked = matched_heuristic(edits, report, fake)
    assert len(random) == len(ranked) == len(edits) == 4
    assert [x.layer for x in random] == [8, 8, 16, 16]
    assert sorted(x.multiplier for x in random) == sorted(x.multiplier for x in edits)


def test_editor_prompt_contains_only_supplied_examples():
    text = proposal_prompt({"layers": {}}, [
        {"family": "arithmetic", "prompt": "practice", "expected": "3",
         "response": "4", "correct": False},
    ])
    assert '"practice"' in text
    assert '"heldout"' not in text
