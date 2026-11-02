import pytest
import torch
from torch import nn
from weightsmith.targeted import choose_fence_token, patched_output_row


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [4, 5] if text.startswith(chr(96) * 3) else [7]
    def decode(self, ids):
        return chr(96) * 3 if ids[0] == 4 else "x"


def test_fence_detection():
    assert choose_fence_token(FakeTokenizer(), [
        {"response": chr(96) * 3 + 'json\n{}\n' + chr(96) * 3}
    ]) == (4, chr(96) * 3)
    with pytest.raises(ValueError):
        choose_fence_token(FakeTokenizer(), [{"response": "{}"}])


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = nn.Linear(5, 10, bias=False)
    def get_output_embeddings(self):
        return self.lm_head


def test_edit_changes_real_weights_and_restores():
    model = FakeModel()
    before = model.lm_head.weight.clone()
    with patched_output_row(model, 4, torch.ones(5) * .2) as changed:
        assert changed == 5
        torch.testing.assert_close(model.lm_head.weight[4], before[4] - .2)
    torch.testing.assert_close(model.lm_head.weight, before)


def test_edit_restores_after_error():
    model = FakeModel()
    before = model.lm_head.weight.clone()
    with pytest.raises(RuntimeError):
        with patched_output_row(model, 4, torch.ones(5) * .2):
            raise RuntimeError("failure")
    torch.testing.assert_close(model.lm_head.weight, before)
