import torch
from torch import nn
from weightsmith.diagnose import candidate_report


class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(7, 5, bias=False)
    def forward(self, x):
        return self.down_proj(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = Module()
    def forward(self, x):
        return self.mlp(x)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block()])
    def forward(self, input_ids, use_cache=False):
        x = torch.stack((input_ids.float(),) * 7, dim=-1)
        return self.model.layers[0](x)


class Tokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, return_tensors):
        return torch.tensor([[1, 2, 3 if messages[0]["content"] == "a" else 5]])


def test_diagnostics_and_no_hooks_remaining():
    model = TinyModel()
    rows = [{"prompt": "a", "correct": False}, {"prompt": "b", "correct": True}]
    report = candidate_report(model, Tokenizer(), rows, layers=(0,), topk=3)
    assert report["n_failed"] == 1
    assert len(report["layers"]["0"]["candidates"]) == 3
    assert len(report["layers"]["0"]["candidates"][0]["weight_fingerprint"]) == 5
    assert len(model.model.layers[0].mlp.down_proj._forward_pre_hooks) == 0
