from types import SimpleNamespace
import pytest
from weightsmith.context_editor import gate_decision,novel_pairs,fit_direction,gate_weight_patch

def test_router_and_novel():
    assert gate_decision("Return JSON. No Markdown.")
    assert not gate_decision("Return Markdown JSON.")
    assert not gate_decision("Return an integer. No Markdown.")
    raw,fenced=novel_pairs(123,8)
    assert all(a.expected==b.expected for a,b in zip(raw,fenced))
    assert sum(gate_decision(t.prompt) for t in raw)==4
    assert not any(gate_decision(t.prompt) for t in fenced)

def test_direction_and_restore():
    torch=pytest.importorskip("torch")
    x=torch.eye(3);y=torch.tensor([-1.,0.,0.])
    d=fit_direction(x,y,1.)
    assert torch.allclose(d,torch.tensor([-.5,0.,0.]))
    gate=torch.nn.Linear(3,2,bias=False)
    model=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(gate_proj=gate))]))
    old=gate.weight.detach().clone()
    with pytest.raises(RuntimeError):
        with gate_weight_patch(model,[(0,1,d)]):
            assert torch.allclose(gate.weight[1],old[1]+d)
            raise RuntimeError("restore")
    assert torch.equal(gate.weight,old)
