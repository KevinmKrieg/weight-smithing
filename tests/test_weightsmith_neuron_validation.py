import json
from weightsmith.neuron_validation import diverse_pairs, choose_ranked, parse_editor

def test_diverse_pairs_and_seeds():
    a,b=diverse_pairs(1,12,"evaluation")
    c,d=diverse_pairs(2,12,"evaluation")
    assert len(a)==len(b)==12
    assert all(x.expected==y.expected and x.prompt!=y.prompt for x,y in zip(a,b))
    assert all(isinstance(json.loads(t.expected),dict) for t in a)
    assert not {t.id for t in a}&{t.id for t in c}

def test_jacobian_rank():
    import pytest
    torch=pytest.importorskip("torch")
    # First neuron is helpful on raw and destructive on fenced; second is selective.
    s={12:torch.tensor([[5.,3.,-2.],[5.,3.,-2.],[8.,0.,0.],[8.,0.,0.]])}
    assert choose_ranked(s,2,2)==[(12,1,0.)]

def test_editor_validation():
    import pytest
    from types import SimpleNamespace
    torch=pytest.importorskip("torch")
    layer=SimpleNamespace(mlp=SimpleNamespace(down_proj=torch.nn.Linear(3,2)))
    model=SimpleNamespace(model=SimpleNamespace(layers=[layer]*24))
    report={"candidates":[{"layer":23,"channel":1}]}
    assert parse_editor('{"edits":[{"layer":23,"channel":1,"multiplier":0.5}],"rationale":"test"}',model,report)==[(23,1,.5)]
    with pytest.raises(ValueError):
        parse_editor('{"edits":[{"layer":23,"channel":2,"multiplier":0.5}],"rationale":"test"}',model,report)
