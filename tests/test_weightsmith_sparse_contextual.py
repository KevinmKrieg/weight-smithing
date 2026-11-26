import pytest
from weightsmith.sparse_contextual import choose_trial,weighted_direction,classify_json

def trial(raw,loss,support,gain=2.):
    return {"support":support,"gain":gain,"feedback":{"correct":{"raw":raw},"losses":{"fenced":loss,"other":0}}}

def test_selection():
    rows=[trial(0,0,[],0.),trial(8,1,[0,1]),trial(6,0,[1]),trial(6,0,[0,1])]
    assert choose_trial(rows,0)["support"]==[1]
    assert choose_trial(rows,1)["support"]==[0,1]

def test_weighted_direction():
    torch=pytest.importorskip("torch")
    x=torch.tensor([[1.,0.],[1.,1.]])
    y=torch.tensor([-1.,0.])
    a=weighted_direction(x,y,torch.ones(2))
    b=weighted_direction(x,y,torch.tensor([1.,100.]))
    assert abs(float(x[1]@b))<abs(float(x[1]@a))

def test_decomposition():
    expected='{"a":1}'
    assert classify_json('{"a":1}',expected)=="correct_raw"
    assert classify_json(chr(96)*3+'json\n{"a":1}\n'+chr(96)*3,expected)=="correct_content_wrapped"
    assert classify_json('{"a":2}',expected)=="raw_wrong_content"
    assert classify_json('Here: {"a":1}',expected)=="other_invalid"
