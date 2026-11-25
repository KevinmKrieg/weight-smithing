from weightsmith.regression_editor import choose_final,pareto,format_only,format_tasks
from weightsmith.benchmark import Task

def row(raw,loss,margin=0):
    return {"patch":[(1,raw,0.)],"feedback":{"correct":{"raw":raw},"regression_losses":loss,
            "raw_margin_gain":margin,"fence_margin_harm":0.}}

def test_constraint():
    base={"correct":{"raw":0}}
    assert choose_final([row(4,1),row(2,0)],base)==[(1,2,0.)]
    assert choose_final([row(4,1)],base)==[]
    assert choose_final([row(0,0,100)],base)==[]

def test_pareto():
    a,b,c=row(4,1),row(2,0),row(1,1)
    assert pareto([a,b,c])==[a,b]

def test_format_only():
    t=Task("x","label","color","red")
    assert format_only(t,"Answer: red")
    assert format_only(t,"red.")
    assert not format_only(t,"red")
    assert not format_only(t,"red and blue")
    a=format_tasks(1,18,"discovery","csv")
    b=format_tasks(2,24,"confirmation","csv")
    assert not {t.id for t in a}&{t.id for t in b}
