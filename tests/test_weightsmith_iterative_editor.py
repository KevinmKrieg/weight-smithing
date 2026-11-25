import json
from weightsmith.iterative_editor import extraction_pairs,unrelated_tasks,action_space,apply_action,objective

def test_extraction_pairs():
    raw,fenced=extraction_pairs(3,12,"practice")
    assert len(raw)==len(fenced)==12
    assert all(a.expected==b.expected and a.prompt!=b.prompt for a,b in zip(raw,fenced))
    assert all(isinstance(json.loads(t.expected),dict) for t in raw)
    assert not {t.id for t in raw}&{t.id for t in extraction_pairs(4,12,"heldout")[0]}

def test_actions():
    p=[(1,2,.5)]
    actions=action_space([(1,2),(2,3)],p)
    assert {"op":"remove","layer":1,"channel":2} in actions
    assert apply_action(p,{"op":"set","layer":2,"channel":3,"multiplier":0.})==[(1,2,.5),(2,3,0.)]
    assert apply_action(p,{"op":"remove","layer":1,"channel":2})==[]
    assert apply_action(p,{"op":"keep"})==p

def test_objective():
    assert objective({"correct":{"raw":1},"losses":{"fenced":0,"other":0}})>objective({"correct":{"raw":5},"losses":{"fenced":1,"other":0}})
    assert len(unrelated_tasks(1,12,"practice"))==12
