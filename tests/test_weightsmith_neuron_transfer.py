import json
from weightsmith.neuron_transfer import allowed_next,grammar_options,choose_scale,scaled,label_tasks

def test_grammar():
    assert allowed_next([(1,2,3),(1,4,3)],(1,))==[2,4]
    assert allowed_next([(1,2,3)],(2,))==[]
    opts=[json.loads(x) for x in grammar_options([{"layer":3,"channel":2},{"layer":3,"channel":4}],{(3,2)})]
    assert len(opts)==5
    assert all(o.get("stop") or o["channel"]==4 for o in opts)

def test_scale_selection():
    rows=[{"scale":s,"raw_correct":r,"fenced_losses":f,"unrelated_losses":u}
          for s,r,f,u in [(0.,5,1,0),(.25,4,0,0),(.5,4,0,0),(1.,0,0,0)]]
    assert choose_scale(rows)==.5
    assert scaled([(1,2,0.)],1.)==[]

def test_label_pairs():
    a,b=label_tasks(1,8,"practice")
    c,d=label_tasks(2,8,"heldout")
    assert all(y.expected==f"The color is {x.expected}." for x,y in zip(a,b))
    assert not {t.id for t in a}&{t.id for t in c}
