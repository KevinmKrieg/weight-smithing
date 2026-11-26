import json
from weightsmith.contextual_validation import schema_pairs,csv_pairs,wrapper_failure
def test_new_schemas():
    r,f=schema_pairs(1,8)
    assert all(a.expected==b.expected for a,b in zip(r,f))
    assert isinstance(json.loads(r[1].expected)["scores"],list)
    assert not {t.id for t in r}&{t.id for t in schema_pairs(2,8)[0]}
def test_csv_wrapper():
    r,f=csv_pairs(1,8,"practice")
    assert wrapper_failure(r[0],{"response":f[0].expected})
    assert not wrapper_failure(r[0],{"response":r[0].expected})
