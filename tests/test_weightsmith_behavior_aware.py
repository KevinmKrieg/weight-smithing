import pytest
from weightsmith.behavior_aware import alternate_pairs,broad_other,wrapper_category,discovery_tasks,logit_gradient

def test_pairs():
    r,f=alternate_pairs(1,12)
    assert len(r)==len(f)==12
    assert all(a.expected==b.expected for a,b in zip(r,f))
    assert len(broad_other(1,12))==12
    assert not {t.id for t in r}&{t.id for t in alternate_pairs(2,12)[0]}

def test_discovery():
    tasks=discovery_tasks(1,8)
    assert len(tasks)==5 and all(len(v)==8 for v in tasks.values())
    assert wrapper_category("abc","abc")=="correct_raw"
    assert wrapper_category(chr(96)*3+"csv\nabc\n"+chr(96)*3,"abc")=="wrapper_only"
    assert wrapper_category(chr(96)*3+"\nwrong\n"+chr(96)*3,"abc")=="wrapped_wrong"

def test_logit_gradient():
    torch=pytest.importorskip("torch")
    from types import SimpleNamespace
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb=torch.nn.Embedding(8,3)
            self.layers=torch.nn.ModuleList([torch.nn.Module()])
            self.layers[0].mlp=torch.nn.Module()
            self.layers[0].mlp.gate_proj=torch.nn.Linear(3,3,bias=False)
        def forward(self,input_ids,use_cache=False):
            x=self.emb(input_ids)
            return SimpleNamespace(last_hidden_state=self.layers[0].mlp.gate_proj(x).cumsum(1))
    m=torch.nn.Module();m.model=Decoder();m.lm_head=torch.nn.Linear(3,8,bias=False)
    ids=torch.tensor([[1,2]])
    w=m.model.layers[0].mlp.gate_proj.weight
    logits=m.lm_head(m.model(ids).last_hidden_state[:,-1])[0]
    expected=torch.autograd.grad(logits[3]-logits[4],w)[0][1]
    for p in m.parameters():p.requires_grad_(False)
    actual,_=logit_gradient(m,None,ids,(0,1,0),3,4)
    assert torch.allclose(actual,expected,atol=1e-6)
