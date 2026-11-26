"""025–027: sparse contextual interventions, protected fitting, schema decomposition."""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
from .context_editor import gate_inputs, fit_direction, gate_weight_patch
from .contextual_validation import fit_rows, apply_rows, evaluate_edits, serialize, schema_pairs
from .iterative_editor import extraction_pairs, unrelated_tasks, score_patch, feedback
from .neuron_basis import REVISION, scaled_columns, save
from .neuron_transfer import JAC_PATCH
from .runtime import MODEL, load_model

def protection_losses(f):
    return f["losses"]["fenced"]+f["losses"]["other"]

def choose_trial(trials,budget):
    eligible=[t for t in trials if protection_losses(t["feedback"])<=budget]
    return max(eligible,key=lambda t:(t["feedback"]["correct"]["raw"],
        -protection_losses(t["feedback"]),-len(t["support"]),-t["gain"]))

def sparse_trials(model,tok,edits,raw,fenced,other):
    base=score_patch(model,tok,[],raw,fenced,other)
    trials=[{"support":[],"gain":0.,"feedback":feedback(base,base)}]
    for k in (1,2,3,4):
        for support in itertools.combinations(range(4),k):
            for gain in (1.,2.):
                rows=apply_rows(model,tok,[edits[i] for i in support],gain,raw,fenced,other)
                trials.append({"support":list(support),"gain":gain,"feedback":feedback(rows,base)})
    return trials

def pilot025(args,out):
    model,tok,env=load_model(MODEL,args.device,REVISION)
    results=[]
    for seed in (25001,25002,25003):
        sub=out/str(seed);sub.mkdir()
        raw,fenced=extraction_pairs(seed,12,"practice")
        other=unrelated_tasks(seed,12,"practice")
        edits,fit=fit_rows(model,tok,raw,fenced+other)
        trials=sparse_trials(model,tok,edits,raw,fenced,other)
        zero=choose_trial(trials,0);one=choose_trial(trials,1)
        variants={"full":(edits,2.),"omit_0":(edits[1:],2.),
                  "omit_2":([e for i,e in enumerate(edits) if i!=2],2.),
                  "single_1":([edits[1]],2.)}
        for name,t in (("zero_loss",zero),("one_loss",one)):
            variants[name]=([edits[i] for i in t["support"]],t["gain"])
        save(sub/"practice.json",trials)
        save(sub/"locked-patch.json",{k:{"gain":g,"edits":serialize(e)} for k,(e,g) in variants.items()})
        results.append({"practice_seed":seed,"trials":len(trials),"zero_loss":zero,"one_loss":one,
            "evaluations":[evaluate_edits(model,tok,variants,seed*10+i,args.count,k,sub)
            for i,k in enumerate(("familiar","paraphrase"))]})
        save(out/"completed.json",results)
    return {"pilot":"025","environment":env,"replications":results,
        "caveat":"Prior exploratory supports independently tested; selections made on practice only. No held-out support search."}

def weighted_direction(x,y,weights,ridge=1.):
    """Weighted ridge in dual form: min ||sqrt(W)(Xd-y)||² + ridge||d||²."""
    import torch
    z=x.float()*weights.sqrt().unsqueeze(1)
    target=y.float()*weights.sqrt()
    return z.T@torch.linalg.solve(z@z.T+ridge*torch.eye(len(z)),target)

def protected_rows(model,tok,raw,protect,penalty):
    import torch
    layers=sorted({l for l,n,s in JAC_PATCH})
    inputs=gate_inputs(model,tok,raw+protect,layers)
    y=torch.tensor([-1.]*len(raw)+[0.]*len(protect))
    weights=torch.tensor([1.]*len(raw)+[float(penalty)]*len(protect))
    dirs={l:weighted_direction(inputs[l],y,weights) for l in layers}
    # Hold perturbation norm constant relative to ordinary ridge to isolate direction.
    for l in layers:
        ref=fit_direction(inputs[l],y)
        dirs[l]=dirs[l]*(ref.norm()/dirs[l].norm().clamp_min(1e-12))
    edits=[]
    for l,n,_ in JAC_PATCH:
        w=model.model.layers[l].mlp.gate_proj.weight[n].detach().float().cpu()
        median=float((inputs[l][:len(raw)]@w).median())
        edits.append((l,n,dirs[l]*(1. if median>=0 else -1.)*max(abs(median),.1)))
    diag={str(l):{"norm":float(d.norm()),"raw_mse":float(((inputs[l][:len(raw)]@d)+1).square().mean()),
        "protected_mse":float((inputs[l][len(raw):]@d).square().mean())} for l,d in dirs.items()}
    return edits,diag

def pilot026(args,out):
    model,tok,env=load_model(MODEL,args.device,REVISION)
    results=[]
    for seed in (26001,26002,26003):
        sub=out/str(seed);sub.mkdir()
        raw,fenced=extraction_pairs(seed,12,"practice")
        other=unrelated_tasks(seed,12,"practice")
        # Separate fitting and practice selection data; additional protective paraphrases.
        from .context_editor import novel_pairs
        fit_raw,fit_fenced=extraction_pairs(seed+100,12,"practice")
        extra_raw,extra_fenced=novel_pairs(seed+200,12)
        fit_other=unrelated_tasks(seed+100,12,"practice")
        fit_raw=fit_raw+extra_raw
        protect=fit_fenced+extra_fenced+fit_other
        base=score_patch(model,tok,[],raw,fenced,other)
        trials=[{"penalty":1,"gain":0.,"support":[],"feedback":feedback(base,base)}]
        edits_by_penalty={};diags={}
        for p in (1.,10.,100.):
            edits,diag=protected_rows(model,tok,fit_raw,protect,p)
            edits_by_penalty[p]=edits;diags[str(p)]=diag
            for g in (.5,1.,2.):
                rows=apply_rows(model,tok,edits,g,raw,fenced,other)
                trials.append({"penalty":p,"gain":g,"support":list(range(4)),"feedback":feedback(rows,base)})
        zero=choose_trial(trials,0);one=choose_trial(trials,1)
        ordinary=choose_trial([t for t in trials if t["penalty"]==1],0)
        variants={"ordinary_fixed2":(edits_by_penalty[1.],2.),"protected10_fixed2":(edits_by_penalty[10.],2.),
                  "protected100_fixed2":(edits_by_penalty[100.],2.)}
        for name,t in (("ordinary_selected",ordinary),("zero_loss",zero),("one_loss",one)):
            variants[name]=(edits_by_penalty[t["penalty"]],t["gain"])
        save(sub/"practice.json",{"trials":trials,"fit_diagnostics":diags})
        save(sub/"locked-patch.json",{k:{"gain":g,"edits":serialize(e)} for k,(e,g) in variants.items()})
        results.append({"practice_seed":seed,"zero_loss":zero,"one_loss":one,"ordinary":ordinary,
            "fit_diagnostics":diags,"trials":trials,
            "evaluations":[evaluate_edits(model,tok,variants,seed*10+i,args.count,k,sub)
                for i,k in enumerate(("familiar","paraphrase"))]})
        save(out/"completed.json",results)
    return {"pilot":"026","environment":env,"replications":results,
        "caveat":"Weighted activation-space preservation proxy, not a hard behavioral constraint. Equal norms at fixed gain; selection budgets differ from held-out guarantees."}

def classify_json(response,expected):
    """Strict full-response JSON; remove only an entire outer Markdown fence."""
    s=response.strip();target=json.loads(expected)
    def parse(v):
        try:return True,json.loads(v)
        except (ValueError,TypeError):return False,None
    ok,val=parse(s)
    if ok:return "correct_raw" if val==target else "raw_wrong_content"
    lines=s.splitlines()
    if len(lines)>=3 and lines[0].strip().lower() in (chr(96)*3,chr(96)*3+"json") and lines[-1].strip()==chr(96)*3:
        ok,val=parse("\n".join(lines[1:-1]))
        if ok:return "correct_content_wrapped" if val==target else "wrapped_wrong_content"
    return "other_invalid"

def pilot027(args,out):
    from collections import Counter
    model,tok,env=load_model(MODEL,args.device,REVISION)
    raw,fenced=extraction_pairs(27001,12,"practice")
    other=unrelated_tasks(27001,12,"practice")
    edits,_=fit_rows(model,tok,raw,fenced+other)
    save(out/"locked-patch.json",{"gain":2.,"edits":serialize(edits)})
    results=[]
    for seed in (27009,27010):
        for kind in ("familiar","schema"):
            r,f=extraction_pairs(seed,args.count,"heldout") if kind=="familiar" else schema_pairs(seed,args.count)
            o=unrelated_tasks(seed,args.count,"heldout")
            rows={"base":score_patch(model,tok,[],r,f,o)}
            with scaled_columns(model,JAC_PATCH):rows["global"]=score_patch(model,tok,[],r,f,o)
            rows["contextual"]=apply_rows(model,tok,edits,2.,r,f,o)
            for name,groups in rows.items():
                for t,row in zip(r,groups["raw"]):row["error_category"]=classify_json(row["response"],t.expected)
            save(out/f"{seed}-{kind}.json",rows)
            results.append({"seed":seed,"kind":kind,"n":len(r),
                "categories":{name:dict(Counter(v["error_category"] for v in groups["raw"])) for name,groups in rows.items()},
                "paired":{name:feedback(groups,rows["base"]) for name,groups in rows.items() if name!="base"}})
    return {"pilot":"027","environment":env,"evaluations":results,
        "caveat":"Semantic object equality after full response parse or single outer fence removal; not arbitrary substring extraction. Python equality may conflate bool/int; official exact scorer retained separately."}

def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot",required=True,choices=["025","026","027"])
    p.add_argument("--out",required=True)
    p.add_argument("--device",default="cpu")
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--count",type=int,default=24)
    a=p.parse_args()
    if a.count<4 or a.threads<1:p.error("Invalid counts")
    torch.set_num_threads(a.threads);torch.manual_seed(252627)
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    result={"025":pilot025,"026":pilot026,"027":pilot027}[a.pilot](a,out)
    save(out/"summary.json",result)
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
