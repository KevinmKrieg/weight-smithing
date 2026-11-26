"""022–024: contextual edit replication, causal controls and new behavior."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from .benchmark import Task, paired_bootstrap
from .context_editor import gate_inputs, fit_direction, gate_weight_patch, summarize_rows, novel_pairs
from .iterative_editor import extraction_pairs, unrelated_tasks, score_patch, feedback, exact_rows
from .neuron_basis import REVISION, scaled_columns, save
from .neuron_transfer import JAC_PATCH
from .runtime import MODEL, load_model

def fit_rows(model,tok,raw,protect,seed=None,mode="true"):
    import torch
    layers=sorted({l for l,n,s in JAC_PATCH})
    x=gate_inputs(model,tok,raw+protect,layers)
    y=torch.tensor([-1.]*len(raw)+[0.]*len(protect))
    rng=torch.Generator().manual_seed(seed or 0)
    true={l:fit_direction(x[l],y) for l in layers}
    if mode=="shuffle":
        yp=y[torch.randperm(len(y),generator=rng)]
        dirs={l:fit_direction(x[l],yp) for l in layers}
    elif mode=="random":
        dirs={l:torch.randn(true[l].shape,generator=rng) for l in layers}
    elif mode=="true":dirs=true
    else:raise ValueError(mode)
    # Match per-layer direction norms exactly, before identical per-row scaling.
    dirs={l:d*(true[l].norm()/d.norm().clamp_min(1e-12)) for l,d in dirs.items()}
    edits=[]
    for l,n,_ in JAC_PATCH:
        w=model.model.layers[l].mlp.gate_proj.weight[n].detach().float().cpu()
        med=float((x[l][:len(raw)]@w).median())
        edits.append((l,n,dirs[l]*(1. if med>=0 else -1.)*max(abs(med),.1)))
    return edits,{str(l):{"norm":float(d.norm()),"predictions":(x[l]@d).tolist()} for l,d in dirs.items()}

def apply_rows(model,tok,edits,gain,raw,fenced,other):
    with gate_weight_patch(model,[(l,n,d*gain) for l,n,d in edits]):
        return score_patch(model,tok,[],raw,fenced,other)

def select_gain(model,tok,edits,raw,fenced,other):
    base=score_patch(model,tok,[],raw,fenced,other)
    trials=[]
    for g in (0.,.25,.5,1.,2.):
        rows=apply_rows(model,tok,edits,g,raw,fenced,other)
        trials.append({"gain":g,"feedback":feedback(rows,base),"examples":rows})
    safe=[t for t in trials if t["feedback"]["losses"]["fenced"]+t["feedback"]["losses"]["other"]==0]
    chosen=max(safe,key=lambda t:(t["feedback"]["correct"]["raw"],-t["gain"]))["gain"]
    return chosen,trials

def serialize(edits):
    return [{"layer":l,"channel":n,"delta":d.tolist(),"norm":float(d.norm())} for l,n,d in edits]

def schema_pairs(seed,n):
    rng=random.Random(seed);raw=[];fenced=[]
    for i in range(n):
        a,b=rng.sample(range(10,90),2)
        name=rng.choice(["Iris","Eli","Ada","Kai"])
        cases=[
          (f"Order {a} contains {b} units. Extract order number as id and units as quantity.",{"id":a,"quantity":b}),
          (f"Scores for {name}: first {a}, second {b}. Extract name and scores (array of integers).",{"name":name,"scores":[a,b]}),
          (f"Account {a} is disabled. Extract account as integer and enabled as boolean.",{"account":a,"enabled":False}),
          (f"Room {a} holds {b} seats. Extract nested room object with number and seats.",{"room":{"number":a,"seats":b}})]
        prompt,obj=cases[i%4];expected=json.dumps(obj,separators=(",",":"))
        r=[" Emit only the object; the first character must be an opening curly brace.",
           " Return bare JSON suitable for direct parsing.",
           " Your entire answer must be a JSON object without backticks.",
           " Supply the JSON object itself, with no surrounding text."][i%4]
        f=[" Enclose the JSON in a json-labeled Markdown code block.",
           " Present the object inside triple backticks with language json.",
           " Format the answer as fenced JSON.",
           " Use a Markdown code fence tagged json, and nothing else."][i%4]
        raw.append(Task(f"schema-{seed}-{i}-raw","json",prompt+r,expected))
        fenced.append(Task(f"schema-{seed}-{i}-fence","fenced_json",prompt+f,expected))
    return raw,fenced

def evaluate_edits(model,tok,variants,seed,n,kind,out):
    raw,fenced={"familiar":lambda:extraction_pairs(seed,n,"heldout"),
                "paraphrase":lambda:novel_pairs(seed,n),
                "schema":lambda:schema_pairs(seed,n)}[kind]()
    other=unrelated_tasks(seed,n,"heldout")
    rows={"base":score_patch(model,tok,[],raw,fenced,other),
          "global":score_patch(model,tok,JAC_PATCH,raw,fenced,other)}
    for name,(edits,gain) in variants.items():
        rows[name]=apply_rows(model,tok,edits,gain,raw,fenced,other)
    save(out/f"examples-{seed}.json",rows)
    return {"seed":seed,"kind":kind,"n_per_group":n,**summarize_rows(rows)}

def pilot022(args,out):
    model,tok,env=load_model(MODEL,args.device,REVISION)
    results=[]
    for seed in (22001,22002,22003):
        sub=out/str(seed);sub.mkdir()
        raw,fenced=extraction_pairs(seed,12,"practice")
        other=unrelated_tasks(seed,12,"practice")
        edits,fit=fit_rows(model,tok,raw,fenced+other)
        gain,trials=select_gain(model,tok,edits,raw,fenced,other)
        save(sub/"practice.json",trials)
        save(sub/"locked-patch.json",{"gain":gain,"edits":serialize(edits),"fit":fit})
        variants={"contextual":(edits,gain)}
        results.append({"practice_seed":seed,"gain":gain,"practice":[{"gain":t["gain"],"feedback":t["feedback"]} for t in trials],
             "evaluations":[evaluate_edits(model,tok,variants,seed*10+i,args.count,k,sub)
                 for i,k in enumerate(("familiar","paraphrase","schema"))]})
        save(out/"completed.json",results)
    return {"pilot":"022","environment":env,"replications":results}

def pilot023(args,out):
    model,tok,env=load_model(MODEL,args.device,REVISION)
    raw,fenced=extraction_pairs(23001,12,"practice")
    other=unrelated_tasks(23001,12,"practice")
    true,fit=fit_rows(model,tok,raw,fenced+other)
    gain,trials=select_gain(model,tok,true,raw,fenced,other)
    # All directions evaluated at the exact same gain, including no-op if selected.
    variants={"true_selected":(true,gain),"true_fixed2":(true,2.)}
    controls={}
    for mode in ("random","shuffle"):
        for seed in (23002,23003,23004):
            edits,diag=fit_rows(model,tok,raw,fenced+other,seed,mode)
            name=f"{mode}_{seed}"
            variants[name]=(edits,2.)
            controls[name]=diag
    for i,e in enumerate(true):
        variants[f"single_{i}"]=([e],2.)
        variants[f"leaveout_{i}"]=([x for j,x in enumerate(true) if j!=i],2.)
    save(out/"practice.json",trials)
    save(out/"locked-patch.json",{"variants":{k:{"gain":g,"edits":serialize(e)} for k,(e,g) in variants.items()},"fit":fit,"controls":controls})
    return {"pilot":"023","environment":env,"selected_gain":gain,
       "evaluations":[evaluate_edits(model,tok,variants,s,args.count,k,out)
           for s,k in ((23009,"familiar"),(23010,"paraphrase"))],
       "caveat":"Fixed gain 2 is predeclared from pilot 020, norm-matched controls use identical rows and scaling. Single/leaveout are not norm matched to full patch."}

def csv_pairs(seed,n,split):
    rng=random.Random(seed);raw=[];fenced=[]
    for i in range(n):
        name=rng.choice(["iris","cedar","maple","birch"]);a=rng.randrange(10,90)
        stem=f"Record: name={name}; quantity={a}. Return one CSV row in name,quantity order, no header."
        raw.append(Task(f"{split}-{seed}-{i}-raw","csv",stem+" Output only the row, no Markdown.",f"{name},{a}"))
        fenced.append(Task(f"{split}-{seed}-{i}-fenced","csv",stem+" Wrap the row in a Markdown code block labeled csv.",chr(96)*3+"csv\n"+f"{name},{a}"+"\n"+chr(96)*3))
    return raw,fenced

def wrapper_failure(t,r):
    s=r["response"].strip().splitlines()
    return len(s)==3 and s[0].startswith(chr(96)*3) and s[-1]==chr(96)*3 and s[1].strip()==t.expected

def csv_score(model,tok,raw,fenced,other):
    return {"raw":exact_rows(model,tok,raw),"fenced":exact_rows(model,tok,fenced),"other":exact_rows(model,tok,other)}

def pilot024(args,out):
    model,tok,env=load_model(MODEL,args.device,REVISION)
    raw,fenced=csv_pairs(24001,18,"practice")
    other=unrelated_tasks(24001,18,"practice")
    base=csv_score(model,tok,raw,fenced,other)
    eligible=sum(wrapper_failure(t,r) for t,r in zip(raw,base["raw"]))
    save(out/"discovery.json",{"eligible":eligible,"examples":base})
    if eligible<3:
        return {"pilot":"024","environment":env,"status":"insufficient_csv_wrapper_failures","eligible":eligible,"n":18}
    edits,fit=fit_rows(model,tok,raw,fenced+other)
    trials=[]
    for g in (0.,.25,.5,1.,2.):
        with gate_weight_patch(model,[(l,n,d*g) for l,n,d in edits]):
            rows=csv_score(model,tok,raw,fenced,other)
        trials.append({"gain":g,"feedback":feedback(rows,base),"examples":rows})
    safe=[t for t in trials if t["feedback"]["losses"]["fenced"]+t["feedback"]["losses"]["other"]==0]
    gain=max(safe,key=lambda t:(t["feedback"]["correct"]["raw"],-t["gain"]))["gain"]
    save(out/"practice.json",trials)
    save(out/"locked-patch.json",{"gain":gain,"edits":serialize(edits),"fit":fit})
    evaluations=[]
    for seed in (24009,24010):
        r,f=csv_pairs(seed,args.count,"heldout");o=unrelated_tasks(seed,args.count,"heldout")
        rows={"base":csv_score(model,tok,r,f,o)}
        with scaled_columns(model,JAC_PATCH):rows["global"]=csv_score(model,tok,r,f,o)
        with gate_weight_patch(model,[(l,n,d*gain) for l,n,d in edits]):rows["contextual"]=csv_score(model,tok,r,f,o)
        # Also test preservation of original JSON behavior on distinct prompts.
        jr,jf=extraction_pairs(seed,args.count,"heldout")
        jsonrows={"base":score_patch(model,tok,[],jr,jf,o)}
        with gate_weight_patch(model,[(l,n,d*gain) for l,n,d in edits]):jsonrows["contextual"]=score_patch(model,tok,[],jr,jf,o)
        save(out/f"examples-{seed}.json",{"csv":rows,"json":jsonrows})
        evaluations.append({"seed":seed,"csv":summarize_rows(rows),"json_regression":summarize_rows(jsonrows)})
    return {"pilot":"024","environment":env,"eligible":eligible,"gain":gain,"evaluations":evaluations,
        "caveat":"Fixed JSON-derived neuron support; CSV wrapper feasibility and exact-string scoring, not broad capability benchmark."}

def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot",required=True,choices=["022","023","024"])
    p.add_argument("--out",required=True)
    p.add_argument("--device",default="cpu")
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--count",type=int,default=24)
    a=p.parse_args()
    if a.count<4 or a.threads<1:p.error("Invalid counts")
    torch.set_num_threads(a.threads);torch.manual_seed(222324)
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    result={"022":pilot022,"023":pilot023,"024":pilot024}[a.pilot](a,out)
    save(out/"summary.json",result)
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
