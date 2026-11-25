"""Pilots 019–021: oracle gating, context-sensitive weights, replicated editor search."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import random

from .benchmark import paired_bootstrap
from .iterative_editor import (EDITOR, EDITOR_REVISION, extraction_pairs, unrelated_tasks,
    score_patch, feedback, action_space, apply_action, final_evaluation, exact_rows)
from .neuron_basis import REVISION, FENCE, chat_ids, scaled_columns, save, fenced_eval
from .neuron_transfer import JAC_PATCH, constrained_choice
from .regression_editor import base_setup, continuous_feedback, choose_final, rank, key
from .neuron_validation import prepare_diagnostics
from .runtime import MODEL, load_model, evaluate

def gate_decision(prompt):
    """Text-only router; never consult task family, labels or expected output."""
    p=prompt.lower()
    if "no markdown" in p or "without markdown" in p or "no code fence" in p:
        return "json" in p
    return False

def summarize_rows(rows):
    return {"correct":{name:{g:sum(r["correct"] for r in v) for g,v in groups.items()}
                       for name,groups in rows.items()},
            "paired":{name:{g:paired_bootstrap(rows["base"][g],v) for g,v in groups.items()}
                      for name,groups in rows.items() if name!="base"}}

def novel_pairs(seed,n):
    raw,fenced=extraction_pairs(seed,n,"heldout")
    from .benchmark import Task
    raw_suffixes=[" Output the JSON object directly, without Markdown.",
        " The receiving parser expects raw JSON, no code fence.",
        " Respond with the bare JSON object only.",
        " Do not surround the JSON object with triple backticks."]
    fenced_suffixes=[" Put the answer in a Markdown code block labeled json.",
        " Return fenced JSON using triple backticks.",
        " Use a JSON code fence; no text outside.",
        " The response must be a Markdown json block."]
    def stem(t):return t.prompt.rsplit(" Return only",1)[0]
    return ([Task(t.id,t.family,stem(t)+raw_suffixes[i%4],t.expected) for i,t in enumerate(raw)],
            [Task(t.id,t.family,stem(t)+fenced_suffixes[i%4],t.expected) for i,t in enumerate(fenced)])

def routed_rows(model,tok,raw,fenced,other,oracle=False):
    rows={}
    for g,tasks in (("raw",raw),("fenced",fenced),("other",other)):
        output=[]
        for t in tasks:
            on=(g=="raw") if oracle else gate_decision(t.prompt)
            with scaled_columns(model,JAC_PATCH if on else []):
                fn=fenced_eval if g=="fenced" else exact_rows if g=="other" else evaluate
                output.extend(fn(model,tok,[t],80))
        rows[g]=output
    return rows

def pilot019(args,out):
    model,tok,env=load_model(MODEL,args.device,REVISION)
    results=[]
    for seed,novel in ((19009,False),(19010,True)):
        raw,fenced=novel_pairs(seed,args.count) if novel else extraction_pairs(seed,args.count,"heldout")
        other=unrelated_tasks(seed,args.count,"heldout")
        rows={"base":score_patch(model,tok,[],raw,fenced,other),
              "global":score_patch(model,tok,JAC_PATCH,raw,fenced,other),
              "oracle":routed_rows(model,tok,raw,fenced,other,True),
              "text_router":routed_rows(model,tok,raw,fenced,other)}
        save(out/f"examples-{seed}.json",rows)
        results.append({"seed":seed,"novel_instruction_templates":novel,
                        "n_per_group":args.count,
                        "router_positive":{g:sum(gate_decision(t.prompt) for t in ts)
                            for g,ts in (("raw",raw),("fenced",fenced),("other",other))},
                        **summarize_rows(rows)})
    return {"pilot":"019","environment":env,"evaluations":results,
        "caveat":"Oracle uses task type; text router uses explicit prompt cues. Both are runtime conditional interventions, not weight-only repairs."}

def gate_inputs(model,tok,tasks,layers):
    import torch
    collected={l:[] for l in layers};hooks=[]
    for l in layers:
        def capture(m,a,layer=l):
            collected[layer].append(a[0][0,-1].detach().float().cpu())
        hooks.append(model.model.layers[l].mlp.gate_proj.register_forward_pre_hook(capture))
    try:
        with torch.inference_mode():
            for t in tasks:
                model.model(input_ids=chat_ids(model,tok,t),use_cache=False)
    finally:
        for h in hooks:h.remove()
    return {l:torch.stack(v) for l,v in collected.items()}

def fit_direction(x,y,ridge=1.):
    """Dual ridge solve for a no-intercept readout; fit only practice representations."""
    import torch
    x=x.float();y=y.float()
    return x.T@torch.linalg.solve(x@x.T+ridge*torch.eye(len(x)),y)

@contextmanager
def gate_weight_patch(model,edits):
    """Actual gate_proj row additions; no inference-time hooks."""
    import torch
    saved=[]
    with torch.no_grad():
        try:
            for l,n,d in edits:
                w=model.model.layers[l].mlp.gate_proj.weight
                saved.append((w,n,w[n].clone()))
                w[n].add_(d.to(device=w.device,dtype=w.dtype))
            yield
        finally:
            for w,n,old in reversed(saved):w[n].copy_(old)

def pilot020(args,out):
    import torch
    model,tok,env=load_model(MODEL,args.device,REVISION)
    layers=sorted({l for l,n,s in JAC_PATCH})
    raw,fenced=extraction_pairs(20001,12,"practice")
    other=unrelated_tasks(20001,12,"practice")
    inputs=gate_inputs(model,tok,raw+fenced+other,layers)
    # Desired gate-preactivation shift: negative on raw contexts, zero otherwise.
    # This is not an exact SiLU suppression objective; validate behavior directly.
    y=torch.tensor([-1.]*len(raw)+[0.]*(len(fenced)+len(other)))
    dirs={l:fit_direction(inputs[l],y) for l in layers}
    evidence={str(l):{"practice_predictions":(inputs[l]@dirs[l]).tolist(),
                       "direction_norm":float(dirs[l].norm())} for l in layers}
    save(out/"direction-fit.json",evidence)
    base=score_patch(model,tok,[],raw,fenced,other)
    practice=[]
    # Common gain and per-row sign chosen from original preactivation on raw examples.
    edits=[]
    for l,n,_ in JAC_PATCH:
        w=model.model.layers[l].mlp.gate_proj.weight[n].detach().float().cpu()
        median=float((inputs[l][:len(raw)]@w).median())
        # Negative shift drives positive gates toward zero; negative gates receive
        # positive shift instead. This heuristic may cross zero at larger gains.
        sign=1. if median>=0 else -1.
        edits.append((l,n,dirs[l]*sign*max(abs(median),.1)))
    for gain in (0.,.25,.5,1.,2.):
        with gate_weight_patch(model,[(l,n,d*gain) for l,n,d in edits]):
            rows=score_patch(model,tok,[],raw,fenced,other)
        f=feedback(rows,base)
        practice.append({"gain":gain,"feedback":f,"examples":rows})
    valid=[r for r in practice if r["feedback"]["losses"]["fenced"]+r["feedback"]["losses"]["other"]==0]
    selected=max(valid,key=lambda r:(r["feedback"]["correct"]["raw"],-r["gain"]))["gain"]
    unconstrained=max(practice,key=lambda r:(r["feedback"]["correct"]["raw"],
        -r["feedback"]["losses"]["fenced"]-r["feedback"]["losses"]["other"],-r["gain"]))["gain"]
    save(out/"practice.json",practice)
    save(out/"locked-patch.json",{"selected_gain":selected,"unconstrained_gain":unconstrained,
        "rows":[{"layer":l,"channel":n,"delta_per_unit_gain":d.tolist()} for l,n,d in edits]})
    results=[]
    for seed,novel in ((20009,False),(20010,True)):
        r,f=novel_pairs(seed,args.count) if novel else extraction_pairs(seed,args.count,"heldout")
        o=unrelated_tasks(seed,args.count,"heldout")
        rows={"base":score_patch(model,tok,[],r,f,o),
              "global":score_patch(model,tok,JAC_PATCH,r,f,o)}
        for name,gain in (("weight_selected",selected),("weight_unconstrained",unconstrained)):
            with gate_weight_patch(model,[(l,n,d*gain) for l,n,d in edits]):
                rows[name]=score_patch(model,tok,[],r,f,o)
        save(out/f"examples-{seed}.json",rows)
        results.append({"seed":seed,"novel_instruction_templates":novel,
                        "n_per_group":args.count,**summarize_rows(rows)})
    return {"pilot":"020","environment":env,"fit":evidence,"selected_gain":selected,
        "unconstrained_gain":unconstrained,
        "practice":[{k:v for k,v in p.items() if k!="examples"} for p in practice],
        "evaluations":results,
        "caveat":"Practice-only closed-form gate-row direction, not gradient-free learning in the broad sense. No runtime gate. Last-position fit may not transfer to later generation positions."}

def pilot021(args,out):
    """Three independent practice/evaluation splits; equal search budgets per arm."""
    all_results=[]
    for seed in (21001,21002,21003):
        sub=out/str(seed);sub.mkdir()
        model,tok,env=load_model(MODEL,args.device,REVISION)
        report=prepare_diagnostics(model,tok,seed,sub)
        candidates=[(r["layer"],r["channel"]) for r in report["candidates"]]
        raw,fenced,other,base,bm=base_setup(model,tok,seed+100,12)
        evidence=[]
        for l,n in candidates:
            f=continuous_feedback(model,tok,[(l,n,0.)],raw,fenced,other,base,bm)
            evidence.append({"layer":l,"channel":n,"feedback":{k:v for k,v in f.items() if k!="examples"}})
        save(sub/"calibration.json",evidence)
        del model,tok;gc.collect()
        lookup={(e["layer"],e["channel"]):e["feedback"] for e in evidence}
        rng=random.Random(seed)
        variants={"base":[],"jacobian":report["variants"]["jacobian"]}
        histories={}
        for arm in ("editor","greedy","random"):
            history=[];current=[];seen={key([])}
            for trial in range(6):
                actions=[a for a in action_space(candidates,current) if key(apply_action(current,a)) not in seen]
                if not actions:break
                ids=[]
                if arm=="editor":
                    editor,etok,_=load_model(EDITOR,args.device,EDITOR_REVISION)
                    prompt=("Choose one weight edit; improve raw JSON while preserving fenced JSON and unrelated tasks. "
                        "Positive raw_margin_gain helps; positive fence_margin_harm hurts. Current: "+json.dumps(current)+
                        "\nCalibration: "+json.dumps(evidence,separators=(",",":"))+
                        "\nRecent practice: "+json.dumps([{**h,"feedback":{**h["feedback"],"examples":h["feedback"]["examples"][:4]}}
                            for h in history[-2:]],separators=(",",":"))+
                        "\nAllowed actions: "+json.dumps(actions,separators=(",",":")))
                    chosen,ids=constrained_choice(editor,etok,prompt,[json.dumps(a,separators=(",",":")) for a in actions])
                    action=json.loads(chosen)
                    del editor,etok;gc.collect()
                elif arm=="random":action=rng.choice(actions)
                else:
                    def proxy(a):
                        return sum((1-s)*(lookup[l,n]["raw_margin_gain"]-lookup[l,n]["fence_margin_harm"]-
                            4*lookup[l,n]["regression_losses"]) for l,n,s in apply_action(current,a))
                    action=max(actions,key=proxy)
                p=apply_action(current,action);seen.add(key(p))
                model,tok,_=load_model(MODEL,args.device,REVISION)
                f=continuous_feedback(model,tok,p,raw,fenced,other,base,bm)
                del model,tok;gc.collect()
                history.append({"trial":trial+1,"action":action,"patch":p,"feedback":f,"tokens":ids})
                current=max(history,key=lambda h:rank(h["feedback"]))["patch"]
                save(sub/f"{arm}-history.json",history)
            variants[arm+"_selected"]=choose_final(history,feedback(base,base))
            histories[arm]=[{**h,"feedback":{k:v for k,v in h["feedback"].items() if k!="examples"}} for h in history]
        save(sub/"locked-patch.json",variants)
        model,tok,_=load_model(MODEL,args.device,REVISION)
        result={"seed":seed,"variants":variants,"histories":histories,
                "calibration_trials":len(evidence),"budget_per_arm":6,
                "evaluation":final_evaluation(model,tok,variants,seed+900,args.count,sub)}
        all_results.append(result)
        save(out/"completed-seeds.json",all_results)
        del model,tok;gc.collect()
    return {"pilot":"021","replications":all_results,
        "caveat":"Three seeds only; correlated synthetic templates, descriptive comparisons, not a powered superiority test."}

def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot",required=True,choices=["019","020","021"])
    p.add_argument("--out",required=True)
    p.add_argument("--device",default="cpu")
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--count",type=int,default=24)
    a=p.parse_args()
    if a.count<6 or a.threads<1:p.error("Invalid counts")
    torch.set_num_threads(a.threads);torch.manual_seed(192021)
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    result={"019":pilot019,"020":pilot020,"021":pilot021}[a.pilot](a,out)
    save(out/"summary.json",result)
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
