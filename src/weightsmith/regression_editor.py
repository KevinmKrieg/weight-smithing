"""Pilots 016–018: richer editor feedback, regression constraints, failure discovery."""
from __future__ import annotations
import argparse
import gc
import itertools
import json
import random
from pathlib import Path

from .benchmark import Task, is_correct
from .iterative_editor import (EDITOR, EDITOR_REVISION, extraction_pairs, unrelated_tasks,
    score_patch, feedback, action_space, apply_action, final_evaluation)
from .neuron_basis import REVISION, FENCE, first_step, scaled_columns, save, margin
from .neuron_transfer import constrained_choice, JAC_PATCH, label_tasks
from .neuron_validation import prepare_diagnostics
from .runtime import MODEL, load_model, evaluate

def key(patch):
    return tuple(sorted(tuple(x) for x in patch))

def continuous_feedback(model,tok,patch,raw,fenced,other,base,base_margin):
    rows=score_patch(model,tok,patch,raw,fenced,other)
    f=feedback(rows,base)
    f["regression_losses"]=f["losses"]["fenced"]+f["losses"]["other"]
    with scaled_columns(model,patch):
        logits,_=first_step(model,tok,raw+fenced)
    fence=tok.encode(FENCE,add_special_tokens=False)
    brace=tok.encode("{",add_special_tokens=False)
    if len(fence)!=1 or len(brace)!=1:raise ValueError("Expected single-token margin")
    m=margin(logits,fence[0],brace[0])
    delta=base_margin-m
    f["raw_margin_gain"]=float(delta[:len(raw)].mean())
    f["fence_margin_harm"]=float(delta[len(raw):].clamp_min(0).mean())
    f["per_prompt_margin_gain"]=delta.tolist()
    f["examples"]=[{"id":r["id"],"group":g,"prompt":r["prompt"],
        "expected":r["expected"],"response":r["response"],"correct":r["correct"]}
        for g in ("raw","fenced","other") for r in rows[g]]
    return f

def rank(f):
    return (-f["regression_losses"],f["correct"]["raw"],
            f["raw_margin_gain"]-f["fence_margin_harm"])

def choose_final(history,base):
    # Margin is a search guide, not evidence of actual behavioral improvement.
    valid=[r for r in history if r["feedback"]["regression_losses"]==0
           and r["feedback"]["correct"]["raw"]>base["correct"]["raw"]]
    if not valid:return []
    return max(valid,key=lambda r:(rank(r["feedback"]),-len(r["patch"])))["patch"]

def base_setup(model,tok,seed,n):
    raw,fenced=extraction_pairs(seed,n,"practice")
    other=unrelated_tasks(seed,n,"practice")
    base=score_patch(model,tok,[],raw,fenced,other)
    logits,_=first_step(model,tok,raw+fenced)
    f=tok.encode(FENCE,add_special_tokens=False)
    b=tok.encode("{",add_special_tokens=False)
    if len(f)!=1 or len(b)!=1:raise ValueError("Expected single-token margin")
    return raw,fenced,other,base,margin(logits,f[0],b[0])

def pilot016(args,out):
    model,tok,env=load_model(MODEL,args.device,REVISION)
    report=prepare_diagnostics(model,tok,16001,out)
    candidates=[(r["layer"],r["channel"]) for r in report["candidates"]]
    raw,fenced,other,base,base_margin=base_setup(model,tok,16002,6)
    evidence=[]
    # Identical shared calibration information for all three search arms.
    for l,n in candidates:
        f=continuous_feedback(model,tok,[(l,n,0.)],raw,fenced,other,base,base_margin)
        evidence.append({"layer":l,"channel":n,"feedback":{k:v for k,v in f.items() if k!="examples"}})
    save(out/"shared-calibration.json",evidence)
    del model,tok;gc.collect()
    rng=random.Random(16003)
    variants={"base":[],"jacobian":report["variants"]["jacobian"]}
    histories={}
    for arm in ("editor","greedy","random"):
        history=[];current=[];seen={key([])}
        for trial in range(6):
            actions=[a for a in action_space(candidates,current)
                     if key(apply_action(current,a)) not in seen]
            if not actions:break
            if arm=="editor":
                editor,etok,eenv=load_model(EDITOR,args.device,EDITOR_REVISION)
                prompt=("Choose one allowed weight edit to improve raw JSON extraction while preserving "
                    "required Markdown and unrelated answers. Positive raw margin gain helps, "
                    "positive fence margin harm hurts. Never repeat a tested patch. "
                    "Current patch: "+json.dumps(current)+"\nSingle-neuron practice calibration: "+
                    json.dumps(evidence,separators=(",",":"))+"\nPrior practice trials: "+
                    json.dumps([{**h,"feedback":{**h["feedback"],"examples":h["feedback"]["examples"][:4]}}
                                for h in history[-2:]],separators=(",",":"))+
                    "\nAllowed actions: "+json.dumps(actions,separators=(",",":")))
                choice,ids=constrained_choice(editor,etok,prompt,[json.dumps(a,separators=(",",":")) for a in actions])
                action=json.loads(choice)
                del editor,etok;gc.collect()
            elif arm=="random":
                action=rng.choice(actions);ids=[]
            else:
                # Additive proxy uses exactly the shared single-neuron measurements.
                lookup={(e["layer"],e["channel"]):e["feedback"] for e in evidence}
                def proxy(a):
                    p=apply_action(current,a)
                    return sum((1-s)*(lookup[l,n]["raw_margin_gain"]-
                        lookup[l,n]["fence_margin_harm"]-
                        4*lookup[l,n]["regression_losses"]) for l,n,s in p)
                action=max(actions,key=proxy);ids=[]
            patch=apply_action(current,action);seen.add(key(patch))
            model,tok,_=load_model(MODEL,args.device,REVISION)
            f=continuous_feedback(model,tok,patch,raw,fenced,other,base,base_margin)
            del model,tok;gc.collect()
            history.append({"trial":trial+1,"action":action,"patch":patch,"feedback":f,"tokens":ids})
            # Continue from best exploratory trial, even if exact accuracy has not improved.
            current=max(history,key=lambda h:rank(h["feedback"]))["patch"]
            save(out/f"{arm}-history.json",history)
        variants[arm+"_selected"]=choose_final(history,feedback(base,base))
        variants[arm+"_first"]=history[0]["patch"] if history else []
        histories[arm]=[{**h,"feedback":{k:v for k,v in h["feedback"].items() if k!="examples"}} for h in history]
    save(out/"locked-patch.json",variants)
    model,tok,_=load_model(MODEL,args.device,REVISION)
    return {"pilot":"016","environment":env,"shared_calibration_trials":len(evidence),
        "trial_budget":6,"histories":histories,"variants":variants,
        "evaluation":final_evaluation(model,tok,variants,16009,args.count,out)}

def pareto(rows):
    return [r for r in rows if not any(
        q["feedback"]["correct"]["raw"]>=r["feedback"]["correct"]["raw"]
        and q["feedback"]["regression_losses"]<=r["feedback"]["regression_losses"]
        and (q["feedback"]["correct"]["raw"]>r["feedback"]["correct"]["raw"]
            or q["feedback"]["regression_losses"]<r["feedback"]["regression_losses"])
        for q in rows)]

def pilot017(args,out):
    model,tok,env=load_model(MODEL,args.device,REVISION)
    raw,fenced,other,base,base_margin=base_setup(model,tok,17001,12)
    # Fixed four-column Jacobian support; independently vary attenuation.
    # Screen all 81 combinations by measured practice margin, not generation.
    grid=[]
    f=tok.encode(FENCE,add_special_tokens=False)[0]
    b=tok.encode("{",add_special_tokens=False)[0]
    for scales in itertools.product((0.,.5,1.),repeat=4):
        patch=[(l,n,s) for (l,n,_),s in zip(JAC_PATCH,scales) if s!=1.]
        with scaled_columns(model,patch):
            logits,_=first_step(model,tok,raw+fenced)
        delta=base_margin-margin(logits,f,b)
        grid.append({"patch":patch,"raw_margin_gain":float(delta[:len(raw)].mean()),
                     "fence_margin_harm":float(delta[len(raw):].clamp_min(0).mean())})
    save(out/"margin-grid.json",grid)
    # Predeclared shortlist across three trade-off weights plus endpoints.
    short={key([]):[],key(JAC_PATCH):JAC_PATCH}
    for penalty in (0.,1.,4.):
        for r in sorted(grid,key=lambda r:r["raw_margin_gain"]-penalty*r["fence_margin_harm"],reverse=True)[:4]:
            short[key(r["patch"])]=r["patch"]
    rows=[{"patch":p,"feedback":continuous_feedback(model,tok,p,raw,fenced,other,base,base_margin)}
          for p in short.values()]
    frontier=pareto(rows)
    variants={"base":[],"full_jacobian":JAC_PATCH,
              "zero_loss_selected":choose_final(rows,feedback(base,base))}
    # Lock a second, explicitly relaxed practice budget of one loss.
    relaxed=[r for r in rows if r["feedback"]["regression_losses"]<=1]
    variants["one_loss_selected"]=max(relaxed,key=lambda r:(r["feedback"]["correct"]["raw"],
        -r["feedback"]["regression_losses"],-len(r["patch"])))["patch"]
    save(out/"practice-frontier.json",{"trials":rows,"frontier":frontier})
    save(out/"locked-patch.json",variants)
    return {"pilot":"017","environment":env,"margin_screened":len(grid),
        "generation_evaluated":len(rows),"practice_frontier":[{"patch":r["patch"],
        "raw_correct":r["feedback"]["correct"]["raw"],"regression_losses":r["feedback"]["regression_losses"]} for r in frontier],
        "variants":variants,"evaluations":[final_evaluation(model,tok,variants,s,args.count,out) for s in (17009,17010)]}

def format_tasks(seed,n,split,family):
    rng=random.Random(f"discovery:{seed}:{split}:{family}")
    if family=="json":
        return extraction_pairs(seed,n,split)[0]
    if family=="label":
        return label_tasks(seed,n,split)[0]
    tasks=[]
    for i in range(n):
        a,b=rng.sample(range(11,90),2)
        word=rng.choice(["amber","cedar","delta","silver","violet","willow"])
        if family=="integer":prompt=f"Calculate {a}+{b}. Output only the integer, no words.";expected=str(a+b)
        elif family=="uppercase":prompt=f"Convert {word} to uppercase. Return only the word, no quotes or punctuation.";expected=word.upper()
        elif family=="csv":prompt=f"Return one CSV row containing name {word} and integer {a}, no header, no Markdown, no spaces.";expected=f"{word},{a}"
        else:raise ValueError(family)
        tasks.append(Task(f"{split}-{family}-{seed}-{i}",family,prompt,expected))
    return tasks

def format_only(task,response):
    # Strict, reproducible wrapper-only correction; no substring matching.
    s=response.strip()
    if is_correct(task,s):return False
    candidates=[]
    lines=s.splitlines()
    if len(lines)>=3 and lines[0].startswith(FENCE) and lines[-1].strip()==FENCE:
        candidates.append("\n".join(lines[1:-1]).strip())
    for prefix in ("Answer: ","The answer is ","Here is the answer: ","Result: "):
        if s.startswith(prefix):candidates.append(s[len(prefix):].strip())
    if len(s)>=2 and s[0]==s[-1] and s[0] in ("'",'"'):candidates.append(s[1:-1])
    if s.endswith("."):candidates.append(s[:-1])
    return any(is_correct(task,v) for v in candidates)

def pilot018(args,out):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    checkpoints=[("HuggingFaceTB/SmolLM2-360M-Instruct","a10cc1512eabd3dde888204e902eca88bddb4951"),
                 ("HuggingFaceTB/SmolLM2-135M-Instruct",None)]
    findings=[]
    for checkpoint,revision in checkpoints:
        kwargs={"revision":revision} if revision else {}
        tok=AutoTokenizer.from_pretrained(checkpoint,trust_remote_code=False,**kwargs)
        model=AutoModelForCausalLM.from_pretrained(checkpoint,torch_dtype=torch.float32,trust_remote_code=False,**kwargs).to(args.device).eval()
        for family in ("json","label","integer","uppercase","csv"):
            tasks=format_tasks(18001,18,"discovery",family)
            rows=evaluate(model,tok,tasks,80)
            count=sum(format_only(t,r["response"]) for t,r in zip(tasks,rows))
            finding={"checkpoint":checkpoint,"revision":getattr(model.config,"_commit_hash",None),
                "family":family,"n":len(tasks),"correct":sum(r["correct"] for r in rows),
                "format_only":count,"discovery_examples":rows}
            # Confirm on a new sample only after passing the fixed discovery threshold.
            if count>=3:
                confirm=format_tasks(18002,24,"confirmation",family)
                vals=evaluate(model,tok,confirm,80)
                finding["confirmation_count"]=sum(format_only(t,r["response"]) for t,r in zip(confirm,vals))
                finding["confirmation_examples"]=vals
                finding["eligible_for_future_repair"]=finding["confirmation_count"]>=3
            findings.append(finding)
            save(out/"discovery.json",findings)
        del model,tok;gc.collect()
    return {"pilot":"018","findings":[{k:v for k,v in r.items() if not k.endswith("examples")} for r in findings],
        "caveat":"Discovery and independent confirmation only; no weight edits or transfer claims."}

def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot",required=True,choices=["016","017","018"])
    p.add_argument("--out",required=True)
    p.add_argument("--device",default="cpu")
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--count",type=int,default=24)
    a=p.parse_args()
    if a.count<6 or a.threads<1:p.error("Invalid counts")
    torch.set_num_threads(a.threads);torch.manual_seed(161718)
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    result={"016":pilot016,"017":pilot017,"018":pilot018}[a.pilot](a,out)
    save(out/"summary.json",result)
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
