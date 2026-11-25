"""Pilots 007–009: locked-patch transfer, local Jacobian readouts and autonomous editing."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import random

from .benchmark import Task, make_tasks, paired_bootstrap, summarize
from .neuron_basis import (FENCE, REVISION, candidate_rows, chat_ids, choose_causal,
                           fenced_eval, first_step, margin, save, scaled_columns)
from .patch import validate_edits
from .runtime import MODEL, evaluate, generate_one, load_model

FIXED = [(23,641,0.), (23,1057,0.), (20,266,0.), (20,4752,0.)]
ACTIVATION = [(20,467,0.), (20,4474,0.), (23,1396,0.), (23,641,0.)]
RANDOM = [(20,975,0.), (20,4544,0.), (23,1669,0.), (23,4252,0.)]
LAYERS = [12,16,20,23]
EDITOR = "Qwen/Qwen2.5-1.5B-Instruct"

def diverse_pairs(seed, count, split):
    rng = random.Random(f"neuron-validation:{seed}:{split}")
    raw, fenced = [], []
    for i in range(count):
        name = rng.choice(["Mira","Omar","Sofia","Noah","Lina"])
        number = rng.randrange(100,999)
        enabled = bool(rng.randrange(2))
        data = [
            {"name":name,"score":number},
            {"item":name,"active":enabled},
            {"id":number,"tags":["red","blue"]},
            {"record":{"name":name,"id":number},"ok":enabled},
            {"city":name,"count":number,"ready":enabled},
            {"first":number,"second":number+1},
        ][i%6]
        expected = json.dumps(data, ensure_ascii=False, separators=(",",":"))
        stem = [
            f"Serialize this object to JSON: {expected}.",
            f"An API expects the following object: {expected}.",
            f"Convert the following data into a JSON object: {expected}.",
            f"Return a JSON representation of this record: {expected}.",
            f"Produce exactly the object {expected} in valid JSON.",
            f"Reproduce this object as JSON, preserving values and types: {expected}.",
        ][i%6]
        no_md = ["Output only the JSON, with no Markdown.", "The response must be directly parseable by json.loads; no code fences.", "Start with the opening curly brace and end with the closing curly brace. Do not add formatting."][i%3]
        yes_md = ["Wrap the JSON in a Markdown code fence labeled json. No surrounding text.", "Return a fenced Markdown json code block only.", "Use a triple-backtick JSON code block with the json language label. No other text."][i%3]
        key = f"{split}-{seed}-{i}"
        raw.append(Task(key+"-raw","json",stem+" "+no_md,expected))
        fenced.append(Task(key+"-fence","fenced_json",stem+" "+yes_md,expected))
    return raw,fenced

def evaluate_variants(model,tokenizer,variants,seed,count,out,max_tokens=80):
    raw,fenced = diverse_pairs(seed,count,"evaluation")
    unrelated = [t for t in make_tasks("heldout",count//3 or 1,seed) if t.family!="json"]
    tasks = raw+unrelated
    results,reg = {},{}
    for name,edits in variants.items():
        with scaled_columns(model,edits):
            results[name]=evaluate(model,tokenizer,tasks,max_tokens)
            reg[name]=fenced_eval(model,tokenizer,fenced,max_tokens)
    summary = {"seed":seed,"raw_json_n":len(raw),"unrelated_n":len(unrelated),
        "required_fences_n":len(fenced),
        "heldout":{k:summarize(v) for k,v in results.items()},
        "vs_base":{k:paired_bootstrap(results["base"],v) for k,v in results.items() if k!="base"},
        "required_fences":{k:sum(r["correct"] for r in v)/len(v) for k,v in reg.items()},
        "fence_vs_base":{k:paired_bootstrap(reg["base"],v) for k,v in reg.items() if k!="base"}}
    save(out/f"examples-{seed}.json",{"heldout":results,"fenced":reg})
    return summary

def jacobian_scores(model,tokenizer,tasks,layers,fence,brace):
    """Target-specific local derivative of next-token margin, all prompt positions.
    Derivatives diagnose the model; no weight gradients or optimizer steps.
    """
    import torch
    params=list(model.parameters())
    flags=[p.requires_grad for p in params]
    for p in params: p.requires_grad_(False)
    hooks=[]
    inputs,outputs={},{}
    scores={l:[] for l in layers}
    try:
        def enable_input(module,args,out):
            return out.detach().requires_grad_(True)
        hooks.append(model.model.embed_tokens.register_forward_hook(enable_input))
        for l in layers:
            def capture(module,args,out,layer=l):
                inputs[layer]=args[0].detach()
                outputs[layer]=out
            hooks.append(model.model.layers[l].mlp.down_proj.register_forward_hook(capture))
        with torch.enable_grad():
            for task in tasks:
                h=model.model(input_ids=chat_ids(model,tokenizer,task),use_cache=False).last_hidden_state[0,-1]
                u=model.lm_head.weight.detach()
                scalar=(h*(u[fence]-u[brace])).sum()
                gs=torch.autograd.grad(scalar,[outputs[l] for l in layers])
                for l,g in zip(layers,gs):
                    w=model.model.layers[l].mlp.down_proj.weight.detach()
                    # d margin / d column scale = sum_t a[t,i]*(g[t] dot w[:,i])
                    s=(inputs[l].float()*(g.float()@w.float())).sum(dim=(0,1))
                    scores[l].append(s.detach().cpu())
                inputs.clear(); outputs.clear()
    finally:
        for hook in hooks: hook.remove()
        for p,flag in zip(params,flags): p.requires_grad_(flag)
    return {l:torch.stack(v) for l,v in scores.items()}

def choose_ranked(scores,raw_n,budget=4):
    ranked=[]
    for l,s in scores.items():
        raw=s[:raw_n].mean(0)
        fenced=s[raw_n:].mean(0)
        score=raw-fenced.clamp_min(0)
        for n in score.topk(budget).indices.tolist():
            if raw[n]>0 and score[n]>0:
                ranked.append((float(score[n]),l,n))
    return [(l,n,0.) for _,l,n in sorted(ranked,reverse=True)[:budget]]

def prepare_diagnostics(model,tokenizer,seed,out):
    raw,fenced=diverse_pairs(seed,6,"practice")
    tasks=raw+fenced
    fence=tokenizer.encode(FENCE,add_special_tokens=False)
    brace=tokenizer.encode("{",add_special_tokens=False)
    if len(fence)!=1 or len(brace)!=1: raise ValueError("Token assumptions failed")
    f,b=fence[0],brace[0]
    base,acts=first_step(model,tokenizer,tasks,LAYERS)
    direct={}
    u=model.lm_head.weight.detach()
    for l in LAYERS:
        w=model.model.layers[l].mlp.down_proj.weight.detach()
        direct[l]=acts[l]*((u[f]-u[b]).float()@w.float()).cpu()
    jac=jacobian_scores(model,tokenizer,tasks,LAYERS,f,b)
    direct_patch=choose_ranked(direct,len(raw))
    jac_patch=choose_ranked(jac,len(raw))
    # Compare on common, preselected union of channels; both methods get same causal tests.
    pairs=sorted({(l,n) for l,n,_ in direct_patch+jac_patch})
    rows=[]
    base_margin=margin(base,f,b)
    for l,n in pairs:
        with scaled_columns(model,[(l,n,0.)]):
            ablated,_=first_step(model,tokenizer,tasks)
        actual=base_margin-margin(ablated,f,b)
        row={"layer":l,"channel":n,"raw_reduction":float(actual[:len(raw)].mean()),
            "fenced_reduction":float(actual[len(raw):].mean()),
            "direct_prediction":direct[l][:,n].tolist(),"jacobian_prediction":jac[l][:,n].tolist(),
            "actual_reduction":actual.tolist()}
        row["causal_score"]=row["raw_reduction"]-max(0.,row["fenced_reduction"])
        diff=(base-ablated)[:len(raw)].mean(0)
        row["promoted_tokens"]=[tokenizer.decode([i]) for i in diff.topk(5).indices.tolist()]
        row["suppressed_tokens"]=[tokenizer.decode([i]) for i in (-diff).topk(5).indices.tolist()]
        rows.append(row)
    causal=[(r["layer"],r["channel"],0.) for r in choose_causal(rows,4)]
    import numpy as np
    metrics={}
    actual=np.array([v for r in rows for v in r["actual_reduction"]])
    for method in ["direct_prediction","jacobian_prediction"]:
        pred=np.array([v for r in rows for v in r[method]])
        corr=float(np.corrcoef(pred,actual)[0,1]) if len(actual)>1 and np.std(pred)>0 and np.std(actual)>0 else None
        metrics[method]={"mae":float(np.mean(abs(pred-actual))) if len(actual) else None,"pearson":corr}
    report={"practice_seed":seed,"practice":[t.asdict() for t in tasks],
        "candidates":rows,"prediction_metrics":metrics,
        "variants":{"base":[],"direct_lens":direct_patch,"jacobian":jac_patch,"causal_union":causal},
        "caveat":"Target-specific local Jacobian at one margin; not full-vocabulary averaged J-lens."}
    save(out/"diagnostics.json",report)
    return report

def editor_prompt(report):
    # Opaque heldout dataset absent. No code execution, one attempt, bounded columns only.
    return ("Choose up to four listed MLP neurons to attenuate in another language model. "
            "Goal: fix unwanted JSON Markdown fences while preserving required fences. "
            "Positive raw_reduction helps; positive fenced_reduction may harm required fences. "
            "Return ONLY JSON with exactly keys edits and rationale. Each edit has integer layer, "
            "integer channel, and multiplier 0, 0.25, 0.5 or 0.75. No other keys. "
            "Example: "+json.dumps({"edits":[{"layer":23,"channel":123,"multiplier":0.5}],"rationale":"brief reason"})+
            "\nPractice evidence:\n"+json.dumps(report["candidates"],separators=(",",":")))

def parse_editor(text,model,report):
    raw=json.loads(text.strip())
    edits=validate_edits(model,raw,max_columns=4,allowed_layers=tuple(LAYERS))
    allowed={(r["layer"],r["channel"]) for r in report["candidates"]}
    if any((e.layer,e.channel) not in allowed or e.multiplier not in (0.,.25,.5,.75) for e in edits):
        raise ValueError("Editor selected unavailable candidate or multiplier")
    return [(e.layer,e.channel,e.multiplier) for e in edits]

def run(args):
    import torch
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out=Path(args.out)
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    model,tokenizer,env=load_model(MODEL,args.device,REVISION)
    save(out/"environment.json",env)
    if args.pilot=="007":
        variants={"base":[],"fixed_causal":FIXED,"fixed_activation":ACTIVATION,"fixed_random":RANDOM}
        save(out/"locked-patch.json",variants)
        result={"pilot":"007","variants":variants,"seeds":[
            evaluate_variants(model,tokenizer,variants,s,args.count,out) for s in [90701,90702]]}
    else:
        report=prepare_diagnostics(model,tokenizer,args.seed,out)
        variants=report["variants"]
        if args.pilot=="009":
            prompt=editor_prompt(report)
            save(out/"editor-input.json",{"prompt":prompt})
            # Release target model to bound memory; the editor is a distinct checkpoint.
            del model,tokenizer
            gc.collect()
            editor,etok,eenv=load_model(EDITOR,args.device,args.editor_revision)
            task=Task("editor","editor",prompt,"")
            response=generate_one(editor,etok,task,384)
            save(out/"editor-output.json",{"environment":eenv,"response":response})
            del editor,etok
            gc.collect()
            model,tokenizer,_=load_model(MODEL,args.device,REVISION)
            try:
                variants["autonomous_editor"]=parse_editor(response["response"],model,report)
            except (ValueError,TypeError,KeyError) as exc:
                result={"pilot":"009","status":"invalid_editor_proposal","error":str(exc),
                        "caveat":"No fallback or manual repair; no autonomous patch evaluated."}
                save(out/"summary.json",result)
                return result
        save(out/"locked-patch.json",variants)
        result={"pilot":args.pilot,"variants":variants,"prediction_metrics":report["prediction_metrics"],
                "evaluation":evaluate_variants(model,tokenizer,variants,90801 if args.pilot=="008" else 90901,args.count,out)}
    save(out/"summary.json",result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot",required=True,choices=["007","008","009"])
    p.add_argument("--device",default="cpu")
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--count",type=int,default=24)
    p.add_argument("--seed",type=int,default=80801)
    p.add_argument("--editor-revision",default=None)
    p.add_argument("--out",required=True)
    args=p.parse_args()
    if args.count<3 or args.threads<1: p.error("Invalid counts")
    print(json.dumps(run(args),indent=2))

if __name__=="__main__": main()
