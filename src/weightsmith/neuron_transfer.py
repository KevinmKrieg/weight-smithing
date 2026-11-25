"""Pilots 010–012: constrained editor, partial attenuation, and cross-model repair."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import random

from .benchmark import Task, make_tasks, paired_bootstrap, summarize
from .neuron_basis import first_step, scaled_columns, save, fenced_eval, chat_ids
from .neuron_validation import (FIXED, REVISION, LAYERS, diverse_pairs, evaluate_variants,
                                prepare_diagnostics, parse_editor, editor_prompt)
from .runtime import MODEL, load_model, evaluate, generate_one

EDITORS = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-Coder-1.5B-Instruct"]
SCALES = [0., .25, .5, .75, 1.]
JAC_PATCH = [(23,641,0.), (16,3532,0.), (23,2959,0.), (12,765,0.)]

def allowed_next(sequences, prefix):
    return sorted({s[len(prefix)] for s in sequences
                   if len(s)>len(prefix) and s[:len(prefix)]==prefix})

def constrained_choice(model, tokenizer, prompt, options):
    """Greedy finite JSON grammar; only tokens on a valid completion path allowed."""
    import torch
    ids = chat_ids(model,tokenizer,Task("editor","editor",prompt,""))
    eos = tokenizer.eos_token_id
    sequences = {tuple(tokenizer.encode(o,add_special_tokens=False)+[eos]):o for o in options}
    if len(sequences)!=len(options): raise ValueError("Ambiguous grammar encodings")
    prefix=[]
    # Cache the prompt and each emitted token; mask by grammar, never modify model weights.
    with torch.inference_mode():
        result=model(input_ids=ids,use_cache=True)
        for _ in range(max(map(len,sequences))):
            candidates=allowed_next(list(sequences),tuple(prefix))
            if not candidates: raise ValueError("Grammar dead end")
            logits=result.logits[0,-1,candidates]
            token=candidates[int(logits.argmax())]
            prefix.append(token)
            if tuple(prefix) in sequences: return sequences[tuple(prefix)],prefix
            result=model(input_ids=torch.tensor([[token]],device=ids.device),
                         past_key_values=result.past_key_values,use_cache=True)
    raise ValueError("Grammar did not terminate")

def grammar_options(rows,used):
    options=[json.dumps({"stop":True},separators=(",",":"))]
    for r in rows:
        l,n=r["layer"],r["channel"]
        if (l,n) in used: continue
        for m in (0.,.25,.5,.75):
            options.append(json.dumps({"layer":l,"channel":n,"multiplier":m},separators=(",",":")))
    return options

def constrained_editor(model,tok,report):
    edits,trace=[],[]
    rows=report["candidates"]
    compact=[{k:r[k] for k in ("layer","channel","raw_reduction","fenced_reduction",
                                "causal_score","promoted_tokens","suppressed_tokens")} for r in rows]
    for i in range(4):
        used={(e["layer"],e["channel"]) for e in edits}
        opts=grammar_options(rows,used)
        prompt=("You are editing another language model's weights. Choose one additional neuron "
                "and attenuation multiplier to reduce unwanted JSON Markdown while preserving required "
                "Markdown. Positive raw_reduction helps; positive fenced_reduction can harm. "
                "Choose only from candidates; multiplier 0, 0.25, 0.5, or 0.75. "
                "Return one JSON edit object or {\"stop\":true}. "
                "Stop if further edits are not worthwhile. Already chosen: "+json.dumps(edits)+
                "\nPractice evidence: "+json.dumps(compact,separators=(",",":")))
        choice,ids=constrained_choice(model,tok,prompt,opts)
        trace.append({"prompt":prompt,"allowed":opts,"chosen":choice,"token_ids":ids})
        obj=json.loads(choice)
        if obj.get("stop"): break
        edits.append(obj)
    return {"edits":edits,"rationale":"Sequential constrained editor choices; no generated rationale."},trace

def pilot010(args,out):
    target,tok,env=load_model(MODEL,args.device,REVISION)
    save(out/"target-environment.json",env)
    report=prepare_diagnostics(target,tok,101001,out)
    variants=dict(report["variants"])
    del target,tok
    gc.collect()
    proposals={}
    for i,checkpoint in enumerate(EDITORS):
        editor,etok,eenv=load_model(checkpoint,args.device,None)
        raw=generate_one(editor,etok,Task("raw","editor",editor_prompt(report),""),384)
        proposal,trace=constrained_editor(editor,etok,report)
        proposals[f"editor_{i}"]={"environment":eenv,"raw":raw,"constrained":proposal,"trace":trace}
        save(out/"proposals.json",proposals)
        del editor,etok
        gc.collect()
    target,tok,_=load_model(MODEL,args.device,REVISION)
    for name,p in proposals.items():
        try:
            variants[name+"_raw"]=parse_editor(p["raw"]["response"],target,report)
            p["raw_status"]="valid"
        except (ValueError,TypeError,KeyError) as exc:
            p["raw_status"]="invalid";p["raw_error"]=str(exc)
        proposal=p["constrained"]
        variants[name+"_constrained"]=parse_editor(json.dumps(proposal),target,report) if proposal["edits"] else []
        p["constrained_status"]="valid" if proposal["edits"] else "abstained"
    save(out/"proposals.json",proposals)
    save(out/"locked-patch.json",variants)
    return {"pilot":"010","proposals":{n:{"environment":p["environment"],"raw_status":p["raw_status"],
              "constrained_status":p["constrained_status"],"constrained":p["constrained"]} for n,p in proposals.items()},
            "variants":variants,"evaluation":evaluate_variants(target,tok,variants,101009,args.count,out)}

def scaled(patch,scale):
    return [(l,n,scale) for l,n,_ in patch] if scale!=1. else []

def choose_scale(rows):
    """Prefer no observed practice regression, then raw gains, then minimal intervention."""
    baseline=next(r for r in rows if r["scale"]==1.)
    eligible=[r for r in rows if r["fenced_losses"]==0 and r["unrelated_losses"]==0
              and r["raw_correct"]>=baseline["raw_correct"]]
    return max(eligible,key=lambda r:(r["raw_correct"],r["scale"]))["scale"]

def pilot011(args,out):
    model,tok,env=load_model(MODEL,args.device,REVISION)
    raw,fenced=diverse_pairs(111001,12,"practice")
    unrelated=[t for t in make_tasks("practice",4,111001) if t.family!="json"]
    base_f=fenced_eval(model,tok,fenced,80);base_u=evaluate(model,tok,unrelated,80)
    variants={"base":[]}
    practice={}
    for name,patch in [("original",FIXED),("jacobian",JAC_PATCH)]:
        rows=[]
        for s in SCALES:
            with scaled_columns(model,scaled(patch,s)):
                r=evaluate(model,tok,raw,80)
                f=fenced_eval(model,tok,fenced,80)
                u=evaluate(model,tok,unrelated,80)
            rows.append({"scale":s,"raw_correct":sum(x["correct"] for x in r),
                         "fenced_correct":sum(x["correct"] for x in f),
                         "unrelated_correct":sum(x["correct"] for x in u),
                         "fenced_losses":sum(a["correct"] and not b["correct"] for a,b in zip(base_f,f)),
                         "unrelated_losses":sum(a["correct"] and not b["correct"] for a,b in zip(base_u,u)),
                         "examples":{"raw":r,"fenced":f,"unrelated":u}})
        chosen=choose_scale(rows)
        practice[name]={"chosen_scale":chosen,"rows":rows}
        variants[name+"_selected"]=scaled(patch,chosen)
        variants[name+"_full"]=scaled(patch,0.)
    save(out/"practice.json",practice)
    save(out/"locked-patch.json",variants)
    return {"pilot":"011","environment":env,"selected_scales":{n:p["chosen_scale"] for n,p in practice.items()},
            "variants":variants,"evaluation":evaluate_variants(model,tok,variants,111009,args.count,out)}

def label_tasks(seed,n,split):
    """Exact single-label extraction, paired with a legitimate sentence response."""
    rng=random.Random(f"label:{seed}:{split}")
    labels=["red","blue","green","yellow","orange","purple","black","white"]
    raw,reg=[],[]
    for i in range(n):
        a,b=rng.sample(labels,2)
        key=f"{split}-{seed}-{i}"
        stem=[f"The first marker is {a} and the second is {b}. What color is the first marker?",
              f"A package has a {a} sticker on top and a {b} sticker on the bottom. Which color is on top?",
              f"Record: primary={a}; secondary={b}. What is the primary color?",
              f"Among the labels {a}, {b}, identify the first label."][i%4]
        raw.append(Task(key+"-raw","label",stem+" Return only the lowercase color word, no punctuation.",a))
        reg.append(Task(key+"-sentence","sentence",stem+" Reply exactly in the form 'The color is COLOR.'",f"The color is {a}."))
    return raw,reg

def margin_derivatives(model,tok,tasks,token_pairs,layers):
    """One scalar bad-vs-good logit margin per prompt; all-position column derivatives."""
    import torch
    params=list(model.parameters()); flags=[p.requires_grad for p in params]
    for p in params:p.requires_grad_(False)
    ins,outs,hooks={}, {}, []
    direct={l:[] for l in layers}; jac={l:[] for l in layers}
    try:
        hooks.append(model.model.embed_tokens.register_forward_hook(lambda m,a,o:o.detach().requires_grad_(True)))
        for l in layers:
            def capture(m,a,o,layer=l):
                ins[layer]=a[0].detach();outs[layer]=o
            hooks.append(model.model.layers[l].mlp.down_proj.register_forward_hook(capture))
        with torch.enable_grad():
            for task,(bad,good) in zip(tasks,token_pairs):
                u=model.lm_head.weight.detach()
                diff=(u[bad]-u[good]).float()
                h=model.model(input_ids=chat_ids(model,tok,task),use_cache=False).last_hidden_state[0,-1]
                grads=torch.autograd.grad((h.float()*diff).sum(),[outs[l] for l in layers])
                for l,g in zip(layers,grads):
                    w=model.model.layers[l].mlp.down_proj.weight.detach().float()
                    direct[l].append((ins[l][0,-1].float()*(diff@w)).detach().cpu())
                    jac[l].append((ins[l].float()*(g.float()@w)).sum((0,1)).detach().cpu())
                ins.clear();outs.clear()
    finally:
        for h in hooks:h.remove()
        for p,f in zip(params,flags):p.requires_grad_(f)
    return {l:torch.stack(v) for l,v in direct.items()},{l:torch.stack(v) for l,v in jac.items()}

def choose_transfer(scores,n,budget=4):
    ranked=[]
    for l,s in scores.items():
        raw=s[:n].mean(0)
        # On legitimate sentences, prefer retaining the prefix over the bare label.
        penalty=s[n:].mean(0).clamp_min(0)
        total=raw-penalty
        for k in total.topk(budget).indices.tolist():
            if total[k]>0 and raw[k]>0:ranked.append((float(total[k]),l,k))
    return [(l,n,0.) for _,l,n in sorted(ranked,reverse=True)[:budget]]

def pilot012(args,out):
    import torch
    model,tok,env=load_model(EDITORS[0],args.device,None)
    save(out/"environment.json",env)
    raw,reg=label_tasks(121001,16,"practice")
    base=evaluate(model,tok,raw,24)
    failures=[]
    for t,r,paired in zip(raw,base,reg):
        # Only genuine extra-prefix failures: correct word present, but first token wrong.
        bad=tok.encode(r["response"],add_special_tokens=False)
        good=tok.encode(t.expected,add_special_tokens=False)
        if not r["correct"] and t.expected in r["response"].lower().split() and bad and good and bad[0]!=good[0]:
            failures.append((t,paired,bad[0],good[0]))
    save(out/"practice.json",{"rows":base,"failures":[{"id":t.id,"bad":b,"good":g} for t,_,b,g in failures]})
    if len(failures)<2:
        return {"pilot":"012","status":"insufficient_practice_prefix_failures","failure_count":len(failures),
                "environment":env,"caveat":"No alternative failure mode selected after observing practice."}
    layers=[len(model.model.layers)*k//4-1 for k in (1,2,3,4)]
    tasks=[t for t,_,_,_ in failures]+[p for _,p,_,_ in failures]
    pairs=[(b,g) for _,_,b,g in failures]+[(b,g) for _,_,b,g in failures]
    direct,jac=margin_derivatives(model,tok,tasks,pairs,layers)
    variants={"base":[],"direct":choose_transfer(direct,len(failures)),
              "jacobian":choose_transfer(jac,len(failures))}
    rng=random.Random(121002)
    used=set()
    control=[]
    for l,n,_ in variants["jacobian"]:
        pool=[i for i in range(model.model.layers[l].mlp.down_proj.in_features) if (l,i) not in used and (l,i,0.) not in variants["jacobian"]]
        c=rng.choice(pool);used.add((l,c));control.append((l,c,0.))
    variants["random"]=control
    save(out/"locked-patch.json",variants)
    test,testreg=label_tasks(121009,args.count,"heldout")
    rows={}
    for name,patch in variants.items():
        with scaled_columns(model,patch):
            rows[name]={"raw":evaluate(model,tok,test,24),"sentence":evaluate(model,tok,testreg,32)}
    save(out/"examples.json",rows)
    return {"pilot":"012","status":"evaluated","environment":env,"practice_failures":len(failures),
            "variants":variants,"metrics":{name:{k:sum(r["correct"] for r in v)/len(v) for k,v in d.items()} for name,d in rows.items()},
            "paired":{name:{k:paired_bootstrap(rows["base"][k],v) for k,v in d.items()} for name,d in rows.items() if name!="base"}}

def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot",required=True,choices=["010","011","012"])
    p.add_argument("--out",required=True)
    p.add_argument("--device",default="cpu")
    p.add_argument("--count",type=int,default=24)
    p.add_argument("--threads",type=int,default=2)
    a=p.parse_args()
    if a.count<3 or a.threads<1:p.error("Invalid counts")
    torch.set_num_threads(a.threads)
    torch.manual_seed(101112)
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    result={"010":pilot010,"011":pilot011,"012":pilot012}[a.pilot](a,out)
    save(out/"summary.json",result)
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
