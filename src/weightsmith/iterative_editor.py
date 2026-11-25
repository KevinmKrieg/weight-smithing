"""Pilots 013–015: feedback-driven editor, broader regression, cross-architecture feasibility."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

from .benchmark import Task, make_tasks, paired_bootstrap, summarize
from .neuron_basis import REVISION, chat_ids, first_step, scaled_columns, save, fenced_eval
from .neuron_validation import FIXED, prepare_diagnostics, evaluate_variants
from .neuron_transfer import JAC_PATCH, constrained_choice
from .runtime import MODEL, load_model, evaluate, generate_one

EDITOR = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
EDITOR_REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
TRIALS = 4

def extraction_pairs(seed, n, split):
    rng = random.Random(f"realistic:{seed}:{split}")
    raw, fenced = [], []
    for i in range(n):
        who = rng.choice(["Maya","Leo","Sam","Amina","Nora","Ravi"])
        value = rng.randrange(10, 99)
        status = rng.choice(["open","closed","pending"])
        city = rng.choice(["Paris","Rome","Oslo","Lyon","Boston"])
        distractor = rng.randrange(100, 999)
        specs = [
            (f"Ticket note: reference {distractor}; owner {who}; priority {value}. Extract owner (string) and priority (integer).", {"owner":who,"priority":value}),
            (f"Shipment update: {who} sent {value} units to {city}; invoice {distractor}. Extract destination and units, using keys city and units.", {"city":city,"units":value}),
            (f"Support log: case {value} is {status}; previous case {distractor} was archived. Extract case (integer) and status.", {"case":value,"status":status}),
            (f"Inventory: {who} has {value} items; alert number {distractor}. Return an object with nested key inventory containing count (integer).", {"inventory":{"count":value}}),
            (f"Meeting notes: {who} will visit {city}; backup room {distractor}. Extract name and city.", {"name":who,"city":city}),
            (f"An order for {value} units is confirmed, not cancelled. Reference {distractor}. Extract quantity (integer) and confirmed (boolean).", {"quantity":value,"confirmed":True}),
        ]
        prompt, obj = specs[i%len(specs)]
        key=f"{split}-{seed}-{i}"
        expected=json.dumps(obj,separators=(",",":"))
        raw.append(Task(key+"-raw","json",prompt+" Return only valid JSON. No Markdown.",expected))
        fenced.append(Task(key+"-fenced","fenced_json",prompt+" Return only a Markdown code block labeled json.",expected))
    return raw,fenced

def unrelated_tasks(seed,n,split):
    rng=random.Random(f"unrelated:{seed}:{split}")
    tasks=[]
    for i in range(n):
        a,b=rng.sample(range(10,90),2)
        word=rng.choice(["lantern","garden","marble","silver","orange","bridge"])
        cases=[
            ("arithmetic",f"Return only the integer difference between {max(a,b)} and {min(a,b)}.",str(abs(a-b))),
            ("uppercase",f"Write the word {word} in uppercase only.",word.upper()),
            ("selection",f"Return only the larger number: {a} or {b}.",str(max(a,b))),
            ("copy",f"Copy exactly this identifier, without quotes: ID-{a}-{b}",f"ID-{a}-{b}"),
            ("choice",f"Answer only yes or no. Is {a} greater than {b}?","yes" if a>b else "no"),
            ("reverse",f"Reverse these two words, separated by one space: {word} blue",f"blue {word}"),
        ]
        family,prompt,expected=cases[i%len(cases)]
        tasks.append(Task(f"{split}-other-{seed}-{i}",family,prompt,expected))
    return tasks

def exact_rows(model,tok,tasks,cap=80):
    rows=[]
    for t in tasks:
        r=generate_one(model,tok,t,cap)
        rows.append({"id":t.id,"family":t.family,"prompt":t.prompt,"expected":t.expected,
                     "response":r["response"],"correct":r["response"].strip()==t.expected})
    return rows

def score_patch(model,tok,patch,raw,fenced,other):
    with scaled_columns(model,patch):
        return {"raw":evaluate(model,tok,raw,80),
                "fenced":fenced_eval(model,tok,fenced,80),
                "other":exact_rows(model,tok,other)}

def counts(rows):
    return {k:sum(r["correct"] for r in v) for k,v in rows.items()}

def feedback(rows,base):
    return {"correct":counts(rows),
            "losses":{k:sum(a["correct"] and not b["correct"] for a,b in zip(base[k],v)) for k,v in rows.items()},
            "wins":{k:sum(not a["correct"] and b["correct"] for a,b in zip(base[k],v)) for k,v in rows.items()}}

def objective(f):
    # Same precommitted selector for both search arms: prioritize no regressions.
    return (-(f["losses"]["fenced"]+f["losses"]["other"]),
            f["correct"]["raw"])

def action_space(candidates,patch):
    """One operation: add/update/remove a candidate, or retain current patch."""
    current={(l,n):s for l,n,s in patch}
    actions=[{"op":"keep"}]
    for l,n in candidates:
        if (l,n) in current:
            actions.append({"op":"remove","layer":l,"channel":n})
        if (l,n) in current or len(current)<4:
            for s in (0.,.25,.5,.75):
                if current.get((l,n))!=s:
                    actions.append({"op":"set","layer":l,"channel":n,"multiplier":s})
    return actions

def apply_action(patch,action):
    current={(l,n):s for l,n,s in patch}
    if action["op"]=="keep":return list(patch)
    key=(action["layer"],action["channel"])
    if action["op"]=="remove":del current[key]
    elif action["op"]=="set":current[key]=action["multiplier"]
    else:raise ValueError("Unknown action")
    if len(current)>4:raise ValueError("Patch budget")
    return [(l,n,s) for (l,n),s in sorted(current.items())]

def final_evaluation(model,tok,variants,seed,n,out):
    raw,fenced=extraction_pairs(seed,n,"heldout")
    other=unrelated_tasks(seed,n,"heldout")
    rows={k:score_patch(model,tok,p,raw,fenced,other) for k,p in variants.items()}
    save(out/f"heldout-{seed}.json",rows)
    return {"seed":seed,"n_per_group":n,"correct":{k:counts(v) for k,v in rows.items()},
            "paired":{k:{g:paired_bootstrap(rows["base"][g],r) for g,r in v.items()}
                      for k,v in rows.items() if k!="base"}}

def pilot013(args,out):
    import gc
    model,tok,env=load_model(MODEL,args.device,REVISION)
    report=prepare_diagnostics(model,tok,13001,out)
    candidates=[(r["layer"],r["channel"]) for r in report["candidates"]]
    raw,fenced=extraction_pairs(13002,6,"practice")
    other=unrelated_tasks(13002,6,"practice")
    base=score_patch(model,tok,[],raw,fenced,other)
    save(out/"practice-base.json",base)
    # Keep the target and editor sequentially resident to fit ordinary CPU runners.
    del model,tok;gc.collect()
    editor,etok,eenv=load_model(EDITOR,args.device,EDITOR_REVISION)
    del editor,etok;gc.collect()
    histories={}
    variants={"base":[],"jacobian":report["variants"]["jacobian"]}
    rng=random.Random(13003)
    for arm in ("editor","random"):
        current=[];best=[];best_score=objective(feedback(base,base))
        history=[]
        for trial in range(TRIALS):
            actions=action_space(candidates,current)
            if arm=="editor":
                editor,etok,_=load_model(EDITOR,args.device,EDITOR_REVISION)
                compact=[{k:r[k] for k in ("layer","channel","raw_reduction","fenced_reduction","causal_score")} for r in report["candidates"]]
                prompt=("Edit another model's MLP weights to fix raw JSON extraction without breaking fenced JSON or unrelated tasks. "
                        "Choose exactly one JSON operation: set, remove, or keep. Maximum four channels. "
                        "Positive raw reduction is useful; positive fenced reduction can be harmful. "
                        "Current patch: "+json.dumps(current)+"\nCandidate evidence: "+json.dumps(compact)+
                        "\nPrior practice trials: "+json.dumps([{k:v for k,v in h.items() if k!="examples"} for h in history])+
                        "\nChoose a new operation or keep. No held-out data.")
                chosen,ids=constrained_choice(editor,etok,prompt,[json.dumps(a,separators=(",",":")) for a in actions])
                action=json.loads(chosen)
                del editor,etok;gc.collect()
            else:
                action=rng.choice(actions);ids=[]
            current=apply_action(current,action)
            model,tok,_=load_model(MODEL,args.device,REVISION)
            rows=score_patch(model,tok,current,raw,fenced,other)
            del model,tok;gc.collect()
            f=feedback(rows,base)
            history.append({"trial":trial+1,"action":action,"patch":list(current),"feedback":f,"token_ids":ids,"examples":rows})
            if objective(f)>best_score:
                best_score=objective(f);best=list(current)
            save(out/f"{arm}-history.json",history)
        histories[arm]=[{"trial":h["trial"],"action":h["action"],"patch":h["patch"],"feedback":h["feedback"]} for h in history]
        variants[arm+"_selected"]=best
        variants[arm+"_first"]=history[0]["patch"]
    save(out/"locked-patch.json",variants)
    model,tok,_=load_model(MODEL,args.device,REVISION)
    return {"pilot":"013","environment":env,"editor_environment":eenv,"trial_budget_per_arm":TRIALS,
            "histories":histories,"variants":variants,
            "evaluation":final_evaluation(model,tok,variants,13009,args.count,out),
            "caveat":"Finite grammar, candidate-restricted, four practice evaluations per arm; best practice patch or no-op. Random search is one realization."}

def pilot014(args,out):
    model,tok,env=load_model(MODEL,args.device,REVISION)
    coder=[(23,785,.5),(23,641,.75),(23,2959,.75),(20,4752,.5)]
    variants={"base":[],"fixed_original":FIXED,"fixed_jacobian":JAC_PATCH,"fixed_coder":coder}
    save(out/"locked-patch.json",variants)
    return {"pilot":"014","environment":env,"variants":variants,
            "evaluations":[final_evaluation(model,tok,variants,s,args.count,out) for s in (14009,14010)],
            "caveat":"Synthetic extraction and exact-answer regression suite, not a public general-capability benchmark."}

def pilot015(args,out):
    """Cross-family target: SmolLM2 (Llama architecture), actual observed prefix failures."""
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from .neuron_transfer import margin_derivatives,choose_transfer
    checkpoint="HuggingFaceTB/SmolLM2-360M-Instruct"
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(checkpoint,torch_dtype=torch.float32,trust_remote_code=False).to(args.device).eval()
    if model.config.model_type!="llama":raise ValueError("Unexpected cross-family architecture")
    env={"checkpoint":checkpoint,"resolved_checkpoint_revision":getattr(model.config,"_commit_hash",None),"model_type":model.config.model_type}
    save(out/"environment.json",env)
    raw,fenced=extraction_pairs(15001,18,"practice")
    base=evaluate(model,tokenizer,raw,80)
    failed=[]
    for t,r,paired in zip(raw,base,fenced):
        # Restrict to actual fenced valid JSON (format-only) failures.
        if r["correct"]:continue
        s=r["response"].strip().splitlines()
        if len(s)<3 or not s[0].startswith(chr(96)*3) or s[-1].strip()!=chr(96)*3:continue
        try:
            if json.loads("\n".join(s[1:-1]))!=json.loads(t.expected):continue
        except ValueError:continue
        b=tokenizer.encode(r["response"],add_special_tokens=False)
        g=tokenizer.encode("{",add_special_tokens=False)
        if b and g and b[0]!=g[0]:failed.append((t,paired,b[0],g[0]))
    save(out/"practice.json",{"rows":base,"eligible_ids":[t.id for t,_,_,_ in failed]})
    if len(failed)<2:
        return {"pilot":"015","status":"insufficient_format_only_failures","eligible":len(failed),"environment":env}
    layers=[len(model.model.layers)*k//4-1 for k in (1,2,3,4)]
    tasks=[t for t,_,_,_ in failed]+[p for _,p,_,_ in failed]
    pairs=[(b,g) for _,_,b,g in failed]*2
    direct,jac=margin_derivatives(model,tokenizer,tasks,pairs,layers)
    variants={"base":[],"direct":choose_transfer(direct,len(failed)),
              "jacobian":choose_transfer(jac,len(failed))}
    rng=random.Random(15002);used={(l,n) for l,n,_ in variants["jacobian"]};control=[]
    for l,n,_ in variants["jacobian"]:
        pool=[i for i in range(model.model.layers[l].mlp.down_proj.in_features) if (l,i) not in used]
        c=rng.choice(pool);used.add((l,c));control.append((l,c,0.))
    variants["random"]=control
    save(out/"locked-patch.json",variants)
    return {"pilot":"015","status":"evaluated","environment":env,"eligible":len(failed),"variants":variants,
            "evaluation":final_evaluation(model,tokenizer,variants,15009,args.count,out)}

def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot",required=True,choices=["013","014","015"])
    p.add_argument("--out",required=True)
    p.add_argument("--device",default="cpu")
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--count",type=int,default=24)
    a=p.parse_args()
    if a.count<6 or a.threads<1:p.error("Invalid counts")
    torch.set_num_threads(a.threads);torch.manual_seed(131415)
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    result={"013":pilot013,"014":pilot014,"015":pilot015}[a.pilot](a,out)
    save(out/"summary.json",result)
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
