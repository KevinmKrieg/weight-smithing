"""028–030: single-row robustness, output-sensitive fitting, failure discovery."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from collections import Counter
from .benchmark import Task
from .context_editor import gate_weight_patch, summarize_rows, novel_pairs
from .contextual_validation import fit_rows, apply_rows, serialize
from .iterative_editor import extraction_pairs, unrelated_tasks, score_patch, feedback, exact_rows
from .neuron_basis import REVISION, chat_ids, scaled_columns, save
from .neuron_transfer import JAC_PATCH
from .runtime import MODEL, load_model, generate_one
from .sparse_contextual import choose_trial, weighted_direction

ROW=1

def alternate_pairs(seed,n):
    rng=random.Random(seed);raw=[];fenced=[]
    for i in range(n):
        a=rng.randrange(10,99);b=rng.randrange(100,999)
        who=rng.choice(["Maya","Leo","Nora","Ravi"])
        city=rng.choice(["Rome","Oslo","Paris","Lyon"])
        prompt,obj=[
            (f"Archive record {b}: assigned to {who}; urgency {a}. Extract owner and priority.",{"owner":who,"priority":a}),
            (f"Delivery manifest {b}: {a} packages bound for {city}. Extract city and units.",{"city":city,"units":a}),
            (f"Visitor register {b}: {who} is scheduled in {city}. Extract name and city.",{"name":who,"city":city}),
        ][i%3]
        expected=json.dumps(obj,separators=(",",":"))
        raw.append(Task(f"alt-{seed}-{i}-r","json",prompt+" Output just the JSON object, without any code block.",expected))
        fenced.append(Task(f"alt-{seed}-{i}-f","fenced_json",prompt+" Output a JSON-labeled Markdown code block only.",expected))
    return raw,fenced

def broad_other(seed,n):
    rng=random.Random(seed);tasks=[]
    for i in range(n):
        a,b=rng.sample(range(2,20),2);word=rng.choice(["amber","cedar","violet","maple"])
        prompt,expected=[
            (f"Return only the product of {a} and {b}.",str(a*b)),
            (f"Return only the number of letters in {word}.",str(len(word))),
            (f"Return the lowercase version of {word.upper()} and nothing else.",word),
            (f"Sort these integers ascending, separated by commas, no spaces: {a}, {b}",f"{min(a,b)},{max(a,b)}"),
            (f"Reply with exactly the following two words: {word} tree",word+" tree"),
            (f"Answer only true or false: {a} is less than {b}.",str(a<b).lower()),
        ][i%6]
        tasks.append(Task(f"broad-{seed}-{i}","protected",prompt,expected))
    return tasks

def eval_set(model,tok,variants,seed,n,kind,out):
    raw,fenced={"familiar":lambda:extraction_pairs(seed,n,"heldout"),
        "paraphrase":lambda:novel_pairs(seed,n),"alternate":lambda:alternate_pairs(seed,n)}[kind]()
    other=unrelated_tasks(seed,n,"heldout")+broad_other(seed,n)
    rows={"base":score_patch(model,tok,[],raw,fenced,other),
          "global":score_patch(model,tok,JAC_PATCH,raw,fenced,other)}
    for name,(edits,gain) in variants.items():
        rows[name]=apply_rows(model,tok,edits,gain,raw,fenced,other)
    save(out/f"{seed}-{kind}.json",rows)
    return {"seed":seed,"kind":kind,"n_raw":len(raw),"n_protected":len(fenced)+len(other),**summarize_rows(rows)}

def pilot028(a,out):
    model,tok,env=load_model(MODEL,a.device,REVISION);reps=[]
    for seed in (28001,28002,28003):
        sub=out/str(seed);sub.mkdir()
        raw,fenced=extraction_pairs(seed,12,"practice");other=unrelated_tasks(seed,12,"practice")
        edits,_=fit_rows(model,tok,raw,fenced+other);single=[edits[ROW]]
        base=score_patch(model,tok,[],raw,fenced,other)
        trials=[]
        for gain in (0.,1.,2.,3.,4.):
            rows=apply_rows(model,tok,single,gain,raw,fenced,other)
            trials.append({"gain":gain,"support":[ROW] if gain else [],"feedback":feedback(rows,base)})
        chosen=choose_trial(trials,0)
        variants={"single_fixed2":(single,2.),"single_selected":(single,chosen["gain"]),
                  "single_fixed4":(single,4.),"full_fixed2":(edits,2.)}
        save(sub/"locked.json",{k:{"gain":g,"edits":serialize(e)} for k,(e,g) in variants.items()})
        reps.append({"seed":seed,"practice":trials,"selected":chosen,
            "evaluations":[eval_set(model,tok,variants,seed*10+i,a.count,k,sub)
                for i,k in enumerate(("familiar","paraphrase","alternate"))]})
        save(out/"completed.json",reps)
    return {"pilot":"028","environment":env,"replications":reps,
        "caveat":"New synthetic templates and protected families, not broad real-world capability coverage."}

def logit_gradient(model,tok,ids,row,target,competitor):
    """Gradient of one next-token margin wrt a single actual gate row."""
    import torch
    layer,channel,_=row
    module=model.model.layers[layer].mlp.gate_proj
    captured=[]
    def hook(m,args,output):
        # Base model parameters frozen; leaf output allows downstream backprop.
        x=args[0].detach()
        z=output.detach().requires_grad_(True)
        captured.append((x,z))
        return z
    h=module.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            hidden=model.model(input_ids=ids,use_cache=False).last_hidden_state[:,-1]
            logits=model.lm_head(hidden)[0].float()
            margin=logits[target]-logits[competitor]
            x,z=captured[0]
            grad=torch.autograd.grad(margin,z)[0][0,:,channel].float()
            return (grad[:,None]*x[0].float()).sum(0).detach().cpu(),float(margin.detach())
    finally:h.remove()

def margin_examples(model,tok,raw,protect,row):
    import torch
    X=[];y=[];groups=[];meta=[]
    # Raw: target first token of expected object against baseline strongest rival.
    # Protected: preserve baseline top-two margins along greedy generated prefixes.
    for group,tasks in (("raw",raw),("protected",protect)):
        for task in tasks:
            ids=chat_ids(model,tok,task)
            steps=1 if group=="raw" else 3
            for step in range(steps):
                with torch.no_grad():
                    logits=model.lm_head(model.model(input_ids=ids,use_cache=False).last_hidden_state[:,-1])[0].float()
                    top=logits.topk(2).indices.tolist()
                if group=="raw":
                    target=tok.encode(task.expected,add_special_tokens=False)[0]
                    competitor=top[0] if top[0]!=target else top[1]
                else:target,competitor=top
                g,m=logit_gradient(model,tok,ids,row,target,competitor)
                X.append(g);y.append(2.0 if group=="raw" else 0.0);groups.append(group)
                meta.append({"id":task.id,"group":group,"step":step,"margin":m,"target":target,"competitor":competitor})
                if top[0]==tok.eos_token_id:break
                ids=torch.cat([ids,torch.tensor([[top[0]]],device=ids.device)],dim=1)
    return torch.stack(X),torch.tensor(y),groups,meta

def pilot029(a,out):
    import torch
    model,tok,env=load_model(MODEL,a.device,REVISION)
    for p in model.parameters():p.requires_grad_(False)
    reps=[]
    for seed in (29001,29002,29003):
        sub=out/str(seed);sub.mkdir()
        raw,fenced=extraction_pairs(seed,8,"fit");other=unrelated_tasks(seed,8,"fit")
        ordinary,_=fit_rows(model,tok,raw,fenced+other)
        l,n,d=ordinary[ROW];norm=d.norm()
        x,y,groups,meta=margin_examples(model,tok,raw,fenced+other,JAC_PATCH[ROW])
        edits={"activation": [(l,n,d)]};diag={}
        for penalty in (1.,10.,100.):
            weights=torch.tensor([1. if g=="raw" else penalty for g in groups])
            v=weighted_direction(x,y,weights)
            v=v*(norm/v.norm().clamp_min(1e-12))
            name=f"logit_{int(penalty)}";edits[name]=[(l,n,v)]
            diag[name]={"norm":float(v.norm()),"predicted_changes":(x@v).tolist()}
        save(sub/"fit.json",{"examples":meta,"diagnostics":diag})
        r,f=extraction_pairs(seed+100,12,"practice");o=unrelated_tasks(seed+100,12,"practice")
        base=score_patch(model,tok,[],r,f,o)
        trials=[{"method":"activation","gain":0.,"support":[],"feedback":feedback(base,base)}]
        for name,e in edits.items():
            for gain in (1.,2.,4.):
                rows=apply_rows(model,tok,e,gain,r,f,o)
                trials.append({"method":name,"gain":gain,"support":[ROW],"feedback":feedback(rows,base)})
        selected=choose_trial(trials,0)
        ordinary_selected=choose_trial([t for t in trials if t["method"]=="activation"],0)
        variants={name+"_fixed2":(e,2.) for name,e in edits.items()}
        for name,t in (("selected",selected),("activation_selected",ordinary_selected)):
            variants[name]=(edits[t["method"]],t["gain"])
        save(sub/"locked.json",{k:{"gain":g,"edits":serialize(e)} for k,(e,g) in variants.items()})
        reps.append({"seed":seed,"trials":trials,"selected":selected,"activation_selected":ordinary_selected,
            "evaluations":[eval_set(model,tok,variants,seed*10+i,a.count,k,sub)
                for i,k in enumerate(("familiar","paraphrase"))]})
        save(out/"completed.json",reps)
    return {"pilot":"029","environment":env,"replications":reps,
        "caveat":"Local first-order margin surrogate; three protected greedy prefix positions, not full distribution or generation guarantee. Gradient-based diagnostics, not autonomous editing."}

def discovery_tasks(seed,n):
    rng=random.Random(seed);families={}
    for i in range(n):
        a,b=rng.sample(range(10,90),2);word=rng.choice(["cedar","iris","maple","birch"])
        cases=[
            ("csv",f"Record name={word}, quantity={a}. Emit one CSV row name,quantity without header.",f"{word},{a}"),
            ("integer",f"Return the integer {a} plus {b}.",str(a+b)),
            ("label",f"Classify as HIGH if {a} is greater than 50, otherwise LOW.", "HIGH" if a>50 else "LOW"),
            ("yaml",f"Record: name {word}. Return one YAML line with key name.",f"name: {word}"),
            ("copy",f"Copy exactly the identifier ID-{a}-{b}.",f"ID-{a}-{b}"),
        ]
        for family,prompt,expected in cases:
            families.setdefault(family,[]).append(Task(f"{seed}-{family}-{i}",family,prompt+" Return only the requested content, no Markdown or explanation.",expected))
    return families

def wrapper_category(response,expected):
    s=response.strip()
    if s==expected:return "correct_raw"
    lines=s.splitlines()
    if len(lines)>=3 and lines[0].strip().startswith(chr(96)*3) and lines[-1].strip()==chr(96)*3:
        return "wrapper_only" if "\n".join(lines[1:-1]).strip()==expected else "wrapped_wrong"
    return "other_wrong"

def pilot030(a,out):
    model,tok,env=load_model(MODEL,a.device,REVISION);results=[]
    # Independent discovery and confirmation; no weight patch is chosen here.
    for seed in (30001,30009):
        families=discovery_tasks(seed,a.count);summary={}
        for name,tasks in families.items():
            rows=exact_rows(model,tok,tasks)
            for t,r in zip(tasks,rows):r["category"]=wrapper_category(r["response"],t.expected)
            save(out/f"{seed}-{name}.json",rows)
            summary[name]=dict(Counter(r["category"] for r in rows))
        results.append({"seed":seed,"n_per_family":a.count,"categories":summary})
    eligible=[k for k in results[0]["categories"] if all(s["categories"][k].get("wrapper_only",0)>=3 for s in results)]
    return {"pilot":"030","environment":env,"evaluations":results,"confirmed_families":eligible,
        "caveat":"Strict exact content after outer fence removal; discovery only, no transfer intervention."}

def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot",required=True,choices=["028","029","030"])
    p.add_argument("--out",required=True);p.add_argument("--device",default="cpu")
    p.add_argument("--threads",type=int,default=2);p.add_argument("--count",type=int,default=24)
    a=p.parse_args()
    if a.count<4 or a.threads<1:p.error("Invalid counts")
    torch.set_num_threads(a.threads);torch.manual_seed(282930)
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    result={"028":pilot028,"029":pilot029,"030":pilot030}[a.pilot](a,out)
    save(out/"summary.json",result);print(json.dumps(result,indent=2))
if __name__=="__main__":main()
