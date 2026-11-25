# Pilots 013–015: iterative editing and generalization

Status: completed; 48 tests passed per job.

013 is the priority: Qwen2.5-Coder-1.5B-Instruct (pinned to the revision
observed in Pilot 010) receives candidate-level causal evidence and feedback
from four successive practice evaluations. Each trial selects one bounded
add/update/remove/keep operation through a finite token grammar. No more than
four columns can be active. A random walk receives the same candidate set,
allowed operations and four target practice evaluations. Both start at the
unedited model and use the same precommitted best-practice selection rule:
minimize losses on fenced and unrelated examples first, then maximize raw
JSON accuracy; ties retain the earlier patch, including a possible no-op.
The editor sees trial-level group counts, wins and losses, not held-out
data. Compare first and selected patches against base and Jacobian baseline
on a new held-out seed. The random control is one realization; this is not
a statistically powered editor superiority test. No gradients optimize
weights, although diagnostic derivatives and discrete search are used.

014 freezes Pilot 004, 008 and 010 patches. Two new seeds each contain 24
raw extraction prompts, 24 paired required-fence prompts and 24 unrelated
tasks covering arithmetic, uppercase, numeric comparison, copy, yes/no
and reversal. The source facts are prose rather than already serialized
JSON. Report all paired gains/losses and raw outputs, no reselection.
This remains synthetic and does not establish broad capability retention.

015 switches to HuggingFaceTB/SmolLM2-360M-Instruct (Llama architecture).
Require at least two format-only practice failures, defined as correct
JSON enclosed in Markdown despite a raw-only instruction. If absent,
record benchmark infeasibility without intervention. Otherwise compare
direct and local derivative selection with a layer/count-matched random
control on a fresh extraction and regression set. Record resolved model
revision; no portability claim about Qwen neuron IDs. The target-specific
margin proxy uses the actual first erroneous token versus opening brace.

Three independent CPU jobs, 120-minute limits; no paid API or GPU requested.
All raw outputs and locked patches uploaded. No result will be claimed
until actual evaluation completes.

## Verified outcomes

013: four editor trials all 0/6 raw, selected no-op. Random practice counts 0,1,0,1; selected trial 2. Held-out raw: base/editor 0/24, random 3/24 (CI 0 to 25 pp), Jacobian 19/24 (CI 62.5 to 95.83 pp). Jacobian fenced 17/24 vs 18/24 and other 10/24 vs 9/24; random and editor had no observed losses. One random realization, no general superiority claim.

014: across two seeds, raw/fenced/other counts base 0/45/17, original 8/38/17, Coder 6/41/17, Jacobian 37/36/18 (48 per group). Jacobian total 91/144 vs 62/144, 39 wins and 10 losses (nine fenced, one unrelated). Seed-level paired CIs and all variants in JSON. No claim of regression-free editing.

015: SmolLM2 exhibited zero eligible format-only failures across 18 practice prompts. No patch or held-out evaluation; feasibility limitation, not evidence against transfer.

[Verified job summaries](../results/iterative-editor-013-015.json) · [Raw artifacts](https://github.com/KevinmKrieg/weight-smithing/actions/runs/45681202347).
