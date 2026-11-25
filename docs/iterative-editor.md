# Pilots 013–015: iterative editing and generalization

Status: submitted for CPU CI; no results claimed.

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
