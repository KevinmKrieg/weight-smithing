# Neuron basis research: pilots 003–006

Inspired by Anthropic's [theory](https://alignment.anthropic.com/2026/neuron-basis-theory/)
and [applications](https://alignment.anthropic.com/2026/neuron-basis-applications/) posts.
These are independent small-model experiments, not replications of their reported scores.

## Pilots 003–004: atlas and context-sensitive repair

Status: implementation submitted for CI; no measured results claimed until an actual run succeeds.

Target: pinned Qwen2.5-0.5B-Instruct, exactly as in pilots 001–002.
At the first generation position, capture signed MLP intermediate activations for six paired
raw-JSON and explicitly fenced-JSON prompts. In layers 12, 16, 20, 23, rank channels by
the difference in mean signed activation times the fence-vs-brace unembedding direction.
Record the top four candidates per layer, with positive/negative cosine vocabulary readouts
and their practice activation values. This restricted atlas is not a full census, and the
cosine logit lens is not the J-lens (no downstream Jacobian or final-norm correction).

Ablate every candidate's down-projection column and measure the actual downstream first-token
logit-margin changes on practice. Rank by mean reduction on raw JSON minus positive reduction
on required-fence JSON. Select at most four channels only if both raw reduction and the
penalized score are positive; permit an empty patch. Lock the patch before evaluating.
The ablation also records the largest actual mean raw-prompt logit changes; these are
context-dependent finite differences, not normalized J-lens readouts.

Compare base, causal edit, activation-only and random-column controls. Controls match
column counts and layers, not weight-change norm. No gradients, training, token blacklist
or inference-time hooks are used for edited generation. The selection is a deterministic
heuristic designed in ChatGPT, not an autonomous LLM editor.

Fresh seed 84031: 48 held-out tasks across the four existing task families, plus 12
independently generated regression JSON tasks where Markdown fences are required.
Report paired bootstrap CIs and per-task raw generations, including negative results.
A useful repair must improve raw JSON and not materially degrade required fences or
other families. This small synthetic set cannot establish general instruction following.

Run:
    python -m pip install -e '.[target,dev]'
    pytest -q
    python -m weightsmith.neuron_basis --device cpu

Artifacts: atlas.json, causal-atlas.json, locked-patch.json, summary.json,
examples.json, required-fences.json. CI uploads them even after partial failure.

## Pilot 005: blind benign-behavior auditing (planned)

Prepare separate patches/adapters for benign changes (language, brevity, rhyme, formatting),
with ground truth hidden from auditor, including null controls. Compare output-only,
weight-delta-only, and neuron-readout-assisted LLM auditors; keep the same model and
inference budget. Verify behavior before including a patch. Report blind exact concept
accuracy and false positives, not anecdotal successful guesses. No results yet.

## Pilot 006: native neuron concept detection (planned)

Start with benign semantic concepts, e.g. French, arithmetic and cooking, using disjoint
selection and evaluation texts and hard negatives. Compare signed neuron activation
or residual-direction cosine scoring to keyword, embedding-similarity and supervised
probe baselines. Report AUROC and uncertainty on unseen prompts. Distinguish truly
zero-label selection from few-shot selection using labeled examples. No results yet.
