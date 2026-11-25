# Neuron basis research: pilots 003–006

Inspired by Anthropic's [theory](https://alignment.anthropic.com/2026/neuron-basis-theory/)
and [applications](https://alignment.anthropic.com/2026/neuron-basis-applications/) posts.
These are independent small-model experiments, not replications of their reported scores.

## Pilots 003–004: atlas and context-sensitive repair

Status: completed successfully on November 25, 2026. See the measured results below.

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

## Pilot 005: closed-set synthetic benign-direction audit

Implemented as a deliberately small first rung, not a full blind LLM or LoRA audit.
Create six direction patches (two each for arithmetic, French and cooking) in late
MLP down-projection columns, plus two unmodified null cases. Patch magnitudes are
relative to the original column norms; direction exemplars are distinct from the
auditor's vocabulary. Patches and nulls are randomly ordered with opaque IDs.
An independent set of prompts checks whether the intended next-token logit shift
actually occurs. Compare output-logit, signed weight-readout and weight-assisted
heuristics; report all cases and the separately validated subset. This is a
closed-set toy test, not a test of learned hidden behavior. Export blind packets,
predictions and ground truth separately for a subsequent LLM audit. No API key
required. Do not infer verified free-generation behavior from logit shifts.

## Pilot 006: few-shot native neuron concept detection

Use 18 labeled selection texts for three benign concepts (arithmetic, French,
cooking), and 36 held-out texts including keyword-confusable hard negatives.
Select eight native neurons per concept using signed standardized mean activation
contrasts across four layers. Lock neuron IDs and signs before processing the
test set. Compare native activation scores to keyword matching, hidden-state
centroid similarity and supervised ridge probes fitted on the same training set.
Report per-concept and macro AUROC, with raw per-text scores. This is few-shot
rather than the zero-label detector in the Anthropic post, and synthetic examples
are too few for strong generalization claims.

Run both applications with:
    python -m weightsmith.neuron_applications --device cpu

## Verified results

Both workflows succeeded; the latest applications workflow reported 39 passing tests. The repair workflow reported 36 tests on its earlier commit. The checked-in [result JSON](../results/neuron-basis-pilots-003-006.json) contains the full summaries parsed from the successful job logs. Raw outputs remain in the linked CI artifacts.

- Pilot 004: 19/48 to 31/48, +25 pp [12.5, 37.5]; 12 wins and 0 losses. Required fences: 12/12 across all variants. Activation-only 30/48 (13 wins, 2 losses); random 19/48. No claim of statistically significant superiority over activation-only.
- Pilot 005: 8/8 across all three heuristic auditors; all eight cases validated by independent next-token logit shifts, not free generation.
- Pilot 006: macro AUROC 1.0 for native neurons, centroid and ridge; keyword 0.8705357. Ceiling effects and small hand-authored data limit conclusions.

The selected columns were (layer, channel) (23,641), (23,1057), (20,266), (20,4752), all scaled to zero. The intervention is a practice-only deterministic heuristic, not an autonomous LLM edit. The required-fence set was small, and no observed losses is not proof of no regressions.
