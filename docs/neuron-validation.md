# Pilots 007–009: replication, downstream derivatives, autonomous editing

Status: all three CPU jobs completed successfully; 42 tests passed in each job. Pilot 009 is a negative scientific result despite successful CI.

007 freezes all three prior four-column patches before generating two fresh evaluation
sets (seeds 90701 and 90702), each with 24 raw JSON and 24 paired fenced JSON tasks,
plus 24 unrelated arithmetic/uppercase/selection tasks. Six new prompt/data templates
include nested values, arrays and booleans. These are synthetic transfer tests; copying
the presented object is easier than open-ended data extraction. Score raw JSON exactly
using the existing benchmark, required fences separately, and report paired CIs.
No reselection of the previous patch is allowed.

008 measures the local derivative of the next-token fence-versus-brace margin with
respect to the MLP output at every prompt position, multiplied by each channel's
signed contribution. This includes downstream attention/MLP and final normalization
effects, unlike direct unembedding; it is a target-specific Jacobian approximation,
not Anthropic's full averaged vocabulary J-lens. Parameters are frozen and there is
no optimizer. Rank four neurons per method from practice only, then measure real
ablation effects on the union and report prediction error and correlation. Compare
direct lens, Jacobian lens and causally selected patches on a fresh evaluation seed.
The candidate union is a selected sample, so correlations are not a population estimate.

009 gives the same new practice-only causal evidence to a separate local
Qwen2.5-1.5B-Instruct editor, once, without an external API. The editor checkpoint
revision is resolved and recorded at runtime; use --editor-revision to pin a replay.
A strict JSON validator permits at most four candidate columns with multipliers
0, 0.25, 0.5 or 0.75. Invalid output is recorded as a failed proposal, never repaired
or replaced by a heuristic. A valid proposal is locked before evaluation on seed
90901. Compare with the direct, derivative and causal heuristics; no inference-time
hooks or gradient-based weight optimization in generation. This is autonomous
bounded selection, not arbitrary weight-space programming.

Each pilot runs separately and uploads full predictions and locked patches. CPU
jobs have 90-minute caps. No paid API calls or cloud GPU resources are requested.

Run locally:
    python -m weightsmith.neuron_validation --pilot 007 --out artifacts/pilot-007
    python -m weightsmith.neuron_validation --pilot 008 --out artifacts/pilot-008
    python -m weightsmith.neuron_validation --pilot 009 --out artifacts/pilot-009

## Verified outcomes

- 007: fixed causal patch 47/96 versus base and random 31/96, activation-only 37/96. Both seeds gained 8/48, with 95% paired bootstrap CI +6.25 to +27.08 pp each. Required fences 41/48 versus 43/48 base; two losses. No unrelated losses for causal patch.
- 008: Jacobian and causal union selected identical four columns. MAE 0.073769 vs 0.641759 for direct lens; Pearson 0.990152 vs 0.773772 on selected candidates only. Jacobian 31/48 vs base 18/48 and direct 26/48; +27.08 pp [14.58,41.67], 14 wins, one loss. Required fences 21/24 vs 22/24 (one win, two losses). Selection accuracy declined from 3/8 to 2/8. Not a full J-lens replication or regression-free repair.
- 009: invalid JSON, parser error at line 1 column 1; no patch evaluated, no fallback. Do not infer editor efficacy.

[Exact job summaries and artifact IDs](../results/neuron-validation-007-009.json), parsed from successful job logs. [Raw run artifacts](https://github.com/KevinmKrieg/weight-smithing/actions/runs/45659572618).
