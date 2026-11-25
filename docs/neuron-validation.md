# Pilots 007–009: replication, downstream derivatives, autonomous editing

Status: submitted for CPU CI; no new results claimed before completed runs.

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
