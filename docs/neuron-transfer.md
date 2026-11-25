# Pilots 010–012: constrained editors, attenuation, transfer

Status: implementation submitted for CI; no results claimed.

010 compares Qwen2.5-1.5B-Instruct and Qwen2.5-Coder-1.5B-Instruct as editor
models, both distinct from the 0.5B target. Each gets the same practice-only
causal candidates. Record a free JSON proposal and a separate constrained
proposal. The constrained decoder greedily follows a finite trie of valid
single-edit JSON objects (four allowed multipliers, no duplicate channels) or
a stop object, sequentially choosing up to four edits. The harness assembles
the outer JSON and supplies a fixed rationale; this is not unconstrained full
JSON synthesis. An empty constrained selection is an explicit abstention.
Invalid free proposals are not repaired; only valid ones enter evaluation.
Compare against the direct, Jacobian and causal heuristics on fresh seed
101009. Editor revisions are resolved at runtime and recorded. No paid API.

011 holds the original Pilot 004 and Pilot 008 column sets fixed. For each,
evaluate attenuation multipliers 0, .25, .5, .75, 1 on a new practice set
(12 raw, 12 fenced, 12 unrelated). Require no per-example losses on
practice fenced or unrelated tasks. Choose maximal raw accuracy, breaking
ties toward the least intervention; multiplier 1 is an allowed no-op.
Lock the selections and compare against full ablation and base on seed
111009. This is discrete practice selection, not gradient optimization.

012 switches target to Qwen2.5-1.5B-Instruct and exact lowercase label
extraction rather than JSON. The paired regression asks for a full sentence.
Identify actual unwanted-prefix errors in a fixed practice set only, requiring
at least two failures containing the correct label but beginning with another
token. If insufficient, report infeasibility; do not substitute a different
failure mode after observing results. On eligible examples compare direct
last-position readout and downstream all-position local derivatives of
the bad-vs-good first-token margin. Penalize suppression of the same prefix
on paired sentence prompts; this proxy is imperfect because some generated
bad prefixes may not be the required sentence's first token. Lock up to four
columns per method and a matched random control, evaluate on a fresh seed.
The larger checkpoint revision is recorded at runtime. This is a feasibility
test, not a claim of cross-model patch portability.

All jobs run on CPU with a 90-minute cap and upload raw artifacts. No external
API or paid GPU requested. Derivatives are diagnostic only; weights are changed
by direct column scaling, and target inference is unconstrained.
