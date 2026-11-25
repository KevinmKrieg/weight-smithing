# Pilots 019–021: context-discriminative editing

Status: submitted for CPU CI; no results claimed.

019 compares unmodified Qwen, global Jacobian attenuation, an oracle
task-type gate, and a fixed text-only router. Two fresh evaluation sets
of 72 prompts each use familiar and alternative instruction wording.
Oracle gating is a feasibility upper-reference, not deployable evidence
of weight-only improvement. Text routing deliberately uses only explicit
lexical cues and may fail on paraphrases. Both conditionally apply and
restore the same weight patch for each prompt; they are runtime interventions,
not permanently context-sensitive weights.

020 fits a no-intercept ridge direction to last-position gate-projection
inputs from 12 raw, 12 paired fenced and 12 unrelated practice prompts.
Target shift -1 on raw, 0 elsewhere. For each of the four fixed Jacobian
neurons, add a scaled direction to its gate projection row, with sign and
magnitude based on the original median raw preactivation. This directly
changes weights and needs no router or inference-time hook. The edit is
rank one within each affected layer; fitting uses a closed-form supervised
linear solve rather than backpropagation or gradient-based weight training.
Five global gains are evaluated on practice. Lock the best zero-observed-loss
selection and unconstrained raw-accuracy selection before testing fresh
seeds and alternative instruction wording. No-op is valid. The practice
fit is at the final prompt position only and may behave differently at
other token positions or during autoregressive generation.

021 repeats Pilot 016 on three independent practice/evaluation seed pairs,
with 12 practice examples per group, six trials per arm, shared calibration,
identical candidate/action spaces and selection rule. Compare the same pinned
Coder editor, greedy and random search and fixed Jacobian diagnostic baseline.
No held-out feedback. Report individual seed outcomes and regressions rather
than claim statistical superiority from three seeds. Synthetic templates
remain shared. Separate jobs on standard CPU runners with a four-hour cap;
no paid APIs or GPUs requested. Partial artifacts are uploaded on failure.
