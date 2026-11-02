# Experiment 01 — Direct LLM-to-LLM weight editing

> **Status update (November 2026):** This document describes the original constrained MLP-column editing protocol, which was executed as **Pilot 001** and produced **no measurable improvement** (12/32 for all variants). A subsequent **different intervention, Pilot 002**, edited a single output-token weight row and improved exact JSON formatting on a fresh 32-question set (15/32 → 23/32). For the completed result and distinct protocol, see [Pilot 002](qwen-output-edit.md). It would be incorrect to attribute Pilot 002's gains to the MLP method specified below.

**Goal:** Test whether an editor language model can improve strict instruction-following in a different pretrained LLM by directly attenuating selected weight columns. This is our own study, not a replication of the source paper's checkpoint or scores.

## Precommitted design

| Factor | Setting |
|---|---|
| Target | Qwen/Qwen2.5-0.5B-Instruct (~0.49B parameters) |
| Editor | API-backed LLM (default gpt-4.1); or an explicitly attributed manually reviewed patch |
| Benchmark | Custom exact-match, automatically generated synthetic instructions |
| Families | Integer math, JSON, uppercase single-word, alphabetical selection |
| Practice | 4 examples/family (16 total), seed 41 |
| Held-out | 10 examples/family (40 total), seed 41 in a *different RNG namespace* |
| Extra regression split | 5 examples/family (20 total), different RNG namespace; **same task families**, not a general-purpose ability test |
| Decoding | Greedy with a 48-new-token cap; identical for all variants |
| Candidate layers | 8, 16 |
| Patch | At most 8 MLP down-projection columns attenuated by factors 0 ≤ m < 1 |
| Controls | Unmodified model, activation-ranked edits (non-LLM), matched random-column edits |
| Metrics | Exact match, paired bootstrap 95% CI, average output tokens, 4-gram repetition, cap-hit fraction |

The target is an **actual pretrained language model**, not our toy Breakout network. The editor is constrained to choose layer/channel/multipliers rather than writing arbitrary billions of tensor values. That is a deliberate limitation; no claim of unconstrained model-weight understanding.

## Steps to reproduce

1. Load and pin the actual target checkpoint. Record resolved revision, model, tokenizer, dtype, decoding and seed.
2. Generate and score practice prompts. Record every baseline generation (including failures).
3. From **practice prompts only**, capture input activations to MLP down-projection matrices at the final prompt token. Compute standardized failure-vs-success channel contrasts. Include L2 norms and a small fingerprint of real weight values from the top-ranked candidates.
4. Give an editor LLM the practice examples and diagnostics. Ask it for JSON with selected column indices and multipliers. The editor does **not** receive held-out examples or rewards. The validated proposal is saved **before** any held-out evaluation.
5. Apply the exact proposed direct tensor edits, with gradients disabled, and compare against the unchanged model. Restore original weights afterward. Evaluate two equal-budget controls: top-ranked channels without an editor LLM, and random channels in the same layers with the same multipliers.
6. On held-out prompts, report exact-match accuracy, paired bootstrap differences, individual task outputs, token counts, cap hits and repetition diagnostics. Publish successes, failures and regressions. Do not hide failed editor proposals.

## Successful result criteria

- A **positive** held-out exact-match difference from the untouched base model, with a positive 95% paired bootstrap CI; and
- Better observed performance than both matched controls, without major regressions.

An inconclusive or negative result is valuable evidence. The default held-out set is a small pilot; reliable claims require larger runs, multiple seeds and additional unrelated capability tests.

## Running

~~~bash
python -m pip install -e '.[target,editor,dev]'
pytest -q
python -m weightsmith.practice --per-family 4 --out artifacts/practice-001

# In a shell with your OpenAI API key; the editor call may cost money.
python -m weightsmith.experiment --editor openai --editor-model gpt-4.1 \
  --heldout-per-family 10 --out artifacts/qwen-001

python -m weightsmith.plot --run artifacts/qwen-001 --out artifacts/qwen-001/charts
~~~

The full experiment is **not runnable offline without model weights**, and an automatic LLM-authored patch needs API access. To test with no editor API call, use **--editor heuristic** and call it a *non-LLM control*. A patch file can be provided using --editor file --proposal-file; its author must be documented separately.

### Safety and data boundaries

- No generated Python is executed. Editor output must pass strict JSON validation.
- Only approved internal Qwen MLP columns may be changed, not file paths, tokenizers or checkpoints.
- Weights are restored even if a model evaluation raises an error.
- Held-out examples must never be passed to the editor or used to select patch coefficients.
- Save outputs and failures rather than choosing flattering examples.
- Never commit full checkpoint shards, API keys, or private prompts.
- Practice diagnostics can suggest correlation, not prove a causal neuron mechanism.
- In a publication-grade run, pin the exact checkpoint revision and code commit before measuring.

## Existing older experiment

The independent hand-programmed toy Breakout project is preserved in [breakout.md](breakout.md), but is not direct evidence for the new language-model editing hypothesis.
