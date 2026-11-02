# Experiment 8: Reproducing LLM-directed edits to Qwen weights

**Status: design complete; no Qwen weights have yet been edited or evaluated by this repository.**

## Research question

Can a general-purpose LLM identify a behavioral failure in an existing LLM and propose a **sparse direct edit to its parameters**, improving held-out performance without a gradient-based update?

The target is [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), matching the model in [Takehi et al. (2026), Experiment 8](https://rikiyat.github.io/blog/weight-space-programming/). The experiment's editor was GPT-6 Astra. The [authors released logs](https://github.com/RikiyaT/lm_understand-logs); inspect their exact procedure and code before claiming fidelity.

## What the paper actually did

- Qwen3.5-0.8B **thinking on**, 24 layers, 0.8 billion parameters.
- A Python session exposed model weights; the editor had **10, 25 or 50 budgets of practice forward passes** over GSM8K/MATH *training* problems. This was **not** a blind one-shot edit without feedback.
- No gradients, training losses or weight-optimization training loop.
- At the 50-pass setting, Astra identified a direction associated with repetitive reasoning and subtracted it from 64 output columns each in the MLPs at layers 15 and 19, changing **131,072 weights (~0.016%)**.
- Held-out evaluation at temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, max 32,768 tokens, thinking mode enabled.
- Paper-reported MATH-500 **7.9 → 48.1%**, GSM8K **13.8 → 67.1%**, IFEval **21.0 → 62.3%**; MATH-500 completions hitting length cap **93.0 → 33.1%**.

Scores have varying numbers of completions per example: MATH-500 four, GSM8K and IFEval two. Exact processing, grading, prompt formatting and checkpoint revision must be matched; results may be expensive to reproduce.

## Phased implementation

### 1. Ground the environment

Pin the Qwen checkpoint SHA, Transformers/version, tokenizer, sampling settings, hardware and numeric precision. Verify model's weight names and matrix shapes; never assume a layer index or a row/column orientation from a written description. Record sha256 of original tensor shards, and exclude weights and dataset artifacts from Git.

### 2. Create a visible failure case

Evaluate **practice-only** questions from the appropriate train splits. Save prompt, original completion, response token length, sampled setting and whether it hits the cap. Do not use MATH-500, GSM8K test or IFEval as editing feedback. Establish baseline loop rate at the exact decoding settings.

### 3. Reproduce the original direction removal

Read the author's published logs and replay only the exact numerical operation. Save a compact patch manifest: model checkpoint digest, tensor names, column indices, direction vector or deterministic recipe, original/new tensor hash, number of edited entries, and magnitude norms. Never execute unreviewed code supplied by a model. Track nonzero deltas, not just the count of addresses assigned.

### 4. Independent editor run

Give an editor LLM documentation, controlled weight summaries/inspections and a fixed budget of *practice* forward passes. Require a structured edit proposal, validate shapes/magnitudes and save the proposal before touching any held-out benchmark. Do not imply the script in the Breakout demo can edit a 0.8B-model; it currently supports only the 600→32→32→6 MLP.

### 5. Fair controls and evaluation

Compare:
1. Original Qwen checkpoint.
2. Recreated authors' edit.
3. Independent editor LLM proposed edit.
4. Equal-sized random-column, shuffled-direction and scale-matched intervention controls.
5. Optional decoding-only baselines at matching latency/length budgets.

Run same evaluator/prompt templates/seeds/precision. Measure MATH-500, GSM8K, IFEval (when available), repetition rate, fraction length-capped, mean output tokens, edit density, accuracy conditioned on completing. Show confidence intervals and potential regressions; report all attempts including failures. **Do not** infer the edit created new mathematical knowledge if improvement arises from reduced looping.

### 6. Visual demonstrations

Show actual identical prompt completions for original and patched models; mark truncation, tokens emitted, correctness and elapsed time. Avoid cherry-picked demo as quantitative evidence. Display benchmark bars, token-cap reduction and per-layer delta heatmaps once measured.

## Engineering / safety rules

- Keep practice sets separate from all held-out benchmarks.
- No automatic execution of free-form model-generated Python.
- Pin every dependency and source commit; use unmodified model copy for each candidate.
- Test for behavioral regressions and avoid claiming broad generalization from a few math metrics.
- Store both author-published reference scores and locally measured scores as **separate namespaces**, never on a shared chart without clear labeling.

## Current repository content

The implemented `weightspace` package remains a small **toy Breakout** reproduction only. See [Breakout archive](breakout.md) for running code and charts. No experiment here has edited the weights of Qwen or any other pretrained LLM yet.
