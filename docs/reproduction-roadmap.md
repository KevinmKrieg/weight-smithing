# Weight Smithing — Roadmap

The project now centers on **Experiment 8** of [Takehi et al. (2026)](https://rikiyat.github.io/blog/weight-space-programming/): an LLM modifies another LLM's pretrained weights to fix a reasoning-loop failure.

## Completed
- [x] Reproducible small-network direct-weight-writing demonstration on a **toy** Breakout
- [x] Benchmark and sticky-action charts for that demonstration (archived on [Breakout page](breakout.md))
- [x] Verified Experiment 8 paper claims, provenance and reproducibility limitations
- [x] Front-page experiment comparison with reported numbers **clearly attributed to the original authors**

## Next priority: real language model
- [ ] Pin Qwen3.5-0.8B checkpoint / tokenizer / environment and record model manifest
- [ ] Replicate baseline loop behavior on practice examples (no held-out evaluation leakage)
- [ ] Replay exact authors' numerical patch from released logs and verify sparse diffs
- [ ] Generate a fresh patch via an LLM editor without gradients
- [ ] Evaluate held-out math, instruction following, length-caps and failure regressions
- [ ] Add actual side-by-side examples and locally measured benchmark charts

See [Experiment 8 protocol](experiment8-plan.md).

**Scope disclosure:** The repo contains no implemented Qwen benchmark or LLM-authored patch. The paper charts on the homepage are illustrative *published results*, not claimed local measurements.
