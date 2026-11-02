# Weight Smithing — research roadmap

Our main question is whether an LLM can **directly change the weights of another pretrained LLM** to fix a measurable behavior, without fine-tuning or gradients. We use our own model, prompts, and controls rather than attempting to reproduce the original paper's leaderboard.

## Completed and measured

- [x] Deterministic custom instruction-following tasks, practice/heldout separation, and strict automatic scoring
- [x] Reversible in-place edits to real pretrained Qwen2.5-0.5B-Instruct parameters
- [x] Real practice inference, activation measurements and weight inspection
- [x] **Pilot 001 (negative):** GPT-6-selected six MLP weight columns (5,376 values); base, edited and both controls scored **12/32 = 37.5%** with identical responses
- [x] **Pilot 002 (positive, narrow):** GPT-6-designed intervention on a single output-token weight row (896 values); heldout strict instruction accuracy **15/32 = 46.9% → 23/32 = 71.9%**, with **8/8 strict JSON outputs fixed**
- [x] Matched random direction/token controls remained at **15/32 = 46.9%**; paired 95% bootstrap CI on the improvement **+9.38 to +40.62 percentage points**
- [x] Preserved raw outputs and patch provenance, designed and published the README charts, passed CI

**Scope:** Pilot 002 is a focused output-format repair. It does **not** demonstrate improved reasoning, general model quality, or that GPT-6 wrote 896 scalar values individually. GPT-6 proposed the intervention, while the edit vector was computed deterministically from practice activations.

## Next research priorities

- [ ] Replicate Pilot 002 across larger heldout sets and multiple independent seeds
- [ ] Test behavioral regressions where Markdown fences are explicitly requested
- [ ] Repeat the intervention on a different open-weight target model
- [ ] Attempt a more substantive failure mode beyond output formatting
- [ ] Let an API-based editor LLM propose a fresh numerical weight patch, freeze it, and compare it against stronger controls
- [ ] Evaluate across broader, unrelated tasks and publish negative as well as positive outcomes

## Results and methods

- [Pilot history, including the negative result](qwen-pilots.md)
- [Pilot 002 detailed methods and raw-run link](qwen-output-edit.md)
- [Machine-readable measured Pilot 002 summary](../results/qwen-output-row-pilot-002.json)
- [Original MLP editing protocol (Pilot 001)](research-protocol.md)
- [Older toy Breakout supporting demonstration](breakout.md)
