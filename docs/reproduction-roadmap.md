# Weight Smithing roadmap

Focus: one LLM directly editing a **different small pretrained LLM**, with a new benchmark and rigorous controls.

- [x] Synthetic automatically scored practice and held-out tasks
- [x] Safe validation and reversible changes to real MLP weight columns
- [x] Practice-only activation and weight diagnostics
- [x] LLM-editor interface and matched non-LLM controls
- [x] Real-results-only SVG plotting script and offline tests
- [ ] Run target Qwen2.5-0.5B-Instruct on real practice prompts
- [ ] Have an LLM propose a patch using the real weight diagnostics
- [ ] Evaluate against held-out questions, random edits and ranked non-LLM edits
- [ ] Publish actual results, negative or positive, with paired confidence intervals
- [ ] Repeat with larger samples, additional target models and general-capability checks

**No Qwen benchmark improvement has yet been verified.**

Read [research protocol](research-protocol.md) and [supporting Breakout experiment](breakout.md).
