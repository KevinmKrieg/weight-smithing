# Two controlled Qwen pilots

## Pilot 001 — Sparse MLP columns (completed, no effect)

- [Source GitHub Actions run](https://github.com/KevinmKrieg/weight-smithing/actions/runs/41531166416) and [complete JSON outputs/artifact](https://github.com/KevinmKrieg/weight-smithing/actions/runs/41531166416/artifacts/13182417559).
- Qwen2.5-0.5B-Instruct, pinned revision 7ae557604adf67be50417f59c2c2f167def9a775.
- Eight practice examples, then 32 separate heldout examples, greedy decoding.
- GPT-6 inspected practice activation/weight diagnostics and selected six down-projection columns (5,376 changed scalar weights). The patch was saved before heldout testing.
- Base, GPT-6 edit, activation-ranked control, and matched random control all scored **12/32 (37.5%)**.
- The edited model returned **identical text on every heldout question**: the intervention did not improve the benchmark.

This is a **negative result**, not evidence of weight-editing success. All runs and control outputs remain available in the artifact. Do not hide this when presenting a later positive experiment.

## Pilot 002 — Targeted output-weight row (precommitted, separate experiment)

Observation from *practice only*: Qwen answered strict JSON questions with fenced Markdown rather than raw JSON. GPT-6 proposed directly suppressing the output-projection row for the code-fence token. The intervention derives a *single* weight-row delta from the average final hidden representation on **practice JSON failures**, with a pre-fixed target-logit reduction of 6.0. It is not gradient descent and uses no reward feedback.

Two norm-matched controls: identical weight-row norm change in a random direction, and identical direction on a random vocabulary row. The new heldout tasks use **seed 7802** rather than seed 41, so the heldout results of Pilot 001 were not available when the second design was fixed.

This is **a narrow output-formatting intervention**, not a claim that a language model learned more general reasoning skills. See [research protocol](research-protocol.md). Do not label results before its evaluation completes.
