# Two controlled Qwen pilots

## Pilot 001 — Sparse MLP columns (completed, no effect)

- [Source GitHub Actions run](https://github.com/KevinmKrieg/weight-smithing/actions/runs/41531166416) and [complete JSON outputs/artifact](https://github.com/KevinmKrieg/weight-smithing/actions/runs/41531166416/artifacts/13182417559).
- Qwen2.5-0.5B-Instruct, pinned revision 7ae557604adf67be50417f59c2c2f167def9a775.
- Eight practice examples, then 32 separate heldout examples, greedy decoding.
- GPT-6 inspected practice activation/weight diagnostics and selected six down-projection columns (5,376 changed scalar weights). The patch was saved before heldout testing.
- Base, GPT-6 edit, activation-ranked control, and matched random control all scored **12/32 (37.5%)**.
- The edited model returned **identical text on every heldout question**: the intervention did not improve the benchmark.

This is a **negative result**, not evidence of weight-editing success. All runs and control outputs remain available in the artifact. Do not hide this when presenting a later positive experiment.

## Pilot 002 — Targeted output-weight row (completed, targeted improvement)

Observation from **practice prompts only**: Qwen repeatedly returned strict JSON inside Markdown code fences. GPT-6 proposed suppressing the code-fence token by editing that token's **output-projection weight row**, using a direction computed from final hidden states on practice JSON failures. The precommitted target-logit reduction was 6.0. No gradients, optimizer, or inference-time token blacklist were used.

We pinned Qwen2.5-0.5B-Instruct (revision 7ae557604adf67be50417f59c2c2f167def9a775) and used a **fresh heldout seed (7802)**, separate from Pilot 001's heldout seed. The edit altered **896 of 494,032,768 parameters**, or approximately **0.00018%**.

| Condition | Fresh 32-prompt heldout accuracy |
|---|---:|
| Original Qwen | 15/32 — **46.9%** |
| GPT-6-designed edit | 23/32 — **71.9%** |
| Random direction, same row and edit norm | 15/32 — **46.9%** |
| Random vocabulary row, same edit norm and direction | 15/32 — **46.9%** |

The 8 changed responses were all strict-JSON cases (0/8 → 8/8). There were no observed changes on the other 24 prompts. Paired improvement: **+25.0 percentage points**, bootstrap 95% CI **[+9.38, +40.62]**.

**Interpretation:** Successful narrow formatting repair, not more general mathematical reasoning. GPT-6 devised the edit; Python computed the complete 896-element vector from practice activations. No autonomous editor API supplied individual values.

[Complete methods and output examples](qwen-output-edit.md) · [Raw model-run artifact](https://github.com/KevinmKrieg/weight-smithing/actions/runs/41532760652/artifacts/13181044922) · [Measured summary](../results/qwen-output-row-pilot-002.json).
