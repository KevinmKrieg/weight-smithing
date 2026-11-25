<div align="center">
  <img src="assets/llm-weight-editor.svg" width="100%" alt="Weight Smithing: can one LLM alter the weights of another LLM?" />
</div>

# Weight Smithing

**Can an LLM improve a different, pretrained language model by directly modifying its neural weights—with no gradient descent?**

We tried it on [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), using our own held-out instruction-following benchmark. A GPT-6-designed intervention changed a single output-weight row, fixing a specific JSON-formatting failure without fine-tuning or an inference-time token blacklist.

This is **our measured experiment**, not a reproduction of another paper's benchmark numbers.

## The result

<img src="assets/qwen-heldout-accuracy.svg" width="100%" alt="Measured Qwen accuracy: 46.9% original, 71.9% after direct weight editing, 46.9% for both controls" />

| Fresh held-out benchmark | Original Qwen | Edited Qwen |
|:--|--:|--:|
| **Overall accuracy (32 prompts)** | 46.9% (15/32) | **71.9% (23/32)** |
| Exact JSON (8 prompts) | 0% | **100%** |
| Arithmetic (8 prompts) | 75% | 75% |
| Uppercase-only response (8 prompts) | 100% | 100% |
| Alphabetical selection (8 prompts) | 12.5% | 12.5% |

**+25.0 percentage points**, paired 95% bootstrap CI **[+9.38, +40.62]**. Eight held-out answers improved; none regressed. Both norm-matched controls—altering the same output row in a random direction or the same direction in a random token row—remained at **46.9%**.

Only **896 of 494,032,768 parameters** changed: approximately **0.00018% of Qwen's weights**.

[Full measured results, model revision and source links](results/qwen-output-row-pilot-002.json) · [Raw generations and saved edit vector](https://github.com/KevinmKrieg/weight-smithing/actions/runs/41532760652/artifacts/13181044922) · [Successful GitHub Actions run](https://github.com/KevinmKrieg/weight-smithing/actions/runs/41532760652)

## What actually changed?

The unmodified model kept returning valid JSON **inside Markdown code fences**, even when instructed to return raw JSON. GPT-6 identified that practice-set failure mode and proposed directly changing the output-weight row responsible for the code-fence token.

<img src="assets/qwen-before-after.svg" width="100%" alt="An actual held-out Qwen response showing incorrect Markdown-wrapped JSON before editing and valid raw JSON after editing" />

Our program took the model's final hidden representations on **practice failures only**, computed a small vector aligned with those states, and subtracted it from the code-fence token's output-projection weights. The targeted token's average practice-prompt logit was lowered by a fixed, precommitted margin of 6.0. Then the edited model generated normally under exactly the same decoding settings as the unmodified model.

- **Editor:** GPT-6 designed the intervention in ChatGPT after reading practice-only model diagnostics. It did **not** individually generate all 896 numerical edits or use an autonomous editor API for this trial.
- **Target:** Pretrained Qwen2.5-0.5B-Instruct, pinned to revision 7ae557604adf67be50417f59c2c2f167def9a775.
- **Actual edit:** A single 896-weight LM-head row (token ID 73594, the Markdown code-fence token); L2 magnitude approximately 0.0217.
- **Optimization:** Zero gradients, optimizer steps, LoRA updates, fine-tuning examples or loss-minimization steps.
- **Evaluation:** Greedy decoding on 32 independently generated held-out tasks, using a fresh seed not used for the preceding failed experiment.

<img src="assets/qwen-by-task.svg" width="100%" alt="Measured accuracy by task: targeted improvement from 0 to 100 percent on JSON, with other families unchanged" />

**Interpretation:** This is a real, narrow output-formatting repair, **not** evidence that the model learned new mathematics or improved general intelligence. It is also more constrained than unrestricted LLM authoring of an arbitrary weight patch. Additional out-of-domain benchmarks and seeds are required before making wider claims.

[Detailed experiment, validity limitations and exact reproduction settings](docs/qwen-output-edit.md)

## We also recorded what did *not* work

Our initial attempt used GPT-6 to select **six internal MLP columns** from practice activation statistics, editing 5,376 weights. On a separate 32-prompt held-out set, all four model variants—including the edited version, an activation-ranked baseline and a random control—scored **37.5%**, producing identical responses.

We preserved this negative result rather than replacing it with the successful trial: [Pilot 001 details](docs/qwen-pilots.md) · [Full source run](https://github.com/KevinmKrieg/weight-smithing/actions/runs/41531166416).

## Reproduce the weight edit

Python 3.10+ and sufficient memory for a ~0.5B-parameter model:

~~~bash
git clone https://github.com/KevinmKrieg/weight-smithing.git
cd weight-smithing
python -m pip install -e '.[target,dev]'
pytest -q

python -m weightsmith.targeted \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --seed 41 --heldout-seed 7802 \
  --practice-per-family 2 --heldout-per-family 8 \
  --regression-per-family 3 --max-new-tokens 32 \
  --logit-margin 6.0 \
  --out artifacts/qwen-output-edit-001
~~~

This command downloads the target checkpoint, computes the intervention using practice prompts, locks the patch **before** any held-out scoring, and compares base, edited and both controls. No editor API key is required to replay this fixed GPT-6-designed strategy.

For an **API-driven editor LLM** that chooses sparse internal MLP columns from new practice diagnostics, see [the broader experiment harness](docs/research-protocol.md). That path is available, but its first recorded attempt had **no improvement**.

The project also retains our earlier [small-network Breakout demonstration](docs/breakout.md) as supporting background.

## Neuron-basis interpretability (new research phase)

**Pilot 004: the first successful internal MLP weight repair.** Four down-projection columns (3,584 weight positions) were zeroed using practice-only causal evidence. On 48 held-out tasks, accuracy rose from 39.6% to 64.6% (+25 pp, paired bootstrap 95% CI +12.5 to +37.5 pp; 12 wins, zero losses). Both base and edited models scored 12/12 on separate required-fence JSON prompts. Activation-only editing scored 62.5%, random editing 39.6%; superiority over activation-only is not established.

| Variant | Accuracy | Raw JSON | Required fences |
|---|---:|---:|---:|
| Base | 19/48 | 0/12 | 12/12 |
| Causal edit | 31/48 | 12/12 | 12/12 |
| Activation only | 30/48 | 12/12 | 12/12 |
| Random | 19/48 | 0/12 | 12/12 |

Pilots 005–006: all three auditing heuristics correctly identified 8/8 synthetic direction patches/nulls; native neuron, centroid and ridge concept scores each reached macro AUROC 1.0 on 36 held-out examples (keyword 0.871). These easy benchmarks establish feasibility, not superiority.

[Protocol and limitations](docs/neuron-basis.md) · [Verified run summaries](results/neuron-basis-pilots-003-006.json) · [Atlas/repair run](https://github.com/KevinmKrieg/weight-smithing/actions/runs/45644467541) · [Applications run](https://github.com/KevinmKrieg/weight-smithing/actions/runs/45644786421).

## Next research steps

[Pilots 013–015](docs/iterative-editor.md) evaluate iterative constrained LLM editing with practice feedback and equal-budget random search, frozen patches on prose extraction and regression tasks, and a cross-architecture SmolLM2 target. **Verified results:** Pilot 013: editor selected no-op after four trials (0/24 raw), random search repaired 3/24, Jacobian 19/24 with one fence loss. Pilot 014: frozen Jacobian raw 37/48 vs 0/48, fenced 36/48 vs 45/48, unrelated 18/48 vs 17/48; overall 91/144 vs 62/144, 39 wins and 10 losses. Pilot 015: zero eligible failures in 18 practice prompts; no intervention evaluated. [Full summaries](results/iterative-editor-013-015.json) · [CI](https://github.com/KevinmKrieg/weight-smithing/actions/runs/45681202347).

[Pilots 010–012](docs/neuron-transfer.md) test two constrained local editors, partial attenuation chosen on practice data, and transfer of the selection method to a larger model and exact-label failure mode. **Verified results:** Pilot 010: Coder selected four constrained edits, 17/48 vs 15/48 base (+4.17 pp, 95% CI [0,10.42]), zero observed losses; Instruct abstained and both free proposals were invalid. Jacobian scored 29/48, with one required-fence win and one loss. Pilot 011: both patch families selected full ablation on practice; Jacobian 31/48 vs 16/48 base (+31.25 pp, CI [18.75,45.83]), zero observed losses including required fences. Pilot 012: zero eligible prefix failures in 16 practice prompts; no edit evaluated. [Full summaries](results/neuron-transfer-010-012.json) · [CI](https://github.com/KevinmKrieg/weight-smithing/actions/runs/45671554202).

[Validation pilots 007–009](docs/neuron-validation.md) extend the internal edit to fresh prompt templates, compare target-specific downstream Jacobian readouts, and test a separate autonomous local model editor. **Results verified November 25, 2026.** Pilot 007: the unchanged four-column causal patch improved 31/96 to 47/96 (+16.7 pp), including 16/48 raw JSON repairs, but required-fence accuracy fell from 43/48 to 41/48. Pilot 008: local Jacobian prediction MAE 0.07377 vs direct lens 0.64176, correlation 0.990 vs 0.774 on the selected candidate union; held-out accuracy 31/48 vs 18/48 base, with one unrelated regression and net one required-fence loss (21/24 vs 22/24). Pilot 009: autonomous Qwen 1.5B output failed JSON parsing; no autonomous edit was applied or evaluated. Workflow success does not imply editor success. [Full verified summaries](results/neuron-validation-007-009.json) · [CI run](https://github.com/KevinmKrieg/weight-smithing/actions/runs/45659572618).

- Scale beyond 32 held-out prompts and validate with multiple different seeds.
- Measure regressions on tasks where Markdown fences are *required* or useful.
- Test a second small open LLM and a more substantial failure mode.
- Allow an editor LLM to propose less-constrained numerical changes without gradient feedback.

---

<sub>Independent MIT-licensed project inspired by weight-space programming. Published numbers here are from our own project experiments, with raw results and provenance linked. Not affiliated with the source paper's authors or Qwen developers.</sub>
