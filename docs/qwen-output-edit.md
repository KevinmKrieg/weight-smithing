# Pilot 002 — Targeted output-weight edit

**Completed with a positive, narrow result.** The underlying raw completions, saved patch vector, model environment, and all controls are available as [GitHub Actions artifact](https://github.com/KevinmKrieg/weight-smithing/actions/runs/41532760652/artifacts/13181044922).

## Precommitted intervention

During the practice run, Qwen2.5-0.5B-Instruct frequently wrapped otherwise valid JSON in Markdown code fences, even when asked to output raw JSON. GPT-6 proposed directly editing the output-weight row for the observed fence token, rather than selecting MLP columns using correlation alone.

The implementation in [targeted.py](../src/weightsmith/targeted.py) identifies the specific fence token **from practice outputs only**, computes the average final hidden state on practice JSON failures, and subtracts a fixed directional change from the selected output-weight row. The coefficient is chosen to decrease that token's logit by a prespecified value (6.0) on the mean practice representation. It uses **no gradients, loss function or training**. The edited model generates normally with greedy decoding, without a token blacklist or inference-time logits processor.

GPT-6 chose the strategy. The actual 896-element edit vector was **computed deterministically** from practice forward passes, not individually authored one scalar at a time by GPT-6. This is an important limitation of the claim.

To prevent reusing heldout feedback from the failed first MLP attempt, this second trial uses a **fresh heldout seed of 7802**.

## Measured heldout results

| Variant | Exact-match correct | Accuracy |
|---|---:|---:|
| Original pretrained Qwen | 15/32 | 46.9% |
| GPT-6-designed output-row edit | **23/32** | **71.9%** |
| Random direction, same output row and L2 norm | 15/32 | 46.9% |
| Same direction and L2 norm, random token row | 15/32 | 46.9% |

**+25.0 percentage points; paired bootstrap 95% CI [+9.38, +40.62].** All 8 gains are strict-JSON prompts, where correctness increases 0/8 → 8/8. On 24 other tasks, answers did not change.

Only **896 / 494,032,768 model parameters** changed (0.000181%). The patch targets the token corresponding to a triple-backtick Markdown fence (token ID 73594). Its L2 size was just ~0.0217.

This establishes a **narrow formatting improvement**, not general reasoning or knowledge improvement. It is different from the source paper's model, benchmark and intervention.

For precise outcomes and source links, see [the committed summary](../results/qwen-output-row-pilot-002.json). For the failed first MLP intervention and controls, see [the pilot comparison](qwen-pilots.md).

## Reproduce

~~~bash
python -m pip install -e '.[target,dev]'
python -m weightsmith.targeted \
  --checkpoint Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --seed 41 --heldout-seed 7802 \
  --practice-per-family 2 --heldout-per-family 8 --regression-per-family 3 \
  --max-new-tokens 32 --logit-margin 6.0 \
  --out artifacts/qwen-output-edit-001
~~~

For a formal follow-up, expand heldout size, check additional datasets, run more random seeds, and test any regressions where Markdown code fences are actually desired.
