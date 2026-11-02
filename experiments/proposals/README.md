# Qwen pilot 001: GPT-6-authored sparse patch

- **Target:** Qwen/Qwen2.5-0.5B-Instruct
- **Exact HF revision:** 7ae557604adf67be50417f59c2c2f167def9a775
- **Practice run:** [GitHub Actions 41530657440](https://github.com/KevinmKrieg/weight-smithing/actions/runs/41530657440)
- **Practice accuracy:** 4/8 (arithmetic 2/2, JSON 0/2, uppercase 1/2, selection 1/2)
- **Diagnostic inputs:** 8 practice-only prompts, generated baseline responses, and MLP down-projection channel statistics and weight fingerprints from layers 8 and 16.
- **Author:** GPT-6 (ChatGPT), human-in-the-loop creation of this JSON proposal. This is *not* an OpenAI API-driven autonomous editor.
- **Method:** Six selected MLP input channels, each mapped to a full 896-weight column of the corresponding down-projection matrix, scaled by the specified multiplier. Potentially 5,376 changed weights if all elements are nonzero. No gradients or model training.
- **Controls:** Original Qwen, equal-count/scale random columns, and activation-ranked-only column selection.
- **Heldout split:** 8 items per family (32 total), separate deterministic RNG namespace. Extra 3 per family (12) as a same-family regression split.
- **Editing constraint:** Fixed before heldout evaluation. Scores were unknown when this patch was written.

**Hypothesis, not a prediction:** The selected neurons show practice failure-correlated activations. Attenuation may be beneficial, harmful, or inert. We must report whichever result occurs. Observational channel contrasts do not prove causal function.

This README records provenance, not performance. Run the accompanying workflow to produce independently scored evaluations.
