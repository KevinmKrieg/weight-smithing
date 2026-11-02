"""Practice-only forward-pass diagnostics for real MLP channel weights.

No gradients and no heldout prompts. Channel suggestions are only correlations.
"""
from __future__ import annotations
import numpy as np
from .patch import down_projection


def candidate_report(model, tokenizer, practice_rows: list[dict],
                     layers: tuple[int, ...] = (8, 16), topk: int = 12) -> dict:
    import torch
    readings = {layer: [] for layer in layers}
    hooks = []
    try:
        for layer in layers:
            module = model.model.layers[layer].mlp.down_proj

            def capture(module, inputs, layer=layer):
                readings[layer].append(inputs[0][0, -1, :].detach().float().cpu().numpy().copy())

            hooks.append(module.register_forward_pre_hook(capture))
        device = next(model.parameters()).device
        for row in practice_rows:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                add_generation_prompt=True, return_tensors="pt"
            ).to(device)
            with torch.inference_mode():
                model(input_ids=ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()

    failures = np.array([not r["correct"] for r in practice_rows])
    contrast = failures.any() and (~failures).any()
    result = {
        "source": "practice prompts only; heldout unseen",
        "comparison": "failed-minus-successful standardized activation" if contrast
                      else "no correct/incorrect contrast: absolute mean activation",
        "n_failed": int(failures.sum()), "n_correct": int((~failures).sum()), "layers": {},
    }
    for layer in layers:
        activations = np.stack(readings[layer])
        weights = down_projection(model, layer).detach().float().cpu().numpy()
        norms = np.linalg.norm(weights, axis=0)
        if contrast:
            failure_mean = activations[failures].mean(axis=0)
            correct_mean = activations[~failures].mean(axis=0)
            strength = np.abs((failure_mean - correct_mean) /
                              (np.std(activations, axis=0) + 1e-3))
        else:
            failure_mean = activations.mean(axis=0)
            correct_mean = np.zeros(activations.shape[1])
            strength = np.abs(failure_mean) / (np.std(activations, axis=0) + 1e-3)
        strength = np.nan_to_num(strength, nan=0, posinf=0, neginf=0)
        indices = np.argsort(-strength, kind="stable")[:topk]
        result["layers"][str(layer)] = {
            "projection_shape": list(weights.shape),
            "candidates": [{
                "channel": int(i),
                "score": round(float(strength[i]), 4),
                "failed_mean_activation": round(float(failure_mean[i]), 5),
                "correct_mean_activation": round(float(correct_mean[i]), 5),
                "weight_column_l2": round(float(norms[i]), 4),
                "weight_fingerprint": [
                    {"output_dim": int(j), "value": round(float(weights[j, i]), 5)}
                    for j in np.argsort(-np.abs(weights[:, i]), kind="stable")[:5]
                ],
            } for i in indices],
        }
    return result
