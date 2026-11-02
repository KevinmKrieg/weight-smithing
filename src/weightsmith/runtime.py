"""Hugging Face Qwen2.5 target loader and deterministic text generation."""
from __future__ import annotations
import platform

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def load_model(checkpoint: str = MODEL, device: str = "auto", revision: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    kwargs = {"revision": revision} if revision else {}
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=False, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=dtype, trust_remote_code=False, **kwargs
    ).to(device).eval()
    if model.config.model_type != "qwen2":
        raise ValueError(f"Only qwen2-family architecture supported, got {model.config.model_type}")
    return model, tokenizer, {
        "checkpoint": checkpoint, "revision": revision or "main",
        "resolved_checkpoint_revision": getattr(model.config, "_commit_hash", None),
        "device": device, "dtype": str(dtype),
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__, "python": platform.python_version(),
    }


def generate_one(model, tokenizer, task, max_new_tokens: int) -> dict:
    import torch
    ids = tokenizer.apply_chat_template([{"role": "user", "content": task.prompt}],
                                        add_generation_prompt=True,
                                        return_tensors="pt").to(next(model.parameters()).device)
    with torch.inference_mode():
        output = model.generate(input_ids=ids, do_sample=False, max_new_tokens=max_new_tokens,
                                use_cache=True, pad_token_id=tokenizer.eos_token_id)
    tokens = output[0, ids.shape[1]:].tolist()
    return {"response": tokenizer.decode(tokens, skip_special_tokens=True),
            "generated_token_ids": tokens}


def evaluate(model, tokenizer, tasks, max_new_tokens=64):
    from .benchmark import score_rows
    return score_rows(tasks, [generate_one(model, tokenizer, t, max_new_tokens) for t in tasks],
                      max_new_tokens)
