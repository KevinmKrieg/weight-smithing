"""LLM selects small, actual projection-column edits using practice-only diagnostics."""
from __future__ import annotations
import json
from .patch import validate_edits

SYSTEM = """You are an ML researcher editing the WEIGHTS of a separate small language model.
The target model failed some strict-format and reasoning practice prompts.
You have practice-only activation statistics and a fingerprint of real MLP weights,
not heldout labels or scores. Select up to 8 MLP down-projection columns using
the supplied ranked candidates, with multiplier values between 0.25 and 0.85.
Multiplying a whole column by less than one directly attenuates that neuron.
Do NOT pretend the patch was tested or optimized with gradients.
Return only raw JSON:
{"edits":[{"layer":8,"channel":123,"multiplier":0.6}],
 "rationale":"one or two sentences explaining the testable hypothesis"}
No markdown, no other keys, no Python code."""


def proposal_prompt(report: dict, practice_rows: list[dict]) -> str:
    failed = [r for r in practice_rows if not r["correct"]][:6]
    correct = [r for r in practice_rows if r["correct"]][:3]
    cases = [{k: r[k] for k in ("family", "prompt", "expected", "response")}
             for r in failed + correct]
    return json.dumps({
        "goal": "Improve exact-format and small reasoning task accuracy via weight edits only",
        "limits": "Select only reported channels, no heldout information, no iterative reward feedback",
        "practice_examples": cases,
        "weight_and_activation_diagnostics": report,
    }, ensure_ascii=False)


def heuristic_proposal(report: dict, n_per_layer: int = 4) -> dict:
    edits = []
    for layer, info in report["layers"].items():
        for candidate in info["candidates"][:n_per_layer]:
            edits.append({"layer": int(layer), "channel": candidate["channel"], "multiplier": 0.6})
    return {"edits": edits,
            "rationale": "Activation-ranked non-LLM comparison. This was not proposed by an LLM."}


def llm_proposal(report: dict, practice_rows: list[dict], editor_model: str) -> dict:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install editor dependencies: pip install -e '.[editor]'") from exc
    client = OpenAI()
    response = client.responses.create(
        model=editor_model,
        instructions=SYSTEM,
        input=proposal_prompt(report, practice_rows),
        max_output_tokens=1500,
    )
    try:
        return json.loads(response.output_text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("Editor did not return a valid raw JSON proposal; no patch applied") from exc


def choose_proposal(model, report: dict, practice_rows: list[dict],
                    mode: str, editor_model: str = "gpt-4.1",
                    proposal_path: str | None = None,
                    max_columns: int = 16,
                    layers: tuple[int, ...] = (8, 16)) -> dict:
    if mode == "openai":
        raw = llm_proposal(report, practice_rows, editor_model)
    elif mode == "heuristic":
        raw = heuristic_proposal(report)
    elif mode == "file":
        if not proposal_path:
            raise ValueError("--proposal-file is required for --editor file")
        with open(proposal_path, encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raise ValueError(f"Unknown editor mode {mode}")
    edits = validate_edits(model, raw, max_columns=max_columns, allowed_layers=layers)
    allowed = {int(layer): {c["channel"] for c in info["candidates"]}
               for layer, info in report["layers"].items()}
    if any(e.channel not in allowed[e.layer] for e in edits):
        raise ValueError("Selected channel not found in practice-only diagnostics")
    return raw
