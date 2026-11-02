"""Deterministic, automatically scored instruction-following tasks.

These are newly generated synthetic tasks, not IFEval or published benchmark scores.
All generations and model edits must be fixed before heldout evaluation.
"""
from __future__ import annotations
import json
import random
from dataclasses import asdict, dataclass

FAMILIES = ("arithmetic", "json", "uppercase", "selection")
WORDS = (
    "BIRCH", "CORAL", "DELTA", "EMBER", "FROST", "GROVE", "HAZEL", "IVORY",
    "JASPER", "LUNAR", "MAPLE", "NORTH", "OLIVE", "QUARTZ", "RIVER", "SOLAR",
    "TIGER", "UMBRA", "VIOLET", "WILLOW", "XENON", "YARROW", "ZENITH",
)


@dataclass(frozen=True)
class Task:
    id: str
    family: str
    prompt: str
    expected: str

    def asdict(self) -> dict:
        return asdict(self)


def make_tasks(split: str, per_family: int, seed: int = 41) -> list[Task]:
    if split not in ("practice", "heldout", "regression"):
        raise ValueError(f"Unknown split {split}")
    if per_family < 1:
        raise ValueError("per_family must be at least 1")
    rng = random.Random(f"weight-smithing:v1:{seed}:{split}")
    tasks: list[Task] = []
    for i in range(per_family):
        a, b = rng.sample(range(12, 98), 2)
        op = "+" if i % 2 == 0 else "-"
        value = a + b if op == "+" else a - b
        tasks.append(Task(f"{split}-math-{i:03d}", "arithmetic",
                          f"Compute {a} {op} {b}. Return only the integer answer; no words or punctuation.",
                          str(value)))
        ticket = rng.randint(100, 9999)
        status = rng.choice(("open", "closed", "pending"))
        tasks.append(Task(f"{split}-json-{i:03d}", "json",
                          f"Output exactly one valid JSON object with two keys, 'ticket' (integer {ticket}) "
                          f"and 'status' (string '{status}'). No Markdown or extra text.",
                          json.dumps({"ticket": ticket, "status": status}, separators=(",", ":"))))
        chosen = rng.choice(WORDS)
        tasks.append(Task(f"{split}-upper-{i:03d}", "uppercase",
                          f"Reply with the uppercase word {chosen}, and nothing else. "
                          f"Do not explain or add punctuation.", chosen))
        options = rng.sample(WORDS, 4)
        answer = sorted(options)[0]
        tasks.append(Task(f"{split}-select-{i:03d}", "selection",
                          f"From {', '.join(options)}, select the word first in alphabetical order. "
                          "Return that word in uppercase only, without punctuation.", answer))
    rng.shuffle(tasks)
    return tasks


def is_correct(task: Task, response: str) -> bool:
    raw = response.strip()
    if task.family == "json":
        try:
            data = json.loads(raw)
            expected = json.loads(task.expected)
        except (ValueError, TypeError):
            return False
        return type(data) is dict and set(data) == set(expected) and all(
            type(data[k]) is type(expected[k]) and data[k] == expected[k] for k in expected
        )
    return raw == task.expected


def repetition_ratio(tokens: list[int], ngram: int = 4) -> float:
    if len(tokens) < ngram:
        return 0.0
    chunks = [tuple(tokens[i:i + ngram]) for i in range(len(tokens) - ngram + 1)]
    return (len(chunks) - len(set(chunks))) / len(chunks)


def score_rows(tasks: list[Task], outputs: list[dict], max_new_tokens: int) -> list[dict]:
    if len(tasks) != len(outputs):
        raise ValueError("Task/output length mismatch")
    records = []
    for task, output in zip(tasks, outputs):
        tok = list(output["generated_token_ids"])
        response = output["response"]
        records.append({
            "id": task.id, "family": task.family, "prompt": task.prompt,
            "expected": task.expected, "response": response,
            "correct": is_correct(task, response),
            "generated_tokens": len(tok), "length_cap": len(tok) >= max_new_tokens,
            "repeat4": round(repetition_ratio(tok), 5),
        })
    return records


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No evaluation rows")
    return {
        "n": len(rows),
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "mean_tokens": sum(r["generated_tokens"] for r in rows) / len(rows),
        "length_cap_rate": sum(r["length_cap"] for r in rows) / len(rows),
        "mean_repeat4": sum(r["repeat4"] for r in rows) / len(rows),
        "accuracy_by_family": {
            family: sum(r["correct"] for r in rows if r["family"] == family) /
                    sum(r["family"] == family for r in rows)
            for family in FAMILIES if any(r["family"] == family for r in rows)
        },
    }


def paired_bootstrap(base: list[dict], edited: list[dict], seed: int = 2026,
                     replications: int = 3000) -> dict:
    """Paired sample-with-replacement CI for accuracy difference, in percentage points."""
    if [x["id"] for x in base] != [x["id"] for x in edited]:
        raise ValueError("Paired rows must be identical and in the same order")
    if not base:
        raise ValueError("Empty rows")
    gains = [int(b["correct"]) - int(a["correct"]) for a, b in zip(base, edited)]
    rng = random.Random(seed)
    diffs = sorted(sum(gains[rng.randrange(len(gains))] for _ in gains) / len(gains)
                   for _ in range(replications))
    return {
        "accuracy_delta_pp": round(100 * sum(gains) / len(gains), 2),
        "ci95_pp": [round(100 * diffs[int((len(diffs) - 1) * pct)], 2)
                    for pct in (.025, .975)],
        "wins": sum(x > 0 for x in gains),
        "losses": sum(x < 0 for x in gains),
        "ties": sum(x == 0 for x in gains),
    }
