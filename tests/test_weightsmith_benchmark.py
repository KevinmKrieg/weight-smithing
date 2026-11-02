import json
import pytest
from weightsmith.benchmark import (
    make_tasks, is_correct, repetition_ratio, score_rows, summarize, paired_bootstrap
)


def test_splits_are_deterministic_balanced_disjoint():
    a = make_tasks("practice", 7, seed=12)
    assert a == make_tasks("practice", 7, seed=12)
    assert len(a) == 28
    assert {t.family for t in a} == {"arithmetic", "json", "uppercase", "selection"}
    assert not ({t.id for t in a} & {t.id for t in make_tasks("heldout", 7, seed=12)})
    assert not ({t.id for t in a} & {t.id for t in make_tasks("regression", 7, seed=12)})


def test_exact_scoring_and_json_strictness():
    for task in make_tasks("practice", 3):
        assert is_correct(task, task.expected)
        assert is_correct(task, "  \n" + task.expected + "\n")
        assert not is_correct(task, task.expected + "\nextra text")
        if task.family == "json":
            expected = json.loads(task.expected)
            assert is_correct(task, json.dumps({"status": expected["status"],
                                                "ticket": expected["ticket"]}))
            assert not is_correct(task, json.dumps({**expected, "other": True}))
            assert not is_correct(task, json.dumps({**expected,
                                                    "ticket": str(expected["ticket"])}))


def test_repetition_measure():
    assert repetition_ratio([1, 2]) == 0
    assert repetition_ratio([1, 2, 3, 4, 5]) == 0
    assert repetition_ratio([2, 2, 2, 2, 2]) == pytest.approx(.5)


def test_score_summary_and_paired_bootstrap():
    tasks = make_tasks("practice", 1)
    rows = score_rows(tasks, [
        {"response": task.expected, "generated_token_ids": [1, 2, 3]}
        for task in tasks
    ], 10)
    assert summarize(rows)["accuracy"] == 1
    bad = [{**row, "correct": False} for row in rows]
    result = paired_bootstrap(rows, bad, replications=100)
    assert result["accuracy_delta_pp"] == -100
    assert result["wins"] == 0 and result["losses"] == len(rows)
    with pytest.raises(ValueError):
        paired_bootstrap(rows, bad[::-1])
