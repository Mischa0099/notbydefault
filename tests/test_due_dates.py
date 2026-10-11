"""due_dates_present: no item without a due date the notes clearly state (v4 verifier gap)."""
import json
from pathlib import Path

import pytest

from notbydefault.fit.checks import FACTORIES, due_dates_present

ROOT = Path(__file__).resolve().parents[1]
HELDOUT = json.loads((ROOT / "eval/cases/heldout/due_dates.json").read_text(encoding="utf-8"))
CASES = [json.loads(l) for l in
         (ROOT / "demo-app/eval/action_item_cases.jsonl").read_text(encoding="utf-8").splitlines()
         if l.strip()]
# The one held-out case a deterministic check doesn't get: the date is in a follow-up
# sentence that only says "He ... it" (no pronoun resolution, on purpose).
KNOWN_MISS = {"hd-dd-pronoun-next-sentence"}


def run(notes_or_message: str, items: list[dict], meeting: str | None = None):
    text = (f"Meeting date: {meeting} (Monday)\n\nNotes:\n{notes_or_message}" if meeting
            else notes_or_message)
    return due_dates_present().run(text, json.dumps({"action_items": items}))


@pytest.mark.parametrize("case", HELDOUT["cases"], ids=lambda c: c["id"])
def test_heldout_cases(case):
    ok, why = run(case["notes"], case["items"], HELDOUT["meeting_date"])
    if case["id"] in KNOWN_MISS:
        assert ok is True          # documented limitation, see KNOWN_MISS
    else:
        assert ok is case["pass"], why


def test_heldout_score_is_reported_honestly():
    right = sum(run(c["notes"], c["items"], HELDOUT["meeting_date"])[0] is c["pass"]
                for c in HELDOUT["cases"])
    assert right == len(HELDOUT["cases"]) - len(KNOWN_MISS)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_reference_answers_never_fail(case):
    """A correct answer must never be flagged (no false alarms)."""
    ok, why = run(case["user_message"], case["reference"]["action_items"])
    assert ok, why


def test_the_c03_gap_is_caught_with_the_resolved_date():
    case = next(c for c in CASES if c["id"].startswith("c03"))
    items = [dict(i) for i in case["reference"]["action_items"]]
    items[1]["due_date"] = None          # what gpt-oss-120b(+verifier) returned
    ok, why = run(case["user_message"], items)
    assert not ok and "next wednesday" in why and "2026-10-21" in why


def test_only_the_best_matching_sentence_counts():
    notes = "- Mo will fix the login bug by Friday.\n- Mo will also tidy the login page copy."
    items = [{"task": "fix the login bug", "owner": "Mo", "due_date": "2026-11-06", "priority": "high"},
             {"task": "tidy the login page copy", "owner": "Mo", "due_date": None, "priority": "low"}]
    assert run(notes, items, "2026-11-02")[0]


def test_invalid_json_fails_and_it_is_in_the_library():
    assert due_dates_present().run("x", "not json")[0] is False
    assert "due_dates_present" in FACTORIES
    assert due_dates_present().retryable and not due_dates_present().needs_reference


def test_reverify_compares_verifier_sets_without_requests(tmp_path):
    from notbydefault.fit.reverify import reverify
    case = next(c for c in CASES if c["id"].startswith("c03"))
    wrong = [dict(i) for i in case["reference"]["action_items"]]
    wrong[1]["due_date"] = None
    old = tmp_path / "old.py"
    old.write_text("from notbydefault.fit.checks import *\nCHECKS = [dates_normalized()]\n"
                   "RUBRIC_CHECK = None\n", encoding="utf-8")
    new = tmp_path / "new.py"
    new.write_text("from notbydefault.fit.checks import *\n"
                   "CHECKS = [dates_normalized(), due_dates_present()]\nRUBRIC_CHECK = None\n",
                   encoding="utf-8")
    out = lambda items: json.dumps({"action_items": items})
    results = tmp_path / "21-x.json"
    results.write_text(json.dumps({
        "verifiers": {"path": str(old)}, "config": {"verifier_model": "groq:m"},
        "samples": [case],
        "rows": [{"sample": 0, "model": "groq:m", "setup": "alone", "output": out(wrong),
                  "correct": False},
                 {"sample": 0, "model": "groq:m", "setup": "verified", "output": out(wrong),
                  "correct": False},
                 {"sample": 0, "model": "groq:n", "setup": "alone",
                  "output": out(case["reference"]["action_items"]), "correct": True}]}),
        encoding="utf-8")
    lines = []
    report = reverify(tmp_path, 21, new, results, out=lines.append)
    s = report["setups"]
    assert s["groq:m"]["false_pass_before"] == 1 and s["groq:m"]["false_pass_after"] == 0
    assert s["groq:m + verifier"]["retries_needed"] == 1
    assert s["groq:n"]["passed_after"] == 1               # the correct answer still passes
    assert report["requests_made"] == 0
