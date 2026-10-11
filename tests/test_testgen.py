"""Generated tests: cases from the failure bank and reference answers, a fixed template, the
model call mocked; run on the demo app's call site before and after switching its model."""
import json
import shutil
from pathlib import Path

from notbydefault import testgen
from notbydefault.fit import failures as failures_mod
from notbydefault.resolve import Site

DEMO = Path(__file__).resolve().parents[1] / "demo-app"
ROWS = [json.loads(l) for l in (DEMO / "eval" / "action_item_cases.jsonl")
        .read_text(encoding="utf-8").splitlines() if l.strip()]
SITE = Site("actions.py", "extract_action_items", 28, "uses literal")
NEW = {"model": "gemini-3.5-flash-lite", "base_url": None}


def test_cases_bank_first_then_reference_one_per_input(tmp_path):
    rows = [{"id": "r1", "user_message": "a", "reference": {"x": 1}, "notes": "n"},
            {"id": "plain", "user_message": "b"},                     # no reference, not banked
            {"id": "r2", "user_message": "c", "reference": "said c", "hard": True},
            {"id": "dup", "user_message": "a", "reference": {"x": 2}}]
    failures_mod.append(tmp_path, 1, [{"input": "c", "output": "bad c", "model": "g:m"},
                                      {"input": "c", "output": "worse c", "model": "g:n"},
                                      {"input": "z", "output": "bad z", "model": "g:m"}])
    rows.insert(0, {"user_message": "z", "source": "failure-bank"})
    got = testgen.cases(tmp_path, 1, rows)
    assert [(c["id"], c["source"]) for c in got] == [
        ("case-1", "failure bank"), ("r1", "reference"), ("r2", "failure bank")]
    z, r1, r2 = got
    assert z["reply"] == "bad z" and z["expect"] is None             # banked output replayed
    assert r1["reply"] == '{"x": 1}' and r1["expect"] == {"x": 1}
    assert r1["fields"] == {"notes": "n"}                             # only call-site inputs
    assert r2["reply"] == "said c"                                    # reference beats banked


def test_template_is_deterministic(tmp_path):
    cases = testgen.cases(tmp_path, 1, ROWS[:3])
    a = testgen.render(tmp_path, 1, "gemini:gemini-3.5-flash-lite", [SITE], SITE.describe(),
                       cases, NEW, ["GROQ_API_KEY"])
    b = testgen.render(tmp_path, 1, "gemini:gemini-3.5-flash-lite", [SITE], SITE.describe(),
                       cases, NEW, ["GROQ_API_KEY"])
    assert a == b
    compile(a, "generated.py", "exec")
    assert "'primary': True" in a and ROWS[0]["id"] in a


def test_methods_and_module_level_sites_are_not_called(tmp_path):
    text = testgen.render(tmp_path, 1, "g:m", [SITE, Site("app.py", "Handler.do_POST", 9, "calls"),
                                               Site("other.py", None, None, "touched")],
                          None, [], NEW, [])
    assert "Handler.do_POST" not in text and "other.py" not in text


def test_generated_tests_fail_before_and_pass_after_the_switch(tmp_path):
    """The demo call site takes (notes, meeting_date): both come from the case's fields, the
    date converted from its annotation; fed the reference, it returns the reference items."""
    shutil.copy2(DEMO / "actions.py", tmp_path / "actions.py")
    cases = testgen.cases(tmp_path, 1, ROWS[:4])
    path = testgen.test_path(tmp_path, 1)
    path.parent.mkdir()
    path.write_text(testgen.render(tmp_path, 1, "gemini:gemini-3.5-flash-lite", [SITE],
                                   SITE.describe(), cases, NEW, ["GROQ_API_KEY"]),
                    encoding="utf-8")
    before = testgen.run(tmp_path, path)
    assert (before["passed"], before["failed"], before["errors"]) == (0, 4, 0), before
    src = tmp_path / "actions.py"
    src.write_text(src.read_text(encoding="utf-8").replace('"openai/gpt-oss-120b"',
                                                           '"gemini-3.5-flash-lite"'),
                   encoding="utf-8")
    after = testgen.run(tmp_path, path)
    assert (after["passed"], after["total"]) == (4, 4), after
    assert testgen.counts_text(before) == "0/4 passed" and testgen.counts_text(after) == "4/4 passed"
    assert not (tmp_path / "tests" / "__pycache__").exists()


def test_a_reply_the_code_cannot_parse_fails_the_reference_check(tmp_path):
    shutil.copy2(DEMO / "actions.py", tmp_path / "actions.py")
    src = tmp_path / "actions.py"
    src.write_text(src.read_text(encoding="utf-8").replace('"openai/gpt-oss-120b"',
                                                           '"gemini-3.5-flash-lite"'),
                   encoding="utf-8")
    [case] = testgen.cases(tmp_path, 1, ROWS[:1])
    case["reply"] = "```json\n" + case["reply"] + "\n```"             # fenced: json.loads fails
    path = testgen.test_path(tmp_path, 1)
    path.parent.mkdir()
    path.write_text(testgen.render(tmp_path, 1, "gemini:gemini-3.5-flash-lite", [SITE],
                                   SITE.describe(), [case], NEW, ["GROQ_API_KEY"]),
                    encoding="utf-8")
    got = testgen.run(tmp_path, path)
    assert (got["passed"], got["failed"]) == (0, 1), got
