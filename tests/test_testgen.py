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


CLASSES = """import os
from openai import OpenAI

MODEL = "openai/gpt-oss-120b"


def _ask(text):
    client = OpenAI(base_url="https://groq.test/v1", api_key=os.environ["FAKE_GROQ_KEY"])
    reply = client.chat.completions.create(model=MODEL,
                                           messages=[{"role": "user", "content": text}])
    return reply.choices[0].message.content


class Plain:                       # no-arg constructor
    def __init__(self, tone="short"):
        self.tone = tone

    def run(self, notes):
        return _ask(f"{self.tone}: {notes}")


class NeedsArgs:                   # __init__ mocked, self.notes set from the case
    def __init__(self, notes, db):
        raise RuntimeError("would connect")

    def run(self):
        return _ask(self.notes)


class Hopeless:                    # neither: self.conn must be a real connection
    def __init__(self, conn):
        self.conn = conn

    def run(self):
        return _ask(self.conn.cursor().fetch())
"""


def test_methods_get_an_instance_or_are_skipped_with_the_reason(tmp_path):
    (tmp_path / "svc.py").write_text(CLASSES, encoding="utf-8")
    sites = [Site("svc.py", f"{c}.run", 1, "uses literal") for c in ("Plain", "NeedsArgs",
                                                                    "Hopeless")]
    sites.append(Site("svc.py", None, None, "touched"))              # no symbol: not callable
    case = {"id": "c1", "source": "reference", "user_message": "ship it",
            "fields": {"notes": "ship it"}, "reply": "done", "expect": "done"}
    path = testgen.test_path(tmp_path, 1)
    path.parent.mkdir()
    text = testgen.render(tmp_path, 1, "g:m", sites, None, [case], NEW, ["FAKE_GROQ_KEY"])
    assert "'svc.py:None'" not in text
    path.write_text(text, encoding="utf-8")
    before = testgen.run(tmp_path, path)
    assert (before["passed"], before["failed"], before["skipped"]) == (0, 2, 1), before
    (tmp_path / "svc.py").write_text(CLASSES.replace("openai/gpt-oss-120b",
                                                     "gemini-3.5-flash-lite"), encoding="utf-8")
    after = testgen.run(tmp_path, path)
    assert (after["passed"], after["failed"], after["skipped"]) == (2, 0, 1), after
    assert testgen.counts_text(after) == "2/2 passed, 1 skipped"
    [reason] = after["skip_reasons"]
    assert reason.startswith("Hopeless.run not reached: constructor: TypeError")
    assert "mocked __init__: AttributeError" in reason


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
    assert testgen.counts_text(before) == "0/4 passed"
    assert testgen.counts_text(after) == "4/4 passed"
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
