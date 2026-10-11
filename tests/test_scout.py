"""Milestone 4 part 6: test scout over Entire checkpoint transcripts (mocked CLI and LLM)."""
import json

import pytest

from conftest import FakeEntireRunner, FakeTransport, make_client
from notbydefault.entire import Entire
from notbydefault.fit.tracing import Tracer
from notbydefault.graph import Graph
from notbydefault.scout import scout, suggested_path

LIST = json.dumps([{"checkpoint_id": "01TOUCH", "session_id": "sess-a"},
                   {"checkpoint_id": "01OTHER", "session_id": "sess-b"}])
TRANSCRIPT = "\n".join(json.dumps(r) for r in [
    {"type": "user", "message": {"role": "user", "content": "In app.py, what about empty notes? "
                                                             "Blank lines might crash summarize_notes"}},
    {"type": "assistant", "message": {"content": [{"type": "text", "text":
        "Good point: summarize_notes with empty notes is an edge case we haven't tested."}]}},
    {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Edit"}]}},
    {"type": "user", "message": {"content": "unrelated: rename the README"}},
])


def runner(transcript=TRANSCRIPT, listing=LIST):
    return FakeEntireRunner({
        ("checkpoint", "list"): listing,
        ("checkpoint", "explain", "01TOUCH", "--json"): json.dumps({"files_touched": ["app.py"]}),
        ("checkpoint", "explain", "01OTHER", "--json"): json.dumps({"files_touched": ["README.md"]}),
        ("checkpoint", "explain", "01TOUCH", "--transcript"): transcript,
    })


@pytest.fixture
def project(tmp_path):
    with Graph.for_project(tmp_path) as g:
        did = g.add_node("decision", "openai/gpt-oss-120b", {
            "kind": "model", "category": "ai-model", "status": "reviewed", "file": "app.py",
            "files": ["app.py"], "affected_source": "entire-graph", "affected_files": ["app.py"],
            "affected_code": [{"path": "app.py", "symbol": "summarize_notes", "line": 22,
                               "via": "uses literal", "depth": 0}]})
    return tmp_path, did


def run(project, did, transport=None, r=None):
    out, t = [], Tracer()
    r = r or runner()
    tr = transport or FakeTransport()
    code = scout(project, did, client=make_client(transport=tr),
                 entire=Entire(exe="entire", runner=r), tracer=t, out=out.append)
    return code, "\n".join(out), t, tr, r


def test_scout_keeps_quoted_untested_cases_as_suggested(project):
    p, did = project
    code, text, t, tr, r = run(p, did)
    assert code == 0 and [k for _, k in tr.calls] == ["scout"]
    # only the checkpoint that touched app.py has its transcript read
    assert ["checkpoint", "explain", "01OTHER", "--transcript"] not in r.calls
    [case] = [json.loads(l) for l in suggested_path(p, did).read_text().splitlines()]
    assert case["status"] == "suggested" and case["checkpoint"] == "01TOUCH"
    assert case["input"] == "notes with an empty line"
    assert "[suggested] notes with an empty line" in text and "[checkpoint 01TOUCH]" in text
    names = [e["name"] for e in t.journal]
    assert "drop_tested_case" in names            # the fake marks "a single note" as tested


def test_cases_with_invented_quotes_are_dropped(project):
    p, did = project
    transcript = TRANSCRIPT.replace("what about empty notes?", "what about long notes?")
    code, text, t, _, _ = run(p, did, r=runner(transcript))
    assert "0 kept" in text
    assert any(e["name"] == "drop_unquoted_case" and e["focus_area"] == "recovery" for e in t.journal)


def test_rerun_does_not_duplicate_suggestions(project):
    p, did = project
    run(p, did)
    run(p, did)
    assert len(suggested_path(p, did).read_text().splitlines()) == 1


def test_no_checkpoints_touching_the_code_means_no_llm_call(project):
    p, did = project
    listing = json.dumps([{"checkpoint_id": "01OTHER", "session_id": "sess-b"}])
    code, text, _, tr, _ = run(p, did, r=runner(listing=listing))
    assert "No discussed edge cases" in text and tr.calls == []


def test_entire_unavailable_degrades(project):
    p, did = project
    code, text, t, tr, _ = run(p, did, r=FakeEntireRunner({}))
    assert code == 0 and "No checkpoints available" in text and tr.calls == []
    assert any(e["name"] == "checkpoints_unavailable" for e in t.journal)


def test_existing_tests_per_call_site_and_caller(tmp_path):
    from notbydefault.resolve import Site
    from notbydefault.scout import existing_tests
    (tmp_path / "tests").mkdir()
    (tmp_path / "pkg").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text(
        "import app\n\ndef test_summary():\n    assert app.summarize_notes(['a'])\n\n"
        "class TestHandler:\n    def test_post(self):\n        Handler.do_POST\n\n"
        "def helper():\n    summarize_notes\n")
    (tmp_path / "tests" / "test_other.py").write_text("from pkg import jobs\n\ndef test_x():\n"
                                                      "    pass\n")
    (tmp_path / "pkg" / "jobs_test.py").write_text("def test_run():\n    run_jobs()\n")
    (tmp_path / "tests" / "broken.py").write_text("def (:\n")       # unparsable: skipped
    sites = [Site("app.py", "summarize_notes", 3, "uses literal"),
             Site("app.py", "Handler.do_POST", 9, "calls summarize_notes"),
             Site("pkg/jobs.py", "run_jobs", 1, "calls"),
             Site("billing.py", "charge", 1, "calls")]
    got = existing_tests(tmp_path, sites)
    assert got == {
        "app.py:3 summarize_notes": ["tests/test_app.py::test_summary"],    # helper() isn't a test
        "app.py:9 Handler.do_POST": ["tests/test_app.py::TestHandler::test_post"],
        "pkg/jobs.py:1 run_jobs": ["pkg/jobs_test.py::test_run",
                                   "tests/test_other.py (imports pkg.jobs)"],
        "billing.py:1 charge": []}
