"""Milestone 4 part 4: the hook's briefing (graph + cached Entire lookups), mocked CLI."""
import json

import pytest

from conftest import FakeEntireRunner, demo_graph_responses, demo_project
from notbydefault import affected as affected_mod
from notbydefault import briefing
from notbydefault.entire import Entire
from notbydefault.graph import Graph
from notbydefault.hooks.pre_tool import handle

CHECKPOINTS = json.dumps([
    {"checkpoint_id": "01AAAAAAAAAAAAAAAAAAAAAAAA", "session_id": "sess-old",
     "date": "2026-10-10T12:00:00+05:30", "message": "Summaries on gpt-oss-120b via Groq",
     "session_ids": ["sess-old"]},
    {"checkpoint_id": "01BBBBBBBBBBBBBBBBBBBBBBBB", "session_id": "sess-x",
     "message": "Unrelated docs tweak"}])
EXPLAIN_A = json.dumps({"checkpoint_id": "01AAAAAAAAAAAAAAAAAAAAAAAA",
                        "files_touched": ["app.py", "README.md"]})
EXPLAIN_B = json.dumps({"checkpoint_id": "01BBBBBBBBBBBBBBBBBBBBBBBB", "files_touched": ["README.md"]})


def responses(search=1):
    r = demo_graph_responses()
    r.update({("checkpoint", "list"): CHECKPOINTS,
              ("checkpoint", "explain", "01AAAAAAAAAAAAAAAAAAAAAAAA"): EXPLAIN_A,
              ("checkpoint", "explain", "01BBBBBBBBBBBBBBBBBBBBBBBB"): EXPLAIN_B,
              ("search",): search})
    return r


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(affected_mod, "repo_root", lambda p: tmp_path)
    demo_project(tmp_path)
    with Graph.for_project(tmp_path) as g:
        old = g.add_node("decision", "openai/gpt-oss-120b", {
            "kind": "model", "category": "ai-model", "status": "reviewed",
            "verdict_summary": "Keep gpt-oss-120b: passes every check at $1.51/month.",
            "decision_record": "decisions/0001-ai-model-openai-gpt-oss-120b.md",
            "file": "app.py", "files": ["app.py"]})
        alt = g.add_node("alternative", "groq:openai/gpt-oss-20b",
                         {"reason": "invents facts in 3 of 8 outputs"})
        g.add_edge(old, alt, "rejects")
        g.add_node("decision", "OpenAI", {"kind": "service", "category": "ai-provider",
                                          "status": "unreviewed", "evidence": "openai",
                                          "file": "requirements.txt"})
    return tmp_path


def refresh(project, search=1):
    runner = FakeEntireRunner(responses(search))
    return briefing.refresh_cache(project, entire=Entire(exe="entire", runner=runner)), runner


def test_refresh_cache_collects_checkpoints_and_sdk_users_and_degrades_search(project):
    cache, runner = refresh(project)
    assert [c["files"] for c in cache["checkpoints"]] == [["app.py", "README.md"], ["README.md"]]
    users = {u["where"] for u in cache["sdk_users"]["openai"]}
    assert "actions.py:28 extract_action_items" in users
    assert "search" in cache["errors"] and "local checkpoints" in cache["errors"]["search"]
    # checkpoints are immutable: a second refresh doesn't explain them again
    _, runner2 = refresh(project)
    assert not [c for c in runner2.calls if c[:2] == ["checkpoint", "explain"]]


def test_search_hits_are_used_when_entire_search_works(project):
    hits = json.dumps([{"id": "01CCCCCCCCCCCCCCCCCCCCCCCC", "type": "checkpoint",
                        "title": "Chose gpt-oss-120b for summaries"}])
    refresh(project, search=hits)
    with Graph.for_project(project) as g:
        d = g.add_node("decision", "openai/gpt-oss-120b", {"kind": "model", "category": "ai-model",
                                                           "status": "unreviewed", "file": "actions.py"})
        lines = briefing.build(project, g, g.get_node(d), limit=None)
    assert any("[entire search 01CCCCCCCCCCCCCCCCCCCCCCCC]" in l for l in lines)


def test_briefing_has_each_kind_with_a_source_id(project):
    refresh(project)
    with Graph.for_project(project) as g:
        new = g.add_node("decision", "gpt-oss-20b", {"kind": "model", "category": "ai-model",
                                                      "status": "unreviewed", "file": "app.py",
                                                      "files": ["app.py"]})
        lines = briefing.build(project, g, g.get_node(new))
    assert len(lines) == 4
    past, rejected, hist, reuse = lines
    assert past.startswith("Past decision #1 openai/gpt-oss-120b") and "[decision #1, decisions/0001" in past
    assert rejected.startswith("Rejected before: `groq:openai/gpt-oss-20b`") and "[alternative #" in rejected
    assert "[checkpoint 01AAAAAAAAAAAAAAAAAAAAAAAA, session sess-old]" in hist
    assert reuse.startswith("Reuse: ") and "[entire-graph " in reuse


def test_hook_shows_at_most_four_lines_from_cache_only(project, monkeypatch):
    refresh(project)
    calls = []
    monkeypatch.setattr("subprocess.run", lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(
        AssertionError("hook must not run subprocesses")))
    (project / "summary.py").write_text("")
    ev = {"session_id": "s", "tool_name": "Write", "cwd": str(project),
          "tool_input": {"file_path": str(project / "summary.py"),
                         "content": 'MODEL = "openai/gpt-oss-120b"\n'}}
    out = handle(ev, project)
    reason = out["hookSpecificOutput"]["permissionDecisionReason"]
    head, brief = reason.split("\nBriefing (full version in `check`):\n")
    lines = brief.splitlines()
    assert 1 <= len(lines) <= 4 and all(l.startswith("- ") and l.endswith("]") for l in lines)
    assert calls == []


def test_hook_without_cache_still_briefs_from_the_graph(project):
    (project / "summary.py").write_text("")
    ev = {"session_id": "s", "tool_name": "Write", "cwd": str(project),
          "tool_input": {"file_path": str(project / "summary.py"),
                         "content": 'MODEL = "openai/gpt-oss-120b"\n'}}
    reason = handle(ev, project)["hookSpecificOutput"]["permissionDecisionReason"]
    assert "Past decision #1" in reason


def test_hook_journal_tags_focus_areas(project):
    from notbydefault.fit.tracing import FOCUS_AREAS, Tracer
    t = Tracer()
    ev = {"session_id": "s", "tool_name": "Bash", "cwd": str(project),
          "tool_input": {"command": "pip install pymongo"}}
    handle(ev, project, tracer=t)
    steps = [e for e in t.journal if e["kind"] == "step"]
    assert [s["name"] for s in steps] == ["detect", "record", "briefing"]
    assert all(e["focus_area"] in FOCUS_AREAS for e in t.journal)
