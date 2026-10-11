"""Milestone 4 part 5: `notbydefault why` answers only from records (mocked CLI and LLM)."""
import json

import pytest

from conftest import FakeEntireRunner, FakeTransport, make_client
from notbydefault.entire import Entire
from notbydefault.fit.tracing import Tracer
from notbydefault.graph import Graph
from notbydefault.why import why

CHECKPOINTS = json.dumps([{"checkpoint_id": "01CPCPCPCPCPCPCPCPCPCPCPCP", "session_id": "sess-9",
                           "message": "Fit test: keep gpt-oss-120b for summaries"}])
HITS = json.dumps([{"id": "01SRSRSRSRSRSRSRSRSRSRSRSR", "type": "checkpoint",
                    "title": "chose gpt-oss-120b after the fit test"}])


@pytest.fixture
def project(tmp_path):
    with Graph.for_project(tmp_path) as g:
        d = g.add_node("decision", "openai/gpt-oss-120b", {
            "kind": "model", "category": "ai-model", "status": "reviewed",
            "verdict_summary": "Keep it: 100% pass rate at $1.51/month.",
            "decision_record": "decisions/0001-ai-model-openai-gpt-oss-120b.md",
            "affected_source": "entire-graph",
            "affected_code": [{"path": "app.py", "symbol": "summarize_notes", "line": 22,
                               "via": "uses 'openai/gpt-oss-120b' (line 27)", "depth": 0}]})
        a = g.add_node("alternative", "groq:openai/gpt-oss-20b", {"reason": "invents facts"})
        g.add_edge(d, a, "rejects")
        e = g.add_node("evidence", "Fit test results", {"path": ".notbydefault/results/1.json"})
        g.add_edge(d, e, "supported_by")
        g.add_node("decision", "MongoDB", {"kind": "service", "category": "database",
                                           "status": "unreviewed", "tool_name": "Bash",
                                           "evidence": "pymongo"}, "sess-1")
    return tmp_path


def run(project, q, search=1, transport=None, use_llm=True):  # tests opt in
    out, t = [], Tracer()
    runner = FakeEntireRunner({("search",): search, ("checkpoint", "list"): CHECKPOINTS})
    tr = transport or FakeTransport()
    code = why(project, q, client=make_client(transport=tr), entire=Entire(exe="entire", runner=runner),
               tracer=t, out=out.append, use_llm=use_llm)
    return code, "\n".join(out), t, tr


def test_answer_cites_record_evidence_code_and_checkpoint(project):
    code, text, t, tr = run(project, "why did we pick gpt-oss-120b?")
    assert code == 0 and "The fit test kept it [F1]." in text
    for src in ("decision #1, decisions/0001-ai-model-openai-gpt-oss-120b.md", "alternative #2",
                "evidence #3, .notbydefault/results/1.json", "entire-graph: app.py:22",
                "checkpoint 01CPCPCPCPCPCPCPCPCPCPCPCP, session sess-9"):
        assert src in text, src
    assert "local checkpoints and git log (entire search unavailable" in text
    assert any(e["name"] == "search_fallback" and e["focus_area"] == "recovery" for e in t.journal)
    assert [k for _, k in tr.calls] == ["why"]


def test_entire_search_hits_are_cited_when_available(project):
    _, text, _, _ = run(project, "why gpt-oss-120b", search=HITS)
    assert "entire checkpoint 01SRSRSRSRSRSRSRSRSRSRSRSR" in text and "searched with: entire search" in text


def test_no_recorded_reason_is_said_and_no_llm_call(project):
    code, text, _, tr = run(project, "why MongoDB?")
    assert "There is no recorded reason for this decision." in text
    assert "session sess-1" in text and tr.calls == []          # nothing to phrase: no model


def test_unknown_topic_says_no_recorded_reason(project):
    code, text, _, tr = run(project, "why Kubernetes?")
    assert "There is no recorded reason" in text and tr.calls == []


def test_answer_citing_unknown_ids_is_discarded(project):
    _, text, t, _ = run(project, "why did we invent gpt-oss-120b?")
    assert "2019" not in text and "CTO" not in text
    assert "Keep it: 100% pass rate" in text                      # the plain recorded facts
    assert any(e["name"] == "citations_rejected" for e in t.journal)


def test_steps_carry_focus_areas(project):
    from notbydefault.fit.tracing import FOCUS_AREAS
    _, _, t, _ = run(project, "why gpt-oss-120b", use_llm=False)
    steps = {e["name"]: e["focus_area"] for e in t.journal if e["kind"] == "step"}
    assert steps["plan"] == "planning" and steps["graph_lookup"] == "state"
    assert steps["history_lookup"] == "context"
    assert all(e["focus_area"] in FOCUS_AREAS for e in t.journal)


def test_needs_review_note_is_not_a_reason_it_was_chosen(project):
    """Found live: a model turned 'needs review: depends on deployment' into 'chosen because'."""
    with Graph.for_project(project) as g:
        [mongo] = [d for d in g.nodes("decision") if d["label"] == "MongoDB"]
        g.update_data(mongo["id"], status="needs_review",
                      review_reason="where its data lives depends on how it is deployed")
    code, text, _, tr = run(project, "why MongoDB?")
    assert "There is no recorded reason for this decision." in text
    assert "Status (not a reason it was chosen)" in text and tr.calls == []
