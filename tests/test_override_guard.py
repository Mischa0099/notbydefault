import json

from notbydefault.detect import Detection
from notbydefault.graph import Graph
from notbydefault.hooks.pre_tool import record


def _reviewed_decision(g, project, label="openai/gpt-oss-120b", status="reviewed"):
    results = project / ".notbydefault" / "results" / "21-test.json"
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(json.dumps({"trace_id": "0f65667e9f5c521dfc124c89f188f9ca"}))
    return g.add_node("decision", label, {"kind": "model", "category": "ai-model",
                                          "status": status,
                                          "results": ".notbydefault/results/21-test.json"})


def _new_model(g, project, chosen):
    det = Detection("model", chosen, "ai-model", chosen, "code", str(project / "tags.py"))
    ev = {"session_id": "s", "tool_name": "Write", "tool_use_id": "t"}    # no transcript: no I/O
    return g.get_node(record(g, det, ev, project))


def test_new_decision_overriding_a_reviewed_one_is_flagged(tmp_path):
    with Graph.for_project(tmp_path) as g:
        old = _reviewed_decision(g, tmp_path)
        new = _new_model(g, tmp_path, "llama-3.3-70b-versatile")
        assert new["data"]["status"] == "unreviewed"
        assert new["data"]["override_guard"] == {
            "status": "needs_evidence", "overrides_decision_id": old,
            "old_evidence_trace": "0f65667e9f5c521dfc124c89f188f9ca"}


def test_no_guard_without_a_reviewed_or_stale_decision_of_that_type(tmp_path):
    with Graph.for_project(tmp_path) as g:
        _reviewed_decision(g, tmp_path, status="unreviewed")
        g.add_node("decision", "MongoDB", {"kind": "service", "category": "database",
                                           "status": "reviewed"})
        new = _new_model(g, tmp_path, "llama-3.3-70b-versatile")
        assert "override_guard" not in new["data"]
