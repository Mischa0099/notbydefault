"""Milestone 4 part 1: affected code from Entire Graph (mocked CLI)."""
import pytest

from conftest import FakeEntireRunner, demo_graph_responses, demo_project
from notbydefault import affected as affected_mod
from notbydefault.affected import Index, analyze, record_affected, refresh
from notbydefault.entire import Entire
from notbydefault.graph import Graph, affected_entries, find_affected_code


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(affected_mod, "repo_root", lambda p: tmp_path)
    return demo_project(tmp_path)


def entire(responses=None):
    return Entire(exe="entire", runner=FakeEntireRunner(responses or demo_graph_responses()))


def decision(g, label, kind, category, files, **extra):
    did = g.add_node("decision", label, {"kind": kind, "category": category, "files": files,
                                         "file": files[0], "status": "unreviewed", **extra})
    return g.get_node(did)


def by_symbol(res):
    return {(e["path"], e["symbol"]): e for e in res.entries}


def test_model_decision_reaches_every_function_using_the_literal(project):
    with Graph.for_project(project) as g:
        d = decision(g, "openai/gpt-oss-120b", "model", "ai-model", ["app.py"])
    res = analyze(Index(project, entire()), d)
    found = by_symbol(res)
    assert res.source == "entire-graph"
    # the tool call only touched app.py; the graph also finds actions.py and the caller
    assert found[("app.py", "summarize_notes")]["depth"] == 0
    assert ("actions.py", "extract_action_items") in found
    assert found[("app.py", "Handler.do_POST")]["depth"] == 1
    assert found[("app.py", "Handler.do_POST")]["via"].startswith("calls ")
    assert set(res.files) == {"app.py", "actions.py"}


def test_service_decision_follows_imports_module_names_and_manifest(project):
    with Graph.for_project(project) as g:
        d = decision(g, "MongoDB", "service", "database", ["requirements.txt"], evidence="pymongo")
    found = by_symbol(analyze(Index(project, entire()), d))
    assert ("app.py", None) in found                       # imports pymongo
    assert ("requirements.txt", None) in found             # lists pymongo
    # module-level `counters = db["counters"]` / `notes = db["notes"]` are followed
    assert found[("app.py", "next_note_id")]["via"].startswith("uses counters")
    assert ("app.py", "Handler.do_GET") in found
    # a prompt string that mentions "notes" is not a use of the `notes` collection
    assert ("app.py", "summarize_notes") not in found
    assert ("actions.py", "extract_action_items") not in found


def test_sdk_calls_from_graph_edges(project):
    with Graph.for_project(project) as g:
        d = decision(g, "OpenAI", "service", "ai-provider", ["requirements.txt"], evidence="openai")
    found = by_symbol(analyze(Index(project, entire()), d))
    assert found[("actions.py", "extract_action_items")]["via"] == "calls openai.OpenAI"
    assert found[("app.py", "summarize_notes")]["via"] == "calls openai.OpenAI"


def test_unknown_literal_falls_back_to_touched_files(project):
    with Graph.for_project(project) as g:
        d = decision(g, "gpt-4o", "model", "ai-model", ["app.py"])
    res = analyze(Index(project, entire()), d)
    assert res.source == "touched-files" and res.files == ["app.py"]
    assert "found no code" in res.note


def test_graph_plugin_missing_falls_back(project):
    with Graph.for_project(project) as g:
        d = decision(g, "openai/gpt-oss-120b", "model", "ai-model", ["app.py"])
    idx = Index(project, entire({("plugin", "list"): "Managed plugin directory: x\n"}))
    res = analyze(idx, d)
    assert not idx.ok and "entire-graph is not installed" in idx.error
    assert res.source == "touched-files" and res.files == ["app.py"]


def test_entire_binary_missing_falls_back(project, monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: None)

    def missing(*a, **k):
        raise FileNotFoundError("entire")
    with Graph.for_project(project) as g:
        d = decision(g, "openai/gpt-oss-120b", "model", "ai-model", ["app.py"])
    res = analyze(Index(project, Entire(exe="entire", runner=missing)), d)
    assert res.source == "touched-files" and "not installed" in res.note


def test_graph_error_falls_back(project):
    responses = dict(demo_graph_responses())
    responses[("graph", "edges")] = 2
    with Graph.for_project(project) as g:
        d = decision(g, "openai/gpt-oss-120b", "model", "ai-model", ["app.py"])
    res = analyze(Index(project, entire(responses)), d)
    assert res.source == "touched-files" and "exited 2" in res.note


def test_not_a_git_repo_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(affected_mod, "repo_root", lambda p: None)
    with Graph.for_project(tmp_path) as g:
        d = decision(g, "openai/gpt-oss-120b", "model", "ai-model", ["app.py"])
    res = analyze(Index(tmp_path, entire()), d)
    assert res.source == "touched-files" and "git repository" in res.note


def test_project_in_a_subdirectory_of_the_repo(tmp_path, monkeypatch):
    sub = tmp_path / "svc"
    sub.mkdir()
    demo_project(sub)
    monkeypatch.setattr(affected_mod, "repo_root", lambda p: tmp_path)
    responses = demo_graph_responses()
    for key in (("graph", "symbols"), ("graph", "edges")):   # paths relative to the repo root
        responses[key] = (responses[key].replace('"file_path":"', '"file_path":"svc/')
                          .replace("local/demo:Python:", "local/demo:Python:svc/")
                          .replace("local/demo:file:", "local/demo:file:svc/"))
    with Graph.for_project(sub) as g:
        d = decision(g, "openai/gpt-oss-120b", "model", "ai-model", ["app.py"])
    res = analyze(Index(sub, entire(responses)), d)
    assert set(res.files) == {"app.py", "actions.py"}


def test_refresh_stores_code_nodes_with_affects_edges(project, monkeypatch):
    monkeypatch.setattr(affected_mod, "Entire", lambda: entire())
    with Graph.for_project(project) as g:
        d = decision(g, "openai/gpt-oss-120b", "model", "ai-model", ["app.py"])
    refresh(project)
    with Graph.for_project(project) as g:
        stored = g.get_node(d["id"])
        labels = {n["label"]: n for n in g.nodes("code")}
        assert {"app.py:summarize_notes", "actions.py:extract_action_items",
                "app.py:Handler.do_POST"} <= set(labels)
        affects = {e["dst"] for e in g.edges() if e["src"] == d["id"] and e["type"] == "affects"}
        assert labels["actions.py:extract_action_items"]["id"] in affects
        assert stored["data"]["affected_source"] == "entire-graph"
        # stored results feed every existing caller (stale judge, pivot impact, callsite)
        assert set(find_affected_code(stored)) == {"app.py", "actions.py"}
        assert any(e["symbol"] == "extract_action_items" for e in affected_entries(stored))
    refresh(project)        # idempotent: no duplicate code nodes
    with Graph.for_project(project) as g:
        assert len(g.nodes("code")) == len(labels)


def test_find_affected_code_with_project_queries_the_graph(project):
    with Graph.for_project(project) as g:
        d = decision(g, "openai/gpt-oss-120b", "model", "ai-model", ["app.py"])
    assert set(find_affected_code(d, project, entire())) == {"app.py", "actions.py"}


def test_record_affected_logs_a_recovery_on_fallback(project):
    from notbydefault.fit.tracing import Tracer
    t = Tracer()
    with Graph.for_project(project) as g:
        d = decision(g, "gpt-4o", "model", "ai-model", ["app.py"])
        record_affected(g, d["id"], analyze(Index(project, entire()), d), t)
    [e] = [e for e in t.journal if e["kind"] == "event"]
    assert e["name"] == "affected_code_fallback" and e["focus_area"] == "recovery"
