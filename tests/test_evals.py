"""Milestone 5: `notbydefault eval` scoring, offline (fake provider, no spend, no Gemini)."""
import json

import pytest

from conftest import make_client
from notbydefault import evals
from notbydefault.fit.tracing import Tracer


@pytest.fixture
def fake(monkeypatch, tmp_path):
    client = evals.CountingClient(make_client(cache_dir=tmp_path / "cache"))
    monkeypatch.setattr(evals, "make_client", lambda project: client)
    from notbydefault import pivot
    monkeypatch.setattr(pivot, "_openrouter_models", lambda project_dir: [])
    return client


def scored(name):
    res = evals.RUNNERS[name](Tracer())
    evals.score(res, name)
    return res


def test_detection_section_scores_every_case():
    res = scored("detection")
    m = res["metrics"]
    h = res["heldout_metrics"]
    assert m["cases"] == len(evals.load_cases("detection", "original"))
    assert h["cases"] == len(evals.load_cases("detection", "heldout"))
    assert m["passed"] + h["passed"] == sum(r["ok"] for r in res["cases"])
    assert len(res["failures"]) == m["cases"] + h["cases"] - m["passed"] - h["passed"]
    row = next(r for r in res["cases"] if r["case"] == "det-pnpm-orm-and-db")
    assert row["ok"] and sorted(row["flagged"]) == ["Drizzle", "PostgreSQL"]
    assert next(r for r in res["cases"] if r["case"] == "neg-read-file")["asked"] is False


def test_pivot_section_counts_llm_calls_per_ordinary_prompt(fake):
    res = scored("pivot")
    m = res["metrics"]
    assert m["llm_calls"] + res["heldout_metrics"]["llm_calls"] == fake.calls
    ordinary = [r for r in res["cases"] if not r["expected_pivot"] and r["split"] == "original"]
    assert m["llm_calls_per_ordinary_prompt"] == round(
        sum(r["llm_calls"] for r in ordinary) / len(ordinary), 3)
    for r in res["cases"]:
        assert r["llm_calls"] == (0 if r["stage"] == "prefilter" else 1)


def test_requirements_section_scores_each_decision(fake):
    res = scored("requirements")
    eu = next(r for r in res["cases"] if r["case"] == "req-eu-residency")
    # documented facts decide these, whatever the (fake) judge says
    assert eu["decisions"]["gpt_oss_120b"]["got"] == "stale"
    assert eu["decisions"]["mongodb"]["got"] == "needs_review"
    original = [r for r in res["cases"] if r["split"] == "original"]
    total = sum(len(r["decisions"]) for r in original)
    right = sum(d["ok"] for r in original for d in r["decisions"].values())
    assert res["metrics"]["decisions_right"] == f"{right}/{total}"


def test_gemini_calls_are_refused(fake):
    with pytest.raises(evals.GeminiRefused):
        fake.chat("gemini:gemini-3.8-flash", [{"role": "user", "content": "hi"}])
    assert fake.calls == 0


def _project_with_decision(tmp_path):
    from notbydefault.graph import Graph
    with Graph.for_project(tmp_path) as g:
        return g.add_node("decision", "openai/gpt-oss-120b", {"kind": "model", "status": "unreviewed"})


def test_fit_is_pending_when_the_run_stopped_at_the_daily_limit(tmp_path):
    did = _project_with_decision(tmp_path)
    j = tmp_path / ".notbydefault" / "journal" / f"{did}-20261010T180513Z-daily_limit.json"
    j.parent.mkdir(parents=True)
    j.write_text(json.dumps([
        {"kind": "step", "name": "notbydefault.check", "trace_id": "abc"},
        {"kind": "step", "name": "plan", "judge": "gemini:gemini-3.8-flash"},
        {"kind": "event", "name": "daily_limit_stop", "model": "gemini:gemini-3.8-flash",
         "reason": "429"},
        {"kind": "event", "name": "verdict", "verdict": "switch", "reason": "Cheapest: x"}]))
    st = evals.fit_status(tmp_path, did)
    assert st["status"] == "pending" and st["provisional_verdict"] == "switch"
    assert st["check_trace_id"] == "abc"


def test_fit_is_final_with_a_finished_results_file(tmp_path):
    did = _project_with_decision(tmp_path)
    r = tmp_path / ".notbydefault" / "results" / f"{did}-20261011T080000Z.json"
    r.parent.mkdir(parents=True)
    r.write_text(json.dumps({"verdict": {"kind": "keep", "summary": "Keep it."}, "stopped": None,
                             "config": {"judge": "gemini:gemini-3.8-flash"}, "trace_id": "t1"}))
    st = evals.fit_status(tmp_path, did)
    assert st["status"] == "final" and st["verdict"] == "keep" and st["check_trace_id"] == "t1"


def test_heldout_cases_are_scored_separately():
    res = scored("detection")
    held = [r["case"] for r in res["cases"] if r["split"] == "heldout"]
    assert held == [c["id"] for c in evals.load("heldout/detection")["cases"]]
    assert res["heldout_metrics"]["cases"] == len(held)
    assert res["metrics"]["cases"] == len(res["cases"]) - len(held)


def test_requirements_heldout_cases_use_their_own_project(fake):
    res = scored("requirements")
    eu = next(r for r in res["cases"] if r["case"] == "hd-req-eu")
    assert eu["split"] == "heldout"
    assert set(eu["decisions"]) == {"supabase", "stripe", "anthropic_sdk", "gpt_oss_20b",
                                    "claude_sonnet", "prisma"}


def test_case_sets_can_be_run_alone():
    assert {c["split"] for c in evals.load_cases("pivot", "heldout")} == {"heldout"}
    assert {c["split"] for c in evals.load_cases("pivot", "original")} == {"original"}
    assert evals.load_cases("fit", "heldout") == []


def test_run_prints_a_progress_line_per_case_and_records_telemetry(tmp_path):
    lines = []
    out = tmp_path / "r.json"
    evals.run("t", ["detection"], out, out=lines.append, which="heldout")
    n = len(evals.load_cases("detection", "heldout"))
    progress = [l for l in lines if l.strip().startswith("[")]
    assert len(progress) == n and progress[0].strip().startswith(f"[1/{n}] ")
    assert all("[held-out]" in l for l in progress)
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["telemetry"]["exporter"] == "off"    # NOTBYDEFAULT_TELEMETRY=0 in tests
