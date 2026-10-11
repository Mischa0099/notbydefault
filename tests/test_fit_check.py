import json

import pytest

from conftest import CHOSEN, JUDGE, SMALL_OTHER, SMALL_SAME, FakeTransport, make_client
from notbydefault.cli import main as cli
from notbydefault.fit import labels
from notbydefault.fit.run import CheckOptions, cache_dir, check_model_decision
from notbydefault.graph import Graph

APP = '''import os
from openai import OpenAI

client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"])


def summarize_notes(texts):
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "Summarize the user's notes in a few concise sentences."},
            {"role": "user", "content": "\\n".join(f"- {t}" for t in texts)},
        ],
    )
    return response.choices[0].message.content
'''
LITERAL = "openai/gpt-oss-120b"
LITE_VERIFIED = f"{SMALL_OTHER} + verifier (<=2 retries)"


def make_project(path):
    cli(["init", str(path), "--goal", "Notes API", "--users", "500", "--budget", "25",
         "--team", "2", "--no-input"])
    (path / "app.py").write_text(APP)
    with Graph.for_project(path) as g:
        did = g.add_node("decision", LITERAL, {"kind": "model", "category": "ai-model",
                                               "status": "unreviewed", "file": "app.py",
                                               "files": ["app.py"]})
        for n in g.nodes():
            if n["type"] in ("requirement", "constraint"):
                g.add_edge(did, n["id"], "depends_on")
            if n["type"] == "goal":
                g.add_edge(did, n["id"], "serves")
        code = g.add_node("code", "app.py", {"path": "app.py"})
        g.add_edge(did, code, "affects")
    return path, did


@pytest.fixture
def project(tmp_path):
    return make_project(tmp_path)


def client_for(project_dir, transport=None):
    return make_client(cache_dir=cache_dir(project_dir), transport=transport)


def run(project_dir, did, client=None, **opts):
    out = []
    opts.setdefault("yes", True)
    opts.setdefault("calls_per_user", 30)
    client = client or client_for(project_dir)
    code = check_model_decision(project_dir, did, CheckOptions(workers=2, **opts), client=client,
                                ask=lambda _: "", out=out.append)
    return code, "\n".join(out)


def test_full_check_writes_report_graph_and_files(project):
    p, did = project
    code, report = run(p, did)
    assert code == 0, report
    nbd = p / ".notbydefault"

    assert json.loads((nbd / "callsites" / f"{did}.json").read_text())["task"]
    samples = (nbd / "samples" / f"{did}.jsonl").read_text().splitlines()
    assert len(samples) == 8 and all(json.loads(s)["source"] == "synthetic" for s in samples)
    assert (nbd / "verifiers" / f"{did}.py").exists()
    cfg = (nbd / "candidates.yaml").read_text()
    assert CHOSEN in cfg and SMALL_SAME in cfg and SMALL_OTHER in cfg and f"judge: {JUDGE}" in cfg

    # the small Gemini model rambles, but with the verifier it matches the large model's
    # pass rate and quality for less money: "cheaper model + verifier"
    assert f"Switch to {LITE_VERIFIED}:" in report
    assert "Run on free tier; costs are list-price estimates." in report
    assert "https://groq.test/models (fetched 2026-10-10)" in report
    assert "SYNTHETIC" in report and "pass rate (95% CI)" in report and "fits budget?" in report
    assert "30 calls per user per month (ASSUMPTION)" in report

    bank = [json.loads(l) for l in (nbd / "failures" / f"{did}.jsonl").read_text().splitlines()]
    assert {b["model"] for b in bank} == {SMALL_OTHER, SMALL_SAME}
    assert {"max_words", "llm_rubric"} <= {f for b in bank for f in b["failed"]}

    results = json.loads(next((nbd / "results").glob(f"{did}-*.json")).read_text())
    by = {(r["model"], r["setup"], r["sample"]): r for r in results["rows"]}
    lite_alone, lite_ver = by[(SMALL_OTHER, "alone", 0)], by[(SMALL_OTHER, "verified", 0)]
    assert not lite_alone["passed"] and lite_ver["passed"] and lite_ver["attempts"] == 2
    assert lite_ver["cost"] > lite_alone["cost"]          # retries + verifier calls are paid for
    assert by[(SMALL_SAME, "verified", 0)]["attempts"] == 3   # 1 + 2 retries, still failing

    with Graph.for_project(p) as g:
        d = g.get_node(did)
        assert d["data"]["status"] == "reviewed" and d["data"]["verdict"] == "switch"
        assert d["data"]["recommended"] == LITE_VERIFIED
        kinds = {n["data"].get("kind") for n in g.neighbors(did, ("supported_by",))}
        assert kinds == {"fit_test", "pricing", "verifiers"}
        rejected = {n["label"]: n["data"]["reason"] for n in g.neighbors(did, ("rejects",))}
        assert "pass rate" in rejected[SMALL_SAME]
    [record] = (p / "decisions").glob("0001-ai-model-openai-gpt-oss-120b.md")
    text = record.read_text()
    assert f"Change recommended: {LITE_VERIFIED}" in text and "free tier" in text


def test_second_run_spends_no_requests_and_runs_failures_first(project):
    p, did = project
    run(p, did)
    client = client_for(p)
    code, report = run(p, did, client)
    assert code == 0
    # failure-bank inputs are new prompts for the healthy models, but nothing repeats
    kinds = [k for _, k in client.transport.calls]
    assert not {"extract", "samples", "verifiers"} & set(kinds)
    samples = json.loads(sorted((p / ".notbydefault" / "results").glob(f"{did}-*.json"))[-1]
                         .read_text())["samples"]
    assert samples[0].get("from_failure_bank") is True
    # one record per decision, rewritten in place on every verdict
    assert [r.name for r in (p / "decisions").glob("*.md")] == ["0001-ai-model-openai-gpt-oss-120b.md"]
    record = (p / "decisions" / "0001-ai-model-openai-gpt-oss-120b.md").read_text(encoding="utf-8")
    assert record.count("## Summary (NotByDefault)") == 1 and record.count("fit check verdict") == 2
    third = client_for(p)
    run(p, did, third)
    assert third.transport.calls == []        # every request of an identical rerun is cached


def test_daily_limit_stops_cleanly_and_resumes_without_repeating(project, tmp_path_factory):
    p, did = project
    first = FakeTransport(daily_limit_after=12)
    code, report = run(p, did, client_for(p, first))
    assert code == 4
    assert "daily free-tier limit reached" in report and "Progress is saved" in report
    assert "check" in report and "again after the limit resets" in report
    with Graph.for_project(p) as g:
        assert g.get_node(did)["data"]["status"] == "unreviewed"
    assert not (p / "decisions").exists()
    [jpath] = (p / ".notbydefault" / "journal").glob(f"{did}-*-daily_limit.json")
    journal = json.loads(jpath.read_text())
    stop = next(e for e in journal if e["name"] == "daily_limit_stop")
    assert stop["focus_area"] == "recovery" and "requests per day" in stop["reason"]

    second = FakeTransport()
    code, report = run(p, did, client_for(p, second))
    assert code == 0, report
    # same total as one uninterrupted run: nothing was requested twice
    clean_dir, clean_did = make_project(tmp_path_factory.mktemp("clean"))
    clean = FakeTransport()
    run(clean_dir, clean_did, client_for(clean_dir, clean))
    assert len(first.calls) + len(second.calls) == len(clean.calls)


def test_estimate_over_max_cost_spends_nothing(project):
    p, did = project
    client = client_for(p)
    code, report = run(p, did, client, max_cost=0.000001)
    assert code == 2 and "exceeds --max-cost" in report
    assert client.transport.calls == []


def test_cancel_at_confirmation(project):
    p, did = project
    client = client_for(p)
    out = []
    code = check_model_decision(p, did, CheckOptions(calls_per_user=30), client=client,
                                ask=lambda _: "n", out=out.append)
    assert code == 1 and client.transport.calls == [] and "nothing was spent" in out[-1]


def test_calls_per_user_asked_once_and_stored(project):
    p, did = project
    answers = iter(["45", "y"])
    check_model_decision(p, did, CheckOptions(workers=2), client=client_for(p),
                         ask=lambda _: next(answers), out=lambda _: None)
    with Graph.for_project(p) as g:
        [node] = [n for n in g.nodes("requirement") if n["data"].get("key") == "calls_per_user"]
        assert node["data"]["value"] == 45
    asked = []
    check_model_decision(p, did, CheckOptions(workers=2, yes=True), client=client_for(p),
                         ask=lambda q: asked.append(q) or "", out=lambda _: None)
    assert asked == []


def test_api_error_during_setup_is_reported_not_raised(project):
    from notbydefault.llm import LLMError
    p, did = project

    class Broke(FakeTransport):
        def complete(self, *a, **k):
            raise LLMError("groq:openai/gpt-oss-120b: HTTP 401: invalid api key")

    code, report = run(p, did, client_for(p, Broke()))
    assert code == 3 and "invalid api key" in report


def test_service_decision_is_milestone_3(tmp_path):
    with Graph.for_project(tmp_path) as g:
        did = g.add_node("decision", "MongoDB", {"kind": "service", "category": "database"})
    out = []
    assert check_model_decision(tmp_path, did, CheckOptions(), client=make_client(),
                                out=out.append) == 0
    assert out[-1] == "service fit: not implemented yet (only model decisions can be re-tested)"


def test_label_measures_verifier_agreement(project):
    p, did = project
    run(p, did)
    answers = iter(["p"] * 10)     # I think everything is fine
    out = []
    assert labels.label(p, did, ask=lambda _: next(answers), out=out.append) == 0
    t = labels.agreement(p, did)
    assert t["n"] == 10
    assert t["false_fail"] == 10 - t["agree"] and t["false_fail"] >= 1   # verifier failed some
    assert any("Verifiers agree with you" in line for line in out)
    _, report = run(p, did)
    assert f"agree with your labels on {t['agree']}/10" in report


def journal_of(project_dir, did):
    path = sorted((project_dir / ".notbydefault" / "results").glob(f"{did}-*.json"))[-1]
    return json.loads(path.read_text())["journal"]


def test_plan_is_an_explicit_planning_step_with_reasons(project):
    p, did = project
    run(p, did)
    journal = journal_of(p, did)
    plan = next(e for e in journal if e["kind"] == "step" and e["name"] == "plan")
    assert plan["focus_area"] == "planning" and "Candidates:" in plan["plan"]
    picked = [e for e in journal if e["name"] == "candidate_selected"]
    assert {e["model"] for e in picked} == {CHOSEN, SMALL_SAME, SMALL_OTHER}
    assert all(e["reason"] for e in picked)
    assert any(e["name"] == "judge_selected" and e["model"] == JUDGE for e in journal)
    steps = [e["name"] for e in journal if e["kind"] == "step"]
    assert steps[:6] == ["notbydefault.check", "plan", "preflight", "context", "verifiers",
                         "execute"]
    assert {e["focus_area"] for e in journal} <= {"planning", "tool_use", "state", "context",
                                                  "execution", "recovery", "verification",
                                                  "iteration"}
    retries = [e for e in journal if e["name"] == "retry_with_feedback"]
    assert retries and all(e["focus_area"] == "recovery" and "failed" in e["reason"]
                           for e in retries)


def test_unavailable_candidate_is_swapped_with_reason(project):
    from conftest import SWAP_IN
    p, did = project
    transport = FakeTransport(unavailable={"gemini-3.5-flash-lite"})
    code, report = run(p, did, client_for(p, transport))
    assert code == 0, report
    swap = next(e for e in journal_of(p, did) if e["name"] == "swap_candidate")
    assert swap["old"] == SMALL_OTHER and swap["new"] == SWAP_IN
    assert "unavailable" in swap["reason"] and "limit: 0" in swap["reason"]
    cfg = (p / ".notbydefault" / "candidates.yaml").read_text()
    assert f"model: {SWAP_IN}" in cfg and "swaps:" in cfg        # later runs stay comparable
    assert SWAP_IN in report


def test_unavailable_chosen_model_stops_clearly(project):
    p, did = project
    code, report = run(p, did, client_for(p, FakeTransport(unavailable={"openai/gpt-oss-120b"})))
    assert code == 3 and "chosen model" in report and "unavailable" in report


def test_too_close_to_call_expands_the_sample_with_a_reason(project, monkeypatch):
    from notbydefault.fit import run as run_mod
    from notbydefault.fit.stats import Verdict
    real = run_mod.decide
    calls = []

    def first_too_close(stats, chosen, excluded=None):
        calls.append(1)
        v = real(stats, chosen, excluded)
        if len(calls) == 1:
            return Verdict("insufficient", v.reference, v.recommended, "intervals overlap", {})
        return v
    monkeypatch.setattr(run_mod, "decide", first_too_close)
    p, did = project
    code, report = run(p, did)
    assert code == 0 and len(calls) == 2
    samples = (p / ".notbydefault" / "samples" / f"{did}.jsonl").read_text().splitlines()
    assert len(samples) == 12
    assert json.loads(samples[-1])["expansion_reason"].startswith("verdict too close to call")
    exp = [e for e in journal_of(p, did) if e["name"] == "expand_sample"]
    assert exp[-1]["focus_area"] == "iteration" and "intervals overlap" in exp[-1]["reason"]
    assert "12 synthetic" in report


def test_single_provider_run_is_labelled(tmp_path):
    import copy
    from conftest import CATALOG
    cat = copy.deepcopy(CATALOG)
    for m in cat["providers"]["gemini"]["models"].values():
        m["available"] = False
    cat["providers"]["groq"]["models"]["llama-4-maverick"] = {
        "family": "llama", "size": "large", "strength": 2, "in": 0.2, "out": 0.6}
    p, did = make_project(tmp_path)
    client = make_client(cache_dir=cache_dir(p), catalog=cat)
    code, report = run(p, did, client)
    assert code == 0, report
    assert "[judge: groq:llama-4-maverick, single-provider run]" in report


def test_multi_provider_run_has_no_single_provider_label(project):
    p, did = project
    _, report = run(p, did)
    assert "single-provider run" not in report
