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
    assert retries and all(e["focus_area"] == "iteration" and "failed" in e["reason"]
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


# --- v4: judge crash, latency, wording ----------------------------------------------------

# --- no judge left: degrade instead of fail (Groq-only 3-model setup, as in the throwaway copy)

def groq_only_client(p, unavailable=()):
    """gpt-oss-120b (today's), gpt-oss-20b and qwen: every free model becomes a candidate."""
    import copy
    from conftest import CATALOG
    catalog = copy.deepcopy(CATALOG)
    catalog["providers"] = {"groq": catalog["providers"]["groq"]}
    catalog["providers"]["groq"]["models"].pop("llama-3.1-8b-instant")
    t = FakeTransport(unavailable=unavailable)
    return make_client(cache_dir=cache_dir(p), transport=t, catalog=catalog), t


def run_traced(p, did, client, **opts):
    from notbydefault.fit.tracing import Tracer
    out, tracer = [], Tracer()
    code = check_model_decision(p, did, CheckOptions(yes=True, calls_per_user=30, workers=2,
                                                     **opts),
                                client=client, tracer=tracer, ask=lambda _: "", out=out.append)
    summary = next(e for e in reversed(tracer.journal) if e["name"] == "root_summary")
    return code, "\n".join(out), tracer, summary


def candidates_entry(p, did):
    import yaml
    doc = yaml.safe_load((p / ".notbydefault" / "candidates.yaml").read_text(encoding="utf-8"))
    return doc["decisions"][str(did)]


def write_reference_samples(p, did, n=6):
    """Cases with reference answers, so a run without a judge can still score correctness."""
    rows = [{"id": f"r{i}", "hard": i % 2 == 0,
             "user_message": f"- buy milk {i}\n- call Sam about launch {i}",
             "reference": {"action_items": [{"task": f"buy milk {i}", "owner": None,
                                             "due_date": None, "priority": "medium"}]}}
            for i in range(n)]
    path = p / ".notbydefault" / "samples" / f"{did}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_all_candidates_frees_one_to_judge_and_records_a_swap(project):
    p, did = project
    client, t = groq_only_client(p)
    code, report, tracer, summary = run_traced(p, did, client)
    assert code == 0, report
    entry = candidates_entry(p, did)
    # qwen is not trusted as a sole judge (JUDGE_FALLBACK_SKIP), so gpt-oss-20b is freed
    assert entry["judge"] == "groq:openai/gpt-oss-20b"
    assert [c["model"] for c in entry["candidates"]] == ["groq:openai/gpt-oss-120b",
                                                          "groq:qwen/qwen3.8-27b"]
    assert entry["swaps"][0]["role"] == "judge_from_candidate"
    swaps = [e for e in tracer.journal if e["name"] == "swap_candidate"]
    assert swaps and swaps[0]["role"] == "judge_from_candidate"
    assert summary["nbd.swap_count"] >= 1 and summary["nbd.judge_mode"] == "judge"
    assert "freed from the candidates to judge" in report
    assert any(k == "judge" for _, k in t.calls)            # the freed model really judges


def test_no_judge_at_all_falls_back_to_reference_only_scoring(project):
    p, did = project
    write_reference_samples(p, did)
    # the freed judge (20b) turns out unavailable; qwen can't judge alone; only 120b is left
    client, t = groq_only_client(p, unavailable=("openai/gpt-oss-20b",))
    code, report, tracer, summary = run_traced(p, did, client)
    assert code == 0, report
    assert summary["nbd.judge_mode"] == "reference_only"
    assert "REFERENCE-ONLY" in report and "no quality scores" in report
    assert "judge groq:openai/gpt-oss-20b unavailable" in report
    assert not any(k == "judge" for _, k in t.calls)        # no judge calls at all
    assert any(e["name"] == "reference_only" for e in tracer.journal)
    assert candidates_entry(p, did)["judge"] is None


def test_nothing_can_run_exits_5_with_one_line_listing_what_was_tried(project):
    p, did = project      # no reference answers: synthetic samples would need a judge
    client, t = groq_only_client(p, unavailable=("openai/gpt-oss-20b",))
    code, report, tracer, summary = run_traced(p, did, client)
    assert code == 5
    line = next(l for l in report.splitlines() if l.startswith("Can't run the fit check"))
    assert "no judge and no reference answers" in line and "tried:" in line
    assert "freed from the candidates to judge" in line
    assert "judge groq:openai/gpt-oss-20b unavailable" in line
    assert "no non-current candidate can be freed" in line
    assert summary["nbd.stopped_early"] is True and summary["nbd.judge_mode"] == "none"
    assert {k for _, k in t.calls} <= {"probe"}             # stopped before any real work


def test_a_second_judge_is_promoted_before_freeing_a_candidate(tmp_path):
    from notbydefault.fit.run import replace_judge
    from notbydefault.fit.tracing import Tracer
    cfg = {"chosen": "groq:a", "candidates": ["groq:a", "groq:b"], "judge": "gemini:j",
           "judges": ["gemini:j", "groq:k"], "why": {}, "judge_tried": []}
    swaps = []
    import notbydefault.fit.candidates as cm
    orig = cm.swap
    cm.swap = lambda *a, **k: swaps.append(a)
    try:
        queue = [("groq:k", "judge2")]
        assert replace_judge(cfg, "gemini:j", "quota", {"gemini:j"}, queue, None, tmp_path,
                             {"id": 1}, Tracer())
    finally:
        cm.swap = orig
    assert cfg["judge"] == "groq:k" and cfg["judges"] == ["groq:k"]
    assert cfg["candidates"] == ["groq:a", "groq:b"]         # nothing taken out of the comparison
    assert queue == [("groq:k", "judge")] and swaps[0][2] == "judge"


def latest_verdict(p):
    path = sorted((p / ".notbydefault" / "results").glob("*.json"))[-1]
    return json.loads(path.read_text(encoding="utf-8"))["verdict"]


def test_latency_requirement_overrides_the_verdict(project):
    p, did = project
    code, before = run(p, did)
    assert code == 0 and latest_verdict(p)["recommended"][1] == "verified"
    # the fake answers in 0.01 s per call: a verified setup (model + verifier) takes longer
    code, after = run(p, did, max_latency_s=0.015)
    assert code == 0
    assert latest_verdict(p)["recommended"][1] == "alone"
    assert "too slow" in after and "Latency requirement: at most 0.015s" in after


def test_latency_requirement_from_the_graph(project):
    from notbydefault.fit.run import latency_requirement
    p, did = project
    with Graph.for_project(p) as g:
        assert latency_requirement(g) is None
        g.add_node("requirement", "Responses within 2 seconds", {"key": "max_latency_s", "value": "2"})
        g.add_node("requirement", "p95 under 1.5s", {"constraints": {"max_latency_s": 1.5}})
        assert latency_requirement(g) == 1.5


def test_latency_tradeoff_is_mentioned_when_much_slower():
    from types import SimpleNamespace as NS
    from notbydefault.fit.report import latency_tradeoff
    from notbydefault.fit.stats import Verdict
    today = NS(key=("a", "alone"), name="a", setup="alone", avg_latency_s=0.6)
    slow = NS(key=("b", "verified"), name="b + verifier", setup="verified", avg_latency_s=1.5)
    near = NS(key=("c", "alone"), name="c", setup="alone", avg_latency_s=1.1)
    stats = {s.key: s for s in (today, slow, near)}
    line = latency_tradeoff(stats, Verdict("switch", today.key, slow.key, "", {}))
    assert line.startswith("Trade-off: b + verifier is 2.5x slower") and "retries" in line
    assert latency_tradeoff(stats, Verdict("switch", today.key, near.key, "", {})) == ""
    assert latency_tradeoff(stats, Verdict("keep", today.key, today.key, "", {})) == ""


def test_summary_reads_as_english():
    from notbydefault.fit.report import task_phrase
    assert task_phrase("Extracts structured action items from meeting notes.") == \
        "extract structured action items from meeting notes"
    assert task_phrase("Summarizes the notes") == "summarize the notes"
    assert task_phrase("Classifies tickets") == "classify tickets"
    assert task_phrase("Matches invoices") == "match invoices"
    assert task_phrase("Generate a summary.") == "generate a summary"
    assert task_phrase("") == ""


# --- the judge in the report and the plan header ---------------------------------------------

def test_freed_same_family_judge_is_flagged_as_possible_self_preference(project):
    p, did = project
    client, _ = groq_only_client(p)
    code, report, _, _ = run_traced(p, did, client)
    assert code == 0, report
    assert ("Possible self-preference: the judge groq:openai/gpt-oss-20b is from the same model "
            "family (gpt-oss) as today's model groq:openai/gpt-oss-120b, because it was freed "
            "from the candidates to judge") in report


def test_other_family_judge_gets_no_self_preference_caveat():
    from types import SimpleNamespace as NS
    from notbydefault.fit.run import same_family_judges
    pricing = NS(models=[{"id": "groq:a", "family": "gpt-oss"}, {"id": "gemini:j", "family": "gemini"},
                         {"id": "groq:b", "family": "gpt-oss"}])
    cfg = {"chosen": "groq:a", "judges": ["gemini:j"], "why": {}}
    assert same_family_judges(cfg, pricing) == []
    cfg = {"chosen": "groq:a", "judges": ["groq:b"], "why": {"groq:b": "next strongest free judge"}}
    [line] = same_family_judges(cfg, pricing)
    assert "same model family (gpt-oss)" in line and "freed" not in line


def test_header_prints_judge_changes_made_by_preflight(project):
    p, did = project
    t = FakeTransport(unavailable=("gemini-3.8-flash",))       # the configured judge
    code, report = run(p, did, client=client_for(p, t))
    assert code == 0, report
    line = next(l for l in report.splitlines() if l.strip().startswith("judge changed:"))
    assert "gemini:gemini-3.8-flash -> " in line and "(unavailable: free tier limit: 0)" in line
    assert report.index("judges:") < report.index("judge changed:")


def test_header_prints_the_drop_to_no_judge(project):
    p, did = project
    write_reference_samples(p, did)
    client, _ = groq_only_client(p, unavailable=("openai/gpt-oss-20b",))
    code, report, _, _ = run_traced(p, did, client)
    assert code == 0
    assert "(freed from the candidates: groq:openai/gpt-oss-20b)" in report
    assert "judge changed: groq:openai/gpt-oss-20b -> none (unavailable: free tier limit: 0" in report
