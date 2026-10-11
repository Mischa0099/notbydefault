"""Milestone 3: Pivot Check (UserPromptSubmit) and stale decisions (`require`)."""
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import CATALOG, CHOSEN, SMALL_OTHER, FakeTransport, make_client
from notbydefault import PACKAGE_DIR
from notbydefault.cli import main as cli
from notbydefault.fit.run import CheckOptions, cache_dir, check_model_decision
from notbydefault.graph import Graph
from notbydefault.hooks.user_prompt import handle
from notbydefault.pivot import detect_choice, prefilter, stack_terms
from notbydefault.stale import require

ORDINARY = [
    "add a unit test for next_note_id",
    "fix the off-by-one in the pagination loop",
    "rename summarize_notes to summarize",
    "why does the POST /notes handler return 404?",
    "actually the test still fails, can you look again",
    "replace the hardcoded port 8000 with an environment variable",
    "drop the unused json import",
    "use f-strings instead of format() in app.py",
    "add type hints to actions.py",
    "write a README section about running the app",
    "make the error message friendlier",
    "can you explain how the date resolver works",
    "add logging when a note is created",
    "refactor the handler into smaller functions",
    "what if we add a /health endpoint",
    "sort notes by id descending instead",
    "move the constants to the top of the file",
    "add input validation for empty notes",
    "let's add a test for the action items endpoint",
    "remove the trailing whitespace",
]
PIVOTS = [
    "actually, let's switch to gpt-4o for action items, it's the best",
    "what if we use Postgres instead of MongoDB",
    "drop Groq and move to OpenAI for the summaries",
    "let's replace the model with claude",
]
LABELS = ["openai/gpt-oss-120b", "MongoDB", "OpenAI"]
OPENROUTER = [{"id": "openai/gpt-4o", "pricing": {"prompt": "0.0000025", "completion": "0.00001"}},
              {"id": "openai/gpt-4o-mini", "pricing": {"prompt": "0.00000015", "completion": "0.0000006"}}]


@pytest.fixture(autouse=True)
def no_network_prices(monkeypatch):
    """Price lookups for models outside the catalog use a canned OpenRouter list."""
    from notbydefault import pivot
    monkeypatch.setattr(pivot, "_openrouter_models", lambda project_dir: OPENROUTER)


# --- pre-filter ---------------------------------------------------------------------

def test_prefilter_lets_ordinary_prompts_through_without_llm():
    terms = stack_terms(LABELS)
    passed = [p for p in ORDINARY if prefilter(p, terms)]
    assert passed == []          # 0 of 20 ordinary prompts would trigger an LLM call


@pytest.mark.parametrize("prompt", PIVOTS)
def test_prefilter_catches_pivots(prompt):
    assert prefilter(prompt, stack_terms(LABELS))


def test_prefilter_is_fast():
    terms = stack_terms(LABELS)
    prefilter("warm up", terms)
    start = time.perf_counter()
    for p in ORDINARY + PIVOTS:
        prefilter(p, terms)
    per_prompt_ms = (time.perf_counter() - start) * 1000 / len(ORDINARY + PIVOTS)
    assert per_prompt_ms < 50


@pytest.mark.parametrize("text,choice", [
    ("switch anyway", "accepted"), ("ok, let's adjust the plan", "adjusting"),
    ("keep the current plan", "kept"), ("keep plan", "kept"), ("sounds good", None)])
def test_detect_choice(text, choice):
    assert detect_choice(text) == choice


# --- project fixture with a reviewed decision and evidence ----------------------------

@pytest.fixture
def project(tmp_path):
    cli(["init", str(tmp_path), "--goal", "Notes API", "--users", "20000", "--budget", "375",
         "--no-input"])
    (tmp_path / "actions.py").write_text('MODEL = "openai/gpt-oss-120b"\n')
    with Graph.for_project(tmp_path) as g:
        did = g.add_node("decision", "openai/gpt-oss-120b", {
            "kind": "model", "category": "ai-model", "status": "reviewed", "verdict": "switch",
            "verdict_summary": "Cheapest setup that meets the bar: gpt-oss-20b + verifier.",
            "file": "actions.py", "files": ["actions.py"]})
        db = g.add_node("decision", "MongoDB", {"kind": "service", "category": "database",
                                                "status": "unreviewed", "file": "requirements.txt"})
        g.add_node("requirement", "Calls per user per month: 60 (demo scenario assumption)",
                   {"source": "check", "key": "calls_per_user", "value": 60})
        ev = g.add_node("evidence", "Fit test results 2026-10-11 (12 inputs)",
                        {"kind": "fit_test", "path": ".notbydefault/results/21-x.json"})
        g.add_edge(did, ev, "supported_by")
        for n in g.nodes():
            if n["type"] in ("requirement", "constraint"):
                g.add_edge(did, n["id"], "depends_on")
    return tmp_path, did, db


def hook(project_dir, prompt, transport=None, session="s1"):
    t = transport or FakeTransport()
    client = make_client(cache_dir=cache_dir(project_dir), transport=t)
    out, info = handle({"prompt": prompt, "session_id": session}, project_dir, client=client)
    return out, info, t


# --- pivot check --------------------------------------------------------------------

def test_ordinary_prompt_adds_nothing_and_calls_no_llm(project):
    p, _, _ = project
    for prompt in ORDINARY:
        out, info, t = hook(p, prompt)
        assert out is None and info["stage"] == "prefilter" and t.calls == []


def test_pivot_is_confirmed_and_explained_from_the_graph(project):
    p, did, _ = project
    out, info, t = hook(p, "actually, let's switch to gpt-4o for action items, it's the best")
    assert [k for _, k in t.calls] == ["pivot"]                  # one fast model call
    ctx = out["hookSpecificOutput"]
    assert ctx["hookEventName"] == "UserPromptSubmit"
    text = ctx["additionalContext"]
    assert "Monthly budget: 375" in text                                         # (a)
    # deterministic: 20,000 users x 60 calls x (1000 in x $2.50/M + 300 out x $10/M) = $6,600
    assert "about $6,600/month for openai/gpt-4o at list price, 18x the budget" in text
    assert "Cost check: openai/gpt-4o" in text and "OVER BUDGET" in text
    assert "assumed 1000 input + 300 output tokens per call" in text
    assert f"#{did} openai/gpt-oss-120b" in text and "Cheapest setup" in text      # (b) reason
    assert "Fit test results 2026-10-11" in text and "results/21-x.json" in text   # evidence
    assert "Code: actions.py" in text                                              # (c)
    assert "at most 6 lines" in text and "switch anyway" in text and "keep the current plan" in text
    assert len(text) < 10_000
    with Graph.for_project(p) as g:
        [pivot] = g.nodes("pivot")
        assert pivot["data"]["status"] == "proposed" and pivot["session_id"] == "s1"
        assert {"src": pivot["id"], "dst": did, "type": "affects"} in g.edges()


def test_not_a_pivot_after_llm_adds_nothing(project):
    p, _, _ = project
    out, info, t = hook(p, "let's use a dataclass instead of a dict for the model config")
    assert [k for _, k in t.calls] == ["pivot"] and out is None and info["stage"] == "not_pivot"
    with Graph.for_project(p) as g:
        assert g.nodes("pivot") == []


def test_users_answer_resolves_the_pivot(project):
    p, did, _ = project
    hook(p, "actually, let's switch to gpt-4o for action items")
    out, info, t = hook(p, "switch anyway")
    assert t.calls == [] and info["choice"] == "accepted"
    assert "pivot #" in out["hookSpecificOutput"]["additionalContext"]
    with Graph.for_project(p) as g:
        assert g.nodes("pivot")[0]["data"]["status"] == "accepted"
        d = g.get_node(did)["data"]
        assert d["status"] == "stale" and "gpt-4o" in d["stale_reason"]


def test_answer_from_another_session_is_ignored(project):
    p, _, _ = project
    hook(p, "actually, let's switch to gpt-4o for action items", session="a")
    out, info, _ = hook(p, "keep the current plan", session="b")
    assert out is None
    with Graph.for_project(p) as g:
        assert g.nodes("pivot")[0]["data"]["status"] == "proposed"


def test_no_graph_means_no_work(tmp_path):
    out, info, t = hook(tmp_path, "actually, let's switch to gpt-4o")
    assert out is None and info["stage"] == "no_graph" and t.calls == []
    assert not (tmp_path / ".notbydefault").exists()


def test_llm_failure_never_blocks(project):
    from notbydefault.llm import LLMError

    class Down(FakeTransport):
        def complete(self, *a, **k):
            raise LLMError("groq down")
    with pytest.raises(LLMError):           # handle() raises; main() logs and adds nothing
        hook(project[0], "actually, let's switch to gpt-4o", transport=Down())


def run_hook_process(project_dir, raw):
    env = dict(os.environ, NOTBYDEFAULT_TELEMETRY="0", GROQ_API_KEY="", GEMINI_API_KEY="")
    return subprocess.run([sys.executable, str(PACKAGE_DIR / "hooks" / "user_prompt.py"),
                           "--project", str(project_dir)], input=raw.encode(),
                          capture_output=True, env=env, timeout=60)


def test_hook_process_never_fails(project):
    p, _, _ = project
    for raw in ["not json", "", "{}", json.dumps({"prompt": "add a test"}),
                json.dumps({"prompt": "actually, let's switch to gpt-4o"})]:   # no key: errors
        proc = run_hook_process(p, raw)
        assert proc.returncode == 0 and proc.stdout == b""
    assert "pivot-check error" in (p / ".notbydefault" / "hook.log").read_text()


def test_install_adds_user_prompt_hook_without_touching_others(tmp_path):
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir()
    other = {"type": "command", "command": "entire hooks claude-code user-prompt-submit"}
    settings.write_text(json.dumps({"hooks": {"UserPromptSubmit": [{"matcher": "", "hooks": [other]}]}}))
    assert cli(["install", str(tmp_path)]) == 0 and cli(["install", str(tmp_path)]) == 0
    ups = json.loads(settings.read_text())["hooks"]["UserPromptSubmit"]
    assert ups[0] == {"matcher": "", "hooks": [other]}
    assert len(ups) == 2 and "matcher" not in ups[1]
    assert ups[1]["hooks"][0]["args"][0].endswith("user_prompt.py")


# --- stale decisions ----------------------------------------------------------------

def run_require(project_dir, text, transport=None):
    out = []
    t = transport or FakeTransport()
    client = make_client(cache_dir=cache_dir(project_dir), transport=t)
    code = require(project_dir, text, client=client, out=out.append)
    return code, "\n".join(out), t


def test_require_marks_affected_decisions_stale(project):
    p, did, db = project
    code, report, t = run_require(p, "We signed an EU customer: EU data residency required")
    assert code == 0
    kinds = [k for _, k in t.calls]
    assert kinds.count("extract") == 1 and kinds.count("stale_judge") == 2
    with Graph.for_project(p) as g:
        [req] = [n for n in g.nodes("requirement") if n["data"].get("source") == "require"]
        assert req["data"]["constraints"] == {"data_residency": "EU"}
        assert req["data"]["constraints_source"] == "model"
        model = g.get_node(did)["data"]
        assert model["status"] == "stale" and "documents data residency" in model["stale_reason"]
        assert model["stale_basis"] == "provider documentation"
        # MongoDB can be self-hosted in the EU: a review, not a provable conflict
        mongo = g.get_node(db)["data"]
        assert mongo["status"] == "needs_review" and "deployed" in mongo["review_reason"]
        assert mongo["review_basis"] == "depends on deployment"
        assert {"src": did, "dst": req["id"], "type": "depends_on"} in g.edges()
        assert {"src": db, "dst": req["id"], "type": "depends_on"} in g.edges()
    assert "1 of 2 decisions provably conflict" in report
    assert "1 of 2 decisions NEED REVIEW" in report
    assert report.index("STALE") < report.index("#" + str(db) + " MongoDB")
    assert "code:  actions.py" in report and "code:  requirements.txt" in report


def test_failed_extraction_still_reads_the_region_the_text_names(project):
    from notbydefault.llm import LLMError

    class Down(FakeTransport):
        def complete(self, *a, **k):
            raise LLMError("no key")
    p, did, _db = project
    code, report, _ = run_require(p, "data must stay in the EU", Down())
    assert code == 0
    with Graph.for_project(p) as g:
        [req] = [n for n in g.nodes("requirement") if n["data"].get("source") == "require"]
        assert req["data"]["constraints"] == {"data_residency": "EU"}
        assert req["data"]["constraints_source"] == "regex-fallback"
        assert g.get_node(did)["data"]["status"] == "stale"     # Groq documents US only


def test_model_judgement_alone_never_marks_stale(project):
    """No documented fact for this service: the judge's 'affected' only asks for a review."""
    p, did, db = project
    with Graph.for_project(p) as g:
        stripe = g.add_node("decision", "Stripe", {"kind": "service", "category": "database",
                                                   "status": "unreviewed", "file": "pay.py"})
    code, report, _ = run_require(p, "We signed an EU customer: EU data residency required")
    with Graph.for_project(p) as g:
        d = g.get_node(stripe)["data"]
        assert d["status"] == "needs_review" and d["review_basis"].startswith("model judgement")
        assert "stale_reason" not in d


def test_no_documented_guarantee_needs_review_but_another_region_is_stale(project):
    """Gemini documents no residency guarantee: it may or may not comply -> needs review.
    Groq documents US only: EU provably breaks -> stale."""
    p, did, db = project
    with Graph.for_project(p) as g:
        lite = g.add_node("decision", "gemini-3.5-flash-lite", {
            "kind": "model", "category": "ai-model", "status": "reviewed", "file": "s.py"})
    run_require(p, "We signed an EU customer: EU data residency required")
    with Graph.for_project(p) as g:
        gem, groq = g.get_node(lite)["data"], g.get_node(did)["data"]
    assert gem["status"] == "needs_review"
    assert gem["review_basis"] == "no documented residency guarantee"
    assert "documents no data-residency guarantee" in gem["review_reason"]
    assert gem["review_source"] == ""          # the fake catalog has no residency source for gemini
    assert groq["status"] == "stale" and groq["stale_basis"] == "provider documentation"
    assert "documents data residency: US" in groq["stale_reason"]


def test_a_resolved_decision_is_judged_on_the_model_it_now_uses(project):
    p, did, db = project
    with Graph.for_project(p) as g:     # resolve --apply switched it from groq to gemini
        g.update_data(did, resolution={
            "at": "2026-10-11T10:00:00+00:00", "branch": "nbd/x", "from": "groq:openai/gpt-oss-120b",
            "to": "gemini:gemini-3.5-flash-lite", "files": ["app.py"], "sites": ["app.py"],
            "left_alone": [], "tests": "passed", "verifiers": "8/8 production", "trace_id": None})
    run_require(p, "We signed an EU customer: EU data residency required")
    with Graph.for_project(p) as g:
        d = g.get_node(did)["data"]
    assert d["status"] == "needs_review"
    assert d["review_reason"].startswith("gemini:gemini-3.5-flash-lite runs on gemini, which "
                                         "documents no data-residency guarantee")


def test_service_without_a_documented_region_needs_review():
    from notbydefault.stale import service_residency
    facts = {"vague": {"name": "Vague", "residency": [], "source": "https://vague"}}
    svc = {"label": "Vague", "data": {"kind": "service"}}
    r = service_residency(svc, {"data_residency": "EU"}, {}, facts)
    assert r["status"] == "needs_review" and "no data-residency guarantee" in r["reason"]


def test_service_with_documented_region_is_stale(project):
    p, did, db = project
    with Graph.for_project(p) as g:
        groq = g.add_node("decision", "Groq", {"kind": "service", "category": "ai-provider",
                                               "status": "unreviewed", "file": "requirements.txt"})
    run_require(p, "We signed an EU customer: EU data residency required")
    with Graph.for_project(p) as g:
        d = g.get_node(groq)["data"]
        assert d["status"] == "stale" and d["stale_basis"] == "service documentation"
        assert d["stale_source"] == "https://groq.test/your-data"


def test_classify_rules():
    from notbydefault.stale import classify, service_residency
    facts = {"mongodb": {"name": "MongoDB", "residency": "deployment", "how": "self-hosted",
                         "source": "https://x"},
             "acme": {"name": "Acme", "residency": ["US"], "source": "https://acme"}}
    eu = {"data_residency": "EU"}
    svc = lambda label: {"label": label, "data": {"kind": "service"}}
    assert service_residency(svc("MongoDB"), eu, {}, facts)["status"] == "needs_review"
    assert service_residency(svc("Acme"), eu, {}, facts)["status"] == "stale"
    assert service_residency(svc("Acme"), {"data_residency": "US"}, {}, facts) is None
    assert service_residency(svc("MongoDB"), {}, {}, facts) is None
    c = classify(svc("Other"), eu, {"affected": True, "reason": "r"}, None, {}, facts)
    assert c["status"] == "needs_review"
    c = classify(svc("Other"), eu, {"affected": False, "reason": "r"}, None, {}, facts)
    assert c["status"] == "valid"
    c = classify(svc("Other"), eu, {"affected": None, "reason": "err"}, None, {}, facts)
    assert c["status"] == "unclear"


def test_require_without_matching_decisions(project):
    p, did, db = project
    code, report, _ = run_require(p, "All API responses must include a request id")
    with Graph.for_project(p) as g:
        assert g.get_node(did)["data"]["status"] == "reviewed"
    assert "No decisions are affected" in report and "Still valid:" in report


# --- check on a stale decision -----------------------------------------------------------

APP = '''def summarize(client, msg):
    return client.chat.completions.create(model="openai/gpt-oss-120b", messages=[
        {"role": "system", "content": "Summarize the user's notes in a few concise sentences."},
        {"role": "user", "content": msg}])
'''


def stale_project(tmp_path):
    cli(["init", str(tmp_path), "--goal", "Notes API", "--users", "500", "--budget", "375",
         "--no-input"])
    (tmp_path / "app.py").write_text(APP)
    with Graph.for_project(tmp_path) as g:
        did = g.add_node("decision", "openai/gpt-oss-120b", {
            "kind": "model", "category": "ai-model", "status": "unreviewed",
            "file": "app.py", "files": ["app.py"]})
    return tmp_path, did


def test_check_on_stale_decision_blocks_when_no_provider_complies(tmp_path):
    p, did = stale_project(tmp_path)
    run_require(p, "EU data residency required")
    t = FakeTransport()
    out = []
    code = check_model_decision(p, did, CheckOptions(yes=True, calls_per_user=30),
                                client=make_client(cache_dir=cache_dir(p), transport=t),
                                ask=lambda _: "", out=out.append)
    text = "\n".join(out)
    assert code == 0 and t.calls == []                       # no requests at all
    assert "is STALE" in text and "no model in the catalog meets" in text
    assert "No requests were made" in text
    with Graph.for_project(p) as g:
        d = g.get_node(did)
        assert d["data"]["status"] == "stale"
        assert any(e["data"].get("kind") == "requirement_check"
                   for e in g.neighbors(did, ("supported_by",)))


def test_check_on_stale_decision_filters_candidates_and_excludes_the_violator(tmp_path):
    from conftest import FakeTransport as FT
    cat = copy.deepcopy(CATALOG)
    cat["providers"]["gemini"]["data_residency"] = ["EU"]          # pretend Gemini were EU-only
    p, did = stale_project(tmp_path)
    t = FT()
    client = make_client(cache_dir=cache_dir(p), transport=t, catalog=cat)
    out = []
    require(p, "EU data residency required", client=client, out=out.append)
    out = []
    code = check_model_decision(p, did, CheckOptions(yes=True, calls_per_user=30, workers=2),
                                client=make_client(cache_dir=cache_dir(p), transport=t, catalog=cat),
                                ask=lambda _: "", out=out.append)
    report = "\n".join(out)
    assert code == 0, report
    results = json.loads(next((p / ".notbydefault" / "results").glob("*.json")).read_text())
    cands = results["config"]["candidates"]
    assert cands[0] == CHOSEN and all(c.startswith("gemini:") for c in cands[1:])
    assert not results["verdict"]["recommended"][0].startswith("groq:")
    assert any("violates a requirement" in why for why in results["verdict"]["rejected"].values())
    yaml_text = (p / ".notbydefault" / "candidates.yaml").read_text()
    assert "refreshed_because" in yaml_text
    with Graph.for_project(p) as g:
        d = g.get_node(did)["data"]
        assert d["status"] == "reviewed" and d["stale_reason"] is None
        assert "EU" in d["previous_stale_reason"] or "residency" in d["previous_stale_reason"]


def test_budget_conflict_does_not_depend_on_the_model(project):
    p, did, _ = project

    class NoConflicts(FakeTransport):
        def respond(self, model, messages):
            kind, text = super().respond(model, messages)
            if kind == "pivot":
                data = json.loads(text)
                data["conflicts"] = []
                text = json.dumps(data)
            return kind, text
    out, _, _ = hook(p, "actually, let's switch to gpt-4o for action items", transport=NoConflicts())
    text = out["hookSpecificOutput"]["additionalContext"]
    assert "Monthly budget: 375" in text and "18x the budget" in text


def test_require_writes_decision_records_for_both_lists(project):
    p, did, db = project
    code, report, _ = run_require(p, "We signed an EU customer: EU data residency required")
    recs = sorted((p / "decisions").glob("*.md"))
    assert len(recs) == 2 and "Decision records updated:" in report
    assert "Entire Brain is not installed" in report          # graceful skip, with how-to
    texts = {r.name: r.read_text(encoding="utf-8") for r in recs}
    stale = next(t for t in texts.values() if "openai/gpt-oss-120b" in t)
    review = next(t for t in texts.values() if "MongoDB" in t)
    assert "STALE" in stale and "documents data residency" in stale
    assert "Needs review" in review and "mongodb.com" in review


# --- Milestone 5 fixes: grounded constraints (R2) and cost facts (R1) ---------------------

class OverreachingExtractor(FakeTransport):
    """An extractor that infers constraints the requirement never states, and a judge that
    calls every decision affected."""

    def __init__(self, extracted, prop="handles the data"):
        super().__init__()
        self.extracted, self.prop = extracted, prop

    def respond(self, model, messages):
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        if system.startswith("You turn a product requirement"):
            return "extract", json.dumps({**self.extracted, "summary": "fake"})
        if system.startswith("You check whether an existing technical decision"):
            return "stale_judge", json.dumps({"affected": True, "property": self.prop,
                                              "reason": "judge thinks so"})
        return super().respond(model, messages)


def test_hipaa_alone_never_makes_anything_stale(project):
    p, did, db = project
    t = OverreachingExtractor({"data_residency": "US", "max_monthly_cost_usd": 10,
                               "expected_users": 1_000_000})
    run_require(p, "We will store patient health records, so we must be HIPAA compliant", t)
    with Graph.for_project(p) as g:
        [req] = [n for n in g.nodes("requirement") if n["data"].get("source") == "require"]
        assert req["data"]["constraints"] == {}            # nothing the text states
        for d in g.nodes("decision"):
            assert d["data"]["status"] != "stale", d
            assert d["data"]["status"] == "needs_review"   # the judge may still ask for review


def test_grounded_keeps_only_what_the_text_states():
    from notbydefault.stale import grounded
    assert grounded({"data_residency": "EU"}, "All user data must stay in the EU")[0] == \
        {"data_residency": "EU"}
    assert grounded({"data_residency": "US"}, "Must be HIPAA compliant")[0] == {}
    assert grounded({"data_residency": "US"}, "Tell us where it is stored")[0] == {}
    assert grounded({"max_monthly_cost_usd": 50}, "Budget is cut to $50 a month")[0] == \
        {"max_monthly_cost_usd": 50}
    assert grounded({"max_monthly_cost_usd": 40}, "Budget is cut to $50 a month")[0] == {}
    assert grounded({}, "Plan for 10x the users", recorded_users=20000)[0] == \
        {"expected_users": 200000}
    assert grounded({"expected_users": 200000}, "Expect 200,000 users")[0] == \
        {"expected_users": 200000}


def test_budget_cut_marks_an_over_budget_model_stale(project):
    p, did, db = project
    t = OverreachingExtractor({"max_monthly_cost_usd": 50})
    code, report, _ = run_require(p, "The monthly budget is cut to $50.", t)
    with Graph.for_project(p) as g:
        model = g.get_node(did)["data"]
        assert model["status"] == "stale"
        assert model["stale_basis"] == "list price x recorded volume"
        assert "vs a $50/month budget (OVER BUDGET)" in model["stale_reason"]
        assert "1,200,000 calls/month" in model["stale_reason"]   # 20,000 users x 60 calls
        # a service has no list price: the judge can only ask for a review
        assert g.get_node(db)["data"]["status"] == "needs_review"
    assert "OVER BUDGET" in report


def test_user_growth_uses_the_new_volume_against_the_recorded_budget(project):
    p, did, db = project
    t = OverreachingExtractor({})
    run_require(p, "Plan for 10x the users.", t)
    with Graph.for_project(p) as g:
        model = g.get_node(did)["data"]
        assert model["status"] == "stale"
        assert "12,000,000 calls/month" in model["stale_reason"]  # 200,000 users x 60 calls
        assert "vs a $375/month budget" in model["stale_reason"]


def test_cost_facts_reach_the_judge(project):
    p, did, db = project
    seen = []

    class Spy(OverreachingExtractor):
        def respond(self, model, messages):
            if messages[0]["content"].startswith("You check whether"):
                seen.append(messages[-1]["content"])
            return super().respond(model, messages)
    run_require(p, "The monthly budget is cut to $50.", Spy({"max_monthly_cost_usd": 50}))
    [prompt] = [s for s in seen if "openai/gpt-oss-120b" in s.split("Kind:")[0]]
    assert "Cost check:" in prompt and "OVER BUDGET" in prompt


def test_requirement_that_changes_no_cost_inputs_adds_no_cost_fact(project):
    p, did, db = project
    run_require(p, "We signed an EU customer: EU data residency required")
    with Graph.for_project(p) as g:
        assert g.get_node(did)["data"]["stale_basis"] == "provider documentation"


# --- widened pre-filter (P1) ----------------------------------------------------------------

@pytest.mark.parametrize("prompt", [
    "Use Postgres for storage from now on, not Mongo.",
    "We're going with Gemini 3.8 Flash for action items.",
    "Use Postgres, not Mongo",
])
def test_prefilter_catches_pivots_without_the_old_phrases(prompt):
    assert prefilter(prompt, stack_terms(LABELS))


@pytest.mark.parametrize("prompt", [
    "this is not a bug, the test is wrong",
    "do not drop the table",
    "from now on, write tests before the code",
])
def test_widened_prefilter_still_skips_non_stack_prompts(prompt):
    assert prefilter(prompt, stack_terms(LABELS)) == []


def test_spoken_model_names_get_a_price_lookup_id():
    from notbydefault.pivot import named_models
    assert named_models("We're going with Gemini 3.8 Flash and GPT 4o") == \
        ["gemini-3.8-flash", "gpt-4o"]


# --- goal / deadline changes (P2) -----------------------------------------------------------

PLAN = ["A notes API for a small team, MVP in 2 weeks", "Expected users: 20,000",
        "Monthly budget: $375", "Team size: 2"]


@pytest.mark.parametrize("prompt", [
    "Forget the 2-week MVP, we're building this as a full enterprise platform now.",
    "We now need this by Friday.",
    "The deadline moved: we have to ship within 3 days.",
])
def test_prefilter_catches_goal_and_deadline_changes_from_the_plan(prompt):
    assert prefilter(prompt, stack_terms(LABELS), PLAN)


def test_plan_changes_need_a_recorded_goal_to_touch():
    assert prefilter("We now need this by Friday.", stack_terms(LABELS), []) == []
    assert prefilter("We now need this by Friday.", stack_terms(LABELS),
                     ["A notes API for a small team"]) == []     # no deadline recorded


def test_ordinary_prompts_stay_free_with_the_plan():
    terms = stack_terms(LABELS)
    assert [p for p in ORDINARY if prefilter(p, terms, PLAN)] == []


def test_hook_passes_the_plan_to_the_prefilter(project):
    p, _, _ = project
    out, info, t = hook(p, "We now need this by Friday.")
    assert info["stage"] == "prefilter" and t.calls == []    # no deadline recorded yet
    with Graph.for_project(p) as g:
        g.add_node("goal", "MVP in 2 weeks", {"source": "init", "key": "goal"})
    out, info, t = hook(p, "We now need this by Friday.")
    assert info["stage"] != "prefilter" and any("plan:" in x for x in info["phrases"])
    assert len(t.calls) == 1


# --- the judge must name a property (R3) ------------------------------------------------------

def test_affected_without_a_named_property_is_valid(project):
    p, did, db = project
    run_require(p, "All API responses must be JSON", OverreachingExtractor({}, prop=""))
    with Graph.for_project(p) as g:
        for d in g.nodes("decision"):
            assert d["data"]["status"] not in ("needs_review", "stale"), d
        [req] = [n for n in g.nodes("requirement") if n["data"].get("source") == "require"]
        assert all(j["status"] == "valid" and "named no property" in j["reason"]
                   for j in req["data"]["judged"].values())


def test_needs_review_reason_shows_the_property(project):
    p, did, db = project
    code, report, _ = run_require(p, "We must be HIPAA compliant",
                                  OverreachingExtractor({}, prop="stores personal data"))
    with Graph.for_project(p) as g:
        mongo = g.get_node(db)["data"]
        assert mongo["status"] == "needs_review"
        assert mongo["review_reason"].startswith("stores personal data: ")
    assert "why:   stores personal data: " in report


# --- mapping a pivot by purpose ---------------------------------------------------------------

VIEW = [
    {"id": 1, "label": "openai/gpt-oss-120b", "kind": "model", "files": ["actions.py"],
     "used_for": "", "verdict": "Keep it on Groq for action items."},
    {"id": 2, "label": "gemini-3.1-flash-lite", "kind": "model", "files": ["summarize.py"],
     "used_for": "", "verdict": ""},
    {"id": 3, "label": "MongoDB", "kind": "service", "files": ["app.py", "requirements.txt"],
     "used_for": "", "verdict": ""},
]


def test_purpose_beats_model_family():
    from notbydefault.pivot import purpose_fix
    # the model mapped the Gemini summarizer because the prompt names Gemini
    assert purpose_fix("We're going with Gemini 3.8 Flash for action items.", VIEW, [2]) == [1]


def test_purpose_from_a_file_name():
    from notbydefault.pivot import purpose_fix
    assert purpose_fix("Replace the model in summarize.py with claude-sonnet-4-5", VIEW, [1]) == [2]


def test_no_stated_purpose_keeps_the_models_mapping():
    from notbydefault.pivot import purpose_fix
    assert purpose_fix("Drop Gemini, use Claude everywhere", VIEW, [2]) == [2]
    assert purpose_fix("Switch to Postgres for the notes", VIEW, [3]) == [3]   # nothing matches


def test_purpose_keeps_mapped_decisions_of_another_kind():
    from notbydefault.pivot import purpose_fix
    assert purpose_fix("Use Claude for action items and drop MongoDB", VIEW, [2, 3]) == [1, 3]


def test_confirm_applies_the_purpose_mapping():
    from notbydefault.pivot import confirm

    class Reply:
        text = json.dumps({"is_pivot": True, "change": "gemini for action items",
                           "decision_ids": [2], "conflicts": []})
        latency_s, cached = 0.1, False

    class Caller:
        def chat(self, *a, **k):
            return Reply()
    view = {"goals": [], "requirements_and_constraints": [], "decisions": VIEW}
    r = confirm(Caller(), "m", "We're going with Gemini 3.8 Flash for action items.", view)
    assert r["decision_ids"] == [1] and r["mapped_by_model"] == [2]
