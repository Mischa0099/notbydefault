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
        model = g.get_node(did)["data"]
        assert model["status"] == "stale" and "documents data residency" in model["stale_reason"]
        assert g.get_node(db)["data"]["status"] == "stale"          # judged by the model
        assert {"src": did, "dst": req["id"], "type": "depends_on"} in g.edges()
    assert "2 of 2 decisions are now stale" in report
    assert "code: actions.py" in report and "code: requirements.txt" in report


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
