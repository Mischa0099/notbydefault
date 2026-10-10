"""Reference answers, objective verifiers, two judges and two recommendations."""
import json
from datetime import date
from pathlib import Path

import pytest

from conftest import CHOSEN, JUDGE, JUDGE2, SMALL_OTHER, SMALL_SAME, FakeTransport, make_client
from notbydefault.cli import main as cli
from notbydefault.fit import checks
from notbydefault.fit.callsite import source_system_prompt
from notbydefault.fit.dates import mentioned_dates
from notbydefault.fit.evaluate import ALONE, VERIFIED
from notbydefault.fit.reference import score_case
from notbydefault.fit.run import CheckOptions, cache_dir, check_model_decision
from notbydefault.fit.stats import SetupStats, decide_with_references
from notbydefault.graph import Graph

THU = date(2026, 10, 8)   # a Thursday


# --- dates -------------------------------------------------------------------------

@pytest.mark.parametrize("phrase,expected", [
    ("tomorrow", "2026-10-09"),
    ("Friday", "2026-10-09"),                 # first Friday after the meeting
    ("next Friday", "2026-10-16"),            # Friday of the following week
    ("end of the week", "2026-10-09"),
    ("Monday", "2026-10-12"),
    ("end of the month", "2026-10-31"),
    ("October 20", "2026-10-20"),
    ("the 3rd of November", "2026-11-03"),
    ("20/10", "2026-10-20"),
    ("2026-12-01", "2026-12-01"),
    ("el viernes", "2026-10-09"),
    ("el próximo martes", "2026-10-13"),
    ("mañana", "2026-10-09"),
    ("in 3 days", "2026-10-11"),
])
def test_mentioned_dates(phrase, expected):
    found = {d.isoformat() for d in mentioned_dates(f"do it {phrase}", THU)}
    assert expected in found


def test_next_weekday_is_not_also_this_weekday():
    found = {d.isoformat() for d in mentioned_dates("review by next Friday", THU)}
    assert "2026-10-16" in found and "2026-10-09" not in found


# --- objective checks ----------------------------------------------------------------

NOTES = "Meeting date: 2026-10-08 (Thursday)\n\nNotes:\n- Priya will send the quote tomorrow."
GOOD = {"action_items": [{"task": "Send the quote", "owner": "Priya", "due_date": "2026-10-09",
                          "priority": "medium"}]}
SCHEMA = {"type": "object", "required": ["action_items"], "properties": {"action_items": {
    "type": "array", "items": {"type": "object", "required": ["task", "owner", "due_date", "priority"],
                               "properties": {"priority": {"enum": ["high", "medium", "low"]}}}}}}


def bad(**changes):
    item = dict(GOOD["action_items"][0], **changes)
    return json.dumps({"action_items": [item]})


@pytest.mark.parametrize("check,output,ok", [
    (checks.json_matches_schema(SCHEMA), json.dumps(GOOD), True),
    (checks.json_matches_schema(SCHEMA), "```json\n" + json.dumps(GOOD) + "\n```", False),
    (checks.json_matches_schema(SCHEMA), bad(priority="urgent"), False),
    (checks.json_matches_schema(SCHEMA), json.dumps({"items": []}), False),
    (checks.owners_in_input(), json.dumps(GOOD), True),
    (checks.owners_in_input(), bad(owner="Priyanka"), False),
    (checks.owners_in_input(), bad(owner=None), True),
    (checks.dates_in_input(), json.dumps(GOOD), True),
    (checks.dates_in_input(), bad(due_date="2026-10-16"), False),     # nobody said next Friday
    (checks.dates_normalized(), bad(due_date="Oct 9"), False),
    (checks.dates_normalized(), bad(due_date="2026-02-30"), False),
    (checks.dates_normalized(), bad(due_date=None), True),
])
def test_objective_checks(check, output, ok):
    assert check.run(NOTES, output)[0] is ok


def test_reference_checks_are_evaluation_only():
    ref = {"action_items": GOOD["action_items"] + [
        {"task": "Book the venue", "owner": "Dana", "due_date": None, "priority": "medium"}]}
    missing = checks.no_missing_items()
    extra = checks.no_extra_items()
    assert missing.needs_reference and not missing.retryable
    assert not missing.run(NOTES, json.dumps(GOOD), ref)[0]
    assert extra.run(NOTES, json.dumps(GOOD), ref)[0]
    assert not extra.run(NOTES, json.dumps(ref), GOOD)[0]


# --- reference scoring ----------------------------------------------------------------

def test_score_case_correct_and_problems():
    assert score_case(json.dumps(GOOD), GOOD) == {"correct": True, "item_f1": 1.0,
                                                  "valid_json": True, "problems": []}
    s = score_case(bad(due_date="2026-10-16"), GOOD)
    assert not s["correct"] and "due" in s["problems"][0] and s["item_f1"] == 0.0
    assert score_case("not json", GOOD)["valid_json"] is False
    assert score_case(json.dumps({"action_items": []}), {"action_items": []})["correct"]


def test_multiple_owners_compare_as_a_set():
    ref = {"action_items": [dict(GOOD["action_items"][0], owner="Priya, Tom")]}
    assert score_case(bad(owner="Tom, Priya"), ref)["correct"]
    assert not score_case(bad(owner="Priya"), ref)["correct"]


# --- verdict with references -----------------------------------------------------------

def S(model, setup, correct, n, cost, monthly, fits=True, pass_rate=1.0):
    s = SetupStats(model=model, setup=setup, name=f"{model}/{setup}", calls=n, errors=0, scored=n,
                   mean=4.0, lo=3.5, hi=4.5, pass_rate=pass_rate, pass_lo=0, pass_hi=1,
                   avg_attempts=1, avg_latency_s=1, avg_cost=cost, monthly_cost=monthly,
                   fits_budget=fits)
    s.n_ref, s.correct_rate, s.correct_lo, s.correct_hi = n, correct / n, 0, 1
    s.item_f1 = correct / n
    return s


def test_two_recommendations_cheapest_bar_and_best_within_budget():
    stats = {s.key: s for s in [
        S("big", ALONE, 10, 12, 0.0010, 1200.0, fits=False),
        S("small", ALONE, 8, 12, 0.0001, 120.0),
        S("small", VERIFIED, 9, 12, 0.00015, 180.0),
        S("mid", ALONE, 11, 12, 0.0003, 360.0, fits=False),
        S("lite", VERIFIED, 11, 12, 0.0002, 240.0),
    ]}
    v = decide_with_references(stats, "big", budget=250.0)
    assert v.recommended == ("small", VERIFIED)          # 9/12 >= 10/12 - one case
    assert v.best_quality == ("lite", VERIFIED)          # most correct that fits $250
    assert "below the bar" in v.rejected["small/alone"]
    assert "over budget" in v.rejected["mid/alone"]
    assert "Cheapest setup that meets the bar" in v.summary and "Best quality" in v.summary


def test_no_setup_within_budget_is_said_plainly():
    stats = {s.key: s for s in [S("big", ALONE, 10, 12, 0.001, 1200.0, fits=False)]}
    v = decide_with_references(stats, "big", budget=25.0)
    assert v.best_quality is None and "no setup fits" in v.summary


# --- system prompt read from source ---------------------------------------------------

def test_source_system_prompt_constant_and_literal(tmp_path):
    f = tmp_path / "a.py"
    f.write_text('P = "rules\\nmore rules"\n'
                 'def a():\n    return [{"role": "system", "content": P}]\n'
                 'def b():\n    return [{"role": "system", "content": "inline prompt"}]\n')
    assert source_system_prompt(f, 3) == "rules\nmore rules"
    assert source_system_prompt(f, 5) == "inline prompt"
    assert source_system_prompt(tmp_path / "missing.py") is None


# --- end to end: references, retries that rescue, two judges ---------------------------

APP = '''SYSTEM_PROMPT = "Extract action items as JSON."

def extract(client, msg):
    return client.chat.completions.create(model="openai/gpt-oss-120b", messages=[
        {"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": msg}])
'''


def case(cid, notes, items, hard=False):
    msg = f"Meeting date: 2026-10-08 (Thursday)\n\nNotes:\n{notes}"
    return {"id": cid, "hard": hard, "user_message": msg, "source": "hand-written",
            "reference": {"action_items": items}}


def it(task, owner, due, priority="medium"):
    return {"task": task, "owner": owner, "due_date": due, "priority": priority}


CASES = [
    case("k1", "- Priya will send the quote tomorrow.", [it("Send the quote", "Priya", "2026-10-09")]),
    case("k2", "- Tom to review the questionnaire by next Friday.",
         [it("Review the questionnaire", "Tom", "2026-10-16")], hard=True),
    case("k3", "- Jun updates the doc by end of the week.", [it("Update the doc", "Jun", "2026-10-09")]),
    case("k4", "- Dana books the venue.\n- Later: scratch the venue.", [], hard=True),
    case("k5", "- Leo fixes the build ASAP.", [it("Fix the build", "Leo", None, "high")]),
    case("k6", "- Mei sends notes on Monday.", [it("Send notes", "Mei", "2026-10-12")]),
]


class JsonTransport(FakeTransport):
    """Candidates answer the action-items task: the large model is always right; the small
    one gets a relative date wrong (fixable on retry) and keeps a cancelled item (not
    fixable: only a reference check sees it); the lite one wraps JSON in code fences."""

    def respond(self, model, messages):
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        if system != "Extract action items as JSON.":
            return super().respond(model, messages)
        msg = messages[1]["content"]
        retry = "failed an automatic check" in messages[-1]["content"]
        c = next(c for c in CASES if c["user_message"] == msg)
        items = [dict(x) for x in c["reference"]["action_items"]]
        if model == "openai/gpt-oss-20b":
            if c["id"] == "k2" and not retry:
                items[0]["due_date"] = "2026-10-09"           # took "next Friday" as this Friday
            if c["id"] == "k4":
                items = [it("Book the venue", "Dana", None)]  # missed the cancellation
        text = json.dumps({"action_items": items})
        if model == "gemini-3.5-flash-lite" and not retry:
            text = "```json\n" + text + "\n```"
        return "candidate", text


@pytest.fixture
def ref_project(tmp_path):
    cli(["init", str(tmp_path), "--goal", "Notes API", "--users", "20000", "--budget", "100",
         "--no-input"])
    (tmp_path / "actions.py").write_text(APP)
    with Graph.for_project(tmp_path) as g:
        did = g.add_node("decision", "openai/gpt-oss-120b", {
            "kind": "model", "category": "ai-model", "status": "unreviewed",
            "file": "actions.py", "files": ["actions.py"]})
        for n in g.nodes():
            if n["type"] in ("requirement", "constraint"):
                g.add_edge(did, n["id"], "depends_on")
    nbd = tmp_path / ".notbydefault"
    (nbd / "samples").mkdir(parents=True)
    (nbd / "samples" / f"{did}.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in CASES), encoding="utf-8")
    (nbd / "verifiers").mkdir()
    verifiers = Path(__file__).resolve().parents[1] / "demo-app" / "eval" / "action_item_verifiers.py"
    (nbd / "verifiers" / f"{did}.py").write_text(verifiers.read_text(encoding="utf-8"),
                                                 encoding="utf-8")
    return tmp_path, did


def run_ref(p, did):
    out = []
    client = make_client(cache_dir=cache_dir(p), transport=JsonTransport())
    code = check_model_decision(p, did, CheckOptions(yes=True, calls_per_user=60, workers=2),
                                client=client, ask=lambda _: "", out=out.append)
    return code, "\n".join(out), client


def test_reference_run_rescues_and_recommends_twice(ref_project):
    p, did = ref_project
    code, report, client = run_ref(p, did)
    assert code == 0, report
    results = json.loads(next((p / ".notbydefault" / "results").glob(f"{did}-*.json")).read_text())
    st = results["stats"]
    small, small_v = st[SMALL_SAME], st[f"{SMALL_SAME} + verifier (<=2 retries)"]
    lite, lite_v = st[SMALL_OTHER], st[f"{SMALL_OTHER} + verifier (<=2 retries)"]
    assert st[CHOSEN]["correct_rate"] == 1.0
    assert round(small["correct_rate"] * 6) == 4 and round(small_v["correct_rate"] * 6) == 5
    assert lite["correct_rate"] == 0.0 and lite_v["correct_rate"] == 1.0
    assert set(small_v["failed_cases"]) == {"k4"}          # the cancelled item: no retry can fix
    # both judges scored, from different families, and agreement is reported
    assert set(st[CHOSEN]["judge_means"]) == {JUDGE, JUDGE2}
    assert results["judge_agreement"]["n"] > 0
    # the two recommendations
    v = results["verdict"]
    assert tuple(v["recommended"]) == (SMALL_SAME, "verified")      # 5/6 meets 6/6 - 1
    assert tuple(v["best_quality"]) == (SMALL_OTHER, "verified")    # 6/6 and cheaper than 120b
    assert "1. Cheapest setup that meets the bar: groq:openai/gpt-oss-20b + verifier" in report
    assert "2. Best quality within the budget:    gemini:gemini-3.5-flash-lite + verifier" in report
    # case results: what failed, what retries rescued, what slipped past production checks
    assert "rescued 1 (k2)" in report
    assert "k4 [hard]: extra: Book the venue" in report
    assert "rescued 6" in report                           # every fenced lite answer fixed
    assert "Two blind judges from different families" in report
    # no LLM rubric: verified setups pay only for retries
    assert not [k for _, k in client.transport.calls if k == "rubric"]
    journal = results["journal"]
    no_retry = [e for e in journal if e["name"] == "no_retry"]
    assert [(e["model"], e["sample"]) for e in no_retry] == [(SMALL_SAME, 3)]   # k4
    assert "only reference checks" in no_retry[0]["reason"]
    assert any(e["name"] == "judge_agreement" and e["focus_area"] == "verification" for e in journal)


def test_reference_failures_are_never_retry_hints(ref_project):
    p, did = ref_project
    _, _, client = run_ref(p, did)
    retries = [m for m, k in client.transport.calls if k == "candidate"]
    # 6 cases x 3 models first attempts + 1 retry (small, k2) + 6 retries (lite fences)
    assert len(retries) == 18 + 1 + 6


# --- regressions found on the real #21 run ------------------------------------------

def test_meeting_date_header_is_not_a_due_date():
    notes = "Meeting date: 2026-10-21 (Wednesday)\n\nNotes:\n- Raj will patch the auth bypass ASAP."
    asap = json.dumps({"action_items": [{"task": "Patch", "owner": "Raj",
                                         "due_date": "2026-10-21", "priority": "high"}]})
    assert not checks.dates_in_input().run(notes, asap)[0]
    today = notes.replace("ASAP", "today")
    assert checks.dates_in_input().run(today, asap)[0]


def test_empty_responses_are_cached_too(tmp_path):
    class Empty(FakeTransport):
        def respond(self, model, messages):
            return "candidate", ""
    t = Empty()
    client = make_client(cache_dir=tmp_path, transport=t)
    msgs = [{"role": "user", "content": "x"}]
    assert client.chat(CHOSEN, msgs).text == ""
    assert client.chat(CHOSEN, msgs).cached and len(t.calls) == 1


def test_reference_mode_skips_unscorable_failure_bank_inputs(ref_project):
    from notbydefault.fit import failures
    from notbydefault.fit.run import assemble_inputs
    from notbydefault.fit.samples import load
    p, did = ref_project
    failures.append(p, did, [
        {"input": "an old input with no reference", "output": "x", "model": "m", "failed": []},
        {"input": CASES[1]["user_message"], "output": "x", "model": "m", "failed": []}])
    inputs = assemble_inputs(p, did, load(p, did))
    assert [i.get("id") for i in inputs] == ["k2", "k1", "k3", "k4", "k5", "k6"]
    assert inputs[0]["from_failure_bank"] and all(i.get("reference") is not None for i in inputs)


def test_failed_calls_appear_as_wrong_cases():
    from notbydefault.fit.report import ReportContext, case_results
    rows = [{"case": "k1", "hard": False, "has_reference": True, "model": "m", "setup": s,
             "correct": None, "passed": False, "attempts": 0, "error": "HTTP 500",
             "problems": [], "failed_checks": []} for s in ("alone", "verified")]
    ctx = ReportContext(1, "x", "m", "", 1, {}, "", "", [], "", 2, None, 1, None, 0, "", "", "",
                        case_rows=rows)
    lines = case_results(ctx)
    assert lines[0] == "m: wrong on 1/1 (k1)" and "HTTP 500" in lines[1]
    assert "0 wrong answer(s) passed every production check" in lines[2]


def test_revised_cases_are_disclosed_in_report_and_record(ref_project):
    p, did = ref_project
    path = p / ".notbydefault" / "samples" / f"{did}.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    rows[4]["revision_note"] = "reworded during development after all models omitted the item"
    path.write_text("".join(json.dumps(r) + chr(10) for r in rows))
    code, report, _ = run_ref(p, did)
    assert code == 0
    assert "Caveats:" in report
    assert "Case k5: reworded during development after all models omitted the item." in report
    assert "Only 6 cases" in report
    record = next((p / "decisions").glob("*.md")).read_text()
    assert "## Caveats" in record and "Case k5: reworded" in record
