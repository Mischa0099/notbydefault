"""Claim check (`claims`) and watched assumptions (`watch`).

Fit checks run on the fake providers; the claim split's model call is faked too (or fails, so the
keyword fallback is used). Nothing here reaches the network."""
import json

import pytest

from conftest import CATALOG, FakeTransport, make_client
from notbydefault import assumptions, claims
from notbydefault import watch as watch_mod
from notbydefault.cli import main as cli
from notbydefault.graph import Graph
from notbydefault.hooks.pre_tool import handle
from test_fit_check import client_for, make_project, run

REASON = ("I picked openai/gpt-oss-120b because it is the cheapest option. It is the most "
          "accurate model for extracting notes. It is fast. Data stays in the EU. "
          "Everyone uses it, and it is mature and battle-tested.")


@pytest.fixture
def checked(tmp_path):
    """A fit-checked model decision whose agent gave REASON."""
    p, did = make_project(tmp_path / "proj")
    with Graph.for_project(p) as g:
        g.update_data(did, agent_reason=REASON, agent_reason_source="session transcript")
    client = client_for(p)
    code, report = run(p, did, client=client)
    assert code == 0, report
    return p, did, client, report


def node(p, nid):
    with Graph.for_project(p) as g:
        return g.get_node(nid)


# --- claims -------------------------------------------------------------------------------

def test_keyword_split_types_every_kind_of_claim():
    got = {c["type"]: c["text"] for c in claims.keyword_claims(REASON)}
    assert set(got) == {"cost", "quality", "latency", "compliance", "popularity", "maturity"}
    assert got["compliance"] == "Data stays in the EU"
    assert "Everyone uses it" in got["popularity"]


def test_reason_from_the_hook_event_uses_description_and_transcript(tmp_path):
    t = tmp_path / "t.jsonl"
    t.write_text("\n".join(json.dumps(r) for r in [
        {"type": "user", "message": {"content": "add a database"}},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "MongoDB is popular and cheap to run."},
            {"type": "tool_use", "id": "toolu_1", "name": "Bash"}]}}]))
    reason, source = claims.reason_from_event({"transcript_path": str(t), "tool_input": {
        "command": "pip install pymongo", "description": "Install the MongoDB driver"}})
    assert reason == "Install the MongoDB driver MongoDB is popular and cheap to run."
    assert source == "tool call description + session transcript"


def test_hook_records_the_agents_stated_reason(tmp_path):
    cli(["init", str(tmp_path), "--goal", "Notes", "--users", "100", "--no-input"])
    t = tmp_path / "t.jsonl"
    t.write_text(json.dumps({"type": "assistant", "message": {"content": [
        {"type": "text", "text": "I'll use MongoDB, it is the most popular database."}]}}))
    event = {"session_id": "s", "tool_name": "Bash", "tool_use_id": "toolu_9",
             "transcript_path": str(t), "tool_input": {"command": "pip install pymongo"}}
    assert handle(event, tmp_path)
    with Graph.for_project(tmp_path) as g:
        [d] = g.list_decisions()
    assert d["data"]["agent_reason"] == "I'll use MongoDB, it is the most popular database."
    assert d["data"]["agent_reason_source"] == "session transcript"


def test_claims_are_checked_against_the_fit_check_and_documented_facts(checked):
    p, did, _client, report = checked
    c = node(p, did)["data"]["claims"]
    by = {i["type"]: i for i in c["items"]}
    assert c["classified_by"] == "keywords"          # the fake model doesn't answer the split
    results = node(p, did)["data"]["results"]
    for kind in ("cost", "quality", "latency"):
        assert by[kind]["verdict"] in ("true", "false")
        assert by[kind]["source"] == f"fit check {results}"
    assert by["compliance"]["verdict"] == "false"    # groq documents US-only residency
    assert "documented data residency: US" in by["compliance"]["detail"]
    assert by["popularity"]["verdict"] == "untested"
    assert "not evidence for your project" in by["popularity"]["detail"]
    assert by["maturity"]["verdict"] == "untested"
    # shown in the check report and the decision record
    assert "Claim check: the agent's reason (session transcript) makes 6 claim(s)" in report
    record = (p / node(p, did)["data"]["decision_record"]).read_text(encoding="utf-8")
    assert "### Claim check" in record and "**FALSE** (compliance)" in record


def test_cost_and_latency_verdicts_follow_the_numbers(checked):
    p, did, client, _ = checked
    data = node(p, did)["data"]
    res = json.loads((p / data["results"]).read_text(encoding="utf-8"))
    stats = list(res["stats"].values())
    cur = next(s for s in stats if s["model"] == res["config"]["chosen"] and s["setup"] == "alone")
    cheapest = min((s for s in stats if s["setup"] == "alone"), key=lambda s: s["avg_cost"])
    by = {i["type"]: i for i in data["claims"]["items"]}
    expected = cur["fits_budget"] is not False and cur["avg_cost"] <= cheapest["avg_cost"] * 1.05
    assert by["cost"]["verdict"] == ("true" if expected else "false")
    assert f"${cur['avg_cost']:,.6f}/call" in by["cost"]["detail"]


def test_model_split_keeps_only_quoted_claims(tmp_path):
    class Splitter(FakeTransport):
        def respond(self, model, messages):
            if messages[0]["content"].startswith("You split a coding agent"):
                return "split", json.dumps({"claims": [
                    {"type": "cost", "text": "it is the cheapest option"},
                    {"type": "quality", "text": "it won every benchmark"},       # not in the text
                    {"type": "vibes", "text": "It is fast"}]})                   # unknown type
            return super().respond(model, messages)
    p, did = make_project(tmp_path)
    with Graph.for_project(p) as g:
        g.update_data(did, agent_reason=REASON)
        c = claims.evaluate(p, g, g.get_node(did), make_client(transport=Splitter()))
    assert c["classified_by"] == "groq:openai/gpt-oss-20b"
    assert [(i["type"], i["text"]) for i in c["items"]] == [("cost", "it is the cheapest option")]
    assert c["items"][0]["verdict"] == "untested"     # no fit check, no recorded call volume


def test_claims_cli_stores_and_prints_and_says_when_there_is_no_reason(tmp_path, monkeypatch):
    p, did = make_project(tmp_path)
    out = []
    assert claims.run(p, did, client=make_client(), entire=None, out=out.append) == 0
    assert "no stated reason recorded" in out[0]
    with Graph.for_project(p) as g:
        g.update_data(did, agent_reason="Popular with everyone. It is fast.")
    out = []
    assert claims.run(p, did, client=make_client(), out=out.append) == 0
    text = "\n".join(out)
    assert "UNTESTED  popularity" in text and "UNTESTED  latency" in text
    assert node(p, did)["data"]["claims"]["items"][1]["detail"].startswith("not measured")


# --- watch ------------------------------------------------------------------------------

def test_check_records_assumptions_with_sources(checked):
    p, did, _, _ = checked
    snap = node(p, did)["data"]["assumptions"]
    items = {i["key"]: i for i in snap["items"]}
    assert items["users"]["value"] == 500 and items["budget"]["value"] == 25
    assert items["calls_per_user"]["value"] == 30
    assert items["data_region"]["value"] == "any"
    assert any(k.startswith("price:groq:") for k in items)
    assert "groq:openai/gpt-oss-120b" in items["models"]["value"]
    assert snap["basis"].startswith("fit check .notbydefault/results/")


def test_unchecked_decisions_get_assumptions_from_graph_and_catalog(tmp_path):
    p, did = make_project(tmp_path)
    with Graph.for_project(p) as g:
        db = g.add_node("decision", "MongoDB", {"kind": "service", "category": "database"})
        eu = g.add_node("requirement", "EU only", {"constraints": {"data_residency": "EU"}})
        model = assumptions.snapshot(p, g, g.get_node(did), CATALOG)
        service = assumptions.snapshot(p, g, g.get_node(db), CATALOG)
    items = {i["key"]: i for i in model["items"]}
    assert items["price:groq:openai/gpt-oss-120b"]["value"] == {"in": 2.5, "out": 10.0}
    assert items["price:groq:openai/gpt-oss-120b"]["fetched"] == "2026-10-10"
    assert items["data_region"] == {"key": "data_region", "label": "data region", "value": "EU",
                                    "source": f'requirement #{eu} "EU only"'}
    assert eu in items["requirements"]["value"] and items["models"]["value"]
    assert all(i["source"] for i in model["items"] + service["items"])
    assert not any(i["key"].startswith(("price:", "models")) for i in service["items"])


def test_watch_reopens_a_decision_when_a_requirement_breaks_an_assumption(checked):
    p, did, client, _ = checked
    with Graph.for_project(p) as g:
        g.add_node("requirement", "EU only", {"source": "require",
                                              "constraints": {"data_residency": "EU"}})
    out = []
    assert watch_mod.watch(p, client=client, out=out.append) == 0
    assert f"re-opened #{did}: 'data region: any' broken by 'EU only'" in out
    d = node(p, did)["data"]
    assert d["status"] == "needs_review" and d["review_basis"] == "watched assumption"
    assert d["assumptions"]["broken"][0]["now"] == "EU"
    assert "re-opened by watch" in (p / d["decision_record"]).read_text(encoding="utf-8")
    out = []                                  # a second run does not re-open it again
    watch_mod.watch(p, client=client, out=out.append)
    assert any(l.startswith(f"already    #{did}") for l in out)


def test_watch_sees_price_changes_new_models_and_real_volume(checked):
    p, did, _, _ = checked
    import copy
    cat = copy.deepcopy(CATALOG)
    cat["providers"]["groq"]["models"]["openai/gpt-oss-120b"]["in"] = 0.5
    cat["providers"]["groq"]["models"]["brand-new-model"] = dict(
        cat["providers"]["groq"]["models"]["openai/gpt-oss-20b"])
    with Graph.for_project(p) as g:
        items = assumptions.broken(p, g, g.get_node(did), cat, observed_calls=60_000,
                                   observed_source="neatlogs traces, last 30 days")
    by = {b["key"]: b for b in items}
    assert by["price:groq:openai/gpt-oss-120b"]["now"]["in"] == 0.5
    assert "groq:brand-new-model" in by["models"]["cause"]
    assert by["call_volume"]["was"] == 15_000 and by["call_volume"]["now"] == 60_000


def test_watch_records_baselines_first_and_reports_unmeasured_volume(tmp_path):
    p, did = make_project(tmp_path)
    out = []
    watch_mod.watch(p, client=make_client(), out=out.append)
    assert f"baseline   #{did} openai/gpt-oss-120b: assumptions recorded" in out
    assert "real call volume: not measured" in out[-1]
    assert node(p, did)["data"]["status"] == "unreviewed"


def test_auto_check_runs_reopened_model_decisions_within_max_cost(checked, tmp_path):
    p, did, client, _ = checked
    with Graph.for_project(p) as g:
        other = g.add_node("decision", "gpt-x", {"kind": "model", "category": "ai-model",
                                                 "status": "reviewed"})
        assumptions.record_snapshot(p, g, other, client.catalog)
        g.add_node("requirement", "EU only", {"constraints": {"data_residency": "EU"}})
    calls = []

    def fake_check(project, decision_id, opts, client=None):
        calls.append((decision_id, opts.max_cost, opts.yes))
        folder = project / ".notbydefault" / "results"
        (folder / f"{decision_id}-29991231T000000Z.json").write_text(json.dumps({"spent": 0.7}))
        return 0
    out = []
    watch_mod.watch(p, auto_check=True, max_cost=0.7, client=client, out=out.append,
                      check_fn=fake_check)
    assert calls == [(did, 0.7, True)]
    assert f"queued     #{other}: not checked, --max-cost 0.70 used up" in out


def test_real_call_volume_from_neatlogs_traces_breaks_the_volume_assumption(checked):
    p, did, client, _ = checked
    asked = []

    def fetch(url, headers, timeout):
        asked.append((url, headers["Authorization"], headers["x-project-id"]))
        spans = [{"spanId": f"s{i}", "spanType": "LLM", "spanName": "chat",
                  "attributes": {"gen_ai.request.model": "openai/gpt-oss-120b"}} for i in range(3)]
        spans.append({"spanId": "x", "spanType": "tool", "spanName": "openai/gpt-oss-120b"})
        return 200, json.dumps({"data": {"spans": spans, "page": {}}}).encode()
    env = {"NEATLOGS_TOKEN": "tok", "NEATLOGS_PROJECT_ID": "proj"}
    src = watch_mod.traces_volume(["t1", "t2"], days=1, env=env, fetch=fetch)
    calls, why = src(p, node(p, did))
    assert calls == 6 * 30 and "6 call(s)" in why and "2 production trace(s)" in why
    assert asked[0] == ("https://app.neatlogs.com/api/v1/public/traces/t1/spans?limit=50",
                        "Bearer tok", "proj")
    out = []
    watch_mod.watch(p, client=client, out=out.append, volume_source=src)
    assert any("'calls per month: 15,000' broken by 'neatlogs: 6 call(s)" in l for l in out)
    assert not any(l.startswith("real call volume: not measured") for l in out)


def test_volume_is_not_measured_without_a_token_or_trace_ids():
    assert watch_mod.volume_note({}) == "NEATLOGS_TOKEN not set"
    assert "no trace-listing endpoint" in watch_mod.volume_note({"NEATLOGS_TOKEN": "t"})
    assert watch_mod.trace_spans("t1", env={}) is None


def test_reopen_puts_failure_bank_cases_first(checked):
    p, did, client, _ = checked
    from notbydefault.fit import failures
    banked = len(failures.load(p, did))
    assert banked, "the fake fit check banks failing outputs"
    with Graph.for_project(p) as g:
        g.add_node("requirement", "EU only", {"constraints": {"data_residency": "EU"}})
    watch_mod.watch(p, client=client, out=lambda _: None)
    assert node(p, did)["data"]["rerun_first"] == banked
    from notbydefault.fit.run import assemble_inputs
    if banked:
        first = assemble_inputs(p, did, [{"user_message": "zzz new", "source": "synthetic"}])[0]
        assert first["source"] != "synthetic"


def test_auto_check_stops_the_queue_when_a_check_hits_a_quota(checked):
    p, did, client, _ = checked
    with Graph.for_project(p) as g:
        other = g.add_node("decision", "gpt-x", {"kind": "model", "category": "ai-model",
                                                 "status": "reviewed"})
        assumptions.record_snapshot(p, g, other, client.catalog)
        g.add_node("requirement", "EU only", {"constraints": {"data_residency": "EU"}})
    calls = []

    def limited(project, decision_id, opts, client=None):
        calls.append(decision_id)
        (project / ".notbydefault" / "results" / f"{decision_id}-29991231T000000Z.json") \
            .write_text(json.dumps({"spent": 0.01, "stopped": True}))
        return 0
    out = []
    watch_mod.watch(p, auto_check=True, max_cost=5, client=client, out=out.append,
                    check_fn=limited)
    assert calls == [did]
    assert any(l.startswith(f"stopped    #{did}") for l in out)


def test_compliance_claim_on_a_provider_without_a_guarantee_is_untested(tmp_path):
    p, did = make_project(tmp_path)
    with Graph.for_project(p) as g:
        g.update_data(did, agent_reason="Data stays in the EU.",
                      resolution={"to": "gemini:gemini-3.5-flash-lite"})
        c = claims.evaluate(p, g, g.get_node(did), make_client(), split=False)
    [item] = c["items"]
    assert item["verdict"] == "untested"
    assert "no documented data-residency guarantee" in item["detail"]


def test_cheaper_than_claim_compares_list_prices_with_the_named_model(tmp_path):
    p, did = make_project(tmp_path)
    with Graph.for_project(p) as g:      # fake list prices: gpt-oss-120b 2.5/10, qwen 0.20/0.80
        g.update_data(did, agent_reason="It is cheaper than qwen3.8-27b.")
        c = claims.evaluate(p, g, g.get_node(did), make_client(), split=False)
    [item] = c["items"]
    assert item["verdict"] == "false"
    assert "for groq:qwen/qwen3.8-27b" in item["detail"] and "list price $" in item["detail"]
