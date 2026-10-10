import json

import pytest

from conftest import MODELS
from notbydefault.fit import candidates, checks
from notbydefault.fit.callsite import CallSite
from notbydefault.fit.evaluate import ALONE, VERIFIED, BudgetExceeded, Caller, SpendGuard
from notbydefault.fit.stats import SetupStats, bootstrap_ci, decide, parse_number
from notbydefault.fit.tracing import Tracer
from notbydefault.fit.verifiers import Verifiers, generate, load, verify
from notbydefault.openrouter import Price, parse_json, pricing_from_models

PRICING = pricing_from_models(MODELS)


# --- openrouter helpers --------------------------------------------------------

def test_price_cost_is_per_token():
    assert Price(2.5e-6, 10e-6).cost(1000, 100) == pytest.approx(0.0035)
    assert PRICING.price("openai/gpt-4o").prompt == pytest.approx(2.5e-6)


@pytest.mark.parametrize("text", ['{"a": 1}', '```json\n{"a": 1}\n```', 'Sure! {"a": 1} done'])
def test_parse_json_tolerates_chatter(text):
    assert parse_json(text) == {"a": 1}


# --- candidates -------------------------------------------------------------------

@pytest.mark.parametrize("literal,expected", [
    ("gpt-4o", "openai/gpt-4o"),
    ("openai/gpt-4o", "openai/gpt-4o"),
    ("claude-sonnet-4-5", "anthropic/claude-sonnet-4.5"),
    ("claude-sonnet-4-5-20250929", "anthropic/claude-sonnet-4.5"),
])
def test_resolve_model_id(literal, expected):
    assert candidates.resolve_model_id(literal, PRICING) == expected


def test_resolve_unknown_model_raises():
    with pytest.raises(LookupError):
        candidates.resolve_model_id("gpt-99-ultra", PRICING)


def test_alternatives_are_cheaper_newest_and_from_other_providers():
    alts = candidates.pick_alternatives("openai/gpt-4o", PRICING)
    assert alts == ["anthropic/claude-haiku-5.5", "google/gemini-3.8-flash",
                    "meta-llama/llama-4-scout"]


def test_judge_is_strong_and_not_a_candidate():
    assert candidates.pick_judge(["openai/gpt-4o"], PRICING) == "anthropic/claude-opus-5.5"
    assert candidates.pick_judge(["anthropic/claude-opus-5.5"], PRICING) == \
        "anthropic/claude-sonnet-4.5"


def test_candidates_yaml_created_once_with_price_source(tmp_path):
    decision = {"id": 10, "label": "gpt-4o"}
    cfg, created = candidates.load_or_create(tmp_path, decision, PRICING)
    assert created and cfg["chosen"] == "openai/gpt-4o" and len(cfg["candidates"]) == 4
    assert cfg["verifier_model"] == "meta-llama/llama-4-scout"   # cheapest candidate
    text = candidates.config_path(tmp_path).read_text()
    assert "pricing_source: https://openrouter.ai/api/v1/models" in text
    assert "pricing_fetched_at:" in text
    cfg2, created2 = candidates.load_or_create(tmp_path, decision, PRICING)
    assert not created2 and cfg2 == cfg


# --- checks library ----------------------------------------------------------------

def test_basic_checks():
    assert checks.max_words(3).run("", "one two three")[0]
    assert not checks.max_words(2).run("", "one two three")[0]
    assert not checks.not_empty().run("", "  ")[0]
    assert not checks.no_preamble().run("", "Sure! Here you go")[0]
    assert checks.no_preamble().run("", "You need to buy milk.")[0]
    assert not checks.no_markdown().run("", "## Summary\n- a")[0]
    assert checks.max_sentences(2).run("", "One. Two.")[0]
    assert not checks.forbid_regex(r"as an ai").run("", "As an AI, I")[0]
    assert checks.json_object(["a"]).run("", '{"a": 1}')[0]
    assert not checks.json_object(["b"]).run("", '{"a": 1}')[0]


def test_covers_inputs():
    c = checks.covers_inputs(0.6)
    notes = "- buy milk\n- call Sam about launch\n- book dentist"
    assert c.run(notes, "Buy milk, call Sam and book the dentist.")[0]
    ok, why = c.run(notes, "Buy milk.")
    assert not ok and "1/3" in why
    assert c.run("no list here", "anything")[0]


def test_validate_args_rejects_unknown_and_bad_regex():
    assert checks.validate_args("max_words", {"n": "50"}) == {"n": 50}
    with pytest.raises(ValueError):
        checks.validate_args("rm_rf", {})
    with pytest.raises(ValueError):
        checks.validate_args("max_words", {})
    with pytest.raises(Exception):
        checks.validate_args("forbid_regex", {"pattern": "("})


# --- verifiers ------------------------------------------------------------------------

SITE = CallSite(file="app.py", line=3, snippet="model='gpt-4o'", task="Summarize notes.",
                system_prompt="Summarize.", user_message_format="- note", rubric=["concise"])


def caller_for(client, max_cost=1.0):
    return Caller(client, SpendGuard(max_cost), Tracer())


def test_generated_verifier_file_is_reviewable_and_loads(tmp_path, fake_client):
    c = caller_for(fake_client)
    v = generate(c.analyst("anthropic/claude-opus-5.5"), tmp_path, {"id": 10, "label": "gpt-4o"},
                 SITE, {"goal": "Notes app", "constraints": ["Budget: $25"]}, "opus")
    text = v.path.read_text()
    assert "delete_files" not in text                       # outside the library: dropped
    assert "max_words(n=60" in text and "RUBRIC_CHECK = llm_rubric(" in text
    assert v.names() == ["not_empty", "max_words", "covers_inputs", "no_preamble", "llm_rubric"]
    assert load(v.path).sha == v.sha


def test_verify_skips_llm_when_deterministic_check_fails(fake_client):
    v = Verifiers([checks.max_words(3)], checks.llm_rubric("no invented facts"), None, "x")
    r = verify(caller_for(fake_client), "meta-llama/llama-4-scout", v, "- a", "one two three four")
    assert not r.passed and r.failed == ["max_words"]
    assert fake_client.calls == []
    r = verify(caller_for(fake_client), "meta-llama/llama-4-scout", v, "- a", "INVENTED fact")
    assert not r.passed and r.failed == ["llm_rubric"] and r.cost > 0


# --- spend guard ----------------------------------------------------------------------

def test_spend_guard_reserves_worst_case():
    g = SpendGuard(0.10)
    g.reserve(0.06)
    with pytest.raises(BudgetExceeded):
        g.reserve(0.05)
    assert g.stopped
    g.settle(0.06, 0.01)
    assert g.spent == pytest.approx(0.01) and g.reserved == pytest.approx(0)


def test_caller_refuses_call_over_cap(fake_client):
    c = caller_for(fake_client, max_cost=1e-9)
    with pytest.raises(BudgetExceeded):
        c.chat("openai/gpt-4o", [{"role": "user", "content": "hi"}], span="x", role="candidate")
    assert fake_client.calls == []


# --- stats + verdict ---------------------------------------------------------------------

def test_bootstrap_is_deterministic_and_brackets_mean():
    scores = [5, 4, 5, 3, 4, 5, 4, 4, 5, 3]
    mean, lo, hi = bootstrap_ci(scores, seed=1)
    assert (mean, lo, hi) == bootstrap_ci(scores, seed=1)
    assert lo <= mean <= hi and mean == pytest.approx(4.2)
    assert bootstrap_ci([4.0]) == (4.0, 4.0, 4.0)


@pytest.mark.parametrize("text,value", [("500", 500), ("1,200 users", 1200), ("$25", 25),
                                        ("1.5k", 1500), ("2M", 2e6), ("", None), ("lots", None)])
def test_parse_number(text, value):
    assert parse_number(text) == value


def S(model, setup, mean, pass_rate, cost, lo=None, hi=None, scored=20, errors=0):
    return SetupStats(model=model, setup=setup, name=f"{model}/{setup}", calls=20, errors=errors,
                      scored=scored, mean=mean, lo=lo if lo is not None else mean - 0.2,
                      hi=hi if hi is not None else mean + 0.2, pass_rate=pass_rate,
                      pass_lo=pass_rate - 0.1, pass_hi=pass_rate + 0.1, avg_attempts=1,
                      avg_latency_s=1, avg_cost=cost)


def stats_of(*items):
    return {(s.model, s.setup): s for s in items}


def test_verdict_switches_to_cheaper_equivalent():
    v = decide(stats_of(S("big", ALONE, 4.5, 0.9, 0.01), S("small", ALONE, 4.4, 0.9, 0.001)), "big")
    assert v.kind == "switch" and v.recommended == ("small", ALONE)


def test_verdict_keeps_when_cheaper_is_worse():
    v = decide(stats_of(S("big", ALONE, 4.5, 0.9, 0.01), S("small", ALONE, 3.0, 0.9, 0.001)), "big")
    assert v.kind == "keep" and "meaningfully below" in v.rejected["small/alone"]


def test_verdict_recommends_cheaper_model_plus_verifier():
    v = decide(stats_of(
        S("big", ALONE, 4.5, 0.95, 0.01),
        S("small", ALONE, 4.4, 0.70, 0.001),          # cheaper but fails verifiers too often
        S("small", VERIFIED, 4.5, 0.95, 0.002)), "big")
    assert v.kind == "switch" and v.recommended == ("small", VERIFIED)
    assert "pass rate" in v.rejected["small/alone"]


def test_verdict_not_enough_evidence_when_only_intervals_overlap():
    v = decide(stats_of(S("big", ALONE, 4.5, 0.9, 0.01, lo=3.5, hi=5.0),
                        S("small", ALONE, 4.0, 0.9, 0.001, lo=3.2, hi=4.6)), "big")
    assert v.kind == "insufficient" and v.recommended == ("small", ALONE)


def test_verdict_not_enough_evidence_with_few_scores():
    v = decide(stats_of(S("big", ALONE, 4.5, 0.9, 0.01, scored=4)), "big")
    assert v.kind == "insufficient"
