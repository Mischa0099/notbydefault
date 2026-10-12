import json

import pytest

from conftest import CATALOG, CHOSEN, JUDGE, SMALL_OTHER, SMALL_SAME, FakeTransport, make_client
from notbydefault.fit import candidates, checks
from notbydefault.fit.callsite import CallSite
from notbydefault.fit.evaluate import ALONE, VERIFIED, BudgetExceeded, Caller, SpendGuard
from notbydefault.fit.providers import RateLimiter, load_catalog
from notbydefault.fit.stats import SetupStats, bootstrap_ci, decide, parse_number
from notbydefault.fit.tracing import Tracer
from notbydefault.fit.verifiers import Verifiers, generate, load, verify
from notbydefault.llm import DailyLimitReached, Price, classify_429, parse_json

PRICING = make_client().pricing()


# --- llm helpers ------------------------------------------------------------------

def test_price_cost_is_per_token():
    assert Price(2.5e-6, 10e-6).cost(1000, 100) == pytest.approx(0.0035)
    p = PRICING.price(CHOSEN)
    assert p.prompt == pytest.approx(2.5e-6)
    assert (p.source_url, p.fetched_at) == ("https://groq.test/models", "2026-10-10")


@pytest.mark.parametrize("text", ['{"a": 1}', '```json\n{"a": 1}\n```', 'Sure! {"a": 1} done'])
def test_parse_json_tolerates_chatter(text):
    assert parse_json(text) == {"a": 1}


GEMINI_DAILY = json.dumps([{"error": {"code": 429, "message": "You exceeded your current quota",
    "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
        {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "40s"}]}}])
GEMINI_MINUTE = GEMINI_DAILY.replace("PerDay", "PerMinute")
GEMINI_NOT_FREE = json.dumps([{"error": {"code": 429, "message":
    "Quota exceeded for metric: generate_content_free_tier_requests, limit: 0, model: gemini-3.1-pro",
    "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "29954s"}]}}])
GROQ_DAILY = json.dumps({"error": {"message": "Rate limit reached for model `openai/gpt-oss-120b` "
    "on requests per day (RPD): Limit 1000, Used 1000", "type": "requests", "code": "rate_limit_exceeded"}})
GROQ_MINUTE = json.dumps({"error": {"message": "Rate limit reached for model `openai/gpt-oss-120b` "
    "on tokens per minute (TPM): Limit 8000, Used 7900", "code": "rate_limit_exceeded"}})


@pytest.mark.parametrize("body,headers,kind,wait", [
    (GEMINI_DAILY, {}, "daily", 40),
    (GEMINI_MINUTE, {}, "minute", 40),
    (GEMINI_NOT_FREE, {}, "unavailable", 29954),
    (GROQ_DAILY, {"retry-after": "3600"}, "daily", 3600),
    (GROQ_MINUTE, {"retry-after": "7"}, "minute", 7),
    ("not json", {"retry-after": "1m26.4s"}, "minute", 86.4),
])
def test_classify_429(body, headers, kind, wait):
    assert classify_429(body, headers) == (kind, pytest.approx(wait))


# --- providers: catalog, throttle, cache --------------------------------------------

def test_shipped_catalog_has_free_groq_and_gemini_with_price_sources():
    cat = load_catalog()["providers"]
    assert cat["groq"]["base_url"] == "https://api.groq.com/openai/v1"
    assert cat["gemini"]["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert cat["groq"]["tier"] == cat["gemini"]["tier"] == "free"
    assert cat["openrouter"]["enabled"] is False
    for p in ("groq", "gemini"):
        assert cat[p]["prices_source"].startswith("https://") and cat[p]["prices_fetched"]


def test_project_catalog_override_merges(tmp_path):
    (tmp_path / ".notbydefault").mkdir()
    (tmp_path / ".notbydefault" / "providers.yaml").write_text(
        "providers:\n  groq:\n    models:\n      openai/gpt-oss-20b: {rpm: 2}\n")
    m = load_catalog(tmp_path)["providers"]["groq"]["models"]["openai/gpt-oss-20b"]
    assert m["rpm"] == 2 and m["family"] == "gpt-oss"


def test_rate_limiter_waits_for_rpm_and_tpm():
    now = [0.0]
    waits = []

    def sleep(s):
        waits.append(s)
        now[0] += s
    lim = RateLimiter(rpm=2, tpm=1000, sleep=sleep, clock=lambda: now[0])
    lim.acquire(100)
    lim.acquire(100)
    lim.acquire(100)                   # third request in the same minute must wait
    assert waits and sum(waits) == pytest.approx(60)
    lim2 = RateLimiter(rpm=100, tpm=1000, sleep=sleep, clock=lambda: now[0])
    waits.clear()
    lim2.acquire(900)
    lim2.acquire(200)                  # token budget exceeded: waits
    assert waits


def test_cache_means_a_request_is_never_made_twice(tmp_path):
    client = make_client(cache_dir=tmp_path)
    msgs = [{"role": "user", "content": "- a\n- b"}]
    first = client.chat(CHOSEN, msgs)
    second = client.chat(CHOSEN, msgs)
    assert not first.cached and second.cached and second.text == first.text
    assert second.cost == pytest.approx(first.cost)          # list-price estimate either way
    assert len(client.transport.calls) == 1
    client.chat(CHOSEN, msgs, max_tokens=99)                  # different settings: new request
    assert len(client.transport.calls) == 2
    fresh = make_client(cache_dir=tmp_path)                   # survives restarts
    assert fresh.chat(CHOSEN, msgs).cached and fresh.transport.calls == []


def test_free_tier_detection():
    client = make_client()
    assert client.free_tier([CHOSEN, JUDGE])
    assert not client.free_tier([CHOSEN, "paidco:tiny-1"])


# --- candidates -------------------------------------------------------------------

@pytest.mark.parametrize("literal,expected", [
    ("openai/gpt-oss-120b", CHOSEN),
    ("gpt-oss-120b", CHOSEN),
    ("gemini-3.8-flash", JUDGE),
    (CHOSEN, CHOSEN),
])
def test_resolve_model_id(literal, expected):
    assert candidates.resolve_model_id(literal, PRICING) == expected


def test_resolve_unknown_model_raises():
    with pytest.raises(LookupError):
        candidates.resolve_model_id("gpt-99-ultra", PRICING)


def test_alternatives_are_small_free_models_of_same_and_other_family():
    assert candidates.pick_alternatives(CHOSEN, PRICING) == [SMALL_SAME, SMALL_OTHER]


def test_paid_models_are_never_picked():
    assert "paidco:tiny-1" not in candidates.pick_alternatives(CHOSEN, PRICING, n=5)
    assert candidates.pick_judge([], PRICING) != "paidco:tiny-1"


def test_judge_is_a_strong_gemini_not_a_candidate():
    assert candidates.pick_judge([CHOSEN, SMALL_SAME, SMALL_OTHER], PRICING) == JUDGE
    assert candidates.pick_judge([JUDGE], PRICING) != JUDGE


def test_candidates_yaml_created_once_with_price_source(tmp_path):
    decision = {"id": 10, "label": "openai/gpt-oss-120b"}
    cfg, created = candidates.load_or_create(tmp_path, decision, PRICING)
    assert created and cfg["chosen"] == CHOSEN
    assert cfg["candidates"] == [CHOSEN, SMALL_SAME, SMALL_OTHER]
    assert cfg["judge"] == JUDGE and cfg["verifier_model"] == SMALL_SAME   # cheapest candidate
    assert cfg["max_retries"] == 2
    text = candidates.config_path(tmp_path).read_text()
    assert "price_source: https://groq.test/models" in text and "price_fetched: '2026-10-10'" in text
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
    v = generate(c.analyst(JUDGE), tmp_path, {"id": 10, "label": "gpt-4o"},
                 SITE, {"goal": "Notes app", "constraints": ["Budget: $25"]}, "opus")
    text = v.path.read_text()
    assert "delete_files" not in text                       # outside the library: dropped
    assert "max_words(n=60" in text and "RUBRIC_CHECK = llm_rubric(" in text
    assert v.names() == ["not_empty", "max_words", "covers_inputs", "no_preamble", "llm_rubric"]
    assert load(v.path).sha == v.sha


def test_verify_skips_llm_when_deterministic_check_fails(fake_client):
    v = Verifiers([checks.max_words(3)], checks.llm_rubric("no invented facts"), None, "x")
    r = verify(caller_for(fake_client), SMALL_SAME, v, "- a", "one two three four")
    assert not r.passed and r.failed == ["max_words"]
    assert fake_client.transport.calls == []
    r = verify(caller_for(fake_client), SMALL_SAME, v, "- a", "INVENTED fact")
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
        c.chat(CHOSEN, [{"role": "user", "content": "hi"}], span="x", role="candidate")
    assert fake_client.transport.calls == []


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
    v = decide(stats_of(S("big", ALONE, 4.5, 0.9, 0.01, scored=3)), "big")
    assert v.kind == "insufficient"


# --- transport recoveries -----------------------------------------------------------

class _Resp:
    def __init__(self, body):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_error(code, body, headers=None):
    import io
    import urllib.error
    from email.message import Message
    msg = Message()
    for k, v in (headers or {}).items():
        msg[k] = v
    return urllib.error.HTTPError("https://x.test", code, "err", msg, io.BytesIO(body.encode()))


def test_transport_backs_off_on_429_and_logs_the_reason(monkeypatch):
    import urllib.request
    from notbydefault.llm import OpenAICompatClient
    ok = json.dumps({"choices": [{"message": {"content": "hi"}}], "usage": {}}).encode()
    answers = [_http_error(429, GROQ_MINUTE, {"retry-after": "3"}),
               _http_error(503, "overloaded"), _Resp(ok)]

    def fake_urlopen(req, timeout):
        a = answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    events, sleeps = [], []
    c = OpenAICompatClient("groq", "https://x.test", "k", sleep=sleeps.append,
                           on_event=lambda *a, **k: events.append((a, k)))
    data, _ = c.complete("openai/gpt-oss-120b", [{"role": "user", "content": "x"}])
    assert data["choices"][0]["message"]["content"] == "hi"
    assert [e[0][0] for e in events] == ["rate_limit_backoff", "retry_failed_call"]
    assert all(e[0][1] == "recovery" for e in events)
    assert "429" in events[0][0][2] and "3.5s" in events[0][0][2]
    assert "HTTP 503" in events[1][0][2]
    assert sleeps[0] == pytest.approx(3.5)


@pytest.mark.parametrize("code,body,unavailable", [
    (429, GROQ_DAILY, False),
    (429, GEMINI_DAILY, False),
    (429, GEMINI_NOT_FREE, True),
    (404, '{"error": {"message": "no longer available to new users"}}', True),
])
def test_transport_daily_and_unavailable_are_not_retried(monkeypatch, code, body, unavailable):
    import urllib.request
    from notbydefault.llm import ModelUnavailable, OpenAICompatClient
    tries = []

    def fake_urlopen(req, timeout):
        tries.append(1)
        raise _http_error(code, body)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    c = OpenAICompatClient("gemini", "https://x.test", "k", sleep=lambda s: None)
    with pytest.raises(ModelUnavailable if unavailable else DailyLimitReached):
        c.complete("m", [{"role": "user", "content": "x"}])
    assert len(tries) == 1


def test_402_prepaid_credits_means_unavailable(monkeypatch):
    import urllib.request
    from notbydefault.llm import ModelUnavailable, OpenAICompatClient
    tries = []

    def fake_urlopen(req, timeout):
        tries.append(1)
        raise _http_error(402, '[{"error": {"message": "Your prepayment credits are depleted."}}]')
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    c = OpenAICompatClient("gemini", "https://x.test", "k", sleep=lambda s: None)
    with pytest.raises(ModelUnavailable, match="402"):
        c.complete("gemini-3.6-flash", [{"role": "user", "content": "x"}])
    assert len(tries) == 1


def test_hung_model_is_retried_once_then_reported(monkeypatch):
    import urllib.request
    from notbydefault.llm import LLMError, OpenAICompatClient
    tries, events = [], []

    def fake_urlopen(req, timeout):
        tries.append(1)
        raise TimeoutError("The read operation timed out")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    c = OpenAICompatClient("gemini", "https://x.test", "k", sleep=lambda s: None,
                           on_event=lambda *a, **k: events.append(a))
    with pytest.raises(LLMError, match="timed out"):
        c.complete("gemini-3.8-flash", [{"role": "user", "content": "x"}])
    assert len(tries) == 2                                   # not 6 x 120 s
    assert [e[0] for e in events] == ["retry_failed_call"] and "timed out" in events[0][2]


def test_catalog_unavailable_flag_keeps_model_out_of_selection():
    import copy
    cat = copy.deepcopy(CATALOG)
    cat["providers"]["gemini"]["models"]["gemini-3.8-flash"]["available"] = False
    pricing = make_client(catalog=cat).pricing()
    assert candidates.pick_judge([CHOSEN, SMALL_SAME], pricing) == SMALL_OTHER   # not JUDGE


def _without_gemini(extra_groq=None):
    import copy
    cat = copy.deepcopy(CATALOG)
    for m in cat["providers"]["gemini"]["models"].values():
        m["available"] = False
    cat["providers"]["groq"]["models"].update(extra_groq or {})
    return make_client(catalog=cat).pricing()


def test_judge_falls_back_to_strongest_non_qwen_when_no_gemini_works():
    pricing = _without_gemini({"llama-4-maverick": {"family": "llama", "size": "large",
                                                    "strength": 2, "in": 0.2, "out": 0.6}})
    assert candidates.pick_judge([CHOSEN, SMALL_SAME], pricing) == "groq:llama-4-maverick"


def test_qwen_is_never_the_fallback_judge():
    from conftest import SWAP_IN
    pricing = _without_gemini()      # only qwen is left that isn't excluded
    with pytest.raises(LookupError):
        candidates.pick_judge([CHOSEN, SMALL_SAME, SWAP_IN], pricing)


def test_rejection_reason_says_when_a_pricier_setup_is_clearly_better():
    v = decide(stats_of(S("big", ALONE, 3.25, 1.0, 0.0001, lo=2.75, hi=3.62),
                        S("small", ALONE, 3.5, 1.0, 0.00007, lo=3.12, hi=3.88),
                        S("lite", ALONE, 4.75, 1.0, 0.00011, lo=4.38, hi=5.0)), "big")
    assert v.recommended == ("small", ALONE)
    assert v.rejected["lite/alone"].startswith("higher quality 4.75")


def test_a_request_that_never_completes_times_out_on_the_wall_clock(monkeypatch):
    """urlopen's socket timeout doesn't cover DNS or a trickling server: bound the whole call."""
    import threading
    import time
    import urllib.request
    from notbydefault import llm
    from notbydefault.llm import LLMError, OpenAICompatClient
    release = threading.Event()

    def stuck_urlopen(req, timeout):
        release.wait(30)              # ignores its socket timeout
        raise TimeoutError("late")
    monkeypatch.setattr(urllib.request, "urlopen", stuck_urlopen)
    monkeypatch.setattr(llm, "REQUEST_SLACK_S", 0.2)
    c = OpenAICompatClient("groq", "https://x.test", "k", timeout=0.2, sleep=lambda s: None)
    start = time.monotonic()
    with pytest.raises(LLMError, match="no complete response"):
        c.complete("m", [{"role": "user", "content": "x"}])
    release.set()
    assert time.monotonic() - start < 5


def test_retries_stop_at_the_call_budget(monkeypatch):
    import urllib.request
    from notbydefault.llm import LLMError, OpenAICompatClient
    tries = []

    def failing_urlopen(req, timeout):
        tries.append(1)
        raise _http_error(503, "overloaded")
    monkeypatch.setattr(urllib.request, "urlopen", failing_urlopen)
    clock = iter(range(0, 1000, 50))
    monkeypatch.setattr("notbydefault.llm.time.monotonic", lambda: next(clock))
    c = OpenAICompatClient("groq", "https://x.test", "k", timeout=40, retries=6,
                           sleep=lambda s: None, call_budget=120)
    with pytest.raises(LLMError, match="gave up after 120s"):
        c.complete("m", [{"role": "user", "content": "x"}])
    assert len(tries) < 6


def test_cache_tolerates_concurrent_writes_of_the_same_key(tmp_path):
    """Identical requests from two worker threads put the same key at once (Windows rename)."""
    from concurrent.futures import ThreadPoolExecutor
    from notbydefault.fit.providers import ResponseCache
    cache = ResponseCache(tmp_path)
    key = cache.key("groq:m", [{"role": "user", "content": "x"}], 0.0, 10, False)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda i: cache.put(key, {"text": "same", "i": 0}), range(64)))
    assert cache.get(key)["text"] == "same"
    assert not list(tmp_path.rglob("*.tmp"))

def test_net_saving_line_from_the_unrounded_numbers():
    from notbydefault.fit.report import net_saving_line
    # decision #21's demo fit check, as stored in its results file
    assert net_saving_line(407.35499999999996, 345.075, 0.0375157) == (
        "Net saving: $407.35/mo → $345.07/mo = $62.28/mo saved. Check cost once at "
        "$0.0375 (list). First-month ROI: $62.24.")
    assert net_saving_line(300.0, 345.07, 0.04) == ""        # recommended costs more: no saving
    assert net_saving_line(None, 345.07, 0.04) == ""
