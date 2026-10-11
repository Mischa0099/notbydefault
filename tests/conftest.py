"""Shared fakes: a deterministic stand-in for provider HTTP APIs (no network, no spend).

FakeTransport replaces OpenAICompatClient inside the real ProviderSet, so routing, caching,
throttling and daily-limit handling are exercised exactly as in production.
"""
import json
import re

import pytest

from notbydefault.fit.providers import ProviderSet
from notbydefault.llm import DailyLimitReached, ModelUnavailable


def _m(family, size, strength, inp, out, **extra):
    return {"family": family, "size": size, "strength": strength, "in": inp, "out": out, **extra}


CATALOG = {"roles": {"pivot": "groq:openai/gpt-oss-20b", "stale_judge": "groq:openai/gpt-oss-120b",
                     "why": "groq:openai/gpt-oss-120b", "scout": "groq:openai/gpt-oss-120b"},
           "providers": {
    "groq": {"base_url": "https://groq.test/v1", "key_env": "FAKE_GROQ_KEY", "tier": "free",
             "data_residency": ["US"], "data_residency_source": "https://groq.test/your-data",
             "prices_source": "https://groq.test/models", "prices_fetched": "2026-10-10",
             "models": {
                 "openai/gpt-oss-120b": _m("gpt-oss", "large", 2, 2.5, 10.0),
                 "openai/gpt-oss-20b": _m("gpt-oss", "small", 1, 0.075, 0.30),
                 "qwen/qwen3.8-27b": _m("qwen", "medium", 1, 0.20, 0.80),   # second judge
                 "llama-3.1-8b-instant": _m("llama", "small", 0, 0.05, 0.08),  # swap target
             }},
    "gemini": {"base_url": "https://gemini.test/openai", "key_env": "FAKE_GEMINI_KEY",
               "tier": "free", "prices_source": "https://gemini.test/pricing",
               "prices_fetched": "2026-10-10",
               "models": {
                   "gemini-3.8-flash": _m("gemini", "large", 3, 0.75, 3.75),
                   "gemini-3.5-flash-lite": _m("gemini", "small", 1, 0.10, 0.40),
               }},
    "paidco": {"base_url": "https://paid.test", "key_env": "FAKE_PAID_KEY", "tier": "paid",
               "prices_source": "https://paid.test/pricing", "prices_fetched": "2026-10-10",
               "models": {"tiny-1": _m("tiny", "small", 9, 0.01, 0.01)}},   # never auto-picked
}}

CHOSEN = "groq:openai/gpt-oss-120b"
SMALL_SAME = "groq:openai/gpt-oss-20b"
SMALL_OTHER = "gemini:gemini-3.5-flash-lite"
JUDGE = "gemini:gemini-3.8-flash"
SWAP_IN = "groq:llama-3.1-8b-instant"
JUDGE2 = "groq:qwen/qwen3.8-27b"

GOOD_ITEMS_RX = re.compile(r"^\s*-\s+(.+)$", re.M)


def concise(user_message: str) -> str:
    return "You need to " + "; ".join(GOOD_ITEMS_RX.findall(user_message)) + "."


class FakeTransport:
    """Answers like each call type in the pipeline, keyed by system prompt and model."""

    def __init__(self, n_samples=8, daily_limit_after=None, unavailable=()):
        self.calls = []                       # (model, kind)
        self.n_samples = n_samples
        self.daily_limit_after = daily_limit_after
        self.unavailable = set(unavailable)   # native model ids answering "limit: 0"

    def respond(self, model, messages):
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        user = messages[-1]["content"]
        if "analyse application code" in system:
            return "extract", json.dumps({
                "task": "Summarize a user's notes in a few sentences.",
                "system_prompt": "Summarize the user's notes in a few concise sentences.",
                "user_message_format": "one '- note' line per note",
                "input_description": "short to-do style notes",
                "rubric": ["covers every note", "concise", "no invented facts"]})
        if user == "Reply with the single word: ok":
            return "probe", "ok"
        if system.startswith("You review messages a developer sends"):
            said = user.split("The developer just wrote:", 1)[1].lower()
            pivot = any(w in said for w in ("gpt-4o", "postgres"))
            view = json.loads(user.split("The project's Decision Graph:", 1)[1]
                              .split("Price facts", 1)[0])
            return "pivot", json.dumps({
                "is_pivot": pivot, "change": "use gpt-4o for action items" if pivot else "",
                "decision_ids": [d["id"] for d in view["decisions"] if d["kind"] == "model"] if pivot else [],
                "conflicts": [{"node_id": n["id"], "why": "gpt-4o costs more than the budget allows"}
                              for n in view["requirements_and_constraints"]
                              if "budget" in n["label"].lower()] if pivot else []})
        if system.startswith("You answer a developer's question"):
            bad = "invent" in user.split("Facts", 1)[0].lower()
            return "why", json.dumps({"answer": "It was chosen in 2019 by the CTO [F99]." if bad
                                      else "The fit test kept it [F1].", "cited": ["F1"]})
        if system.startswith("You extract edge cases"):
            return "scout", json.dumps({"cases": [
                {"input": "notes with an empty line", "why": "agent said blank notes crash it",
                 "evidence": "what about empty notes?", "already_tested": False},
                {"input": "a single note", "why": "covered", "evidence": "Blank lines might crash",
                 "already_tested": True}]})
        if system.startswith("You turn a product requirement"):
            text = user.split("Requirement:", 1)[1].splitlines()[0].lower()
            return "extract", json.dumps({"data_residency": "EU" if " eu " in f" {text} " else None,
                                          "max_monthly_cost_usd": None, "summary": "fake"})
        if system.startswith("You check whether an existing technical decision"):
            affected = "category: database" in user and "residency" in user.split("Machine-readable", 1)[0].lower()
            return "stale_judge", json.dumps({
                "affected": affected, "property": "stores customer data" if affected else "",
                "reason": "stores customer data outside the EU" if affected
                else "not related to where data is stored"})
        if "realistic test data" in system and "already exist" in user:
            n = int(re.search(r"Write (\d+) NEW", user).group(1))
            return "extend", json.dumps({"inputs": [
                f"- water plants {k}\n- email Ana re budget {k}" for k in range(n)]})
        if "realistic test data" in system:
            return "samples", json.dumps({"inputs": [
                f"- buy milk {i}\n- call Sam about launch {i}\n- book dentist appointment"
                for i in range(self.n_samples)]})
        if "design automatic output checks" in system:
            return "verifiers", json.dumps({
                "checks": [
                    {"check": "not_empty", "args": {}, "why": "must answer"},
                    {"check": "max_words", "args": {"n": 60}, "why": "a quick glance"},
                    {"check": "covers_inputs", "args": {"min_fraction": 0.6}, "why": "every note"},
                    {"check": "delete_files", "args": {"path": "/"}, "why": "not in the library"},
                    {"check": "no_preamble", "args": {}, "why": "straight to the point"}],
                "rubric_check": {"criterion": "No facts absent from the input.", "why": "trust"}})
        if "check one output" in system:
            ok = "INVENTED" not in user.split("Output to check:", 1)[1]
            return "rubric", json.dumps({"pass": ok, "reason": "ok" if ok else "invents a meeting"})
        if "impartial evaluator" in system:
            scores = {}
            for label, out in re.findall(r"Response ([A-Z]):\n<<<\n(.*?)\n>>>", user, re.S):
                s = 2 if "INVENTED" in out else 3 if "commentary" in out else 5
                scores[label] = {"score": s, "reason": "fake"}
            return "judge", json.dumps({"scores": scores})
        original = next(m["content"] for m in messages if m["role"] == "user")
        retry = "failed an automatic check" in user
        base = concise(original)
        if model == "gemini-3.5-flash-lite":           # rambles until told to fix it
            return "candidate", base if retry else base + " This is additional commentary." * 20
        if model == "openai/gpt-oss-20b":              # always invents a fact
            return "candidate", base + " Also an INVENTED meeting with the CEO."
        return "candidate", base

    def complete(self, model, messages, temperature=0.0, max_tokens=1024, json_mode=False,
                 extra_body=None):
        if model in self.unavailable:
            raise ModelUnavailable("fake", model, "free tier limit: 0")
        kind, text = self.respond(model, messages)
        if self.daily_limit_after is not None and len(self.calls) >= self.daily_limit_after:
            raise DailyLimitReached("groq", model, "Rate limit reached: requests per day (RPD)",
                                    "in 3h")
        self.calls.append((model, kind))
        tin = sum(len(m["content"]) for m in messages) // 4
        return ({"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": tin, "completion_tokens": len(text) // 4 + 1}}, 0.01)


def make_client(cache_dir=None, transport=None, catalog=CATALOG):
    t = transport or FakeTransport()
    client = ProviderSet(catalog, cache_dir=cache_dir, env={},
                         transports={"groq": t, "gemini": t, "paidco": t},
                         sleep=lambda s: None)
    client.transport = t
    return client


@pytest.fixture(autouse=True)
def no_telemetry(monkeypatch):
    """Tests never send spans to neatlogs."""
    monkeypatch.setenv("NOTBYDEFAULT_TELEMETRY", "0")
    monkeypatch.setenv("NOTBYDEFAULT_ENTIRE", "0")   # the real Entire CLI is never called
    from notbydefault import openrouter

    def offline(*a, **k):
        raise ConnectionError("network is disabled in tests (OpenRouter models API)")
    monkeypatch.setattr(openrouter, "fetch_models", offline)


@pytest.fixture
def fake_client():
    return make_client()


DEMO_STATE = (__import__("pathlib").Path(__file__).resolve().parents[1]
              / "demo-app" / ".notbydefault").resolve()


@pytest.fixture(autouse=True)
def never_touch_demo_state(monkeypatch):
    """Tests must never open the live demo graph or cache (a scheduled check may be using them)."""
    from pathlib import Path
    from notbydefault import graph as graph_mod
    from notbydefault.fit import providers as providers_mod
    real_graph, real_cache = graph_mod.Graph.__init__, providers_mod.ResponseCache.__init__

    def guard(path):
        p = Path(path).resolve()
        assert DEMO_STATE not in p.parents and p != DEMO_STATE, f"test touched {p}"

    def graph_init(self, path):
        guard(path)
        real_graph(self, path)

    def cache_init(self, folder):
        guard(folder)
        real_cache(self, folder)
    monkeypatch.setattr(graph_mod.Graph, "__init__", graph_init)
    monkeypatch.setattr(providers_mod.ResponseCache, "__init__", cache_init)
    try:    # the UI's read-only reader opens graph.db without Graph.__init__
        from notbydefault.ui import data as ui_data
    except ImportError:
        return
    real_ro = ui_data.ReadGraph.__init__

    def ro_init(self, path):
        guard(path)
        real_ro(self, path)
    monkeypatch.setattr(ui_data.ReadGraph, "__init__", ro_init)


# --- a fake Entire CLI ----------------------------------------------------------------

FIXTURES = __import__("pathlib").Path(__file__).resolve().parent / "fixtures"


class FakeEntireRunner:
    """Stands in for subprocess.run(["entire", ...]): canned stdout per command prefix.

    `responses` maps a tuple of leading args (after the exe) to stdout text, a callable
    (args -> stdout), an exception to raise, or an int exit code (failure). Unknown commands
    exit 1. The longest matching prefix wins.
    """

    def __init__(self, responses: dict | None = None):
        self.responses = dict(responses or {})
        self.calls: list[list[str]] = []

    def __call__(self, cmd, cwd=None, capture_output=True, timeout=None, **_):
        import subprocess
        args = list(cmd[1:])
        self.calls.append(args)
        best = None
        for prefix, value in self.responses.items():
            if tuple(args[:len(prefix)]) == prefix and (best is None or len(prefix) > len(best[0])):
                best = (prefix, value)
        value = best[1] if best else 1
        if callable(value) and not isinstance(value, type):
            value = value(args)
        if isinstance(value, BaseException):
            raise value
        if isinstance(value, int):
            return subprocess.CompletedProcess(cmd, value, b"",
                                               f"fake failure for {args[:2]}".encode())
        return subprocess.CompletedProcess(cmd, 0, value.encode("utf-8"), b"")


def demo_graph_responses() -> dict:
    """entire-graph output recorded on a throwaway copy of demo-app (2026-10-11)."""
    d = FIXTURES / "demo_graph"
    return {("plugin", "list"): (d / "plugin_list.txt").read_text(encoding="utf-8"),
            ("graph", "symbols"): (d / "symbols.ndjson").read_text(encoding="utf-8"),
            ("graph", "edges"): (d / "edges.ndjson").read_text(encoding="utf-8")}


def demo_project(path):
    """The demo app's source files in `path` (the graph fixtures index them)."""
    import shutil
    for name in ("app.py", "actions.py", "requirements.txt"):
        shutil.copy2(FIXTURES / "demo_graph" / name, path / name)
    return path
