"""Shared fakes: a deterministic stand-in for the OpenRouter API (no network, no spend)."""
import json
import re

import pytest

from notbydefault.openrouter import ChatResult, pricing_from_models


def model(mid, prompt_per_m, completion_per_m, created, **extra):
    return {"id": mid, "created": created,
            "pricing": {"prompt": str(prompt_per_m / 1e6), "completion": str(completion_per_m / 1e6)},
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "supported_parameters": ["temperature", "max_tokens"], **extra}


MODELS = [
    model("openai/gpt-4o", 2.5, 10, 100),
    model("openai/gpt-4o-mini", 0.15, 0.6, 101),            # same provider: never an alternative
    model("anthropic/claude-haiku-5.5", 0.1, 0.5, 300),
    model("anthropic/claude-haiku-4.5", 0.08, 0.4, 200),     # older than haiku-5.5
    model("anthropic/claude-haiku-5.5:batch", 0.05, 0.25, 301),
    model("anthropic/claude-opus-5.5", 4, 20, 302),          # judge
    model("anthropic/claude-sonnet-4.5", 3, 15, 150),
    model("google/gemini-3.8-flash", 0.75, 3.75, 400),
    model("google/gemini-guard-1", 0.01, 0.01, 401),         # specialty: excluded
    model("meta-llama/llama-4-scout", 0.1, 0.3, 500),
    model("x-ai/grok-4.7", 2, 6, 600),                       # not cheap enough
]

GOOD_ITEMS_RX = re.compile(r"^\s*-\s+(.+)$", re.M)


def concise(user_message: str) -> str:
    items = GOOD_ITEMS_RX.findall(user_message)
    return "You need to " + "; ".join(items) + "."


class FakeClient:
    """Routes each request by its system prompt, like the real pipeline's call types."""

    def __init__(self, models=MODELS, n_samples=20):
        self._pricing = pricing_from_models(models, fetched_at="2026-10-10T00:00:00+00:00")
        self.calls = []          # (model, kind)
        self.n_samples = n_samples

    def pricing(self):
        return self._pricing

    def chat(self, model, messages, temperature=0.0, max_tokens=1024, json_mode=False):
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        user = messages[-1]["content"]
        if "analyse application code" in system:
            kind, text = "extract", json.dumps({
                "task": "Summarize a user's notes in a few sentences.",
                "system_prompt": "Summarize the user's notes in a few concise sentences.",
                "user_message_format": "one '- note' line per note",
                "input_description": "short to-do style notes",
                "rubric": ["covers every note", "concise", "no invented facts"]})
        elif "realistic test data" in system:
            kind = "samples"
            text = json.dumps({"inputs": [f"- buy milk {i}\n- call Sam about launch {i}\n"
                                          f"- book dentist appointment" for i in range(self.n_samples)]})
        elif "design automatic output checks" in system:
            kind, text = "verifiers", json.dumps({
                "checks": [
                    {"check": "not_empty", "args": {}, "why": "must answer"},
                    {"check": "max_words", "args": {"n": 60}, "why": "a quick glance"},
                    {"check": "covers_inputs", "args": {"min_fraction": 0.6}, "why": "every note"},
                    {"check": "delete_files", "args": {"path": "/"}, "why": "not in the library"},
                    {"check": "no_preamble", "args": {}, "why": "straight to the point"}],
                "rubric_check": {"criterion": "No facts absent from the input.", "why": "trust"}})
        elif "check one output" in system:
            kind = "rubric"
            output = user.split("Output to check:", 1)[1]
            ok = "INVENTED" not in output
            text = json.dumps({"pass": ok, "reason": "ok" if ok else "mentions a CEO meeting"})
        elif "impartial evaluator" in system:
            kind = "judge"
            scores = {}
            for label, out in re.findall(r"Response ([A-Z]):\n<<<\n(.*?)\n>>>", user, re.S):
                s = 2 if "INVENTED" in out else 3 if "commentary" in out else 5
                scores[label] = {"score": s, "reason": "fake"}
            text = json.dumps({"scores": scores})
        else:
            kind = "candidate"
            original = next(m["content"] for m in messages if m["role"] == "user")
            retry = "failed an automatic check" in user
            base = concise(original)
            if model.startswith("google/"):
                text = base if retry else base + " This is additional commentary." * 20
            elif model.startswith("meta-llama/"):
                text = base + " Also an INVENTED meeting with the CEO."
            else:
                text = base
        self.calls.append((model, kind))
        tin = sum(len(m["content"]) for m in messages) // 4
        tout = len(text) // 4 + 1
        return ChatResult(model=model, text=text, input_tokens=tin, output_tokens=tout,
                          latency_s=0.01, cost=self._pricing.price(model).cost(tin, tout),
                          finish_reason="stop")


@pytest.fixture(autouse=True)
def no_telemetry(monkeypatch):
    """Tests never send spans to neatlogs."""
    monkeypatch.setenv("NOTBYDEFAULT_TELEMETRY", "0")


@pytest.fixture
def fake_client():
    return FakeClient()
