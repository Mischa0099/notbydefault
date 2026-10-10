"""Minimal OpenRouter client (OpenAI-compatible chat API + public models/pricing API)."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone

API_BASE = "https://openrouter.ai/api/v1"
MODELS_URL = f"{API_BASE}/models"
CHAT_URL = f"{API_BASE}/chat/completions"


class LLMError(RuntimeError):
    pass


@dataclass
class Price:
    """USD per token, as published by OpenRouter's models API."""
    prompt: float
    completion: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return input_tokens * self.prompt + output_tokens * self.completion


@dataclass
class Pricing:
    prices: dict[str, Price]
    source_url: str
    fetched_at: str  # ISO 8601, UTC
    models: list[dict] = field(default_factory=list)

    def price(self, model: str) -> Price:
        if model not in self.prices:
            raise LLMError(f"model {model!r} not found in {self.source_url}")
        return self.prices[model]


@dataclass
class ChatResult:
    model: str
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost: float
    finish_reason: str | None = None
    reported_cost: float | None = None


def pricing_from_models(models: list[dict], source_url: str = MODELS_URL,
                        fetched_at: str | None = None) -> Pricing:
    prices = {}
    for m in models:
        p = m.get("pricing") or {}
        try:
            prices[m["id"]] = Price(float(p["prompt"]), float(p["completion"]))
        except (KeyError, TypeError, ValueError):
            continue
    return Pricing(prices, source_url,
                   fetched_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   models)


class OpenRouterClient:
    def __init__(self, api_key: str, timeout: float = 120.0, retries: int = 3):
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self._pricing: Pricing | None = None

    def _request(self, url: str, body: dict | None = None) -> dict:
        headers = {"Content-Type": "application/json", "X-Title": "NotByDefault",
                   "HTTP-Referer": "https://github.com/Mischa0099/notbydefault"}
        if body is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(body).encode() if body is not None else None
        last: Exception | None = None
        for attempt in range(self.retries):
            req = urllib.request.Request(url, data=data, headers=headers,
                                         method="POST" if body is not None else "GET")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")[:500]
                last = LLMError(f"HTTP {e.code} from {url}: {detail}")
                if e.code not in (408, 429, 500, 502, 503, 504):
                    break
            except (urllib.error.URLError, TimeoutError) as e:
                last = LLMError(f"network error calling {url}: {e}")
            time.sleep(2 ** attempt)
        raise last  # type: ignore[misc]

    def pricing(self) -> Pricing:
        """Live prices from OpenRouter's models API (fetched once per client)."""
        if self._pricing is None:
            self._pricing = pricing_from_models(self._request(MODELS_URL)["data"])
        return self._pricing

    def chat(self, model: str, messages: list[dict], temperature: float = 0.0,
             max_tokens: int = 1024, json_mode: bool = False) -> ChatResult:
        body: dict = {"model": model, "messages": messages, "temperature": temperature,
                      "max_tokens": max_tokens, "usage": {"include": True}}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        start = time.perf_counter()
        data = self._request(CHAT_URL, body)
        latency = time.perf_counter() - start
        if "error" in data:
            raise LLMError(f"{model}: {data['error']}")
        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage") or {}
        tin, tout = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        return ChatResult(
            model=model,
            text=(choice.get("message") or {}).get("content") or "",
            input_tokens=tin,
            output_tokens=tout,
            latency_s=latency,
            cost=self.pricing().price(model).cost(tin, tout),
            finish_reason=choice.get("finish_reason"),
            reported_cost=usage.get("cost"),
        )


def parse_json(text: str):
    """Parse a JSON object/array from an LLM reply, tolerating code fences and chatter."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_, close in (("{", "}"), ("[", "]")):
        i, j = text.find(open_), text.rfind(close)
        if i != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"could not parse JSON from model reply: {text[:200]!r}")
