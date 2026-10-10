"""OpenAI-compatible chat client shared by every provider (Groq, Gemini, OpenRouter, ...)."""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

MAX_INLINE_WAIT_S = 90   # a 429 asking us to wait longer than this is treated as a daily limit
MAX_TIMEOUT_RETRIES = 1  # timeouts are retried once (a model that hangs twice is reported)


class LLMError(RuntimeError):
    pass


class ModelUnavailable(LLMError):
    """The model can't be used with this key at all (HTTP 404/403, or a free-tier limit of 0)."""

    def __init__(self, provider: str, model: str, detail: str):
        self.provider, self.model, self.detail = provider, model, detail
        super().__init__(f"{provider}:{model} is unavailable: {detail}")


class DailyLimitReached(RuntimeError):
    """A quota that won't reset within minutes (daily limit, or not on the free tier)."""

    def __init__(self, provider: str, model: str, detail: str, resets: str = ""):
        self.provider, self.model, self.detail, self.resets = provider, model, detail, resets
        super().__init__(f"{provider}:{model}: {detail}" + (f" (resets {resets})" if resets else ""))


@dataclass
class Price:
    """USD per token (list price)."""
    prompt: float
    completion: float
    source_url: str = ""
    fetched_at: str = ""

    def cost(self, input_tokens: float, output_tokens: float) -> float:
        return input_tokens * self.prompt + output_tokens * self.completion


@dataclass
class Pricing:
    prices: dict[str, Price]
    source_url: str          # human-readable summary of where prices came from
    fetched_at: str
    models: list[dict] = field(default_factory=list)

    def price(self, model: str) -> Price:
        if model not in self.prices:
            raise LLMError(f"no price for model {model!r} ({self.source_url})")
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
    cached: bool = False


def _duration_s(text: str | None) -> float | None:
    """'2.5', '29954s', '1m26.4s', '7.66s' -> seconds."""
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    total, found = 0.0, False
    for value, unit in re.findall(r"([\d.]+)(ms|h|m|s)", text):
        found = True
        total += float(value) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total if found else None


def classify_429(body: str, headers) -> tuple[str, float | None]:
    """('minute' | 'daily' | 'unavailable', wait_s). 'unavailable' = free-tier limit of 0."""
    wait = _duration_s(headers.get("retry-after") if headers else None)
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        data = {}
    err = (data[0] if isinstance(data, list) and data else data).get("error", {}) \
        if isinstance(data, (list, dict)) else {}
    message = str(err.get("message", body))
    for detail in err.get("details", []) if isinstance(err, dict) else []:
        if "RetryInfo" in str(detail.get("@type", "")):
            wait = _duration_s(detail.get("retryDelay")) or wait
    quota_ids = " ".join(v.get("quotaId", "") for d in err.get("details", []) if isinstance(d, dict)
                         for v in d.get("violations", []) or []) if isinstance(err, dict) else ""
    if re.search(r"limit:\s*0\b", message):                  # not on the free tier at all
        return "unavailable", wait
    daily = (
        re.search(r"per ?day|PerDay|\b(RPD|TPD)\b", message + " " + quota_ids, re.I) is not None
        or (wait is not None and wait > MAX_INLINE_WAIT_S)
    )
    return ("daily" if daily else "minute"), wait


class OpenAICompatClient:
    """POST {base_url}/chat/completions with bearer auth; retries 429/5xx with backoff."""

    def __init__(self, provider: str, base_url: str, api_key: str, timeout: float = 120.0,
                 retries: int = 6, extra_headers: dict | None = None, sleep=time.sleep,
                 on_event=None):
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.extra_headers = extra_headers or {}
        self.sleep = sleep
        self.on_event = on_event   # on_event(name, focus_area, reason, **attrs) for recoveries

    def _emit(self, name: str, reason: str, **attrs) -> None:
        if self.on_event:
            try:
                self.on_event(name, "recovery", reason, provider=self.provider, **attrs)
            except Exception:
                pass

    def _post(self, path: str, body: dict, model: str) -> dict:
        headers = {"Content-Type": "application/json", "User-Agent": "notbydefault",
                   "Authorization": f"Bearer {self.api_key}", **self.extra_headers}
        data = json.dumps(body).encode()
        last: Exception | None = None
        timeouts = 0
        for attempt in range(self.retries):
            req = urllib.request.Request(f"{self.base_url}{path}", data=data, headers=headers,
                                         method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")
                if e.code == 429:
                    kind, wait = classify_429(detail, e.headers)
                    if kind == "unavailable":
                        raise ModelUnavailable(self.provider, model, _short(detail))
                    if kind == "daily":
                        raise DailyLimitReached(self.provider, model, _short(detail),
                                                _reset_hint(self.provider, e.headers, wait))
                    pause = min(MAX_INLINE_WAIT_S, (wait or 2 ** attempt) + 0.5)
                    self._emit("rate_limit_backoff",
                               f"HTTP 429 per-minute limit on {model}; waiting {pause:.1f}s "
                               f"(attempt {attempt + 1}/{self.retries})",
                               model=model, wait_s=pause, attempt=attempt + 1)
                    last = LLMError(f"{self.provider}:{model}: rate limited (HTTP 429)")
                    self.sleep(pause)
                    continue
                if e.code in (402, 403, 404):   # 402: model needs paid credits on this key
                    raise ModelUnavailable(self.provider, model, f"HTTP {e.code}: {_short(detail)}")
                last = LLMError(f"{self.provider}:{model}: HTTP {e.code}: {_short(detail)}")
                if e.code not in (408, 500, 502, 503, 504):
                    break
            except (urllib.error.URLError, TimeoutError) as e:
                last = LLMError(f"{self.provider}:{model}: network error: {e}")
                if isinstance(e, TimeoutError) or "timed out" in str(e):
                    timeouts += 1
                    if timeouts > MAX_TIMEOUT_RETRIES:   # a hung model shouldn't stall the run
                        break
            pause = min(30, 2 ** attempt)
            if attempt + 1 < self.retries:
                self._emit("retry_failed_call", f"{last}; retrying in {pause}s "
                           f"(attempt {attempt + 1}/{self.retries})",
                           model=model, wait_s=pause, attempt=attempt + 1)
            self.sleep(pause)
        raise last  # type: ignore[misc]

    def complete(self, model: str, messages: list[dict], temperature: float = 0.0,
                 max_tokens: int = 1024, json_mode: bool = False,
                 extra_body: dict | None = None) -> tuple[dict, float]:
        """Raw response JSON and latency in seconds."""
        body: dict = {"model": model, "messages": messages, "temperature": temperature,
                      "max_tokens": max_tokens, **(extra_body or {})}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        start = time.perf_counter()
        data = self._post("/chat/completions", body, model)
        if isinstance(data, list):  # Gemini sometimes wraps responses in a list
            data = data[0] if data else {}
        if "error" in data:
            raise LLMError(f"{self.provider}:{model}: {data['error']}")
        return data, time.perf_counter() - start


def parse_usage(data: dict) -> tuple[str, int, int, str | None, float | None]:
    choice = (data.get("choices") or [{}])[0]
    usage = data.get("usage") or {}
    return ((choice.get("message") or {}).get("content") or "",
            int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0),
            choice.get("finish_reason"), usage.get("cost"))


def _short(text: str, n: int = 300) -> str:
    return " ".join(text.split())[:n]


def _reset_hint(provider: str, headers, wait: float | None) -> str:
    if provider == "gemini":
        return "at midnight Pacific time"
    reset = headers.get("x-ratelimit-reset-requests") if headers else None
    if reset:
        return f"in {reset}"
    return f"in ~{wait / 3600:.1f} h" if wait else ""


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
