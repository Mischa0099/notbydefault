"""OpenRouter specifics (optional provider): live prices from its public models API."""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone

from notbydefault.llm import (ChatResult, LLMError, OpenAICompatClient, Price, Pricing,  # noqa: F401
                              parse_json, parse_usage)

API_BASE = "https://openrouter.ai/api/v1"
MODELS_URL = f"{API_BASE}/models"


def pricing_from_models(models: list[dict], source_url: str = MODELS_URL,
                        fetched_at: str | None = None) -> Pricing:
    fetched_at = fetched_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    prices = {}
    for m in models:
        p = m.get("pricing") or {}
        try:
            prices[m["id"]] = Price(float(p["prompt"]), float(p["completion"]), source_url,
                                    fetched_at)
        except (KeyError, TypeError, ValueError):
            continue
    return Pricing(prices, source_url, fetched_at, models)


def fetch_models(timeout: float = 60) -> list[dict]:
    req = urllib.request.Request(MODELS_URL, headers={"User-Agent": "notbydefault"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))["data"]


class OpenRouterClient(OpenAICompatClient):
    def __init__(self, api_key: str, **kwargs):
        super().__init__("openrouter", API_BASE, api_key, extra_headers={
            "X-Title": "NotByDefault",
            "HTTP-Referer": "https://github.com/Mischa0099/notbydefault"}, **kwargs)
