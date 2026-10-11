"""Multiple OpenAI-compatible providers behind one client: routing, throttling, disk cache.

Model ids are "<provider>:<provider's model id>", e.g. "groq:openai/gpt-oss-120b".
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import yaml

from notbydefault import PACKAGE_DIR
from notbydefault.llm import (ChatResult, LLMError, OpenAICompatClient, Price, Pricing,
                              parse_usage)

DEFAULT_CATALOG = PACKAGE_DIR / "fit" / "providers.yaml"


def role_model(catalog: dict, role: str) -> str:
    """Model configured for a role (e.g. "pivot", "stale_judge") in the catalog's `roles`."""
    model = (catalog.get("roles") or {}).get(role)
    if not model:
        raise LLMError(f"no model configured for role {role!r} in providers.yaml")
    return model


def residency(catalog: dict, model_id: str) -> list[str]:
    """Regions where a provider guarantees data stays (empty: no guarantee)."""
    provider = model_id.split(":", 1)[0]
    cfg = (catalog.get("providers") or {}).get(provider) or {}
    return [str(r).upper() for r in cfg.get("data_residency") or []]


def split_id(model_id: str) -> tuple[str, str]:
    if ":" not in model_id:
        raise LLMError(f"model id {model_id!r} must look like <provider>:<model>")
    provider, model = model_id.split(":", 1)
    return provider, model


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_catalog(project: Path | None = None) -> dict:
    catalog = yaml.safe_load(DEFAULT_CATALOG.read_text(encoding="utf-8"))
    override = Path(project) / ".notbydefault" / "providers.yaml" if project else None
    if override and override.exists():
        catalog = _merge(catalog, yaml.safe_load(override.read_text(encoding="utf-8")) or {})
    return catalog


class RateLimiter:
    """Sliding 60 s window on requests and (estimated) tokens per model."""

    def __init__(self, rpm: int | None, tpm: int | None, sleep=time.sleep, clock=time.monotonic):
        self.rpm, self.tpm = rpm, tpm
        self.sleep, self.clock = sleep, clock
        self.events: deque[tuple[float, int]] = deque()
        self.lock = threading.Lock()

    def acquire(self, tokens: int) -> float:
        """Block until a request of `tokens` fits; returns seconds waited."""
        waited = 0.0
        while True:
            with self.lock:
                now = self.clock()
                while self.events and now - self.events[0][0] >= 60:
                    self.events.popleft()
                used = sum(t for _, t in self.events)
                fits_rpm = self.rpm is None or len(self.events) < self.rpm
                fits_tpm = self.tpm is None or used + tokens <= self.tpm or not self.events
                if fits_rpm and fits_tpm:
                    self.events.append((now, tokens))
                    return waited
                wait = max(0.05, 60 - (now - self.events[0][0]))
            self.sleep(wait)
            waited += wait


class ResponseCache:
    """One JSON file per (model, messages, settings); reruns never repeat a request."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)

    @staticmethod
    def key(model_id: str, messages: list[dict], temperature: float, max_tokens: int,
            json_mode: bool, extra_body: dict | None = None) -> str:
        parts = [model_id, messages, temperature, max_tokens, json_mode]
        if extra_body:   # only when set, so existing cache entries keep their keys
            parts.append(extra_body)
        blob = json.dumps(parts,
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.folder / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def put(self, key: str, record: dict) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # one temp file per writer: two threads can put the same key at once (identical
        # requests), and on Windows a shared temp name makes the second rename fail
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        try:
            tmp.replace(path)
        except PermissionError:      # another writer is replacing it right now: same answer
            tmp.unlink(missing_ok=True)

    def count(self) -> int:
        return sum(1 for _ in self.folder.glob("*/*.json")) if self.folder.exists() else 0


class ProviderSet:
    """Same interface as a single client: pricing() and chat()."""

    def __init__(self, catalog: dict, cache_dir: Path | None = None, env=None,
                 transports: dict | None = None, sleep=time.sleep):
        self.catalog = catalog
        self.env = os.environ if env is None else env
        self.cache = ResponseCache(cache_dir) if cache_dir else None
        self.sleep = sleep
        self._transports = dict(transports or {})
        self._limiters: dict[str, RateLimiter] = {}
        self._lock = threading.Lock()
        self._pricing: Pricing | None = None
        self.requests_made = 0
        self.cache_hits = 0
        self.on_event = None   # on_event(name, focus_area, reason, **attrs); set by the check

    # --- catalog ---------------------------------------------------------------

    def providers(self) -> dict[str, dict]:
        return {name: cfg for name, cfg in (self.catalog.get("providers") or {}).items()
                if cfg.get("enabled", True)}

    def has_key(self, provider: str) -> bool:
        return provider in self._transports or bool(self.env.get(self.providers()[provider]["key_env"]))

    def pricing(self) -> Pricing:
        if self._pricing is not None:
            return self._pricing
        prices, models, sources = {}, [], []
        for name, cfg in self.providers().items():
            if cfg.get("prices") == "live":
                prices_, models_ = self._live_prices(name)
                prices.update(prices_)
                models += models_
                sources.append(f"{name}: live from its models API")
                continue
            src, fetched = cfg.get("prices_source", ""), str(cfg.get("prices_fetched", ""))
            sources.append(f"{name}: {src} (fetched {fetched})")
            for mid, m in (cfg.get("models") or {}).items():
                full = f"{name}:{mid}"
                prices[full] = Price(float(m["in"]) / 1e6, float(m["out"]) / 1e6, src, fetched)
                models.append({"id": full, "provider": name, "model": mid,
                               "tier": cfg.get("tier", "paid"),
                               "available": self.has_key(name) and m.get("available", True),
                               **{k: m.get(k) for k in ("family", "size", "strength", "rpm",
                                                        "tpm", "rpd")}})
        fetched_dates = sorted({p.fetched_at for p in prices.values() if p.fetched_at})
        self._pricing = Pricing(prices, "; ".join(sources),
                                ", ".join(fetched_dates) or "unknown", models)
        return self._pricing

    def _live_prices(self, provider: str):
        from notbydefault.openrouter import fetch_models, pricing_from_models
        live = pricing_from_models(fetch_models())
        prices = {f"{provider}:{k}": v for k, v in live.prices.items()}
        models = [{"id": f"{provider}:{m['id']}", "provider": provider, "model": m["id"],
                   "tier": "paid", "available": self.has_key(provider)} for m in live.models]
        return prices, models

    def free_tier(self, model_ids: list[str]) -> bool:
        providers = self.providers()
        return all(providers.get(split_id(m)[0], {}).get("tier") == "free" for m in model_ids)

    # --- calls -----------------------------------------------------------------

    def _transport(self, provider: str):
        with self._lock:
            if provider not in self._transports:
                cfg = self.providers().get(provider)
                if cfg is None:
                    raise LLMError(f"unknown or disabled provider {provider!r}")
                key = self.env.get(cfg["key_env"])
                if not key:
                    raise LLMError(f"{cfg['key_env']} is not set (needed for {provider})")
                self._transports[provider] = OpenAICompatClient(
                    provider, cfg["base_url"], key, sleep=self.sleep,
                    timeout=float(cfg.get("timeout", 120)),
                    on_event=lambda *a, **k: self._emit(*a, **k))
            return self._transports[provider]

    def _emit(self, name: str, focus_area: str, reason: str, **attrs) -> None:
        if self.on_event:
            try:
                self.on_event(name, focus_area, reason, **attrs)
            except Exception:
                pass

    def _limiter(self, model_id: str) -> RateLimiter:
        with self._lock:
            if model_id not in self._limiters:
                provider, mid = split_id(model_id)
                m = (self.providers().get(provider, {}).get("models") or {}).get(mid, {})
                self._limiters[model_id] = RateLimiter(m.get("rpm"), m.get("tpm"), self.sleep)
            return self._limiters[model_id]

    def is_cached(self, model: str, messages: list[dict], temperature: float = 0.0,
                  max_tokens: int = 1024, json_mode: bool = False,
                  extra_body: dict | None = None) -> bool:
        return bool(self.cache and self.cache.get(
            self.cache.key(model, messages, temperature, max_tokens, json_mode, extra_body)))

    def chat(self, model: str, messages: list[dict], temperature: float = 0.0,
             max_tokens: int = 1024, json_mode: bool = False,
             extra_body: dict | None = None) -> ChatResult:
        price = self.pricing().price(model)
        key = (self.cache.key(model, messages, temperature, max_tokens, json_mode, extra_body)
               if self.cache else None)
        hit = self.cache.get(key) if self.cache else None
        if hit:
            with self._lock:
                self.cache_hits += 1
            return ChatResult(model=model, text=hit["text"], input_tokens=hit["input_tokens"],
                              output_tokens=hit["output_tokens"], latency_s=hit["latency_s"],
                              cost=price.cost(hit["input_tokens"], hit["output_tokens"]),
                              finish_reason=hit.get("finish_reason"), cached=True)
        provider, native = split_id(model)
        est = int(sum(len(m["content"]) for m in messages) / 3.5) + min(max_tokens, 1024)
        waited = self._limiter(model).acquire(est)
        if waited:
            self._emit("throttle_wait", "execution",
                       f"stayed under {model}'s per-minute limit by waiting {waited:.1f}s",
                       model=model, wait_s=round(waited, 2))
        data, latency = self._transport(provider).complete(
            native, messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode,
            extra_body=extra_body)
        text, tin, tout, finish, reported = parse_usage(data)
        with self._lock:
            self.requests_made += 1
        if self.cache:   # empty answers too: they are real behaviour and cost a request
            self.cache.put(key, {"model": model, "text": text, "input_tokens": tin,
                                 "output_tokens": tout, "latency_s": latency,
                                 "finish_reason": finish,
                                 "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        return ChatResult(model=model, text=text, input_tokens=tin, output_tokens=tout,
                          latency_s=latency, cost=price.cost(tin, tout), finish_reason=finish,
                          reported_cost=reported)
