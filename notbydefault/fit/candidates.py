"""Step 3: candidate models in .notbydefault/candidates.yaml, chosen from OpenRouter's live list.

Prices are never hardcoded: they come from OpenRouter's models API, and the file records
the source URL and when they were fetched. Edit the file to change candidates or the judge.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from notbydefault.detect import model_provider
from notbydefault.openrouter import Price, Pricing

# detect.py provider names -> OpenRouter id prefixes
OPENROUTER_PREFIX = {"OpenAI": "openai", "Anthropic": "anthropic", "Google": "google",
                     "Meta": "meta-llama", "Mistral": "mistralai", "DeepSeek": "deepseek",
                     "xAI": "x-ai", "Alibaba": "qwen", "Cohere": "cohere"}
# Providers tried, in order, when picking cheaper alternatives.
ALTERNATIVE_PROVIDERS = ["anthropic", "google", "meta-llama", "mistralai", "deepseek",
                         "qwen", "x-ai", "openai", "cohere"]
# Judge preference: newest model matching the first pattern that isn't a candidate.
JUDGE_PATTERNS = [r"^anthropic/claude-opus-[\d.]+$", r"^anthropic/claude-sonnet-[\d.]+$",
                  r"^google/gemini-[\d.]+-pro$", r"^openai/gpt-[\d.]+$"]
SPECIALTY = re.compile(r"guard|embed|tts|audio|image|vision|banana|omni|search|-exp|preview|"
                       r"ocr|moderation|realtime|coder|devstral|codestral|math", re.I)
CHEAPER_RATIO = 0.5   # an alternative must cost at most half the chosen model per typical call
TYPICAL_IN, TYPICAL_OUT = 1000, 300

DEFAULTS = {"temperature": 0.0, "max_tokens": 1024, "max_retries": 2}


def config_path(project: Path) -> Path:
    return project / ".notbydefault" / "candidates.yaml"


def typical_cost(price: Price) -> float:
    return price.cost(TYPICAL_IN, TYPICAL_OUT)


def provider(model_id: str) -> str:
    return model_id.split("/", 1)[0]


def eligible(model: dict) -> bool:
    mid = model.get("id", "")
    arch = model.get("architecture") or {}
    if ":" in mid or SPECIALTY.search(mid) or model.get("expiration_date"):
        return False
    if "text" not in (arch.get("input_modalities") or ["text"]):
        return False
    if "text" not in (arch.get("output_modalities") or ["text"]):
        return False
    if "temperature" not in (model.get("supported_parameters") or ["temperature"]):
        return False
    p = model.get("pricing") or {}
    try:
        return float(p.get("prompt", 0)) > 0 and float(p.get("completion", 0)) > 0
    except (TypeError, ValueError):
        return False


def _norm(s: str) -> str:
    s = re.sub(r"-\d{8}$|-\d{4}-\d{2}-\d{2}$", "", s.lower())  # drop date suffixes
    return s.replace(".", "-")


def resolve_model_id(literal: str, pricing: Pricing) -> str:
    """Map a literal from code (e.g. "gpt-4o", "claude-sonnet-4-5") to an OpenRouter id."""
    if literal in pricing.prices:
        return literal
    prefix = OPENROUTER_PREFIX.get(model_provider(literal))
    ids = [m for m in pricing.prices if ":" not in m]
    if prefix and f"{prefix}/{literal}" in pricing.prices:
        return f"{prefix}/{literal}"
    target = _norm(literal.split("/")[-1])
    matches = [m for m in ids if _norm(m.split("/", 1)[-1]) == target]
    if prefix:
        matches = [m for m in matches if provider(m) == prefix] or matches
    if not matches:
        raise LookupError(f"{literal!r} not found on OpenRouter ({pricing.source_url}); "
                          "add it to candidates.yaml by hand")
    return sorted(matches, key=len)[0]


def pick_alternatives(chosen: str, pricing: Pricing, n: int = 3) -> list[str]:
    """Newest eligible model per provider costing <= CHEAPER_RATIO of the chosen one."""
    limit = typical_cost(pricing.price(chosen)) * CHEAPER_RATIO
    by_provider: dict[str, dict] = {}
    for m in pricing.models:
        mid = m["id"]
        if (not eligible(m) or provider(mid) == provider(chosen) or mid not in pricing.prices
                or typical_cost(pricing.prices[mid]) > limit):
            continue
        best = by_provider.get(provider(mid))
        if best is None or m.get("created", 0) > best.get("created", 0):
            by_provider[provider(mid)] = m
    order = ALTERNATIVE_PROVIDERS + sorted(set(by_provider) - set(ALTERNATIVE_PROVIDERS))
    return [by_provider[p]["id"] for p in order if p in by_provider][:n]


def pick_judge(exclude: list[str], pricing: Pricing) -> str:
    models = [m for m in pricing.models if eligible(m) and m["id"] not in exclude]
    for pattern in JUDGE_PATTERNS:
        hits = [m for m in models if re.match(pattern, m["id"])]
        if hits:
            return max(hits, key=lambda m: m.get("created", 0))["id"]
    big = [m for m in models if provider(m["id"]) in ("anthropic", "openai", "google")]
    if not big:
        raise LookupError("no judge model available on OpenRouter")
    return max(big, key=lambda m: typical_cost(pricing.prices[m["id"]]))["id"]


def _price_entry(model: str, role: str, pricing: Pricing) -> dict:
    p = pricing.price(model)
    return {"model": model, "role": role,
            "usd_per_mtok_in": round(p.prompt * 1e6, 6),
            "usd_per_mtok_out": round(p.completion * 1e6, 6)}


def load_or_create(project: Path, decision: dict, pricing: Pricing) -> tuple[dict, bool]:
    """Config for this decision: chosen, candidates, judge, verifier_model, temperature, ..."""
    path = config_path(project)
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
    doc = doc or {}
    decisions = doc.setdefault("decisions", {})
    key = str(decision["id"])
    created = False
    if key not in decisions:
        chosen = resolve_model_id(decision["label"], pricing)
        alternatives = pick_alternatives(chosen, pricing)
        entries = [_price_entry(chosen, "chosen", pricing)]
        entries += [_price_entry(m, "alternative", pricing) for m in alternatives]
        decisions[key] = {
            "literal": decision["label"],
            "chosen": chosen,
            "candidates": entries,
            "judge": pick_judge([e["model"] for e in entries], pricing),
            # runs the verifiers' LLM rubric check; cheap because verified setups pay for it
            "verifier_model": min([chosen] + alternatives,
                                  key=lambda m: typical_cost(pricing.price(m))),
            "pricing_source": pricing.source_url,
            "pricing_fetched_at": pricing.fetched_at,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        for k, v in DEFAULTS.items():
            doc.setdefault(k, v)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = ("# NotByDefault fit-test candidates, per decision id. Edit freely.\n"
                  "# usd_per_mtok_* are a snapshot from pricing_source at pricing_fetched_at;\n"
                  "# every check re-fetches live prices and uses those for cost.\n")
        path.write_text(header + yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        created = True
    entry = decisions[key]
    return {
        "chosen": entry["chosen"],
        "candidates": [c["model"] for c in entry["candidates"]],
        "judge": entry["judge"],
        "verifier_model": entry.get("verifier_model") or entry["chosen"],
        "max_retries": int(doc.get("max_retries", DEFAULTS["max_retries"])),
        "temperature": float(doc.get("temperature", DEFAULTS["temperature"])),
        "max_tokens": int(doc.get("max_tokens", DEFAULTS["max_tokens"])),
    }, created
