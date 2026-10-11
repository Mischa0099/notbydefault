"""Step 3: candidate models in .notbydefault/candidates.yaml, chosen from the provider catalog.

Only models on a free tier with a configured API key are picked automatically: the chosen
model plus small models of the same family and of other families. The judge is the strongest
free model (Gemini first) that isn't a candidate. Prices are the providers' published paid
list prices (catalog: notbydefault/fit/providers.yaml), recorded with source URL and date.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from notbydefault.llm import Price, Pricing

N_ALTERNATIVES = 2            # chosen + 2 = 3 candidates by default
SIZE_ORDER = {"small": 0, "medium": 1, "large": 2}
JUDGE_FAMILIES = ["gemini"]   # judge preference: a strong Gemini model
JUDGE_FALLBACK_SKIP = {"qwen"}  # produced unusable analyst output as a stand-in judge (2026-10-10)
TYPICAL_IN, TYPICAL_OUT = 1000, 300

DEFAULTS = {"temperature": 0.0, "max_tokens": 2048, "max_retries": 2}


def config_path(project: Path) -> Path:
    return project / ".notbydefault" / "candidates.yaml"


def typical_cost(price: Price) -> float:
    return price.cost(TYPICAL_IN, TYPICAL_OUT)


def usable(m: dict) -> bool:
    return m.get("tier") == "free" and m.get("available", True)



def _norm(s: str) -> str:
    s = re.sub(r"-\d{8}$|-\d{4}-\d{2}-\d{2}$", "", s.lower())  # drop date suffixes
    return s.replace(".", "-")


def resolve_model_id(literal: str, pricing: Pricing) -> str:
    """Map a literal from code ("openai/gpt-oss-120b", "gemini-3.8-flash") to "<provider>:<id>"."""
    if literal in pricing.prices:
        return literal
    target = _norm(literal)
    matches = [m for m in pricing.models
               if _norm(m["model"]) == target or _norm(m["model"].split("/")[-1]) == target]
    if not matches:
        raise LookupError(f"{literal!r} is not in the provider catalog ({pricing.source_url}); "
                          "add it to .notbydefault/providers.yaml")
    matches.sort(key=lambda m: (not usable(m), m["provider"]))
    return matches[0]["id"]


def _entry(pricing: Pricing, model_id: str) -> dict:
    return next((m for m in pricing.models if m["id"] == model_id),
                {"id": model_id, "provider": model_id.split(":")[0], "model": model_id})


def pick_alternatives(chosen: str, pricing: Pricing, n: int = N_ALTERNATIVES,
                      exclude: tuple | list | set = (), allowed=None) -> list[str]:
    """Small free models: first of the same family, then of other families (one per family)."""
    c = _entry(pricing, chosen)
    pool = [m for m in pricing.models if usable(m) and m["id"] != chosen and m["id"] not in exclude
            and (allowed is None or allowed(m["id"]))
            and SIZE_ORDER.get(m.get("size"), 1) < SIZE_ORDER.get(c.get("size"), 2)]
    cost = lambda m: typical_cost(pricing.price(m["id"]))
    same = sorted((m for m in pool if m.get("family") == c.get("family")), key=cost)
    other = sorted((m for m in pool if m.get("family") != c.get("family")),
                   key=lambda m: (SIZE_ORDER.get(m.get("size"), 1), m["provider"] == c.get("provider"), cost(m)))
    picks, families = [], set()
    for m in same[:1] + other:
        if m.get("family") in families and m.get("family") != c.get("family"):
            continue
        picks.append(m["id"])
        families.add(m.get("family"))
        if len(picks) == n:
            break
    return picks


def pick_judge(exclude: list[str], pricing: Pricing) -> str:
    """Strongest working Gemini that isn't excluded; else the strongest other free model,
    skipping families too weak to judge (JUDGE_FALLBACK_SKIP)."""
    pool = [m for m in pricing.models if usable(m) and m["id"] not in exclude]   # judges see test data only
    preferred = [m for m in pool if m.get("family") in JUDGE_FAMILIES]
    pool = preferred or [m for m in pool if m.get("family") not in JUDGE_FALLBACK_SKIP]
    if not pool:
        raise LookupError("no free model left to act as judge; add one to providers.yaml")
    return max(pool, key=lambda m: (m.get("strength") or 0,
                                    typical_cost(pricing.price(m["id"]))))["id"]


def freeable_judge(chosen: str, candidates: list[str], pricing: Pricing,
                   exclude: set | frozenset = frozenset()) -> str | None:
    """A non-current candidate that may be taken out of the comparison to act as judge, or None.

    Same trust rules as pick_judge (a preferred judge family first, never a family in
    JUDGE_FALLBACK_SKIP); among those, one from another family than today's model (less
    self-preference), then the strongest."""
    family = _entry(pricing, chosen).get("family")
    pool = [_entry(pricing, m) for m in candidates if m != chosen and m not in exclude]
    pool = [m for m in pool if m.get("family") in JUDGE_FAMILIES] or \
        [m for m in pool if m.get("family") not in JUDGE_FALLBACK_SKIP]
    if not pool:
        return None
    return max(pool, key=lambda m: (m.get("family") != family, m.get("strength") or 0,
                                    typical_cost(pricing.price(m["id"]))))["id"]


def pick_second_judge(exclude: list[str], first: str, pricing: Pricing) -> str | None:
    """A strong judge from a different family than the first, to expose self-preference.

    qwen is allowed here: this judge is cross-checked against the first, not trusted alone.
    """
    family = _entry(pricing, first).get("family")
    pool = [m for m in pricing.models if usable(m) and m["id"] not in exclude
            and m["id"] != first and m.get("family") != family]
    if not pool:
        return None
    return max(pool, key=lambda m: (m.get("strength") or 0,
                                    typical_cost(pricing.price(m["id"]))))["id"]


def why_picked(model: str, chosen: str, pricing: Pricing) -> str:
    m, c = _entry(pricing, model), _entry(pricing, chosen)
    if model == chosen:
        return "the model the code uses today"
    family = ("same family" if m.get("family") == c.get("family")
              else f"other family ({m.get('family')})")
    ratio = typical_cost(pricing.price(chosen)) / max(typical_cost(pricing.price(model)), 1e-12)
    return (f"{family}, {m.get('size')} model on {m.get('provider')}'s free tier, "
            f"{ratio:.1f}x cheaper at list price")


def _price_entry(model: str, role: str, pricing: Pricing) -> dict:
    p = pricing.price(model)
    return {"model": model, "role": role, "why": "",
            "usd_per_mtok_in": round(p.prompt * 1e6, 6),
            "usd_per_mtok_out": round(p.completion * 1e6, 6),
            "price_source": p.source_url, "price_fetched": p.fetched_at}


def load_or_create(project: Path, decision: dict, pricing: Pricing, allowed=None,
                   refresh: str | None = None) -> tuple[dict, bool]:
    """Config for this decision: chosen, candidates, judge, verifier_model, temperature, ...

    allowed(model_id) -> bool filters which alternatives may be picked (e.g. data residency).
    refresh: a reason to re-pick candidates (e.g. the decision went stale); the old entry is
    kept under `history`.
    """
    path = config_path(project)
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
    doc = doc or {}
    decisions = doc.setdefault("decisions", {})
    key = str(decision["id"])
    created = False
    previous = None
    if refresh and key in decisions:
        previous = decisions.pop(key)
        previous["replaced_because"] = refresh
    if key not in decisions:
        chosen = resolve_model_id(decision["label"], pricing)
        alternatives = pick_alternatives(chosen, pricing, allowed=allowed)
        entries = [_price_entry(chosen, "chosen", pricing)]
        entries += [_price_entry(m, "alternative", pricing) for m in alternatives]
        for e in entries:
            e["why"] = why_picked(e["model"], chosen, pricing)
        swaps = []
        try:
            judge = pick_judge([e["model"] for e in entries], pricing)
            judge_why = (f"strongest free model (family {_entry(pricing, judge).get('family')}, "
                         "strength rank) that is not a candidate")
        except LookupError:
            # every free model is a candidate: take one out of the comparison to judge
            judge = freeable_judge(chosen, [e["model"] for e in entries], pricing)
            if judge:
                entries = [e for e in entries if e["model"] != judge]
                judge_why = ("freed from the candidates to judge: every free model was a "
                             "candidate, none was left to act as judge")
                swaps.append({"role": "judge_from_candidate", "old": judge, "new": judge,
                              "reason": "every free model was a candidate",
                              "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            else:
                judge_why = ("no judge: every free model is a candidate and none can be freed "
                             "(only today's model, or families not trusted as a sole judge); "
                             "reference-only scoring")
        second = (pick_second_judge([e["model"] for e in entries] + [judge], judge, pricing)
                  if judge else None)
        alternatives = [e["model"] for e in entries if e["model"] != chosen]
        decisions[key] = {
            "literal": decision["label"],
            "chosen": chosen,
            "candidates": entries,
            "judge": judge,
            "judge_why": judge_why,
            "judges": ([judge] if judge else []) + ([second] if second else []),
            "second_judge_why": (f"strongest non-candidate outside the {_entry(pricing, judge).get('family')} "
                                 "family, to check the first judge for self-preference")
            if second else "no second judge available",
            # runs the verifiers' LLM rubric check; cheap because verified setups pay for it
            "verifier_model": min([chosen] + alternatives,
                                  key=lambda m: typical_cost(pricing.price(m))),
            "verifier_why": "cheapest candidate: verified setups pay for every rubric check",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if swaps:
            decisions[key]["swaps"] = swaps
        if previous:
            decisions[key]["history"] = (previous.pop("history", []) or []) + [previous]
        if refresh:
            decisions[key]["refreshed_because"] = refresh
        for k, v in DEFAULTS.items():
            doc.setdefault(k, v)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = ("# NotByDefault fit-test candidates, per decision id. Edit freely.\n"
                  "# Model ids are <provider>:<model>; providers and their published list\n"
                  "# prices live in notbydefault/fit/providers.yaml (override per project in\n"
                  "# .notbydefault/providers.yaml). usd_per_mtok_* here are a snapshot.\n")
        path.write_text(header + yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        created = True
    entry = decisions[key]
    judges = entry.get("judges")
    if judges is None and entry.get("judge"):   # entries written before two-judge support
        second = pick_second_judge([c["model"] for c in entry["candidates"]], entry["judge"],
                                   pricing)
        judges = [entry["judge"]] + ([second] if second else [])
    judges = judges or []
    why = {c["model"]: c.get("why") or "listed in candidates.yaml" for c in entry["candidates"]}
    if entry.get("judge"):
        why[entry["judge"]] = entry.get("judge_why") or "judge listed in candidates.yaml"
    for j in judges[1:]:
        why[j] = entry.get("second_judge_why") or "second judge from another family"
    freed = [s_["old"] for s_ in entry.get("swaps", []) if s_.get("role") == "judge_from_candidate"]
    return {
        "chosen": entry["chosen"],
        "judges": judges,
        "candidates": [c["model"] for c in entry["candidates"]],
        "why": why,
        "judge": entry.get("judge"),
        "judge_why": entry.get("judge_why", ""),
        # a candidate freed to judge in this call (the run records it as a swap)
        "freed_for_judge": freed[-1] if created and freed else None,
        "verifier_model": entry.get("verifier_model") or entry["chosen"],
        "max_retries": int(doc.get("max_retries", DEFAULTS["max_retries"])),
        "temperature": float(doc.get("temperature", DEFAULTS["temperature"])),
        "max_tokens": int(doc.get("max_tokens", DEFAULTS["max_tokens"])),
    }, created


def swap(project: Path, decision: dict, role: str, old: str, new: str, reason: str) -> None:
    """Persist a swap (candidate, judge or verifier model) so later runs stay comparable."""
    path = config_path(project)
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    entry = doc["decisions"][str(decision["id"])]
    if role == "candidate":
        for c in entry["candidates"]:
            if c["model"] == old:
                c.update(model=new, why=f"replaced {old}: {reason}")
    elif role == "judge":
        entry["judge"], entry["judge_why"] = new, f"replaced {old}: {reason}"
        entry["judges"] = [new if j == old else j for j in entry.get("judges", [new])]
    elif role == "judge2":
        entry["judges"] = [new if j == old else j for j in entry.get("judges", [])]
        entry["second_judge_why"] = f"replaced {old}: {reason}"
    elif role == "judge_from_candidate":   # candidate `new` leaves the comparison to judge
        entry["candidates"] = [c for c in entry["candidates"] if c["model"] != new]
        entry["judge"], entry["judge_why"] = new, f"freed from the candidates to judge: {reason}"
        entry["judges"] = [new] + [j for j in entry.get("judges", []) if j not in (old, new)]
    elif role == "no_judge":
        entry["judge"], entry["judges"] = None, []
        entry["judge_why"] = f"no judge: {reason}; reference-only scoring"
    if entry.get("verifier_model") == old:
        entry["verifier_model"], entry["verifier_why"] = new, f"replaced {old}: {reason}"
    entry.setdefault("swaps", []).append({
        "role": role, "old": old, "new": new, "reason": reason,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    text = path.read_text(encoding="utf-8")
    header = "".join(l + "\n" for l in text.splitlines() if l.startswith("#"))
    path.write_text(header + yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
