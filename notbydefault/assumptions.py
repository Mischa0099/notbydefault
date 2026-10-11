"""Watched assumptions: what each decision's verdict relied on, and which of it no longer holds.

Every decision keeps the assumptions its verdict relied on (`data.assumptions`), each with a
source: expected users, calls per user, monthly budget, list prices (+ fetched date), the data
region in force, the providers' model list, and the requirements in force. `check` records them
when it reaches a verdict; `watch` records a baseline for decisions that have none.

`broken()` compares them with now: requirements in force (a new residency, budget or user
count), prices in fit/providers.yaml, new models in the providers' lists, and real call volume
when one is measured. `notbydefault watch` (watch.py) re-opens what broke.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

VOLUME_TOLERANCE = 0.25    # real call volume this far from the assumed one breaks it


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def region_in_force(graph) -> tuple[str, str, int | None]:
    """(region or "any", source, requirement node id) from the requirements in force."""
    from notbydefault.stale import states_region
    found = ("any", "no data-residency requirement recorded", None)
    for n in graph.nodes("requirement") + graph.nodes("constraint"):
        c = (n["data"].get("constraints") or {}).get("data_residency")
        if c:
            found = (str(c).upper(), f"requirement #{n['id']} \"{n['label']}\"", n["id"])
        elif n["data"].get("key") == "region":
            for r in ("EU", "US", "UK"):
                if states_region(n["label"], r):
                    found = (r, f"constraint #{n['id']} \"{n['label']}\"", n["id"])
    return found


def stated_limits(graph) -> dict:
    """Budget / users stated by requirements added with `require` (they override init's)."""
    out = {}
    for n in graph.nodes("requirement") + graph.nodes("constraint"):
        c = n["data"].get("constraints") or {}
        if c.get("max_monthly_cost_usd") is not None:
            out["budget"] = (float(c["max_monthly_cost_usd"]), n)
        if c.get("expected_users") is not None:
            out["users"] = (float(c["expected_users"]), n)
    return out


def _models(catalog: dict, providers: set[str]) -> list[str]:
    return sorted(f"{p}:{m}" for p, cfg in (catalog.get("providers") or {}).items()
                  if p in providers for m in (cfg.get("models") or {}))


def snapshot(project: Path, graph, decision: dict, catalog: dict | None = None,
             results: dict | None = None) -> dict:
    """The assumptions this decision's verdict relies on, as they are now."""
    from notbydefault.fit.providers import load_catalog
    from notbydefault.pivot import volume
    catalog = catalog or load_catalog(project)
    data = decision["data"]
    v, limits = volume(graph), stated_limits(graph)
    a = (results or {}).get("assumptions") or {}
    src = f"fit check {data.get('results')}" if results else "Decision Graph"
    items = []

    def add(key, label, value, source, **extra):
        items.append({"key": key, "label": label, "value": value, "source": source, **extra})

    users = limits.get("users", (a.get("users") or v["users"], None))
    add("users", "expected users", users[0],
        f"requirement #{users[1]['id']}" if users[1] else src)
    if data.get("kind") == "model":
        add("calls_per_user", "calls per user per month", a.get("calls_per_user")
            or v["calls_per_user"], src)
    budget = limits.get("budget", (a.get("budget") if a.get("budget") is not None
                                   else v["budget"], None))
    add("budget", "monthly budget", budget[0],
        f"requirement #{budget[1]['id']}" if budget[1] else src)
    region, rsrc, _rid = region_in_force(graph)
    add("data_region", "data region", region, rsrc)
    if data.get("kind") == "model":
        prices = ((results or {}).get("pricing") or {}).get("prices") or {}
        providers = {m.split(":", 1)[0] for m in prices}
        for m, p in sorted(prices.items()):
            add(f"price:{m}", f"price of {m}",
                {"in": p.get("usd_per_mtok_in"), "out": p.get("usd_per_mtok_out")},
                p.get("source", "fit/providers.yaml"), fetched=p.get("fetched"))
        if not prices:        # not fit-checked yet: the chosen model's catalog list price
            for pname, cfg in (catalog.get("providers") or {}).items():
                for native, m in (cfg.get("models") or {}).items():
                    if decision["label"].lower() in (native.lower(), native.split("/")[-1].lower()):
                        providers.add(pname)
                        add(f"price:{pname}:{native}", f"price of {pname}:{native}",
                            {"in": m.get("in"), "out": m.get("out")},
                            str(cfg.get("prices_source", "fit/providers.yaml")),
                            fetched=str(cfg.get("prices_fetched", "")))
        if not providers and data.get("openrouter_id"):
            providers = {data["openrouter_id"].split(":", 1)[0]}
        providers = providers or set((catalog.get("providers") or {}))
        add("models", "model list", _models(catalog, providers),
            "fit/providers.yaml model lists")
    reqs = sorted(n["id"] for n in graph.nodes("requirement") + graph.nodes("constraint"))
    add("requirements", "requirements in force", reqs, "Decision Graph")
    return {"recorded_at": _now(), "basis": src, "items": items}


def current_price(catalog: dict, model: str) -> dict | None:
    provider, _, native = model.partition(":")
    m = ((catalog.get("providers") or {}).get(provider) or {}).get("models", {}).get(native)
    if not m:
        return None
    return {"in": m.get("in"), "out": m.get("out")}


def _fmt(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return f"{value:,.0f}"
    if isinstance(value, dict):
        return f"${value.get('in')}/M in, ${value.get('out')}/M out"
    return "unknown" if value is None else str(value)


def _cause_of(graph, nid: int | None, fallback: str) -> tuple[str, str]:
    """(what broke it, its node type: requirement / constraint, or "" when not a node)."""
    n = graph.get_node(nid) if nid else None
    return (n["label"], n["type"]) if n else (fallback, "")


def broken(project: Path, graph, decision: dict, catalog: dict,
           observed_calls: float | None = None, observed_source: str = "") -> list[dict]:
    """Assumptions recorded on the decision that no longer hold, each with what broke it."""
    saved = (decision["data"].get("assumptions") or {}).get("items") or []
    if not saved:
        return []
    now = {i["key"]: i for i in snapshot(project, graph, decision, catalog)["items"]}
    limits = stated_limits(graph)
    _region, _rsrc, region_node = region_in_force(graph)
    out = []

    def broke(item, now_value, cause, cause_type=""):
        out.append({"key": item["key"], "label": item["label"], "was": item["value"],
                    "now": now_value, "cause": cause, "cause_type": cause_type,
                    "source": item.get("source", "")})

    for item in saved:
        key, was = item["key"], item["value"]
        if key in ("users", "budget", "calls_per_user", "data_region"):
            cur = now.get(key, {}).get("value")
            if cur is not None and cur != was:
                node = (region_node if key == "data_region"
                        else (limits.get(key) or (None, None))[1])
                broke(item, cur, *_cause_of(graph, node["id"] if isinstance(node, dict) else node,
                                            f"the graph's {item['label']}"))
        elif key.startswith("price:"):
            cur = current_price(catalog, key[6:])
            if cur and was and any(abs((cur[k] or 0) - (was.get(k) or 0)) > 1e-9
                                   for k in ("in", "out")):
                broke(item, cur, "fit/providers.yaml list prices "
                      f"(fetched {(catalog.get('providers') or {}).get(key[6:].split(':')[0], {}).get('prices_fetched', '?')})")
        elif key == "models":
            new = sorted(set(now.get("models", {}).get("value") or []) - set(was or []))
            if new:
                broke(item, (was or []) + new, "new models in the providers' lists: "
                      + ", ".join(new[:6]) + (" ..." if len(new) > 6 else ""))
    if observed_calls is not None:
        users = next((i["value"] for i in saved if i["key"] == "users"), None)
        cpu = next((i["value"] for i in saved if i["key"] == "calls_per_user"), None)
        if users and cpu and abs(observed_calls - users * cpu) > VOLUME_TOLERANCE * users * cpu:
            out.append({"key": "call_volume", "label": "calls per month",
                        "was": users * cpu, "now": observed_calls,
                        "cause": observed_source or "observed call volume", "source": ""})
    return out


def describe(b: dict) -> str:
    kind = f"{b['cause_type']} " if b.get("cause_type") else ""
    return f"'{b['label']}: {_fmt(b['was'])}' broken by {kind}'{b['cause']}'"


def record_snapshot(project: Path, graph, decision_id: int, catalog: dict | None = None,
                    results: dict | None = None) -> dict:
    d = graph.get_node(decision_id)
    snap = snapshot(project, graph, d, catalog, results)
    graph.update_data(decision_id, assumptions=snap)
    return snap
