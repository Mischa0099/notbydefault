"""Pivot Check: notice when the user changes direction and show what it would undo.

Pipeline for each prompt (hooks/user_prompt.py):
  1. pending choice: if a pivot from this session awaits "switch anyway / adjust / keep plan",
     record the user's answer (no LLM)
  2. pre-filter: cheap regex for change-of-direction phrases; most prompts stop here (no LLM)
  3. confirm: one fast Groq call decides whether it really is a pivot and maps it onto the graph
  4. impact: goals/constraints it conflicts with, decisions it would invalidate (with their
     recorded reasons and evidence) and the code those decisions affect, read from the graph
  5. record a "pivot" node (status: proposed) linked to the affected decisions
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

PIVOT_PHRASES = [
    r"switch(?:ing)? (?:to|from|over)", r"\binstead\b", r"\bactually\b", r"what if we",
    r"let'?s (?:use|go with|try|switch|move)", r"let us use", r"\bdrop(?:ping)?\b",
    r"\breplac(?:e|ing)\b", r"rather than", r"\bgo with\b", r"\bmigrat(?:e|ing) (?:to|from|off)",
    r"\bmove (?:to|off|away from)\b", r"\bchange (?:to|from)\b", r"\bswap\b", r"\bditch\b",
    r"\bscrap\b", r"get rid of", r"\bno longer\b", r"\bforget (?:about )?(?:the|our)\b",
]
PIVOT_RX = re.compile("|".join(PIVOT_PHRASES), re.IGNORECASE)

CHOICES = [  # (status, phrases that answer "switch anyway / adjust / keep plan")
    ("accepted", r"\bswitch anyway\b|\bgo ahead with the (?:switch|change)\b"),
    ("adjusting", r"\badjust(?: the plan)?\b"),
    ("kept", r"\bkeep (?:the )?(?:current )?plan\b|\bkeep (?:it|things) as (?:is|planned)\b"),
]
MAX_CONTEXT = 9000   # Claude Code caps additionalContext at 10,000 characters


STACK_WORDS = re.compile(
    r"\b(models?|llms?|providers?|vendors?|apis?|sdks?|databases?|db|orms?|frameworks?|hosting|"
    r"host|auth(?:entication)?|cloud|backend|stack|architecture|queue|cache|storage|region|"
    r"budget|cost|pricing|plan)\b", re.IGNORECASE)


def stack_terms(graph_labels: list[str] | None = None) -> list[str]:
    """Names that make a prompt about the stack: known services/models + the graph's decisions."""
    from notbydefault.detect import KNOWN_SERVICES, MODEL_PROVIDERS
    names = {label for label, _ in KNOWN_SERVICES.values()}
    names |= {p.rstrip("-/") for p, _ in MODEL_PROVIDERS if len(p.rstrip("-/")) > 2}
    names |= {"openai", "anthropic", "gemini", "groq", "mistral", "llama", "claude", "gpt",
              "postgres", "mysql", "sqlite", "dynamo", "aws", "gcp", "azure", "vertex"}
    names |= {l for l in (graph_labels or []) if l}
    return sorted(names, key=len, reverse=True)


def prefilter(prompt: str, terms: list[str] | None = None) -> list[str]:
    """Change-of-direction phrases, but only in prompts about the stack; [] means no LLM call."""
    phrases = [m.group(0) for m in PIVOT_RX.finditer(prompt or "")]
    if not phrases:
        return []
    low = prompt.lower()
    if STACK_WORDS.search(prompt) or any(
            re.search(rf"(?<![\w]){re.escape(t.lower())}(?![\w])", low)
            for t in (terms or stack_terms())):
        return phrases
    return []


def detect_choice(prompt: str) -> str | None:
    text = (prompt or "").lower()
    for status, rx in CHOICES:
        if re.search(rx, text):
            return status
    return None


# --- graph snapshot sent to the model ------------------------------------------------

def used_for(project_dir, decision_id: int) -> str:
    """What a decision's code does, from the call-site analysis `check` saved (if any)."""
    import json
    from pathlib import Path
    if project_dir is None:
        return ""
    path = Path(project_dir) / ".notbydefault" / "callsites" / f"{decision_id}.json"
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("task") or "")
    except (OSError, ValueError):
        return ""


def snapshot(graph, project_dir=None) -> dict:
    """Compact view of goals, requirements, constraints and decisions (ids are graph ids)."""
    def short(text, n=220):
        text = " ".join(str(text or "").split())
        return text if len(text) <= n else text[:n - 1] + "…"

    nodes = {t: graph.nodes(t) for t in ("goal", "requirement", "constraint", "decision")}
    decisions = []
    for d in nodes["decision"]:
        data = d["data"]
        decisions.append({
            "id": d["id"], "label": d["label"], "kind": data.get("kind"),
            "category": data.get("category"), "status": data.get("status"),
            "used_for": short(used_for(project_dir, d["id"]), 160),
            "verdict": short(data.get("verdict_summary")),
            "recommended": data.get("recommended"), "files": data.get("files") or [],
        })
    return {"goals": [{"id": n["id"], "label": n["label"]} for n in nodes["goal"]],
            "requirements_and_constraints": [{"id": n["id"], "label": n["label"]}
                                             for n in nodes["requirement"] + nodes["constraint"]],
            "decisions": decisions}


CONFIRM_SYSTEM = ("You review messages a developer sends to a coding agent. Decide whether the "
                  "message changes the direction of an earlier technical decision. Answer with "
                  "one JSON object only.")

CONFIRM_PROMPT = """The project's Decision Graph:
{graph}

Price facts (list prices; use them for budget conflicts): {prices}

The developer just wrote:
<<<
{prompt}
>>>

A pivot is a change of direction on something already decided: a different model, provider,
database, framework, architecture, or dropping a recorded goal or constraint. Ordinary requests
(fix a bug, add a feature, rename, refactor, explain, "use X instead of Y" for a variable) are
not pivots.

Return JSON:
{{"is_pivot": true or false,
  "change": "<one line: what the developer wants to change, or empty>",
  "decision_ids": [<ids of decisions above this would invalidate>],
  "conflicts": [{{"node_id": <id of a goal/requirement/constraint above>, "why": "<one line>"}}]}}"""


def confirm(caller, model: str, prompt: str, graph_view: dict, prices: str = "none") -> dict:
    """One fast LLM call: is it a pivot, and which graph nodes does it touch?"""
    import json

    from notbydefault.llm import parse_json
    reply = caller.chat(model, [
        {"role": "system", "content": CONFIRM_SYSTEM},
        {"role": "user", "content": CONFIRM_PROMPT.format(
            graph=json.dumps(graph_view, ensure_ascii=False, indent=1), prompt=prompt[:4000],
            prices=prices)},
    ], span="notbydefault.pivot_confirm", role="pivot", max_tokens=1500, json_mode=True,
        extra_body={"reasoning_effort": "low"}, focus_area="verification")
    data = parse_json(reply.text)
    ids = {n["id"] for k in ("goals", "requirements_and_constraints", "decisions")
           for n in graph_view[k]}
    decision_ids = {d["id"] for d in graph_view["decisions"]}
    return {
        "is_pivot": bool(data.get("is_pivot")),
        "change": str(data.get("change") or "").strip(),
        "decision_ids": [int(i) for i in data.get("decision_ids") or []
                         if str(i).lstrip("#").isdigit() and int(str(i).lstrip("#")) in decision_ids],
        "conflicts": [{"node_id": int(str(c.get("node_id")).lstrip("#")), "why": str(c.get("why", ""))}
                      for c in data.get("conflicts") or []
                      if str(c.get("node_id", "")).lstrip("#").isdigit()
                      and int(str(c.get("node_id")).lstrip("#")) in ids],
        "model": model, "latency_s": reply.latency_s, "cached": reply.cached,
    }


# --- impact from the graph (deterministic) -------------------------------------------

def impact(graph, result: dict, project_dir=None, cost: dict | None = None) -> dict:
    from notbydefault.graph import find_affected_code
    conflicts = []
    for c in result["conflicts"]:
        node = graph.get_node(c["node_id"])
        if node:
            conflicts.append({"id": node["id"], "type": node["type"], "label": node["label"],
                              "why": c["why"]})
    decisions = []
    for did in result["decision_ids"]:
        d = graph.get_node(did)
        if not d:
            continue
        data = d["data"]
        evidence = [{"id": e["id"], "label": e["label"],
                     "path": e["data"].get("path") or e["data"].get("url")}
                    for e in graph.neighbors(did, ("supported_by",))]
        decisions.append({
            "id": did, "label": d["label"], "kind": data.get("kind"),
            "status": data.get("status"), "used_for": used_for(project_dir, did),
            "reason": data.get("verdict_summary") or (
                "unreviewed: no fit test yet" if data.get("status") == "unreviewed"
                else data.get("stale_reason") or ""),
            "recommended": data.get("recommended"),
            "evidence": evidence, "code": find_affected_code(d),
        })
    # the budget conflict is computed, not left to the model's judgement
    if cost and cost["over_budget"] and cost["budget_node"]:
        b = cost["budget_node"]
        conflicts = [c for c in conflicts if c["id"] != b["id"]]
        conflicts.insert(0, {"id": b["id"], "type": b["type"], "label": b["label"],
                             "why": f"about ${cost['monthly']:,.0f}/month for {cost['model']} at "
                                    f"list price, {cost['monthly'] / cost['budget']:.0f}x the budget"})
    return {"change": result["change"], "conflicts": conflicts, "decisions": decisions,
            "cost_line": cost_line(cost) if cost else ""}


def render_context(imp: dict, pivot_id: int) -> str:
    """additionalContext for Claude Code (not shown to the user directly)."""
    lines = [f"NotByDefault Pivot Check (pivot #{pivot_id}): the user's message changes an earlier "
             f"decision: {imp['change'] or 'see the message'}."]
    if imp["conflicts"]:
        lines.append("It conflicts with these recorded goals/constraints:")
        lines += [f"- #{c['id']} {c['type']} \"{c['label']}\": {c['why']}" for c in imp["conflicts"]]
    if imp["decisions"]:
        lines.append("It would invalidate these decisions:")
        for d in imp["decisions"]:
            ev = "; ".join(f"#{e['id']} {e['label']}" + (f" ({e['path']})" if e["path"] else "")
                           for e in d["evidence"]) or "none recorded"
            what = f" ({d['used_for'].rstrip('.')})" if d.get("used_for") else ""
            lines.append(f"- #{d['id']} {d['label']}{what} [{d['kind']}, {d['status']}]: {d['reason']} "
                         f"Evidence: {ev}. Code: {', '.join(d['code']) or 'none recorded'}.")
    if imp.get("cost_line"):
        lines.append(imp["cost_line"])
    lines.append(
        "Before doing anything else, show the user a summary of this in at most 6 lines "
        "(the change, the conflicts, the decisions and code affected), then ask them to choose: "
        "(1) switch anyway, (2) adjust the plan, (3) keep the current plan. Do not start the "
        "change until they answer.")
    text = "\n".join(lines)
    return text if len(text) <= MAX_CONTEXT else text[:MAX_CONTEXT - 20] + "\n[truncated]"


def record(graph, prompt: str, imp: dict, session_id: str | None, result: dict) -> int:
    label = (imp["change"] or prompt)[:120]
    pid = graph.add_node("pivot", label, {
        "status": "proposed", "prompt": prompt[:2000], "change": imp["change"],
        "conflicts": imp["conflicts"], "decisions": [d["id"] for d in imp["decisions"]],
        "confirmed_by": result.get("model"),
        "proposed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, session_id)
    for d in imp["decisions"]:
        graph.add_edge(pid, d["id"], "affects")
    return pid


def pending(graph, session_id: str | None) -> dict | None:
    """The latest proposed pivot of this session, if any."""
    for p in reversed(graph.nodes("pivot")):
        if p["data"].get("status") == "proposed" and (session_id is None or p["session_id"] == session_id):
            return p
    return None


def resolve(graph, pivot: dict, status: str, prompt: str = "") -> list[int]:
    """Record the user's choice. An accepted pivot marks the decisions it affects stale."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    graph.update_data(pivot["id"], status=status, resolved_at=now, user_answer=prompt[:500])
    stale = []
    if status == "accepted":
        for did in pivot["data"].get("decisions", []):
            d = graph.get_node(did)
            if d and d["type"] == "decision":
                graph.update_data(did, status="stale", stale_since=now,
                                  stale_reason=f"pivot #{pivot['id']} accepted: {pivot['data'].get('change')}",
                                  stale_cause=pivot["id"])
                stale.append(did)
    if stale and getattr(graph, "path", None) is not None:
        from notbydefault.records import write_record   # fast: no Brain call from the hook
        for did in stale:
            write_record(graph.path.parents[1], graph, did,
                         f"stale: pivot #{pivot['id']} accepted ({pivot['data'].get('change')})")
    return stale



# --- cost check (deterministic) ------------------------------------------------------

LOOSE_MODEL = re.compile(
    r"\b(gpt-oss-[\w.-]+|gpt-\d[\w.-]*|chatgpt-[\w.-]+|o[134](?:-mini|-pro)?|claude-[\w.-]+|"
    r"gemini-[\w.-]+|llama-?\d[\w.-]*|mistral-[\w.-]+|deepseek-[\w.-]+|grok-\d[\w.-]*|"
    r"qwen[\d.]*-[\w.-]+)\b", re.IGNORECASE)
DEFAULT_TOKENS = (1000, 300)   # per-call assumption when nothing was measured
OPENROUTER_TTL_S = 24 * 3600


def named_models(prompt: str) -> list[str]:
    seen: dict[str, None] = {}
    for m in LOOSE_MODEL.finditer(prompt or ""):
        seen.setdefault(m.group(1).lower().rstrip(".,"), None)
    return list(seen)


def _openrouter_models(project_dir) -> list[dict]:
    """OpenRouter's public model list (no key needed), cached on disk for a day."""
    import json
    import time
    from pathlib import Path
    from notbydefault.openrouter import fetch_models
    cache = Path(project_dir) / ".notbydefault" / "cache" / "openrouter-models.json"
    try:
        if cache.exists() and time.time() - cache.stat().st_mtime < OPENROUTER_TTL_S:
            return json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    models = fetch_models(timeout=5)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(models), encoding="utf-8")
    return models


def list_price(model: str, client, project_dir) -> dict | None:
    """{"model", "in", "out" (USD per token), "source", "fetched"} or None."""
    from datetime import datetime, timezone
    from notbydefault.fit.candidates import resolve_model_id
    try:
        mid = resolve_model_id(model, client.pricing())
        pr = client.pricing().price(mid)
        return {"model": mid, "in": pr.prompt, "out": pr.completion,
                "source": pr.source_url, "fetched": pr.fetched_at}
    except (LookupError, Exception):
        pass
    try:
        models = _openrouter_models(project_dir)
    except Exception:
        return None
    exact = [m for m in models if m["id"].split("/")[-1].lower() == model.lower()]
    if not exact:
        return None
    m = min(exact, key=lambda m: len(m["id"]))
    try:
        return {"model": m["id"], "in": float(m["pricing"]["prompt"]),
                "out": float(m["pricing"]["completion"]),
                "source": "https://openrouter.ai/api/v1/models",
                "fetched": datetime.now(timezone.utc).date().isoformat()}
    except (KeyError, TypeError, ValueError):
        return None


def volume(graph) -> dict:
    """Users, calls per user per month and monthly budget from the graph (None if unknown)."""
    from notbydefault.fit.stats import parse_number
    out = {"users": None, "calls_per_user": None, "budget": None, "budget_node": None}
    for n in graph.nodes("requirement") + graph.nodes("constraint"):
        key, value = n["data"].get("key"), n["data"].get("value")
        if key == "users":
            out["users"] = parse_number(str(value))
        elif key == "calls_per_user":
            out["calls_per_user"] = parse_number(str(value))
        elif key == "budget":
            out["budget"], out["budget_node"] = parse_number(str(value)), n
    return out


def tokens_per_call(graph, project_dir, decision_ids: list[int]) -> tuple[int, int, str]:
    """Measured average tokens per call of an affected decision's current model, or the default."""
    import json
    from pathlib import Path
    for did in decision_ids:
        d = graph.get_node(did)
        res = d and d["data"].get("results")
        if not res or project_dir is None:
            continue
        try:
            data = json.loads((Path(project_dir) / res).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        chosen = data.get("config", {}).get("chosen")
        rows = [r for r in data.get("rows", []) if r.get("model") == chosen
                and r.get("setup") == "alone" and not r.get("error")]
        if rows:
            tin = round(sum(r["input_tokens"] for r in rows) / len(rows))
            tout = round(sum(r["output_tokens"] for r in rows) / len(rows))
            return tin, tout, f"measured on decision #{did}'s fit test ({res})"
    return DEFAULT_TOKENS[0], DEFAULT_TOKENS[1], (
        f"assumed {DEFAULT_TOKENS[0]} input + {DEFAULT_TOKENS[1]} output tokens per call "
        "(no fit test measured yet)")


def price_facts(prompt: str, client, project_dir) -> list[dict]:
    return [p for p in (list_price(m, client, project_dir) for m in named_models(prompt)) if p]


def cost_check(graph, project_dir, prices: list[dict], decision_ids: list[int]) -> dict | None:
    """Projected monthly cost of the model the user wants, against the recorded budget."""
    if not prices:
        return None
    v = volume(graph)
    if not (v["users"] and v["calls_per_user"]):
        return None
    tin, tout, basis = tokens_per_call(graph, project_dir, decision_ids)
    calls = v["users"] * v["calls_per_user"]
    p = prices[0]
    monthly = calls * (tin * p["in"] + tout * p["out"])
    return {"model": p["model"], "monthly": monthly, "calls": calls, "tokens_in": tin,
            "tokens_out": tout, "basis": basis, "budget": v["budget"],
            "budget_node": v["budget_node"], "price": p,
            "over_budget": v["budget"] is not None and monthly > v["budget"]}


def cost_line(c: dict) -> str:
    p = c["price"]
    line = (f"Cost check: {c['model']} at list price (${p['in'] * 1e6:.2f}/M input, "
            f"${p['out'] * 1e6:.2f}/M output; {p['source']}, {p['fetched']}) x "
            f"{c['calls']:,.0f} calls/month x {c['tokens_in']} in + {c['tokens_out']} out tokens "
            f"({c['basis']}) = about ${c['monthly']:,.0f}/month")
    if c["budget"] is not None:
        line += (f" vs a ${c['budget']:,.0f}/month budget"
                 + (" (OVER BUDGET)" if c["over_budget"] else ""))
    return line + "."
