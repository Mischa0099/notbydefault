"""Claim check: `notbydefault claims <id>`.

Takes the agent's stated reason for a decision (recorded by the hook from the tool call's
description and the transcript, else read back from the Entire session transcript), splits it
into typed claims, and checks each one against evidence we already have:

  cost        list price x recorded volume vs the budget (and the fit check's measured cost)
  quality     the fit check's results (correct vs reference answers, or judge scores)
  latency     the fit check's measured latency
  compliance  documented data residency (fit/providers.yaml, service_facts.yaml)
  maturity    nothing measures it here: untested
  popularity  never evidence for your project

One cheap model call splits the reason (only spans quoted verbatim from it are kept); when the
call fails, a keyword split does it instead. Verdicts: true / false / untested, each with its
source (results file + trace id, price source + date, residency source URL).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

CLAIM_TYPES = ("cost", "quality", "latency", "compliance", "maturity", "popularity")
KEYWORDS = {
    "popularity": r"popular|widely used|everyone|most used|industry standard|de facto|community|"
                  r"stars|downloads|trending|go-to|well[- ]known|common choice",
    "compliance": r"\bEU\b|europe|GDPR|data residen|\bregion|compliant|compliance|HIPAA|SOC ?2|"
              r"sovereign|\bUS\b|\bUK\b|data stays?|hosted in",
    "latency": r"fast|latency|quick|speed|low[- ]latency|real[- ]?time|responsive|snappy|"
               r"tokens per second|throughput",
    "cost": r"cheap|cost|price|pricing|afford|budget|inexpensive|free tier|\$|economical|"
            r"save money|low[- ]cost",
    "quality": r"accura|quality|best|smart|capable|reliab|correct|good at|strong|better|"
               r"powerful|state[- ]of[- ]the[- ]art|benchmark|handles? .* well",
    "maturity": r"mature|stable|battle[- ]tested|production[- ]ready|proven|well[- ]supported|"
                r"long[- ]term support|\bLTS\b|established|robust ecosystem",
}
SPLIT_SYSTEM = ("You split a coding agent's stated reason for a technical choice into separate "
                "claims. Answer with one JSON object only.")
SPLIT_PROMPT = """The agent chose {label} ({kind}/{category}) and said:
<<<
{reason}
>>>
Split this into the separate claims it makes about the choice. For each, copy the exact words
from the text (verbatim, 2 to 25 words) and give its type, one of:
cost, quality, latency, compliance, maturity, popularity.
(compliance = data residency, regulations, where it is hosted; quality = how well it does the
task; popularity = how widely others use it.) Skip sentences that make no claim.
Return {{"claims": [{{"type": "<type>", "text": "<verbatim words>"}}]}}"""
MAX_REASON = 2000
CLAIM_MARK = {"true": "TRUE", "false": "FALSE", "untested": "UNTESTED"}


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s$%.]", " ", s.lower()).split())


def keyword_claims(reason: str) -> list[dict]:
    """Deterministic split: one claim per clause that matches a type's keywords."""
    out = []
    for clause in re.split(r"(?<=[.!?;])\s+|\n+|,\s+(?=and |but |plus )|\s+and\s+(?=it |its |the )",
                           reason):
        clause = clause.strip(" .;,-")
        if len(clause) < 4:
            continue
        for kind in ("popularity", "compliance", "latency", "cost", "quality", "maturity"):
            if re.search(KEYWORDS[kind], clause, re.IGNORECASE):
                out.append({"type": kind, "text": clause})
                break
    return out


def split_claims(reason: str, decision: dict, caller=None, model: str | None = None,
                 tracer=None) -> tuple[list[dict], str]:
    """([{"type", "text"}], how) using one cheap model call, else keywords."""
    if caller is not None and model:
        from notbydefault.llm import LLMError, parse_json
        data = decision["data"]
        try:
            reply = caller.chat(model, [
                {"role": "system", "content": SPLIT_SYSTEM},
                {"role": "user", "content": SPLIT_PROMPT.format(
                    label=decision["label"], kind=data.get("kind"), category=data.get("category"),
                    reason=reason[:MAX_REASON])}],
                span="notbydefault.split_claims", role="pivot", max_tokens=800, json_mode=True,
                focus_area="tool_use", attrs={"notbydefault.decision_id": decision["id"]})
            raw = parse_json(reply.text).get("claims") or []
            claims = [{"type": str(c.get("type", "")).lower(), "text": str(c.get("text", "")).strip()}
                      for c in raw if isinstance(c, dict)]
            norm = _norm(reason)
            kept = [c for c in claims if c["type"] in CLAIM_TYPES and c["text"]
                    and _norm(c["text"]) in norm]
            if tracer is not None and len(kept) < len(claims):
                tracer.event("claims_dropped", "verification",
                             f"{len(claims) - len(kept)} claim(s) not quoted from the reason or "
                             "of an unknown type were dropped")
            if kept:
                return kept, model
        except Exception as e:     # noqa: BLE001  (LLMError, BudgetExceeded, bad JSON...)
            if tracer is not None:
                tracer.event("split_failed", "recovery", f"claim split failed ({e}); using keywords")
    return keyword_claims(reason), "keywords"


# --- the agent's stated reason --------------------------------------------------------

def _text(message) -> str:
    from notbydefault.scout import _text as text_of
    return text_of(message)


def reason_from_event(event: dict, limit: int = 1200) -> tuple[str, str]:
    """At hook time: the tool call's own description plus the agent's last words before it
    (read from the tail of the session transcript). Cheap: one bounded file read."""
    parts, sources = [], []
    desc = ((event.get("tool_input") or {}).get("description") or "").strip()
    if desc:
        parts.append(desc)
        sources.append("tool call description")
    path = event.get("transcript_path")
    if path:
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 200_000))
                lines = f.read().decode("utf-8", "replace").splitlines()
            for line in reversed(lines):
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("type") != "assistant":
                    continue
                text = " ".join(_text(row.get("message")).split())
                if text:
                    parts.append(text[-limit:])
                    sources.append("session transcript")
                    break
        except OSError:
            pass
    return " ".join(parts)[:MAX_REASON], " + ".join(sources)


def reason_from_entire(project: Path, decision: dict, entire) -> tuple[str, str]:
    """Read back from the Entire checkpoints of the decision's session: the assistant text of
    the message that made the tool call (or the last one mentioning the choice)."""
    sid, tool_id = decision.get("session_id"), decision["data"].get("tool_use_id")
    rows = entire.checkpoint_list(project) if sid else None
    for row in rows or []:
        if sid not in (row.get("session_ids") or [row.get("session_id")]):
            continue
        transcript = entire.checkpoint_transcript(row.get("checkpoint_id"), project) or []
        last, hit = "", ""
        for t in transcript:
            if t.get("type") != "assistant":
                continue
            msg = t.get("message") or {}
            text = " ".join(_text(msg).split())
            content = msg.get("content") if isinstance(msg, dict) else None
            uses = [c.get("id") for c in content or [] if isinstance(c, dict)
                    and c.get("type") == "tool_use"]
            if text:
                last = text
                if decision["label"].lower() in text.lower():
                    hit = text
            if tool_id and tool_id in uses:
                return (text or last)[:MAX_REASON], f"Entire checkpoint {row.get('checkpoint_id')}"
        if hit:
            return hit[:MAX_REASON], f"Entire checkpoint {row.get('checkpoint_id')}"
    return "", ""


def stated_reason(project: Path, decision: dict, entire=None) -> tuple[str, str]:
    data = decision["data"]
    if data.get("agent_reason"):
        return data["agent_reason"], data.get("agent_reason_source") or "hook"
    if entire is not None:
        text, src = reason_from_entire(project, decision, entire)
        if text:
            return text, src
    return "", ""


# --- checks ------------------------------------------------------------------------------

SUPERLATIVE = re.compile(r"\b(best|most|better|smartest|strongest|top|highest|cheapest|cheaper|"
                         r"fastest|faster|lowest|least)\b", re.IGNORECASE)
REGION_NAMES = {"EU": r"\bEU\b|\bEEA\b|europe|GDPR", "US": r"\bUS\b|\bUSA\b|united states|america",
                "UK": r"\bUK\b|united kingdom|britain"}


def load_context(project: Path, graph, decision: dict, client=None) -> dict:
    """What the checks read: the latest fit check, prices, residency facts, recorded volume."""
    from notbydefault.stale import load_service_facts
    data = decision["data"]
    results, results_rel = None, data.get("results")
    if results_rel and (Path(project) / results_rel).is_file():
        try:
            results = json.loads((Path(project) / results_rel).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            results = None
    if client is None:
        try:
            from notbydefault.fit.providers import ProviderSet, load_catalog
            client = ProviderSet(load_catalog(project), cache_dir=None, env={})
        except Exception:    # noqa: BLE001
            client = None
    return {"project": Path(project), "graph": graph, "decision": decision, "results": results,
            "results_rel": results_rel, "client": client,
            "catalog": getattr(client, "catalog", None) or {},
            "facts": load_service_facts(project)}


def _verdict(kind: str, claim: dict, verdict: str, detail: str, source: str = "",
             trace_id: str | None = None) -> dict:
    return {"type": kind, "text": claim["text"], "verdict": verdict, "detail": detail,
            "source": source, "trace_id": trace_id}


def _chosen_stats(results: dict) -> tuple[dict | None, list[dict]]:
    stats = list((results.get("stats") or {}).values())
    chosen = (results.get("config") or {}).get("chosen")
    cur = next((s for s in stats if s.get("model") == chosen and s.get("setup") == "alone"), None)
    return cur, stats


def _fit_source(ctx: dict) -> tuple[str, str | None]:
    r = ctx["results"] or {}
    return f"fit check {ctx['results_rel']}", r.get("trace_id")


def _usd(x, places: int = 2) -> str:
    return "n/a" if x is None else "$" + format(float(x), f",.{places}f")


def check_cost(claim: dict, ctx: dict) -> dict:
    d = ctx["decision"]
    if d["data"].get("kind") != "model":
        return _verdict("cost", claim, "untested", "no list price is recorded for services")
    if ctx["results"]:
        cur, stats = _chosen_stats(ctx["results"])
        if cur:
            src, trace = _fit_source(ctx)
            budget = (ctx["results"].get("assumptions") or {}).get("budget")
            alone = [s for s in stats if s.get("setup") == "alone" and s.get("avg_cost")]
            cheapest = min(alone, key=lambda s: s["avg_cost"]) if alone else cur
            within = cur.get("fits_budget") is not False
            rel = bool(SUPERLATIVE.search(claim["text"]))
            ok = within and (not rel or cur["avg_cost"] <= cheapest["avg_cost"] * 1.05)
            detail = (f"measured {_usd(cur['avg_cost'], 6)}/call, "
                      f"{_usd(cur.get('monthly_cost'))}/month at the assumed volume"
                      + (f" vs a {_usd(budget)} budget" if budget is not None else "")
                      + ("" if within else " (OVER BUDGET)")
                      + (f"; cheapest tested: {cheapest['name']} at "
                         f"{_usd(cheapest['avg_cost'], 6)}/call"
                         if cheapest is not cur else "; the cheapest setup tested"))
            return _verdict("cost", claim, "true" if ok else "false", detail, src, trace)
    from notbydefault.pivot import DEFAULT_TOKENS, cost_check, cost_line, list_price, named_models
    client = ctx["client"]
    price = list_price(d["label"], client, ctx["project"]) if client is not None else None
    c = cost_check(ctx["graph"], ctx["project"], [price], [d["id"]]) if price else None
    # "cheaper than X": compare list prices per call with the model the claim names
    tin, tout = (c["tokens_in"], c["tokens_out"]) if c else DEFAULT_TOKENS
    others = [m for m in named_models(claim["text"]) if m.lower() not in d["label"].lower()]
    other = list_price(others[0], client, ctx["project"]) if price and others else None
    compared = None
    if other:
        mine, theirs = tin * price["in"] + tout * price["out"], tin * other["in"] + tout * other["out"]
        compared = (mine <= theirs, f"list price {_usd(mine, 6)}/call vs {_usd(theirs, 6)}/call for "
                                    f"{other['model']} ({tin} in + {tout} out tokens)")
    if not c and not compared:
        return _verdict("cost", claim, "untested",
                        "no list price or no recorded volume (users x calls per user) to check it")
    parts, ok = [], True
    if compared:
        parts.append(compared[1])
        ok = compared[0]
    if c:
        parts.append(cost_line(c))
        ok = ok and not c["over_budget"]
    verdict = "true" if ok else "false"
    if ok and not compared and c["budget"] is None:
        verdict = "untested"
    return _verdict("cost", claim, verdict, "; ".join(parts),
                    f"{price['source']} (fetched {price['fetched']})")


def check_quality(claim: dict, ctx: dict) -> dict:
    if not ctx["results"]:
        return _verdict("quality", claim, "untested",
                        f"no fit check yet: run `notbydefault check {ctx['decision']['id']}`")
    cur, stats = _chosen_stats(ctx["results"])
    if not cur:
        return _verdict("quality", claim, "untested", "the fit check has no row for this model")
    src, trace = _fit_source(ctx)
    rel = bool(SUPERLATIVE.search(claim["text"]))
    if cur.get("n_ref"):
        def count(s):
            return round(s["correct_rate"] * s["n_ref"])
        best = max(stats, key=count)
        ok = (count(best) - count(cur) <= 1) if rel else cur["correct_rate"] >= 0.75
        detail = (f"correct on {count(cur)}/{cur['n_ref']} reference cases"
                  + (f", hard {cur['hard_correct']}/{cur['hard_total']}"
                     if cur.get("hard_total") else "")
                  + (f"; best tested: {best['name']} {count(best)}/{best['n_ref']}"
                     if best is not cur else "; the best setup tested"))
    else:
        from notbydefault.fit.stats import MEAN_TOLERANCE
        best = max(stats, key=lambda s: s.get("mean") or 0)
        ok = (best["mean"] - cur["mean"] <= MEAN_TOLERANCE) if rel else cur["pass_rate"] >= 0.8
        detail = (f"judge score {cur['mean']:.2f}/5, verifier pass {cur['pass_rate']:.0%}"
                  + (f"; best tested: {best['name']} {best['mean']:.2f}" if best is not cur else ""))
    return _verdict("quality", claim, "true" if ok else "false", detail, src, trace)


def check_latency(claim: dict, ctx: dict) -> dict:
    if not ctx["results"]:
        return _verdict("latency", claim, "untested",
                        f"not measured: run `notbydefault check {ctx['decision']['id']}`")
    cur, stats = _chosen_stats(ctx["results"])
    if not cur:
        return _verdict("latency", claim, "untested", "the fit check has no row for this model")
    alone = [s for s in stats if s.get("setup") == "alone" and s.get("avg_latency_s")]
    fastest = min(alone, key=lambda s: s["avg_latency_s"]) if alone else cur
    ok = cur["avg_latency_s"] <= fastest["avg_latency_s"] * 1.25
    src, trace = _fit_source(ctx)
    return _verdict("latency", claim, "true" if ok else "false",
                    f"measured {cur['avg_latency_s']:.1f}s per call"
                    + (f"; fastest tested: {fastest['name']} {fastest['avg_latency_s']:.1f}s"
                       if fastest is not cur else "; the fastest setup tested"), src, trace)


def documented_regions(ctx: dict) -> tuple[list[str] | None, str, str]:
    """(regions, or None when unknown or deployment-dependent; source; note)."""
    from notbydefault.stale import provider_facts
    d = ctx["decision"]
    if d["data"].get("kind") == "model" and ctx["client"] is not None:
        model, regions, source = provider_facts(d, ctx["client"].pricing(), ctx["catalog"])
        if model:
            return [r.upper() for r in regions], source, f"{model} runs on {model.split(':')[0]}"
    f = ctx["facts"].get(d["label"].lower())
    if not f:
        return None, "", "no documented data residency for it"
    if f.get("residency") == "deployment":
        return None, f.get("source", ""), f"depends on deployment ({f.get('how', '')})"
    if f.get("provider"):
        cfg = (ctx["catalog"].get("providers") or {}).get(f["provider"], {})
        return ([str(r).upper() for r in cfg.get("data_residency") or []],
                str(cfg.get("data_residency_source", "")), f"provider {f['provider']}")
    return [str(r).upper() for r in f.get("residency") or []], f.get("source", ""), ""


def check_compliance(claim: dict, ctx: dict) -> dict:
    regions, source, note = documented_regions(ctx)
    named = [r for r, rx in REGION_NAMES.items() if re.search(rx, claim["text"], re.IGNORECASE)]
    if regions is None:
        return _verdict("compliance", claim, "untested", note, source)
    if not regions:
        return _verdict("compliance", claim, "untested",
                        f"{note + ': ' if note else ''}no documented data-residency guarantee, so "
                        "it is neither shown nor ruled out", source)
    where = ", ".join(regions)
    if not named:
        return _verdict("compliance", claim, "untested",
                        f"names no region; documented data residency: {where}", source)
    missing = [r for r in named if r not in regions]
    return _verdict("compliance", claim, "false" if missing else "true",
                    f"{note + ': ' if note else ''}documented data residency: {where}", source)


def check_claim(claim: dict, ctx: dict) -> dict:
    kind = claim["type"]
    if kind == "cost":
        return check_cost(claim, ctx)
    if kind == "quality":
        return check_quality(claim, ctx)
    if kind == "latency":
        return check_latency(claim, ctx)
    if kind == "compliance":
        return check_compliance(claim, ctx)
    if kind == "popularity":
        return _verdict(kind, claim, "untested",
                        "not evidence for your project: how many others use it says nothing about "
                        "your task, volume, budget or region")
    return _verdict(kind, claim, "untested",
                    "nothing here measures maturity; check release history and support terms "
                    "yourself")


# --- run ---------------------------------------------------------------------------------

def evaluate(project: Path, graph, decision: dict, client=None, entire=None, tracer=None,
             split: bool = True) -> dict | None:
    """The claim check for one decision (not stored), or None when no reason was stated."""
    reason, source = stated_reason(project, decision, entire)
    if not reason:
        return None
    ctx = load_context(project, graph, decision, client)
    caller = model = None
    if split and ctx["client"] is not None and hasattr(ctx["client"], "chat"):
        from notbydefault.fit.evaluate import Caller, SpendGuard
        from notbydefault.fit.providers import role_model
        from notbydefault.fit.tracing import Tracer
        try:
            model = role_model(ctx["catalog"], "pivot")
            caller = Caller(ctx["client"], SpendGuard(0.05), tracer or Tracer())
        except Exception:    # noqa: BLE001  (no role configured)
            caller = model = None
    claims, how = split_claims(reason, decision, caller, model, tracer)
    items = [check_claim(c, ctx) for c in claims]
    return {"reason": reason, "reason_source": source, "classified_by": how, "items": items,
            "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "results": ctx["results_rel"], "trace_id": getattr(tracer, "trace_id", None)}


def counts(claims: dict | None) -> dict[str, int]:
    items = (claims or {}).get("items") or []
    return {v: sum(i["verdict"] == v for i in items) for v in ("true", "false", "untested")}


def block_lines(claims: dict | None, markdown: bool = False) -> list[str]:
    """The "Claim check" block, as the check report, decision record and PR show it."""
    if not claims:
        return []
    c = counts(claims)
    head = (f"Claim check: the agent's reason ({claims.get('reason_source') or 'recorded'}) makes "
            f"{len(claims['items'])} claim(s): {c['true']} true, {c['false']} false, "
            f"{c['untested']} untested (split by {claims.get('classified_by')}).")
    reason = " ".join(claims["reason"].split())
    lines = [head, f'Reason: "{reason[:400]}{"..." if len(reason) > 400 else ""}"']
    for i in claims["items"]:
        src = "; ".join(x for x in (i.get("source"), f"trace {i['trace_id']}" if i.get("trace_id")
                                    else "") if x)
        if markdown:
            lines.append(f"- **{CLAIM_MARK[i['verdict']]}** ({i['type']}) \"{i['text']}\": "
                         f"{i['detail']}" + (f" [{src}]" if src else ""))
        else:
            lines.append(f"  {CLAIM_MARK[i['verdict']]:<9} {i['type']:<10} \"{i['text']}\"")
            lines.append(f"            {i['detail']}" + (f" [{src}]" if src else ""))
    if not claims["items"]:
        lines.append("  (no claim found in the reason)")
    return lines


def store(project: Path, graph, decision_id: int, claims: dict) -> None:
    from notbydefault.records import write_record
    graph.update_data(decision_id, claims=claims)
    c = counts(claims)
    write_record(project, graph, decision_id,
                 f"claim check: {c['true']} true, {c['false']} false, {c['untested']} untested")


def run(project: Path, decision_id: int, client=None, entire=None, tracer=None,
        out=print) -> int:
    """`notbydefault claims <id>`."""
    from notbydefault.entire import Entire
    from notbydefault.fit.tracing import make_tracer
    from notbydefault.graph import Graph, db_path
    project = Path(project).resolve()
    if not db_path(project).exists():
        out(f"No Decision Graph at {db_path(project)}.")
        return 1
    if client is None:
        from notbydefault.env import load_env
        from notbydefault.fit.providers import ProviderSet, load_catalog
        load_env()
        client = ProviderSet(load_catalog(project), cache_dir=project / ".notbydefault" / "cache")
    tracer = tracer or make_tracer()
    if hasattr(client, "on_event"):
        client.on_event = tracer.event
    with Graph.for_project(project) as g:
        d = g.get_node(decision_id)
        if not d or d["type"] != "decision":
            out(f"No decision #{decision_id}")
            return 1
        with tracer.root("notbydefault.claims", {"notbydefault.decision_id": decision_id}):
            with tracer.step("claim_check", "verification",
                             reason="split the agent's stated reason into claims and check each "
                                    "against the evidence we have"):
                claims = evaluate(project, g, d, client, entire or Entire(), tracer)
                if claims:
                    claims["trace_id"] = tracer.trace_id
                    store(project, g, decision_id, claims)
    if not claims:
        out(f"#{decision_id} {d['label']}: no stated reason recorded (the hook records the agent's "
            "words from now on; none were found in the Entire transcripts either).")
        tracer.flush()
        return 0
    out(f"#{decision_id} {d['label']}")
    out("\n".join(block_lines(claims)))
    if tracer.trace_id:
        out(f"\nneatlogs trace id: {tracer.trace_id} (workflow notbydefault)")
    tracer.flush()
    return 0
