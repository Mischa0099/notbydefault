"""Stale decisions: `notbydefault require <project> "<new requirement>"`.

Adds a requirement node, turns it into machine-checkable constraints, and sorts every existing
decision into one of three lists:
  stale         it provably conflicts: a documented fact breaks the requirement (e.g. the
                provider documents that data stays in the US and EU residency is required), or
                list price x recorded volume exceeds the budget (a budget cut, more users)
  needs review  it might conflict, but that depends on how it is deployed (self-hosted MongoDB
                can run in the EU), the provider documents no residency guarantee at all
                (neither ruled in nor out), or only the model's judgement says so
  still valid   neither
Each affected decision gets the reason, its basis and source, and an edge to the requirement.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from notbydefault.graph import Graph, db_path, find_affected_code
from notbydefault.llm import LLMError, parse_json

EXTRACT_SYSTEM = ("You turn a product requirement into machine-checkable constraints. "
                  "Answer with one JSON object only.")
EXTRACT_PROMPT = """Requirement: {text}

Return JSON with these keys. Use null unless the requirement itself states the value; never infer
one (a compliance regime such as HIPAA, SOC 2 or PCI DSS names no region and no budget):
{{"data_residency": "<region code the requirement says the data must stay in, e.g. EU, US, UK, or null>",
  "max_monthly_cost_usd": <the monthly budget in USD the requirement states, or null>,
  "expected_users": <the number of users the requirement states, or null>,
  "summary": "<the requirement in at most 12 words>"}}"""

# A constraint is kept only when the requirement's own words state it (the model may not infer).
REGION_CODES = {"EU": ("EU", "EEA"), "US": ("US", "USA", "U.S.", "U.S.A."), "UK": ("UK",)}
REGION_WORDS = {"EU": r"europe(?:an)?(?: union)?", "US": r"united states|america",
                "UK": r"united kingdom|britain|british"}
LOCATION_WORDS = re.compile(r"\b(stay|stays|stored?|stor(?:age|ing)|resid(?:e|es|ency|ent)|located|"
                            r"locat(?:ion|ed)|kept|keep|hosted|host|remain|within|sovereign(?:ty)?|"
                            r"region|leave|inside)\b", re.IGNORECASE)
NUMBER = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*([kKmM]\b)?")
MULTIPLIER = re.compile(r"\b(\d+(?:\.\d+)?)\s*x\b", re.IGNORECASE)


def _numbers(text: str) -> set[float]:
    return {float(n.replace(",", "")) * {"k": 1e3, "m": 1e6}.get((u or "").lower(), 1)
            for n, u in NUMBER.findall(text)}


def states_region(text: str, region: str) -> bool:
    """Does the requirement's text name this region as where data must be?"""
    region = region.upper()
    codes = REGION_CODES.get(region, (region,))
    named = any(re.search(rf"(?<![\w.]){re.escape(c)}(?![\w])", text) for c in codes) or bool(
        REGION_WORDS.get(region) and re.search(rf"\b(?:{REGION_WORDS[region]})\b", text, re.I))
    return named and bool(LOCATION_WORDS.search(text))


def grounded(raw: dict, text: str, recorded_users: float | None = None) -> tuple[dict, list[str]]:
    """The extracted constraints the text actually states, and what was dropped (and why)."""
    out, dropped = {}, []
    numbers = _numbers(text)
    region = raw.get("data_residency")
    if region not in (None, ""):
        if states_region(text, str(region)):
            out["data_residency"] = str(region).upper()
        else:
            dropped.append(f"data_residency={region} (the requirement names no such region)")
    for key in ("max_monthly_cost_usd", "expected_users"):
        value = raw.get(key)
        if value in (None, ""):
            continue
        try:
            value = float(str(value).replace(",", "").replace("$", ""))
        except ValueError:
            dropped.append(f"{key}={value} (not a number)")
            continue
        ok = value in numbers
        if key == "expected_users" and not ok and recorded_users:   # "10x the users"
            ok = any(abs(float(k) * recorded_users - value) < 1 for k in MULTIPLIER.findall(text))
        if ok:
            out[key] = int(value) if value.is_integer() else value
        else:
            dropped.append(f"{key}={value:g} (not stated in the requirement)")
    if "expected_users" not in out and recorded_users and re.search(r"\buser", text, re.I):
        for k in MULTIPLIER.findall(text):
            out["expected_users"] = int(float(k) * recorded_users)
            break
    return out, dropped


JUDGE_SYSTEM = ("You check whether an existing technical decision still holds after a new "
                "requirement. Answer with one JSON object only.")
JUDGE_PROMPT = """New requirement: {text}
Machine-readable constraints: {constraints}

Existing decision #{id}: {label}
Kind: {kind}; category: {category}; status: {status}
What it is used for: {used_for}
Code it affects: {files}
Known facts: {facts}

Is this decision affected, i.e. must it be re-evaluated because of the new requirement?
It is affected only if the requirement touches a concrete property of THIS decision itself
(e.g. "stores personal data", "sends user data to a third-party API", "cost scales with call
volume", "runs only in one region"), not of other parts of the system. If you cannot name such
a property, it is not affected.
Return {{"affected": true or false,
  "property": "<the concrete property of this decision the requirement touches, or empty>",
  "reason": "<one line>"}}"""


def constraints_of(node: dict) -> dict:
    return (node.get("data") or {}).get("constraints") or {}


def provider_facts(decision: dict, pricing, catalog: dict) -> tuple[str | None, list[str], str]:
    """(model id, guaranteed residency regions, source) for a model decision; (None, [], "")
    when the decision isn't a model or its provider isn't in the catalog. A resolved decision
    is judged on the model it now uses (`resolution.to`)."""
    from notbydefault.fit.candidates import resolve_model_id
    from notbydefault.fit.providers import residency
    if decision["data"].get("kind") != "model":
        return None, [], ""
    try:
        current = (decision["data"].get("resolution") or {}).get("to") or decision["label"]
        model = resolve_model_id(current, pricing)
    except LookupError:
        return None, [], ""
    cfg = catalog["providers"].get(model.split(":", 1)[0], {})
    return model, residency(catalog, model), str(cfg.get("data_residency_source", ""))


def load_service_facts(project: Path | None = None) -> dict:
    """{service name lowercased: facts} from service_facts.yaml (+ the project's override)."""
    import yaml
    from notbydefault import PACKAGE_DIR
    facts = {}
    paths = [PACKAGE_DIR / "service_facts.yaml"]
    if project is not None:
        paths.append(Path(project) / ".notbydefault" / "service_facts.yaml")
    for path in paths:
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            for name, f in (data.get("services") or {}).items():
                facts[str(name).lower()] = dict(f or {}, name=name)
    return facts


def service_residency(decision: dict, constraints: dict, catalog: dict,
                      facts: dict) -> dict | None:
    """Documented residency verdict for a service decision:
    {"status": "stale"|"needs_review", "reason", "source"} or None (no fact / no constraint)."""
    region = (constraints.get("data_residency") or "").upper()
    if not region or decision["data"].get("kind") != "service":
        return None
    f = facts.get(decision["label"].lower())
    if not f:
        return None
    if f.get("residency") == "deployment":
        return {"status": "needs_review", "source": f.get("source", ""),
                "reason": f"{f['name']}: where its data lives depends on how it is deployed "
                          f"({f.get('how', 'not documented')}); confirm it runs in {region}"}
    if f.get("provider"):
        cfg = (catalog.get("providers") or {}).get(f["provider"], {})
        regions = [str(r).upper() for r in cfg.get("data_residency") or []]
        source = str(cfg.get("data_residency_source", ""))
    else:
        regions = [str(r).upper() for r in f.get("residency") or []]
        source = f.get("source", "")
    if region in regions:
        return None
    if not regions:
        return {"status": "needs_review", "source": source,
                "reason": f"{f['name']} documents no data-residency guarantee (source: {source}); "
                          f"confirm with the vendor that {region} residency is possible"}
    return {"status": "stale", "source": source,
            "reason": f"{f['name']} documents data residency: "
                      f"{', '.join(regions) or 'no region guaranteed'} (source: {source}); "
                      f"{region} residency is required"}


def residency_violation(decision: dict, constraints: dict, pricing, catalog: dict) -> str | None:
    """A documented reason the decision breaks the residency constraint (the provider names
    its regions and the required one isn't among them), or None."""
    region = (constraints.get("data_residency") or "").upper()
    if not region:
        return None
    model, regions, source = provider_facts(decision, pricing, catalog)
    if model and regions and region not in regions:
        return (f"{model} runs on {model.split(':', 1)[0]}, which documents data residency: "
                f"{', '.join(regions)} (source: {source}); {region} residency is required")
    return None


def residency_unguaranteed(decision: dict, constraints: dict, pricing, catalog: dict) -> str | None:
    """The provider documents no residency guarantee at all: it may or may not comply."""
    region = (constraints.get("data_residency") or "").upper()
    if not region:
        return None
    model, regions, source = provider_facts(decision, pricing, catalog)
    if model and not regions:
        return (f"{model} runs on {model.split(':', 1)[0]}, which documents no data-residency "
                f"guarantee (source: {source}); confirm {region} processing with the provider "
                "(e.g. a regional or enterprise endpoint) or switch")
    return None


def cost_fact(graph, project: Path, decision: dict, client, constraints: dict) -> dict | None:
    """Projected monthly cost of a model decision under the requirement (Pivot Check's math:
    list price x recorded calls per user x users), or None when the requirement changes neither
    the budget nor the number of users, or no price is known."""
    if decision["data"].get("kind") != "model":
        return None
    budget, users = constraints.get("max_monthly_cost_usd"), constraints.get("expected_users")
    if budget is None and users is None:
        return None
    from notbydefault.pivot import cost_check, list_price
    price = list_price(decision["label"], client, project)
    if not price:
        return None
    return cost_check(graph, project, [price], [decision["id"]], users=users, budget=budget)


def judge_decision(caller, model: str, decision: dict, text: str, constraints: dict,
                   facts: str) -> dict:
    data = decision["data"]
    reply = caller.chat(model, [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_PROMPT.format(
            text=text, constraints=json.dumps(constraints), id=decision["id"],
            label=decision["label"], kind=data.get("kind"), category=data.get("category"),
            status=data.get("status"),
            used_for=data.get("verdict_summary") or data.get("evidence") or "not recorded",
            files=", ".join(find_affected_code(decision)) or "none recorded",
            facts=facts or "none")},
    ], span="notbydefault.stale_judge", role="stale_judge", max_tokens=1500, json_mode=True,
        focus_area="verification", attrs={"notbydefault.decision_id": decision["id"]})
    out = parse_json(reply.text)
    prop = str(out.get("property") or "").strip()
    reason = str(out.get("reason") or "").strip()
    if out.get("affected") and not prop:    # no concrete property named: not affected
        return {"affected": False, "property": "", "no_property": True,
                "reason": "the judge named no property of this decision that the requirement "
                          f"touches (it said: {reason or 'affected'})"}
    return {"affected": bool(out.get("affected")), "property": prop, "reason": reason}


def classify(decision: dict, constraints: dict, verdict: dict, pricing, catalog: dict,
             facts: dict, cost: dict | None = None) -> dict:
    """{"status": "stale"|"needs_review"|"valid"|"unclear", "reason", "basis", "source"}.

    Only a documented fact (provider docs, or list price x recorded volume over the budget)
    makes a decision stale; the model's judgement alone can at most ask for a review, so a
    guess never invalidates a decision.
    """
    documented = residency_violation(decision, constraints, pricing, catalog)
    if documented:
        _m, _r, source = provider_facts(decision, pricing, catalog)
        return {"status": "stale", "reason": documented, "basis": "provider documentation",
                "source": source}
    if cost and cost["over_budget"]:
        from notbydefault.pivot import cost_line
        return {"status": "stale", "reason": cost_line(cost), "basis": "list price x recorded volume",
                "source": f"{cost['price']['source']} ({cost['price']['fetched']})"}
    unguaranteed = residency_unguaranteed(decision, constraints, pricing, catalog)
    if unguaranteed:
        _m, _r, source = provider_facts(decision, pricing, catalog)
        return {"status": "needs_review", "reason": unguaranteed,
                "basis": "no documented residency guarantee", "source": source}
    svc = service_residency(decision, constraints, catalog, facts)
    if svc:
        basis = ("service documentation" if svc["status"] == "stale"
                 else "no documented residency guarantee" if "no data-residency guarantee"
                 in svc["reason"] else "depends on deployment")
        return {**svc, "basis": basis}
    if verdict.get("affected") is None:
        return {"status": "unclear", "reason": verdict.get("reason", ""), "basis": "", "source": ""}
    if verdict["affected"]:
        prop = verdict.get("property") or ""
        reason = verdict.get("reason") or "the judge thinks it is affected"
        return {"status": "needs_review", "basis": "model judgement (nothing documented)",
                "source": "", "property": prop, "reason": f"{prop}: {reason}" if prop else reason}
    return {"status": "valid", "reason": verdict.get("reason", ""), "basis": "", "source": ""}


def require(project: Path, text: str, client=None, tracer=None, out=print,
            workers: int = 4) -> int:
    project = Path(project).resolve()
    if not db_path(project).exists():
        out(f"No Decision Graph at {db_path(project)}. Run `init` first.")
        return 1
    from notbydefault.env import load_env
    from notbydefault.fit.evaluate import BudgetExceeded, Caller, SpendGuard
    from notbydefault.fit.providers import ProviderSet, load_catalog, role_model
    from notbydefault.fit.tracing import make_tracer

    if client is None:
        load_env()
        client = ProviderSet(load_catalog(project), cache_dir=project / ".notbydefault" / "cache")
    catalog = client.catalog
    tracer = tracer or make_tracer()
    if hasattr(client, "on_event"):
        client.on_event = tracer.event
    caller = Caller(client, SpendGuard(0.50), tracer)
    model = role_model(catalog, "stale_judge")
    pricing = client.pricing()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results = []
    with Graph.for_project(project) as g, tracer.root("notbydefault.require",
                                                     {"notbydefault.requirement": text}):
        with tracer.step("extract_constraints", "tool_use",
                         reason="turn the requirement into checks the evaluator can enforce"):
            try:
                reply = caller.chat(model, [{"role": "system", "content": EXTRACT_SYSTEM},
                                            {"role": "user", "content": EXTRACT_PROMPT.format(text=text)}],
                                    span="notbydefault.extract_constraints", role="stale_judge",
                                    max_tokens=1000, json_mode=True, focus_area="tool_use")
                raw = parse_json(reply.text)
                extracted_by = "model"
            except (LLMError, BudgetExceeded) as e:
                # the region the text itself names still counts (grounded() keeps only that)
                raw = {"data_residency": next((r for r in REGION_CODES if states_region(text, r)),
                                              None)}
                extracted_by = "regex-fallback"
                tracer.event("extract_failed", "recovery",
                             f"could not extract constraints ({e}); judging with the text only"
                             + (f" (data residency {raw['data_residency']} read from the text)"
                                if raw["data_residency"] else ""))
            from notbydefault.pivot import volume
            constraints, dropped = grounded(raw, text, volume(g)["users"])
            if dropped:
                tracer.event("constraint_dropped", "verification",
                             "the model extracted constraints the requirement does not state: "
                             + "; ".join(dropped), dropped=dropped)
        with tracer.step("add_requirement", "state", reason="record the new requirement"):
            rid = g.add_node("requirement", text, {
                "source": "require", "constraints": constraints,
                "constraints_source": extracted_by, "added_at": now,
                "summary": str(raw.get("summary") or "")})
            tracer.event("requirement_added", "state", f"#{rid}: {text}", requirement_id=rid,
                         constraints=constraints)
        with tracer.step("affected_code", "context",
                         reason="find every function each decision touches (Entire Graph), so "
                                "the judge and the report see all of it"):
            from notbydefault.affected import refresh as refresh_affected
            refresh_affected(project, graph=g, tracer=tracer)
        decisions = g.nodes("decision")
        service_facts = load_service_facts(project)
        costs = {}
        with tracer.step("cost_facts", "context",
                         reason="list price x recorded volume for each model decision, when the "
                                "requirement changes the budget or the number of users"):
            for d in decisions:
                c = cost_fact(g, project, d, client, constraints)
                if c:
                    from notbydefault.pivot import cost_line
                    costs[d["id"]] = c
                    tracer.event("cost_fact", "verification", f"#{d['id']}: {cost_line(c)}",
                                 decision_id=d["id"], monthly=round(c["monthly"], 2),
                                 budget=c["budget"], over_budget=c["over_budget"])

        def one(d: dict) -> dict:
            m, regions, source = provider_facts(d, pricing, catalog)
            facts = (f"provider model {m}; documented data residency: "
                     f"{', '.join(regions) or 'none guaranteed'} ({source})") if m else ""
            if d["id"] in costs:
                from notbydefault.pivot import cost_line
                facts = "; ".join(x for x in (facts, cost_line(costs[d["id"]])) if x)
            try:
                verdict = judge_decision(caller, model, d, text, constraints, facts)
            except (LLMError, BudgetExceeded) as e:
                verdict = {"affected": None, "reason": f"could not judge: {e}"}
                tracer.event("judge_failed", "recovery",
                             f"decision #{d['id']}: {e}; left unchanged", decision_id=d["id"])
            c = classify(d, constraints, verdict, pricing, catalog, service_facts, costs.get(d["id"]))
            if verdict.get("no_property"):
                tracer.event("no_property_valid", "verification",
                             f"#{d['id']}: the judge said affected but named no concrete property "
                             "of the decision; counting it as valid", decision_id=d["id"])
            if verdict.get("affected") and c["status"] == "needs_review" \
                    and c["basis"].startswith("model"):
                tracer.event("downgrade_to_review", "verification",
                             f"#{d['id']}: the judge says affected but nothing documented proves "
                             "a conflict; asking for review instead of marking it stale",
                             decision_id=d["id"])
            return {"decision": d, "affected": verdict.get("affected"), **c}

        with tracer.step("judge_decisions", "verification",
                         reason=f"check {len(decisions)} decisions against the new requirement"):
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(tracer.bind(one), decisions))
        with tracer.step("mark_stale", "state",
                         reason="mark provable conflicts stale and the rest needs-review"):
            for r in results:
                d = r["decision"]
                if r["status"] == "stale":
                    g.update_data(d["id"], status="stale", stale_reason=r["reason"],
                                  stale_basis=r["basis"], stale_source=r["source"],
                                  stale_since=now, stale_cause=rid,
                                  status_before_stale=d["data"].get("status"))
                elif r["status"] == "needs_review":
                    g.update_data(d["id"], status="needs_review", review_reason=r["reason"],
                                  review_basis=r["basis"], review_source=r["source"],
                                  review_since=now, review_cause=rid,
                                  status_before_review=d["data"].get("status"))
                else:
                    continue
                g.add_edge(d["id"], rid, "depends_on")
                tracer.event("decision_" + r["status"], "planning",
                             f"#{d['id']} {d['label']}: {r['reason']} ({r['basis']})",
                             decision_id=d["id"], requirement_id=rid)
            g.update_data(rid, judged={str(r["decision"]["id"]): {
                "status": r["status"], "affected": r["affected"], "reason": r["reason"],
                "basis": r["basis"], "source": r["source"],
                "property": r.get("property", "")} for r in results})
        changed = [r for r in results if r["status"] in ("stale", "needs_review")]
        brain_notes: list[str] = []
        if changed:
            with tracer.step("decision_records", "state",
                             reason="write each verdict into its decision record (and Brain)"):
                from notbydefault.records import write_records
                for r in changed:
                    write_records(project, g, [r["decision"]["id"]],
                                  f"{r['status'].replace('_', ' ')} after requirement #{rid} "
                                  f"\"{text}\": {r['reason']}", tracer, out=brain_notes.append)
        for r in results:   # reload so the report shows what was stored
            r["decision"] = g.get_node(r["decision"]["id"])

    report(rid, text, constraints, results, out)
    recs = [r["decision"]["data"].get("decision_record") for r in results
            if r["status"] in ("stale", "needs_review")]
    if recs:
        out("\nDecision records updated: " + ", ".join(sorted({x for x in recs if x})))
        for note in dict.fromkeys(brain_notes):
            out(note)
    if tracer.trace_id:
        out(f"\nneatlogs trace id: {tracer.trace_id} (workflow notbydefault)")
    tracer.flush()
    return 0


def _code(d: dict) -> str:
    from notbydefault.affected import describe
    from notbydefault.graph import affected_entries
    entries = affected_entries(d)
    shown = [describe(e) for e in entries[:6]]
    if len(entries) > 6:
        shown.append(f"+{len(entries) - 6} more")
    return ", ".join(shown) or "no code recorded"


def report(rid: int, text: str, constraints: dict, results: list[dict], out) -> None:
    by = {s: [r for r in results if r["status"] == s]
          for s in ("stale", "needs_review", "unclear", "valid")}
    out(f"Requirement #{rid}: {text}")
    if constraints:
        out(f"  checks: {json.dumps(constraints)}")
    if not by["stale"] and not by["needs_review"]:
        out(f"\nNo decisions are affected ({len(results)} checked).")
    for status, title in (("stale", "provably conflict (documented), now STALE"),
                          ("needs_review", "NEED REVIEW (may conflict; not provable from the docs)")):
        if not by[status]:
            continue
        out(f"\n{len(by[status])} of {len(results)} decisions {title}:")
        for r in by[status]:
            d = r["decision"]
            out(f"  #{d['id']} {d['label']} [{d['data'].get('kind')}/{d['data'].get('category')}]")
            out(f"      why:   {r['reason']}")
            out(f"      basis: {r['basis']}" + (f" ({r['source']})" if r["source"] else ""))
            out(f"      code:  {_code(d)}")
    for r in by["unclear"]:
        out(f"  #{r['decision']['id']} {r['decision']['label']}: {r['reason']}")
    ok = by["valid"]
    if ok:
        out("\nStill valid:")
        out("\n".join(f"  #{r['decision']['id']} {r['decision']['label']}: {r['reason']}" for r in ok))
    if by["stale"]:
        out("\nRe-test a stale model decision with `python -m notbydefault check <id>`; it "
            "applies the new requirement's checks to the candidates.")
    if by["needs_review"]:
        out("Needs-review decisions are not invalid: confirm how each is deployed (or that the "
            "judge is wrong) and record it in its decision record.")
