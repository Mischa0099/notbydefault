"""Stale decisions: `notbydefault require <project> "<new requirement>"`.

Adds a requirement node, turns it into machine-checkable constraints, and sorts every existing
decision into one of three lists:
  stale         it provably conflicts: a documented fact breaks the requirement (e.g. the
                provider documents that data stays in the US and EU residency is required)
  needs review  it might conflict, but that depends on how it is deployed (self-hosted MongoDB
                can run in the EU), or only the model's judgement says so (nothing documented)
  still valid   neither
Each affected decision gets the reason, its basis and source, and an edge to the requirement.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from notbydefault.graph import Graph, db_path, find_affected_code
from notbydefault.llm import LLMError, parse_json

EXTRACT_SYSTEM = ("You turn a product requirement into machine-checkable constraints. "
                  "Answer with one JSON object only.")
EXTRACT_PROMPT = """Requirement: {text}

Return JSON with these keys (null when the requirement says nothing about it):
{{"data_residency": "<region code the data must stay in, e.g. EU, US, UK, or null>",
  "max_monthly_cost_usd": <number or null>,
  "summary": "<the requirement in at most 12 words>"}}"""

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
Return {{"affected": true or false, "reason": "<one line>"}}"""


def constraints_of(node: dict) -> dict:
    return (node.get("data") or {}).get("constraints") or {}


def provider_facts(decision: dict, pricing, catalog: dict) -> tuple[str | None, list[str], str]:
    """(model id, guaranteed residency regions, source) for a model decision; (None, [], "")
    when the decision isn't a model or its provider isn't in the catalog."""
    from notbydefault.fit.candidates import resolve_model_id
    from notbydefault.fit.providers import residency
    if decision["data"].get("kind") != "model":
        return None, [], ""
    try:
        model = resolve_model_id(decision["label"], pricing)
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
    return {"status": "stale", "source": source,
            "reason": f"{f['name']} documents data residency: "
                      f"{', '.join(regions) or 'no region guaranteed'} (source: {source}); "
                      f"{region} residency is required"}


def residency_violation(decision: dict, constraints: dict, pricing, catalog: dict) -> str | None:
    """A documented reason the decision breaks the residency constraint, or None."""
    region = (constraints.get("data_residency") or "").upper()
    if not region:
        return None
    model, regions, source = provider_facts(decision, pricing, catalog)
    if model and region not in regions:
        where = ", ".join(regions) if regions else "no region guaranteed"
        return (f"{model} runs on {model.split(':', 1)[0]}, which documents data residency: "
                f"{where} (source: {source}); {region} residency is required")
    return None


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
    return {"affected": bool(out.get("affected")), "reason": str(out.get("reason") or "").strip()}


def classify(decision: dict, constraints: dict, verdict: dict, pricing, catalog: dict,
             facts: dict) -> dict:
    """{"status": "stale"|"needs_review"|"valid"|"unclear", "reason", "basis", "source"}.

    Only a documented fact makes a decision stale; the model's judgement alone can at most ask
    for a review, so a guess never invalidates a decision.
    """
    documented = residency_violation(decision, constraints, pricing, catalog)
    if documented:
        _m, _r, source = provider_facts(decision, pricing, catalog)
        return {"status": "stale", "reason": documented, "basis": "provider documentation",
                "source": source}
    svc = service_residency(decision, constraints, catalog, facts)
    if svc:
        basis = ("service documentation" if svc["status"] == "stale"
                 else "depends on deployment")
        return {**svc, "basis": basis}
    if verdict.get("affected") is None:
        return {"status": "unclear", "reason": verdict.get("reason", ""), "basis": "", "source": ""}
    if verdict["affected"]:
        return {"status": "needs_review", "basis": "model judgement (nothing documented)",
                "source": "", "reason": verdict.get("reason") or "the judge thinks it is affected"}
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
            except (LLMError, BudgetExceeded) as e:
                tracer.event("extract_failed", "recovery",
                             f"could not extract constraints ({e}); judging with the text only")
                raw = {}
            constraints = {k: v for k, v in raw.items()
                           if k in ("data_residency", "max_monthly_cost_usd") and v not in (None, "")}
        with tracer.step("add_requirement", "state", reason="record the new requirement"):
            rid = g.add_node("requirement", text, {
                "source": "require", "constraints": constraints, "added_at": now,
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

        def one(d: dict) -> dict:
            m, regions, source = provider_facts(d, pricing, catalog)
            facts = (f"provider model {m}; documented data residency: "
                     f"{', '.join(regions) or 'none guaranteed'} ({source})") if m else ""
            try:
                verdict = judge_decision(caller, model, d, text, constraints, facts)
            except (LLMError, BudgetExceeded) as e:
                verdict = {"affected": None, "reason": f"could not judge: {e}"}
                tracer.event("judge_failed", "recovery",
                             f"decision #{d['id']}: {e}; left unchanged", decision_id=d["id"])
            c = classify(d, constraints, verdict, pricing, catalog, service_facts)
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
                "basis": r["basis"], "source": r["source"]} for r in results})
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
