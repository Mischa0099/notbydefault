"""`check <id>` for model decisions: verifiers, setups, judge, verdict, graph + record."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from notbydefault.fit import callsite as callsite_mod
from notbydefault.fit import candidates as candidates_mod
from notbydefault.fit import failures as failures_mod
from notbydefault.fit import labels as labels_mod
from notbydefault.fit import samples as samples_mod
from notbydefault.fit import verifiers as verifiers_mod
from notbydefault.fit.evaluate import ALONE, VERIFIED, BudgetExceeded, Caller, SpendGuard, judge, \
    run_setups
from notbydefault.fit.report import ReportContext, render, write_decision_record
from notbydefault.fit.stats import Verdict, decide, parse_number, project_monthly, summarize
from notbydefault.fit.tracing import Tracer, make_tracer
from notbydefault.graph import Graph, find_affected_code

DEFAULT_CALLS_PER_USER = 30
EXPECTED_OUT_TOKENS = 350     # cost estimate only; the spend guard uses worst cases
EXPECTED_EXTRA_ATTEMPTS = 0.5  # cost estimate only: average retries per verified call
SETUPS = (ALONE, VERIFIED)


@dataclass
class CheckOptions:
    max_cost: float = 1.0
    yes: bool = False
    calls_per_user: float | None = None
    n_samples: int = 20
    workers: int = 6
    seed: int = 0


def graph_context(g: Graph, decision: dict) -> dict:
    deps = g.get_dependencies(decision["id"])
    by_key = {n["data"].get("key"): n for n in deps}
    return {
        "goal": next((n["label"] for n in deps if n["type"] == "goal"), None),
        "constraints": [n["label"] for n in deps if n["type"] in ("requirement", "constraint")],
        "files": find_affected_code(decision),
        "users": parse_number((by_key.get("users") or {}).get("data", {}).get("value")),
        "budget": parse_number((by_key.get("budget") or {}).get("data", {}).get("value")),
        "calls_per_user_node": by_key.get("calls_per_user"),
    }


def resolve_calls_per_user(g: Graph, decision: dict, ctx: dict, opts: CheckOptions,
                           ask) -> float:
    """Ask once (default 30); stored as an assumption requirement node in the graph."""
    node = ctx["calls_per_user_node"]
    if opts.calls_per_user is None and node:
        return float(node["data"]["value"])
    value = opts.calls_per_user
    if value is None:
        answer = ""
        if not opts.yes:
            try:
                answer = ask(f"How many calls to this {decision['label']} feature per user per "
                             f"month? [{DEFAULT_CALLS_PER_USER}] ").strip()
            except EOFError:
                answer = ""
        value = parse_number(answer) or DEFAULT_CALLS_PER_USER
    label = f"Calls per user per month: {value:g} (assumption)"
    data = {"source": "check", "key": "calls_per_user", "value": value, "assumption": True}
    if node:
        g.update_data(node["id"], **data)
        g.conn.execute("UPDATE nodes SET label = ? WHERE id = ?", (label, node["id"]))
        g.conn.commit()
    else:
        nid = g.add_node("requirement", label, data)
        g.add_edge(decision["id"], nid, "depends_on")
    return value


def estimate_cost(pricing, cfg: dict, site, have_site: bool, have_verifiers: bool,
                  inputs: list[str] | None, n_inputs: int, system_prompt: str) -> float:
    analyst = pricing.price(cfg["judge"])
    snippet_tokens = len(site.snippet) / 3.5
    total = 0.0
    if not have_site:
        total += analyst.cost(snippet_tokens + 400, 700)
    if not have_verifiers:
        total += analyst.cost(snippet_tokens + 1200, 700)
    if inputs is None:
        total += analyst.cost(700, n_inputs * 150)
    avg_in = (sum(len(i) for i in inputs) / len(inputs) / 3.5 if inputs else 200) \
        + len(system_prompt) / 3.5 + 10
    attempts = 1 + EXPECTED_EXTRA_ATTEMPTS
    verifier = pricing.price(cfg["verifier_model"])
    for m in cfg["candidates"]:
        p = pricing.price(m)
        total += n_inputs * attempts * p.cost(avg_in * 1.5, EXPECTED_OUT_TOKENS)
        total += n_inputs * attempts * verifier.cost(avg_in + EXPECTED_OUT_TOKENS + 150, 60)
    outputs = 1.5 * len(cfg["candidates"])
    total += n_inputs * pricing.price(cfg["judge"]).cost(
        700 + avg_in + outputs * EXPECTED_OUT_TOKENS, 60 * outputs + 50)
    return total


def assemble_inputs(project: Path, decision_id: int, samples: list[dict]) -> list[dict]:
    """Failure-bank cases first, then the sample set (deduplicated)."""
    rows = [{"user_message": t, "source": "failure-bank"}
            for t in failures_mod.inputs(project, decision_id)]
    seen = {r["user_message"] for r in rows}
    return rows + [s for s in samples if s["user_message"] not in seen]


def _counts(inputs: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in inputs:
        out[r.get("source", "provided")] = out.get(r.get("source", "provided"), 0) + 1
    return out


def check_model_decision(project: Path, decision_id: int, opts: CheckOptions, client=None,
                         tracer: Tracer | None = None, ask=input, out=print) -> int:
    project = Path(project).resolve()
    with Graph.for_project(project) as g:
        decision = g.get_node(decision_id)
        if decision is None or decision["type"] != "decision":
            out(f"No decision #{decision_id} in {project}")
            return 1
        if decision["data"].get("kind") != "model":
            out(f"Decision #{decision_id}: {decision['label']} "
                f"[{decision['data'].get('kind')}/{decision['data'].get('category')}]")
            out("service fit: coming in Milestone 3")
            return 0

        if client is None:
            from notbydefault.env import require_key
            from notbydefault.openrouter import OpenRouterClient
            client = OpenRouterClient(require_key("OPENROUTER_API_KEY"))
        pricing = client.pricing()
        cfg, created = candidates_mod.load_or_create(project, decision, pricing)
        site = callsite_mod.locate(project, decision)
        cached_site = callsite_mod.load_cached(project, decision_id, site)
        vpath = verifiers_mod.verifier_path(project, decision_id)
        samples = samples_mod.load(project, decision_id)
        ctx = graph_context(g, decision)
        cpu = resolve_calls_per_user(g, decision, ctx, opts, ask)
        bank = failures_mod.inputs(project, decision_id)
        planned = assemble_inputs(project, decision_id, samples) if samples else None
        n_inputs = len(planned) if planned else len(bank) + opts.n_samples

        est = estimate_cost(pricing, cfg, site, cached_site is not None, vpath.exists(),
                            [r["user_message"] for r in planned] if planned else None, n_inputs,
                            (cached_site or site).system_prompt)
        out(f"Fit test for decision #{decision_id}: {decision['label']} -> {cfg['chosen']}")
        out(f"  call site:   {site.file}:{site.line}")
        out(f"  candidates:  {', '.join(cfg['candidates'])}"
            + (f"  (new: {candidates_mod.config_path(project)})" if created else ""))
        out(f"  setups:      each model alone, and + verifier with <= {cfg['max_retries']} retries")
        out(f"  judge:       {cfg['judge']}    verifier model: {cfg['verifier_model']}")
        out(f"  inputs:      {n_inputs} ({len(bank)} from failure bank, "
            + (f"{len(samples)} from {samples_mod.samples_path(project, decision_id)}"
               if samples else f"{opts.n_samples} synthetic to generate") + ")")
        out(f"  prices:      {pricing.source_url} (fetched {pricing.fetched_at})")
        out(f"  estimated cost: ${est:.2f}   (hard cap --max-cost ${opts.max_cost:.2f})")
        if est > opts.max_cost:
            out(f"Estimated cost exceeds --max-cost ${opts.max_cost:.2f}; not running. "
                "Raise --max-cost or use fewer samples/candidates.")
            return 2
        if not opts.yes:
            try:
                go = ask("Run it? [y/N] ").strip().lower()
            except EOFError:
                go = ""
            if go not in ("y", "yes"):
                out("Cancelled; nothing was spent.")
                return 1

        tracer = tracer or make_tracer(decision.get("session_id"))
        guard = SpendGuard(opts.max_cost)
        caller = Caller(client, guard, tracer)
        run_at = datetime.now(timezone.utc)
        with tracer.root("notbydefault.check", {
                "notbydefault.decision_id": decision_id, "notbydefault.literal": decision["label"],
                "notbydefault.chosen": cfg["chosen"], "notbydefault.candidates": cfg["candidates"],
                "notbydefault.max_cost": opts.max_cost, "notbydefault.estimated_cost": est}):
            try:
                site = cached_site or callsite_mod.extract(
                    caller.analyst(cfg["judge"], 1500), project, decision, site)
                verifiers = verifiers_mod.load(vpath) if vpath.exists() else verifiers_mod.generate(
                    caller.analyst(cfg["judge"], 1500), project, decision, site, ctx, cfg["judge"])
                if samples is None:
                    samples = samples_mod.generate(
                        caller.analyst(cfg["judge"], max(2000, opts.n_samples * 250)), project,
                        decision_id, site, n=opts.n_samples, generator=cfg["judge"])
            except BudgetExceeded as e:
                out(f"Stopped before testing: {e}")
                return 3
            inputs = assemble_inputs(project, decision_id, samples)
            rows, failures = run_setups(caller, site, inputs, cfg["candidates"], verifiers,
                                        cfg["verifier_model"], cfg["temperature"],
                                        cfg["max_tokens"], cfg["max_retries"], opts.workers)
            judgements = judge(caller, cfg["judge"], site, inputs, rows, opts.seed, opts.workers)
            stats = summarize(rows, cfg["candidates"], SETUPS, cfg["max_retries"], opts.seed)
            project_monthly(stats, ctx["users"], cpu, ctx["budget"])
            stopped = guard.stopped
            verdict = decide(stats, cfg["chosen"])
            if stopped:
                verdict = Verdict("insufficient", verdict.reference, verdict.reference,
                                  "Not enough evidence: the run stopped at --max-cost "
                                  f"${opts.max_cost:.2f} before every call finished.",
                                  {})
            tracer.annotate({"notbydefault.verdict": verdict.kind,
                             "notbydefault.recommended": stats[verdict.recommended].name,
                             "notbydefault.spent_usd": round(guard.spent, 6),
                             **{f"notbydefault.pass_rate.{s.name}": round(s.pass_rate, 4)
                                for s in stats.values() if s.calls > s.errors},
                             **{f"notbydefault.mean_score.{s.name}": round(s.mean, 4)
                                for s in stats.values() if s.scored}})

        added = failures_mod.append(project, decision_id, failures)
        stamp = run_at.strftime("%Y%m%dT%H%M%SZ")
        results_path = labels_mod.results_dir(project) / f"{decision_id}-{stamp}.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        used = sorted(set(cfg["candidates"]) | {cfg["judge"], cfg["verifier_model"]})
        prices = {m: {"usd_per_mtok_in": pricing.price(m).prompt * 1e6,
                      "usd_per_mtok_out": pricing.price(m).completion * 1e6} for m in used}
        results_path.write_text(json.dumps({
            "decision_id": decision_id, "literal": decision["label"], "run_at": run_at.isoformat(),
            "config": cfg, "callsite": site.to_dict(),
            "verifiers": {"path": str(vpath), "sha": verifiers.sha, "checks": verifiers.names(),
                          "rubric": verifiers.rubric.criterion if verifiers.rubric else None},
            "pricing": {"source": pricing.source_url, "fetched_at": pricing.fetched_at,
                        "prices": prices},
            "assumptions": {"users": ctx["users"], "calls_per_user": cpu,
                            "budget": ctx["budget"]},
            "samples": inputs, "rows": [r.to_dict() for r in rows], "judge": judgements,
            "stats": {s.name: asdict(s) for s in stats.values()}, "verdict": asdict(verdict),
            "spent": guard.spent, "stopped": stopped,
        }, indent=2, default=str), encoding="utf-8")

        rctx = ReportContext(
            decision_id=decision_id, literal=decision["label"], chosen=cfg["chosen"],
            task=site.task, n_samples=len(inputs), sample_counts=_counts(inputs),
            judge=cfg["judge"], verifier_model=cfg["verifier_model"],
            verifier_checks=verifiers.names(), verifier_path=_rel(vpath, project),
            max_retries=cfg["max_retries"], users=ctx["users"], calls_per_user=cpu,
            budget=ctx["budget"], spent=guard.spent, pricing_source=pricing.source_url,
            pricing_fetched_at=pricing.fetched_at, results_path=_rel(results_path, project),
            failures_added=added,
            failures_path=_rel(failures_mod.bank_path(project, decision_id), project),
            trust=labels_mod.agreement(project, decision_id), stopped=stopped)

        if not stopped:
            record = write_decision_record(project, stats, verdict, rctx, run_at.date().isoformat())
            rctx.notes.append(f"Decision record: {_rel(record, project)}")
            record_review(g, decision, verdict, stats, rctx, results_path, record, pricing,
                          prices, vpath)
        out("")
        out(render(stats, verdict, rctx))
    tracer.flush()
    return 0


def _rel(path: Path, project: Path) -> str:
    try:
        return Path(path).resolve().relative_to(project).as_posix()
    except ValueError:
        return str(path)


def record_review(g: Graph, decision: dict, verdict: Verdict, stats: dict, rctx: ReportContext,
                  results_path: Path, record: Path, pricing, prices: dict, vpath: Path) -> None:
    """Evidence + alternatives into the graph; decision status -> reviewed."""
    project_root = results_path.parents[2]
    did = decision["id"]
    evidence = [
        ("Fit test results " + rctx.pricing_fetched_at[:10] + f" ({rctx.n_samples} inputs)",
         {"kind": "fit_test", "path": _rel(results_path, project_root), "spent": rctx.spent,
          "inputs": rctx.sample_counts, "judge": rctx.judge}),
        (f"OpenRouter pricing fetched {pricing.fetched_at}",
         {"kind": "pricing", "url": pricing.source_url, "fetched_at": pricing.fetched_at,
          "prices_usd_per_mtok": prices}),
        (f"Verifiers for decision #{did}",
         {"kind": "verifiers", "path": _rel(vpath, project_root), "checks": rctx.verifier_checks,
          "trust": rctx.trust}),
    ]
    for label, data in evidence:
        existing = g.find_node("evidence", label) if data["kind"] == "verifiers" else None
        if existing:
            g.update_data(existing["id"], **data)
            eid = existing["id"]
        else:
            eid = g.add_node("evidence", label, data)
        g.add_edge(did, eid, "supported_by")
    for name, reason in verdict.rejected.items():
        node = g.find_node("alternative", name)
        nid = node["id"] if node else g.add_node("alternative", name, {})
        g.update_data(nid, reason=reason, rejected_by_check=rctx.results_path)
        g.add_edge(did, nid, "rejects")
    rec = stats[verdict.recommended]
    g.update_data(did, status="reviewed", verdict=verdict.kind, verdict_summary=verdict.summary,
                  recommended=rec.name, recommended_setup=rec.setup,
                  reviewed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  results=rctx.results_path, decision_record=_rel(record, project_root),
                  openrouter_id=stats[verdict.reference].model)
