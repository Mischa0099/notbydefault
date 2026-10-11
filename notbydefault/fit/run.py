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
from notbydefault.fit.stats import (Verdict, decide, decide_with_references, judge_agreement,
                                    parse_number, project_monthly, summarize)
from notbydefault.fit.tracing import Tracer, make_tracer
from notbydefault.graph import Graph, find_affected_code
from notbydefault.llm import LLMError, ModelUnavailable

DEFAULT_CALLS_PER_USER = 30
EXPECTED_OUT_TOKENS = 350     # cost estimate only; the spend guard uses worst cases
EXPECTED_EXTRA_ATTEMPTS = 0.5  # cost estimate only: average retries per verified call
SETUPS = (ALONE, VERIFIED)
RECOVERY_EVENTS = {"rate_limit_backoff", "retry_failed_call", "retry_with_feedback", "call_failed",
                   "judge_failed", "swap_candidate", "drop_candidate", "daily_limit_stop",
                   "budget_stop", "expand_failed", "setup_failed"}
PROBE = [{"role": "user", "content": "Reply with the single word: ok"}]


@dataclass
class CheckOptions:
    max_cost: float = 1.0
    yes: bool = False
    calls_per_user: float | None = None
    n_samples: int = 8
    workers: int = 6
    seed: int = 0
    expand_rounds: int = 1      # if the verdict is too close to call, add inputs this many times
    expand_by: int = 4


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
    by_message = {s["user_message"]: s for s in samples}
    referenced = bool(samples) and all(s.get("reference") is not None for s in samples)
    rows = [dict(by_message[t], from_failure_bank=True) if t in by_message
            else {"user_message": t, "source": "failure-bank"}
            for t in failures_mod.inputs(project, decision_id)
            if t in by_message or not referenced]   # unscorable banked inputs are skipped
    seen = {r["user_message"] for r in rows}
    return rows + [s for s in samples if s["user_message"] not in seen]


def _counts(inputs: list[dict]) -> dict[str, int]:
    """Inputs per source; banked failures re-run first are counted again under their own key."""
    out: dict[str, int] = {}
    for r in inputs:
        out[r.get("source", "provided")] = out.get(r.get("source", "provided"), 0) + 1
    banked = sum(1 for r in inputs if r.get("from_failure_bank"))
    if banked:
        out["re-run first from the failure bank"] = banked
    return out


def check_model_decision(project: Path, decision_id: int, opts: CheckOptions, client=None,
                         tracer: Tracer | None = None, ask=input, out=print) -> int:
    project = Path(project).resolve()
    with Graph.for_project(project) as g:
        decision = g.get_node(decision_id)
        if decision is None or decision["type"] != "decision":
            out(f"No decision #{decision_id} in {project}")
            return 1
        if decision["data"].get("status") == "stale":
            out(f"Decision #{decision_id} is STALE: {decision['data'].get('stale_reason')}")
        if decision["data"].get("kind") != "model":
            out(f"Decision #{decision_id}: {decision['label']} "
                f"[{decision['data'].get('kind')}/{decision['data'].get('category')}]")
            print_briefing(project, g, decision_id, out)
            out("service fit: not implemented yet (only model decisions can be re-tested)")
            return 0

        if client is None:
            from notbydefault.env import load_env
            from notbydefault.fit.providers import ProviderSet, load_catalog
            load_env()
            client = ProviderSet(load_catalog(project), cache_dir=cache_dir(project))
        pricing = client.pricing()
        reqs = requirement_constraints(g)
        allowed, why_not = compliance(reqs, getattr(client, "catalog", {}))
        stale_reason = (decision["data"].get("stale_reason")
                        if decision["data"].get("status") == "stale" else None)
        cfg, created = candidates_mod.load_or_create(project, decision, pricing, allowed=allowed,
                                                     refresh=stale_reason)
        excluded = {}
        if allowed:
            excluded = {(m, setup): why_not(m) for m in cfg["candidates"] if not allowed(m)
                        for setup in SETUPS}
            if all(not allowed(m) for m in cfg["candidates"]):
                return blocked_by_requirements(g, project, decision, cfg, reqs, client, why_not,
                                               tracer, out)
        from notbydefault.affected import refresh as refresh_affected
        affected = refresh_affected(project, [decision_id], graph=g)[decision_id]
        decision = g.get_node(decision_id)
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
        out(f"  affects:     " + (", ".join(_describe_entries(affected.entries)) or "nothing found")
            + f"  ({affected.source})")
        out(f"  candidates:  {', '.join(cfg['candidates'])}"
            + (f"  (new: {candidates_mod.config_path(project)})" if created else ""))
        out(f"  setups:      each model alone, and + verifier with <= {cfg['max_retries']} retries")
        out(f"  judges:      {', '.join(cfg['judges'])}    verifier model: {cfg['verifier_model']}")
        out(f"  inputs:      {n_inputs} ({len(bank)} from failure bank, "
            + (f"{len(samples)} from {samples_mod.samples_path(project, decision_id)}"
               if samples else f"{opts.n_samples} synthetic to generate") + ")")
        out(f"  prices:      {pricing.source_url}")
        free = _free_tier(client, cfg)
        out(f"  requests:    ~{estimate_requests(cfg, n_inputs, cached_site is not None, vpath.exists(), samples is not None)}"
            " (cached responses are reused, never re-requested)")
        out(f"  estimated cost: ${est:.2f}" + (" at list price (run on free tier)" if free else "")
            + f"   (hard cap --max-cost ${opts.max_cost:.2f})")
        print_briefing(project, g, decision_id, out)
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
        if hasattr(client, "on_event"):
            client.on_event = tracer.event      # 429 backoffs, retried calls, throttling
        run_at = datetime.now(timezone.utc)
        state: dict = {"rows": [], "early": False}
        with tracer.root("notbydefault.check", {
                "notbydefault.decision_id": decision_id, "notbydefault.literal": decision["label"],
                "notbydefault.chosen": cfg["chosen"], "notbydefault.candidates": cfg["candidates"],
                "notbydefault.max_cost": opts.max_cost, "notbydefault.estimated_cost": est},
                finish=lambda: check_summary(tracer, guard, state)):
            with tracer.step("plan", "planning",
                             reason="decide what to test before spending any request",
                             attrs={"plan": plan_text(decision, cfg, site, n_inputs, est, opts),
                                    "candidates": cfg["candidates"], "judge": cfg["judge"],
                                    "verifier_model": cfg["verifier_model"],
                                    "inputs": n_inputs, "estimated_cost_usd": round(est, 6)}):
                tracer.event("affected_code" if affected.source == "entire-graph"
                             else "affected_code_fallback",
                             "context" if affected.source == "entire-graph" else "recovery",
                             f"{len(affected.entries)} code locations in {len(affected.files)} files "
                             f"use {decision['label']} ({affected.source})"
                             + (f": {affected.note}" if affected.note else ""),
                             files=affected.files)
                for model, why in cfg["why"].items():
                    tracer.event("judge_selected" if model in cfg["judges"] else "candidate_selected",
                                 "planning", why, model=model)
                tracer.event("verifier_model_selected", "planning",
                             "cheapest candidate: verified setups pay for every rubric check",
                             model=cfg["verifier_model"])
            try:
                with tracer.step("preflight", "tool_use",
                                 reason="probe each model once (cached) so unavailable models are "
                                        "swapped before the run, not halfway through"):
                    cfg = preflight(caller, cfg, pricing, project, decision, tracer, allowed)
                with tracer.step("context", "context",
                                 reason="understand the call site and gather test inputs"):
                    site = cached_site or callsite_mod.extract(
                        caller.analyst(cfg["judge"]), project, decision, site)
                    if samples is None:
                        tracer.event("generate_samples", "context",
                                     f"no samples file for decision #{decision_id} yet; generating "
                                     f"{opts.n_samples} synthetic inputs and saving them so runs "
                                     "are reproducible", added=opts.n_samples)
                        samples = samples_mod.generate(
                            caller.analyst(cfg["judge"], max(8192, opts.n_samples * 400)), project,
                            decision_id, site, n=opts.n_samples, generator=cfg["judge"])
                    else:
                        tracer.event("load_samples", "context",
                                     f"reusing {len(samples)} saved inputs so results stay "
                                     "comparable across runs", count=len(samples))
                    if bank:
                        tracer.event("expand_sample", "context",
                                     f"{len(bank)} inputs failed a verifier in earlier runs; adding "
                                     "them first so known failures are always re-tested",
                                     added=len(bank))
                with tracer.step("verifiers", "verification",
                                 reason="decision-specific checks every output must pass"):
                    if vpath.exists():
                        verifiers = verifiers_mod.load(vpath)
                        tracer.event("load_verifiers", "verification",
                                     f"reusing reviewed verifiers in {_rel(vpath, project)}")
                    else:
                        verifiers = verifiers_mod.generate(
                            caller.analyst(cfg["judge"]), project, decision, site, ctx, cfg["judge"])
            except (BudgetExceeded, LLMError, ValueError, LookupError) as e:
                state["early"] = True
                if guard.halt_reason:
                    out(daily_stop_message(guard.halt_reason, client, project, decision_id))
                    out(f"Journal: {save_journal(project, decision_id, tracer, run_at, 'daily_limit')}")
                    tracer.flush()
                    return 4
                tracer.event("setup_failed", "recovery", f"{e}; nothing to test, stopping")
                out(f"Stopped before testing (spent ${guard.spent:.4f}): {e}")
                out(f"Journal: {save_journal(project, decision_id, tracer, run_at, 'setup_failed')}")
                tracer.flush()
                return 3

            all_failures: list[dict] = []
            for rnd in range(opts.expand_rounds + 1):
                inputs = assemble_inputs(project, decision_id, samples)
                with tracer.step("execute", "execution",
                                 reason=f"run {len(inputs)} inputs on {len(cfg['candidates'])} "
                                        "models, alone and with the verifier "
                                        f"(<= {cfg['max_retries']} retries)",
                                 attrs={"round": rnd, "inputs": len(inputs)}):
                    rows, failures = run_setups(caller, site, inputs, cfg["candidates"], verifiers,
                                                cfg["verifier_model"], cfg["temperature"],
                                                cfg["max_tokens"], cfg["max_retries"], opts.workers)
                    state["rows"] = rows
                all_failures += failures
                judgements = []
                for jm in cfg["judges"]:
                    with tracer.step("judge", "verification",
                                     reason=f"blind, order-randomized scoring of every distinct output "
                                            f"by {jm}" + (" (second judge from another family, to "
                                                          "check for self-preference)"
                                                          if jm != cfg["judges"][0] else ""),
                                     attrs={"round": rnd, "judge": jm}):
                        judgements += [dict(j, judge=jm) for j in
                                       judge(caller, jm, site, inputs, rows, opts.seed, opts.workers)]
                with tracer.step("decide", "verification",
                                 reason="compare every setup with what the code does today",
                                 attrs={"round": rnd}):
                    stats = summarize(rows, cfg["candidates"], SETUPS, cfg["max_retries"], opts.seed,
                                      samples=inputs, judges=cfg["judges"])
                    project_monthly(stats, ctx["users"], cpu, ctx["budget"])
                    agreement = judge_agreement(rows, cfg["judges"])
                    if has_references(inputs):
                        verdict = decide_with_references(stats, cfg["chosen"], ctx["budget"],
                                                         excluded)
                    else:
                        verdict = decide(stats, cfg["chosen"], excluded)
                    if agreement.get("n"):
                        tracer.event("judge_agreement", "verification",
                                     f"judges agree within 1 point on {agreement['within_1']:.0%} of "
                                     f"{agreement['n']} outputs (mean gap {agreement['mean_abs_diff']:.2f})",
                                     **{k: round(v, 4) if isinstance(v, float) else v
                                        for k, v in agreement.items()})
                    tracer.event("verdict", "verification", verdict.summary, verdict=verdict.kind,
                                 round=rnd)
                if guard.stopped or verdict.kind != "insufficient" or rnd == opts.expand_rounds:
                    break
                if has_references(inputs):
                    tracer.event("no_expansion", "iteration",
                                 "verdict is close, but the cases have hand-written reference answers "
                                 "and synthetic additions would not; add reference cases instead")
                    break
                reason = (f"verdict too close to call after {len(inputs)} inputs "
                          f"({verdict.summary}); adding {opts.expand_by} new inputs to narrow the "
                          "intervals (earlier calls are cached, so only new inputs cost requests)")
                with tracer.step("expand_sample", "iteration", reason=reason,
                                 attrs={"round": rnd + 1}):
                    tracer.event("expand_sample", "iteration", reason, added=opts.expand_by,
                                 round=rnd + 1)
                    try:
                        samples = samples_mod.extend(caller.analyst(cfg["judge"]), project,
                                                     decision_id, site, samples, opts.expand_by,
                                                     reason, cfg["judge"])
                    except (BudgetExceeded, LLMError, ValueError) as e:
                        tracer.event("expand_failed", "recovery",
                                     f"could not add inputs ({e}); keeping the current verdict")
                        break
            failures = all_failures
            stopped = guard.stopped
            if guard.halt_reason:
                state["early"] = True
                failures_mod.append(project, decision_id, failures)
                out(daily_stop_message(guard.halt_reason, client, project, decision_id))
                out(f"Journal: {save_journal(project, decision_id, tracer, run_at, 'daily_limit')}")
                tracer.annotate({"notbydefault.stopped": guard.halt_reason})
                tracer.flush()
                return 4
            if stopped:
                tracer.event("budget_stop", "recovery",
                             f"--max-cost ${opts.max_cost:.2f} reached before every call finished; "
                             "reporting without a verdict")
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

            with tracer.step("record", "state",
                             reason="save results, failure bank, graph evidence and decision record"):
                added = failures_mod.append(project, decision_id, failures)
                stamp = run_at.strftime("%Y%m%dT%H%M%SZ")
                results_path = labels_mod.results_dir(project) / f"{decision_id}-{stamp}.json"
                results_path.parent.mkdir(parents=True, exist_ok=True)
                used = sorted(set(cfg["candidates"]) | {cfg["judge"], cfg["verifier_model"]})
                prices = {m: {"usd_per_mtok_in": pricing.price(m).prompt * 1e6,
                              "usd_per_mtok_out": pricing.price(m).completion * 1e6,
                              "source": pricing.price(m).source_url,
                              "fetched": pricing.price(m).fetched_at} for m in used}
                rctx = ReportContext(
                    decision_id=decision_id, literal=decision["label"], chosen=cfg["chosen"],
                    task=site.task, n_samples=len(inputs), sample_counts=_counts(inputs),
                    judge=cfg["judge"], judges=cfg["judges"], agreement=agreement,
                    verifier_model=cfg["verifier_model"], case_rows=case_rows(rows, inputs),
                    bar=verdict.bar, best_quality=verdict.best_quality,
                    caveats=caveats(inputs, cfg),
                    verifier_checks=verifiers.names(), verifier_path=_rel(vpath, project),
                    max_retries=cfg["max_retries"], users=ctx["users"], calls_per_user=cpu,
                    budget=ctx["budget"], spent=guard.spent, pricing_source=pricing.source_url,
                    pricing_fetched_at=pricing.fetched_at, results_path=_rel(results_path, project),
                    failures_added=added,
                    failures_path=_rel(failures_mod.bank_path(project, decision_id), project),
                    trust=labels_mod.agreement(project, decision_id), stopped=stopped,
                    free_tier=free, single_provider=_single_provider(cfg),
                    trace_id=tracer.trace_id)
                recoveries = {k: v for k, v in tracer.summary().items() if k in RECOVERY_EVENTS}
                rctx.notes.append("Agent journal: "
                                  + (", ".join(f"{v} {k}" for k, v in sorted(recoveries.items()))
                                     or "no recoveries needed")
                                  + f" (full journal in {rctx.results_path})")
                if not stopped:
                    from notbydefault import records as records_mod
                    record = write_decision_record(project, stats, verdict, rctx,
                                                   run_at.date().isoformat(),
                                                   records_mod.existing_path(project, decision))
                    rctx.notes.insert(0, f"Decision record: {_rel(record, project)}")
                    record_review(g, decision, verdict, stats, rctx, results_path, record, pricing,
                                  prices, vpath)
                    records_mod.write_records(project, g, [decision_id],
                                              f"fit check verdict: {verdict.kind} ({verdict.summary})",
                                              tracer, out=rctx.notes.append)
                results_path.write_text(json.dumps({
                    "decision_id": decision_id, "literal": decision["label"],
                    "run_at": run_at.isoformat(), "config": cfg, "callsite": site.to_dict(),
                    "verifiers": {"path": str(vpath), "sha": verifiers.sha,
                                  "checks": verifiers.names(),
                                  "rubric": verifiers.rubric.criterion if verifiers.rubric else None},
                    "pricing": {"source": pricing.source_url, "fetched_at": pricing.fetched_at,
                                "prices": prices, "free_tier": free},
                    "assumptions": {"users": ctx["users"], "calls_per_user": cpu,
                                    "budget": ctx["budget"]},
                    "samples": inputs, "rows": [r.to_dict() for r in rows], "judge": judgements,
                    "stats": {s.name: asdict(s) for s in stats.values()},
                    "judge_agreement": agreement,
                    "verdict": asdict(verdict), "spent": guard.spent, "stopped": stopped,
                    "requests": {"made": getattr(client, "requests_made", None),
                                 "cache_hits": getattr(client, "cache_hits", None)},
                    "trace_id": tracer.trace_id,
                    "journal": getattr(tracer, "journal", []),
                }, indent=2, default=str), encoding="utf-8")
        out("")
        out(render(stats, verdict, rctx))
    tracer.flush()
    return 0


def print_briefing(project: Path, g: Graph, decision_id: int, out) -> None:
    """The hook's briefing without its 4-line limit, after a live refresh of the Entire cache."""
    from notbydefault.briefing import build, refresh_cache
    cache = refresh_cache(project, [decision_id])
    lines = build(project, g, g.get_node(decision_id), limit=None)
    out("  briefing:    " + ("" if lines else "nothing related recorded yet"))
    for line in lines:
        out(f"    - {line}")
    for key, why in (cache.get("errors") or {}).items():
        out(f"    ({key}: {why})")


def _describe_entries(entries: list[dict], limit: int = 6) -> list[str]:
    from notbydefault.affected import describe
    out = [describe(e) for e in entries[:limit]]
    return out + ([f"+{len(entries) - limit} more"] if len(entries) > limit else [])


def plan_text(decision: dict, cfg: dict, site, n_inputs: int, est: float,
              opts: CheckOptions) -> str:
    return (f"Test decision #{decision['id']} ({decision['label']} -> {cfg['chosen']}) at "
            f"{site.file}:{site.line}. Candidates: {', '.join(cfg['candidates'])}; each alone and "
            f"with the verifier (<= {cfg['max_retries']} retries). Inputs: {n_inputs} "
            f"(failure bank first). Judge {cfg['judge']} scores blind; verifier model "
            f"{cfg['verifier_model']}. If too close to call, add {opts.expand_by} inputs up to "
            f"{opts.expand_rounds}x. Estimated ${est:.4f} at list price, cap ${opts.max_cost:.2f}.")


def preflight(caller, cfg: dict, pricing, project: Path, decision: dict, tracer,
              allowed=None) -> dict:
    """Probe every model once; swap unavailable alternatives/judge for the next eligible model."""
    cfg = dict(cfg, candidates=list(cfg["candidates"]), why=dict(cfg["why"]),
               judges=list(cfg["judges"]))
    queue = [(m, "candidate") for m in cfg["candidates"]] + [(cfg["judge"], "judge")]
    queue += [(j, "judge2") for j in cfg["judges"][1:]]
    if cfg["verifier_model"] not in cfg["candidates"]:
        queue.append((cfg["verifier_model"], "verifier"))
    failed: set[str] = set()
    while queue:
        model, role = queue.pop(0)
        try:
            caller.chat(model, PROBE, span="notbydefault.preflight", role="preflight",
                        max_tokens=256, attrs={"notbydefault.for_role": role})
            continue
        except ModelUnavailable as e:
            failed.add(model)
            reason = e.detail
        if model == cfg["chosen"]:
            raise LLMError(f"the chosen model {model} is unavailable ({reason}); it can't be "
                           "tested with this key")
        taken = failed | set(cfg["candidates"]) | set(cfg["judges"])
        try:
            if role == "candidate":
                new = (candidates_mod.pick_alternatives(cfg["chosen"], pricing, n=1,
                                                        exclude=taken, allowed=allowed) or [None])[0]
            elif role == "judge":
                new = candidates_mod.pick_judge(sorted(taken), pricing)
            elif role == "judge2":
                new = candidates_mod.pick_second_judge(sorted(taken), cfg["judge"], pricing)
            else:
                new = next((m for m in cfg["candidates"] if m not in failed), None)
        except LookupError:
            new = None
        if new is None:
            tracer.event("drop_candidate", "recovery",
                         f"{model} is unavailable ({reason}) and no free replacement is left; "
                         "continuing without it", model=model, role=role)
            if role == "candidate":
                cfg["candidates"].remove(model)
                continue
            if role == "judge2":
                cfg["judges"].remove(model)
                continue
            raise LLMError(f"no available {role} model ({model}: {reason})")
        why = (candidates_mod.why_picked(new, cfg["chosen"], pricing) if role == "candidate"
               else f"next strongest free {role}")
        tracer.event("swap_candidate", "recovery",
                     f"{model} is unavailable ({reason}); swapping in {new}: {why}",
                     old=model, new=new, role=role)
        candidates_mod.swap(project, decision, role, model, new, reason)
        if role == "candidate":
            cfg["candidates"][cfg["candidates"].index(model)] = new
        elif role == "judge":
            cfg["judge"] = new
            cfg["judges"][0] = new
        elif role == "judge2":
            cfg["judges"][cfg["judges"].index(model)] = new
        if cfg["verifier_model"] == model:
            cfg["verifier_model"] = new
        cfg["why"].pop(model, None)
        cfg["why"][new] = why
        queue.insert(0, (new, role))
    return cfg


def caveats(inputs: list[dict], cfg: dict) -> list[str]:
    """What a reader must know to weigh the evidence: revised cases, sample size, judges."""
    out = [f"Case {s.get('id') or i}: {s['revision_note']}."
           for i, s in enumerate(inputs) if s.get("revision_note")]
    if has_references(inputs):
        out.append(f"Only {len(inputs)} cases: one case is {100 / len(inputs):.0f} percentage points "
                   "of correctness, so intervals are wide and small differences are not meaningful.")
    if any(j.split(":", 1)[0] in {c.split(":", 1)[0] for c in cfg["candidates"]}
           for j in cfg.get("judges", [])):
        out.append("A judge shares a provider with a candidate; compare the two judges' columns "
                   "for self-preference (correctness against references is unaffected).")
    return out


def requirement_constraints(g: Graph) -> list[dict]:
    """Requirement nodes that carry machine-checkable constraints (added by `require`)."""
    return [n for n in g.nodes("requirement") if (n["data"].get("constraints") or {})]


def compliance(reqs: list[dict], catalog: dict):
    """(allowed(model_id) -> bool, why_not(model_id) -> str) for the requirements' constraints,
    or (None, None) when nothing constrains the choice of model."""
    from notbydefault.fit.providers import residency
    regions = {(n["id"], str(n["data"]["constraints"]["data_residency"]).upper())
               for n in reqs if n["data"]["constraints"].get("data_residency")}
    if not regions:
        return None, None

    def allowed(model_id: str) -> bool:
        return all(r in residency(catalog, model_id) for _, r in regions)

    def why_not(model_id: str) -> str:
        provider = model_id.split(":", 1)[0]
        cfg = (catalog.get("providers") or {}).get(provider, {})
        have = ", ".join(residency(catalog, model_id)) or "no region guaranteed"
        need = ", ".join(f"{r} (requirement #{rid})" for rid, r in sorted(regions))
        return (f"{provider} documents data residency: {have} "
                f"(source: {cfg.get('data_residency_source', 'not recorded')}); needs {need}")
    return allowed, why_not


def blocked_by_requirements(g: Graph, project: Path, decision: dict, cfg: dict, reqs: list[dict],
                            client, why_not, tracer, out) -> int:
    """No candidate complies: explain with the documented facts, record it, make no requests."""
    catalog = getattr(client, "catalog", {})
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    providers = sorted({m.split(":", 1)[0] for m in (getattr(client, "pricing")().prices)
                        if m.split(":", 1)[0] in (catalog.get("providers") or {})})
    facts = {p: why_not(f"{p}:x") for p in providers}
    tracer = tracer or make_tracer(decision.get("session_id"))
    with tracer.root("notbydefault.check", {"notbydefault.decision_id": decision["id"]},
                     finish=lambda: {"nbd.swap_count": 0, "nbd.retry_rescued": 0,
                                     "nbd.stopped_early": True, "nbd.cost_usd": 0.0}):
        with tracer.step("requirements", "planning",
                         reason="the new requirements rule out every model in the catalog"):
            tracer.event("blocked_by_requirement", "planning",
                         "no compliant candidate; not spending any requests",
                         **{f"provider.{p}": f for p, f in facts.items()})
        with tracer.step("record", "state", reason="record why the decision can't be re-tested"):
            eid = g.add_node("evidence", f"Requirement check {now[:10]}: no compliant model", {
                "kind": "requirement_check", "requirements": [n["id"] for n in reqs],
                "providers": facts, "checked_at": now})
            g.add_edge(decision["id"], eid, "supported_by")
            g.update_data(decision["id"], blocked_by=[n["id"] for n in reqs], last_check=now)
            from notbydefault import records as records_mod
            records_mod.write_records(project, g, [decision["id"]],
                                      "fit check blocked: no model in the catalog meets the "
                                      "requirements", tracer)
    out(f"Decision #{decision['id']} ({decision['label']} -> {cfg['chosen']}) can't be re-tested: "
        "no model in the catalog meets the project's requirements.")
    for n in reqs:
        out(f"  requirement #{n['id']}: {n['label']}  checks: {json.dumps(n['data']['constraints'])}")
    out("Providers in the catalog:")
    for p, f in facts.items():
        out(f"  - {f}")
    out(f"The current choice {cfg['chosen']} breaks the requirement, so the decision stays stale.")
    out("To fix it, add a provider that contractually keeps data in the required region to "
        ".notbydefault/providers.yaml (data_residency: [...], with its source), then run "
        f"`python -m notbydefault check {decision['id']}` again. No requests were made.")
    tracer.flush()
    return 0


def has_references(inputs: list[dict]) -> bool:
    return bool(inputs) and all(r.get("reference") is not None for r in inputs)


def case_rows(rows, inputs: list[dict]) -> list[dict]:
    """Per case x setup outcome, for the report's case table."""
    out = []
    for r in rows:
        s = inputs[r.sample]
        out.append({"case": s.get("id") or f"input {r.sample}", "hard": bool(s.get("hard")),
                    "has_reference": s.get("reference") is not None,
                    "model": r.model, "setup": r.setup, "correct": r.correct,
                    "passed": r.passed, "attempts": r.attempts, "error": r.error,
                    "problems": r.problems, "failed_checks": r.failed})
    return out


def _single_provider(cfg: dict) -> bool:
    used = list(cfg["candidates"]) + [cfg["judge"], cfg["verifier_model"]]
    return len({m.split(":", 1)[0] for m in used}) == 1


def check_summary(tracer, guard, state: dict) -> dict:
    """Root-span attributes for neatlogs Conditions: candidate swaps, retries that rescued an
    output the model alone got wrong, whether the run stopped early, and what it spent."""
    from notbydefault.fit.evaluate import ALONE, VERIFIED
    rows = {(r.sample, r.model, r.setup): r for r in state.get("rows") or []}
    rescued = sum(1 for (sample, model, setup), r in rows.items()
                  if setup == VERIFIED and r.passed and r.attempts > 1
                  and not getattr(rows.get((sample, model, ALONE)), "passed", True))
    return {"nbd.swap_count": int(tracer.summary().get("swap_candidate", 0)),
            "nbd.retry_rescued": rescued,
            "nbd.stopped_early": bool(state.get("early") or guard.stopped or guard.halt_reason),
            "nbd.cost_usd": round(float(guard.spent), 6)}


def save_journal(project: Path, decision_id: int, tracer, run_at: datetime, why: str) -> str:
    """Keep the agent journal of a run that stopped early (finished runs keep it in results)."""
    path = Path(project) / ".notbydefault" / "journal" / \
        f"{decision_id}-{run_at.strftime('%Y%m%dT%H%M%SZ')}-{why}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(getattr(tracer, "journal", []), indent=2, default=str),
                    encoding="utf-8")
    return _rel(path, Path(project).resolve())


def cache_dir(project: Path) -> Path:
    return Path(project) / ".notbydefault" / "cache"


def _free_tier(client, cfg: dict) -> bool:
    free = getattr(client, "free_tier", None)
    used = list(cfg["candidates"]) + [cfg["judge"], cfg["verifier_model"]]
    return bool(free and free(used))


def estimate_requests(cfg: dict, n_inputs: int, have_site: bool, have_verifiers: bool,
                      have_samples: bool) -> int:
    probes = len(set(cfg["candidates"]) | set(cfg["judges"]) | {cfg["verifier_model"]})  # cached later
    prep = (not have_site) + (not have_verifiers) + (not have_samples) + probes
    attempts = 1 + EXPECTED_EXTRA_ATTEMPTS
    per_input = len(cfg["candidates"]) * attempts * 2 + len(cfg["judges"])   # call + check, judges
    return int(prep + n_inputs * per_input)


def daily_stop_message(reason: str, client, project: Path, decision_id: int) -> str:
    cache = getattr(client, "cache", None)
    saved = (f"{cache.count()} responses are cached in {_rel(cache.folder, project)}" if cache
             else "responses so far are cached")
    return (f"Stopped: {reason}.\n"
            f"Progress is saved: {saved}, plus samples, verifiers and the failure bank. "
            f"Run `check {decision_id}` again after the limit resets to resume; cached calls "
            "are reused, so no request is spent twice.")


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
        (f"Provider list prices fetched {pricing.fetched_at}",
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
    best = stats[verdict.best_quality] if verdict.best_quality else None
    g.update_data(did, status="reviewed", verdict=verdict.kind, verdict_summary=verdict.summary,
                  recommended=rec.name, recommended_setup=rec.setup,
                  best_quality_within_budget=best.name if best else None,
                  reviewed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  results=rctx.results_path, decision_record=_rel(record, project_root),
                  openrouter_id=stats[verdict.reference].model)
    if decision["data"].get("status") == "stale":   # re-tested: no longer stale
        g.update_data(did, stale_resolved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      previous_stale_reason=decision["data"].get("stale_reason"), stale_reason=None)
