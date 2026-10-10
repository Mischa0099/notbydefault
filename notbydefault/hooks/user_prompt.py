"""UserPromptSubmit hook: Pivot Check.

Claude Code runs this with the hook JSON on stdin (fields include "prompt" and "session_id").
Contract (Claude Code hooks docs):
- exit 0 with no output -> the prompt proceeds unchanged
- {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": "..."}}
  -> the text is added to Claude's context alongside the prompt (not shown to the user)
This hook never blocks a prompt and never exits non-zero: on any error it logs and adds nothing.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notbydefault.hooks.pre_tool import _log, _project_dir, _spawn_telemetry  # noqa: E402


def _context(text: str) -> dict:
    return {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                   "additionalContext": text}}


def handle(event: dict, project_dir: Path, client=None, tracer=None) -> tuple[dict | None, dict]:
    """Returns (hook output or None, info for logging/telemetry)."""
    from notbydefault.graph import Graph, db_path
    from notbydefault.pivot import (confirm, cost_check, detect_choice, impact, pending,
                                    prefilter, price_facts, record, render_context, resolve,
                                    snapshot, stack_terms)

    prompt = event.get("prompt") or ""
    session_id = event.get("session_id")
    info: dict = {"stage": "no_graph"}
    if not db_path(project_dir).exists():
        return None, info          # no Decision Graph here: nothing to check against

    with Graph.for_project(project_dir) as g:
        p = pending(g, session_id)
        choice = detect_choice(prompt) if p else None
        if p and choice:
            stale = resolve(g, p, choice, prompt)
            if tracer is not None:
                tracer.event("pivot_resolved", "planning",
                             f"user chose '{choice}' for pivot #{p['id']} ({p['data'].get('change')})"
                             + (f"; decisions {stale} marked stale" if stale else ""),
                             pivot_id=p["id"], choice=choice)
            info.update(stage="choice", pivot_id=p["id"], choice=choice, stale=stale)
            text = (f"NotByDefault recorded the user's answer to pivot #{p['id']}: {choice}."
                    + (f" Decisions now marked stale: {', '.join(f'#{d}' for d in stale)}; run "
                       f"`python -m notbydefault check <id>` on them before relying on them."
                       if stale else ""))
            return _context(text), info
        labels = [d["label"] for d in g.nodes("decision")]

    started = time.perf_counter()
    phrases = prefilter(prompt, stack_terms(labels))
    info.update(stage="prefilter", phrases=phrases,
                prefilter_ms=round((time.perf_counter() - started) * 1000, 3))
    if not phrases:
        return None, info

    from notbydefault.env import load_env
    from notbydefault.fit.evaluate import Caller, SpendGuard
    from notbydefault.fit.providers import ProviderSet, load_catalog, role_model
    from notbydefault.fit.tracing import Tracer

    tracer = tracer or Tracer()
    tracer.record_calls = True
    tracer.event("prefilter_match", "context",
                 f"change-of-direction phrases {phrases} in a prompt about the stack; "
                 "asking a fast model to confirm", phrases=phrases)
    if client is None:
        load_env()
        catalog = load_catalog(project_dir)
        client = ProviderSet(catalog, cache_dir=project_dir / ".notbydefault" / "cache")
    model = role_model(client.catalog, "pivot")
    with Graph.for_project(project_dir) as g:
        view = snapshot(g, project_dir)
    with tracer.step("price_lookup", "tool_use",
                     reason="list prices for any model the prompt names, for the budget check"):
        prices = price_facts(prompt, client, project_dir)
        tracer.event("prices_found" if prices else "no_prices", "context",
                     ", ".join(f"{p['model']}: ${p['in'] * 1e6:.2f}/M in, ${p['out'] * 1e6:.2f}/M out "
                               f"({p['source']})" for p in prices) or "no model named or no price found")
    price_text = "; ".join(f"{p['model']}: ${p['in'] * 1e6:.2f} per 1M input tokens, "
                           f"${p['out'] * 1e6:.2f} per 1M output tokens" for p in prices) or "none"
    with tracer.step("pivot_confirm", "verification",
                     reason="confirm it is a real change of direction, not an ordinary request"):
        result = confirm(Caller(client, SpendGuard(0.05), tracer), model, prompt, view, price_text)
    info.update(stage="confirmed" if result["is_pivot"] else "not_pivot", model=model,
                llm_latency_s=round(result["latency_s"], 3), change=result["change"])
    if not result["is_pivot"]:
        tracer.event("not_a_pivot", "verification",
                     "the model judged it an ordinary request; adding nothing", model=model)
        return None, info

    with Graph.for_project(project_dir) as g:
        with tracer.step("pivot_impact", "context",
                         reason="collect conflicting constraints, invalidated decisions, evidence "
                                "and affected code from the Decision Graph"):
            cost = cost_check(g, project_dir, prices, result["decision_ids"])
            if cost:
                tracer.event("cost_check", "verification",
                             f"{cost['model']} about ${cost['monthly']:,.0f}/month vs budget "
                             f"{cost['budget']}", over_budget=cost["over_budget"])
            imp = impact(g, result, project_dir, cost)
        with tracer.step("pivot_record", "state", reason="record the pivot as proposed"):
            pid = record(g, prompt, imp, session_id, result)
    tracer.event("plan_change_proposed", "planning",
                 f"user proposes: {imp['change']}; conflicts with "
                 f"{[c['id'] for c in imp['conflicts']]}, would invalidate "
                 f"{[d['id'] for d in imp['decisions']]}; asking the user to choose",
                 pivot_id=pid)
    info.update(pivot_id=pid, conflicts=[c["id"] for c in imp["conflicts"]],
                decisions=[d["id"] for d in imp["decisions"]])
    return _context(render_context(imp, pid)), info


def main(argv: list[str]) -> int:
    start = time.perf_counter()
    project_dir = _project_dir(argv, {}) if "--project" in argv else None
    try:
        event = json.loads(sys.stdin.buffer.read().decode("utf-8", errors="replace") or "{}")
        project_dir = _project_dir(argv, event)
        from notbydefault.fit.tracing import Tracer
        tracer = Tracer()
        out, info = handle(event, project_dir, tracer=tracer)
        if out:
            sys.stdout.write(json.dumps(out))
            sys.stdout.flush()
        ms = (time.perf_counter() - start) * 1000
        if info.get("stage") not in ("no_graph", "prefilter") or info.get("phrases"):
            _log(project_dir, f"pivot-check {info.get('stage')} {ms:.0f}ms "
                              + json.dumps({k: v for k, v in info.items() if k != 'stage'},
                                           default=str))
            _send_trace(event, info, tracer, project_dir)
    except Exception:
        import traceback
        _log(project_dir, "pivot-check error (prompt passed through):\n" + traceback.format_exc())
    return 0


def _send_trace(event: dict, info: dict, tracer, project_dir: Path) -> None:
    import os
    if os.environ.get("NOTBYDEFAULT_TELEMETRY") == "0":
        return
    try:
        _spawn_telemetry({"kind": "trace", "name": "notbydefault.pivot_check",
                          "session_id": event.get("session_id"), "info": info,
                          "journal": tracer.journal}, project_dir)
    except Exception as e:
        _log(project_dir, f"telemetry spawn failed: {e!r}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
