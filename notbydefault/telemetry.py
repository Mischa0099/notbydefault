"""Send one neatlogs span for a PreToolUse hook call, then flush.

Run detached by hooks/pre_tool.py:  python -m notbydefault.telemetry '<json payload>'
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from notbydefault.env import load_env


def _log(payload: dict, message: str) -> None:
    project = payload.get("project_dir")
    if not project:
        return
    try:
        with open(Path(project, ".notbydefault", "hook.log"), "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} telemetry: {message}\n")
    except OSError:
        pass


def send(payload: dict) -> bool:
    load_env()
    if not os.environ.get("NEATLOGS_API_KEY"):
        _log(payload, "NEATLOGS_API_KEY not set; skipped")
        return False

    from notbydefault.fit.tracing import EXPORT_TIMEOUT_S
    for var in ("OTEL_EXPORTER_OTLP_TRACES_TIMEOUT", "OTEL_EXPORTER_OTLP_LOGS_TIMEOUT"):
        os.environ.setdefault(var, str(EXPORT_TIMEOUT_S))
    import neatlogs

    neatlogs.init(workflow_name="notbydefault", tags=["pre_tool_hook"],
                  register_shutdown_handlers=False)
    decisions = payload.get("decisions") or []

    def joined(key):
        return ", ".join(str(d.get(key)) for d in decisions)

    @neatlogs.span(kind="TOOL", name="notbydefault.pre_tool",
                   session_id=payload.get("session_id") or None)
    def pre_tool_hook(kind: str, chosen: str, decision_id: str, tool_name: str,
                      decisions: list) -> dict:
        return {"permission_decision": "ask", "decision_count": len(decisions)}

    pre_tool_hook(kind=joined("kind"), chosen=joined("chosen"),
                  decision_id=joined("decision_id"), tool_name=payload.get("tool_name") or "",
                  decisions=decisions)
    from notbydefault.fit.tracing import FLUSH_DEADLINE_S, _run_with_deadline
    done, _ = _run_with_deadline(lambda: neatlogs.flush(timeout_millis=15000), FLUSH_DEADLINE_S)
    if not done:
        _log(payload, f"neatlogs did not finish exporting within {FLUSH_DEADLINE_S}s; gave up")
        return False
    failures = neatlogs.get_delivery_diagnostics().get("span_export_failures", 0)
    _run_with_deadline(lambda: neatlogs.shutdown(timeout_millis=5000), 6)
    _log(payload, f"sent span for decision(s) {joined('decision_id')}"
                  + (f" ({failures} export failure(s))" if failures else ""))
    return not failures


def replay(payload: dict) -> bool:
    """Re-emit a hook's journal (steps, calls, events) as one neatlogs trace.

    Hooks must stay fast, so they journal what they did and this detached process exports it:
    every step and call becomes a span tagged with its focus_area, and every decision or
    recovery becomes an event with its reason.
    """
    import contextlib

    load_env()
    if not os.environ.get("NEATLOGS_API_KEY"):
        _log(payload, "NEATLOGS_API_KEY not set; skipped")
        return False
    from notbydefault.fit.tracing import NeatlogsTracer

    project = payload.get("project_dir")
    tracer = NeatlogsTracer(payload.get("session_id"),
                            spill_dir=Path(project, ".notbydefault", "telemetry") if project else None)
    with tracer.root(payload.get("name", "notbydefault.hook"),
                     {"notbydefault.info": payload.get("info") or {},
                      **{k: v for k, v in (payload.get("nbd") or {}).items() if v is not None}}):
        with contextlib.ExitStack() as stack:
            for e in payload.get("journal") or []:
                extras = {k: v for k, v in e.items()
                          if k not in ("t", "kind", "name", "focus_area", "reason", "attrs",
                                       "span_kind")}
                if e["kind"] == "step":
                    stack.close()
                    stack.enter_context(tracer.step(e["name"], e["focus_area"], e.get("reason", ""),
                                                    extras))
                elif e["kind"] == "call":
                    tracer.llm(e["name"], lambda: None, before=e.get("attrs") or {},
                               after=lambda _r: {}, kind=e.get("span_kind", "TOOL"),
                               focus_area=e["focus_area"])
                else:
                    tracer.event(e["name"], e["focus_area"], e.get("reason", ""), **extras)
    trace_id = tracer.trace_id
    tracer.flush()
    if not tracer.export_ok:
        _log(payload, f"{payload.get('name')} trace {trace_id} not exported ({tracer.export_status}); "
                      f"journal kept in {tracer.journal_path}")
        return False
    _log(payload, f"sent {payload.get('name')} trace {trace_id}")
    return True


def main(argv: list[str]) -> int:
    payload: dict = {}
    try:
        payload = json.loads(argv[1])
        if payload.get("kind") == "trace":
            replay(payload)
        else:
            send(payload)
    except Exception as e:
        _log(payload, f"error: {e!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
