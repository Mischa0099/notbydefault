"""Step 9: one neatlogs trace per check, a span per agent step and per model/judge call.

Every step and call span carries a `focus_area` attribute (one of FOCUS_AREAS). Decisions
and recoveries are logged as events with a reason: on the current span, as a neatlogs log
line, and in an in-memory journal that is saved with the results.
"""
from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from typing import Any, Callable

from notbydefault.env import load_env

MAX_ATTR_CHARS = 4000
FOCUS_AREAS = ("planning", "tool_use", "state", "context", "execution", "recovery",
               "verification", "iteration")
# neatlogs span kinds per call role (its decorator has no "LLM" kind)
SPAN_KIND = {"candidate": "TOOL", "analyst": "TOOL", "preflight": "TOOL",
             "judge": "EVALUATOR", "verifier": "GUARDRAIL"}
# focus_area per call role
ROLE_FOCUS = {"candidate": "tool_use", "analyst": "context", "preflight": "tool_use",
              "judge": "verification", "verifier": "verification"}


def _check_focus(focus_area: str) -> str:
    if focus_area not in FOCUS_AREAS:
        raise ValueError(f"focus_area must be one of {FOCUS_AREAS}, not {focus_area!r}")
    return focus_area


class Tracer:
    """No-op tracer (tests, or no NEATLOGS_API_KEY) that still keeps the journal."""
    enabled = False
    trace_id: str | None = None   # neatlogs trace id of the current check, when exporting
    record_calls = False          # journal every model call too (hooks replay it later)

    def __init__(self):
        self.journal: list[dict] = []
        self._journal_lock = threading.Lock()

    def _note(self, kind: str, name: str, focus_area: str, reason: str = "", **attrs) -> None:
        if not hasattr(self, "journal"):   # subclasses that skip __init__
            Tracer.__init__(self)
        with self._journal_lock:
            self.journal.append({"t": round(time.time(), 3), "kind": kind, "name": name,
                                 "focus_area": _check_focus(focus_area), "reason": reason,
                                 **{k: v for k, v in attrs.items() if v is not None}})

    @contextlib.contextmanager
    def root(self, name: str, attrs: dict):
        self._note("step", name, "execution")
        yield

    @contextlib.contextmanager
    def step(self, name: str, focus_area: str, reason: str = "", attrs: dict | None = None):
        """An explicit agent step (its own span)."""
        self._note("step", name, focus_area, reason, **(attrs or {}))
        yield

    def event(self, name: str, focus_area: str, reason: str, **attrs) -> None:
        """A decision or recovery inside the current step/span, with the reason for it."""
        self._note("event", name, focus_area, reason, **attrs)

    def llm(self, name: str, fn: Callable[[], Any], before: dict,
            after: Callable[[Any], dict], kind: str = "TOOL", focus_area: str = "tool_use") -> Any:
        _check_focus(focus_area)
        result = fn()
        if self.record_calls:
            attrs = {k: v for k, v in before.items() if k != "input.value"}
            try:
                attrs.update({k: v for k, v in (after(result) or {}).items() if k != "output.value"})
            except Exception:
                pass
            self._note("call", name, focus_area, "", span_kind=kind, attrs=attrs)
        return result

    def annotate(self, attrs: dict) -> None:
        """Set attributes on the current span (call inside root())."""

    def bind(self, fn: Callable) -> Callable:
        """Wrap fn so it runs inside the current trace when called from a worker thread."""
        return fn

    def flush(self) -> None:
        pass

    def summary(self) -> dict[str, int]:
        """Count of journal events per name, e.g. {"rate_limit_backoff": 2}."""
        out: dict[str, int] = {}
        for e in getattr(self, "journal", []):
            if e["kind"] == "event":
                out[e["name"]] = out.get(e["name"], 0) + 1
        return out


def _attr(value):
    if isinstance(value, (bool, int, float, str)):
        return value if not isinstance(value, str) else value[:MAX_ATTR_CHARS]
    return json.dumps(value, default=str)[:MAX_ATTR_CHARS]


class NeatlogsTracer(Tracer):
    enabled = True

    def __init__(self, session_id: str | None = None):
        super().__init__()
        import neatlogs
        from opentelemetry import context as otel_context
        from opentelemetry import trace as otel_trace

        self.neatlogs, self.otel_context, self.otel_trace = neatlogs, otel_context, otel_trace
        self.session_id = session_id
        neatlogs.init(workflow_name="notbydefault", tags=["check"],
                      register_shutdown_handlers=False)

    def _span(self):
        """The active neatlogs span. In neatlogs' isolated mode it lives on a private
        context key rather than OTel's current span, so ask neatlogs first."""
        try:
            from neatlogs._wrap_utils import _current_neatlogs_parent
            span = _current_neatlogs_parent()
            if span is not None:
                return span
        except Exception:
            pass
        return self.otel_trace.get_current_span()

    def _set(self, attrs: dict) -> None:
        span = self._span()
        for k, v in attrs.items():
            if v is not None:
                span.set_attribute(k, _attr(v))

    @contextlib.contextmanager
    def root(self, name: str, attrs: dict):
        with self.neatlogs.trace(name=name, kind="WORKFLOW", session_id=self.session_id):
            ctx = self._span().get_span_context()
            if ctx.trace_id:
                self.trace_id = format(ctx.trace_id, "032x")
            self._note("step", name, "execution", trace_id=self.trace_id)
            self._set({"focus_area": "execution", **attrs})
            yield

    @contextlib.contextmanager
    def step(self, name: str, focus_area: str, reason: str = "", attrs: dict | None = None):
        self._note("step", name, focus_area, reason, **(attrs or {}))
        with self.neatlogs.trace(name=name, kind="CHAIN"):
            self._set({"focus_area": focus_area, "notbydefault.reason": reason or None,
                       **{f"notbydefault.{k}": v for k, v in (attrs or {}).items()}})
            yield

    def event(self, name: str, focus_area: str, reason: str, **attrs) -> None:
        self._note("event", name, focus_area, reason, **attrs)
        try:
            self._span().add_event(name, {"focus_area": focus_area, "reason": reason[:MAX_ATTR_CHARS],
                                          **{k: _attr(v) for k, v in attrs.items() if v is not None}})
            self.neatlogs.log("{event}: {reason}", level="warning" if focus_area == "recovery"
                              else "info", event=name, focus_area=focus_area, reason=reason,
                              **{k: v for k, v in attrs.items() if v is not None})
        except Exception:
            pass

    def llm(self, name, fn, before, after, kind="TOOL", focus_area="tool_use"):
        _check_focus(focus_area)

        @self.neatlogs.span(kind=kind, name=name, capture_input=False, capture_output=False)
        def _call():
            self._set({"focus_area": focus_area, **before})
            result = fn()
            self._set(after(result))
            return result
        return _call()

    def annotate(self, attrs):
        self._set(attrs)

    def bind(self, fn):
        ctx = self.otel_context.get_current()

        def run(*args, **kwargs):
            token = self.otel_context.attach(ctx)
            try:
                return fn(*args, **kwargs)
            finally:
                self.otel_context.detach(token)
        return run

    def flush(self) -> None:
        self.neatlogs.flush(timeout_millis=30000)
        self.neatlogs.shutdown(timeout_millis=10000)


def make_tracer(session_id: str | None = None) -> Tracer:
    load_env()
    if not os.environ.get("NEATLOGS_API_KEY") or os.environ.get("NOTBYDEFAULT_TELEMETRY") == "0":
        return Tracer()
    try:
        return NeatlogsTracer(session_id)
    except Exception:
        return Tracer()


def chat_attrs(result) -> dict:
    """Span attributes after a model call (OpenInference names + ours)."""
    return {
        "llm.token_count.prompt": result.input_tokens,
        "llm.token_count.completion": result.output_tokens,
        "llm.token_count.total": result.input_tokens + result.output_tokens,
        "notbydefault.cost_usd": round(result.cost, 8),
        "notbydefault.latency_s": round(result.latency_s, 3),
        "notbydefault.finish_reason": result.finish_reason or "",
        "notbydefault.cached": bool(getattr(result, "cached", False)),
        "output.value": result.text,
    }
