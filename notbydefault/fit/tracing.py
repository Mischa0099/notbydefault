"""Step 9: one neatlogs trace per check, one span per model call and per judge call."""
from __future__ import annotations

import contextlib
import json
import os
from typing import Any, Callable

from notbydefault.env import load_env

MAX_ATTR_CHARS = 4000
# neatlogs span kinds per call role (its decorator has no "LLM" kind)
SPAN_KIND = {"candidate": "TOOL", "analyst": "TOOL", "judge": "EVALUATOR", "verifier": "GUARDRAIL"}


class Tracer:
    """No-op tracer (used in tests and when NEATLOGS_API_KEY is missing)."""
    enabled = False

    @contextlib.contextmanager
    def root(self, name: str, attrs: dict):
        yield

    def llm(self, name: str, fn: Callable[[], Any], before: dict,
            after: Callable[[Any], dict], kind: str = "TOOL") -> Any:
        return fn()

    def annotate(self, attrs: dict) -> None:
        """Set attributes on the current span (call inside root())."""

    def bind(self, fn: Callable) -> Callable:
        """Wrap fn so it runs inside the current trace when called from a worker thread."""
        return fn

    def flush(self) -> None:
        pass


def _attr(value):
    if isinstance(value, (bool, int, float, str)):
        return value if not isinstance(value, str) else value[:MAX_ATTR_CHARS]
    return json.dumps(value, default=str)[:MAX_ATTR_CHARS]


class NeatlogsTracer(Tracer):
    enabled = True

    def __init__(self, session_id: str | None = None):
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

    @contextlib.contextmanager
    def root(self, name: str, attrs: dict):
        with self.neatlogs.trace(name=name, kind="WORKFLOW", session_id=self.session_id):
            span = self._span()
            for k, v in attrs.items():
                span.set_attribute(k, _attr(v))
            yield

    def llm(self, name, fn, before, after, kind="TOOL"):
        @self.neatlogs.span(kind=kind, name=name, capture_input=False, capture_output=False)
        def _call():
            span = self._span()
            for k, v in before.items():
                span.set_attribute(k, _attr(v))
            result = fn()
            for k, v in after(result).items():
                if v is not None:
                    span.set_attribute(k, _attr(v))
            return result
        return _call()

    def annotate(self, attrs):
        span = self._span()
        for k, v in attrs.items():
            span.set_attribute(k, _attr(v))

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
        "output.value": result.text,
    }
