"""Step 9: one neatlogs trace per check, a span per agent step and per model/judge call.

Every step and call span carries a `focus_area` attribute (one of FOCUS_AREAS). Decisions
and recoveries are logged as events with a reason: on the current span, as a neatlogs log
line, and in an in-memory journal that is saved with the results.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time
from typing import Any, Callable

from notbydefault.env import load_env

MAX_ATTR_CHARS = 4000
FOCUS_AREAS = ("planning", "tool_use", "state", "context", "execution", "recovery",
               "verification", "iteration")
FOCUS_LABELS = {"planning": "Planning", "tool_use": "Tool use", "state": "State",
                "context": "Context", "execution": "Execution", "recovery": "Recovery",
                "verification": "Verification", "iteration": "Iteration"}
# journals saved before the 2026-10-12 retag carry the old tag: (name, old) -> today's tag
RETAGGED = {("candidate_selected", "planning"): "tool_use",
            ("judge_selected", "planning"): "tool_use",
            ("verifier_model_selected", "planning"): "tool_use",
            ("retry_with_feedback", "recovery"): "iteration",
            ("expand_sample", "context"): "iteration",
            ("refresh_cache", "planning"): "execution"}
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


# a model call that fails in one of these ways is an expected code path the caller handles
# (skip, stop and resume, swap, fall back), so its span ends OK with nbd.outcome; only other
# exceptions mark a span ERROR
HANDLED_OUTCOMES = {"offline_cache_miss", "daily_limit", "preflight_unavailable", "judge_fallback"}


def handled_outcome(error: BaseException, before: dict) -> str | None:
    """The HANDLED_OUTCOMES entry a failed model call stands for, or None for a real failure.
    `before` is the call's span attributes (its role, and for a preflight probe the role probed)."""
    outcome = getattr(error, "outcome", None)
    if outcome in HANDLED_OUTCOMES:
        return outcome
    from notbydefault.llm import ModelUnavailable
    if isinstance(error, ModelUnavailable) and before.get("notbydefault.role") == "preflight":
        # preflight swaps an unavailable model; for the judge it falls back to a second judge,
        # a freed candidate, or reference-only scoring
        return ("judge_fallback" if before.get("notbydefault.for_role") in ("judge", "judge2")
                else "preflight_unavailable")
    return None


def current_focus(entry: dict) -> str:
    """The focus_area the code gives this journal entry today (a saved one may predate a retag)."""
    return RETAGGED.get((entry.get("name"), entry.get("focus_area")), entry.get("focus_area"))


def step_strip(journal: list[dict]) -> str:
    """The run's agent steps in order with their focus areas, as the UI's step strip shows them:
    "plan [planning]  →  judge ×2 [verification]  →  ...". The root span is left out."""
    steps: list[list] = []
    for e in journal:
        if e.get("kind") != "step" or e.get("name") == "notbydefault.check":
            continue
        if steps and steps[-1][0] == e["name"]:
            steps[-1][2] += 1
        else:
            steps.append([e["name"], current_focus(e), 1])
    return "  →  ".join(f"{n}{f' ×{k}' if k > 1 else ''} [{a}]" for n, a, k in steps)


class SpanAttrs:
    """Attributes collected while a span runs, set on it when it ends."""

    def __init__(self):
        self.attrs: dict = {}

    def set(self, **attrs) -> None:
        self.attrs.update(attrs)


def _safe_call(fn: Callable[[], dict]) -> dict:
    try:
        return fn() or {}
    except Exception:
        return {}


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
    def root(self, name: str, attrs: dict, finish: Callable[[], dict] | None = None):
        """The run's root span. `finish()` returns attributes set on it as it ends, on every
        exit path (early returns and errors included)."""
        self._note("step", name, "execution")
        try:
            yield
        finally:
            if finish is not None:
                self._note("event", "root_summary", "execution", "", **_safe_call(finish))

    @contextlib.contextmanager
    def step(self, name: str, focus_area: str, reason: str = "", attrs: dict | None = None):
        """An explicit agent step (its own span)."""
        self._note("step", name, focus_area, reason, **(attrs or {}))
        yield

    @contextlib.contextmanager
    def span(self, name: str, kind: str, focus_area: str, attrs: dict | None = None):
        """A lightweight span of a neatlogs kind (e.g. GUARDRAIL) that is not an agent step:
        not journaled. Yields a SpanAttrs; what is .set() on it lands on the span as it ends."""
        _check_focus(focus_area)
        yield SpanAttrs()

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


EXPORT_TIMEOUT_S = 5        # each OTLP export request (spans and logs)
FLUSH_DEADLINE_S = 20       # flush + shutdown together, however the backend behaves


def _run_with_deadline(fn: Callable[[], Any], seconds: float) -> tuple[bool, Any]:
    """(finished, result) of fn run in a daemon thread; a hung fn is abandoned, not awaited."""
    box: dict = {}

    def target():
        try:
            box["result"] = fn()
        except Exception as e:      # reported to the caller as a failure
            box["error"] = e
    t = threading.Thread(target=target, name="notbydefault-telemetry", daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        return False, None
    if "error" in box:
        raise box["error"]
    return True, box.get("result")


def default_spill_dir() -> "Path":
    import tempfile
    from pathlib import Path
    return Path(tempfile.gettempdir()) / "notbydefault-telemetry"


class NeatlogsTracer(Tracer):
    """Exports to neatlogs, failing open: telemetry never blocks or breaks the work.

    Spans leave through neatlogs' background batch exporter with a short per-request timeout;
    flush and shutdown run against a hard deadline. Any neatlogs error switches export off for
    the rest of the run (one warning), and when export failed or timed out the journal (every
    step, event and call) is written to a local file so nothing is lost.
    """
    enabled = True
    export_ok = True          # False after the first neatlogs failure
    export_status = "ok"      # ok | failed: ... | timed out ...
    journal_path: str | None = None
    spill_dir = None

    def _warn(self, msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    def __init__(self, session_id: str | None = None, spill_dir=None, warn=None):
        super().__init__()
        for var in ("OTEL_EXPORTER_OTLP_TRACES_TIMEOUT", "OTEL_EXPORTER_OTLP_LOGS_TIMEOUT"):
            os.environ.setdefault(var, str(EXPORT_TIMEOUT_S))
        import neatlogs
        from opentelemetry import context as otel_context
        from opentelemetry import trace as otel_trace

        self.neatlogs, self.otel_context, self.otel_trace = neatlogs, otel_context, otel_trace
        self.session_id = session_id
        self.spill_dir = spill_dir
        if warn:
            self._warn = warn
        neatlogs.init(workflow_name="notbydefault", tags=["check"],
                      register_shutdown_handlers=False)

    # --- fail-open plumbing ---------------------------------------------------------

    def _fail(self, what: str, error) -> None:
        if not self.export_ok:
            return
        self.export_ok = False
        self.export_status = f"failed: {what}: {error!r}"[:300]
        self._note("event", "telemetry_failed", "recovery",
                   f"neatlogs {what} failed ({error!r}); continuing without export, journal "
                   "kept locally")
        self._warn(f"NotByDefault: neatlogs {what} failed ({error!r}); continuing without "
                   "telemetry export (the journal is kept locally).")

    def _guard(self, what: str, fn: Callable[[], Any]):
        if not self.export_ok:
            return None
        try:
            return fn()
        except Exception as e:
            self._fail(what, e)
            return None

    @contextlib.contextmanager
    def _span_cm(self, what: str, factory: Callable[[], Any]):
        """Enter a neatlogs context manager if possible; the body runs either way."""
        cm = self._guard(what, factory) if self.export_ok else None
        entered = False
        if cm is not None:
            try:
                cm.__enter__()
                entered = True
            except Exception as e:
                self._fail(what, e)
        try:
            yield entered
        finally:
            if entered:
                try:
                    cm.__exit__(None, None, None)
                except Exception as e:
                    self._fail(what, e)

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
        def go():
            span = self._span()
            for k, v in attrs.items():
                if v is not None:
                    span.set_attribute(k, _attr(v))
        self._guard("set_attribute", go)

    # --- Tracer API -------------------------------------------------------------------

    @contextlib.contextmanager
    def root(self, name: str, attrs: dict, finish: Callable[[], dict] | None = None):
        self._roots_open = getattr(self, "_roots_open", 0) + 1
        try:
            with self._root(name, attrs, finish):
                yield
        finally:
            self._roots_open -= 1
            if not self._roots_open and getattr(self, "_flush_pending", False):
                self._flush_pending = False
                self.flush()

    @contextlib.contextmanager
    def _root(self, name: str, attrs: dict, finish: Callable[[], dict] | None):
        with self._span_cm("trace", lambda: self.neatlogs.trace(
                name=name, kind="WORKFLOW", session_id=self.session_id)) as entered:
            if entered:
                ctx = self._guard("trace id", lambda: self._span().get_span_context())
                if ctx is not None and ctx.trace_id:
                    self.trace_id = format(ctx.trace_id, "032x")
            self._note("step", name, "execution", trace_id=self.trace_id)
            if entered:
                self._set({"focus_area": "execution", **attrs})
            try:
                yield
            finally:
                if finish is not None:
                    summary = _safe_call(finish)
                    self._note("event", "root_summary", "execution", "", **summary)
                    if entered:
                        self._set(summary)

    @contextlib.contextmanager
    def step(self, name: str, focus_area: str, reason: str = "", attrs: dict | None = None):
        self._note("step", name, focus_area, reason, **(attrs or {}))
        with self._span_cm("span", lambda: self.neatlogs.trace(name=name, kind="CHAIN")) as entered:
            if entered:
                self._set({"focus_area": focus_area, "notbydefault.reason": reason or None,
                           **{f"notbydefault.{k}": v for k, v in (attrs or {}).items()}})
            yield

    @contextlib.contextmanager
    def span(self, name: str, kind: str, focus_area: str, attrs: dict | None = None):
        _check_focus(focus_area)
        box = SpanAttrs()
        with self._span_cm("span", lambda: self.neatlogs.trace(name=name, kind=kind)) as entered:
            if entered:
                self._set({"focus_area": focus_area, **(attrs or {})})
            try:
                yield box
            finally:
                if entered and box.attrs:
                    self._set(box.attrs)

    def event(self, name: str, focus_area: str, reason: str, **attrs) -> None:
        self._note("event", name, focus_area, reason, **attrs)

        def go():
            self._span().add_event(name, {"focus_area": focus_area, "reason": reason[:MAX_ATTR_CHARS],
                                          **{k: _attr(v) for k, v in attrs.items() if v is not None}})
            self.neatlogs.log("{event}: {reason}", level="warning" if focus_area == "recovery"
                              else "info", event=name, focus_area=focus_area, reason=reason,
                              **{k: v for k, v in attrs.items() if v is not None})
        self._guard("event", go)

    def llm(self, name, fn, before, after, kind="TOOL", focus_area="tool_use"):
        """A model call as a neatlogs LLM span, so neatlogs shows its tokens and spend.

        neatlogs' span decorator has no LLM kind; its own provider wrappers (neatlogs.wrap)
        start spans with neatlogs.span.kind="llm" and neatlogs.llm.* attributes through its
        provider tracer, and that is what this does for our plain-HTTP calls. `kind` (the
        call's role: judge -> EVALUATOR, verifier -> GUARDRAIL) is kept as an attribute."""
        _check_focus(focus_area)
        if self.record_calls:
            self._note("call", name, focus_area, "", span_kind=kind,
                       attrs={k: v for k, v in before.items() if k != "input.value"})
        if not self.export_ok:
            return fn()
        span = self._guard("llm span", lambda: self._start_llm_span(name, before, kind, focus_area))
        if span is None:
            return fn()
        try:
            result = fn()
        except BaseException as e:      # the model call's own error: recorded, re-raised as is
            outcome = handled_outcome(e, before)

            def failed():
                from opentelemetry.trace import StatusCode
                if outcome:             # expected and handled by the caller: not an error
                    span.set_attribute("nbd.outcome", outcome)
                    span.set_attribute("nbd.outcome_reason", str(e)[:500])
                    span.set_status(StatusCode.OK)
                else:
                    span.set_status(StatusCode.ERROR, str(e)[:500])
                    span.record_exception(e)
                span.end()
            self._guard("llm span", failed)
            raise
        self._guard("llm span", lambda: self._end_llm_span(
            span, {**before, **_safe_after(after, result)}))
        return result

    def _start_llm_span(self, name: str, before: dict, kind: str, focus_area: str):
        from neatlogs._wrap_utils import get_provider_tracer
        model = str(before.get("llm.model_name") or "")
        provider, _, native = model.partition(":") if ":" in model else ("", "", model)
        params = before.get("llm.invocation_parameters") or {}
        attrs = {"neatlogs.span.kind": "llm", "neatlogs.llm.provider": provider or "unknown",
                 "neatlogs.llm.system": provider or "unknown", "neatlogs.llm.model_name": native,
                 "neatlogs.llm.request_model": native, "focus_area": focus_area,
                 "notbydefault.span_kind": kind}
        for p in ("temperature", "max_tokens"):
            if isinstance(params, dict) and params.get(p) is not None:
                attrs[f"neatlogs.llm.{p}"] = params[p]
        if before.get("input.value") is not None:
            attrs["neatlogs.llm.input_messages.0.role"] = "user"
            attrs["neatlogs.llm.input_messages.0.content"] = _attr(before["input.value"])
        attrs.update({k: _attr(v) for k, v in before.items()
                      if v is not None and k != "input.value"})
        return get_provider_tracer().start_span(name=name, attributes=attrs)

    @staticmethod
    def _end_llm_span(span, attrs: dict) -> None:
        from opentelemetry.trace import StatusCode
        tin, tout = attrs.get("llm.token_count.prompt"), attrs.get("llm.token_count.completion")
        cached = bool(attrs.get("notbydefault.cached"))
        usage = {"neatlogs.llm.token_count.prompt": tin,
                 "neatlogs.llm.token_count.completion": tout,
                 "neatlogs.llm.token_count.total": attrs.get("llm.token_count.total"),
                 # spend: what this call cost now (a cache hit costs nothing); the list-price
                 # cost of its tokens stays in notbydefault.cost_usd
                 "neatlogs.llm.cost_usd": 0.0 if cached else attrs.get("notbydefault.cost_usd"),
                 "neatlogs.llm.metrics.duration_ms":
                     round(float(attrs.get("notbydefault.latency_s") or 0) * 1000, 3)}
        if attrs.get("output.value") is not None:
            usage["neatlogs.llm.output_messages.0.role"] = "assistant"
            usage["neatlogs.llm.output_messages.0.content"] = _attr(attrs["output.value"])
        if attrs.get("notbydefault.finish_reason"):
            usage["neatlogs.llm.output_messages.0.finish_reason"] = attrs["notbydefault.finish_reason"]
        rest = {k: v for k, v in attrs.items()
                if k not in ("input.value", "output.value", "llm.invocation_parameters")}
        for k, v in {**rest, **usage}.items():
            if v is not None:
                span.set_attribute(k, _attr(v))
        span.set_status(StatusCode.OK)
        span.end()

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
        """Export what is queued, but never wait longer than FLUSH_DEADLINE_S.

        Called inside a root span (an early return), it waits until the root ends, so the
        root's closing attributes are exported too."""
        if getattr(self, "_roots_open", 0):
            self._flush_pending = True
            return
        if getattr(self, "_flushed", False):
            return
        self._flushed = True
        failures = 0
        if self.export_ok:
            started = time.monotonic()
            try:
                done, _ = _run_with_deadline(
                    lambda: self.neatlogs.flush(timeout_millis=int(FLUSH_DEADLINE_S * 700)),
                    FLUSH_DEADLINE_S)
                if done:
                    try:
                        failures = int(self.neatlogs.get_delivery_diagnostics()
                                       .get("span_export_failures", 0) or 0)
                    except Exception:
                        failures = 0
                    left = max(1.0, FLUSH_DEADLINE_S - (time.monotonic() - started))
                    _run_with_deadline(lambda: self.neatlogs.shutdown(
                        timeout_millis=int(left * 700)), left)
                else:
                    self.export_ok = False
                    self.export_status = f"timed out flushing after {FLUSH_DEADLINE_S}s"
                    self._warn(f"NotByDefault: neatlogs did not finish exporting within "
                               f"{FLUSH_DEADLINE_S}s; continuing (the journal is kept locally).")
            except Exception as e:
                self._fail("flush", e)
            if failures and self.export_ok:
                self.export_ok = False
                self.export_status = f"failed: {failures} span export batch(es) failed"
                self._warn(f"NotByDefault: {failures} neatlogs span export batch(es) failed; "
                           "the journal is kept locally.")
        if not self.export_ok:
            self.journal_path = self.spill()

    def spill(self) -> str | None:
        """Write the journal to a local file (when export failed); returns its path."""
        from pathlib import Path
        try:
            folder = Path(self.spill_dir) if self.spill_dir else default_spill_dir()
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{time.strftime('%Y%m%dT%H%M%S')}-{self.trace_id or 'no-trace'}.json"
            path.write_text(json.dumps({"trace_id": self.trace_id, "session_id": self.session_id,
                                        "export_status": self.export_status,
                                        "journal": self.journal}, indent=1, default=str),
                            encoding="utf-8")
            self._warn(f"NotByDefault: telemetry journal saved to {path}")
            return str(path)
        except OSError as e:
            self._warn(f"NotByDefault: could not save the telemetry journal ({e!r})")
            return None


def _safe_after(after: Callable[[Any], dict], result) -> dict:
    try:
        return after(result) or {}
    except Exception:
        return {}


def make_tracer(session_id: str | None = None, spill_dir=None) -> Tracer:
    load_env()
    if not os.environ.get("NEATLOGS_API_KEY") or os.environ.get("NOTBYDEFAULT_TELEMETRY") == "0":
        return Tracer()
    try:
        return NeatlogsTracer(session_id, spill_dir=spill_dir)
    except Exception as e:
        print(f"NotByDefault: neatlogs unavailable ({e!r}); continuing without telemetry export.",
              file=sys.stderr, flush=True)
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
