import json
"""The real neatlogs tracer, exporting to memory: one trace per check, attributes on spans."""
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from conftest import JUDGE, make_client
from notbydefault.fit import tracing
from notbydefault.fit.run import CheckOptions, check_model_decision
from test_fit_check import project  # noqa: F401  (fixture)


class MemoryTracer(tracing.NeatlogsTracer):
    def __init__(self, provider):
        import neatlogs
        from opentelemetry import context, trace
        tracing.Tracer.__init__(self)
        self.neatlogs, self.otel_context, self.otel_trace = neatlogs, context, trace
        self.session_id = "test"
        neatlogs.init(api_key="test", workflow_name="notbydefault", disable_export=True,
                      tracer_provider=provider, register_shutdown_handlers=False)

    def flush(self):
        pass


def test_one_trace_with_model_judge_and_verifier_spans(project):  # noqa: F811
    import neatlogs
    p, did = project
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    code = check_model_decision(p, did, CheckOptions(yes=True, calls_per_user=30, workers=3),
                                client=make_client(), tracer=MemoryTracer(provider),
                                out=lambda _: None)
    assert code == 0
    spans = exporter.get_finished_spans()
    assert len({s.context.trace_id for s in spans}) == 1
    names = {s.name for s in spans}
    assert {"notbydefault.check", "notbydefault.model_call", "notbydefault.judge_call",
            "notbydefault.verifier_call"} <= names

    model = next(s for s in spans if s.name == "notbydefault.model_call").attributes
    for key in ("llm.model_name", "llm.token_count.prompt", "llm.token_count.completion",
                "notbydefault.cost_usd", "notbydefault.latency_s"):
        assert key in model, key
    judge = next(s for s in spans if s.name == "notbydefault.judge_call").attributes
    assert judge["llm.model_name"] == JUDGE
    assert any(k.startswith("notbydefault.score.") for k in judge)
    root_span = next(s for s in spans if s.name == "notbydefault.check")
    results = json.loads(sorted((p / ".notbydefault" / "results").glob("*.json"))[-1].read_text())
    assert results["trace_id"] == format(root_span.context.trace_id, "032x")
    root = root_span.attributes
    assert root["notbydefault.verdict"] == "switch"

    # every agent step and tool span is tagged with a focus_area
    from notbydefault.fit.tracing import FOCUS_AREAS
    steps = ("plan", "preflight", "context", "verifiers", "execute", "judge", "decide", "record")
    ours = [s for s in spans if s.name.startswith("notbydefault.") or s.name in steps]
    assert {s.name for s in ours} >= set(steps)
    assert all(s.attributes.get("focus_area") in FOCUS_AREAS for s in ours)
    plan = next(s for s in spans if s.name == "plan")
    assert plan.attributes["focus_area"] == "planning"
    assert "Candidates:" in plan.attributes["notbydefault.plan"]
    retries = [s for s in spans if s.name == "notbydefault.model_call"
               and s.attributes.get("notbydefault.attempt", 0) > 0]
    assert retries and all(s.attributes["focus_area"] == "iteration" for s in retries)
    events = [e for s in spans for e in s.events if e.name == "retry_with_feedback"]
    assert events and all(e.attributes["focus_area"] == "recovery" and e.attributes["reason"]
                          for e in events)

    # neatlogs Condition attributes ------------------------------------------------------
    # model calls are neatlogs LLM spans with usage and spend
    for s in (s for s in spans if s.name in ("notbydefault.model_call", "notbydefault.judge_call",
                                             "notbydefault.verifier_call")):
        a = s.attributes
        assert a["neatlogs.span.kind"] == "llm" and a["neatlogs.llm.provider"] in ("groq", "gemini")
        assert a["neatlogs.llm.token_count.prompt"] > 0 and "neatlogs.llm.cost_usd" in a
        norm = neatlogs.normalize_span_v2(s)      # what neatlogs' backend schema sees
        assert norm.kind == "LLM" and norm.semantic["usage"]["input_tokens"] > 0
        assert norm.semantic["usage"]["cost_usd"] is not None
    # every verification is a GUARDRAIL span with nbd.verifier_failed; the rubric call too
    verifies = [s for s in spans if s.name == "notbydefault.verify"]
    assert verifies and all(isinstance(s.attributes["nbd.verifier_failed"], bool) for s in verifies)
    assert {s.attributes["nbd.verifier_failed"] for s in verifies} == {True, False}
    assert all(s.attributes["neatlogs.span.kind"] == "guardrail" for s in verifies)
    assert all("nbd.verifier_failed" in s.attributes for s in spans
               if s.name == "notbydefault.verifier_call")
    # the check root carries the run summary
    assert root["nbd.swap_count"] == 0 and root["nbd.stopped_early"] is False
    assert root["nbd.retry_rescued"] >= 1          # the fake 20b fails alone, passes on retry
    assert root["nbd.cost_usd"] > 0
    neatlogs.shutdown(timeout_millis=1000)


# --- telemetry fails open ---------------------------------------------------------------------

class BrokenNeatlogs:
    """neatlogs whose every call raises (backend down, bad key, SDK bug...)."""
    def trace(self, **k):
        raise ConnectionError("ingest unreachable")

    def span(self, **k):
        raise ConnectionError("ingest unreachable")

    def log(self, *a, **k):
        raise ConnectionError("ingest unreachable")

    def flush(self, **k):
        raise ConnectionError("ingest unreachable")


class HangingNeatlogs(BrokenNeatlogs):
    def trace(self, **k):
        import contextlib
        return contextlib.nullcontext()

    def log(self, *a, **k):
        pass

    def span(self, **k):
        return lambda fn: fn

    def flush(self, **k):
        import time
        time.sleep(30)            # an export that never comes back


def fake_tracer(neatlogs, tmp_path, warnings):
    from opentelemetry import context, trace
    t = tracing.NeatlogsTracer.__new__(tracing.NeatlogsTracer)
    tracing.Tracer.__init__(t)
    t.neatlogs, t.otel_context, t.otel_trace = neatlogs, context, trace
    t.session_id, t.spill_dir, t._warn = "s", tmp_path, warnings.append
    return t


def test_broken_neatlogs_never_breaks_the_work(tmp_path):
    warnings = []
    t = fake_tracer(BrokenNeatlogs(), tmp_path, warnings)
    ran = []
    with t.root("notbydefault.eval.x", {}):
        with t.step("case", "verification", reason="r"):
            t.event("case_pass", "verification", "ok")
            assert t.llm("call", lambda: 42, before={}, after=lambda r: {}) == 42
            ran.append(True)
    t.flush()
    assert ran == [True]
    assert len([w for w in warnings if "failed" in w]) == 1      # one warning, not one per call
    assert t.export_ok is False and t.export_status.startswith("failed: trace")
    saved = json.loads(open(t.journal_path, encoding="utf-8").read())
    assert [e["name"] for e in saved["journal"] if e["kind"] == "event"][-1] == "case_pass"


def test_model_errors_still_propagate_through_a_broken_tracer(tmp_path):
    import pytest
    t = fake_tracer(BrokenNeatlogs(), tmp_path, [])

    def boom():
        raise ValueError("model failed")
    with pytest.raises(ValueError):
        t.llm("call", boom, before={}, after=lambda r: {})


def test_flush_has_a_deadline(tmp_path, monkeypatch):
    import time
    monkeypatch.setattr(tracing, "FLUSH_DEADLINE_S", 0.5)
    warnings = []
    t = fake_tracer(HangingNeatlogs(), tmp_path, warnings)
    with t.root("r", {}):
        t.event("e", "verification", "x")
    start = time.monotonic()
    t.flush()
    assert time.monotonic() - start < 3
    assert "timed out" in t.export_status and t.journal_path
    assert any("did not finish exporting" in w for w in warnings)


def test_make_tracer_falls_back_when_neatlogs_cannot_start(monkeypatch):
    monkeypatch.setenv("NEATLOGS_API_KEY", "x")
    monkeypatch.delenv("NOTBYDEFAULT_TELEMETRY", raising=False)

    def explode(*a, **k):
        raise RuntimeError("cannot init")
    monkeypatch.setattr(tracing.NeatlogsTracer, "__init__", explode)
    t = tracing.make_tracer()
    assert type(t) is tracing.Tracer


# --- eval case and hook spans carry the nbd.* Condition attributes ---------------------------

def memory():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, MemoryTracer(provider)


def test_eval_case_spans_carry_failure_and_version():
    import neatlogs
    from notbydefault import evals
    exporter, t = memory()
    evals._version = "vtest"
    with t.root("notbydefault.eval.detection", {}):
        res = evals.run_detection(t, out=lambda _: None, which="all")
    cases = {s.name[len("case:"):]: s.attributes for s in exporter.get_finished_spans()
             if s.name.startswith("case:")}
    assert set(cases) == {r["case"] for r in res["cases"]}
    for r in res["cases"]:
        a = cases[r["case"]]
        assert a["nbd.eval_case_failed"] is (not r["ok"]) and a["nbd.eval_version"] == "vtest"
        assert a["focus_area"] == "verification"
    neatlogs.shutdown(timeout_millis=1000)


def test_hook_trace_root_carries_interrupted_and_hook_ms(monkeypatch):
    import neatlogs
    from notbydefault import telemetry
    exporter, t = memory()
    monkeypatch.setenv("NEATLOGS_API_KEY", "test")
    monkeypatch.setattr(tracing, "NeatlogsTracer", lambda *a, **k: t)
    telemetry.replay({"kind": "trace", "name": "notbydefault.pre_tool", "info": {},
                      "nbd": {"nbd.interrupted": True, "nbd.hook_ms": 41.5},
                      "journal": [{"kind": "step", "name": "detect", "focus_area": "tool_use"}]})
    root = next(s for s in exporter.get_finished_spans() if s.name == "notbydefault.pre_tool")
    assert root.attributes["nbd.interrupted"] is True and root.attributes["nbd.hook_ms"] == 41.5
    assert root.attributes["focus_area"] == "execution"
    neatlogs.shutdown(timeout_millis=1000)


def test_pre_tool_reports_interrupted_and_hook_ms(tmp_path, monkeypatch):
    from notbydefault.hooks import pre_tool
    sent = []
    monkeypatch.setenv("NOTBYDEFAULT_TELEMETRY", "1")
    monkeypatch.setattr(pre_tool, "_spawn_telemetry", lambda payload, project: sent.append(payload))
    ev = {"session_id": "s", "tool_name": "Bash", "cwd": str(tmp_path),
          "tool_input": {"command": "pip install pymongo"}}
    assert pre_tool.handle(ev, tmp_path)["hookSpecificOutput"]["permissionDecision"] == "ask"
    (tmp_path / "requirements.txt").write_text("pymongo\n")      # now already used: no ask
    assert pre_tool.handle(ev, tmp_path) is None
    assert [p["nbd"]["nbd.interrupted"] for p in sent] == [True, False]
    assert all(isinstance(p["nbd"]["nbd.hook_ms"], float) for p in sent)


def test_flush_inside_the_root_waits_for_it_so_closing_attributes_export(tmp_path):
    flushed = []

    class Recording(HangingNeatlogs):
        def flush(self, **k):
            flushed.append(True)
            return True

        def get_delivery_diagnostics(self):
            return {}

        def shutdown(self, **k):
            pass
    t = fake_tracer(Recording(), tmp_path, [])
    with t.root("notbydefault.check", {}, finish=lambda: {"nbd.stopped_early": True}):
        t.flush()                      # an early return inside the root
        assert flushed == []           # not yet: the root is still open
    assert flushed == [True]
    assert t.journal[-1]["name"] == "root_summary" and t.journal[-1]["nbd.stopped_early"] is True
