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
    import neatlogs
    neatlogs.shutdown(timeout_millis=1000)
