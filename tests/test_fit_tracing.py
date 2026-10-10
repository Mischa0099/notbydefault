"""The real neatlogs tracer, exporting to memory: one trace per check, attributes on spans."""
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from conftest import FakeClient
from notbydefault.fit import tracing
from notbydefault.fit.run import CheckOptions, check_model_decision
from test_fit_check import project  # noqa: F401  (fixture)


class MemoryTracer(tracing.NeatlogsTracer):
    def __init__(self, provider):
        import neatlogs
        from opentelemetry import context, trace
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
                                client=FakeClient(), tracer=MemoryTracer(provider),
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
    assert judge["llm.model_name"] == "anthropic/claude-opus-5.5"
    assert any(k.startswith("notbydefault.score.") for k in judge)
    root = next(s for s in spans if s.name == "notbydefault.check").attributes
    assert root["notbydefault.verdict"] == "switch"
    import neatlogs
    neatlogs.shutdown(timeout_millis=1000)
