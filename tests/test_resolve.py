"""`notbydefault resolve`: each step, the approval/commit rules and the failure paths.

Projects are real git repos in tmp_path; the fit check runs on the fake providers, so its
response cache is what resolve's verify step replays (no network, no spend)."""
import json
import subprocess

import pytest

from conftest import FakeEntireRunner, FakeTransport, demo_graph_responses, demo_project, \
    make_client
from notbydefault import affected as affected_mod
from notbydefault import resolve as resolve_mod
from notbydefault.entire import Entire
from notbydefault.fit.run import cache_dir
from notbydefault.fit.tracing import FOCUS_AREAS, Tracer
from notbydefault.graph import Graph
from notbydefault.resolve import (EditError, Resolver, ResolveOptions, Site, call_contract,
                                  edit_file)
from test_fit_check import make_project, run

TARGET = "gemini:gemini-3.5-flash-lite"
BRANCH = "nbd/{did}-switch-to-gemini-3-5-flash-lite"

# the client is created inside the call site, so switching provider stays local to it
APP = '''import os
from openai import OpenAI

GROQ_BASE_URL = "https://groq.test/v1"
_client = None


def summarize_notes(texts):
    global _client
    if _client is None:  # created lazily so the app starts without FAKE_GROQ_KEY
        _client = OpenAI(base_url=GROQ_BASE_URL, api_key=os.environ["FAKE_GROQ_KEY"])
    response = _client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "Summarize the user's notes in a few concise sentences."},
            {"role": "user", "content": "\\n".join(f"- {t}" for t in texts)},
        ],
    )
    return response.choices[0].message.content
'''
PASSING_TEST = "def test_imports():\n    import ast\n    ast.parse(open('app.py').read())\n"


class Concise(FakeTransport):
    """gemini-3.5-flash-lite answers well without a verifier, so the verdict is a plain switch."""

    def respond(self, model, messages):
        kind, text = super().respond(model, messages)
        if kind == "candidate" and model == "gemini-3.5-flash-lite":
            text = text.split(" This is additional commentary.")[0]
        return kind, text


def git(p, *args):
    return subprocess.run(["git", *args], cwd=p, capture_output=True, text=True,
                          check=True).stdout.rstrip()


def make_repo(path, app=APP, tests=PASSING_TEST, transport=None):
    """A git repo on main with a fit-checked decision; returns (path, decision id, client)."""
    p, did = make_project(path)
    (p / "app.py").write_text(app)
    if tests:
        (p / "tests").mkdir()
        (p / "tests" / "test_app.py").write_text(tests)
    client = make_client(cache_dir=cache_dir(p), transport=transport or Concise())
    code, report = run(p, did, client=client)
    assert code == 0, report
    (p / ".gitignore").write_text(".notbydefault/\n__pycache__/\n.pytest_cache/\n")
    git(p, "init", "-q", "-b", "main")
    git(p, "config", "user.email", "t@example.com")
    git(p, "config", "user.name", "t")
    git(p, "add", "-A")
    git(p, "commit", "-q", "-m", "init")
    return p, did, client


@pytest.fixture
def repo(tmp_path):
    return make_repo(tmp_path)


def resolve(p, did, client, answers=(), tracer=None, **opts):
    out, asked = [], list(answers)
    r = Resolver(p, did, ResolveOptions(**opts), tracer=tracer or Tracer(),
                 ask=lambda q: asked.pop(0) if asked else "", out=out.append, client=client)
    return r.run(), "\n".join(out), r


def snapshot(p):
    return {f: (p / f).read_bytes() for f in git(p, "ls-files").splitlines()}


# --- happy paths ----------------------------------------------------------------------

def test_dry_run_shows_the_diff_and_changes_nothing(repo):
    p, did, client = repo
    before, graph_before = snapshot(p), (p / ".notbydefault" / "graph.db").read_bytes()
    code, out, _ = resolve(p, did, client)
    assert code == 0, out
    assert "Fit check: reusing" in out and f"switch groq:openai/gpt-oss-120b -> {TARGET}" in out
    assert '+        model="gemini-3.5-flash-lite",' in out and "+GEMINI_BASE_URL" in out
    assert "Dry run: nothing changed" in out
    assert snapshot(p) == before and git(p, "status", "--porcelain") == ""
    assert git(p, "branch", "--show-current") == "main" and "nbd/" not in git(p, "branch")
    assert (p / ".notbydefault" / "graph.db").read_bytes() == graph_before
    assert not (p / ".notbydefault" / "resolve").exists()


def test_apply_edits_tests_verifies_records_and_leaves_it_uncommitted(repo):
    p, did, client = repo
    main_sha = git(p, "rev-parse", "main")
    requests = client.transport.calls[:]
    code, out, r = resolve(p, did, client, apply=True, yes=True)
    assert code == 0, out
    branch = BRANCH.format(did=did)
    assert git(p, "branch", "--show-current") == branch
    assert git(p, "rev-parse", "main") == main_sha == git(p, "rev-parse", "HEAD")  # no commit
    app = (p / "app.py").read_text()
    assert 'model="gemini-3.5-flash-lite"' in app and "openai/gpt-oss-120b" not in app
    assert 'GEMINI_BASE_URL = "https://gemini.test/openai"' in app and "GROQ_BASE_URL" not in app
    assert 'os.environ["FAKE_GEMINI_KEY"]' in app and "without FAKE_GEMINI_KEY" in app
    assert call_contract(p / "app.py", None)["model"] == ["gemini-3.5-flash-lite"]
    assert "Test: passed" in out
    assert "(response cache (no requests)" in out
    assert client.transport.calls == requests            # verify replayed, never requested
    assert "Left uncommitted" in out and "Nothing was pushed" in out

    record = (p / r._rel(r.record_path)).read_text(encoding="utf-8")
    assert "### Resolution" in record and branch in record
    with Graph.for_project(p) as g:
        res = g.get_node(did)["data"]["resolution"]
    assert res["to"] == TARGET and res["tests"] == "passed"
    pr = r.pr_path.read_text(encoding="utf-8")
    for part in ("## Evidence", "| latency per call (s) |", "| verifiers on the new code |",
                 "## Traces", "## Checkpoints", "FAKE_GEMINI_KEY", "| verify | verification | 1 | ok |"):
        assert part in pr, part
    changed = {l[3:] for l in git(p, "status", "--porcelain").splitlines()}
    assert changed == {"app.py", r._rel(r.record_path)}


def test_commits_only_when_the_user_says_yes_and_never_on_main(repo):
    p, did, client = repo
    main_sha = git(p, "rev-parse", "main")
    code, out, _ = resolve(p, did, client, answers=["y"], apply=True, yes=True)
    assert code == 0, out
    branch = BRANCH.format(did=did)
    assert git(p, "rev-parse", "main") == main_sha
    assert git(p, "rev-list", "--count", f"main..{branch}") == "1"
    assert git(p, "status", "--porcelain") == ""
    assert git(p, "log", "-1", "--format=%s", branch) == f"Resolve #{did}: switch to {TARGET}"
    assert "Not pushed" in out


def test_approval_is_asked_and_no_means_nothing_changes(repo):
    p, did, client = repo
    before = snapshot(p)
    code, out, _ = resolve(p, did, client, answers=["n"], apply=True)
    assert code == 1 and "Not approved; nothing changed." in out
    assert snapshot(p) == before and git(p, "branch", "--show-current") == "main"


def test_every_step_is_a_span_with_focus_area_and_resolve_step(repo):
    from test_fit_tracing import MemoryTracer
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    p, did, client = repo
    exporter, provider = InMemorySpanExporter(), TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    import neatlogs
    try:
        code, out, _ = resolve(p, did, client, apply=True, yes=True,
                               tracer=MemoryTracer(provider))
    finally:
        neatlogs.shutdown(timeout_millis=1000)
    assert code == 0, out
    spans = exporter.get_finished_spans()
    assert len({s.context.trace_id for s in spans}) == 1
    steps = {s.attributes["nbd.resolve_step"]: s for s in spans
             if "nbd.resolve_step" in s.attributes}
    assert set(steps) == set(resolve_mod.STEPS)
    for name, s in steps.items():
        assert s.name == name
        assert s.attributes["focus_area"] == resolve_mod.STEPS[name][0] in FOCUS_AREAS
    root = next(s for s in spans if s.name == "notbydefault.resolve")
    assert root.attributes["nbd.resolve_outcome"] == "resolved"
    assert any(s.name == "notbydefault.verify" for s in spans)   # the verifiers' own spans


# --- investigate ----------------------------------------------------------------------

def test_investigate_leaves_another_decisions_call_site_alone(tmp_path, monkeypatch):
    """Entire Graph finds the model in actions.py and app.py; app.py belongs to another
    decision whose fit check rejected the target, so only actions.py is changed."""
    monkeypatch.setattr(affected_mod, "repo_root", lambda p: tmp_path)
    p = demo_project(tmp_path)
    with Graph.for_project(p) as g:
        mine = g.add_node("decision", "openai/gpt-oss-120b", {
            "kind": "model", "category": "ai-model", "file": "actions.py", "files": ["actions.py"]})
        other = g.add_node("decision", "openai/gpt-oss-120b", {
            "kind": "model", "category": "ai-model", "file": "app.py", "files": ["app.py"],
            "verdict": "switch"})
        alt = g.add_node("alternative", TARGET, {"reason": "costs more"})
        g.add_edge(other, alt, "rejects")
        d = g.get_node(mine)
    r = Resolver(p, mine, ResolveOptions(), tracer=Tracer(), out=lambda _: None,
                 entire=Entire(exe="entire", runner=FakeEntireRunner(demo_graph_responses())))
    r.decision = d
    r.step_investigate(None)
    assert [s.describe() for s in r.inv["sites"]] == ["actions.py:28 extract_action_items"]
    (site, owner), = r.inv["excluded"]
    assert site.symbol == "summarize_notes" and owner["id"] == other
    assert TARGET in owner["rejects"]
    assert {s.symbol for s in r.inv["callers"]} == {"Handler.do_POST"}


def test_investigate_retry_uses_the_stored_analysis(tmp_path):
    p = demo_project(tmp_path)
    with Graph.for_project(p) as g:
        did = g.add_node("decision", "openai/gpt-oss-120b", {
            "kind": "model", "file": "actions.py", "files": ["actions.py"],
            "affected_code": [{"path": "actions.py", "symbol": "extract_action_items",
                               "kind": "function", "line": 28, "via": "uses it", "depth": 0}]})
        d = g.get_node(did)
    r = Resolver(p, did, ResolveOptions(), tracer=Tracer(), out=lambda _: None)
    r.decision = d
    r.step_investigate("RuntimeError: graph read timed out")
    assert r.inv["source"] == "stored analysis (retry)"
    assert [s.symbol for s in r.inv["sites"]] == ["extract_action_items"]


# --- edits ----------------------------------------------------------------------------

PROVIDERS = {"groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY"},
             "gemini": {"base_url": "https://gemini.example/openai", "key_env": "GEMINI_API_KEY"}}
DEMO_ACTIONS = (__import__("conftest").FIXTURES / "demo_graph" / "actions.py")


def test_edit_switches_model_provider_url_and_key_inside_the_site():
    text = DEMO_ACTIONS.read_text(encoding="utf-8")
    after, notes = edit_file(text, "actions.py", [Site("actions.py", "extract_action_items", 28, "")],
                             ("groq", "openai/gpt-oss-120b"), ("gemini", "gemini-3.1-flash-lite"),
                             PROVIDERS)
    assert 'model="gemini-3.1-flash-lite"' in after and "gpt-oss-120b" not in after
    assert 'GEMINI_BASE_URL = "https://gemini.example/openai"' in after
    assert "GROQ" not in after
    assert notes == ["GROQ_BASE_URL replaced by GEMINI_BASE_URL (no other use in actions.py)"]
    assert len(after.splitlines()) == len(text.splitlines())


def test_edit_keeps_a_shared_constant_and_crlf_line_endings():
    text = APP.replace("    return response", "    print(GROQ_BASE_URL)\n    return response")
    text = text + "\n\ndef other():\n    return GROQ_BASE_URL\n"
    crlf = text.replace("\n", "\r\n")
    after, notes = edit_file(crlf, "app.py", [Site("app.py", "summarize_notes", 8, "")],
                             ("groq", "openai/gpt-oss-120b"), ("gemini", "gemini-3.1-flash-lite"),
                             PROVIDERS | {"groq": {"base_url": "https://groq.test/v1",
                                                  "key_env": "FAKE_GROQ_KEY"}})
    assert "\n" not in after.replace("\r\n", "")
    assert 'GROQ_BASE_URL = "https://groq.test/v1"\r\nGEMINI_BASE_URL = ' in after
    assert "return GROQ_BASE_URL" in after and "print(GEMINI_BASE_URL)" in after
    assert notes and "still used elsewhere" in notes[0]


def test_same_provider_changes_only_the_model():
    text = DEMO_ACTIONS.read_text(encoding="utf-8")
    after, _ = edit_file(text, "actions.py", [Site("actions.py", "extract_action_items", 28, "")],
                         ("groq", "openai/gpt-oss-120b"), ("groq", "openai/gpt-oss-20b"), PROVIDERS)
    assert [l for l in after.splitlines() if l not in text.splitlines()] == \
        ['        model="openai/gpt-oss-20b",']


def test_a_shared_client_cannot_be_switched_from_one_site():
    module_client = 'import os\nfrom openai import OpenAI\n\nclient = OpenAI(base_url="https://' \
        'api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"])\n\n\ndef f():\n' \
        '    return client.chat.completions.create(model="openai/gpt-oss-120b", messages=[])\n'
    with pytest.raises(EditError, match="client is not created here"):
        edit_file(module_client, "app.py", [Site("app.py", "f", 7, "")],
                  ("groq", "openai/gpt-oss-120b"), ("gemini", "gemini-3.1-flash-lite"), PROVIDERS)


# --- stops and failure paths ----------------------------------------------------------

def test_verdict_with_a_verifier_loop_is_not_applied(tmp_path):
    p, did, client = make_repo(tmp_path, transport=FakeTransport())   # verified setup wins
    before = snapshot(p)
    code, out, _ = resolve(p, did, client, apply=True, yes=True)
    assert code == 2 and "with a verifier and retries" in out
    assert snapshot(p) == before and git(p, "branch", "--show-current") == "main"


def test_stale_fit_check_is_not_reused_in_a_dry_run(repo):
    p, did, client = repo
    (p / "app.py").write_text(APP.replace("a few concise", "two"))
    git(p, "commit", "-qam", "change the prompt")
    code, out, _ = resolve(p, did, client)
    assert code == 2 and "app.py changed since the fit check" in out and "--apply" in out


def test_dirty_tree_stops_before_branching(repo):
    p, did, client = repo
    (p / "app.py").write_text(APP + "\n# local edit\n")
    code, out, _ = resolve(p, did, client, apply=True, yes=True)
    assert code == 1 and "Uncommitted changes" in out and "Nothing changed" in out
    assert git(p, "branch", "--show-current") == "main" and "nbd/" not in git(p, "branch")
    assert (p / "app.py").read_text().endswith("# local edit\n")


def test_a_step_that_fails_once_is_retried_with_its_error(repo, monkeypatch):
    p, did, client = repo
    real, seen = Resolver.step_test, []

    def flaky(self, error):
        seen.append(error)
        if error is None:
            raise RuntimeError("pytest crashed (flaky)")
        return real(self, error)
    monkeypatch.setattr(Resolver, "step_test", flaky)
    code, out, r = resolve(p, did, client, apply=True, yes=True)
    assert code == 0, out
    assert seen == [None, "RuntimeError: pytest crashed (flaky)"]
    assert "test failed: RuntimeError: pytest crashed (flaky). Retrying once." in out
    assert {"step": "test", "focus_area": "verification", "attempts": 2, "status": "ok"} \
        in r.state["steps"]
    assert any(e["name"] == "retry_step" and e["focus_area"] == "recovery"
               for e in r.tracer.journal)


def test_failing_project_tests_roll_everything_back(tmp_path):
    failing = "def test_still_groq():\n    assert 'gpt-oss-120b' in open('app.py').read()\n"
    p, did, client = make_repo(tmp_path, tests=failing)
    before = snapshot(p)
    with Graph.for_project(p) as g:
        node_before = g.get_node(did)["data"]
    code, out, r = resolve(p, did, client, apply=True, yes=True)
    assert code == 3
    assert "Stopped at step 'test' after one retry" in out and "exited 1" in out
    assert snapshot(p) == before and git(p, "status", "--porcelain") == ""
    branch = BRANCH.format(did=did)
    assert git(p, "branch", "--show-current") == "main"
    assert git(p, "rev-parse", branch) == git(p, "rev-parse", "main")    # clean, no commits
    assert f"branch {branch} is clean" in out
    patch = (p / ".notbydefault" / "resolve" / f"{did}-failed.patch").read_text()
    assert '+        model="gemini-3.5-flash-lite",' in patch
    with Graph.for_project(p) as g:
        assert g.get_node(did)["data"] == node_before
    assert [s["status"].split(":")[0] for s in r.state["steps"]][-1] == "failed"


def test_failing_verify_after_the_record_restores_the_record_too(repo, monkeypatch):
    p, did, client = repo
    before = snapshot(p)

    def broken_record(self, error):
        Resolver._real_record(self, error)
        raise OSError("disk full")
    monkeypatch.setattr(Resolver, "_real_record", Resolver.step_record, raising=False)
    monkeypatch.setattr(Resolver, "step_record", broken_record)
    code, out, _ = resolve(p, did, client, apply=True, yes=True)
    assert code == 3 and "Stopped at step 'record'" in out and "disk full" in out
    assert snapshot(p) == before and git(p, "status", "--porcelain") == ""
    assert not list((p / ".notbydefault" / "resolve").glob("*-pr.md"))


def test_edit_that_cannot_be_planned_stops_without_a_branch(tmp_path):
    module_client = APP.replace(
        '_client = None', '_client = OpenAI(base_url=GROQ_BASE_URL, '
                          'api_key=os.environ["FAKE_GROQ_KEY"])').replace(
        '    if _client is None:  # created lazily so the app starts without FAKE_GROQ_KEY\n'
        '        _client = OpenAI(base_url=GROQ_BASE_URL, api_key=os.environ["FAKE_GROQ_KEY"])\n',
        '')
    p, did, client = make_repo(tmp_path, app=module_client)
    with Graph.for_project(p) as g:   # function-level sites, as Entire Graph reports them
        g.update_data(did, affected_code=[{"path": "app.py", "symbol": "summarize_notes",
                                           "kind": "function", "line": 8, "via": "uses it",
                                           "depth": 0}])
    code, out, _ = resolve(p, did, client, apply=True, yes=True)
    assert code == 3 and "Stopped at step 'recommend'" in out and "nothing was changed" in out
    assert git(p, "branch", "--show-current") == "main" and "nbd/" not in git(p, "branch")


def test_cli_resolve_dry_run(repo, monkeypatch):
    from notbydefault.cli import main as cli
    p, did, client = repo
    monkeypatch.setattr(resolve_mod.Resolver, "__init__", _with_client(client))
    assert cli(["resolve", str(did), "--project", str(p)]) == 0
    assert git(p, "status", "--porcelain") == ""


def _with_client(client):
    real = resolve_mod.Resolver.__init__

    def init(self, *a, **kw):
        real(self, *a, **kw)
        self.client, self.tracer = client, Tracer()
    return init
