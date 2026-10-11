"""`notbydefault ui`: every page, the live poll, read-only, CSRF, escaping and the actions.

The project is a real fit-checked git repo built on the fake providers (test_resolve.make_repo),
copied per test that writes, so nothing here touches demo-app or the network."""
import json
import shutil
import subprocess

import pytest
from markupsafe import escape

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from conftest import make_client  # noqa: E402
from notbydefault.fit.report import table_rows  # noqa: E402
from notbydefault.graph import Graph  # noqa: E402
from notbydefault.ui import data  # noqa: E402
from notbydefault.ui.app import create_app  # noqa: E402
from test_resolve import Concise, make_repo  # noqa: E402

BASE = "http://127.0.0.1:8765"
XSS = '<script>alert("pwned")</script><img src=x onerror=alert(1)>'


@pytest.fixture(scope="module")
def base_repo(tmp_path_factory):
    p, did, _client = make_repo(tmp_path_factory.mktemp("ui") / "proj")
    with Graph.for_project(p) as g:
        db = g.add_node("decision", "MongoDB", {"kind": "service", "category": "database",
                                                "chosen": "MongoDB", "status": "unreviewed",
                                                "tool_name": "Bash", "evidence": "pymongo",
                                                "file": "requirements.txt"})
    return p, did, db


@pytest.fixture
def repo(base_repo, tmp_path):
    """A private copy for tests that write."""
    src, did, db = base_repo
    dst = tmp_path / "proj"
    shutil.copytree(src, dst)
    return dst, did, db


def client_for(project, read_only=False, transport=None):
    from notbydefault.fit.run import cache_dir
    factory = (lambda: make_client(cache_dir=cache_dir(project), transport=transport or Concise()))
    app = create_app(project, read_only=read_only, client_factory=factory)
    return TestClient(app, base_url=BASE)


def post(c, url, csrf=True, origin=BASE, **form):
    token = c.app.state.cfg.csrf if csrf is True else (csrf or "")
    headers = {"Origin": origin} if origin else {}
    return c.post(url, data=dict(form, csrf_token=token), headers=headers)


def status(project, nid):
    with Graph.for_project(project) as g:
        return g.get_node(nid)["data"].get("status")


# --- pages ----------------------------------------------------------------------------

def test_every_page_renders(base_repo):
    p, did, db = base_repo
    c = client_for(p)
    with Graph.for_project(p) as g:
        goal = g.nodes("goal")[0]["id"]
        record = g.get_node(did)["data"]["decision_record"]
        results = g.get_node(did)["data"]["results"]
    for url in ["/", "/decisions", "/decisions?status=unreviewed", "/decisions?kind=model",
                f"/decisions/{did}", f"/decisions/{db}", f"/nodes/{goal}", "/graph",
                f"/graph?focus={did}", "/timeline", "/evals", "/costs", f"/files/{record}",
                f"/files/{results}", f"/fragments/node/{did}", "/fragments/overview"]:
        r = c.get(url)
        assert r.status_code == 200, url
        assert "text/html" in r.headers["content-type"]
        assert "default-src 'self'" in r.headers["content-security-policy"]
    assert c.get("/costs.csv").text.startswith("decision_id,")
    assert c.get(f"/nodes/{did}", follow_redirects=False).status_code == 303
    assert c.get("/decisions/9999").status_code == 404


def test_overview_tiles_feed_and_per_decision_savings(base_repo):
    p, did, db = base_repo
    html = client_for(p).get("/").text
    assert "decisions caught" in html and "no decision without a why" in html
    assert f"Caught #{db} MongoDB" in html and f"Fit check #{did}" in html
    assert "/mo</span>" in html and "users ×" in html     # savings at its own volume
    assert "Add a requirement" in html


def test_decision_page_shows_the_cli_fit_table_and_sources(base_repo):
    p, did, _ = base_repo
    with Graph.for_project(p) as g:
        d = g.get_node(did)
    res = json.loads((p / d["data"]["results"]).read_text(encoding="utf-8"))
    html = client_for(p).get(f"/decisions/{did}").text
    from notbydefault.fit.stats import SetupStats
    stats = {(s["model"], s["setup"]): SetupStats(**s) for s in res["stats"].values()}
    from types import SimpleNamespace
    headers, rows = table_rows(stats, SimpleNamespace(chosen=res["config"]["chosen"],
                                                      judges=[res["config"]["judge"]]))
    for h in headers:
        assert f">{h}</th>" in html
    for row in rows:
        for cell in row:
            assert str(escape(cell)) in html
    assert d["data"]["decision_record"] in html and d["data"]["results"] in html
    assert "Rejected alternatives" in html and "Failure bank" in html


def test_files_outside_the_project_or_secret_are_refused(repo):
    p, _, _ = repo
    (p / ".env").write_text("GROQ_API_KEY=secret-value")
    (p / "secrets.json").write_text('{"k": "secret-value"}')
    (p.parent / "outside.md").write_text("outside")
    c = client_for(p)
    for url in ["/files/.env", "/files/secrets.json", "/files/..%2Foutside.md",
                "/files/.notbydefault/graph.db", "/files/.git/config"]:
        r = c.get(url)
        assert r.status_code == 404 and "secret-value" not in r.text, url
    assert data.safe_path(p, "../outside.md") is None


def test_wrong_host_header_is_refused(base_repo):
    p, _, _ = base_repo
    c = TestClient(create_app(p), base_url="http://evil.example")
    assert c.get("/").status_code == 400


def test_project_without_a_graph_shows_empty_states_and_creates_nothing(tmp_path):
    c = client_for(tmp_path)
    for url in ["/", "/decisions", "/graph", "/timeline", "/costs"]:
        assert c.get(url).status_code == 200, url
    assert "No activity yet" in c.get("/").text
    assert c.get("/api/graph").json()["nodes"] == []
    assert not (tmp_path / ".notbydefault").exists()


# --- live -----------------------------------------------------------------------------

def test_live_poll_returns_new_nodes_and_the_new_feed_item(repo):
    p, _, _ = repo
    c = client_for(p)
    g0 = c.get("/api/graph").json()
    assert c.get(f"/api/graph?since={g0['max_id']}").json()["nodes"] == []
    v0 = c.get("/api/version").json()["version"]
    assert c.get(f"/fragments/overview?v={v0}").status_code == 204     # nothing changed
    with Graph.for_project(p) as g:      # the hook catches a decision in another window
        nid = g.add_node("decision", "Redis", {"kind": "service", "category": "cache",
                                               "status": "unreviewed", "tool_name": "Bash"})
    new = c.get(f"/api/graph?since={g0['max_id']}").json()
    assert [n["data"]["nid"] for n in new["nodes"]] == [nid]
    assert new["nodes"][0]["data"]["status"] == "unreviewed" and new["version"] != v0
    frag = c.get(f"/fragments/overview?v={v0}")
    assert frag.status_code == 200 and f"Caught #{nid} Redis" in frag.text


def test_reads_never_hold_a_lock(repo):
    p, did, _ = repo
    c = client_for(p)
    import sqlite3
    w = sqlite3.connect(p / ".notbydefault" / "graph.db", timeout=0)    # fail fast if locked
    try:
        for url in ["/", "/graph", f"/decisions/{did}", "/api/graph"]:
            c.get(url)
            w.execute("BEGIN IMMEDIATE")
            w.execute("UPDATE nodes SET label = label WHERE id = ?", (did,))
            w.commit()
    finally:
        w.close()


# --- safety ---------------------------------------------------------------------------

def test_a_malicious_reason_and_label_are_escaped(repo):
    p, did, _ = repo
    with Graph.for_project(p) as g:
        g.update_data(did, reason=XSS)
        g.add_node("decision", XSS, {"kind": "service", "category": XSS, "status": "unreviewed"})
    c = client_for(p)
    for url in ["/", "/decisions", f"/decisions/{did}", "/timeline", "/graph"]:
        html = c.get(url).text
        assert "<script>alert" not in html and "<img src=x" not in html, url
    assert "&lt;script&gt;alert(&#34;pwned&#34;)&lt;/script&gt;" in c.get(f"/decisions/{did}").text


def test_read_only_hides_and_refuses_every_write(repo):
    p, did, db = repo
    c = client_for(p, read_only=True)
    page = c.get(f"/decisions/{did}").text
    assert "Mark reviewed" not in page and "Resolve: dry run" not in page
    assert "Add a requirement" not in c.get("/").text and "read-only" in page
    before = (p / ".notbydefault" / "graph.db").read_bytes()
    for url, form in [(f"/decisions/{db}/review", {"reason": "fine", "mark": "reviewed"}),
                      ("/requirements", {"text": "EU only"}),
                      (f"/decisions/{did}/resolve-dry-run", {})]:
        assert post(c, url, **form).status_code == 403, url
    assert (p / ".notbydefault" / "graph.db").read_bytes() == before
    assert not (p / ".notbydefault" / data.ACTIVITY_LOG).exists()


@pytest.mark.parametrize("csrf,origin", [(False, BASE), ("wrong-token", BASE),
                                         (True, "http://evil.example"), (True, None)])
def test_writes_without_token_or_from_another_origin_are_refused(repo, csrf, origin):
    p, _, db = repo
    c = client_for(p)
    r = post(c, f"/decisions/{db}/review", csrf=csrf, origin=origin, reason="x", mark="reviewed")
    assert r.status_code == 403
    assert status(p, db) == "unreviewed"


# --- actions --------------------------------------------------------------------------

def test_mark_reviewed_with_a_reason_updates_graph_record_and_feed(repo):
    p, _, db = repo
    c = client_for(p)
    r = post(c, f"/decisions/{db}/review", reason="we already run Mongo for billing",
             mark="reviewed")
    assert r.status_code == 200 and f"#{db} marked reviewed" in r.text
    with Graph.for_project(p) as g:
        d = g.get_node(db)["data"]
    assert d["status"] == "reviewed" and d["reason"] == "we already run Mongo for billing"
    record = (p / d["decision_record"]).read_text(encoding="utf-8")
    assert "we already run Mongo for billing (reason given by a person" in record
    page = c.get(f"/decisions/{db}").text
    assert "has a why" in page and "chosen by default" not in page
    assert f"#{db} marked reviewed: MongoDB" in c.get("/").text


def test_reason_only_keeps_the_status(repo):
    p, _, db = repo
    c = client_for(p)
    assert post(c, f"/decisions/{db}/review", reason="", mark="reason").status_code == 400
    assert post(c, f"/decisions/{db}/review", reason="cheap", mark="reason").status_code == 200
    assert status(p, db) == "unreviewed"


def test_add_requirement_runs_require_and_shows_what_changed(repo):
    p, did, db = repo
    c = client_for(p)
    r = post(c, "/requirements", text="Customer data must stay in the EU")
    assert r.status_code == 200, r.text
    assert status(p, db) in ("stale", "needs_review")
    assert f"#{db} MongoDB" in r.text and "What changed" in r.text
    for nid in (db, did):       # every changed decision is listed with its new status
        if status(p, nid) != "reviewed":
            assert f'href="/decisions/{nid}" class="mono"' in r.text
    with Graph.for_project(p) as g:
        rid = g.find_node("requirement", "Customer data must stay in the EU")["id"]
    assert f"/graph?req={rid}" in r.text
    feed = c.get("/").text
    assert f"Requirement #{rid}: Customer data must stay in the EU" in feed
    assert "from the UI" in feed
    node = next(n for n in c.get("/api/graph").json()["nodes"] if n["data"]["nid"] == rid)
    assert db in node["data"]["impacts"]


def test_resolve_dry_run_shows_plan_and_diff_and_changes_nothing(repo):
    p, did, _ = repo
    c = client_for(p)
    git = lambda *a: subprocess.run(["git", *a], cwd=p, capture_output=True, text=True).stdout
    before = git("status", "--porcelain")
    r = post(c, f"/decisions/{did}/resolve-dry-run")
    assert r.status_code == 200, r.text
    assert "dry run (nothing changed)" in r.text and 'class="code diff"' in r.text
    assert "gemini-3.5-flash-lite" in r.text
    assert git("status", "--porcelain") == before and git("branch", "--show-current").strip() == "main"
    assert f"Resolve dry run for #{did}" in c.get("/").text


def test_claim_check_watched_assumptions_and_debt_are_shown(repo):
    from notbydefault import assumptions, claims
    from notbydefault import watch as watch_mod
    p, did, db = repo
    with Graph.for_project(p) as g:
        g.update_data(did, agent_reason="It is the cheapest. Data stays in the EU. Everyone uses it.")
        c = claims.evaluate(p, g, g.get_node(did), make_client(), split=False)
        claims.store(p, g, did, c)
        assumptions.record_snapshot(p, g, did)
        g.add_node("requirement", "EU only", {"constraints": {"data_residency": "EU"}})
    watch_mod.watch(p, client=make_client(), out=lambda _: None)
    cl = client_for(p)
    page = cl.get(f"/decisions/{did}").text
    assert "Claim check" in page and "claim v-false" in page and "not evidence for your project" in page
    assert "Watched assumptions" in page and "broken by requirement EU only" in page
    overview = cl.get("/").text
    assert "decision debt" in overview and "assumptions expired" in overview
    assert "Decision debt:" in overview and f"decision #{db}: mark reviewed with a reason" in overview


def test_claim_checks_and_watch_reopens_appear_in_the_feed(repo):
    from notbydefault import assumptions, claims
    from notbydefault import watch as watch_mod
    p, did, db = repo
    with Graph.for_project(p) as g:
        g.update_data(db, agent_reason="Popular with everyone. Data stays in the EU.")
        claims.store(p, g, db, claims.evaluate(p, g, g.get_node(db), make_client(), split=False))
        assumptions.record_snapshot(p, g, did)
        g.add_node("requirement", "EU only", {"constraints": {"data_residency": "EU"}})
    watch_mod.watch(p, client=make_client(), out=lambda _: None)
    html = client_for(p).get("/timeline").text
    assert f"Claim check #{db} MongoDB:" in html and f"Re-opened #{did}" in html


def test_resolution_receipt_shows_generated_test_counts(repo):
    p, did, _ = repo
    with Graph.for_project(p) as g:
        g.update_data(did, resolution={
            "at": "2026-10-11T12:00:00+00:00", "branch": "nbd/1-switch-to-x", "from": "groq:a",
            "to": "gemini:b", "tests": "passed", "verifiers": "8/8 production, 8/8 vs reference",
            "trace_id": None, "generated_tests": {
                "path": "tests/test_nbd_1_generated.py", "cases": 8, "from_failure_bank": 3,
                "sites": ["app.py:9 summarize_notes"], "before": "0/8 passed",
                "after": "8/8 passed"}})
    html = client_for(p).get(f"/decisions/{did}").text
    assert "tests/test_nbd_1_generated.py" in html
    assert "8 case(s), 3 from the failure bank" in html
    assert "before the edit 0/8 passed → after 8/8 passed" in html
