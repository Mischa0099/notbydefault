"""`notbydefault scan`: decision debt from the code and the graph, with the next command."""
from conftest import CATALOG, make_client
from notbydefault import claims, scan
from notbydefault.graph import Graph
from notbydefault.watch import watch
from test_claims_watch import REASON, checked, node  # noqa: F401  (fixture)
from test_fit_check import make_project


def test_debt_counts_unchecked_false_claims_and_expired_assumptions(checked):  # noqa: F811
    p, did, client, _ = checked
    with Graph.for_project(p) as g:
        db = g.add_node("decision", "MongoDB", {"kind": "service", "category": "database",
                                                "status": "unreviewed"})
        g.add_node("requirement", "EU only", {"constraints": {"data_residency": "EU"}})
        r = scan.compute(p, g, client.catalog)
    false_claims = [i for i in node(p, did)["data"]["claims"]["items"] if i["verdict"] == "false"]
    assert [d["id"] for d in r["unchecked"]] == [db] and r["unrecorded"] == []
    assert len(r["claims_false"]) == len(false_claims) >= 1
    assert [d["id"] for d, _ in r["expired"]] == [did]
    assert r["score"] == 1 + len(false_claims) + 1
    nexts = {(row["count"], row["id"]): row["next"] for row in r["rows"]}
    assert nexts[("unchecked", db)] == f"notbydefault ui (decision #{db}: mark reviewed with a reason)"
    assert nexts[("assumption expired", did)] == "notbydefault watch"     # not re-opened yet
    assert nexts[("claim false", did)] in (f"notbydefault resolve {did}", f"notbydefault check {did}")


def test_expired_after_watch_points_at_a_new_check(checked):  # noqa: F811
    p, did, client, _ = checked
    with Graph.for_project(p) as g:
        g.add_node("requirement", "EU only", {"constraints": {"data_residency": "EU"}})
    watch(p, client=client, out=lambda _: None)
    with Graph.for_project(p) as g:
        r = scan.compute(p, g, client.catalog)
    [row] = [x for x in r["rows"] if x["count"] == "assumption expired"]
    assert row["next"] == f"notbydefault check {did}"
    assert "'data region: any' broken by requirement 'EU only'" in row["why"]


def test_audit_finds_choices_in_the_code_that_no_decision_records(tmp_path):
    p, did = make_project(tmp_path)
    (p / "requirements.txt").write_text("pymongo==4.8\nrequests\n")
    (p / "digest.py").write_text('MODEL = "gpt-4o-mini"\n')
    with Graph.for_project(p) as g:
        r = scan.compute(p, g, CATALOG)
    found = {(f.kind, f.chosen, f.file) for f in r["unrecorded"]}
    assert ("service", "MongoDB", "requirements.txt") in found
    assert ("model", "gpt-4o-mini", "digest.py") in found
    assert not any(f.chosen == "openai/gpt-oss-120b" for f in r["unrecorded"])  # recorded as #did
    assert {row["next"] for row in r["rows"] if row["id"] is None} == {"notbydefault scan --record"}


def test_scan_prints_the_score_and_table_and_record_adds_the_findings(tmp_path):
    p, did = make_project(tmp_path)
    (p / "digest.py").write_text('MODEL = "gpt-4o-mini"\n')
    with Graph.for_project(p) as g:
        g.update_data(did, agent_reason="Popular with everyone.")
        claims.store(p, g, did, claims.evaluate(p, g, g.get_node(did), make_client(), split=False))
    out = []
    assert scan.scan(p, out=out.append) == 0
    assert out[0] == scan.DEBT_LINE
    assert out[1] == ("Decision debt: 2  (2 unchecked, 1 of them not in the graph / 0 claim(s) "
                      "false / 0 with expired assumptions)")
    table = out[3].splitlines()
    assert table[0].split() == ["count", "#", "decision", "next", "command", "why"]
    assert any(l.startswith("unchecked") and f"notbydefault check {did}" in l for l in table)
    out = []
    scan.scan(p, record=True, out=out.append)
    assert out[0].startswith("Recorded 1 choice(s) found in the code as unreviewed decisions")
    with Graph.for_project(p) as g:
        [new] = [d for d in g.nodes("decision") if d["label"] == "gpt-4o-mini"]
        assert new["data"]["source"] == "scan" and new["data"]["status"] == "unreviewed"
        assert {n["label"] for n in g.neighbors(new["id"], ("affects",))} == {"digest.py"}
    assert "not in the graph" not in out[1]


def test_no_graph_says_init_first(tmp_path):
    out = []
    assert scan.scan(tmp_path, out=out.append) == 1 and "init" in out[0]


def test_a_resolved_decisions_new_model_is_not_reported_as_unrecorded(tmp_path):
    p, did = make_project(tmp_path)
    (p / "app.py").write_text('MODEL = "gemini-3.5-flash-lite"\n')
    with Graph.for_project(p) as g:
        g.update_data(did, resolution={"to": "gemini:gemini-3.5-flash-lite"})
        assert scan.unrecorded(p, g) == []
