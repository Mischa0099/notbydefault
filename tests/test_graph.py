import pytest

from notbydefault.graph import Graph, db_path, find_affected_code


@pytest.fixture
def graph(tmp_path):
    g = Graph.for_project(tmp_path)
    yield g
    g.close()


def test_db_lives_under_project(tmp_path):
    Graph.for_project(tmp_path).close()
    assert db_path(tmp_path).exists()
    assert db_path(tmp_path).parent.name == ".notbydefault"


def test_add_and_get_node(graph):
    nid = graph.add_node("goal", "Ship an MVP", {"source": "init"}, session_id="s1")
    node = graph.get_node(nid)
    assert node["type"] == "goal"
    assert node["label"] == "Ship an MVP"
    assert node["data"] == {"source": "init"}
    assert node["session_id"] == "s1"
    assert node["created_at"]


def test_rejects_unknown_types(graph):
    with pytest.raises(ValueError):
        graph.add_node("banana", "x")
    a = graph.add_node("goal", "a")
    with pytest.raises(ValueError):
        graph.add_edge(a, a, "likes")


def test_duplicate_edges_ignored(graph):
    a = graph.add_node("decision", "a")
    b = graph.add_node("goal", "b")
    graph.add_edge(a, b, "serves")
    graph.add_edge(a, b, "serves")
    assert graph.edges() == [{"src": a, "dst": b, "type": "serves"}]


def test_list_decisions_filters_by_status(graph):
    graph.add_node("goal", "g")
    d1 = graph.add_node("decision", "supabase", {"status": "unreviewed"})
    d2 = graph.add_node("decision", "gpt-4o", {"status": "accepted"})
    assert [d["id"] for d in graph.list_decisions()] == [d1, d2]
    assert [d["id"] for d in graph.list_decisions("unreviewed")] == [d1]


def test_get_dependencies_is_transitive_and_skips_non_dependency_edges(graph):
    goal = graph.add_node("goal", "Ship")
    req = graph.add_node("requirement", "10k users")
    con = graph.add_node("constraint", "Budget $50")
    ev = graph.add_node("evidence", "benchmark")
    alt = graph.add_node("alternative", "firebase")
    code = graph.add_node("code", "app.py")
    dec = graph.add_node("decision", "supabase")
    graph.add_edge(dec, req, "depends_on")
    graph.add_edge(dec, ev, "supported_by")
    graph.add_edge(dec, alt, "rejects")
    graph.add_edge(dec, code, "affects")
    graph.add_edge(req, goal, "serves")
    graph.add_edge(req, con, "depends_on")
    deps = {n["id"] for n in graph.get_dependencies(dec)}
    assert deps == {req, ev, goal, con}


def test_get_dependencies_handles_cycles(graph):
    a = graph.add_node("decision", "a")
    b = graph.add_node("requirement", "b")
    graph.add_edge(a, b, "depends_on")
    graph.add_edge(b, a, "depends_on")
    assert [n["id"] for n in graph.get_dependencies(a)] == [b]


def test_update_data_merges(graph):
    d = graph.add_node("decision", "x", {"status": "unreviewed", "kind": "service"})
    graph.update_data(d, status="accepted")
    assert graph.get_node(d)["data"] == {"status": "accepted", "kind": "service"}


def test_delete_cascades_edges(graph):
    a = graph.add_node("decision", "a")
    b = graph.add_node("goal", "b")
    graph.add_edge(a, b, "serves")
    graph.delete_nodes([b])
    assert graph.edges() == []


def test_find_affected_code_returns_touched_file(graph):
    d = graph.add_node("decision", "x", {"file": "src/db.ts"})
    assert find_affected_code(graph.get_node(d)) == ["src/db.ts"]
    e = graph.add_node("decision", "y", {})
    assert find_affected_code(graph.get_node(e)) == []
    f = graph.add_node("decision", "z", {"file": "a.py", "files": ["a.py", "b.py"]})
    assert find_affected_code(graph.get_node(f)) == ["a.py", "b.py"]
