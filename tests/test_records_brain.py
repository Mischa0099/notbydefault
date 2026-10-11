"""Milestone 4 part 3: decision records on every verdict, and Entire Brain (mocked CLI)."""
from conftest import FakeEntireRunner
from notbydefault import brain
from notbydefault.entire import Entire
from notbydefault.graph import Graph
from notbydefault.records import BEGIN, END, write_record, write_records

PLUGINS_NO_BRAIN = "Managed plugin directory: x\n\n  graph  v0.4.0  x\\entire-graph.exe\n"
PLUGINS_BRAIN = PLUGINS_NO_BRAIN + "  brain  v0.1.0  x\\entire-brain.exe\n"
BRAIN_HELP = """Entire Brain\n\nUsage:\n  entire brain [command]\n\nAvailable Commands:
  save        Save a document to your brain
  query       Ask your brain\n\nFlags:\n  -h, --help\n"""
SAVE_HELP = "Usage:\n  entire brain save --file <path> [flags]\n\nFlags:\n  --file string\n"


def setup(tmp_path):
    g = Graph.for_project(tmp_path)
    did = g.add_node("decision", "MongoDB", {"kind": "service", "category": "database",
                                             "status": "unreviewed", "file": "requirements.txt",
                                             "evidence": "pymongo", "tool_name": "Bash"}, "sess-1")
    alt = g.add_node("alternative", "PostgreSQL", {"reason": "team knows Mongo"})
    g.add_edge(did, alt, "rejects")
    ev = g.add_node("evidence", "Fit test results", {"path": ".notbydefault/results/1.json"})
    g.add_edge(did, ev, "supported_by")
    return g, did


def entire(responses):
    return Entire(exe="entire", runner=FakeEntireRunner(responses))


def test_record_is_created_with_why_evidence_alternatives_caveats(tmp_path):
    g, did = setup(tmp_path)
    path = write_record(tmp_path, g, did, "detected")
    text = path.read_text(encoding="utf-8")
    assert path.name == "0001-database-mongodb.md"
    for part in ("### Why", "Bash call (pymongo)", "### Evidence", ".notbydefault/results/1.json",
                 "### Rejected alternatives", "`PostgreSQL`: team knows Mongo", "### Caveats",
                 "### Affected code", "requirements.txt", "sess-1", "detected"):
        assert part in text, part
    assert g.get_node(did)["data"]["decision_record"] == "decisions/0001-database-mongodb.md"
    g.close()


def test_record_is_updated_in_place_and_keeps_hand_written_text(tmp_path):
    g, did = setup(tmp_path)
    path = write_record(tmp_path, g, did, "detected")
    path.write_text("# 0001. My notes\n\nHand-written context.\n\n" + path.read_text(encoding="utf-8")
                    .split("\n\n", 1)[1], encoding="utf-8")
    g.update_data(did, status="needs_review", review_reason="depends on where it runs",
                  review_basis="depends on deployment", review_source="https://docs")
    again = write_record(tmp_path, g, did, "needs review after requirement #9")
    text = again.read_text(encoding="utf-8")
    assert again == path and len(list((tmp_path / "decisions").glob("*.md"))) == 1
    assert "Hand-written context." in text and text.count(BEGIN) == 1 and text.count(END) == 1
    assert "Needs review" in text and "https://docs" in text
    assert "needs review after requirement #9" in text and "detected" in text   # history
    g.close()


def test_check_record_is_reused_for_the_same_decision(tmp_path):
    """A record written by `check` (no markers yet) gets the summary block appended."""
    g, did = setup(tmp_path)
    (tmp_path / "decisions").mkdir()
    old = tmp_path / "decisions" / "0003-ai-model-x.md"
    old.write_text(f"# 0003. AI model\n\n- Decision graph node: #{did} (`x`)\n\nBody.\n",
                   encoding="utf-8")
    assert write_record(tmp_path, g, did, "fit check") == old
    assert "Body." in old.read_text(encoding="utf-8") and BEGIN in old.read_text(encoding="utf-8")
    g.close()


def test_brain_missing_explains_how_to_install_and_skips(tmp_path):
    g, did = setup(tmp_path)
    notes = []
    write_records(tmp_path, g, [did], "detected", out=notes.append,
                  entire=entire({("plugin", "list"): PLUGINS_NO_BRAIN}))
    assert "not installed" in notes[0] and "entire plugin install" in notes[0]
    assert (tmp_path / "decisions" / "0001-database-mongodb.md").exists()   # record still written
    g.close()


def test_brain_installed_saves_with_the_documented_command(tmp_path):
    g, did = setup(tmp_path)
    runner = FakeEntireRunner({("plugin", "list"): PLUGINS_BRAIN, ("brain", "--help"): BRAIN_HELP,
                               ("brain", "save", "--help"): SAVE_HELP, ("brain", "save", "--file"): "ok"})
    notes = []
    write_records(tmp_path, g, [did], "detected", out=notes.append,
                  entire=Entire(exe="entire", runner=runner))
    assert notes and notes[0].startswith("Saved 0001-database-mongodb.md to Entire Brain")
    [save] = [c for c in runner.calls if c[:3] == ["brain", "save", "--file"]]
    assert save[3].endswith("0001-database-mongodb.md")
    g.close()


def test_brain_without_a_documented_save_command_is_skipped(tmp_path):
    e = entire({("plugin", "list"): PLUGINS_BRAIN,
                ("brain", "--help"): "Available Commands:\n  query   Ask\n\n"})
    saved, msg = brain.save(tmp_path, tmp_path / "x.md", None, e)
    assert not saved and "documents no subcommand" in msg


def test_brain_failure_never_raises(tmp_path):
    e = entire({("plugin", "list"): PLUGINS_BRAIN, ("brain", "--help"): BRAIN_HELP,
                ("brain", "save", "--help"): SAVE_HELP, ("brain", "save", "--file"): 3})
    saved, msg = brain.save(tmp_path, tmp_path / "x.md", None, e)
    assert not saved and "failed" in msg
