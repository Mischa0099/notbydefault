"""Milestone 4 part 7: checkpoint of the commit that added a decision record."""
import json
import subprocess

from conftest import FakeEntireRunner
from notbydefault.checkpoints import show
from notbydefault.entire import Entire
from notbydefault.graph import Graph
from notbydefault.records import write_record


def git(path, *args):
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=path,
                   check=True, capture_output=True)


def setup(tmp_path, trailer=True):
    git(tmp_path, "init", "-q")
    git(tmp_path, "remote", "add", "origin", "https://github.com/Mischa0099/notbydefault.git")
    with Graph.for_project(tmp_path) as g:
        did = g.add_node("decision", "MongoDB", {"kind": "service", "category": "database",
                                                 "status": "unreviewed"})
        write_record(tmp_path, g, did, "detected")
    git(tmp_path, "add", "decisions")
    msg = "Add decision record" + ("\n\nEntire-Checkpoint: 01CHECKPOINTID" if trailer else "")
    git(tmp_path, "commit", "-q", "-m", msg)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True,
                         text=True).stdout.strip()
    return did, sha


LIST = json.dumps([{"checkpoint_id": "01CHECKPOINTID", "session_id": "sess-1", "agent": "Claude Code",
                    "date": "2026-10-11T01:03:29+05:30", "message": "Add decision record",
                    "session_ids": ["sess-1"]},
                   {"checkpoint_id": "01OTHER", "message": "x"}])


def test_shows_checkpoint_row_and_full_sha_link(tmp_path):
    did, sha = setup(tmp_path)
    out = []
    show(tmp_path, did, Entire(exe="entire", runner=FakeEntireRunner({("checkpoint", "list"): LIST})),
         out.append)
    text = "\n".join(out)
    assert f"added in commit {sha}" in text and "checkpoint: 01CHECKPOINTID" in text
    assert "01CHECKPOINTID  2026-10-11T01:03:29+05:30  Claude Code" in text and "01OTHER" not in text
    assert f"https://entire.io/gh/Mischa0099/notbydefault/commit/{sha}" in text
    assert len(sha) == 40


def test_falls_back_to_explain_commit_without_trailer(tmp_path):
    did, sha = setup(tmp_path, trailer=False)
    out = []
    runner = FakeEntireRunner({("checkpoint", "list"): LIST,
                               ("checkpoint", "explain", "--commit"):
                                   json.dumps({"checkpoint_id": "01CHECKPOINTID"})})
    show(tmp_path, did, Entire(exe="entire", runner=runner), out.append)
    assert "checkpoint: 01CHECKPOINTID" in "\n".join(out)
    assert ["checkpoint", "explain", "--commit", sha, "--json"] in runner.calls


def test_uncommitted_record_explains_the_link_format(tmp_path):
    git(tmp_path, "init", "-q")
    with Graph.for_project(tmp_path) as g:
        did = g.add_node("decision", "MongoDB", {"kind": "service", "status": "unreviewed"})
        write_record(tmp_path, g, did, "detected")
    out = []
    show(tmp_path, did, Entire(exe="entire", runner=FakeEntireRunner({})), out.append)
    assert "not committed yet" in "\n".join(out) and "/commit/<FULL_SHA>" in "\n".join(out)
