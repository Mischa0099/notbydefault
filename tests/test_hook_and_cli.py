import json
import os
import subprocess
import sys
import time
from pathlib import Path

from notbydefault import PACKAGE_DIR
from notbydefault.cli import main as cli, merge_settings
from notbydefault.graph import Graph

HOOK = PACKAGE_DIR / "hooks" / "pre_tool.py"


def run_hook(project: Path, event, raw: str | None = None):
    env = dict(os.environ, NOTBYDEFAULT_TELEMETRY="0")
    start = time.perf_counter()
    proc = subprocess.run([sys.executable, str(HOOK), "--project", str(project)],
                          input=(raw if raw is not None else json.dumps(event)).encode(),
                          capture_output=True, env=env, timeout=30)
    return proc, (time.perf_counter() - start) * 1000


def event(tool, tool_input, project):
    return {"session_id": "sess-1", "hook_event_name": "PreToolUse", "cwd": str(project),
            "tool_name": tool, "tool_input": tool_input, "tool_use_id": "toolu_1"}


def test_hook_asks_and_records_decision(tmp_path):
    cli(["init", str(tmp_path), "--goal", "Notes app", "--users", "100",
         "--budget", "$0", "--no-input"])
    proc, _ = run_hook(tmp_path, event("Bash", {"command": "pip install pymongo"}, tmp_path))
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)["hookSpecificOutput"]
    assert out["hookEventName"] == "PreToolUse"
    assert out["permissionDecision"] == "ask"
    assert out["permissionDecisionReason"].startswith(
        "NotByDefault: this picks MongoDB as your service (database).")
    with Graph.for_project(tmp_path) as g:
        decision_id = g.list_decisions()[0]["id"]
    # a path with spaces is quoted, with PowerShell's call operator so it still runs
    exe = f'& "{sys.executable}"' if " " in sys.executable else sys.executable
    assert (f"Run `{exe} -m notbydefault check {decision_id} "
            f"--project {tmp_path.resolve()}` to test the fit.") in out["permissionDecisionReason"]

    with Graph.for_project(tmp_path) as g:
        [d] = g.list_decisions()
        assert d["label"] == "MongoDB"
        assert d["data"]["status"] == "unreviewed"
        assert d["session_id"] == "sess-1"
        deps = {n["label"] for n in g.get_dependencies(d["id"])}
        assert deps == {"Notes app", "Expected users: 100", "Monthly budget: $0"}


def test_hook_links_code_and_dedupes_retries(tmp_path):
    f = tmp_path / "ai.py"
    e = event("Write", {"file_path": str(f), "content": 'MODEL = "gpt-4o"\n'}, tmp_path)
    run_hook(tmp_path, e)
    run_hook(tmp_path, e)  # Claude retries the same write after a "no"
    with Graph.for_project(tmp_path) as g:
        [d] = g.list_decisions()
        assert d["data"]["times_seen"] == 2
        [code] = g.nodes("code")
        assert code["label"] == "ai.py"
        assert {"src": d["id"], "dst": code["id"], "type": "affects"} in g.edges()


def test_reused_decision_picks_up_files_from_later_calls(tmp_path):
    from notbydefault.graph import find_affected_code
    run_hook(tmp_path, event("PowerShell", {"command": "pip install pymongo"}, tmp_path))
    req = tmp_path / "requirements.txt"
    run_hook(tmp_path, event("Edit", {"file_path": str(req), "old_string": "",
                                      "new_string": "pymongo>=4.6\n"}, tmp_path))
    with Graph.for_project(tmp_path) as g:
        [d] = g.list_decisions()
        assert find_affected_code(d) == ["requirements.txt"]


def test_hook_is_silent_for_non_decisions(tmp_path):
    proc, ms = run_hook(tmp_path, event("Bash", {"command": "ls -la"}, tmp_path))
    assert proc.returncode == 0 and proc.stdout == b""
    proc, _ = run_hook(tmp_path, event("Read", {"file_path": "x"}, tmp_path))
    assert proc.stdout == b""


def test_hook_never_breaks_on_garbage(tmp_path):
    for raw in ["not json", "", "[]", '{"tool_name": "Bash", "tool_input": null}']:
        proc, _ = run_hook(tmp_path, None, raw=raw)
        assert proc.returncode == 0
        assert proc.stdout == b""


def test_hook_is_fast(tmp_path):
    run_hook(tmp_path, event("Bash", {"command": "npm i stripe"}, tmp_path))  # warm caches
    _, ms_quiet = run_hook(tmp_path, event("Bash", {"command": "ls"}, tmp_path))
    _, ms_ask = run_hook(tmp_path, event("Bash", {"command": "npm i firebase"}, tmp_path))
    assert ms_quiet < 300, ms_quiet
    assert ms_ask < 300, ms_ask


def test_init_skips_blank_answers_and_refuses_rerun(tmp_path, capsys):
    assert cli(["init", str(tmp_path), "--goal", "Shop", "--team", "2", "--no-input"]) == 0
    with Graph.for_project(tmp_path) as g:
        assert [(n["type"], n["label"]) for n in g.nodes()] == [
            ("goal", "Shop"), ("constraint", "Team size: 2")]
    assert cli(["init", str(tmp_path), "--no-input"]) == 1
    assert cli(["init", str(tmp_path), "--goal", "Shop v2", "--no-input", "--force"]) == 0
    with Graph.for_project(tmp_path) as g:
        assert [n["label"] for n in g.nodes()] == ["Shop v2"]


def test_install_merges_without_touching_other_hooks(tmp_path):
    other = {"type": "command", "command": "entire hooks claude-code pre-task"}
    settings_path = tmp_path / ".claude" / "settings.json"
    settings_path.parent.mkdir()
    settings_path.write_text(json.dumps({
        "permissions": {"allow": ["Bash(ls:*)"]},
        "hooks": {"PreToolUse": [{"matcher": "Agent", "hooks": [other]}],
                  "Stop": [{"matcher": "", "hooks": [other]}]},
    }))
    assert cli(["install", str(tmp_path)]) == 0
    assert cli(["install", str(tmp_path)]) == 0  # idempotent
    s = json.loads(settings_path.read_text())
    assert s["permissions"] == {"allow": ["Bash(ls:*)"]}
    assert s["hooks"]["Stop"] == [{"matcher": "", "hooks": [other]}]
    pre = s["hooks"]["PreToolUse"]
    assert pre[0] == {"matcher": "Agent", "hooks": [other]}
    assert len(pre) == 2
    ours = pre[1]["hooks"][0]
    assert pre[1]["matcher"] == "Bash|PowerShell|Write|Edit|MultiEdit"
    assert Path(ours["command"]).is_absolute() and Path(ours["command"]).exists()
    assert Path(ours["args"][0]) == HOOK
    assert ours["args"][1:] == ["--project", str(tmp_path.resolve())]


def test_merge_settings_on_empty():
    s = merge_settings({}, Path("C:/x"))
    assert list(s["hooks"]) == ["PreToolUse", "UserPromptSubmit"]
    assert "matcher" not in s["hooks"]["UserPromptSubmit"][0]   # no matcher support


# --- already decided in this project (D3/D4) ----------------------------------------------

def test_reviewed_decision_does_not_interrupt_again(tmp_path):
    with Graph.for_project(tmp_path) as g:
        did = g.add_node("decision", "OpenAI", {"kind": "service", "category": "ai-provider",
                                                "status": "reviewed", "times_seen": 1})
    proc, _ = run_hook(tmp_path, event("Bash", {"command": "pip install --upgrade openai"}, tmp_path))
    assert proc.returncode == 0 and proc.stdout == b""
    with Graph.for_project(tmp_path) as g:
        assert g.get_node(did)["data"]["times_seen"] == 2     # counted as seen
        assert len(g.list_decisions()) == 1


def test_unreviewed_decision_still_asks(tmp_path):
    with Graph.for_project(tmp_path) as g:
        g.add_node("decision", "OpenAI", {"kind": "service", "category": "ai-provider",
                                          "status": "unreviewed", "times_seen": 1})
    proc, _ = run_hook(tmp_path, event("Bash", {"command": "pip install openai"}, tmp_path))
    assert json.loads(proc.stdout)["hookSpecificOutput"]["permissionDecision"] == "ask"


def test_model_already_used_elsewhere_does_not_interrupt(tmp_path):
    (tmp_path / "app.py").write_text('MODEL = "gpt-4o"\n')
    test_file = tmp_path / "tests" / "test_app.py"
    proc, _ = run_hook(tmp_path, event("Write", {"file_path": str(test_file),
                                                 "content": 'assert MODEL == "gpt-4o"\n'}, tmp_path))
    assert proc.stdout == b""
    # a different model is still a new choice
    proc, _ = run_hook(tmp_path, event("Write", {"file_path": str(test_file),
                                                 "content": 'M = "gpt-4o-mini"\n'}, tmp_path))
    assert json.loads(proc.stdout)["hookSpecificOutput"]["permissionDecision"] == "ask"


def test_service_already_in_a_manifest_does_not_interrupt(tmp_path):
    (tmp_path / "requirements.txt").write_text("pymongo>=4\n")
    proc, _ = run_hook(tmp_path, event("Bash", {"command": "pip install pymongo"}, tmp_path))
    assert proc.stdout == b""


def test_the_file_being_written_does_not_count_as_elsewhere(tmp_path):
    f = tmp_path / "app.py"
    f.write_text('MODEL = "gpt-4o"\n')
    proc, _ = run_hook(tmp_path, event("Edit", {"file_path": str(f), "old_string": '"gpt-4o"',
                                                "new_string": '"gpt-4.1"'}, tmp_path))
    assert json.loads(proc.stdout)["hookSpecificOutput"]["permissionDecision"] == "ask"
