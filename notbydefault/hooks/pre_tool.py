"""PreToolUse hook: flag consequential stack decisions and ask the user to confirm.

Claude Code runs this with the hook JSON on stdin. Contract (see Claude Code hooks docs):
- no output + exit 0  -> no opinion; the normal permission flow applies
- {"hookSpecificOutput": {"permissionDecision": "ask", ...}} -> user is prompted,
  permissionDecisionReason is shown in the prompt
This hook never exits non-zero and never blocks: on any error it logs and stays silent.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _project_dir(argv: list[str], event: dict) -> Path:
    import os
    if "--project" in argv:
        i = argv.index("--project")
        if i + 1 < len(argv):
            return Path(argv[i + 1])
    return Path(os.environ.get("CLAUDE_PROJECT_DIR") or event.get("cwd") or ".")


def _log(project_dir: Path | None, message: str) -> None:
    import tempfile
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n"
    targets = []
    if project_dir:
        targets.append(project_dir / ".notbydefault" / "hook.log")
    targets.append(Path(tempfile.gettempdir()) / "notbydefault-hook.log")
    for path in targets:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
            return
        except OSError:
            continue


def _rel(path: str | None, project_dir: Path) -> str | None:
    if not path:
        return None
    try:
        return Path(path).resolve().relative_to(project_dir.resolve()).as_posix()
    except (ValueError, OSError):
        return path


def record(graph, detection, event: dict, project_dir: Path) -> int:
    """Write an unreviewed decision node linked to project context and touched code."""
    session_id = event.get("session_id")
    file = _rel(detection.file, project_dir)

    existing = [d for d in graph.list_decisions("unreviewed")
                if d["data"].get("kind") == detection.kind
                and d["label"].lower() == detection.chosen.lower()]
    if existing:  # e.g. Claude retried after the user said no: don't pile up duplicates
        decision_id = existing[0]["id"]
        old = existing[0]["data"]
        files = old.get("files") or ([old["file"]] if old.get("file") else [])
        if file and file not in files:
            files.append(file)
        graph.update_data(decision_id, times_seen=old.get("times_seen", 1) + 1, files=files,
                          file=old.get("file") or file)
    else:
        data = detection.to_dict()
        data.update(status="unreviewed", file=file, files=[file] if file else [],
                    tool_name=event.get("tool_name"),
                    tool_use_id=event.get("tool_use_id"), times_seen=1)
        if event.get("tool_name") in ("Bash", "PowerShell"):
            data["command"] = (event.get("tool_input") or {}).get("command")
        decision_id = graph.add_node("decision", detection.chosen, data, session_id)
        for node in graph.nodes("goal"):
            graph.add_edge(decision_id, node["id"], "serves")
        for node_type in ("requirement", "constraint"):
            for node in graph.nodes(node_type):
                graph.add_edge(decision_id, node["id"], "depends_on")

    if file:
        code = graph.find_node("code", file)
        code_id = code["id"] if code else graph.add_node("code", file, {"path": file}, session_id)
        graph.add_edge(decision_id, code_id, "affects")
    return decision_id


def _spawn_telemetry(payload: dict, project_dir: Path) -> None:
    """Send the neatlogs span from a detached process so the hook stays fast.

    neatlogs import + flush takes ~2 s; the child flushes before *it* exits.
    """
    import subprocess
    payload = dict(payload, project_dir=str(project_dir))
    cmd = [sys.executable, "-m", "notbydefault.telemetry", json.dumps(payload)]
    kwargs = dict(cwd=str(REPO_ROOT), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                  stderr=subprocess.DEVNULL, close_fds=True)
    if sys.platform == "win32":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            subprocess.Popen(cmd, creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB,
                             **kwargs)
            return
        except OSError:
            pass  # job doesn't allow breakaway; fall through
        subprocess.Popen(cmd, creationflags=flags, **kwargs)
    else:
        subprocess.Popen(cmd, start_new_session=True, **kwargs)


def _quote(arg: str) -> str:
    return f'"{arg}"' if " " in arg else arg


def check_command(decision_id: int, project_dir: Path) -> str:
    """The exact command to inspect a decision, using the interpreter running this hook."""
    exe = _quote(sys.executable)
    if exe.startswith('"'):  # PowerShell needs the call operator for a quoted exe path
        exe = f"& {exe}"
    return (f"{exe} -m notbydefault check {decision_id} "
            f"--project {_quote(str(project_dir.resolve()))}")


def handle(event: dict, project_dir: Path) -> dict | None:
    from notbydefault.detect import detect

    detections = detect(event)
    if not detections:
        return None

    from notbydefault.graph import Graph

    with Graph.for_project(project_dir) as graph:
        ids = [record(graph, d, event, project_dir) for d in detections]

    parts = [f"this picks {d.chosen} as your {d.kind}"
             + (f" ({d.category})" if d.kind == "service" else "")
             + f". Run `{check_command(i, project_dir)}` to test the fit."
             for d, i in zip(detections, ids)]
    reason = "NotByDefault: " + " Also: ".join(parts)

    import os
    if os.environ.get("NOTBYDEFAULT_TELEMETRY") != "0":
        try:
            _spawn_telemetry({
                "session_id": event.get("session_id"),
                "tool_name": event.get("tool_name"),
                "decisions": [{"decision_id": i, "kind": d.kind, "chosen": d.chosen,
                               "category": d.category, "file": d.file}
                              for d, i in zip(detections, ids)],
            }, project_dir)
        except Exception as e:  # telemetry must never cost the user their prompt
            _log(project_dir, f"telemetry spawn failed: {e!r}")

    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "ask",
            "permissionDecisionReason": reason,
        }
    }


def main(argv: list[str]) -> int:
    start = time.perf_counter()
    project_dir = _project_dir(argv, {}) if "--project" in argv else None
    try:
        event = json.loads(sys.stdin.buffer.read().decode("utf-8", errors="replace") or "{}")
        project_dir = _project_dir(argv, event)
        out = handle(event, project_dir)
        if out:
            sys.stdout.write(json.dumps(out))
            sys.stdout.flush()
            ms = (time.perf_counter() - start) * 1000
            _log(project_dir, f"ask tool={event.get('tool_name')} {ms:.0f}ms "
                              f"{out['hookSpecificOutput']['permissionDecisionReason']}")
    except Exception:
        import traceback
        _log(project_dir, "error (allowed tool call):\n" + traceback.format_exc())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
