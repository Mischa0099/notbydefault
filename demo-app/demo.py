"""Reproducible NotByDefault demo scenarios, run on a throwaway copy of this project.

    python demo.py reset     # copy the app + its Decision Graph into .demo-run/ (read-only on the original)
    python demo.py pivot     # scenario a: the user pivots to gpt-4o; Pivot Check explains the cost
    python demo.py require   # scenario b: a new EU data-residency requirement makes decisions stale
    python demo.py all       # reset, then both scenarios

The live state in .notbydefault/ is never written: the graph is copied with SQLite's online
backup (safe while a `check` is running), and every scenario runs inside .demo-run/.
"""
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIVE = HERE / ".notbydefault"
WORK = HERE / ".demo-run"
PYTHON = sys.executable
HOOK = HERE.parent / "notbydefault" / "hooks" / "user_prompt.py"

PIVOT_PROMPT = "actually, let's switch to gpt-4o for action items, it's the best"
REQUIREMENT = "We signed an EU customer: EU data residency required"
APP_FILES = ["app.py", "actions.py", "requirements.txt"]


def reset() -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    (WORK / ".notbydefault").mkdir(parents=True)
    for name in APP_FILES:
        shutil.copy2(HERE / name, WORK / name)
    src = sqlite3.connect(f"file:{LIVE / 'graph.db'}?mode=ro", uri=True)   # read-only
    dst = sqlite3.connect(WORK / ".notbydefault" / "graph.db")
    src.backup(dst)
    src.close()
    dst.close()
    for name in ("candidates.yaml",):
        if (LIVE / name).exists():
            shutil.copy2(LIVE / name, WORK / ".notbydefault" / name)
    for folder in ("callsites", "results"):   # what each decision is for; measured tokens/call
        if (LIVE / folder).is_dir():
            shutil.copytree(LIVE / folder, WORK / ".notbydefault" / folder)
    print(f"demo copy ready in {WORK} (graph copied read-only from {LIVE / 'graph.db'})")


def pivot() -> None:
    event = {"session_id": "demo-pivot", "hook_event_name": "UserPromptSubmit",
             "cwd": str(WORK), "prompt": PIVOT_PROMPT}
    print(f'User types in Claude Code: "{PIVOT_PROMPT}"\n')
    start = time.perf_counter()
    proc = subprocess.run([PYTHON, str(HOOK), "--project", str(WORK)],
                          input=json.dumps(event).encode(), capture_output=True,
                          env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    ms = (time.perf_counter() - start) * 1000
    print(f"UserPromptSubmit hook: exit {proc.returncode}, {ms:.0f} ms")
    if not proc.stdout:
        print("(no context added; see .demo-run/.notbydefault/hook.log)")
        return
    ctx = json.loads(proc.stdout)["hookSpecificOutput"]["additionalContext"]
    print("Context Claude Code receives alongside the prompt:\n")
    print(ctx)


def require() -> None:
    nbd = [PYTHON, "-m", "notbydefault"]
    print(f'$ python -m notbydefault require .demo-run "{REQUIREMENT}"\n')
    subprocess.run(nbd + ["require", str(WORK), REQUIREMENT], cwd=HERE.parent, check=True)
    sys.path.insert(0, str(HERE.parent))
    from notbydefault.graph import Graph
    with Graph.for_project(WORK) as g:
        stale_models = [d["id"] for d in g.nodes("decision")
                        if d["data"].get("status") == "stale" and d["data"].get("kind") == "model"]
    if stale_models:
        target = max(stale_models)
        print(f"\n$ python -m notbydefault check {target} --project .demo-run\n")
        subprocess.run(nbd + ["check", str(target), "--project", str(WORK), "-y"],
                       cwd=HERE.parent, check=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)   # keep our lines in order with subprocess output
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("reset", "all"):
        reset()
    if cmd in ("pivot", "all"):
        print("\n=== Scenario a: pivot ===")
        pivot()
    if cmd in ("require", "all"):
        print("\n=== Scenario b: new requirement ===")
        require()
    if cmd not in ("reset", "pivot", "require", "all"):
        sys.exit(__doc__)
