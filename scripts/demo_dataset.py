"""Build a demo dataset that exercises every part of NotByDefault, on a throwaway copy of demo-app.

    python scripts/demo_dataset.py [--out .demo-dataset] [--offline]

Steps (each prints what it did; every trace id created is listed at the end):
  1. copy demo-app (graph.db via SQLite's backup API, so the live graph is never touched) into
     <out>/demo-app, make it a git repo on main
  2. watch      record each decision's assumptions (baseline)
  3. resolve 21 dry run, then --apply on branch nbd/21-... (switch to the fit check's
                recommendation, verified from its response cache; commit answered "yes" on the
                copy only). Done BEFORE the requirement, so it lands on the resolved decision:
                Gemini documents no residency guarantee -> needs review (Groq: US only -> stale)
  4. pivot      demo.py's pivot prompt through the UserPromptSubmit hook   (1 Groq call)
  5. require    demo.py's EU data-residency requirement                     (Groq calls)
  6. an agent turn: the PreToolUse hook catches a new model choice; the agent's reason comes
     from a SCRIPTED transcript (scripts/demo_transcript.jsonl), labelled as such
  7. reasons    a SCRIPTED agent reason on each earlier decision (stored as agent_reason, its
                source saying so), since demo-app's own sessions recorded none
  8. claims     on #21 first, then the other decisions and the agent turn's (1 Groq call each)
  9. watch      re-evaluate the assumptions (and record the new decision's baseline)
 10. scan       decision debt
 11. shots      regenerate docs/screenshots from the dataset (#21's receipt as decision-*.png,
                the agent turn's as decision-claims-*.png); --no-shots skips it
--offline skips the steps that call Groq (4, 5, 8) for a quick dry build. A rebuild reuses the
previous build's model-response cache (same prompts answered from disk); --fresh ignores it.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "demo-app"))

# the line a command prints for the trace it created (claim sources also quote older fit-check
# trace ids, which this run did not create)
TRACE = re.compile(r"neatlogs trace id: ([0-9a-f]{32})")
TRANSCRIPT = REPO / "scripts" / "demo_transcript.jsonl"
DIGEST = '''"""Weekly digest of a team's notes (scripted demo turn)."""
import os

from openai import OpenAI

client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"])


def weekly_digest(notes: list[str]) -> str:
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[{"role": "system", "content": "Write a short weekly digest of these notes."},
                  {"role": "user", "content": "\\n".join(f"- {n}" for n in notes)}],
    )
    return response.choices[0].message.content
'''
# What the agent "said" when it made each of demo-app's earlier decisions. Scripted: demo-app's
# sessions predate the hook recording reasons, and these give the claim check a mix of claims.
REASONS = {
    7: "I'll add pymongo and keep the notes in MongoDB. It's the most popular document database "
       "and it's battle-tested in production.",
    9: "I'll add the openai SDK. It's the industry standard for LLM APIs and everyone uses it.",
    10: "I'll call gpt-4o for the summaries. It's the best model for summarisation and it's fast "
        "enough for a web request.",
    13: "Switching the summary to openai/gpt-oss-120b on Groq. It's much cheaper than gpt-4o, it's "
        "really fast on Groq, and its summaries were as accurate as gpt-4o's.",
    21: "I'll write actions.py with openai/gpt-oss-120b on Groq. It's cheap enough for our volume. "
        "It's very fast on Groq. It's the most accurate open model at extracting action items. "
        "Data stays in the EU, so we're GDPR compliant.",
}
REASON_SOURCE = "scripted demo reason (scripts/demo_dataset.py), not a real session"
traces: list[tuple[str, str]] = []


def say(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 70 - len(title)), flush=True)


def note_traces(step: str, text: str) -> None:
    for t in dict.fromkeys(TRACE.findall(text)):
        if (step, t) not in traces:
            traces.append((step, t))


def captured(step: str, fn, *a, **kw):
    """Run fn with out=list.append, print its lines, collect the trace ids it reports."""
    lines: list[str] = []
    code = fn(*a, out=lines.append, **kw)
    text = "\n".join(lines)
    print(text)
    note_traces(step, text)
    return code, text


def git(p: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=p, capture_output=True, text=True,
                          check=True).stdout.strip()


def _remove(path: Path) -> None:
    """rmtree that also removes git's read-only object files (Windows)."""
    import stat

    def writable(func, p, _exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    shutil.rmtree(path, onexc=writable)


def copy_demo(dest: Path, fresh: bool = False) -> Path:
    """Fresh copy; the previous build's model-response cache is carried over (unless `fresh`),
    so a rebuild answers the same prompts from it instead of calling Groq again."""
    src = REPO / "demo-app"
    keep = None
    if dest.exists():
        old_cache = dest / ".notbydefault" / "cache"
        if old_cache.is_dir() and not fresh:
            keep = dest.parent / ".cache-carry"
            shutil.rmtree(keep, ignore_errors=True)
            shutil.copytree(old_cache, keep)
        _remove(dest)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns(
        "graph.db*", "__pycache__", ".demo-run", "*.pyc"))
    live = sqlite3.connect(f"{(src / '.notbydefault' / 'graph.db').resolve().as_uri()}?mode=ro",
                           uri=True)
    copy = sqlite3.connect(dest / ".notbydefault" / "graph.db")
    live.backup(copy)
    live.close()
    copy.close()
    if keep is not None:
        shutil.copytree(keep, dest / ".notbydefault" / "cache", dirs_exist_ok=True)
        shutil.rmtree(keep, ignore_errors=True)
    (dest / ".gitignore").write_text(".notbydefault/\n__pycache__/\n.claude/\n")
    git(dest, "init", "-q", "-b", "main")
    git(dest, "config", "user.email", "demo@notbydefault.local")
    git(dest, "config", "user.name", "NotByDefault demo")
    git(dest, "add", "-A")
    git(dest, "commit", "-q", "-m", "demo-app baseline (copied for the demo dataset)")
    return dest


def resolve_21(p: Path, did: int) -> None:
    from notbydefault.fit.tracing import make_tracer
    from notbydefault.resolve import Resolver, ResolveOptions
    for apply in (False, True):
        say(f"resolve {did} " + ("--apply (on the copy's branch; commit answered yes)" if apply
                                 else "(dry run)"))
        lines: list[str] = []
        r = Resolver(p, did, ResolveOptions(apply=apply, yes=apply),
                     tracer=make_tracer(f"demo-resolve-{did}"),
                     ask=lambda q: "y", out=lines.append)
        code = r.run()
        print("\n".join(lines) + f"\n(exit {code})")
        if r.tracer.trace_id:
            traces.append((f"resolve {did}{' --apply' if apply else ''}", r.tracer.trace_id))


def pivot(p: Path) -> None:
    from demo import PIVOT_PROMPT
    from notbydefault.fit.tracing import make_tracer
    from notbydefault.hooks.user_prompt import handle
    say(f'pivot: user types "{PIVOT_PROMPT}"')
    tracer = make_tracer("demo-pivot")
    with tracer.root("notbydefault.pivot_check", {"notbydefault.prompt": PIVOT_PROMPT}):
        out, info = handle({"session_id": "demo-pivot", "hook_event_name": "UserPromptSubmit",
                            "prompt": PIVOT_PROMPT}, p, tracer=tracer)
    tracer.flush()
    print(out["hookSpecificOutput"]["additionalContext"] if out else f"no pivot: {info}")
    if tracer.trace_id:
        traces.append(("pivot check", tracer.trace_id))


def agent_turn(p: Path) -> int | None:
    """The PreToolUse hook catches a new model choice; the reason comes from the scripted
    transcript, so the claim check has the agent's words to check."""
    from notbydefault.graph import Graph
    from notbydefault.hooks.pre_tool import handle
    say("agent turn: Write digest.py (scripted transcript) -> PreToolUse hook")
    event = {"session_id": "demo-digest", "hook_event_name": "PreToolUse",
             "tool_name": "Write", "tool_use_id": "toolu_demo_digest",
             "transcript_path": str(TRANSCRIPT),
             "tool_input": {"file_path": str(p / "digest.py"), "content": DIGEST}}
    old = os.environ.get("NOTBYDEFAULT_ENTIRE")
    os.environ["NOTBYDEFAULT_ENTIRE"] = "0"       # no detached cache refresh racing the build
    try:
        out = handle(event, p)
    finally:
        if old is None:
            os.environ.pop("NOTBYDEFAULT_ENTIRE", None)
        else:
            os.environ["NOTBYDEFAULT_ENTIRE"] = old
    print(out["hookSpecificOutput"]["permissionDecisionReason"] if out else "(hook stayed silent)")
    (p / "digest.py").write_text(DIGEST, encoding="utf-8")     # the tool call runs after the hook
    with Graph.for_project(p) as g:
        hit = [d for d in g.nodes("decision") if d["session_id"] == "demo-digest"]
    if hit:
        print(f"agent's recorded reason ({hit[-1]['data'].get('agent_reason_source')}): "
              f"{hit[-1]['data'].get('agent_reason')}")
    return hit[-1]["id"] if hit else None


def record_reasons(p: Path) -> list[int]:
    """Store REASONS as each decision's agent_reason (only where none is recorded)."""
    from notbydefault.graph import Graph
    say("agent reasons on the earlier decisions (scripted, labelled as such)")
    done = []
    with Graph.for_project(p) as g:
        for did, reason in REASONS.items():
            d = g.get_node(did)
            if not d or d["type"] != "decision" or d["data"].get("agent_reason"):
                print(f"  #{did}: skipped ({'already has a reason' if d else 'not in the graph'})")
                continue
            g.update_data(did, agent_reason=reason, agent_reason_source=REASON_SOURCE)
            print(f"  #{did} {d['label']}: {reason}")
            done.append(did)
    return done


def screenshots(p: Path, decision: int, claims_decision: int | None) -> int:
    say("screenshots -> docs/screenshots")
    cmd = [sys.executable, str(REPO / "scripts" / "ui_screenshots.py"), "--project", str(p),
           "--decision", str(decision)]
    if claims_decision:
        cmd += ["--claims-decision", str(claims_decision)]
    return subprocess.run(cmd, cwd=REPO).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(REPO / ".demo-dataset"))
    ap.add_argument("--decision", type=int, default=21, help="decision to resolve (default 21)")
    ap.add_argument("--offline", action="store_true", help="skip the steps that call Groq")
    ap.add_argument("--fresh", action="store_true",
                    help="don't reuse the previous build's model-response cache")
    ap.add_argument("--no-shots", action="store_true", help="don't regenerate docs/screenshots")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace", line_buffering=True)
    from demo import REQUIREMENT
    from notbydefault.watch import watch
    from notbydefault.claims import run as claims_run
    from notbydefault.scan import scan
    from notbydefault.env import load_env
    from notbydefault.stale import require
    load_env()

    p = copy_demo(Path(args.out).resolve() / "demo-app", args.fresh)
    say(f"copy ready: {p} (git repo on main)")
    say("watch (baseline)")
    captured("watch (baseline)", watch, p)
    resolve_21(p, args.decision)
    if not args.offline:
        pivot(p)
        say(f'require "{REQUIREMENT}"')
        captured("require", require, p, REQUIREMENT)
    new_id = agent_turn(p)
    reasoned = record_reasons(p)
    if not args.offline:
        first = [args.decision] if args.decision in reasoned else []
        for did in first + [d for d in reasoned if d != args.decision] + [new_id] * bool(new_id):
            say(f"claims {did}")
            captured(f"claims {did}", claims_run, p, did)
    say("watch (after the requirement and the new decision)")
    captured("watch", watch, p)
    say("scan")
    captured("scan", scan, p)

    shots = 0 if args.no_shots else screenshots(p, args.decision, new_id)

    say("trace ids created")
    for step, t in traces:
        print(f"  {step:<28} {t}")
    if not traces:
        print("  none (telemetry off, or --offline)")
    print(f"\nDataset: {p}\nUI:      python -m notbydefault ui --project \"{p}\"")
    if shots:
        print(f"screenshots failed (exit {shots})")
    return shots


if __name__ == "__main__":
    sys.exit(main())
