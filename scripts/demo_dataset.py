"""Build a demo dataset that exercises every part of NotByDefault, on a throwaway copy of demo-app.

    python scripts/demo_dataset.py --out .demo-dataset/demo-app [--offline] [--no-shots]

--out is reset from demo-app/ (graph.db copied via SQLite's backup API, so the live graph is never
touched), made a git repo on main, and then:
  0. watch      record each decision's assumptions (baseline)
  1. reasons    a SCRIPTED agent reason on #13, #21, MongoDB, OpenAI and gpt-4o (stored as
                agent_reason, its source saying so: demo-app's own sessions recorded none)
  2. resolve 21 dry run, then --apply on branch nbd/21-... (commit answered "yes", on the copy)
  3. pivot      the user types PIVOT_PROMPT. The real Pivot Check runs first; when it records
                nothing (its prefilter or confirm call says no, or --offline has no cached
                answer), ONE pivot is written from SEEDED_PIVOT, marked source=seeded
     + an agent turn: the PreToolUse hook catches a new model choice, its reason read from a
                scripted transcript (scripts/demo_transcript.jsonl)
  4. claims     on #21 first, then the other reasoned decisions and the agent turn's
  5. require    REQUIREMENT: Groq documents US only -> #13 stale; #21 now runs on Gemini, which
                documents no region -> needs review
  6. watch      re-evaluate the assumptions ('data region: any' broken on #21)
  7. scan       decision debt
  8. shots      regenerate docs/screenshots (light + dark) from the dataset; --no-shots skips it

Every step prints the trace id it created, the graph ids it added or changed and the files it
wrote. Model calls are answered from the copy's response cache (carried over from the previous
build unless --fresh); every call that would go to a provider is printed with its step. With
--offline those calls are skipped instead (each skip printed), and each step falls back:
claims -> keyword split, require -> the region the text names, stale judge -> unclear,
pivot -> the seeded record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
SHOTS = REPO / "docs" / "screenshots"
PIVOT_PROMPT = "let's drop verifiers, they slow things down"
REQUIREMENT = "data must stay in the EU"
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
# What the agent "said" when it made each of demo-app's earlier decisions (keyed by the
# decision's label in demo-app's graph, which is checked). Scripted: demo-app's sessions predate
# the hook recording reasons.
REASONS = {
    (21, "openai/gpt-oss-120b"): "openai/gpt-oss-120b: cost-effective, strong at structured "
                                 "extraction, widely used in production",
    (13, "openai/gpt-oss-120b"): "openai/gpt-oss-120b: good at summarization and the standard "
                                 "choice",
    (7, "MongoDB"): "schemaless, scales well, popular for AI apps",
    (9, "OpenAI"): "most reliable provider",
    (10, "gpt-4o"): "best general-purpose model",
}
REASON_SOURCE = "scripted demo reason (scripts/demo_dataset.py), not a real session"
SEEDED_PIVOT = {
    "change": "drop the verifiers (they slow things down)",
    "source": "seeded",
    "source_note": "seeded demo record (scripts/demo_dataset.py): the real Pivot Check did not "
                   "record this prompt",
}
traces: list[tuple[str, str]] = []
written: list[str] = []


def say(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 70 - len(title)), flush=True)


def note_traces(step: str, text: str) -> None:
    for t in dict.fromkeys(TRACE.findall(text)):
        if (step, t) not in traces:
            traces.append((step, t))
            print(f"[trace] {step}: {t}")


# --- what each step changed ----------------------------------------------------------------

def graph_state(p: Path) -> dict[int, tuple[str, str, str]]:
    from notbydefault.graph import Graph
    with Graph.for_project(p) as g:
        return {n["id"]: (n["type"], n["label"], hashlib.sha1(json.dumps(
            n["data"], sort_keys=True, default=str).encode()).hexdigest()) for n in g.nodes()}


def file_state(*roots: Path) -> dict[str, tuple[int, int]]:
    out = {}
    for root in roots:
        for f in root.rglob("*"):
            rel = f.relative_to(root).parts
            if not f.is_file() or ".git" in rel or "__pycache__" in rel or "cache" in rel:
                continue
            st = f.stat()
            out[str(f)] = (st.st_mtime_ns, st.st_size)
    return out


class Step:
    """Prints the graph ids a step added or changed and the files it wrote."""

    def __init__(self, p: Path, name: str):
        self.p, self.name = p, name

    def __enter__(self):
        say(self.name)
        self.g, self.f = graph_state(self.p), file_state(self.p, SHOTS)
        return self

    def __exit__(self, *exc):
        if exc[0]:
            return False
        g, f = graph_state(self.p), file_state(self.p, SHOTS)
        for nid, (kind, label, h) in g.items():
            if nid not in self.g:
                print(f"[graph] +#{nid} {kind} {label!r}")
            elif self.g[nid][2] != h:
                print(f"[graph] ~#{nid} {kind} {label!r}")
        for path in sorted(set(f) - set(self.f)) + sorted(
                x for x in f if x in self.f and f[x] != self.f[x]):
            rel = os.path.relpath(path, REPO)
            print(f"[file]  {rel}")
            written.append(rel)
        return False


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


# --- model calls ---------------------------------------------------------------------------

def providers(p: Path, offline: bool):
    """The project's ProviderSet, answering from the response cache; every call that would reach
    a provider is printed with its step, and refused when offline."""
    from notbydefault.fit.providers import ProviderSet, load_catalog
    from notbydefault.llm import OfflineCacheMiss

    class Logged(ProviderSet):
        step = ""

        def _transport(self, provider: str):        # only reached on a cache miss
            if offline:
                print(f"[live call skipped: --offline] step {self.step!r} -> {provider}")
                raise OfflineCacheMiss(f"--offline: no cached answer, live {provider} call skipped")
            print(f"[live call] step {self.step!r} -> {provider}")
            return super()._transport(provider)

    return Logged(load_catalog(p), cache_dir=p / ".notbydefault" / "cache",
                  env={} if offline else None)


# --- steps ---------------------------------------------------------------------------------

def _remove(path: Path) -> None:
    """rmtree that also removes git's read-only object files (Windows)."""
    import stat

    def writable(func, p, _exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    shutil.rmtree(path, onexc=writable)


def copy_demo(dest: Path, fresh: bool = False) -> Path:
    """Fresh copy; the previous build's model-response cache is carried over (unless `fresh`),
    so a rebuild answers the same prompts from it instead of calling a provider again."""
    src = REPO / "demo-app"
    if dest.resolve() == src.resolve():
        raise SystemExit("--out must not be demo-app itself")
    keep = None
    if dest.exists():
        old_cache = dest / ".notbydefault" / "cache"
        if old_cache.is_dir() and not fresh:
            keep = dest.parent / ".cache-carry"
            shutil.rmtree(keep, ignore_errors=True)
            shutil.copytree(old_cache, keep)
        _remove(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
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


def record_reasons(p: Path) -> list[int]:
    """Store REASONS as each decision's agent_reason (only where none is recorded)."""
    from notbydefault.graph import Graph
    done = []
    with Graph.for_project(p) as g:
        for (did, label), reason in REASONS.items():
            d = g.get_node(did)
            if not d or d["type"] != "decision" or d["label"] != label:
                print(f"  #{did} {label}: skipped (not that decision in this graph)")
                continue
            if d["data"].get("agent_reason"):
                print(f"  #{did} {label}: skipped (already has a reason)")
                continue
            g.update_data(did, agent_reason=reason, agent_reason_source=REASON_SOURCE)
            print(f"  #{did} {label}: \"{reason}\"")
            done.append(did)
    return done


def resolve(p: Path, did: int, client, apply: bool) -> None:
    from notbydefault.fit.tracing import make_tracer
    from notbydefault.resolve import Resolver, ResolveOptions
    client.step = f"resolve {did}{' --apply' if apply else ''}"
    lines: list[str] = []
    r = Resolver(p, did, ResolveOptions(apply=apply, yes=apply),
                 tracer=make_tracer(f"demo-resolve-{did}"),
                 ask=lambda q: "y", out=lines.append, client=client)
    code = r.run()
    print("\n".join(lines) + f"\n(exit {code})")
    if r.tracer.trace_id:
        traces.append((client.step, r.tracer.trace_id))
        print(f"[trace] {client.step}: {r.tracer.trace_id}")


def pivot(p: Path, client) -> None:
    """The real Pivot Check on PIVOT_PROMPT; one seeded pivot when it records nothing."""
    from notbydefault.fit.tracing import make_tracer
    from notbydefault.graph import Graph
    from notbydefault.hooks.user_prompt import handle
    from notbydefault.llm import LLMError
    from notbydefault.pivot import record
    client.step = "pivot"
    print(f'user types: "{PIVOT_PROMPT}"')
    tracer = make_tracer("demo-pivot")
    try:
        with tracer.root("notbydefault.pivot_check", {"notbydefault.prompt": PIVOT_PROMPT}):
            out, info = handle({"session_id": "demo-pivot", "hook_event_name": "UserPromptSubmit",
                                "prompt": PIVOT_PROMPT}, p, client=client, tracer=tracer)
    except LLMError as e:
        out, info = None, {"stage": f"confirm call failed ({e})"}
    tracer.flush()
    if tracer.trace_id:
        traces.append(("pivot check", tracer.trace_id))
        print(f"[trace] pivot check: {tracer.trace_id}")
    if out:
        print(out["hookSpecificOutput"]["additionalContext"])
        return
    print(f"real Pivot Check recorded nothing (stage: {info.get('stage')}); writing the seeded "
          "pivot, marked source=seeded")
    with Graph.for_project(p) as g:
        verifiers = next((n for n in g.nodes("evidence")
                          if n["label"] == "Verifiers for decision #21"), None)
        conflicts = [{"id": verifiers["id"], "type": "evidence", "label": verifiers["label"],
                      "why": "the fit check measured the verifier lifting "
                             "gemini-3.1-flash-lite from 11/12 to 12/12 correct (trace 4bcff389)"}
                     ] if verifiers else []
        imp = {"change": SEEDED_PIVOT["change"], "conflicts": conflicts,
               "decisions": [{"id": 21}]}
        pid = record(g, PIVOT_PROMPT, imp, "demo-pivot", {"model": None})
        g.update_data(pid, source=SEEDED_PIVOT["source"], source_note=SEEDED_PIVOT["source_note"])
    print(f"pivot #{pid}: {PIVOT_PROMPT} -> affects #21 (its verifiers)")


def agent_turn(p: Path) -> int | None:
    """The PreToolUse hook catches a new model choice; the reason comes from the scripted
    transcript, so the claim check has the agent's words to check."""
    from notbydefault.graph import Graph
    from notbydefault.hooks.pre_tool import handle
    print("Write digest.py (scripted transcript) -> PreToolUse hook")
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


def screenshots(p: Path, decision: int, claims_decision: int | None) -> int:
    cmd = [sys.executable, str(REPO / "scripts" / "ui_screenshots.py"), "--project", str(p),
           "--decision", str(decision)]
    if claims_decision:
        cmd += ["--claims-decision", str(claims_decision)]
    return subprocess.run(cmd, cwd=REPO).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(REPO / ".demo-dataset" / "demo-app"),
                    help="the copy to (re)build (default .demo-dataset/demo-app)")
    ap.add_argument("--decision", type=int, default=21, help="decision to resolve (default 21)")
    ap.add_argument("--offline", action="store_true",
                    help="no provider calls: answer from the response cache or fall back")
    ap.add_argument("--fresh", action="store_true",
                    help="don't reuse the previous build's model-response cache")
    ap.add_argument("--no-shots", action="store_true", help="don't regenerate docs/screenshots")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace", line_buffering=True)
    from notbydefault.claims import run as claims_run
    from notbydefault.env import load_env
    from notbydefault.scan import scan
    from notbydefault.stale import require
    from notbydefault.watch import watch
    load_env()

    p = copy_demo(Path(args.out).resolve(), args.fresh)
    say(f"copy ready: {p} (git repo on main)")
    client = providers(p, args.offline)
    with Step(p, "0. watch (baseline)"):
        captured("watch (baseline)", watch, p)
    with Step(p, "1. agent reasons (scripted, labelled as such)"):
        reasoned = record_reasons(p)
    with Step(p, f"2a. resolve {args.decision} (dry run)"):
        resolve(p, args.decision, client, apply=False)
    with Step(p, f"2b. resolve {args.decision} --apply (on the copy's branch; commit answered yes)"):
        resolve(p, args.decision, client, apply=True)
    with Step(p, "3. pivot"):
        pivot(p, client)
    with Step(p, "3b. agent turn: digest.py"):
        new_id = agent_turn(p)
    first = [args.decision] if args.decision in reasoned else []
    for did in first + [d for d in reasoned if d != args.decision] + [new_id] * bool(new_id):
        with Step(p, f"4. claims {did}"):
            client.step = f"claims {did}"
            captured(f"claims {did}", claims_run, p, did, client=client)
    with Step(p, f'5. require "{REQUIREMENT}"'):
        client.step = "require"
        captured("require", require, p, REQUIREMENT, client=client)
    with Step(p, "6. watch"):
        captured("watch", watch, p)
    with Step(p, "7. scan"):
        captured("scan", scan, p)
    shots = 0
    if not args.no_shots:
        with Step(p, "8. screenshots -> docs/screenshots (light + dark)"):
            shots = screenshots(p, args.decision, new_id)

    say("summary")
    print("trace ids created:")
    for step, t in traces:
        print(f"  {step:<28} {t}")
    if not traces:
        print("  none (telemetry off)")
    print(f"files written: {len(dict.fromkeys(written))}")
    print(f"provider calls made: {client.requests_made}, answered from cache: {client.cache_hits}")
    print(f"\nDataset: {p}\nUI:      python -m notbydefault ui --project \"{p}\"")
    if shots:
        print(f"screenshots failed (exit {shots})")
    return shots


if __name__ == "__main__":
    sys.exit(main())
