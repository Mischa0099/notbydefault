"""Thin wrapper around the Entire CLI (entire 0.11, entire-graph plugin 0.4).

Only commands documented by `entire agent-help` / `entire graph <cmd> --help` are used:
  entire graph symbols --repo R --format ndjson --worktree
  entire graph edges   --repo R --format ndjson --worktree [--relation T]
  entire graph search  --repo R --query Q --format json --top-k N
  entire search Q --json --compact --limit N            (needs login + a mirrored repo)
  entire checkpoint list --json
  entire checkpoint explain <id|sha> --json / --commit <sha> --json / <id> --transcript
  entire plugin list
Every call degrades gracefully: on a missing binary, a missing plugin, a timeout or an error it
returns None (or []) and records why in `self.errors`, so callers can fall back and say so.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

DEFAULT_TIMEOUT_S = 60
GRAPH_TIMEOUT_S = 30   # a graph read that takes longer is treated as unavailable


class Entire:
    def __init__(self, exe: str | None = None, runner=None, timeout: float = DEFAULT_TIMEOUT_S):
        self.exe = exe or shutil.which("entire") or "entire"
        self.runner = runner or subprocess.run
        self.timeout = timeout
        self.errors: list[str] = []
        self._plugins: dict[str, str] | None = None

    # --- plumbing ---------------------------------------------------------------------

    def run(self, args: list[str], cwd: Path | str | None = None,
            timeout: float | None = None) -> str | None:
        """stdout of `entire <args>`, or None (with the reason in self.errors)."""
        import os
        if os.environ.get("NOTBYDEFAULT_ENTIRE") == "0" and self.runner is subprocess.run:
            self.errors.append("Entire integration disabled (NOTBYDEFAULT_ENTIRE=0)")
            return None
        try:
            proc = self.runner([self.exe, *args], cwd=str(cwd) if cwd else None,
                               capture_output=True, timeout=timeout or self.timeout)
        except FileNotFoundError:
            self.errors.append("entire CLI is not installed or not on PATH")
            return None
        except subprocess.TimeoutExpired:
            self.errors.append(f"`entire {' '.join(args[:2])}` timed out after "
                               f"{timeout or self.timeout:.0f}s")
            return None
        except OSError as e:
            self.errors.append(f"`entire {' '.join(args[:2])}` failed: {e}")
            return None
        out = _text(proc.stdout)
        if proc.returncode != 0:
            msg = (_text(proc.stderr) or out).strip().splitlines()
            self.errors.append(f"`entire {' '.join(args[:2])}` exited {proc.returncode}: "
                               f"{msg[-1] if msg else 'no output'}")
            return None
        return out

    def run_json(self, args: list[str], cwd=None, timeout=None):
        out = self.run(args, cwd, timeout)
        if out is None:
            return None
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            self.errors.append(f"`entire {' '.join(args[:2])}` returned non-JSON output")
            return None

    def run_ndjson(self, args: list[str], cwd=None, timeout=None) -> list[dict] | None:
        out = self.run(args, cwd, timeout)
        if out is None:
            return None
        rows = []
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def last_error(self) -> str:
        return self.errors[-1] if self.errors else ""

    # --- availability -----------------------------------------------------------------

    def plugins(self) -> dict[str, str]:
        """{name: version} from `entire plugin list` (empty when entire is unavailable)."""
        if self._plugins is None:
            out = self.run(["plugin", "list"], timeout=20) or ""
            found = {}
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[1].startswith("v") and ("entire-" + parts[0]) in line:
                    found[parts[0]] = parts[1]
            self._plugins = found
        return self._plugins

    def has_plugin(self, name: str) -> bool:
        if name in self.plugins():
            return True
        return shutil.which(f"entire-{name}") is not None   # unmanaged plugins on PATH

    # --- graph ------------------------------------------------------------------------

    def graph_symbols(self, repo: Path) -> list[dict] | None:
        rows = self.run_ndjson(["graph", "symbols", "--repo", str(repo), "--format", "ndjson",
                                "--worktree"], cwd=repo, timeout=GRAPH_TIMEOUT_S)
        return None if rows is None else [r for r in rows if r.get("record_type") == "symbol"]

    def graph_edges(self, repo: Path, relations: str | None = None) -> list[dict] | None:
        args = ["graph", "edges", "--repo", str(repo), "--format", "ndjson", "--worktree"]
        if relations:
            args += ["--relation", relations]
        rows = self.run_ndjson(args, cwd=repo, timeout=GRAPH_TIMEOUT_S)
        return None if rows is None else [r for r in rows if r.get("record_type") == "relation"]

    def graph_search(self, repo: Path, query: str, top_k: int = 5) -> list[dict] | None:
        data = self.run_json(["graph", "search", "--repo", str(repo), "--query", query,
                              "--format", "json", "--top-k", str(top_k)], cwd=repo)
        if data is None:
            return None
        if isinstance(data, dict):
            for key in ("results", "hits", "candidates"):
                if isinstance(data.get(key), list):
                    return data[key]
            return []
        return data if isinstance(data, list) else []

    # --- sessions, checkpoints, search ------------------------------------------------

    def search(self, query: str, cwd: Path, limit: int = 5) -> list[dict] | None:
        data = self.run_json(["search", query, "--json", "--compact", "--limit", str(limit)],
                             cwd=cwd, timeout=30)
        if data is None:
            return None
        if isinstance(data, dict):
            data = data.get("results") or data.get("hits") or []
        return data if isinstance(data, list) else []

    def checkpoint_list(self, cwd: Path) -> list[dict] | None:
        data = self.run_json(["checkpoint", "list", "--json"], cwd=cwd, timeout=30)
        return data if isinstance(data, list) else (None if data is None else [])

    def checkpoint_explain(self, target: str, cwd: Path, commit: bool = False) -> dict | None:
        args = ["checkpoint", "explain"] + (["--commit", target] if commit else [target]) + ["--json"]
        data = self.run_json(args, cwd=cwd, timeout=30)
        return data if isinstance(data, dict) else None

    def checkpoint_transcript(self, checkpoint_id: str, cwd: Path) -> list[dict] | None:
        return self.run_ndjson(["checkpoint", "explain", checkpoint_id, "--transcript"],
                               cwd=cwd, timeout=60)


def _text(b) -> str:
    if b is None:
        return ""
    return b.decode("utf-8", errors="replace") if isinstance(b, bytes) else str(b)


def git(args: list[str], cwd: Path, runner=None) -> str | None:
    """stdout of a git command, or None."""
    try:
        proc = (runner or subprocess.run)(["git", *args], cwd=str(cwd), capture_output=True,
                                          timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return _text(proc.stdout).strip() if proc.returncode == 0 else None


def repo_root(project: Path, runner=None) -> Path | None:
    out = git(["rev-parse", "--show-toplevel"], project, runner)
    return Path(out).resolve() if out else None


def repo_slug(project: Path, runner=None) -> str | None:
    """'owner/name' of the GitHub origin, for entire.io links."""
    import re
    url = git(["remote", "get-url", "origin"], project, runner) or ""
    m = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?$", url)
    return m.group(1) if m else None


def commit_link(slug: str, sha: str) -> str:
    return f"https://entire.io/gh/{slug}/commit/{sha}"


def recent_commits(project: Path, n: int = 300, runner=None) -> list[dict]:
    """[{sha, subject}] touching the project, newest first; [] when the enclosing repo is the
    home directory (logging it is slow and unrelated) or the integration is disabled."""
    import os
    if os.environ.get("NOTBYDEFAULT_ENTIRE") == "0" and runner is None:
        return []
    project = Path(project).resolve()
    root = repo_root(project, runner)
    if root is None or (root == Path.home().resolve() and project != root):
        return []
    log = git(["log", "-n", str(n), "--format=%H%x09%s", "--", "."], project, runner) or ""
    return [{"sha": l.split("\t", 1)[0], "subject": l.split("\t", 1)[1]}
            for l in log.splitlines() if "\t" in l]
