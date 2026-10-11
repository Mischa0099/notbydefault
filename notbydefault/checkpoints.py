"""`notbydefault checkpoints <id>`: the Entire checkpoint of the commit that added a record.

  1. the decision's record path (decisions/NNNN-<slug>.md)
  2. `git log --diff-filter=A` -> the full SHA of the commit that added it
  3. its checkpoint: the commit's `Entire-Checkpoint:` trailer, else
     `entire checkpoint explain --commit <sha> --json`
  4. the matching rows of `entire checkpoint list --json`, and the link
     https://entire.io/gh/<owner>/<repo>/commit/<FULL_SHA>
"""
from __future__ import annotations

import re
from pathlib import Path

from notbydefault.entire import Entire, commit_link, git, repo_slug


def record_commit(project: Path, record: str) -> str | None:
    """Full SHA of the commit that added `record` (oldest add wins), or None."""
    out = git(["log", "--diff-filter=A", "--format=%H", "--", record], project)
    shas = (out or "").split()
    return shas[-1] if shas else None


def checkpoint_of(project: Path, sha: str, entire: Entire) -> str | None:
    body = git(["log", "-1", "--format=%B", sha], project) or ""
    m = re.search(r"^Entire-Checkpoint:\s*(\S+)", body, re.M)
    if m:
        return m.group(1)
    info = entire.checkpoint_explain(sha, project, commit=True)
    return (info or {}).get("checkpoint_id")


def show(project: Path, decision_id: int, entire: Entire | None = None, out=print) -> int:
    from notbydefault.graph import Graph, db_path
    from notbydefault.records import existing_path
    project = Path(project).resolve()
    if not db_path(project).exists():
        out(f"No Decision Graph at {db_path(project)}.")
        return 1
    with Graph.for_project(project) as g:
        d = g.get_node(decision_id)
    if not d or d["type"] != "decision":
        out(f"No decision #{decision_id}")
        return 1
    path = existing_path(project, d)
    if path is None:
        out(f"Decision #{decision_id} has no decision record yet; run "
            f"`python -m notbydefault records {decision_id}` and commit it.")
        return 1
    rel = path.relative_to(project).as_posix()
    entire = entire or Entire()
    slug = repo_slug(project)
    sha = record_commit(project, rel)
    out(f"Decision #{decision_id} {d['label']}: record {rel}")
    if sha is None:
        out("  not committed yet: commit it, then run this again. Its checkpoint link will be "
            f"https://entire.io/gh/{slug or '<owner>/<repo>'}/commit/<FULL_SHA>")
        return 0
    out(f"  added in commit {sha}")
    cid = checkpoint_of(project, sha, entire)
    out(f"  checkpoint: {cid or 'none linked (' + (entire.last_error() or 'no Entire-Checkpoint trailer') + ')'}")
    rows = entire.checkpoint_list(project)
    if rows is None:
        out(f"  `entire checkpoint list` unavailable: {entire.last_error()}")
    else:
        mine = [r for r in rows if cid and r.get("checkpoint_id") == cid]
        out("  entire checkpoint list:" + ("" if mine else " (no matching row)"))
        for r in mine:
            out(f"    {r.get('checkpoint_id')}  {r.get('date', '')}  {r.get('agent', '')}  "
                f"\"{r.get('message', '')}\"  sessions: {', '.join(r.get('session_ids') or [r.get('session_id') or '-'])}")
    if slug:
        out(f"  link: {commit_link(slug, sha)}")
    else:
        out("  link: (origin is not a GitHub remote; format is "
            "https://entire.io/gh/<owner>/<repo>/commit/<FULL_SHA>)")
    return 0
