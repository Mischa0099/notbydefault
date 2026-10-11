"""Briefing: what the project already knows about a decision the hook just caught.

Four kinds of line, each with a source id:
  past decision        related decisions in the Decision Graph (same choice or category)
  rejected before      alternatives a past check rejected, with the reason
  history              Entire checkpoints/sessions/commits about it (Entire search when the repo
                       is mirrored and you are logged in; otherwise local checkpoints + git log)
  reuse                existing code that already uses the service/model (Entire Graph)

The PreToolUse hook must stay fast, so it reads only the graph and a cache file
(.notbydefault/entire_cache.json) and shows at most HOOK_LINES lines; `refresh_cache` (run
detached after the hook, and live by `check`) does the slow Entire calls.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

HOOK_LINES = 4
CACHE_NAME = "entire_cache.json"
MAX_CHECKPOINTS = 30
TEXT_CHARS = 110


def cache_path(project: Path) -> Path:
    return Path(project) / ".notbydefault" / CACHE_NAME


def load_cache(project: Path) -> dict:
    try:
        return json.loads(cache_path(project).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _short(text, n: int = TEXT_CHARS) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n - 3] + "..."


def terms(decision: dict) -> list[str]:
    """Words that identify the choice: 'openai/gpt-oss-120b' -> gpt-oss-120b, gpt-oss, ..."""
    label = decision["label"]
    out = {label.lower(), label.split("/")[-1].lower()}
    base = re.sub(r"[-_.]?\d[\w.]*$", "", label.split("/")[-1]).lower()
    if len(base) >= 4:
        out.add(base)
    out |= {p.lower() for p in str(decision["data"].get("evidence") or "").split(",") if p.strip()}
    return sorted((t for t in out if len(t) >= 3), key=len, reverse=True)


def _mentions(text: str, words: list[str]) -> bool:
    low = (text or "").lower()
    return any(re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", low) for w in words)


# --- lines -----------------------------------------------------------------------------

def past_decisions(graph, decision: dict) -> list[str]:
    from notbydefault.records import why_text
    data = decision["data"]
    words = terms(decision)
    related = [d for d in reversed(graph.nodes("decision")) if d["id"] != decision["id"]
               and (d["label"].lower() == decision["label"].lower()
                    or d["data"].get("category") == data.get("category")
                    or _mentions(d["label"], words))]
    best: dict[str, dict] = {}     # one line per choice: prefer the one with a record
    for d in related:
        key = d["label"].lower()
        if key not in best or (d["data"].get("decision_record")
                               and not best[key]["data"].get("decision_record")):
            best[key] = d
    lines = []
    for d in best.values():
        rec = d["data"].get("decision_record")
        lines.append(f"Past decision #{d['id']} {d['label']} ({d['data'].get('status')}): "
                     f"{_short(why_text(d), 90)} [decision #{d['id']}"
                     + (f", {rec}" if rec else "") + "]")
    return lines


def rejected_before(graph, decision: dict) -> list[str]:
    words = terms(decision)
    category = decision["data"].get("category")
    by_alt: dict[int, list[dict]] = {}
    for e in graph.edges():
        if e["type"] == "rejects":
            by_alt.setdefault(e["dst"], []).append(graph.get_node(e["src"]))
    same, near = [], []
    for alt in graph.nodes("alternative"):
        owners = [o for o in by_alt.get(alt["id"], []) if o]
        line = (f"Rejected before: `{alt['label']}`"
                + (f" for #{owners[0]['id']} {owners[0]['label']}" if owners else "")
                + f": {_short(alt['data'].get('reason') or 'no reason recorded', 80)} "
                  f"[alternative #{alt['id']}]")
        if _mentions(alt["label"], words):
            same.append(line)            # the agent is picking something rejected earlier
        elif any(o["data"].get("category") == category for o in owners):
            near.append(line)
    return same + near


def history(cache: dict, decision: dict, files: list[str]) -> list[str]:
    words = terms(decision)
    lines = []
    for hit in (cache.get("search") or {}).get(decision["label"].lower(), []):
        lines.append(f"History: {hit.get('type', 'hit')} {hit['id']} "
                     f"\"{_short(hit.get('title'), 70)}\" [entire search {hit['id']}]")
    for cp in cache.get("checkpoints") or []:
        touched = [f for f in cp.get("files") or [] if f in files]
        if _mentions(cp.get("message", ""), words) or touched:
            what = f"touched {', '.join(touched[:2])}" if touched else "mentions it"
            lines.append(f"History: checkpoint {cp['id'][:12]} \"{_short(cp.get('message'), 60)}\""
                         f" {what} [checkpoint {cp['id']}"
                         + (f", session {cp['sessions'][0]}" if cp.get("sessions") else "") + "]")
    for c in cache.get("commits") or []:
        if _mentions(c.get("subject", ""), words):
            lines.append(f"History: commit {c['sha'][:7]} \"{_short(c['subject'], 70)}\" "
                         f"[commit {c['sha']}]")
    return lines


def reuse(graph, cache: dict, decision: dict) -> list[str]:
    """Existing functions already using this model/service (or its SDK)."""
    from notbydefault.affected import describe, markers
    lines, seen = [], set()
    for d in graph.nodes("decision"):
        if d["id"] == decision["id"] or not d["data"].get("affected_source") == "entire-graph":
            continue
        same = (d["label"].lower() == decision["label"].lower()
                or (decision["data"].get("kind") == "model" and d["data"].get("kind") == "service"
                    and d["data"].get("category") == "ai-provider"))
        if not same:
            continue
        for e in d["data"].get("affected_code") or []:
            if e.get("symbol") and e.get("depth") == 0 and describe(e) not in seen:
                seen.add(describe(e))
                lines.append(f"Reuse: {describe(e)} already {e['via']} "
                             f"[entire-graph {e['path']}:{e['line']}]")
    pkgs = markers(decision)["packages"]
    for pkg, users in (cache.get("sdk_users") or {}).items():
        if pkg in pkgs:
            for u in users:
                if u["where"] not in seen:
                    seen.add(u["where"])
                    lines.append(f"Reuse: {u['where']} already {u['via']} [entire-graph {u['loc']}]")
    return lines


def build(project: Path, graph, decision: dict, limit: int | None = HOOK_LINES) -> list[str]:
    """Briefing lines; with `limit`, one line per kind first, then fill up to the limit."""
    from notbydefault.graph import find_affected_code
    cache = load_cache(project)
    groups = [past_decisions(graph, decision), rejected_before(graph, decision),
              history(cache, decision, find_affected_code(decision)),
              reuse(graph, cache, decision)]
    if limit is None:
        return [line for g in groups for line in g]
    out = [g[0] for g in groups if g][:limit]
    for g in groups:
        for line in g[1:]:
            if len(out) >= limit:
                return out
            out.append(line)
    return out


# --- the cache (slow; run detached by the hook, live by `check`) ------------------------

def sdk_users(index) -> dict[str, list[dict]]:
    """Known SDK package -> functions calling it, from the graph's external CALLS edges."""
    from notbydefault.detect import KNOWN_SERVICES
    from notbydefault.affected import IMPORT_NAMES, _pkg_match
    pkgs = sorted({IMPORT_NAMES.get(p, p.replace("-", "_")) for p in KNOWN_SERVICES} | set(KNOWN_SERVICES))
    out: dict[str, list[dict]] = {}
    for e in index.calls:
        if not e["to_id"].startswith("external:symbol:"):
            continue
        target = e["to_id"].split("external:symbol:", 1)[1]
        pkg = _pkg_match(target, pkgs)
        sym = index.symbols.get(e["from_id"])
        if pkg and sym:
            where = f"{sym['path']}:{sym['start_line']} {sym['qualified_name']}"
            users = out.setdefault(pkg, [])
            if all(u["where"] != where for u in users):
                users.append({"where": where, "via": f"calls {target}",
                              "loc": f"{sym['path']}:{sym['start_line']}"})
    return out


def refresh_cache(project: Path, decision_ids: list[int] | None = None, entire=None,
                  tracer=None, search: bool = True) -> dict:
    """Affected code for the decisions, SDK users, checkpoints, search hits -> cache file."""
    from notbydefault.affected import Index, analyze, record_affected
    from notbydefault.entire import Entire, recent_commits, repo_root
    from notbydefault.graph import Graph
    project = Path(project).resolve()
    entire = entire or Entire()
    old = load_cache(project)
    cache = {"updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "errors": {}}

    def recover(key: str, why: str) -> None:
        cache["errors"][key] = why
        if tracer is not None:
            tracer.event(f"{key}_fallback", "recovery", why)

    index = Index(project, entire)
    with Graph.for_project(project) as g:
        decisions = [d for d in g.nodes("decision")
                     if decision_ids is None or d["id"] in decision_ids]
        for d in decisions:
            record_affected(g, d["id"], analyze(index, d), tracer)
    if index.ok:
        cache["sdk_users"] = sdk_users(index)
    else:
        cache["sdk_users"] = old.get("sdk_users") or {}
        recover("graph", f"entire-graph unavailable ({index.error}); reuse hints use the last "
                         "cached graph, if any")

    root = repo_root(project) if os.environ.get("NOTBYDEFAULT_ENTIRE") != "0" else None
    prefix = ""
    if root is not None and root != project:
        try:
            prefix = project.relative_to(root).as_posix() + "/"
        except ValueError:
            prefix = ""
    known = {c["id"]: c for c in old.get("checkpoints") or []}
    listed = entire.checkpoint_list(project)
    if listed is None:
        cache["checkpoints"] = list(known.values())
        recover("checkpoints", f"`entire checkpoint list` unavailable ({entire.last_error()}); "
                               "using cached checkpoints")
    else:
        cps = []
        for c in listed[:MAX_CHECKPOINTS]:
            cid = c.get("checkpoint_id")
            if not cid:
                continue
            if cid in known and known[cid].get("files") is not None:
                cps.append(known[cid])
                continue
            info = entire.checkpoint_explain(cid, project) or {}
            files = [f[len(prefix):] for f in info.get("files_touched") or []
                     if f.startswith(prefix)]
            cps.append({"id": cid, "message": c.get("message", ""), "date": c.get("date"),
                        "sessions": c.get("session_ids") or [c.get("session_id")],
                        "files": files if info else None})
        cache["checkpoints"] = cps

    cache["search"] = dict(old.get("search") or {})
    if search:
        with Graph.for_project(project) as g:
            labels = sorted({d["label"] for d in g.nodes("decision")
                             if decision_ids is None or d["id"] in decision_ids})
        for label in labels:
            hits = entire.search(label, project, limit=3)
            if hits is None:
                recover("search", f"`entire search` unavailable ({entire.last_error()}); "
                                  "using local checkpoints and git history instead")
                break
            cache["search"][label.lower()] = [
                {"id": str(h.get("id")), "type": h.get("type", "hit"),
                 "title": h.get("title") or h.get("snippet") or ""} for h in hits if h.get("id")]
    cache["commits"] = recent_commits(project, 200)
    path = cache_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return cache


def main(argv: list[str]) -> int:
    """python -m notbydefault.briefing <project> [decision ids...]  (detached by the hook)."""
    from notbydefault.hooks.pre_tool import _log
    project = Path(argv[0])
    ids = [int(a) for a in argv[1:]] or None
    started = time.perf_counter()
    try:
        cache = refresh_cache(project, ids)
        _log(project, f"briefing cache refreshed in {time.perf_counter() - started:.1f}s"
                      + (f" (fallbacks: {cache['errors']})" if cache["errors"] else ""))
    except Exception as e:
        _log(project, f"briefing cache refresh failed: {e!r}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
