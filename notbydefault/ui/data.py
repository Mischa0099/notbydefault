"""Everything the UI shows, read from the project's own files. No writes, no model calls.

The Decision Graph is opened read-only (sqlite `mode=ro`) for one short read per request and
closed straight away, so a hook writing from another process is never blocked for long.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from notbydefault.graph import Graph, db_path

STATUSES = ("unreviewed", "reviewed", "needs_review", "stale")
NODE_TYPES = ("goal", "requirement", "constraint", "decision", "evidence", "alternative",
              "code", "pivot")
TRACE_HOST = (os.environ.get("NEATLOGS_HOST") or "https://app.neatlogs.com").rstrip("/")
ACTIVITY_LOG = "ui-activity.jsonl"
VIEWABLE = {".md", ".json", ".jsonl", ".py", ".txt", ".yaml", ".yml", ".csv", ".patch",
            ".diff", ".log"}
SECRET_NAME = re.compile(r"(^\.env|secret|token|credential|password|^id_rsa|\.pem$|\.key$)",
                         re.IGNORECASE)
MAX_VIEW_BYTES = 512 * 1024


class ReadGraph(Graph):
    """The Decision Graph opened read-only: never creates the file or its tables."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.conn = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True,
                                    timeout=2.0)
        self.conn.row_factory = sqlite3.Row


@dataclass
class Snapshot:
    """All nodes and edges, read in one short transaction."""
    nodes: dict[int, dict] = field(default_factory=dict)
    edges: list[dict] = field(default_factory=list)
    exists: bool = False

    def of_type(self, type: str) -> list[dict]:
        return [n for n in self.nodes.values() if n["type"] == type]

    def nodes_list(self, type: str | None = None) -> list[dict]:   # costs.decision_rows API
        return self.of_type(type) if type else list(self.nodes.values())

    def out(self, nid: int, *types: str) -> list[dict]:
        return [dict(self.nodes[e["dst"]], edge_type=e["type"]) for e in self.edges
                if e["src"] == nid and (not types or e["type"] in types) and e["dst"] in self.nodes]

    def into(self, nid: int, *types: str) -> list[dict]:
        return [dict(self.nodes[e["src"]], edge_type=e["type"]) for e in self.edges
                if e["dst"] == nid and (not types or e["type"] in types) and e["src"] in self.nodes]

    @property
    def max_id(self) -> int:
        return max(self.nodes, default=0)


class _GraphView:
    """What costs.decision_rows needs from a Graph (`nodes(type)`), backed by a Snapshot."""

    def __init__(self, snap: Snapshot):
        self.snap = snap

    def nodes(self, type: str | None = None) -> list[dict]:
        return self.snap.nodes_list(type)

    def get_node(self, nid: int) -> dict | None:
        return self.snap.nodes.get(nid)


def snapshot(project: Path) -> Snapshot:
    path = db_path(project)
    if not path.exists():
        return Snapshot()
    g = ReadGraph(path)
    try:
        nodes = {n["id"]: n for n in g.nodes()}
        edges = g.edges()
    except sqlite3.Error:
        return Snapshot()
    finally:
        g.close()
    return Snapshot(nodes, edges, True)


def version(project: Path) -> str:
    """Changes whenever anything the UI shows may have changed (cheap: file stats only)."""
    nbd = Path(project) / ".notbydefault"
    parts = []
    for name in ("graph.db", "graph.db-journal", "graph.db-wal", "hook.log", ACTIVITY_LOG,
                 "results", "resolve"):
        try:
            st = (nbd / name).stat()
            parts.append(f"{st.st_mtime_ns}:{st.st_size}")
        except OSError:
            parts.append("-")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


# --- files --------------------------------------------------------------------------------

def safe_path(project: Path, rel: str) -> Path | None:
    """`rel` inside the project, viewable and not secret-looking; else None."""
    if not rel:
        return None
    root = Path(project).resolve()
    try:
        p = (root / rel).resolve()
        p.relative_to(root)
    except (ValueError, OSError):
        return None
    parts = p.relative_to(root).parts
    if any(SECRET_NAME.search(x) or x in (".git", "cache") for x in parts):
        return None
    if p.suffix.lower() not in VIEWABLE or not p.is_file():
        return None
    return p


def rel_to(project: Path, path: str | None) -> str | None:
    """A stored path (absolute or relative) as a project-relative posix path, if inside."""
    if not path:
        return None
    p = Path(path)
    root = Path(project).resolve()
    try:
        return (p if p.is_absolute() else root / p).resolve().relative_to(root).as_posix()
    except (ValueError, OSError):
        return None


_json_cache: dict[Path, tuple[int, dict]] = {}


def load_json(project: Path, rel: str | None) -> dict | None:
    p = safe_path(project, rel or "")
    if p is None or p.suffix != ".json":
        return None
    mtime = p.stat().st_mtime_ns
    hit = _json_cache.get(p)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    _json_cache[p] = (mtime, data)
    return data


def parse_time(value) -> datetime | None:
    if not value:
        return None
    try:
        t = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.astimezone()   # naive = this machine's local time


def trace_url(trace_id: str | None) -> str | None:
    return f"{TRACE_HOST}/api/v1/public/traces/{trace_id}" if trace_id else None


# --- decisions ----------------------------------------------------------------------------

def status_of(node: dict) -> str:
    return (node.get("data") or {}).get("status") or "unreviewed"


def by_default(d: dict) -> bool:
    """Chosen by default: an agent's tool call picked it and no one recorded why."""
    data = d["data"]
    return bool(data.get("tool_name")) and not data.get("reason") \
        and not data.get("verdict_summary")


def last_evidence(snap: Snapshot, d: dict) -> str:
    data = d["data"]
    from notbydefault.records import status_line
    line = status_line(data)
    if line:
        return line
    ev = sorted(snap.out(d["id"], "supported_by"), key=lambda n: n["id"])
    fit = [e for e in ev if e["data"].get("kind") == "fit_test"]
    if fit:
        return f"{fit[-1]['label']}, verdict: {data.get('verdict') or '-'}"
    if data.get("reason"):
        return f"reason given by a person ({str(data.get('reason_at'))[:10]})"
    how = f" ({data['evidence']})" if data.get("evidence") else ""
    return f"detected from a {data.get('tool_name') or 'tool'} call{how}"


def decision_rows(snap: Snapshot, status: str = "", kind: str = "") -> list[dict]:
    rows = []
    for d in snap.of_type("decision"):
        st = status_of(d)
        if (status and st != status) or (kind and d["data"].get("kind") != kind):
            continue
        rows.append({"d": d, "status": st, "by_default": by_default(d),
                     "evidence": last_evidence(snap, d)})
    return rows


# --- fit check ----------------------------------------------------------------------------

def fit_table(results: dict) -> dict:
    """The results table exactly as the CLI report prints it (fit.report.table_rows)."""
    from notbydefault.fit.report import table_rows
    from notbydefault.fit.stats import SetupStats
    names = {f.name for f in dataclasses.fields(SetupStats)}
    stats = {}
    for raw in (results.get("stats") or {}).values():
        s = SetupStats(**{k: v for k, v in raw.items() if k in names})
        stats[s.key] = s
    cfg = results.get("config") or {}
    ctx = SimpleNamespace(chosen=cfg.get("chosen"),
                          judges=cfg.get("judges") or ([cfg["judge"]] if cfg.get("judge") else []))
    headers, rows = table_rows(stats, ctx)
    v = results.get("verdict") or {}
    rec, best = tuple(v.get("recommended") or ()), tuple(v.get("best_quality") or ())
    marked = []
    for s, cells in zip(stats.values(), rows):
        marked.append({"cells": cells, "current": (s.model, s.setup) == (ctx.chosen, "alone"),
                       "recommended": s.key == rec, "best": s.key == best,
                       "name": s.name, "failed_cases": s.failed_cases})
    return {"headers": headers, "rows": marked, "refs": any(s.n_ref for s in stats.values())}


def record_sections(text: str, *titles: str) -> dict[str, list[str]]:
    """Bullet lines under `## <title>` headings of a decision record."""
    out: dict[str, list[str]] = {t: [] for t in titles}
    current = None
    for line in text.splitlines():
        if line.startswith("## "):
            current = line[3:].strip() if line[3:].strip() in out else None
        elif current and line.startswith("- "):
            out[current].append(line[2:].strip())
    return out


_git_cache: dict[tuple, tuple[float, object]] = {}


def _cached(key: tuple, ttl: float, fn):
    hit = _git_cache.get(key)
    if hit and time.monotonic() - hit[0] < ttl:
        return hit[1]
    value = fn()
    _git_cache[key] = (time.monotonic(), value)
    return value


def record_links(project: Path, rel: str) -> dict:
    """The commit that added a decision record, its Entire checkpoint and link (git only)."""
    from notbydefault.checkpoints import record_commit
    from notbydefault.entire import commit_link, git, repo_slug

    def compute():
        sha = record_commit(project, rel)
        if not sha:
            return {"sha": None, "checkpoint": None, "link": None}
        body = git(["log", "-1", "--format=%B", sha], project) or ""
        m = re.search(r"^Entire-Checkpoint:\s*(\S+)", body, re.M)
        slug = repo_slug(project)
        return {"sha": sha, "checkpoint": m.group(1) if m else None,
                "link": commit_link(slug, sha) if slug else None}
    try:
        mtime = (Path(project) / rel).stat().st_mtime_ns
    except OSError:
        mtime = 0
    return _cached(("record", str(project), rel, mtime), 60.0, compute)


# the 8 focus areas: (area, what it covers, spans/events tagged with it, decision-page section
# where it shows most). tests/test_ui.py checks each span name is emitted with its area.
FOCUS_MAP = [
    ("planning", "the plan step", ("plan", "recommend", "latency_requirement",
                                   "blocked_by_requirement"), "fit-steps"),
    ("recovery", "swaps and retries on 429", ("swap_candidate", "drop_candidate",
                                              "rate_limit_backoff", "retry_failed_call",
                                              "retry_step"), "journal-recovery"),
    ("context", "briefing and call-site analysis", ("briefing", "context", "investigate",
                                                    "affected_code"), "affected"),
    ("execution", "the run, resumable and cached", ("execute", "refresh_cache", "throttle_wait"),
     "journal-execution"),
    ("verification", "verifiers, judges and confidence intervals",
     ("verifiers", "judge", "decide", "verdict", "test", "verify"), "fit-evidence"),
    ("iteration", "the same case re-run with feedback; the failure bank",
     ("retry_with_feedback", "expand_sample"), "failure-bank"),
    ("tool_use", "picking candidates, judges and verifiers",
     ("candidate_selected", "judge_selected", "verifier_model_selected", "preflight"),
     "journal-tool_use"),
    ("state", "graph writes between steps", ("record", "decision_record", "branch", "commit"),
     "history"),
]


def journal_view(journal: list[dict]) -> tuple[list[dict], list[dict]]:
    """(steps, entries) of a saved agent journal, each with today's focus area and label."""
    from notbydefault.fit.tracing import FOCUS_LABELS, current_focus
    entries, seen = [], set()
    for e in journal:
        area = current_focus(e)
        entries.append({"kind": e.get("kind"), "name": e.get("name"), "area": area,
                        "label": FOCUS_LABELS.get(area, area), "reason": e.get("reason") or "",
                        "recorded": e.get("focus_area"), "anchor": area not in seen})
        seen.add(area)
    steps: list[dict] = []
    for e in entries:
        if e["kind"] != "step" or e["name"] == "notbydefault.check":
            continue
        if steps and steps[-1]["name"] == e["name"]:
            steps[-1]["n"] += 1
        else:
            steps.append({**e, "n": 1})
    return steps, entries


def decision_view(project: Path, snap: Snapshot, did: int) -> dict | None:
    """Everything the receipt page shows for one decision, each claim with its source."""
    from notbydefault.fit import failures as failures_mod
    from notbydefault.graph import affected_entries
    from notbydefault.records import existing_path, why_text
    d = snap.nodes.get(did)
    if d is None or d["type"] != "decision":
        return None
    project = Path(project)
    data = d["data"]
    results_rel = rel_to(project, data.get("results"))
    results = load_json(project, results_rel)
    fit = None
    if results and results.get("stats"):
        v = results.get("verdict") or {}
        cfg = results.get("config") or {}
        callsite = results.get("callsite") or {}
        fit = {"table": fit_table(results), "verdict": v, "run_at": results.get("run_at"),
               "trace_id": results.get("trace_id"), "trace_url": trace_url(results.get("trace_id")),
               "spent": results.get("spent"), "requests": results.get("requests") or {},
               "assumptions": results.get("assumptions") or {},
               "free_tier": (results.get("pricing") or {}).get("free_tier"),
               "pricing_source": (results.get("pricing") or {}).get("source"),
               "judges": cfg.get("judges") or [cfg.get("judge")],
               "agreement": results.get("judge_agreement"), "results": results_rel,
               "callsite": {"file": callsite.get("file"), "line": callsite.get("line"),
                            "task": callsite.get("task")},
               "verifiers": rel_to(project, (results.get("verifiers") or {}).get("path"))
               or (f".notbydefault/verifiers/{did}.py"
                   if (project / ".notbydefault" / "verifiers" / f"{did}.py").is_file() else None),
               "checks": (results.get("verifiers") or {}).get("checks") or [],
               "stopped": results.get("stopped")}
        fit["steps"], fit["journal"] = journal_view(results.get("journal") or [])
        from notbydefault.fit.report import net_saving_line
        by_setup = {(s.get("model"), s.get("setup")): s for s in results["stats"].values()}
        cur, rec = (by_setup.get(tuple(v.get(k) or ())) or {} for k in ("reference", "recommended"))
        fit["net_saving"] = net_saving_line(cur.get("monthly_cost"), rec.get("monthly_cost"),
                                            results.get("spent")) if rec is not cur else ""
    record = existing_path(project, d)
    record_rel = rel_to(project, str(record)) if record else None
    sections = {"Caveats": [], "Assumptions and evidence": []}
    if record:
        sections = record_sections(record.read_text(encoding="utf-8"),
                                   "Caveats", "Assumptions and evidence")
    caveats = list(sections["Caveats"])
    if data.get("status") == "needs_review":
        caveats.insert(0, f"Needs review ({data.get('review_basis') or 'judge'}): "
                          f"{data.get('review_reason')}")
    assumptions = [n for n in snap.out(did, "depends_on") if n["data"].get("assumption")]
    cause_id = data.get("stale_cause") if data.get("status") == "stale" else \
        data.get("review_cause") if data.get("status") == "needs_review" else None
    pivots = [p for p in snap.of_type("pivot") if did in (p["data"].get("decisions") or [])]
    resolve_dir = project / ".notbydefault" / "resolve"
    prs = sorted(resolve_dir.glob(f"{did}-*-pr.md")) if resolve_dir.is_dir() else []
    failed_patch = resolve_dir / f"{did}-failed.patch"
    bank = failures_mod.load(project, did)
    return {
        "d": d, "data": data, "status": status_of(d), "by_default": by_default(d),
        "why": why_text(d), "fit": fit,
        "affected": affected_entries(d), "affected_source": data.get("affected_source"),
        "affected_note": data.get("affected_note"),
        "evidence": snap.out(did, "supported_by"),
        "rejected": snap.out(did, "rejects"),
        "depends": snap.out(did, "depends_on", "serves"),
        "code": snap.out(did, "affects"),
        "caveats": caveats, "assumption_notes": sections["Assumptions and evidence"],
        "assumptions": assumptions,
        "cause": snap.nodes.get(cause_id) if cause_id else None,
        "pivots": pivots,
        "failures": bank[-20:], "failures_total": len(bank),
        "failures_file": rel_to(project, str(failures_mod.bank_path(project, did))) if bank else None,
        "record": record_rel,
        "record_links": record_links(project, record_rel) if record_rel else None,
        "resolution": data.get("resolution"),
        "pr": prs[-1].read_text(encoding="utf-8") if prs else None,
        "pr_file": rel_to(project, str(prs[-1])) if prs else None,
        "failed_patch": rel_to(project, str(failed_patch)) if failed_patch.is_file() else None,
        "history": list(reversed(data.get("history") or [])),
        "resolve_diff": resolve_diff(project, data.get("resolution")),
        "claims": data.get("claims"), "watched": data.get("assumptions"),
    }


def resolve_diff(project: Path, res: dict | None) -> str:
    """What a resolve changed on its branch: working tree if checked out, else main...branch."""
    from notbydefault.entire import git
    if not res or not res.get("branch") or not res.get("files"):
        return ""

    def compute():
        branch = git(["rev-parse", "--abbrev-ref", "HEAD"], project)
        if branch == res["branch"]:
            out = git(["diff", "HEAD", "--", *res["files"]], project)
            committed = git(["diff", "HEAD~1", "HEAD", "--", *res["files"]], project)
            return out or committed or ""
        return git(["diff", f"HEAD...{res['branch']}", "--", *res["files"]], project) or ""
    return _cached(("diff", str(project), res["branch"], version(project)), 10.0, compute)


def requirement_impacts(snap: Snapshot, rid: int) -> list[tuple[int, str]]:
    """(decision id, status) a requirement made stale or needs-review."""
    req = snap.nodes.get(rid) or {"data": {}}
    found: dict[int, str] = {}
    for k, v in ((req["data"].get("judged") or {}).items()):
        if v.get("status") in ("stale", "needs_review") and int(k) in snap.nodes:
            found[int(k)] = v["status"]
    for d in snap.of_type("decision"):
        data = d["data"]
        if data.get("stale_cause") == rid:
            found[d["id"]] = "stale"
        elif data.get("review_cause") == rid and d["id"] not in found:
            found[d["id"]] = "needs_review"
    return sorted(found.items())


# --- activity: what happened, with cause -> effect -------------------------------------

HOOK_LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ask tool=(\S+) \d+ms (.*)$")


@dataclass
class Event:
    at: datetime
    kind: str            # caught | hook | pivot | requirement | fit_check | resolve | claims | watch | ui | setup
    title: str
    detail: str = ""
    href: str | None = None
    effects: list[dict] = field(default_factory=list)   # {"label", "href", "status"}
    trace_id: str | None = None
    via_ui: bool = False

    @property
    def key(self) -> str:
        return hashlib.sha1(f"{self.kind}|{self.at.isoformat()}|{self.title}".encode()).hexdigest()[:10]


def _effect(snap: Snapshot, did: int, status: str | None = None) -> dict:
    d = snap.nodes.get(did)
    label = f"#{did} {d['label']}" if d else f"#{did}"
    return {"label": label, "href": f"/decisions/{did}", "status": status or (status_of(d) if d else "")}


def _tail(path: Path, limit: int = 200_000) -> list[str]:
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - limit))
            return f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []


def clip(text: str, limit: int = 220) -> str:
    """The first sentence(s) that fit, else cut at a word, never mid-number."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = cut.rfind(". ")
    return cut[:end + 1] if end > limit // 3 else cut.rsplit(" ", 1)[0] + "…"


def ui_log(project: Path) -> list[dict]:
    out = []
    for line in _tail(Path(project) / ".notbydefault" / ACTIVITY_LOG):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def events(project: Path, snap: Snapshot) -> list[Event]:
    """Newest first."""
    from notbydefault.costs import load_checks
    project = Path(project)
    ev: list[Event] = []
    ui = ui_log(project)
    ui_reqs = {e.get("node") for e in ui if e.get("action") == "require"}
    init = [n for n in snap.nodes.values() if n["data"].get("source") == "init"]
    if init:
        goal = next((n["label"] for n in init if n["type"] == "goal"), "")
        ev.append(Event(parse_time(init[0]["created_at"]), "setup", "Project initialised",
                        "; ".join(n["label"] for n in init if n["type"] != "goal")
                        + (f" — goal: {goal}" if goal else ""), "/graph"))
    for d in snap.of_type("decision"):
        data = d["data"]
        how = f"{data.get('kind')}/{data.get('category')}"
        via = f" via {data['tool_name']} call" if data.get("tool_name") else ""
        ev.append(Event(parse_time(d["created_at"]), "caught", f"Caught #{d['id']} {d['label']}",
                        f"{how}{via}" + (f" ({data['evidence']})" if data.get("evidence") else ""),
                        f"/decisions/{d['id']}", [_effect(snap, d["id"])]))
    for line in _tail(project / ".notbydefault" / "hook.log"):
        m = HOOK_LINE.match(line)
        if not m:
            continue
        msg = m.group(3).split(" Run `", 1)[0].removeprefix("NotByDefault: ")
        did = re.search(r" check (\d+) ", m.group(3))
        tool = m.group(2)
        article = "an" if tool[:1].lower() in "aeiou" else "a"
        ev.append(Event(parse_time(m.group(1)), "hook", f"Hook interrupted {article} {tool} call",
                        msg, f"/decisions/{did.group(1)}" if did else None,
                        [_effect(snap, int(did.group(1)))] if did else []))
    for r in snap.of_type("requirement") + snap.of_type("constraint"):
        if r["data"].get("source") in ("init", "check"):
            continue
        effects = [_effect(snap, did, st) for did, st in requirement_impacts(snap, r["id"])]
        ev.append(Event(parse_time(r["data"].get("added_at") or r["created_at"]), "requirement",
                        f"Requirement #{r['id']}: {r['label']}",
                        f"{len(effects)} decision(s) affected" if effects else
                        "no decision affected", f"/graph?req={r['id']}", effects,
                        via_ui=r["id"] in ui_reqs))
    for p in snap.of_type("pivot"):
        data = p["data"]
        ids = data.get("decisions") or []
        ev.append(Event(parse_time(data.get("proposed_at") or p["created_at"]), "pivot",
                        f"Pivot proposed: {p['label']}"
                        + (" (seeded demo record)" if data.get("source") == "seeded" else ""),
                        "; ".join(c.get("why", "") for c in data.get("conflicts") or [])[:300],
                        f"/nodes/{p['id']}", [_effect(snap, i) for i in ids]))
        if data.get("resolved_at"):
            st = data.get("status")
            ev.append(Event(parse_time(data["resolved_at"]), "pivot", f"Pivot {st}: {p['label']}",
                            "decisions it affects are now stale" if st == "accepted" else "",
                            f"/nodes/{p['id']}",
                            [_effect(snap, i, "stale" if st == "accepted" else None) for i in ids]))
    for c in load_checks(project):
        v = c.data.get("verdict") or {}
        rec = (v.get("recommended") or [None])[0]
        ev.append(Event(parse_time(c.run_at), "fit_check",
                        f"Fit check #{c.decision_id}: {v.get('kind') or 'no verdict'}"
                        + (f" → {rec}" if v.get("kind") == "switch" and rec else ""),
                        clip(v.get("summary") or ""), f"/decisions/{c.decision_id}",
                        [_effect(snap, c.decision_id)], c.data.get("trace_id")))
    for d in snap.of_type("decision"):
        c = d["data"].get("claims")
        if c and c.get("checked_at"):
            items = c.get("items") or []
            n = {v: sum(i["verdict"] == v for i in items) for v in ("true", "false", "untested")}
            ev.append(Event(parse_time(c["checked_at"]), "claims",
                            f"Claim check #{d['id']} {d['label']}: {n['false']} false, "
                            f"{n['true']} true, {n['untested']} untested",
                            "; ".join(f"{i['type']}: \"{i['text']}\"" for i in items
                                      if i["verdict"] == "false")[:280],
                            f"/decisions/{d['id']}", [_effect(snap, d["id"])], c.get("trace_id")))
        broken = (d["data"].get("assumptions") or {})
        if d["data"].get("review_basis") == "watched assumption" and broken.get("broken_at"):
            ev.append(Event(parse_time(broken["broken_at"]), "watch",
                            f"Re-opened #{d['id']} {d['label']}", d["data"].get("review_reason", ""),
                            f"/decisions/{d['id']}", [_effect(snap, d["id"], "needs_review")]))
        res = d["data"].get("resolution")
        if res:
            ev.append(Event(parse_time(res.get("at")), "resolve",
                            f"Resolved #{d['id']}: {res.get('from')} → {res.get('to')}",
                            f"branch {res.get('branch')}; tests {res.get('tests')}; "
                            f"verifiers {res.get('verifiers')}", f"/decisions/{d['id']}",
                            [_effect(snap, d["id"])], res.get("trace_id")))
    for e in ui:
        if e.get("action") == "require":
            continue
        ev.append(Event(parse_time(e.get("at")) or datetime.now(timezone.utc), "ui",
                        e.get("title", "UI action"), e.get("detail", ""), e.get("href"),
                        [_effect(snap, i) for i in e.get("decisions") or [] if i in snap.nodes],
                        via_ui=True))
    ev = [e for e in ev if e.at is not None]
    return sorted(ev, key=lambda e: e.at, reverse=True)


# --- overview, graph, evals, costs ----------------------------------------------------------

def cost_rows(project: Path, snap: Snapshot) -> tuple[list[dict], list]:
    from notbydefault.costs import decision_rows as rows_fn, load_checks
    checks = load_checks(Path(project))
    return rows_fn(Path(project), _GraphView(snap), checks), checks


def overview(project: Path, snap: Snapshot) -> dict:
    decisions = snap.of_type("decision")
    count = {s: sum(status_of(d) == s for d in decisions) for s in STATUSES}
    rows, checks = cost_rows(project, snap)
    savings = [r for r in rows if r["savings_monthly_usd"] not in ("", None)]
    from notbydefault.scan import compute
    try:
        debt = compute(Path(project), _GraphView(snap))
    except Exception:      # noqa: BLE001  (a malformed catalog must not break the page)
        debt = None
    # chips link into the latest fit-checked decision's page (the results file that page shows);
    # a section that page lacks falls back to its agent journal
    from notbydefault.fit.tracing import FOCUS_LABELS, current_focus
    journaled = [c for c in checks if c.data.get("journal") and c.decision_id in snap.nodes
                 and c.path == rel_to(project, snap.nodes[c.decision_id]["data"].get("results"))]
    target = max(journaled, key=lambda c: c.run_at) if journaled else None
    focus_did = target.decision_id if target else None
    present = {f"journal-{current_focus(e)}" for e in target.data["journal"]} if target else set()
    if target and snap.nodes[focus_did]["data"].get("history"):
        present.add("history")
    focus = [{"area": a, "label": FOCUS_LABELS[a], "covers": covers, "spans": spans,
              "href": None if not focus_did else f"/decisions/{focus_did}#"
              + (anchor if anchor in present or anchor in ("fit-steps", "affected", "fit-evidence",
                                                           "failure-bank") else "agent-journal")}
             for a, covers, spans, anchor in FOCUS_MAP]
    return {"decisions": len(decisions), "count": count, "checks": len(checks), "debt": debt,
            "focus": focus, "focus_decision": focus_did,
            "by_default": sum(by_default(d) for d in decisions), "savings": savings,
            "goal": next((g["label"] for g in snap.of_type("goal")), None)}


def graph_elements(snap: Snapshot, since: int = 0) -> dict:
    impacts = {r["id"]: requirement_impacts(snap, r["id"])
               for r in snap.of_type("requirement") + snap.of_type("constraint")}
    nodes = []
    for n in snap.nodes.values():
        if n["id"] <= since:
            continue
        label = f"#{n['id']} {n['label']}" if n["type"] == "decision" else n["label"]
        nodes.append({"data": {
            "id": f"n{n['id']}", "nid": n["id"], "type": n["type"],
            "label": label if len(label) <= 34 else label[:32] + "…", "full": label,
            "status": status_of(n) if n["type"] == "decision" else (n["data"].get("status") or ""),
            "by_default": by_default(n) if n["type"] == "decision" else False,
            "impacts": [i for i, _ in impacts.get(n["id"], [])],
            "href": f"/decisions/{n['id']}" if n["type"] == "decision" else f"/nodes/{n['id']}"}})
    edges = [{"data": {"id": f"e{e['src']}-{e['dst']}-{e['type']}", "source": f"n{e['src']}",
                       "target": f"n{e['dst']}", "type": e["type"]}}
             for e in snap.edges if e["src"] in snap.nodes and e["dst"] in snap.nodes]
    return {"nodes": nodes, "edges": edges, "max_id": snap.max_id}


RATE_METRICS = {"precision", "recall", "accuracy", "mapping_accuracy", "false_interruption_rate"}


def _version_key(name: str):
    m = re.match(r"v(\d+)", name)
    return (int(m.group(1)) if m else 999, name)


def evals() -> dict:
    """eval/results/*.json side by side, original cases then held-out (as `eval --compare`)."""
    from notbydefault.evals import EVAL_DIR, METRICS, _results
    folder = EVAL_DIR / "results"
    names = sorted((p.stem for p in folder.glob("*.json")) if folder.is_dir() else [],
                   key=_version_key)
    versions = [n for n in names if not n.endswith("-heldout")]
    files = {v: _results(v) for v in versions}
    cols = [{"name": v, "res": files[v], "key": "metrics", "heldout": False} for v in versions]
    for v in versions:
        held = _results(f"{v}-heldout") or files[v]
        if any(s.get("heldout_metrics") for s in held["sections"].values()):
            cols.append({"name": f"{v} held-out", "res": held, "key": "heldout_metrics",
                         "heldout": True})
    sections = []
    for sec, keys in METRICS.items():
        rows = []
        for k in keys:
            vals = [((c["res"]["sections"].get(sec) or {}).get(c["key"]) or {}).get(k)
                    for c in cols]
            rows.append({"metric": k.replace("_", " "), "values": vals,
                         "rate": k in RATE_METRICS, "lower_better": k.startswith(("false", "missed"))
                         or k in ("llm_calls", "llm_calls_per_ordinary_prompt",
                                  "ordinary_prompts_reaching_llm", "interruptions")})
        passed = []
        for c in cols:
            m = (c["res"]["sections"].get(sec) or {}).get(c["key"]) or {}
            passed.append(m["passed"] / m["cases"] if m.get("cases") and "passed" in m else None)
        traces = [(c["res"]["sections"].get(sec) or {}).get("trace_id") for c in cols]
        sections.append({"name": sec, "rows": rows, "passed": passed, "traces": traces})
    last = files[versions[-1]] if versions else None
    failures = []
    if last:
        for name, s in last["sections"].items():
            failures += [dict(f, section=name) for f in s.get("failures") or []]
    meta = [{"name": c["name"], "heldout": c["heldout"], "run_at": c["res"].get("run_at"),
             "commit": c["res"].get("commit")} for c in cols]
    return {"columns": meta, "sections": sections, "failures": failures,
            "last": versions[-1] if versions else None}


def costs_markdown(project: Path, snap: Snapshot, checks_per_month: float = 4.0,
                   projects_per_customer: float = 1.0) -> tuple[str, str]:
    """(costs.md, costs.csv) exactly as `export-costs` writes them, computed now."""
    from notbydefault.costs import render_csv, render_md, unit_economics
    rows, checks = cost_rows(project, snap)
    ue = unit_economics(checks, checks_per_month, projects_per_customer)
    return render_md(Path(project).resolve().name, rows, checks, ue), render_csv(rows)


HIDDEN_FIELDS = {"judged", "prices_usd_per_mtok", "affected_code", "history", "prompt",
                 "claims", "assumptions", "agent_reason"}


def node_fields(n: dict) -> list[tuple[str, str]]:
    """Short key/value lines for a node's side panel and page."""
    out = []
    for k, v in (n.get("data") or {}).items():
        if k in HIDDEN_FIELDS or SECRET_NAME.search(k):
            continue
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False)
        v = str(v)
        out.append((k.replace("_", " "), v if len(v) <= 300 else v[:297] + "…"))
    return out


def why_of(n: dict) -> str | None:
    if n["type"] != "decision":
        return None
    from notbydefault.records import why_text
    return why_text(n)
