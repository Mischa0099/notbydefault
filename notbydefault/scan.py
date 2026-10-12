"""Decision debt: `notbydefault scan [--record]` (and the UI overview).

Audits the project's code and its Decision Graph. One point for each:
  unchecked            a decision with no fit-check verdict and no reason from a person, or a
                       model literal / service package in the code that no decision records
                       (chosen by default, never caught: e.g. written before the hook was installed)
  claim false          a claim in an agent's stated reason that the evidence contradicts
  assumption expired   a decision whose verdict rests on an assumption that no longer holds
Each row comes with the next command to pay it off. Read-only, except `--record`, which adds the
unrecorded choices to the graph as unreviewed decisions (re-opening is `watch`'s job).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Found:
    """A choice in the code that the graph doesn't record."""
    kind: str
    chosen: str
    category: str
    file: str
    evidence: str


DEBT_LINE = ("Decision debt: choices that are unchecked, whose claims proved false, or whose "
             "assumptions have expired.")


def unchecked(d: dict) -> bool:
    data = d["data"]
    return not data.get("verdict") and not data.get("reason") \
        and data.get("status", "unreviewed") != "reviewed"


def unrecorded(project: Path, graph) -> list[Found]:
    """Model literals and known service packages in the code with no decision node."""
    from notbydefault.detect import (_project_files, _read, dependency_names, is_manifest,
                                     lookup_service, model_ids)
    project = Path(project).resolve()
    known = set()
    for d in graph.nodes("decision"):
        known.add((d["data"].get("kind"), d["label"].lower()))
        to = (d["data"].get("resolution") or {}).get("to")
        if to:      # resolve switched it: the new model is this decision's, not a new choice
            known.add(("model", to.split(":", 1)[-1].lower()))
    found: dict[tuple[str, str], Found] = {}
    for path in _project_files(project, None):
        text = _read(str(path))
        if not text:
            continue
        rel = path.relative_to(project).as_posix()
        hits = [("model", m, "ai-model", m) for m in model_ids(text, str(path))]
        if is_manifest(str(path)):
            hits += [("service", svc[0], svc[1], pkg) for pkg in sorted(dependency_names(str(path), text))
                     if (svc := lookup_service(pkg))]
        for kind, chosen, category, evidence in hits:
            key = (kind, chosen.lower())
            if key not in known and key not in found:
                found[key] = Found(kind, chosen, category, rel, evidence)
    return sorted(found.values(), key=lambda f: (f.file, f.chosen))


def next_command(d: dict, why: str) -> str:
    did, data = d["id"], d["data"]
    model = data.get("kind") == "model"
    if why == "expired" and data.get("status") not in ("needs_review", "stale"):
        return "notbydefault watch"
    if why == "claim" and model and data.get("verdict") == "switch":
        return f"notbydefault resolve {did}"
    if model:
        return f"notbydefault check {did}"
    return f"notbydefault ui (decision #{did}: mark reviewed with a reason)"


def compute(project: Path, graph, catalog: dict | None = None, audit_code: bool = True) -> dict:
    """{"score", "unchecked", "unrecorded", "claims_false", "expired", "rows"}."""
    from notbydefault.assumptions import broken, describe
    from notbydefault.fit.providers import load_catalog
    catalog = catalog or load_catalog(project)
    decisions = graph.nodes("decision")
    out = {"unchecked": [d for d in decisions if unchecked(d)],
           "unrecorded": unrecorded(project, graph) if audit_code else [],
           "claims_false": [], "expired": []}
    for d in decisions:
        for item in (d["data"].get("claims") or {}).get("items") or []:
            if item["verdict"] == "false":
                out["claims_false"].append((d, item))
        items = broken(project, graph, d, catalog) \
            or (d["data"].get("assumptions") or {}).get("broken") or []
        if items:
            out["expired"].append((d, items))
    rows = [{"count": "unchecked", "id": d["id"], "decision": d["label"],
             "why": f"{d['data'].get('kind')}/{d['data'].get('category')}, "
                    f"{d['data'].get('status', 'unreviewed')}: no fit check, no reason",
             "next": next_command(d, "unchecked")} for d in out["unchecked"]]
    rows += [{"count": "unchecked", "id": None, "decision": f.chosen,
              "why": f"in {f.file} ({f.evidence}), not in the graph", "next": "notbydefault scan --record"}
             for f in out["unrecorded"]]
    rows += [{"count": "claim false", "id": d["id"], "decision": d["label"],
              "why": f"{i['type']}: \"{i['text']}\" ({i['detail']})", "next": next_command(d, "claim")}
             for d, i in out["claims_false"]]
    rows += [{"count": "assumption expired", "id": d["id"], "decision": d["label"],
              "why": "; ".join(describe(b) for b in items), "next": next_command(d, "expired")}
             for d, items in out["expired"]]
    out["rows"] = rows
    out["score"] = (len(out["unchecked"]) + len(out["unrecorded"]) + len(out["claims_false"])
                    + len(out["expired"]))
    return out


def record_found(project: Path, graph, found: list[Found]) -> list[int]:
    ids = []
    for f in found:
        nid = graph.add_node("decision", f.chosen, {
            "kind": f.kind, "chosen": f.chosen, "category": f.category, "evidence": f.evidence,
            "source": "scan", "file": f.file, "files": [f.file], "status": "unreviewed"})
        for n in graph.nodes("goal"):
            graph.add_edge(nid, n["id"], "serves")
        for t in ("requirement", "constraint"):
            for n in graph.nodes(t):
                graph.add_edge(nid, n["id"], "depends_on")
        code = graph.find_node("code", f.file)
        graph.add_edge(nid, code["id"] if code else graph.add_node("code", f.file, {"path": f.file}),
                       "affects")
        ids.append(nid)
    return ids


def _table(rows: list[dict]) -> list[str]:
    cells = [("count", "#", "decision", "next command", "why")] + [
        (r["count"], f"#{r['id']}" if r["id"] else "-", r["decision"], r["next"],
         r["why"] if len(r["why"]) <= 110 else r["why"][:107] + "...") for r in rows]
    w = [max(len(c[i]) for c in cells) for i in range(4)]
    line = lambda c: "  ".join(x.ljust(w[i]) for i, x in enumerate(c[:4])) + "  " + c[4]
    return [line(cells[0]), "-" * (sum(w) + 8 + 20)] + [line(c) for c in cells[1:]]


def scan(project: Path, record: bool = False, out=print) -> int:
    from notbydefault.graph import Graph, db_path
    project = Path(project).resolve()
    if not db_path(project).exists():
        out(f"No Decision Graph at {db_path(project)}. Run `notbydefault init` first.")
        return 1
    with Graph.for_project(project) as g:
        r = compute(project, g)
        if record and r["unrecorded"]:
            ids = record_found(project, g, r["unrecorded"])
            out(f"Recorded {len(ids)} choice(s) found in the code as unreviewed decisions: "
                + ", ".join(f"#{i}" for i in ids))
            r = compute(project, g)
    unchecked_n = len(r["unchecked"]) + len(r["unrecorded"])
    out(DEBT_LINE)
    out(f"Decision debt: {r['score']}  ({unchecked_n} unchecked"
        + (f", {len(r['unrecorded'])} of them not in the graph" if r["unrecorded"] else "")
        + f" / {len(r['claims_false'])} claim(s) false / "
        f"{len(r['expired'])} with expired assumptions)")
    if r["rows"]:
        out("")
        out("\n".join(_table(r["rows"])))
    else:
        out("No decision debt.")
    return 0


# --- retroactive audit: `scan --history` (read-only; nothing is written) --------------------

HISTORY_PATHS = ("*.py", "*.yaml", "*.toml", "requirements*.txt")


def _git(project: Path, *args: str) -> str:
    import subprocess
    return subprocess.run(["git", "-C", str(project), *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", check=True).stdout


def history_hits(diff: str) -> list[tuple[str, str]]:
    """(path, added line) for each added line of a unified diff that the existing detector
    (detect.detect_file_change / detect_shell) calls a decision; a literal or package already
    on a removed line of the same file (a moved or edited line) is not new."""
    from notbydefault.detect import detect_file_change, detect_shell
    files: dict[str, tuple[list[str], list[str]]] = {}
    path = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("--- ") or path is None:
            continue
        elif line.startswith("+"):
            files.setdefault(path, ([], []))[1].append(line[1:])
        elif line.startswith("-"):
            files.setdefault(path, ([], []))[0].append(line[1:])
    hits = []
    for path, (removed, added) in files.items():
        old = "\n".join(removed)
        for line in added:
            if detect_file_change(path, old, line) or detect_shell(line.strip(), None):
                hits.append((path, line.strip()))
    return hits


def scan_history(project: Path, limit: int = 100, out=print) -> int:
    """Run the decision detector over the last `limit` commits' diffs and print one block."""
    import subprocess
    project = Path(project).resolve()
    try:
        log = _git(project, "log", f"-n{limit}", "--format=%H%x09%h%x09%ad", "--date=short")
    except (OSError, subprocess.CalledProcessError) as e:
        out(f"Repo history scan: not a git repository or git failed ({e}).")
        return 1
    commits = [line.split("\t") for line in log.splitlines() if line.count("\t") == 2]
    examples, n_hits, n_commits = [], 0, 0
    for sha, short, date in commits:
        diff = _git(project, "show", "--format=", "--unified=0", "--no-color", "--no-ext-diff",
                    "--diff-merges=first-parent", sha, "--", *HISTORY_PATHS)
        hits = history_hits(diff)
        if hits:
            n_hits += len(hits)
            n_commits += 1
            examples += [(short, date, p, l) for p, l in hits[:3 - len(examples)]]
    out(f"Repo history scan: scanned last {len(commits)} commits. Found {n_hits} "
        f"decision-shaped line(s) across {n_commits} commit(s). Read-only: the graph is not "
        "touched.")
    if examples:
        out("Examples (up to 3):")
        for short, date, p, l in examples:
            out(f"  - {short} {date} {p}: {l[:100]}")
    return 0
