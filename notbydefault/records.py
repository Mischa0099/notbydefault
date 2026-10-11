"""Decision records: decisions/NNNN-<slug>.md, one per decision, rewritten on every verdict.

The file sits in the project's working tree, so it is committed with the code and lands in the
next Entire checkpoint. `check` writes a detailed body (results table, case results); this module
keeps a generated summary block at the end of the file (between the markers below) up to date:
status, why, evidence links, rejected alternatives, caveats, affected code and status history.
Anything outside the markers is left alone.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

BEGIN = "<!-- notbydefault:summary:begin (generated; edits inside are overwritten) -->"
END = "<!-- notbydefault:summary:end -->"
STATUS_TEXT = {"unreviewed": "Unreviewed (detected from an agent tool call; not fit-tested)",
               "reviewed": "Reviewed", "stale": "STALE (provably conflicts with a requirement)",
               "needs_review": "Needs review (may conflict, depending on how it is deployed)",
               "accepted": "Accepted"}


def folder(project: Path) -> Path:
    return Path(project) / "decisions"


def slug(decision: dict) -> str:
    kind = decision["data"].get("kind") or "decision"
    prefix = "ai-model" if kind == "model" else (decision["data"].get("category") or kind)
    return re.sub(r"[^a-z0-9]+", "-", f"{prefix}-{decision['label']}".lower()).strip("-")[:60]


def next_path(project: Path, decision: dict) -> Path:
    f = folder(project)
    numbers = [int(m.group(1)) for p in f.glob("*.md") if (m := re.match(r"(\d{4})-", p.name))]
    return f / f"{max(numbers, default=0) + 1:04d}-{slug(decision)}.md"


def existing_path(project: Path, decision: dict) -> Path | None:
    rel = decision["data"].get("decision_record")
    if rel and (Path(project) / rel).is_file():
        return Path(project) / rel
    for p in sorted(folder(project).glob("*.md")):   # a record written before it was linked
        try:
            if f"Decision graph node: #{decision['id']} " in p.read_text(encoding="utf-8"):
                return p
        except OSError:
            continue
    return None


def _rel(path: Path, project: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(project).resolve()).as_posix()
    except ValueError:
        return str(path)


def note_history(graph, decision_id: int, event: str) -> None:
    """Append one line to the decision's status history (kept on the node)."""
    d = graph.get_node(decision_id)
    history = list(d["data"].get("history") or [])
    history.append({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "status": d["data"].get("status"), "event": event})
    graph.update_data(decision_id, history=history[-30:])


def why_text(d: dict) -> str:
    data = d["data"]
    if data.get("status") == "stale":
        return data.get("stale_reason") or "marked stale"
    if data.get("status") == "needs_review":
        return data.get("review_reason") or "needs review"
    if data.get("verdict_summary"):
        return data["verdict_summary"]
    how = f" ({data.get('evidence')})" if data.get("evidence") else ""
    return (f"Chosen by a coding agent's {data.get('tool_name') or 'tool'} call{how}; "
            "no recorded reason beyond that yet.")


def summary_block(project: Path, graph, d: dict) -> list[str]:
    from notbydefault.affected import describe
    from notbydefault.graph import affected_entries
    data = d["data"]
    lines = [BEGIN, "", "## Summary (NotByDefault)", "",
             f"- Decision graph node: #{d['id']} ({data.get('kind')}/{data.get('category')}: "
             f"`{d['label']}`)",
             f"- Status: {STATUS_TEXT.get(data.get('status'), data.get('status'))}"]
    if data.get("recommended"):
        lines.append(f"- Recommended: `{data['recommended']}`")
    if d.get("session_id"):
        lines.append(f"- Detected in agent session: `{d['session_id']}`")
    lines += ["", "### Why", "", why_text(d), ""]

    evidence = graph.neighbors(d["id"], ("supported_by",))
    lines += ["### Evidence", ""]
    for e in evidence:
        link = e["data"].get("path") or e["data"].get("url")
        lines.append(f"- #{e['id']} {e['label']}" + (f": `{link}`" if link else ""))
    key = {"stale": "stale_source", "needs_review": "review_source"}.get(data.get("status"))
    if key and data.get(key):
        lines.append(f"- Documentation: {data[key]}")
    if data.get("results"):
        lines.append(f"- Raw fit-test results: `{data['results']}`")
    if len(lines) and lines[-1] == "":
        lines.append("- none recorded")
    lines.append("")

    rejected = graph.neighbors(d["id"], ("rejects",))
    lines += ["### Rejected alternatives", ""]
    lines += [f"- `{a['label']}`: {a['data'].get('reason') or 'no reason recorded'}"
              for a in rejected] or ["- none recorded"]
    lines.append("")

    caveats = []
    if data.get("status") == "needs_review":
        caveats.append(f"Needs review ({data.get('review_basis')}): {data.get('review_reason')}")
    for n in graph.neighbors(d["id"], ("depends_on",)):
        if n["data"].get("assumption"):
            caveats.append(f"Assumption #{n['id']}: {n['label']}")
    if data.get("affected_note"):
        caveats.append(f"Affected code: {data['affected_note']}")
    if not data.get("verdict_summary") and data.get("status") == "unreviewed":
        caveats.append("Not fit-tested yet: run `python -m notbydefault check "
                       f"{d['id']}` (model decisions).")
    lines += ["### Caveats", ""] + ([f"- {c}" for c in caveats] or ["- none recorded"]) + [""]

    entries = affected_entries(d)
    lines += [f"### Affected code ({data.get('affected_source') or 'touched files'})", ""]
    lines += [f"- `{describe(e)}`: {e.get('via', '')}" for e in entries] or ["- none recorded"]
    lines.append("")

    history = data.get("history") or []
    if history:
        lines += ["### Status history", ""]
        lines += [f"- {h['at']}: {h['status']}: {h['event']}" for h in history] + [""]
    lines.append(END)
    return lines


def write_record(project: Path, graph, decision_id: int, event: str | None = None) -> Path:
    """Create or update the decision's record; returns its path (also stored on the node)."""
    project = Path(project)
    if event:
        note_history(graph, decision_id, event)
    d = graph.get_node(decision_id)
    path = existing_path(project, d)
    block = "\n".join(summary_block(project, graph, d)) + "\n"
    if path is None:
        path = next_path(project, d)
        path.parent.mkdir(parents=True, exist_ok=True)
        title = ("AI model" if d["data"].get("kind") == "model"
                 else (d["data"].get("category") or "Decision").capitalize())
        text = f"# {path.stem[:4]}. {title}: {d['label']}\n\n" + block
    else:
        text = path.read_text(encoding="utf-8")
        if BEGIN in text and END in text:
            head, rest = text.split(BEGIN, 1)
            text = head + block.rstrip("\n") + rest.split(END, 1)[1]
            text = text.rstrip("\n") + "\n"
        else:
            text = text.rstrip("\n") + "\n\n" + block
    path.write_text(text, encoding="utf-8")
    graph.update_data(decision_id, decision_record=_rel(path, project))
    return path


def write_records(project: Path, graph, decision_ids, event: str, tracer=None,
                  out=None, entire=None) -> list[Path]:
    """write_record for several decisions, then offer each record to Entire Brain."""
    from notbydefault import brain
    decision_ids = list(decision_ids)
    paths = []
    for did in decision_ids:
        path = write_record(project, graph, did, event)
        paths.append(path)
        if tracer is not None:
            tracer.event("decision_record", "state", f"#{did}: {event} -> {_rel(path, project)}",
                         decision_id=did, path=_rel(path, project))
        saved, msg = brain.save(project, path, graph.get_node(did), entire)
        if tracer is not None:
            tracer.event("brain_saved" if saved else "brain_skipped",
                         "state" if saved else "recovery", msg, decision_id=did)
        if out is not None and (saved or did == decision_ids[0]):
            out(msg)
    return paths
