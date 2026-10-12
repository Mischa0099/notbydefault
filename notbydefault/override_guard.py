"""Override Guard: notice when a new decision undoes one that already has evidence.

A new decision of the same type (kind + category) as an older decision that is `reviewed` or
`stale` (i.e. was fit-checked) gets `override_guard` on its data. It only warns (a banner on the
receipt); nothing is enforced.
"""
from __future__ import annotations

import json
from pathlib import Path

GUARDED = ("reviewed", "stale")


def _trace(project_dir: Path | None, old: dict) -> str | None:
    """The trace id of the older decision's fit check, from its results file (None if absent)."""
    rel = old.get("results")
    if not rel or project_dir is None:
        return None
    try:
        return json.loads((Path(project_dir) / rel).read_text(encoding="utf-8")).get("trace_id")
    except (OSError, ValueError, AttributeError):
        return None


def check(graph, data: dict, label: str, project_dir: Path | None = None) -> dict | None:
    """override_guard for a decision about to be recorded, or None when it overrides nothing."""
    kind, category = data.get("kind"), data.get("category")
    if not kind:
        return None
    olds = [d for d in graph.nodes("decision")
            if d["data"].get("status") in GUARDED
            and d["data"].get("kind") == kind and d["data"].get("category") == category
            and d["label"].lower() != label.lower()]
    if not olds:
        return None
    old = max(olds, key=lambda d: (bool(d["data"].get("results")), d["id"]))  # evidence first
    return {"status": "needs_evidence", "overrides_decision_id": old["id"],
            "old_evidence_trace": _trace(project_dir, old["data"])}
