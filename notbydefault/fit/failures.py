"""Failure bank: every failing output, so later runs test those cases first."""
from __future__ import annotations

import json
from pathlib import Path


def bank_path(project: Path, decision_id: int) -> Path:
    return project / ".notbydefault" / "failures" / f"{decision_id}.jsonl"


def load(project: Path, decision_id: int) -> list[dict]:
    path = bank_path(project, decision_id)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def append(project: Path, decision_id: int, records: list[dict]) -> int:
    """Add records not already banked (same input, output and model). Returns how many."""
    existing = {(r["input"], r["output"], r["model"]) for r in load(project, decision_id)}
    new = []
    for r in records:
        key = (r["input"], r["output"], r["model"])
        if key not in existing:
            existing.add(key)
            new.append(r)
    if new:
        path = bank_path(project, decision_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for r in new:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(new)


def inputs(project: Path, decision_id: int) -> list[str]:
    """Unique banked inputs, oldest first."""
    seen: dict[str, None] = {}
    for r in load(project, decision_id):
        seen.setdefault(r["input"], None)
    return list(seen)
