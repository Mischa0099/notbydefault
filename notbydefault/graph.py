"""Decision Graph stored in SQLite at <project-dir>/.notbydefault/graph.db."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

NODE_TYPES = {
    "goal", "requirement", "constraint", "decision",
    "evidence", "alternative", "code", "pivot",
}
EDGE_TYPES = {"depends_on", "supported_by", "rejects", "affects", "serves"}

# Edges a decision follows to reach the things it relies on.
DEPENDENCY_EDGES = ("depends_on", "serves", "supported_by")

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    label TEXT NOT NULL,
    data_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    session_id TEXT
);
CREATE TABLE IF NOT EXISTS edges (
    src INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    dst INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    type TEXT NOT NULL,
    PRIMARY KEY (src, dst, type)
);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
"""


def db_path(project_dir: str | Path) -> Path:
    return Path(project_dir) / ".notbydefault" / "graph.db"


class Graph:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Hooks can fire concurrently for parallel tool calls; wait briefly on locks.
        self.conn = sqlite3.connect(self.path, timeout=2.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)

    @classmethod
    def for_project(cls, project_dir: str | Path) -> "Graph":
        return cls(db_path(project_dir))

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Graph":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- writes -----------------------------------------------------------

    def add_node(self, type: str, label: str, data: dict | None = None,
                 session_id: str | None = None) -> int:
        if type not in NODE_TYPES:
            raise ValueError(f"unknown node type: {type}")
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO nodes (type, label, data_json, created_at, session_id)"
                " VALUES (?, ?, ?, ?, ?)",
                (type, label, json.dumps(data or {}),
                 datetime.now(timezone.utc).isoformat(timespec="seconds"), session_id),
            )
        return cur.lastrowid

    def add_edge(self, src: int, dst: int, type: str) -> None:
        if type not in EDGE_TYPES:
            raise ValueError(f"unknown edge type: {type}")
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO edges (src, dst, type) VALUES (?, ?, ?)",
                (src, dst, type),
            )

    def update_data(self, node_id: int, **fields) -> None:
        node = self.get_node(node_id)
        if node is None:
            raise KeyError(node_id)
        node["data"].update(fields)
        with self.conn:
            self.conn.execute("UPDATE nodes SET data_json = ? WHERE id = ?",
                              (json.dumps(node["data"]), node_id))

    def delete_nodes(self, ids: list[int]) -> None:
        with self.conn:
            self.conn.executemany("DELETE FROM nodes WHERE id = ?", [(i,) for i in ids])

    # --- reads ------------------------------------------------------------

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["data"] = json.loads(d.pop("data_json") or "{}")
        return d

    def get_node(self, node_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return self._row(row) if row else None

    def nodes(self, type: str | None = None) -> list[dict]:
        if type:
            rows = self.conn.execute("SELECT * FROM nodes WHERE type = ? ORDER BY id", (type,))
        else:
            rows = self.conn.execute("SELECT * FROM nodes ORDER BY id")
        return [self._row(r) for r in rows]

    def edges(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT src, dst, type FROM edges ORDER BY src, dst, type")]

    def find_node(self, type: str, label: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM nodes WHERE type = ? AND label = ? ORDER BY id LIMIT 1",
            (type, label)).fetchone()
        return self._row(row) if row else None

    def list_decisions(self, status: str | None = None) -> list[dict]:
        decisions = self.nodes("decision")
        if status:
            decisions = [d for d in decisions if d["data"].get("status") == status]
        return decisions

    def neighbors(self, node_id: int, edge_types: tuple[str, ...] | None = None) -> list[dict]:
        """Nodes reached by outgoing edges from node_id (each with its edge type)."""
        sql = ("SELECT n.*, e.type AS edge_type FROM edges e JOIN nodes n ON n.id = e.dst"
               " WHERE e.src = ?")
        params: list = [node_id]
        if edge_types:
            sql += f" AND e.type IN ({','.join('?' * len(edge_types))})"
            params += edge_types
        return [self._row(r) for r in self.conn.execute(sql + " ORDER BY n.id", params)]

    def get_dependencies(self, decision_id: int,
                         edge_types: tuple[str, ...] = DEPENDENCY_EDGES) -> list[dict]:
        """Everything a decision depends on, following edges transitively."""
        seen: dict[int, dict] = {}
        frontier = [decision_id]
        while frontier:
            current = frontier.pop()
            for node in self.neighbors(current, edge_types):
                if node["id"] not in seen and node["id"] != decision_id:
                    seen[node["id"]] = node
                    frontier.append(node["id"])
        return sorted(seen.values(), key=lambda n: n["id"])


def find_affected_code(decision: dict) -> list[str]:
    """Files a decision affects.

    Stub: later this will query Entire Graph for every caller/consumer of the
    chosen service or model. For now it returns the file(s) the tool calls touched.
    """
    data = decision.get("data", {})
    if data.get("files"):
        return list(data["files"])
    return [data["file"]] if data.get("file") else []
