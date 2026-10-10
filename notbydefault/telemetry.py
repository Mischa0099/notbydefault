"""Send one neatlogs span for a PreToolUse hook call, then flush.

Run detached by hooks/pre_tool.py:  python -m notbydefault.telemetry '<json payload>'
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from notbydefault import REPO_ROOT


def load_env(path: Path = REPO_ROOT / ".env") -> None:
    """Minimal .env reader: KEY=VALUE lines; existing environment wins."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _log(payload: dict, message: str) -> None:
    project = payload.get("project_dir")
    if not project:
        return
    try:
        with open(Path(project, ".notbydefault", "hook.log"), "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} telemetry: {message}\n")
    except OSError:
        pass


def send(payload: dict) -> bool:
    load_env()
    if not os.environ.get("NEATLOGS_API_KEY"):
        _log(payload, "NEATLOGS_API_KEY not set; skipped")
        return False

    import neatlogs

    neatlogs.init(workflow_name="notbydefault", tags=["pre_tool_hook"],
                  register_shutdown_handlers=False)
    decisions = payload.get("decisions") or []

    def joined(key):
        return ", ".join(str(d.get(key)) for d in decisions)

    @neatlogs.span(kind="TOOL", name="notbydefault.pre_tool",
                   session_id=payload.get("session_id") or None)
    def pre_tool_hook(kind: str, chosen: str, decision_id: str, tool_name: str,
                      decisions: list) -> dict:
        return {"permission_decision": "ask", "decision_count": len(decisions)}

    pre_tool_hook(kind=joined("kind"), chosen=joined("chosen"),
                  decision_id=joined("decision_id"), tool_name=payload.get("tool_name") or "",
                  decisions=decisions)
    neatlogs.flush(timeout_millis=15000)
    failures = neatlogs.get_delivery_diagnostics().get("span_export_failures", 0)
    neatlogs.shutdown(timeout_millis=5000)
    _log(payload, f"sent span for decision(s) {joined('decision_id')}"
                  + (f" ({failures} export failure(s))" if failures else ""))
    return not failures


def main(argv: list[str]) -> int:
    payload: dict = {}
    try:
        payload = json.loads(argv[1])
        send(payload)
    except Exception as e:
        _log(payload, f"error: {e!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
