"""Keys come only from the repo's .env (existing environment variables win)."""
from __future__ import annotations

import os
from pathlib import Path

from notbydefault import REPO_ROOT


def load_env(path: Path = REPO_ROOT / ".env") -> None:
    """Minimal .env reader: KEY=VALUE lines."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def require_key(name: str) -> str:
    load_env()
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set. Add `{name}=...` to {REPO_ROOT / '.env'}")
    return value
