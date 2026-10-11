"""Screenshots of every UI page, light and dark, into docs/screenshots/.

Runs against a throwaway copy of demo-app (graph.db copied with SQLite's backup API, so a
check writing to the live graph is never disturbed) and uses the installed Chrome via
Playwright (no `playwright install`). Read-only: no actions are clicked.

    python scripts/ui_screenshots.py [--width 1280] [--out docs/screenshots] [--pages overview,decision]
"""
from __future__ import annotations

import argparse
import shutil
import socket
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PAGES = [("overview", "/"), ("decisions", "/decisions"), ("decision", "/decisions/{did}"),
         ("graph", "/graph"), ("graph-focus", "/graph?focus={did}"), ("timeline", "/timeline"), ("evals", "/evals"),
         ("costs", "/costs"), ("fit-check", "/decisions/{did}")]
# pages shot as a clip: from the top of the first selector to the bottom of the last one
CLIPS = {"fit-check": ("#fit-evidence", ".journal")}


def throwaway_demo(dest: Path) -> Path:
    src = REPO / "demo-app"
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns(
        "graph.db*", "cache", "__pycache__", ".demo-run"))
    live = sqlite3.connect(f"{(src / '.notbydefault' / 'graph.db').resolve().as_uri()}?mode=ro",
                           uri=True)
    copy = sqlite3.connect(dest / ".notbydefault" / "graph.db")
    live.backup(copy)
    live.close()
    copy.close()
    return dest


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=860)
    ap.add_argument("--out", default=str(REPO / "docs" / "screenshots"))
    ap.add_argument("--decision", type=int, default=21)
    ap.add_argument("--project", help="render this project as it is (e.g. the demo dataset) "
                                      "instead of a throwaway copy of demo-app")
    ap.add_argument("--claims-decision", type=int,
                    help="also shoot this decision's page as decision-claims-*.png")
    ap.add_argument("--pages", help="comma-separated subset of: "
                    + ", ".join(name for name, _ in PAGES))
    args = ap.parse_args()
    import uvicorn
    from playwright.sync_api import sync_playwright
    from notbydefault.ui.app import create_app

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pages = PAGES + ([("decision-claims", f"/decisions/{args.claims_decision}")]
                     if args.claims_decision else [])
    with tempfile.TemporaryDirectory() as tmp:
        project = Path(args.project).resolve() if args.project else throwaway_demo(Path(tmp) / "demo-app")
        port = free_port()
        server = uvicorn.Server(uvicorn.Config(create_app(project), host="127.0.0.1", port=port,
                                               log_level="warning"))
        threading.Thread(target=server.run, daemon=True).start()
        while not server.started:
            time.sleep(0.05)
        base = f"http://127.0.0.1:{port}"
        with sync_playwright() as pw:
            browser = pw.chromium.launch(channel="chrome")
            for scheme in ("light", "dark"):
                ctx = browser.new_context(viewport={"width": args.width, "height": args.height},
                                          color_scheme=scheme, device_scale_factor=1)
                page = ctx.new_page()
                errors: list[str] = []
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
                for name, path in pages:
                    if args.pages and name not in args.pages.split(","):
                        continue
                    page.goto(base + path.format(did=args.decision), wait_until="networkidle")
                    if name.startswith("graph"):
                        page.wait_for_timeout(1200)       # dagre layout
                    over = page.evaluate("document.documentElement.scrollWidth - innerWidth")
                    file = out / f"{name}-{scheme}.png"
                    if name in CLIPS:
                        first, last = (page.locator(s).first.bounding_box() for s in CLIPS[name])
                        top = first["y"] - 12
                        page.screenshot(path=str(file), full_page=True, clip={
                            "x": 0, "y": top, "width": args.width,
                            "height": last["y"] + last["height"] + 16 - top})
                    else:
                        page.screenshot(path=str(file), full_page=not name.startswith("graph"))
                    print(f"{file.relative_to(REPO) if file.is_relative_to(REPO) else file}  horizontal overflow: {over}px")
                if errors:
                    print(f"{scheme}: browser errors:\n  " + "\n  ".join(errors))
                ctx.close()
            browser.close()
        server.should_exit = True
    return 0


if __name__ == "__main__":
    sys.exit(main())
