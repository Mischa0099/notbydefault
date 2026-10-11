"""The FastAPI app behind `notbydefault ui`: server-rendered pages (Jinja2, autoescaped), HTMX
fragments for live updates, and a few actions that call the existing Python functions.

Safety: binds 127.0.0.1 only; requests must name 127.0.0.1/localhost as Host (no DNS
rebinding); every write needs the per-run CSRF token AND a same-origin Origin/Referer, and is
refused outright with --read-only; only files inside the project are shown, never secrets.
"""
from __future__ import annotations

import json
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup, escape

from notbydefault.ui import data, mdlite

HERE = Path(__file__).resolve().parent
STATUS_TEXT = {"unreviewed": "unreviewed", "reviewed": "reviewed", "needs_review": "needs review",
               "stale": "stale", "accepted": "accepted", "proposed": "proposed", "kept": "kept",
               "adjusting": "adjusting"}
KIND_TEXT = {"caught": "caught", "hook": "hook", "pivot": "pivot", "requirement": "requirement",
             "fit_check": "fit check", "resolve": "resolve", "ui": "you", "setup": "setup"}


def _ago(t: datetime | None) -> str:
    if t is None:
        return "-"
    s = (datetime.now(timezone.utc) - t).total_seconds()
    if s < 60:
        return "just now"
    for unit, n in (("d", 86400), ("h", 3600), ("m", 60)):
        if s >= n:
            return f"{int(s // n)}{unit} ago"
    return "-"


def _when(value) -> str:
    t = value if isinstance(value, datetime) else data.parse_time(value)
    return t.astimezone().strftime("%Y-%m-%d %H:%M") if t else "-"


def _money(v, places: int = 2) -> str:
    if v in ("", None):
        return "-"
    return f"${float(v):,.{places}f}"


def _diff(text: str) -> Markup:
    """A unified diff as escaped lines, coloured by + / - / @@."""
    out = []
    for line in (text or "").splitlines():
        cls = ("add" if line.startswith("+") and not line.startswith("+++") else
               "del" if line.startswith("-") and not line.startswith("---") else
               "hunk" if line.startswith("@@") else "")
        out.append(f'<span class="{cls}">{escape(line)}</span>' if cls else str(escape(line)))
    return Markup('<pre class="code diff">' + "\n".join(out) + "</pre>")


def make_env() -> Environment:
    env = Environment(loader=FileSystemLoader(str(HERE / "templates")),
                      autoescape=select_autoescape(default=True, default_for_string=True))
    env.filters.update(diff=_diff, ago=_ago, when=_when, money=_money, md=mdlite.render,
                       status_text=lambda s: STATUS_TEXT.get(s, s or "-"),
                       kind_text=lambda k: KIND_TEXT.get(k, k),
                       pct=lambda v: "-" if v is None else f"{v:.0%}")
    return env


class Settings:
    def __init__(self, project: Path, read_only: bool, client_factory, entire,
                 allowed_hosts: tuple[str, ...]):
        self.project = Path(project).resolve()
        self.read_only = read_only
        self.client_factory = client_factory   # tests: fake providers (no network)
        self.entire = entire
        self.allowed_hosts = allowed_hosts
        self.csrf = secrets.token_urlsafe(32)
        self.lock = threading.Lock()           # one action at a time


def create_app(project: Path, read_only: bool = False, client_factory=None, entire=None,
               allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost")) -> FastAPI:
    cfg = Settings(project, read_only, client_factory, entire, allowed_hosts)
    env = make_env()
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.state.cfg = cfg
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]")
        if host not in cfg.allowed_hosts:
            return Response("Bad host", status_code=400)
        response = await call_next(request)
        response.headers.update({
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
            "X-Content-Type-Options": "nosniff", "Referrer-Policy": "same-origin",
            "X-Frame-Options": "DENY", "Cache-Control": "no-store"})
        return response

    def page(request: Request, name: str, status_code: int = 200, **ctx) -> HTMLResponse:
        ctx.update(request=request, read_only=cfg.read_only, csrf=cfg.csrf,
                   project_name=cfg.project.name, project=str(cfg.project),
                   path=request.url.path, version=data.version(cfg.project))
        return HTMLResponse(env.get_template(name).render(**ctx), status_code=status_code)

    app.state.page = page
    app.state.env = env

    from notbydefault.ui import actions, routes
    routes.register(app, cfg, page)
    actions.register(app, cfg, page)
    return app


def same_origin(request: Request) -> bool:
    host = request.headers.get("host", "")
    for header in ("origin", "referer"):
        value = request.headers.get(header)
        if value:
            return urlsplit(value).netloc == host
    return False


def check_write(request: Request, cfg: Settings, form_token: str | None) -> None:
    """403 unless writes are enabled, the request is same-origin and carries the token."""
    if cfg.read_only:
        raise HTTPException(403, "read-only mode: actions are disabled")
    token = request.headers.get("x-csrf-token") or form_token or ""
    if not same_origin(request) or not secrets.compare_digest(token, cfg.csrf):
        raise HTTPException(403, "cross-site request refused (missing or wrong CSRF token)")


def log_action(project: Path, action: str, title: str, detail: str = "", href: str | None = None,
               decisions: list[int] | None = None, node: int | None = None) -> None:
    entry = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "action": action,
             "title": title, "detail": detail, "href": href, "decisions": decisions or [],
             "node": node}
    path = Path(project) / ".notbydefault" / data.ACTIVITY_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def serve(project: Path, port: int = 8765, read_only: bool = False, open_browser: bool = True) -> int:
    import uvicorn
    import webbrowser
    project = Path(project).resolve()
    if not project.is_dir():
        print(f"Not a directory: {project}")
        return 1
    if not data.db_path(project).exists():
        print(f"No Decision Graph at {data.db_path(project)} yet; the UI will show it once "
              "`init` or the hook creates it.")
    url = f"http://127.0.0.1:{port}/"
    print(f"NotByDefault UI for {project}{' (read-only)' if read_only else ''}: {url}  "
          "(Ctrl+C to stop)")
    if open_browser:
        threading.Timer(1.0, webbrowser.open, (url,)).start()
    uvicorn.run(create_app(project, read_only), host="127.0.0.1", port=port, log_level="warning")
    return 0
