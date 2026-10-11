"""UI actions. Each one calls the existing Python function (records.review, stale.require,
resolve.Resolver in dry-run mode), is refused with --read-only, needs the CSRF token and a
same-origin request, runs one at a time, and is logged to the activity feed."""
from __future__ import annotations

from fastapi import FastAPI, Request

from notbydefault.graph import Graph
from notbydefault.ui import data
from notbydefault.ui.app import check_write, log_action

MAX_TEXT = 2000


async def _form(request: Request) -> dict:
    try:
        form = await request.form()
    except Exception:          # noqa: BLE001  (no or malformed body)
        return {}
    return {k: str(v) for k, v in form.items()}


def register(app: FastAPI, cfg, page) -> None:
    project = cfg.project

    def result(request: Request, title: str, ok: bool = True, **ctx):
        name = "_action_result.html"
        if request.headers.get("hx-request"):
            return page(request, name, title=title, ok=ok, **ctx)
        return page(request, "action_page.html", title=title, ok=ok, inner=name, **ctx)

    @app.post("/decisions/{did}/review")
    async def review(request: Request, did: int):
        form = await _form(request)
        check_write(request, cfg, form.get("csrf_token"))
        reason = (form.get("reason") or "").strip()[:MAX_TEXT]
        mark = form.get("mark", "reviewed") != "reason"
        if not mark and not reason:
            return result(request, "Write a reason first.", ok=False, status_code=400)
        from notbydefault.records import review as review_fn
        with cfg.lock, Graph.for_project(project) as g:
            d = g.get_node(did)
            if d is None or d["type"] != "decision":
                return result(request, f"No decision #{did}.", ok=False, status_code=404)
            before = data.status_of(d)
            path = review_fn(project, g, did, reason, mark_reviewed=mark)
        rel = data.rel_to(project, str(path))
        title = (f"#{did} marked reviewed" if mark else f"Reason added to #{did}")
        log_action(project, "review" if mark else "reason", f"{title}: {d['label']}",
                   reason or f"{before} → reviewed", f"/decisions/{did}", [did])
        return result(request, title, lines=[f"status: {before} → {'reviewed' if mark else before}"]
                      + ([f"reason: {reason}"] if reason else []), record=rel,
                      reload=f"/decisions/{did}")

    @app.post("/requirements")
    async def add_requirement(request: Request):
        form = await _form(request)
        check_write(request, cfg, form.get("csrf_token"))
        text = " ".join((form.get("text") or "").split())[:MAX_TEXT]
        if not text:
            return result(request, "Write the requirement first.", ok=False, status_code=400)
        from notbydefault.stale import require
        if not cfg.lock.acquire(blocking=False):
            return result(request, "Another action is still running; try again in a moment.",
                          ok=False, status_code=409)
        try:
            before = {d["id"]: data.status_of(d) for d in data.snapshot(project).of_type("decision")}
            lines: list[str] = []
            client = cfg.client_factory() if cfg.client_factory else None
            code = require(project, text, client=client, out=lines.append)
        finally:
            cfg.lock.release()
        snap = data.snapshot(project)
        reqs = [r for r in snap.of_type("requirement") if r["label"] == text]
        rid = reqs[-1]["id"] if reqs else None
        changed = [(snap.nodes[i], before.get(i, "unreviewed"), data.status_of(snap.nodes[i]))
                   for i in sorted(before) if i in snap.nodes
                   and data.status_of(snap.nodes[i]) != before[i]]
        log_action(project, "require", f"Requirement added: {text}",
                   f"{len(changed)} decision(s) changed", f"/graph?req={rid}" if rid else None,
                   [d["id"] for d, _, _ in changed], node=rid)
        return result(request, f"Requirement #{rid}: {text}" if rid else "Requirement not added",
                      ok=code == 0, changed=changed, report="\n".join(lines), rid=rid)

    @app.post("/decisions/{did}/resolve-dry-run")
    async def resolve_dry_run(request: Request, did: int):
        form = await _form(request)
        check_write(request, cfg, form.get("csrf_token"))
        from notbydefault.resolve import Resolver, ResolveOptions
        if not cfg.lock.acquire(blocking=False):
            return result(request, "Another action is still running; try again in a moment.",
                          ok=False, status_code=409)
        lines: list[str] = []
        try:
            client = cfg.client_factory() if cfg.client_factory else None
            r = Resolver(project, did, ResolveOptions(apply=False), ask=lambda _q: "n",
                         out=lines.append, client=client, entire=cfg.entire)
            code = r.run()
        finally:
            cfg.lock.release()
        diffs = [e.diff() for e in getattr(r, "edits", [])]
        log_action(project, "resolve_dry_run", f"Resolve dry run for #{did}",
                   f"{len(diffs)} file(s) would change; nothing was written", f"/decisions/{did}",
                   [did])
        return result(request, f"Resolve #{did}: dry run (nothing changed)", ok=code in (0, 2),
                      report="\n".join(lines), diff="".join(diffs),
                      trace_id=getattr(r.tracer, "trace_id", None) if r.tracer else None)
