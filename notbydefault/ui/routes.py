"""Pages, fragments and JSON endpoints (read-only). Actions live in ui/actions.py."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import Response

from notbydefault.ui import data

FEED_LIMIT = 14


def unchanged(request: Request, version: str) -> bool:
    """A live poll that already shows this version: 204, so HTMX swaps nothing."""
    return request.query_params.get("v") == version


def register(app: FastAPI, cfg, page) -> None:
    project = cfg.project

    def overview_ctx():
        snap = data.snapshot(project)
        return {"snap": snap, "ov": data.overview(project, snap),
                "events": data.events(project, snap)[:FEED_LIMIT]}

    @app.get("/")
    def overview(request: Request):
        return page(request, "overview.html", **overview_ctx())

    @app.get("/fragments/overview")
    def overview_fragment(request: Request):
        version = data.version(project)
        if unchanged(request, version):
            return Response(status_code=204)
        return page(request, "_overview_live.html", **overview_ctx())

    @app.get("/api/version")
    def api_version():
        return {"version": data.version(project)}

    @app.get("/decisions")
    def decisions(request: Request, status: str = "", kind: str = ""):
        snap = data.snapshot(project)
        kinds = sorted({d["data"].get("kind") or "" for d in snap.of_type("decision")} - {""})
        return page(request, "decisions.html", rows=data.decision_rows(snap, status, kind),
                    status=status, kind=kind, kinds=kinds, statuses=data.STATUSES,
                    total=len(snap.of_type("decision")))

    @app.get("/decisions/{did}")
    def decision(request: Request, did: int):
        snap = data.snapshot(project)
        view = data.decision_view(project, snap, did)
        if view is None:
            return page(request, "missing.html", status_code=404, what=f"decision #{did}")
        return page(request, "decision.html", v=view)

    @app.get("/nodes/{nid}")
    def node(request: Request, nid: int):
        snap = data.snapshot(project)
        n = snap.nodes.get(nid)
        if n is None:
            return page(request, "missing.html", status_code=404, what=f"node #{nid}")
        if n["type"] == "decision":
            from fastapi.responses import RedirectResponse
            return RedirectResponse(f"/decisions/{nid}", status_code=303)
        impacts = data.requirement_impacts(snap, nid)
        return page(request, "node.html", n=n, out=snap.out(nid), into=snap.into(nid),
                    impacts=[(snap.nodes[i], st) for i, st in impacts],
                    path_link=data.rel_to(project, n["data"].get("path")),
                    fields=data.node_fields(n))

    @app.get("/files/{rel:path}")
    def files(request: Request, rel: str):
        p = data.safe_path(project, rel)
        if p is None:
            return page(request, "missing.html", status_code=404,
                        what=f"{rel} (only text files inside the project, never secrets)")
        raw = p.read_bytes()[:data.MAX_VIEW_BYTES].decode("utf-8", "replace")
        if p.suffix == ".json":
            try:
                import json
                raw = json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
            except ValueError:
                pass
        return page(request, "file.html", rel=rel, text=raw, markdown=p.suffix == ".md",
                    truncated=p.stat().st_size > data.MAX_VIEW_BYTES)

    @app.get("/graph")
    def graph(request: Request):
        snap = data.snapshot(project)
        reqs = snap.of_type("requirement") + snap.of_type("constraint")
        return page(request, "graph.html", types=data.NODE_TYPES, empty=not snap.nodes,
                    decisions=snap.of_type("decision"), reqs=sorted(reqs, key=lambda n: n["id"]))

    @app.get("/api/graph")
    def api_graph(since: int = 0):
        """Graph elements; with `since`, only nodes added after that id (live polling)."""
        snap = data.snapshot(project)
        return dict(data.graph_elements(snap, since), version=data.version(project))

    @app.get("/timeline")
    def timeline(request: Request):
        snap = data.snapshot(project)
        days: list[tuple[str, list]] = []
        for e in reversed(data.events(project, snap)):
            day = e.at.astimezone().strftime("%A %d %B %Y")
            if not days or days[-1][0] != day:
                days.append((day, []))
            days[-1][1].append(e)
        return page(request, "timeline.html", days=days)

    @app.get("/evals")
    def evals(request: Request):
        return page(request, "evals.html", ev=data.evals(), trace_url=data.trace_url)

    @app.get("/costs")
    def costs(request: Request):
        md, _ = data.costs_markdown(project, data.snapshot(project))
        if md.startswith("# "):
            md = md.split("\n", 1)[1]          # the page has its own title
        return page(request, "costs.html", md=md)

    @app.get("/costs.csv")
    def costs_csv():
        _, csv_text = data.costs_markdown(project, data.snapshot(project))
        return Response(csv_text, media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="costs.csv"'})

    @app.get("/fragments/node/{nid}")
    def node_panel(request: Request, nid: int):
        snap = data.snapshot(project)
        n = snap.nodes.get(nid)
        if n is None:
            return page(request, "_node_panel.html", status_code=404, n=None)
        return page(request, "_node_panel.html", n=n, status=data.status_of(n),
                    fields=data.node_fields(n)[:14], out=snap.out(nid), into=snap.into(nid),
                    impacts=[(snap.nodes[i], st) for i, st in data.requirement_impacts(snap, nid)],
                    why=data.why_of(n))
