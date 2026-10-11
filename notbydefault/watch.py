"""`notbydefault watch [--auto-check] [--volume-traces FILE]`: re-open decisions whose assumptions broke.

For every decision with recorded assumptions (assumptions.py; a baseline is recorded for those
without), compare them with now: requirements in force, prices in fit/providers.yaml, new models
in the providers' model lists, and real call volume from neatlogs traces when NEATLOGS_TOKEN is
set. A decision whose assumption changed is re-opened: needs review, the broken assumption and
its cause recorded on the node and in its decision record, and its failure-bank cases run first
in the next check. Report: "re-opened #21: 'data region: any' broken by 'EU only'".

Real call volume: neatlogs documents reading a trace by id (GET /api/v1/public/traces/<id> and
/spans, NEATLOGS_TOKEN + NEATLOGS_PROJECT_ID, as `neatlogs trace get` does) but no endpoint that
lists traces, so pass the production trace ids of a window with --volume-traces FILE (one id per
line) and --volume-days N; LLM spans that name the decision's model are counted and scaled to 30
days. Without them the volume is reported as not measured.

--auto-check queues a fit check for each re-opened model decision, one at a time, within
--max-cost in total; a check stopped by a provider's daily limit stops the queue.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote, urlparse

from notbydefault.assumptions import _now, broken, describe, record_snapshot

PAGE_LIMIT, MAX_PAGES = 50, 40


def _fetch(url: str, headers: dict, timeout: float) -> tuple[int, bytes]:
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(1 << 20)
    except urllib.error.HTTPError as e:
        return e.code, b""


def trace_spans(trace_id: str, env=None, fetch=None, timeout: float = 5.0) -> list[dict] | None:
    """Every span of one trace through neatlogs' public read API, or None when unreadable."""
    env = os.environ if env is None else env
    token, project_id = env.get("NEATLOGS_TOKEN", "").strip(), env.get("NEATLOGS_PROJECT_ID", "").strip()
    if not token or not project_id:
        return None
    host = urlparse((env.get("NEATLOGS_HOST") or "https://app.neatlogs.com").strip())
    base = f"{host.scheme}://{host.netloc}/api/v1/public/traces/{quote(trace_id, safe='')}/spans"
    headers = {"Authorization": f"Bearer {token}", "x-project-id": project_id}
    fetch = fetch or _fetch
    spans, cursor = [], None
    for _ in range(MAX_PAGES):
        url = f"{base}?limit={PAGE_LIMIT}" + (f"&cursor={quote(cursor, safe='')}" if cursor else "")
        try:
            status, body = fetch(url, headers, timeout)
            data = json.loads(body).get("data") if 200 <= status < 300 else None
        except (OSError, ValueError, AttributeError):
            data = None
        if not isinstance(data, dict):
            return None
        spans += [s for s in data.get("spans") or [] if isinstance(s, dict)]
        cursor = (data.get("page") or {}).get("nextCursor")
        if not cursor:
            return spans
    return spans


def traces_volume(trace_ids: list[str], days: float, env=None, fetch=None):
    """volume_source for watch(): LLM spans naming the decision's model, scaled to 30 days."""
    cache: dict[str, list[dict] | None] = {}

    def source(project: Path, decision: dict):
        if decision["data"].get("kind") != "model" or not trace_ids or days <= 0:
            return None
        model = str(decision["data"].get("chosen") or decision["label"]).lower()
        calls, read = 0, 0
        for t in trace_ids:
            if t not in cache:
                cache[t] = trace_spans(t, env, fetch)
            spans = cache[t]
            if spans is None:
                continue
            read += 1
            calls += sum(1 for s in spans if "llm" in str(s.get("spanType") or "").lower()
                         and model in json.dumps(s).lower())
        if not read:
            return None
        return calls * 30.0 / days, (f"neatlogs: {calls} call(s) to {model} in {read} production "
                                     f"trace(s) over {days:g} days")
    return source


def volume_note(env=None, traces_file: str | None = None) -> str:
    env = os.environ if env is None else env
    if not env.get("NEATLOGS_TOKEN"):
        return "NEATLOGS_TOKEN not set"
    if not traces_file:
        return ("neatlogs documents no trace-listing endpoint; pass --volume-traces FILE with "
                "production trace ids")
    return "none of the listed traces could be read"


def reopen(project: Path, graph, decision: dict, items: list[dict]) -> None:
    from notbydefault.fit import failures as failures_mod
    from notbydefault.records import write_record
    did, data = decision["id"], decision["data"]
    first = items[0]
    snap = dict(data.get("assumptions") or {})
    snap.update(broken=items, broken_at=_now())
    banked = len(failures_mod.load(project, did))
    graph.update_data(did, assumptions=snap, status="needs_review",
                      review_reason=describe(first) + (f" (+{len(items) - 1} more)"
                                                       if len(items) > 1 else ""),
                      review_basis="watched assumption", review_source=first.get("source", ""),
                      review_since=_now(), status_before_review=data.get("status"),
                      rerun_first=banked)
    write_record(project, graph, did, "re-opened by watch: " + "; ".join(describe(b) for b in items)
                 + (f"; {banked} failure-bank case(s) run first next check" if banked else ""))


def watch(project: Path, auto_check: bool = False, max_cost: float = 1.0, client=None,
          tracer=None, out=print, volume_source=None, check_fn=None,
          volume_note: str = "") -> int:
    """`notbydefault watch`. volume_source(project, decision) -> (calls/month, source) or None."""
    from notbydefault.fit.providers import load_catalog
    from notbydefault.fit.tracing import make_tracer
    from notbydefault.graph import Graph, db_path
    project = Path(project).resolve()
    if not db_path(project).exists():
        out(f"No Decision Graph at {db_path(project)}.")
        return 1
    catalog = getattr(client, "catalog", None) or load_catalog(project)
    tracer = tracer or make_tracer()
    reopened, measured = [], False
    with Graph.for_project(project) as g, tracer.root("notbydefault.watch", {}):
        with tracer.step("watch_assumptions", "verification",
                         reason="re-evaluate what each verdict assumed: requirements, prices, "
                                "model lists, volume"):
            for d in g.nodes("decision"):
                if not (d["data"].get("assumptions") or {}).get("items"):
                    record_snapshot(project, g, d["id"], catalog)
                    out(f"baseline   #{d['id']} {d['label']}: assumptions recorded")
                    continue
                obs = volume_source(project, d) if volume_source else None
                measured = measured or obs is not None
                items = broken(project, g, d, catalog, *(obs or (None, "")))
                if not items:
                    out(f"ok         #{d['id']} {d['label']}")
                    continue
                for b in items:
                    tracer.event("assumption_broken", "verification", f"#{d['id']}: {describe(b)}",
                                 decision_id=d["id"], key=b["key"])
                if d["data"].get("status") in ("needs_review", "stale"):
                    snap = dict(d["data"]["assumptions"], broken=items, broken_at=_now())
                    g.update_data(d["id"], assumptions=snap)
                    out(f"already    #{d['id']} {d['label']} is {d['data']['status']}; "
                        + "; ".join(describe(b) for b in items))
                    continue
                reopen(project, g, d, items)
                reopened.append(d)
                out(f"re-opened #{d['id']}: {describe(items[0])}"
                    + "".join(f"\n           also {describe(b)}" for b in items[1:]))
    if not measured:
        out(f"real call volume: not measured ({volume_note or 'no volume source'})")
    code = 0
    if auto_check:
        code = _auto_check(project, [d for d in reopened if d["data"].get("kind") == "model"],
                           max_cost, client, out, check_fn)
    if tracer.trace_id:
        out(f"\nneatlogs trace id: {tracer.trace_id} (workflow notbydefault)")
    tracer.flush()
    return code


def _auto_check(project: Path, queue: list[dict], max_cost: float, client, out,
                check_fn=None) -> int:
    """Run the queued fit checks one by one within --max-cost in total."""
    import json
    from notbydefault.fit.run import CheckOptions, check_model_decision
    check_fn = check_fn or check_model_decision
    left = max_cost
    for d in queue:
        if left <= 0.0001:
            out(f"queued     #{d['id']}: not checked, --max-cost {max_cost:.2f} used up")
            continue
        out(f"checking   #{d['id']} {d['label']} (up to ${left:.2f}; failure-bank cases first)")
        folder = project / ".notbydefault" / "results"
        before = set(folder.glob(f"{d['id']}-*.json"))
        code = check_fn(project, d["id"], CheckOptions(max_cost=left, yes=True), client=client)
        new = sorted(set(folder.glob(f"{d['id']}-*.json")) - before)
        try:     # only this run's results file counts against the cap
            res = json.loads(new[-1].read_text(encoding="utf-8")) if new else {}
        except (OSError, ValueError):
            res = {}
        left -= float(res.get("spent") or 0)
        if code != 0 or res.get("stopped"):
            out(f"stopped    #{d['id']}: the check ended early (code {code}); remaining "
                "checks stay queued (quota or --max-cost)")
            return code or 0
    return 0
