"""`notbydefault export-costs`: business/costs.csv and business/costs.md, from files only.

Reads the Decision Graph and the fit checks' results files; makes no model calls.

Per decision (its latest fit check):
  - today's monthly cost vs the recommended setup's, and the savings per month, at the volume
    and budget the check assumed (users x calls per user, from the graph)
  - fit-check cost: list price of every model, verifier and judge call in the comparison,
    including calls answered from the response cache (setup calls not included)
Per fit check (every results file), our own cost:
  - requests made, cache hits, and the list-price cost of the requests actually made
    (`spent`: cache hits cost $0; includes setup calls)
Unit economics per customer: our measured cost per check x checks per project per month
(an ASSUMPTION, labelled) x projects per customer (an ASSUMPTION, labelled).
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

from notbydefault.graph import Graph, db_path

DEFAULT_CHECKS_PER_MONTH = 4.0
DEFAULT_PROJECTS_PER_CUSTOMER = 1.0
CSV_COLUMNS = ["decision_id", "label", "kind", "category", "status", "verdict",
               "current_setup", "recommended_setup", "current_monthly_usd",
               "recommended_monthly_usd", "savings_monthly_usd", "assumed_users",
               "assumed_calls_per_user", "budget_monthly_usd", "checks_run", "latest_check_at",
               "fit_check_list_cost_usd", "our_requests_made", "our_cache_hits",
               "our_list_cost_usd", "free_tier", "results_file"]


@dataclass
class Check:
    path: str               # relative to the project
    decision_id: int
    run_at: str
    requests_made: int | None
    cache_hits: int | None
    spent: float            # list price of the requests made
    comparison_cost: float  # list price of every comparison/judge call, cache hits included
    free_tier: bool
    stopped: bool
    data: dict = field(repr=False, default_factory=dict)


def load_checks(project: Path) -> list[Check]:
    out = []
    for p in sorted((project / ".notbydefault" / "results").glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "decision_id" not in d or "rows" not in d:
            continue
        req = d.get("requests") or {}
        out.append(Check(
            path=p.relative_to(project).as_posix(), decision_id=int(d["decision_id"]),
            run_at=str(d.get("run_at", "")), requests_made=req.get("made"),
            cache_hits=req.get("cache_hits"), spent=float(d.get("spent") or 0),
            comparison_cost=sum(float(r.get("cost") or 0) for r in d["rows"])
            + sum(float(j.get("cost") or 0) for j in d.get("judge") or []),
            free_tier=bool((d.get("pricing") or {}).get("free_tier")),
            stopped=bool(d.get("stopped")), data=d))
    return sorted(out, key=lambda c: c.run_at)


def _stat(stats: dict, model_setup) -> dict | None:
    if not model_setup:
        return None
    model, setup = model_setup
    return next((s for s in stats.values()
                 if s.get("model") == model and s.get("setup") == setup), None)


def decision_rows(project: Path, graph: Graph, checks: list[Check]) -> list[dict]:
    rows = []
    for d in graph.nodes("decision"):
        data = d["data"]
        mine = [c for c in checks if c.decision_id == d["id"]]
        linked = next((c for c in mine if c.path == data.get("results")), None)
        latest = linked or (mine[-1] if mine else None)
        row = dict.fromkeys(CSV_COLUMNS, "")
        row.update(decision_id=d["id"], label=d["label"], kind=data.get("kind", ""),
                   category=data.get("category", ""), status=data.get("status", ""),
                   checks_run=len(mine))
        if latest is not None:
            res = latest.data
            verdict = res.get("verdict") or {}
            cur = _stat(res.get("stats") or {}, verdict.get("reference"))
            rec = _stat(res.get("stats") or {}, verdict.get("recommended"))
            a = res.get("assumptions") or {}
            row.update(verdict=verdict.get("kind", ""),
                       current_setup=cur["name"] if cur else "",
                       recommended_setup=rec["name"] if rec else "",
                       assumed_users=a.get("users"), assumed_calls_per_user=a.get("calls_per_user"),
                       budget_monthly_usd=a.get("budget"), latest_check_at=latest.run_at,
                       fit_check_list_cost_usd=round(latest.comparison_cost, 6),
                       our_requests_made=latest.requests_made, our_cache_hits=latest.cache_hits,
                       our_list_cost_usd=round(latest.spent, 6), free_tier=latest.free_tier,
                       results_file=latest.path)
            cm = (cur or {}).get("monthly_cost")
            rm = (rec or {}).get("monthly_cost")
            if cm is not None:
                row["current_monthly_usd"] = round(cm, 2)
            if rm is not None:
                row["recommended_monthly_usd"] = round(rm, 2)
            if cm is not None and rm is not None:
                row["savings_monthly_usd"] = round(cm - rm, 2)
        rows.append({k: ("" if v is None else v) for k, v in row.items()})
    return rows


def unit_economics(checks: list[Check], checks_per_month: float,
                   projects_per_customer: float) -> dict:
    n = len(checks)
    avg = (lambda xs: sum(xs) / len(xs) if xs else 0.0)
    per_check = avg([c.spent for c in checks])
    per_check_full = avg([c.comparison_cost for c in checks])
    made = [c.requests_made for c in checks if c.requests_made is not None]
    per_project = checks_per_month * per_check
    return {"checks": n, "checks_per_month": checks_per_month,
            "projects_per_customer": projects_per_customer,
            "cost_per_check": per_check, "comparison_cost_per_check": per_check_full,
            "requests_per_check": avg(made) if made else None,
            "cost_per_project_month": per_project,
            "cost_per_customer_month": per_project * projects_per_customer,
            "comparison_cost_per_customer_month":
                checks_per_month * per_check_full * projects_per_customer,
            "free_tier_checks": sum(c.free_tier for c in checks)}


def _money(v, places=2) -> str:
    return "-" if v in ("", None) else f"${float(v):,.{places}f}"


def _num(v) -> str:
    if v in ("", None):
        return "-"
    return f"{float(v):,.0f}" if float(v) == int(float(v)) else f"{float(v):,.2f}"


def volume(r: dict) -> str:
    if r["assumed_users"] == "" or r["assumed_calls_per_user"] == "":
        return "-"
    return f"{_num(r['assumed_users'])} users x {_num(r['assumed_calls_per_user'])} calls"


def savings_total(rows: list[dict]) -> str:
    """One total only when every saving was computed at the same volume."""
    priced = [r for r in rows if r["savings_monthly_usd"] != ""]
    if not priced:
        return "No savings to total: no decision has a fit check with a recommendation."
    volumes = {volume(r) for r in priced}
    if len(volumes) > 1:
        return ("Savings are not totalled: the fit checks assumed different volumes ("
                + "; ".join(f"#{r['decision_id']}: {volume(r)}" for r in priced)
                + "). Re-run the older checks at today's volume to compare them.")
    total = sum(float(r["savings_monthly_usd"]) for r in priced)
    return (f"Total savings if every recommendation is applied: {_money(total)}/month "
            f"(at {volumes.pop()} per month).")


def render_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def render_md(project_name: str, rows: list[dict], checks: list[Check], ue: dict) -> str:
    L = [f"# Costs: {project_name}", "",
         "Generated by `python -m notbydefault export-costs` from the Decision Graph and the fit "
         "checks' results files (no model calls). All prices are provider list prices recorded "
         "in each results file.", "",
         "## Per decision", "",
         "Monthly costs are at the volume each fit check assumed (users x calls per user, from "
         "the graph). Fit-check cost: list price of every model, verifier and judge call in the "
         "comparison, including calls answered from the response cache.", "",
         "| # | decision | verdict | today | today/month | recommended | recommended/month "
         "| savings/month | assumed volume | budget/month | fit-check cost | checks |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['decision_id']} | `{r['label']}` ({r['kind']}/{r['category']}, "
                 f"{r['status']}) | {r['verdict'] or 'not checked'} | "
                 f"{('`' + r['current_setup'] + '`') if r['current_setup'] else '-'} | "
                 f"{_money(r['current_monthly_usd'])} | "
                 f"{('`' + r['recommended_setup'] + '`') if r['recommended_setup'] else '-'} | "
                 f"{_money(r['recommended_monthly_usd'])} | {_money(r['savings_monthly_usd'])} | "
                 f"{volume(r)} | {_money(r['budget_monthly_usd'])} | {_money(r['fit_check_list_cost_usd'], 4)} "
                 f"| {r['checks_run']} |")
    L += ["", savings_total(rows), "",
          "## Our cost per fit check", "",
          "What each check cost us: the requests it actually made, at list price (cache hits "
          "cost $0; call-site analysis, sample and verifier generation included).", "",
          "| results file | decision | run at | requests made | cache hits | our cost (list) "
          "| comparison at list price | tier |", "|---|---|---|---|---|---|---|---|"]
    for c in checks:
        L.append(f"| `{c.path}` | #{c.decision_id} | {c.run_at[:16].replace('T', ' ')} | "
                 f"{_num(c.requests_made)} | {_num(c.cache_hits)} | {_money(c.spent, 4)} | "
                 f"{_money(c.comparison_cost, 4)} | "
                 f"{'free (paid $0)' if c.free_tier else 'paid'}"
                 f"{', stopped early' if c.stopped else ''} |")
    if not checks:
        L.append("| no fit checks yet | | | | | | | |")
    L += ["", "## Unit economics per customer", "",
          "| | value | basis |", "|---|---|---|",
          f"| checks per project per month | {ue['checks_per_month']:g} | **ASSUMPTION** "
          "(`--checks-per-month`) |",
          f"| projects per customer | {ue['projects_per_customer']:g} | **ASSUMPTION** "
          "(`--projects-per-customer`) |",
          f"| our cost per check | {_money(ue['cost_per_check'], 4)} | measured: mean over "
          f"{ue['checks']} check(s) of the list price of requests made |",
          f"| requests per check | {_num(ue['requests_per_check'])} | measured |",
          f"| our LLM cost per project per month | {_money(ue['cost_per_project_month'], 4)} | "
          "checks per month x cost per check |",
          f"| **our LLM cost per customer per month** | "
          f"**{_money(ue['cost_per_customer_month'], 4)}** | x projects per customer |",
          f"| alternative: comparison calls only | "
          f"{_money(ue['comparison_cost_per_customer_month'], 4)} | every model, verifier and "
          "judge call in the comparison at list price, cache hits included, setup calls "
          "excluded; neither a lower nor an upper bound |", "",
          f"{ue['free_tier_checks']} of {ue['checks']} check(s) ran on provider free tiers, so "
          "the amount actually paid was $0; the figures above are what the same requests cost "
          "at list price.", ""]
    return "\n".join(L)


def export(project: Path, out_dir: Path, checks_per_month: float = DEFAULT_CHECKS_PER_MONTH,
           projects_per_customer: float = DEFAULT_PROJECTS_PER_CUSTOMER, out=print) -> int:
    project = Path(project).resolve()
    if not db_path(project).exists():
        out(f"No Decision Graph at {db_path(project)}.")
        return 1
    if checks_per_month < 0 or projects_per_customer < 0:
        out("--checks-per-month and --projects-per-customer must not be negative.")
        return 1
    checks = load_checks(project)
    with Graph.for_project(project) as g:
        rows = decision_rows(project, g, checks)
    ue = unit_economics(checks, checks_per_month, projects_per_customer)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "costs.csv").write_text(render_csv(rows), encoding="utf-8", newline="")
    (out_dir / "costs.md").write_text(render_md(project.name, rows, checks, ue),
                                      encoding="utf-8", newline="")
    out(f"Wrote {out_dir / 'costs.csv'} and {out_dir / 'costs.md'}: {len(rows)} decision(s), "
        f"{len(checks)} fit check(s); our LLM cost per customer "
        f"${ue['cost_per_customer_month']:.4f}/month at {checks_per_month:g} check(s)/month "
        "(assumption).")
    return 0
