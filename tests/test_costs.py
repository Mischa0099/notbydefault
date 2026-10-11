"""`notbydefault export-costs`: business/costs.csv + costs.md from the graph and results only."""
import csv
import json

import pytest

from notbydefault.cli import main as cli
from notbydefault.costs import export
from notbydefault.graph import Graph
from test_fit_check import client_for, make_project, run

CUR, REC = "groq:openai/gpt-oss-120b", "gemini:gemini-3.1-flash-lite"


def stat(model, setup, monthly):
    name = model if setup == "alone" else f"{model} + verifier (<=2 retries)"
    return name, {"model": model, "setup": setup, "name": name, "monthly_cost": monthly}


def write_results(project, did, stamp, spent, made, hits, costs, judge_costs=(), users=1000,
                  cpu=30, budget=50.0, kind="switch", cur=400.0, rec=300.0, free=True):
    stats = dict([stat(CUR, "alone", cur), stat(REC, "alone", rec), stat(REC, "verified", 350.0)])
    data = {"decision_id": did, "run_at": f"2026-10-{stamp}T10:00:00+00:00",
            "rows": [{"cost": c} for c in costs], "judge": [{"cost": c} for c in judge_costs],
            "spent": spent, "requests": {"made": made, "cache_hits": hits},
            "pricing": {"free_tier": free}, "stopped": False,
            "assumptions": {"users": users, "calls_per_user": cpu, "budget": budget},
            "stats": stats, "verdict": {"kind": kind, "reference": [CUR, "alone"],
                                        "recommended": [REC if kind == "switch" else CUR,
                                                        "alone"]}}
    folder = project / ".notbydefault" / "results"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{did}-202610{stamp}T100000Z.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path.relative_to(project).as_posix()


def graph(project, *decisions):
    with Graph.for_project(project) as g:
        return [g.add_node("decision", label, {"kind": kind, "category": "ai-model",
                                               "status": "reviewed", **extra})
                for label, kind, extra in decisions]


def read(out):
    rows = list(csv.DictReader((out / "costs.csv").open(encoding="utf-8")))
    return {int(r["decision_id"]): r for r in rows}, (out / "costs.md").read_text(encoding="utf-8")


def test_end_to_end_from_a_real_fit_check_makes_no_model_calls(tmp_path, monkeypatch):
    p, did = make_project(tmp_path / "proj")
    client = client_for(p)
    code, report = run(p, did, client=client)
    assert code == 0, report
    from notbydefault.fit import providers
    monkeypatch.setattr(providers.ProviderSet, "chat",
                        lambda *a, **k: pytest.fail("export-costs made a model call"))
    out = tmp_path / "business"
    assert export(p, out, out=lambda _: None) == 0
    rows, md = read(out)
    with Graph.for_project(p) as g:
        res = json.loads((p / g.get_node(did)["data"]["results"]).read_text())
    r = rows[did]
    by = {(s["model"], s["setup"]): s for s in res["stats"].values()}
    cur, rec = by[tuple(res["verdict"]["reference"])], by[tuple(res["verdict"]["recommended"])]
    assert float(r["current_monthly_usd"]) == round(cur["monthly_cost"], 2)
    assert float(r["recommended_monthly_usd"]) == round(rec["monthly_cost"], 2)
    assert float(r["savings_monthly_usd"]) == round(cur["monthly_cost"] - rec["monthly_cost"], 2)
    assert float(r["our_list_cost_usd"]) == round(res["spent"], 6)
    assert int(r["our_requests_made"]) == res["requests"]["made"]
    full = sum(x["cost"] for x in res["rows"]) + sum(j["cost"] for j in res["judge"])
    assert float(r["fit_check_list_cost_usd"]) == round(full, 6)
    assert "## Unit economics per customer" in md and "**ASSUMPTION**" in md


def test_per_decision_costs_savings_and_unchecked_decisions(tmp_path):
    a, b = graph(tmp_path, ("openai/gpt-oss-120b", "model", {}), ("MongoDB", "service", {}))
    write_results(tmp_path, a, "10", spent=0.02, made=40, hits=2, costs=[0.001, 0.002],
                  judge_costs=[0.0005])
    assert export(tmp_path, tmp_path / "b", out=lambda _: None) == 0
    rows, md = read(tmp_path / "b")
    r = rows[a]
    assert (r["current_setup"], r["recommended_setup"]) == (CUR, REC)
    assert (r["current_monthly_usd"], r["recommended_monthly_usd"], r["savings_monthly_usd"]) == \
        ("400.0", "300.0", "100.0")
    assert r["fit_check_list_cost_usd"] == "0.0035" and r["our_list_cost_usd"] == "0.02"
    assert (r["our_requests_made"], r["our_cache_hits"], r["free_tier"]) == ("40", "2", "True")
    assert rows[b]["verdict"] == "" and rows[b]["checks_run"] == "0"
    assert "| not checked |" in md
    assert "Total savings if every recommendation is applied: $100.00/month" in md


def test_keep_verdict_saves_nothing(tmp_path):
    a, = graph(tmp_path, ("m", "model", {}))
    write_results(tmp_path, a, "10", 0.01, 10, 0, [0.001], kind="keep")
    export(tmp_path, tmp_path / "b", out=lambda _: None)
    r = read(tmp_path / "b")[0][a]
    assert r["recommended_setup"] == CUR and r["savings_monthly_usd"] == "0.0"


def test_linked_check_wins_and_every_check_is_listed(tmp_path):
    a, = graph(tmp_path, ("m", "model", {}))
    old = write_results(tmp_path, a, "09", 0.01, 10, 0, [0.001], rec=380.0)
    write_results(tmp_path, a, "11", 0.03, 30, 5, [0.003])
    with Graph.for_project(tmp_path) as g:
        g.update_data(a, results=old)       # the decision points at the older check
    export(tmp_path, tmp_path / "b", out=lambda _: None)
    rows, md = read(tmp_path / "b")
    assert rows[a]["results_file"] == old and rows[a]["savings_monthly_usd"] == "20.0"
    assert rows[a]["checks_run"] == "2"
    assert md.count(f"| #{a} |") == 2


def test_unit_economics_use_measured_cost_and_labelled_assumptions(tmp_path):
    a, b = graph(tmp_path, ("m", "model", {}), ("n", "model", {}))
    write_results(tmp_path, a, "10", spent=0.02, made=40, hits=0, costs=[0.01])
    write_results(tmp_path, b, "11", spent=0.04, made=20, hits=0, costs=[0.05])
    export(tmp_path, tmp_path / "b", checks_per_month=5, projects_per_customer=2,
           out=lambda _: None)
    md = read(tmp_path / "b")[1]
    assert "| checks per project per month | 5 | **ASSUMPTION** (`--checks-per-month`) |" in md
    assert "| projects per customer | 2 | **ASSUMPTION** (`--projects-per-customer`) |" in md
    assert "| our cost per check | $0.0300 |" in md          # mean of 0.02 and 0.04
    assert "| requests per check | 30 |" in md
    assert "| our LLM cost per project per month | $0.1500 |" in md      # 5 x 0.03
    assert "| **our LLM cost per customer per month** | **$0.3000** |" in md
    assert "| alternative: comparison calls only | $0.3000 |" in md      # 5 x 0.03 x 2
    assert "2 of 2 check(s) ran on provider free tiers" in md


def test_savings_are_not_totalled_across_different_volumes(tmp_path):
    a, b = graph(tmp_path, ("m", "model", {}), ("n", "model", {}))
    write_results(tmp_path, a, "10", 0.01, 1, 0, [0.0], users=500)
    write_results(tmp_path, b, "11", 0.01, 1, 0, [0.0], users=20000)
    export(tmp_path, tmp_path / "b", out=lambda _: None)
    md = read(tmp_path / "b")[1]
    assert "Total savings" not in md and "Savings are not totalled" in md
    assert "500 users x 30 calls" in md and "20,000 users x 30 calls" in md


def test_no_checks_yet_and_bad_results_files(tmp_path):
    graph(tmp_path, ("m", "model", {}))
    folder = tmp_path / ".notbydefault" / "results"
    folder.mkdir(parents=True)
    (folder / "1-broken.json").write_text("{not json")
    (folder / "2-other.json").write_text(json.dumps({"something": "else"}))
    assert export(tmp_path, tmp_path / "b", out=lambda _: None) == 0
    md = read(tmp_path / "b")[1]
    assert "| no fit checks yet |" in md and "| our cost per check | $0.0000 |" in md
    assert "No savings to total" in md


def test_errors(tmp_path):
    msgs = []
    assert export(tmp_path, tmp_path / "b", out=msgs.append) == 1 and "No Decision Graph" in msgs[0]
    graph(tmp_path, ("m", "model", {}))
    assert export(tmp_path, tmp_path / "b", checks_per_month=-1, out=msgs.append) == 1
    assert not (tmp_path / "b").exists()


def test_cli(tmp_path, monkeypatch):
    (tmp_path / "p").mkdir()
    a, = graph(tmp_path / "p", ("m", "model", {}))
    write_results(tmp_path / "p", a, "10", 0.01, 3, 0, [0.0])
    monkeypatch.chdir(tmp_path)
    assert cli(["export-costs", "--project", "p", "--checks-per-month", "2"]) == 0
    assert (tmp_path / "business" / "costs.csv").exists()
    assert "| checks per project per month | 2 |" in (tmp_path / "business" / "costs.md").read_text()
