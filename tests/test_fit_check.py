import json
from pathlib import Path

import pytest

from conftest import FakeClient
from notbydefault.cli import main as cli
from notbydefault.fit import labels
from notbydefault.fit.run import CheckOptions, check_model_decision
from notbydefault.graph import Graph

APP = '''from openai import OpenAI

client = OpenAI()


def summarize_notes(texts):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Summarize the user's notes in a few concise sentences."},
            {"role": "user", "content": "\\n".join(f"- {t}" for t in texts)},
        ],
    )
    return response.choices[0].message.content
'''


@pytest.fixture
def project(tmp_path):
    cli(["init", str(tmp_path), "--goal", "Notes API", "--users", "500", "--budget", "25",
         "--team", "2", "--no-input"])
    (tmp_path / "app.py").write_text(APP)
    with Graph.for_project(tmp_path) as g:
        did = g.add_node("decision", "gpt-4o", {"kind": "model", "category": "ai-model",
                                                "status": "unreviewed", "file": "app.py",
                                                "files": ["app.py"]})
        for n in g.nodes():
            if n["type"] in ("requirement", "constraint"):
                g.add_edge(did, n["id"], "depends_on")
            if n["type"] == "goal":
                g.add_edge(did, n["id"], "serves")
        code = g.add_node("code", "app.py", {"path": "app.py"})
        g.add_edge(did, code, "affects")
    return tmp_path, did


def run(project_dir, did, client, **opts):
    out = []
    opts.setdefault("yes", True)
    opts.setdefault("calls_per_user", 30)
    code = check_model_decision(project_dir, did, CheckOptions(workers=2, **opts), client=client,
                                ask=lambda _: "", out=out.append)
    return code, "\n".join(out)


def test_full_check_writes_report_graph_and_files(project):
    p, did = project
    client = FakeClient()
    code, report = run(p, did, client)
    assert code == 0, report
    nbd = p / ".notbydefault"

    # step 1-3 artifacts
    assert json.loads((nbd / "callsites" / f"{did}.json").read_text())["task"]
    samples = (nbd / "samples" / f"{did}.jsonl").read_text().splitlines()
    assert len(samples) == 20 and all(json.loads(s)["source"] == "synthetic" for s in samples)
    assert (nbd / "verifiers" / f"{did}.py").exists()
    assert (nbd / "candidates.yaml").exists()

    # verdict: haiku alone matches gpt-4o's pass rate and quality at a fraction of the cost
    assert "Switch to anthropic/claude-haiku-5.5:" in report
    assert "SYNTHETIC" in report and "pass rate (95% CI)" in report
    assert "fits budget?" in report and "Calls per user" not in report  # assumption is in prose
    assert "30 calls per user per month (ASSUMPTION)" in report

    # failure bank: gemini rambles first time, llama invents facts every time
    bank = [json.loads(l) for l in (nbd / "failures" / f"{did}.jsonl").read_text().splitlines()]
    assert {b["model"] for b in bank} == {"google/gemini-3.8-flash", "meta-llama/llama-4-scout"}
    assert {"max_words"} <= {f for b in bank for f in b["failed"]}
    assert {"llm_rubric"} <= {f for b in bank for f in b["failed"]}

    results = json.loads(next((nbd / "results").glob(f"{did}-*.json")).read_text())
    by = {(r["model"], r["setup"], r["sample"]): r for r in results["rows"]}
    gem_alone, gem_ver = by[("google/gemini-3.8-flash", "alone", 0)], by[("google/gemini-3.8-flash", "verified", 0)]
    assert not gem_alone["passed"] and gem_ver["passed"] and gem_ver["attempts"] == 2
    assert gem_ver["cost"] > gem_alone["cost"]          # retries + verifier calls are paid for
    assert not by[("meta-llama/llama-4-scout", "verified", 0)]["passed"]
    assert by[("meta-llama/llama-4-scout", "verified", 0)]["attempts"] == 3   # 1 + 2 retries

    # judge was blind: the prompt never names a model
    assert results["pricing"]["source"] == "https://openrouter.ai/api/v1/models"

    # step 8: graph + decision record
    with Graph.for_project(p) as g:
        d = g.get_node(did)
        assert d["data"]["status"] == "reviewed" and d["data"]["verdict"] == "switch"
        assert d["data"]["recommended"] == "anthropic/claude-haiku-5.5"
        kinds = {n["data"].get("kind") for n in g.neighbors(did, ("supported_by",))}
        assert kinds == {"fit_test", "pricing", "verifiers"}
        rejected = {n["label"]: n["data"]["reason"] for n in g.neighbors(did, ("rejects",))}
        assert "meta-llama/llama-4-scout" in rejected and "pass rate" in rejected["meta-llama/llama-4-scout"]
    [record] = (p / "decisions").glob("0001-ai-model-gpt-4o.md")
    assert "Change recommended: anthropic/claude-haiku-5.5" in record.read_text()


def test_second_run_reuses_inputs_and_runs_failures_first(project):
    p, did = project
    run(p, did, FakeClient())
    client = FakeClient()
    code, report = run(p, did, client)
    assert code == 0
    kinds = [k for _, k in client.calls]
    assert "extract" not in kinds and "samples" not in kinds and "verifiers" not in kinds
    results = sorted((p / ".notbydefault" / "results").glob(f"{did}-*.json"))[-1]
    samples = json.loads(results.read_text())["samples"]
    assert samples[0]["source"] == "failure-bank"
    assert "failure-bank" in report
    assert (p / "decisions" / "0002-ai-model-gpt-4o.md").exists()


def test_estimate_over_max_cost_spends_nothing(project):
    p, did = project
    client = FakeClient()
    code, report = run(p, did, client, max_cost=0.0001)
    assert code == 2 and "exceeds --max-cost" in report
    assert client.calls == []


def test_cancel_at_confirmation(project):
    p, did = project
    client = FakeClient()
    out = []
    code = check_model_decision(p, did, CheckOptions(calls_per_user=30), client=client,
                                ask=lambda _: "n", out=out.append)
    assert code == 1 and client.calls == [] and "nothing was spent" in out[-1]


def test_calls_per_user_asked_once_and_stored(project):
    p, did = project
    answers = iter(["45", "y"])
    check_model_decision(p, did, CheckOptions(workers=2), client=FakeClient(),
                         ask=lambda _: next(answers), out=lambda _: None)
    with Graph.for_project(p) as g:
        [node] = [n for n in g.nodes("requirement") if n["data"].get("key") == "calls_per_user"]
        assert node["data"]["value"] == 45
    asked = []
    check_model_decision(p, did, CheckOptions(workers=2, yes=True), client=FakeClient(),
                         ask=lambda q: asked.append(q) or "", out=lambda _: None)
    assert asked == []


def test_service_decision_is_milestone_3(tmp_path):
    with Graph.for_project(tmp_path) as g:
        did = g.add_node("decision", "MongoDB", {"kind": "service", "category": "database"})
    out = []
    assert check_model_decision(tmp_path, did, CheckOptions(), client=FakeClient(),
                                out=out.append) == 0
    assert out[-1] == "service fit: coming in Milestone 3"


def test_label_measures_verifier_agreement(project):
    p, did = project
    run(p, did, FakeClient())
    answers = iter(["p"] * 10)     # I think everything is fine
    out = []
    assert labels.label(p, did, ask=lambda _: next(answers), out=out.append) == 0
    t = labels.agreement(p, did)
    assert t["n"] == 10
    assert t["false_fail"] == 10 - t["agree"] and t["false_fail"] >= 1   # verifier failed some
    assert any("Verifiers agree with you" in line for line in out)
    _, report = run(p, did, FakeClient())
    assert f"agree with your labels on {t['agree']}/10" in report
