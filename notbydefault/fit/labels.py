"""Verifier trust: hand-label outputs and measure how often the verifiers agree."""
from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path


def labels_path(project: Path, decision_id: int) -> Path:
    return project / ".notbydefault" / "labels" / f"{decision_id}.jsonl"


def results_dir(project: Path) -> Path:
    return project / ".notbydefault" / "results"


def latest_results(project: Path, decision_id: int) -> Path | None:
    files = sorted(results_dir(project).glob(f"{decision_id}-*.json"))
    return files[-1] if files else None


def load(project: Path, decision_id: int) -> list[dict]:
    path = labels_path(project, decision_id)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def agreement(project: Path, decision_id: int) -> dict:
    labels = load(project, decision_id)
    agree = sum(1 for l in labels if l["human_passed"] == l["verifier_passed"])
    return {
        "n": len(labels), "agree": agree, "rate": agree / len(labels) if labels else 0.0,
        "false_pass": sum(1 for l in labels if l["verifier_passed"] and not l["human_passed"]),
        "false_fail": sum(1 for l in labels if not l["verifier_passed"] and l["human_passed"]),
    }


def pick(results: dict, already: set[tuple[str, str]], n: int = 10, seed: int = 0) -> list[dict]:
    """Up to n unlabeled outputs, half that the verifiers failed and half they passed."""
    samples = results["samples"]
    seen, items = set(already), []
    for r in results["rows"]:
        if r.get("error"):
            continue
        key = (samples[r["sample"]]["user_message"], r["output"])
        if key in seen:
            continue
        seen.add(key)
        items.append({"input": key[0], "output": key[1], "model": r["model"],
                      "setup": r["setup"], "verifier_passed": r["passed"],
                      "verifier_reasons": r["reasons"]})
    rng = random.Random(seed)
    rng.shuffle(items)
    failed = [i for i in items if not i["verifier_passed"]]
    passed = [i for i in items if i["verifier_passed"]]
    chosen = failed[: n // 2] + passed[: n - min(len(failed), n // 2)]
    chosen += [i for i in failed[n // 2:] if len(chosen) < n]
    rng.shuffle(chosen)
    return chosen[:n]


def label(project: Path, decision_id: int, ask=input, out=print, n: int = 10) -> int:
    path = latest_results(project, decision_id)
    if not path:
        out(f"No check results for decision #{decision_id} yet. Run `check {decision_id}` first.")
        return 1
    results = json.loads(path.read_text(encoding="utf-8"))
    existing = load(project, decision_id)
    items = pick(results, {(l["input"], l["output"]) for l in existing}, n=n)
    if not items:
        out("Every output from the latest run is already labeled.")
    criteria = results.get("callsite", {}).get("rubric") or []
    out(f"Label outputs for decision #{decision_id}: {results.get('callsite', {}).get('task', '')}")
    if criteria:
        out("A good output: " + "; ".join(criteria))
    out("Answer p (pass), f (fail), s (skip) or q (quit). The verifiers' verdict is hidden.\n")
    new = []
    for k, item in enumerate(items, 1):
        out(f"--- {k}/{len(items)} ---\nINPUT:\n{item['input']}\n\nOUTPUT:\n{item['output']}\n")
        answer = ""
        while answer not in ("p", "f", "s", "q"):
            try:
                answer = ask("pass/fail? [p/f/s/q] ").strip().lower()[:1]
            except EOFError:
                answer = "q"
        if answer == "q":
            break
        if answer == "s":
            continue
        new.append(item | {"human_passed": answer == "p", "results_file": path.name,
                           "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    if new:
        lp = labels_path(project, decision_id)
        lp.parent.mkdir(parents=True, exist_ok=True)
        with open(lp, "a", encoding="utf-8") as f:
            for item in new:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    t = agreement(project, decision_id)
    if t["n"]:
        out(f"\nVerifiers agree with you on {t['agree']}/{t['n']} labeled outputs ({t['rate']:.0%}): "
            f"{t['false_pass']} false pass(es), {t['false_fail']} false fail(s).")
    return 0
