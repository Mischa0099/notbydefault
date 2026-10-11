"""`notbydefault reverify`: re-run a fit check's saved outputs through a (new) verifier file.

Read-only. Every output in the results file is checked twice, with the verifiers the run used
and with the given file, so a verifier change shows its before/after effect without spending
a single request: deterministic checks need no model, and an LLM rubric is answered from the
response cache only (outputs it doesn't cover are reported, never requested).

A verified setup's saved output is the one that passed after its retries. When the new
verifiers fail it, production would retry; those retries are new model calls, so they are
counted, not made.
"""
from __future__ import annotations

import glob
import json
from dataclasses import dataclass, field
from pathlib import Path

from notbydefault.fit.evaluate import VERIFIED
from notbydefault.fit.verifiers import Verifiers, load, verify
from notbydefault.llm import LLMError


class NotCached(LLMError):
    pass


class CacheOnlyClient:
    """A ProviderSet that answers from the response cache and refuses everything else."""

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def chat(self, model, messages, **kw):
        if not self.inner.is_cached(model, messages, kw.get("temperature", 0.0),
                                    kw.get("max_tokens", 1024), kw.get("json_mode", False),
                                    kw.get("extra_body")):
            raise NotCached(f"{model}: not in the response cache (reverify makes no requests)")
        return self.inner.chat(model, messages, **kw)


@dataclass
class SetupCounts:
    rows: int = 0
    passed_before: int = 0
    passed_after: int = 0
    false_pass_before: int = 0     # passed every production check but wrong vs the reference
    false_pass_after: int = 0
    caught: list[str] = field(default_factory=list)       # passed before, fail now
    retries_needed: int = 0        # verified outputs production would now retry (not run)


def production(v: Verifiers) -> Verifiers:
    """The checks production runs (evaluation-only reference checks removed)."""
    return Verifiers([c for c in v.checks if not c.needs_reference], v.rubric, v.path, v.sha)


def latest_results(project: Path, decision_id: int) -> Path:
    paths = sorted(glob.glob(str(project / ".notbydefault" / "results" / f"{decision_id}-*.json")))
    if not paths:
        raise FileNotFoundError(f"no results for decision #{decision_id} in {project}")
    return Path(paths[-1])


def reverify(project: Path, decision_id: int, new_verifiers: Path, results: Path | None = None,
             out=print) -> dict:
    from notbydefault.env import load_env
    from notbydefault.fit.evaluate import Caller, SpendGuard
    from notbydefault.fit.providers import ProviderSet, load_catalog
    from notbydefault.fit.tracing import Tracer

    project = Path(project).resolve()
    results = Path(results) if results else latest_results(project, decision_id)
    data = json.loads(results.read_text(encoding="utf-8"))
    old_path = Path(data["verifiers"]["path"])
    if not old_path.exists():
        old_path = project / ".notbydefault" / "verifiers" / f"{decision_id}.py"
    before, after = production(load(old_path)), production(load(Path(new_verifiers)))
    load_env()
    client = CacheOnlyClient(ProviderSet(load_catalog(project),
                                         cache_dir=project / ".notbydefault" / "cache"))
    caller = Caller(client, SpendGuard(0.0), Tracer())
    vmodel = data["config"].get("verifier_model")
    samples = data["samples"]
    counts: dict[str, SetupCounts] = {}
    not_cached = 0
    for r in data["rows"]:
        if r.get("error") or not r.get("output"):
            continue
        s = samples[r["sample"]]
        name = r["model"] + (" + verifier" if r["setup"] == VERIFIED else "")
        c = counts.setdefault(name, SetupCounts())
        c.rows += 1
        results_ = []
        for v in (before, after):
            res = verify(caller, vmodel, v, s["user_message"], r["output"])
            if res.error and "not in the response cache" in res.error:
                not_cached += 1
            results_.append(res)
        ok_b, ok_a = results_[0].passed, results_[1].passed
        wrong = r.get("correct") is False
        c.passed_before += ok_b
        c.passed_after += ok_a
        c.false_pass_before += ok_b and wrong
        c.false_pass_after += ok_a and wrong
        if ok_b and not ok_a:
            c.caught.append(f"{s.get('id', r['sample'])}: {'; '.join(results_[1].reasons)[:160]}")
            if r["setup"] == VERIFIED:
                c.retries_needed += 1
    report = {"results": str(results), "before": str(old_path), "after": str(new_verifiers),
              "before_checks": before.names(), "after_checks": after.names(),
              "requests_made": client.inner.requests_made, "rubric_not_cached": not_cached,
              "setups": {k: vars(v) for k, v in counts.items()}}
    out(f"Re-verified {sum(c.rows for c in counts.values())} saved outputs of decision "
        f"#{decision_id} ({results.name}); no model requests made.")
    out(f"  before: {', '.join(before.names())}")
    out(f"  after:  {', '.join(after.names())}")
    out("")
    rows = [("setup", "pass before", "pass after", "false pass before", "false pass after",
             "retries needed")]
    for name, c in counts.items():
        rows.append((name, f"{c.passed_before}/{c.rows}", f"{c.passed_after}/{c.rows}",
                     str(c.false_pass_before), str(c.false_pass_after),
                     str(c.retries_needed) if name.endswith("verifier") else "-"))
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for k, r in enumerate(rows):
        out("  ".join(f"{x:<{w[i]}}" for i, x in enumerate(r)))
        if k == 0:
            out("  ".join("-" * x for x in w))
    caught = [(n, x) for n, c in counts.items() for x in c.caught]
    if caught:
        out("\nNewly failing (passed before):")
        for n, x in caught:
            out(f"  {n}: {x}")
    if any(c.retries_needed for c in counts.values()):
        out("\nA verified setup would retry the newly failing outputs in production; those "
            "retries are new model calls, so they were not run here.")
    if not_cached:
        out(f"\n{not_cached} rubric check(s) were not in the response cache and were not run.")
    out("False pass = passed every production check but is wrong against the reference answer.")
    return report
