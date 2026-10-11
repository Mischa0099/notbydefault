"""Steps 7-8: plain-English report, results table, decision record."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from notbydefault.fit.stats import SetupStats, Verdict, fmt_pass, fmt_quality


@dataclass
class ReportContext:
    decision_id: int
    literal: str
    chosen: str
    task: str
    n_samples: int
    sample_counts: dict[str, int]        # source -> count, e.g. {"failure-bank": 3, "synthetic": 20}
    judge: str
    verifier_model: str
    verifier_checks: list[str]
    verifier_path: str
    max_retries: int
    users: float | None
    calls_per_user: float
    budget: float | None
    spent: float
    pricing_source: str
    pricing_fetched_at: str
    results_path: str
    failures_added: int = 0
    failures_path: str = ""
    trust: dict | None = None             # from labels.agreement()
    stopped: bool = False
    free_tier: bool = False
    single_provider: bool = False
    trace_id: str | None = None
    notes: list[str] = field(default_factory=list)
    judges: list[str] = field(default_factory=list)
    agreement: dict | None = None
    case_rows: list[dict] = field(default_factory=list)
    bar: str = ""
    best_quality: tuple | None = None
    caveats: list[str] = field(default_factory=list)

    @property
    def has_references(self) -> bool:
        return any(r.get("correct") is not None for r in self.case_rows)


EVAL_ONLY_CHECKS = {"no_missing_items", "no_extra_items"}


FREE_TIER_LABEL = "Run on free tier; costs are list-price estimates."


def _money(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None else f"${x:,.{digits}f}"


def sample_label(ctx: ReportContext) -> str:
    """e.g. "12 hand-written" or "8 synthetic, 3 of them re-run first from the failure bank"."""
    banked = ctx.sample_counts.get("re-run first from the failure bank", 0)
    label = " + ".join(f"{n} {src}" for src, n in ctx.sample_counts.items()
                       if src != "re-run first from the failure bank")
    return label + (f", {banked} of them re-run first from the failure bank" if banked else "")


def _short(model: str) -> str:
    return model.split(":", 1)[-1].split("/")[-1]


def table_rows(stats: dict, ctx: ReportContext) -> tuple[list[str], list[list[str]]]:
    refs = any(s.n_ref for s in stats.values())
    if refs:
        headers = (["setup", "correct (95% CI)", "hard", "item F1", "verifier pass"]
                   + [f"judge {_short(j)}" for j in ctx.judges]
                   + ["latency", "cost/call incl. retries", "monthly", "fits budget?"])
    else:
        headers = ["setup", "pass rate (95% CI)", "quality 1-5 (95% CI)", "avg latency",
                   "cost/call incl. retries", "monthly", "fits budget?"]
    rows = []
    for s in stats.values():
        name = s.name + (" (current)" if (s.model, s.setup) == (ctx.chosen, "alone") else "")
        fits = {True: "yes", False: "NO", None: "n/a"}[s.fits_budget]
        cost = f"${s.avg_cost:.5f}" + (f" ({s.avg_attempts:.2f} tries)" if s.setup != "alone" else "")
        if refs:
            correct = (f"{s.correct_count}/{s.n_ref} [{s.correct_lo:.0%}-{s.correct_hi:.0%}]"
                       + (f" ({s.errors} err)" if s.errors else ""))
            judges = [f"{s.judge_means[j]:.2f}" if s.judge_means.get(j) == s.judge_means.get(j)
                      and j in s.judge_means else "n/a" for j in ctx.judges]
            rows.append([name, correct, f"{s.hard_correct}/{s.hard_total}", f"{s.item_f1:.2f}",
                         f"{s.pass_rate:.0%}", *judges, f"{s.avg_latency_s:.1f}s", cost,
                         _money(s.monthly_cost), fits])
        else:
            quality = fmt_quality(s) + (f" ({s.errors} err)" if s.errors else "")
            rows.append([name, fmt_pass(s), quality, f"{s.avg_latency_s:.1f}s", cost,
                         _money(s.monthly_cost), fits])
    return headers, rows


def text_table(table: tuple[list[str], list[list[str]]]) -> str:
    headers, rows = table
    widths = [max(len(r[i]) for r in [headers] + rows) for i in range(len(headers))]
    line = lambda r: "  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip()
    return "\n".join([line(headers), line(["-" * w for w in widths])] + [line(r) for r in rows])


def markdown_table(table: tuple[list[str], list[list[str]]]) -> str:
    headers, rows = table
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows])


def judges_line(ctx: ReportContext) -> str:
    a = ctx.agreement or {}
    if len(ctx.judges) < 2 or not a.get("n"):
        return f"Judged blind by {', '.join(ctx.judges) or ctx.judge}."
    j1, j2 = (_short(j) for j in ctx.judges[:2])
    return (f"Two blind judges from different families, {ctx.judges[0]} and {ctx.judges[1]}: "
            f"on the same {a['n']} outputs they gave identical scores {a['exact']:.0%} of the time "
            f"and were within 1 point {a['within_1']:.0%} of the time (mean gap "
            f"{a['mean_abs_diff']:.2f}; mean {j1} {a['mean_a']:.2f} vs {j2} {a['mean_b']:.2f}). "
            "Correctness against the references decides; judge scores only break ties.")


def case_results(ctx: ReportContext) -> list[str]:
    """Which cases each setup got wrong, and what verifier retries did about it."""
    by_setup: dict[tuple[str, str], dict[str, dict]] = {}
    for r in ctx.case_rows:
        by_setup.setdefault((r["model"], r["setup"]), {})[r["case"]] = r
    lines = []
    for (model, setup), cases in by_setup.items():
        if setup != "alone":
            continue
        wrong = [c for c, r in cases.items() if r["correct"] is False
                 or (r.get("has_reference") and r["correct"] is None)]
        verified = by_setup.get((model, "verified"), {})
        rescued = [c for c in wrong if verified.get(c, {}).get("correct")]
        broken = [c for c, r in cases.items() if r["correct"] and verified.get(c, {}).get("correct") is False]
        retried = [c for c, r in verified.items() if r["attempts"] > 1]
        slipped = [c for c, r in cases.items() if r["correct"] is False and not r.get("error")
                   and not (set(r["failed_checks"]) - EVAL_ONLY_CHECKS)]
        lines.append(f"{model}: wrong on {len(wrong)}/{len(cases)}"
                     + (f" ({', '.join(wrong)})" if wrong else ""))
        for c in wrong:
            r = cases[c]
            first = (r["problems"] or [r.get("error") or "incorrect"])[0]
            lines.append(f"    {c}{' [hard]' if r['hard'] else ''}: {first}"
                         + (f" (+{len(r['problems']) - 1} more)" if len(r["problems"]) > 1 else ""))
        lines.append(f"    + verifier: retried {len(retried)} case(s)"
                     + (f" ({', '.join(retried)})" if retried else "")
                     + f"; rescued {len(rescued)}" + (f" ({', '.join(rescued)})" if rescued else "")
                     + (f"; made {len(broken)} worse ({', '.join(broken)})" if broken else "")
                     + f"; {len(slipped)} wrong answer(s) passed every production check"
                     + (f" ({', '.join(slipped)})" if slipped else ""))
    return lines


def trust_line(ctx: ReportContext) -> str:
    t = ctx.trust
    if not t or not t.get("n"):
        return (f"Verifier trust: not measured yet - run `python -m notbydefault label "
                f"{ctx.decision_id}` to hand-label 10 outputs.")
    return (f"Verifier trust: verifiers agree with your labels on {t['agree']}/{t['n']} outputs "
            f"({t['rate']:.0%}); {t['false_pass']} false pass(es), {t['false_fail']} false fail(s).")


def assumptions(ctx: ReportContext) -> list[str]:
    users = f"{ctx.users:,.0f} expected users (from the graph)" if ctx.users else \
        "expected users unknown (run `init`)"
    budget = f"monthly budget {_money(ctx.budget)} (from the graph)" if ctx.budget is not None \
        else "no monthly budget recorded"
    synthetic = ctx.sample_counts.get("synthetic", 0)
    hand = ctx.sample_counts.get("hand-written", 0)
    hard = sum(1 for r in {(x["case"], x["hard"]) for x in ctx.case_rows} if r[1])
    return [
        f"Monthly cost = {users} x {ctx.calls_per_user:g} calls per user per month "
        f"(ASSUMPTION) x measured average cost per call; {budget}.",
        (FREE_TIER_LABEL + " " if ctx.free_tier else "")
        + f"Prices: {ctx.pricing_source}.",
        f"Test inputs: {sample_label(ctx)}"
        + (f" ({synthetic} SYNTHETIC: generated by an LLM, not real user data)" if synthetic else "")
        + (f" ({hand} hand-written with reference answers, {hard} marked hard)" if hand else "")
        + "; failure-bank cases always run first.",
        f"Verifier pass = all checks in {ctx.verifier_path} pass ({', '.join(ctx.verifier_checks)}). "
        f"Verified setups retry up to {ctx.max_retries}x, feeding back only the failures production "
        "could see (reference checks are never used as hints); their cost/latency include those "
        "retries and any verifier calls.",
        judges_line(ctx) if ctx.has_references else
        f"Quality judged blind by {ctx.judge}; response order randomized per input.",
        trust_line(ctx),
    ]


def summary_text(stats: dict, verdict: Verdict, ctx: ReportContext) -> str:
    c = stats[verdict.reference]
    parts = [f"Decision #{ctx.decision_id} uses {ctx.literal} ({ctx.chosen})"
             + (f" to {ctx.task.rstrip('.')[0].lower() + ctx.task.rstrip('.')[1:]}." if ctx.task else ".")]
    n_models = len({s.model for s in stats.values()})
    if ctx.has_references:
        parts.append(f"We ran {ctx.n_samples} cases ({sample_label(ctx)}) through {n_models} models, "
                     "each alone and with a verifier, scored every answer against hand-written "
                     f"reference answers, and had {len(ctx.judges)} judge(s) from different "
                     "families score them blind.")
    else:
        parts.append(f"We ran {ctx.n_samples} inputs ({sample_label(ctx)}) through {n_models} models, "
                     f"each alone and with a verifier, checked every output with the verifiers and "
                     f"had {ctx.judge} score them blind.")
    if ctx.stopped:
        parts.append("The run STOPPED EARLY because --max-cost was reached, so results are "
                     "incomplete.")
    parts.append(verdict.summary)
    rec = stats.get(verdict.recommended)
    if c.monthly_cost is not None:
        line = f"At the assumed volume, today's setup costs about {_money(c.monthly_cost)}/month"
        if rec and rec.key != c.key and rec.monthly_cost is not None:
            line += f" vs {_money(rec.monthly_cost)} for {rec.name}"
        if ctx.budget is not None:
            line += f" (budget {_money(ctx.budget)})"
        parts.append(line + ".")
    return " ".join(parts)


def render(stats: dict, verdict: Verdict, ctx: ReportContext) -> str:
    rows = table_rows(stats, ctx)
    best = stats.get(ctx.best_quality) if ctx.best_quality else None
    lines = [f"NotByDefault check: decision #{ctx.decision_id} ({ctx.literal})"
             + (f"  [{FREE_TIER_LABEL}]" if ctx.free_tier else "")
             + (f"  [judge: {ctx.judge}, single-provider run]" if ctx.single_provider else ""),
             "",
             summary_text(stats, verdict, ctx), "", text_table(rows), ""]
    if ctx.has_references:
        lines += ["Recommendations:",
                  f"  1. Cheapest setup that meets the bar: {stats[verdict.recommended].name}",
                  f"  2. Best quality within the budget:    "
                  + (best.name if best else "none fits the budget at this volume"),
                  f"  Bar: {ctx.bar}.", "", "Case results (vs reference answers):"]
        lines += [f"  {l}" for l in case_results(ctx)] + [""]
    if ctx.caveats:
        lines += ["Caveats:"] + [f"  - {c}" for c in ctx.caveats] + [""]
    lines += ["Assumptions:"]
    lines += [f"  - {a}" for a in assumptions(ctx)]
    if verdict.rejected:
        lines += ["", "Not recommended:"] + [f"  - {m}: {why}" for m, why in verdict.rejected.items()]
    if ctx.failures_added:
        lines += ["", f"{ctx.failures_added} failing outputs added to the failure bank "
                      f"({ctx.failures_path}); they run first next time."]
    lines += [f"  - {n}" for n in ctx.notes]
    lines += ["", f"Spent ${ctx.spent:.4f} on this check. Raw results: {ctx.results_path}"]
    if ctx.trace_id:
        lines.append(f"neatlogs trace id: {ctx.trace_id} (workflow notbydefault)")
    return "\n".join(lines)


def write_decision_record(project: Path, stats: dict, verdict: Verdict,
                          ctx: ReportContext, date: str, existing: Path | None = None) -> Path:
    """Write the detailed record; `existing` (the decision's record so far) is rewritten in
    place, keeping its number, so one decision has one record across re-tests."""
    folder = project / "decisions"
    folder.mkdir(parents=True, exist_ok=True)
    numbers = [int(m.group(1)) for p in folder.glob("*.md")
               if (m := re.match(r"(\d{4})-", p.name))]
    slug = re.sub(r"[^a-z0-9]+", "-", f"ai-model-{ctx.literal}".lower()).strip("-")
    path = existing or folder / f"{max(numbers, default=0) + 1:04d}-{slug}.md"
    rec = stats[verdict.recommended]
    status = {"keep": "Accepted (keep current model)",
              "switch": f"Change recommended: {rec.name}",
              "insufficient": "Undecided (not enough evidence)"}[verdict.kind]
    body = [
        f"# {path.stem[:4]}. AI model for: {ctx.task or ctx.literal}", "",
        f"- Date: {date}", f"- Status: {status}",
        *([f"- {FREE_TIER_LABEL}"] if ctx.free_tier else []),
        f"- Decision graph node: #{ctx.decision_id} (`{ctx.literal}` -> `{ctx.chosen}`)", "",
        "## Context", "", summary_text(stats, verdict, ctx), "",
        "## Setups tested", "", markdown_table(table_rows(stats, ctx)), "",
        "## Verdict", "", verdict.summary, "",
    ]
    if ctx.has_references:
        best = stats.get(ctx.best_quality) if ctx.best_quality else None
        body += [f"1. Cheapest setup that meets the bar: `{rec.name}`",
                 "2. Best quality within the budget: "
                 + (f"`{best.name}`" if best else "none fits the budget at this volume"), "",
                 f"Bar: {ctx.bar}.", "", "## Case results (vs reference answers)", "", "```"]
        body += case_results(ctx) + ["```", ""]
    if verdict.rejected:
        body += ["## Rejected alternatives", ""] + [f"- `{m}`: {why}"
                                                   for m, why in verdict.rejected.items()] + [""]
    if ctx.caveats:
        body += ["## Caveats", ""] + [f"- {c}" for c in ctx.caveats] + [""]
    body += ["## Assumptions and evidence", ""] + [f"- {a}" for a in assumptions(ctx)]
    body += [f"- Raw results: `{ctx.results_path}`", ""]
    path.write_text("\n".join(body), encoding="utf-8")
    return path
