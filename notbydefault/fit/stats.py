"""Steps 5-7: bootstrap intervals, monthly cost projection, verdict over setups."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from notbydefault.fit.evaluate import ALONE, setup_name

MEAN_TOLERANCE = 0.3     # quality: "not meaningfully worse" = mean within this many points
PASS_TOLERANCE = 0.05    # pass rate: "matches" = within 5 percentage points (or better)
MIN_SCORED = 5           # fewer scored outputs than this is not enough evidence (default run: 8)
MAX_ERROR_RATE = 0.10    # a setup whose calls fail more often than this is not viable


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """(mean, low, high) with a percentile bootstrap interval."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, mean, mean
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    return mean, means[int(alpha / 2 * n_boot)], means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]


@dataclass
class SetupStats:
    model: str
    setup: str
    name: str
    calls: int
    errors: int
    scored: int
    mean: float
    lo: float
    hi: float
    pass_rate: float
    pass_lo: float
    pass_hi: float
    avg_attempts: float
    avg_latency_s: float
    avg_cost: float
    monthly_cost: float | None = None
    fits_budget: bool | None = None
    # against reference answers (n_ref == 0 when the samples have none)
    n_ref: int = 0
    correct_rate: float = float("nan")
    correct_lo: float = float("nan")
    correct_hi: float = float("nan")
    item_f1: float = float("nan")
    hard_correct: int = 0
    hard_total: int = 0
    judge_means: dict = field(default_factory=dict)   # judge model -> mean score
    failed_cases: dict = field(default_factory=dict)  # case id -> first problems

    @property
    def key(self) -> tuple[str, str]:
        return self.model, self.setup

    @property
    def correct_count(self) -> int:
        return round(self.correct_rate * self.n_ref) if self.n_ref else 0

    @property
    def judge_avg(self) -> float:
        vals = [v for v in self.judge_means.values() if v == v]
        return sum(vals) / len(vals) if vals else float("nan")

    @property
    def error_rate(self) -> float:
        return self.errors / self.calls if self.calls else 1.0


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def case_id(samples: list[dict] | None, i: int) -> str:
    return (samples[i].get("id") if samples else None) or f"input {i}"


def summarize(rows, models: list[str], setups: tuple[str, ...], max_retries: int = 2,
              seed: int = 0, samples: list[dict] | None = None,
              judges: list[str] | None = None) -> dict[tuple[str, str], SetupStats]:
    out = {}
    for m in models:
        for setup in setups:
            mine = [r for r in rows if r.model == m and r.setup == setup]
            ok = [r for r in mine if not r.error]
            scores = [r.score for r in ok if r.score is not None]
            mean, lo, hi = bootstrap_ci(scores, seed=seed)
            p, plo, phi = bootstrap_ci([1.0 if r.passed else 0.0 for r in ok], seed=seed)
            st = SetupStats(
                model=m, setup=setup, name=setup_name(m, setup, max_retries),
                calls=len(mine), errors=len(mine) - len(ok), scored=len(scores),
                mean=mean, lo=lo, hi=hi, pass_rate=p, pass_lo=plo, pass_hi=phi,
                avg_attempts=_avg([r.attempts for r in ok]),
                avg_latency_s=_avg([r.latency_s for r in ok]), avg_cost=_avg([r.cost for r in ok]))
            # rows with a reference: errors count as incorrect (the feature failed that case)
            refd = [r for r in mine if samples and samples[r.sample].get("reference") is not None]
            if refd:
                vals = [1.0 if r.correct else 0.0 for r in refd]
                st.n_ref = len(refd)
                st.correct_rate, st.correct_lo, st.correct_hi = bootstrap_ci(vals, seed=seed)
                st.item_f1 = _avg([r.item_f1 or 0.0 for r in refd])
                hard = [r for r in refd if samples[r.sample].get("hard")]
                st.hard_total, st.hard_correct = len(hard), sum(1 for r in hard if r.correct)
                st.failed_cases = {case_id(samples, r.sample): (r.problems or [r.error or "incorrect"])
                                   for r in refd if not r.correct}
            for j in judges or []:
                js = [r.scores[j] for r in ok if j in r.scores]
                st.judge_means[j] = _avg(js) if js else float("nan")
            out[(m, setup)] = st
    return out


def judge_agreement(rows, judges: list[str]) -> dict:
    """How two judges compare on the same outputs (each distinct output counted once)."""
    if len(judges) < 2:
        return {"n": 0}
    a, b = judges[:2]
    seen, pairs = set(), []
    for r in rows:
        key = (r.sample, r.output)
        if a in r.scores and b in r.scores and key not in seen:
            seen.add(key)
            pairs.append((r.scores[a], r.scores[b]))
    if not pairs:
        return {"n": 0}
    diffs = [abs(x - y) for x, y in pairs]
    return {"n": len(pairs), "exact": sum(d == 0 for d in diffs) / len(pairs),
            "within_1": sum(d <= 1 for d in diffs) / len(pairs),
            "mean_abs_diff": sum(diffs) / len(pairs),
            "mean_a": _avg([x for x, _ in pairs]), "mean_b": _avg([y for _, y in pairs])}


def parse_number(text: str | None) -> float | None:
    """'500', '1,200', '1.5k', '$25/month', '2M' -> float."""
    if not text:
        return None
    m = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*([kKmM])?", text)
    if not m:
        return None
    value = float(m.group(1).replace(",", ""))
    return value * {"k": 1e3, "m": 1e6}.get((m.group(2) or "").lower(), 1)


def project_monthly(stats: dict, users: float | None, calls_per_user: float,
                    budget: float | None) -> None:
    for s in stats.values():
        if users is None:
            continue
        s.monthly_cost = users * calls_per_user * s.avg_cost
        s.fits_budget = None if budget is None else s.monthly_cost <= budget


@dataclass
class Verdict:
    kind: str                     # keep | switch | insufficient
    reference: tuple[str, str]    # (chosen model, ALONE): what the code does today
    recommended: tuple[str, str]  # cheapest setup that meets the bar
    summary: str
    rejected: dict[str, str] = field(default_factory=dict)   # setup name -> reason
    best_quality: tuple[str, str] | None = None   # best quality within the budget
    bar: str = ""


def fmt_quality(s: SetupStats) -> str:
    return "n/a" if not s.scored else f"{s.mean:.2f} [{s.lo:.2f}-{s.hi:.2f}]"


def fmt_pass(s: SetupStats) -> str:
    return "n/a" if s.calls == s.errors else f"{s.pass_rate:.0%} [{s.pass_lo:.0%}-{s.pass_hi:.0%}]"


def decide(stats: dict[tuple[str, str], SetupStats], chosen: str,
           excluded: dict | None = None) -> Verdict:
    """excluded: setup key -> reason it may not be recommended (e.g. breaks a requirement)."""
    excl = excluded or {}
    ref_key = (chosen, ALONE)
    c = stats[ref_key]
    if c.scored < MIN_SCORED or c.error_rate > MAX_ERROR_RATE:
        return Verdict("insufficient", ref_key, ref_key,
                       f"Not enough evidence: only {c.scored} scored outputs for {c.name} "
                       f"({c.errors} errors).")

    def viable(s):
        return s.scored >= MIN_SCORED and s.error_rate <= MAX_ERROR_RATE and s.key not in excl

    def quality_close(s):
        return s.mean >= c.mean - MEAN_TOLERANCE

    def quality_overlaps(s):
        return s.hi >= c.lo and s.lo <= c.hi

    def pass_matches(s):
        return s.pass_rate >= c.pass_rate - PASS_TOLERANCE

    acceptable = [s for s in stats.values() if (s.key == ref_key and ref_key not in excl) or
                  (viable(s) and pass_matches(s) and (quality_close(s) or quality_overlaps(s)))]
    rejected = {s.name: f"violates a requirement: {excl[s.key]}" for s in stats.values()
                if s.key in excl}
    if not acceptable:
        return Verdict("insufficient", ref_key, ref_key,
                       "Not enough evidence: no setup that meets the requirements matched today's "
                       "pass rate and quality.", rejected)
    best = min(acceptable, key=lambda s: (s.avg_cost, s.key != ref_key))

    for s in stats.values():
        if s.key in (ref_key, best.key) or s.key in excl:
            continue
        if not viable(s):
            why = f"unreliable: {s.errors}/{s.calls} calls failed, {s.scored} scored"
        elif not pass_matches(s):
            why = f"pass rate {fmt_pass(s)} is below {c.name}'s {fmt_pass(c)}"
        elif s not in acceptable:
            why = f"quality {fmt_quality(s)} is meaningfully below {c.name} {fmt_quality(c)}"
        else:
            better = s.mean > best.mean + MEAN_TOLERANCE and s.lo > best.hi
            why = ((f"higher quality {fmt_quality(s)} vs {fmt_quality(best)} for {best.name}, "
                    if better else "matches on pass rate and quality, ")
                   + f"but costs ${s.avg_cost:.6f}/call vs ${best.avg_cost:.6f}")
        rejected[s.name] = why

    if best.key == ref_key:
        return Verdict("keep", ref_key, ref_key,
                       f"Keep {chosen}: no cheaper setup matched its pass rate {fmt_pass(c)} and "
                       f"quality {fmt_quality(c)}.", rejected)
    ratio = c.avg_cost / best.avg_cost if best.avg_cost else float("inf")
    detail = (f"pass rate {fmt_pass(best)} vs {fmt_pass(c)} and quality {fmt_quality(best)} vs "
              f"{fmt_quality(c)} for {c.name}, at {ratio:.1f}x lower cost per call"
              + (" including retries and verification" if best.setup != ALONE else ""))
    if quality_close(best):
        return Verdict("switch", ref_key, best.key, f"Switch to {best.name}: {detail}.", rejected)
    return Verdict("insufficient", ref_key, best.key,
                   f"Not enough evidence: {best.name} looks viable ({detail}), but its mean "
                   f"quality is more than {MEAN_TOLERANCE} points lower and only the intervals "
                   f"overlap. Add samples (or label outputs) to decide.", rejected)


def _cases(s: SetupStats) -> str:
    hard = f", {s.hard_correct}/{s.hard_total} hard" if s.hard_total else ""
    return f"correct on {s.correct_count}/{s.n_ref} cases{hard}"


def decide_with_references(stats: dict[tuple[str, str], SetupStats], chosen: str,
                           budget: float | None, excluded: dict | None = None) -> Verdict:
    """Correctness against references decides; judge scores only break ties.

    Recommends (1) the cheapest setup that meets the bar: at least as correct as today's model
    minus one case, with a verifier pass rate no more than one case lower; and (2) the best
    quality setup whose projected monthly cost fits the budget.
    """
    ref_key = (chosen, ALONE)
    c = stats[ref_key]
    n = c.n_ref
    if n < MIN_SCORED:
        return Verdict("insufficient", ref_key, ref_key,
                       f"Not enough evidence: only {n} cases with reference answers.")
    tol = 1 / n + 1e-9
    excl = excluded or {}

    def viable(s):
        return s.n_ref > 0 and s.error_rate <= MAX_ERROR_RATE and s.key not in excl

    def meets(s):
        return viable(s) and s.correct_rate >= c.correct_rate - tol and s.pass_rate >= c.pass_rate - tol

    bar = (f"correct on at least {max(0, c.correct_count - 1)}/{n} cases (today's {c.name}: "
           f"{c.correct_count}/{n}, one case of slack) with a verifier pass rate at most one case "
           "lower")
    meeting = [s for s in stats.values() if (s.key == ref_key and ref_key not in excl) or meets(s)]
    if not meeting:
        return Verdict("insufficient", ref_key, ref_key,
                       f"No setup that meets the requirements reaches the bar ({_cases(c)} for "
                       f"today's {c.name}).",
                       {s.name: f"violates a requirement: {excl[s.key]}" for s in stats.values()
                        if s.key in excl}, None, "")
    cheapest = min(meeting, key=lambda s: (s.avg_cost, s.key != ref_key))
    affordable = [s for s in stats.values() if viable(s) and s.fits_budget is not False]
    best = max(affordable, key=lambda s: (s.correct_rate, s.item_f1,
                                          s.judge_avg if s.judge_avg == s.judge_avg else 0,
                                          -s.avg_cost)) if affordable else None

    def money(s):
        return f"${s.avg_cost:.6f}/call" + (f", ${s.monthly_cost:,.2f}/month"
                                           if s.monthly_cost is not None else "")

    rejected = {}
    for s in stats.values():
        if s.key in excl:
            rejected[s.name] = f"violates a requirement: {excl[s.key]}"
            continue
        if s.key in {ref_key, cheapest.key} | ({best.key} if best else set()):
            continue
        if not viable(s):
            why = f"unreliable: {s.errors}/{s.calls} calls failed"
        elif s.fits_budget is False:
            why = (f"over budget: ${s.monthly_cost:,.2f}/month vs ${budget:,.2f}"
                   + ("" if meets(s) else f"; {_cases(s)}"))
        elif not meets(s):
            why = f"below the bar: {_cases(s)} vs {c.correct_count}/{n} for {c.name}"
        else:
            why = f"meets the bar ({_cases(s)}) but costs {money(s)} vs {money(cheapest)}"
        rejected[s.name] = why

    lines = [f"Cheapest setup that meets the bar: {cheapest.name}, {_cases(cheapest)}, "
             f"verifier pass rate {fmt_pass(cheapest)}, {money(cheapest)}."]
    if best is None:
        lines.append(f"Best quality within the budget: none; no setup fits ${budget:,.2f}/month "
                     "at this volume.")
    else:
        lines.append(f"Best quality within the budget: {best.name}, {_cases(best)}, "
                     f"item F1 {best.item_f1:.2f}, {money(best)}.")
    kind = "keep" if cheapest.key == ref_key else "switch"
    return Verdict(kind, ref_key, cheapest.key, " ".join(lines), rejected,
                   best.key if best else None, bar)
