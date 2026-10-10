"""Steps 5-7: bootstrap intervals, monthly cost projection, verdict over setups."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from notbydefault.fit.evaluate import ALONE, setup_name

MEAN_TOLERANCE = 0.3     # quality: "not meaningfully worse" = mean within this many points
PASS_TOLERANCE = 0.05    # pass rate: "matches" = within 5 percentage points (or better)
MIN_SCORED = 10          # fewer scored outputs than this is not enough evidence
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

    @property
    def key(self) -> tuple[str, str]:
        return self.model, self.setup

    @property
    def error_rate(self) -> float:
        return self.errors / self.calls if self.calls else 1.0


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize(rows, models: list[str], setups: tuple[str, ...], max_retries: int = 2,
              seed: int = 0) -> dict[tuple[str, str], SetupStats]:
    out = {}
    for m in models:
        for setup in setups:
            mine = [r for r in rows if r.model == m and r.setup == setup]
            ok = [r for r in mine if not r.error]
            scores = [r.score for r in ok if r.score is not None]
            mean, lo, hi = bootstrap_ci(scores, seed=seed)
            p, plo, phi = bootstrap_ci([1.0 if r.passed else 0.0 for r in ok], seed=seed)
            out[(m, setup)] = SetupStats(
                model=m, setup=setup, name=setup_name(m, setup, max_retries),
                calls=len(mine), errors=len(mine) - len(ok), scored=len(scores),
                mean=mean, lo=lo, hi=hi, pass_rate=p, pass_lo=plo, pass_hi=phi,
                avg_attempts=_avg([r.attempts for r in ok]),
                avg_latency_s=_avg([r.latency_s for r in ok]), avg_cost=_avg([r.cost for r in ok]))
    return out


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
    recommended: tuple[str, str]
    summary: str
    rejected: dict[str, str] = field(default_factory=dict)   # setup name -> reason


def fmt_quality(s: SetupStats) -> str:
    return "n/a" if not s.scored else f"{s.mean:.2f} [{s.lo:.2f}-{s.hi:.2f}]"


def fmt_pass(s: SetupStats) -> str:
    return "n/a" if s.calls == s.errors else f"{s.pass_rate:.0%} [{s.pass_lo:.0%}-{s.pass_hi:.0%}]"


def decide(stats: dict[tuple[str, str], SetupStats], chosen: str) -> Verdict:
    ref_key = (chosen, ALONE)
    c = stats[ref_key]
    if c.scored < MIN_SCORED or c.error_rate > MAX_ERROR_RATE:
        return Verdict("insufficient", ref_key, ref_key,
                       f"Not enough evidence: only {c.scored} scored outputs for {c.name} "
                       f"({c.errors} errors).")

    def viable(s):
        return s.scored >= MIN_SCORED and s.error_rate <= MAX_ERROR_RATE

    def quality_close(s):
        return s.mean >= c.mean - MEAN_TOLERANCE

    def quality_overlaps(s):
        return s.hi >= c.lo and s.lo <= c.hi

    def pass_matches(s):
        return s.pass_rate >= c.pass_rate - PASS_TOLERANCE

    acceptable = [s for s in stats.values() if s.key == ref_key or
                  (viable(s) and pass_matches(s) and (quality_close(s) or quality_overlaps(s)))]
    best = min(acceptable, key=lambda s: (s.avg_cost, s.key != ref_key))

    rejected = {}
    for s in stats.values():
        if s.key in (ref_key, best.key):
            continue
        if not viable(s):
            why = f"unreliable: {s.errors}/{s.calls} calls failed, {s.scored} scored"
        elif not pass_matches(s):
            why = f"pass rate {fmt_pass(s)} is below {c.name}'s {fmt_pass(c)}"
        elif s not in acceptable:
            why = f"quality {fmt_quality(s)} is meaningfully below {c.name} {fmt_quality(c)}"
        else:
            why = (f"matches on pass rate and quality but costs ${s.avg_cost:.6f}/call vs "
                   f"${best.avg_cost:.6f} for {best.name}")
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
