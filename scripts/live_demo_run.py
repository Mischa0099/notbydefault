"""One live fit check of demo-app decision #21, for the record. Run by hand, never in CI.

    python scripts/live_demo_run.py                         # new live cache, every request live
    python scripts/live_demo_run.py --cache cache-live-...  # resume in that live cache
    python scripts/live_demo_run.py --no-use-fresh-cache    # reuse .notbydefault/cache like `check`

Runs `check 21` on the real demo-app with the real Groq and Gemini keys from .env. By default the
response cache is a new empty folder, .notbydefault/cache-live-<timestamp>/, so every request goes
to the providers; the old cache is left as it is. When a provider's free-tier daily quota stops
the check, the folder keeps every response so far and the script prints when the quota resets and
the command that resumes there (no request is made twice). Each attempt's trace id is kept in
<cache>/live-run.json.

The cap is --max-cost 0.10. If the check's own estimate is above that, it is raised once to 0.15
(the check refuses before any request, so that first attempt costs nothing); above 0.15 it doesn't run.

When the check finishes, the full report goes to the terminal and demo-live-run.txt, the trace id
and cost to docs/demo.md ("live-run trace: <id> ($<cost>)"), and two lines to the decision record:
"Live-run cost: ..." and "Live run: X requests on Y models across Z calendar days ...". Like any
`check`, it also updates demo-app's graph, results, failure bank and decision record.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PROJECT = REPO / "demo-app"
DECISION = 21
MAX_COST = 0.10
RAISED_MAX_COST = 0.15
DAILY_LIMIT = 4                  # check's exit code when a free-tier daily quota stops it
REPORT = REPO / "demo-live-run.txt"
DEMO_DOC = REPO / "docs" / "demo.md"
STATE = "live-run.json"          # in the cache folder; the cache itself only reads */*.json
KEYS = ("GROQ_API_KEY", "GEMINI_API_KEY", "NEATLOGS_API_KEY")   # neatlogs: for the trace id
LINE = re.compile(r"^live-run trace: .*$", re.M)
ESTIMATE = re.compile(r"estimated cost: \$([0-9.]+)")
LIVE_PREFIX = "cache-live-"


# --- Pacific time without tzdata (not installed on Windows) ---------------------------------

def _nth_sunday(year: int, month: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(6 - first.weekday()) % 7 + 7 * (n - 1))


def _pacific_offset(day: date) -> timedelta:
    """UTC offset at local midnight of `day` (US DST: 2nd Sunday of March to 1st Sunday of
    November, switching at 02:00, so midnight of a switch day still has the old offset)."""
    dst = _nth_sunday(day.year, 3, 2) < day <= _nth_sunday(day.year, 11, 1)
    return timedelta(hours=-7 if dst else -8)


def next_pacific_midnight(now: datetime) -> tuple[datetime, str]:
    """The next midnight Pacific time after `now` (aware): (as UTC, as "YYYY-MM-DD 00:00 PDT")."""
    utc = now.astimezone(timezone.utc)
    nxt = (utc + _pacific_offset(utc.date())).date() + timedelta(days=1)
    offset = _pacific_offset(nxt)
    reset = datetime(nxt.year, nxt.month, nxt.day, tzinfo=timezone.utc) - offset
    return reset, f"{nxt:%Y-%m-%d} 00:00 {'PDT' if offset == timedelta(hours=-7) else 'PST'}"


# --- the live cache and its attempts --------------------------------------------------------

def cache_entries(cache: Path) -> list[dict]:
    out = []
    for f in sorted(cache.glob("*/*.json")):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    return out


def load_attempts(cache: Path) -> list[dict]:
    """Attempts recorded in <cache>/live-run.json; for a live cache made before that file
    existed, the daily-limit journals written since the folder was created."""
    state = cache / STATE
    if state.is_file():
        return json.loads(state.read_text(encoding="utf-8"))["attempts"]
    try:
        made = datetime.strptime(cache.name[len(LIVE_PREFIX):], "%Y%m%dT%H%M%SZ") \
            .replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return []
    attempts = []
    for j in sorted((PROJECT / ".notbydefault" / "journal").glob(f"{DECISION}-*-daily_limit.json")):
        journal = json.loads(j.read_text(encoding="utf-8"))
        if journal and journal[0].get("t", 0) >= made and journal[0].get("trace_id"):
            attempts.append({"trace_id": journal[0]["trace_id"], "exit": DAILY_LIMIT,
                             "at": datetime.fromtimestamp(journal[0]["t"], timezone.utc)
                             .isoformat(timespec="seconds")})
    return attempts


def save_attempts(cache: Path, attempts: list[dict]) -> None:
    (cache / STATE).write_text(json.dumps({"decision": DECISION, "attempts": attempts}, indent=1),
                               encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--use-fresh-cache", action=argparse.BooleanOptionalAction, default=True,
                    help="answer nothing from the old response cache (default: on)")
    ap.add_argument("--cache", metavar="DIR",
                    help="resume in this live cache (a folder name under demo-app/.notbydefault/, "
                         "or a path) instead of starting a new one")
    args = ap.parse_args()

    from notbydefault.env import load_env
    from notbydefault.fit.providers import ProviderSet, load_catalog
    from notbydefault.fit.run import CheckOptions, cache_dir, check_model_decision
    from notbydefault.fit.tracing import make_tracer
    from notbydefault.graph import Graph
    from notbydefault.records import existing_path

    load_env()
    missing = [k for k in KEYS if not os.environ.get(k)]
    if missing or os.environ.get("NOTBYDEFAULT_TELEMETRY") == "0":
        print("Not running: " + (f"set {', '.join(missing)} in {REPO / '.env'}" if missing
                                 else "NOTBYDEFAULT_TELEMETRY=0 turns off the trace"))
        return 1

    if args.cache:
        cache = Path(args.cache)
        if not cache.is_absolute() and not cache.is_dir():
            cache = PROJECT / ".notbydefault" / args.cache
        if not cache.is_dir():
            print(f"Not running: no cache folder {cache}")
            return 1
        cache = cache.resolve()
    elif args.use_fresh_cache:
        cache = PROJECT / ".notbydefault" / \
            f"{LIVE_PREFIX}{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        cache.mkdir(parents=True)
    else:
        cache = cache_dir(PROJECT)
    live = cache.name.startswith(LIVE_PREFIX)
    attempts = load_attempts(cache) if live else []

    lines: list[str] = []

    def out(text: str = "") -> None:
        print(text, flush=True)
        lines.append(str(text))

    out(f"response cache: {cache.relative_to(REPO)} " + (
        f"(resuming: {len(cache_entries(cache))} live responses so far, {len(attempts)} earlier "
        "attempt(s); only the rest go to the providers)" if args.cache and live else
        "(new and empty: every request goes to the providers)" if live else
        "(reused: cached requests cost nothing)"))
    client = ProviderSet(load_catalog(PROJECT), cache_dir=cache)

    with Graph.for_project(PROJECT) as g:
        session = (g.get_node(DECISION) or {}).get("session_id")
    tracer = make_tracer(session)
    max_cost = MAX_COST
    from notbydefault.awake import keep_awake
    with keep_awake():
        code = check_model_decision(PROJECT, DECISION, CheckOptions(max_cost=max_cost, yes=True),
                                    client=client, tracer=tracer, out=out)
        est = next((float(m.group(1)) for l in lines if (m := ESTIMATE.search(l))), None)
        if code == 2 and est is not None and est > MAX_COST:
            max_cost = RAISED_MAX_COST
            out(f"\nEstimate ${est:.2f} is above ${MAX_COST:.2f}: raising the cap to "
                f"${max_cost:.2f} and running (nothing was spent on the refused attempt).\n")
            code = check_model_decision(PROJECT, DECISION,
                                        CheckOptions(max_cost=max_cost, yes=True),
                                        client=client, tracer=tracer, out=out)
    trace_id = tracer.trace_id
    if live and trace_id:
        attempts.append({"trace_id": trace_id, "exit": code,
                         "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        save_attempts(cache, attempts)

    with Graph.for_project(PROJECT) as g:
        decision = g.get_node(DECISION)
    results_rel = decision["data"].get("results")
    results = json.loads((PROJECT / results_rel).read_text(encoding="utf-8")) if results_rel else {}
    done = code == 0 and trace_id and results.get("trace_id") == trace_id
    req = (results.get("requests") or {}) if done else {}

    out("")
    out(f"exit code: {code}")
    out(f"cap:       --max-cost ${max_cost:.2f}")
    out(f"trace id:  {trace_id or ('none: the check did not run' if code else 'none (export failed; see the warnings above)')}")

    if code == DAILY_LIMIT and live:
        reset, pacific = next_pacific_midnight(datetime.now(timezone.utc))
        out("")
        out("A provider's free-tier daily quota stopped the check. Nothing is lost: "
            f"{len(cache_entries(cache))} live responses are kept in {cache.relative_to(REPO)}.")
        out(f"Quota resets: {pacific} (midnight Pacific) = "
            f"{reset.astimezone():%Y-%m-%d %H:%M} local ({reset:%H:%M} UTC).")
        out("Resume after that (no request is made twice):")
        out(f"  .venv\\Scripts\\python.exe scripts\\live_demo_run.py --cache {cache.name}")
        REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"report:    {REPORT.relative_to(REPO)}")
        return code

    if not done:
        REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"report:    {REPORT.relative_to(REPO)}")
        print(f"{DEMO_DOC.relative_to(REPO)} and the decision record not updated: the check did "
              f"not finish (exit {code})" + (f"; its partial trace is {trace_id}" if trace_id else "")
              + ".")
        return code or 1

    # a live cache holds exactly the requests this live run made, across every attempt
    entries = cache_entries(cache) if live else []
    pricing = client.pricing()
    spent = (sum(pricing.prices[e["model"]].cost(e["input_tokens"], e["output_tokens"])
                 for e in entries if e.get("model") in pricing.prices)
             if live else float(results.get("spent") or 0))
    out(f"spent:     ${spent:.4f} "
        + (f"(list price of the {len(entries)} live requests in {cache.name}, every attempt)"
           if live else f"({req.get('made', 0)} provider requests, "
                        f"{req.get('cache_hits', 0)} answered from the response cache)"))
    out(f"results:   demo-app/{results_rel}")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report:    {REPORT.relative_to(REPO)}")

    entry = f"live-run trace: {trace_id} (${spent:.4f})"
    doc = DEMO_DOC.read_text(encoding="utf-8") if DEMO_DOC.exists() else "# Demo\n"
    doc = LINE.sub(entry, doc, count=1) if LINE.search(doc) else doc.rstrip("\n") + f"\n\n{entry}\n"
    DEMO_DOC.write_text(doc, encoding="utf-8")
    print(f"{DEMO_DOC.relative_to(REPO)}: {entry}")

    # on free-tier keys the cost is the list price of the requests made, not what was billed
    free_tier = (results.get("pricing") or {}).get("free_tier")
    notes = [f"Live-run cost: ${spent:.2f} ("
             + ("list price of real requests" if free_tier else "real spend")
             + (", fresh cache" if live else ", response cache reused")
             + ("; free tier, not billed" if free_tier else "") + ")"]
    if live:
        models = {e.get("model") for e in entries}
        days = {datetime.fromisoformat(e["at"]).astimezone().date() for e in entries if e.get("at")}
        many = len(days) != 1
        where = f"across {len(days)} calendar day{'s' * many}" if many else "in 1 calendar day"
        if len(attempts) > 1:
            partial = ", ".join(a["trace_id"] for a in attempts[:-1])
            notes.append(f"Live run: {len(entries)} requests on {len(models)} models {where} "
                         f"(quota reset between). Partial trace: {partial}, final trace: {trace_id}.")
        else:
            notes.append(f"Live run: {len(entries)} requests on {len(models)} models {where}. "
                         f"Trace: {trace_id}.")
    record = existing_path(PROJECT, decision)
    if record:
        text = record.read_text(encoding="utf-8")
        record.write_text(text.rstrip("\n") + "\n\n" + "\n".join(notes) + "\n", encoding="utf-8")
        for n in notes:
            print(f"{record.relative_to(REPO)}: {n}")
    else:
        print(f"No decision record for #{DECISION}; not written: " + " / ".join(notes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
