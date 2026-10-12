# Demo claims audit

Audited 2026-10-12 against the files in this repo (commit `e908b29` plus the untracked
`demo-live-run.txt`). Nothing in application logic, evaluation outputs or results files was changed.

## Status labels

- **Verified**: recomputed in this audit from a raw results file, cache or eval output, and it matches.
- **Projected**: the arithmetic checks out, but the figure is a forecast built on stated assumptions,
  not a measurement (every $/month figure falls in this category).
- **Reported, unverified**: written somewhere in the repo, but this audit could not check it against raw data.
- **Seeded / scripted**: written by `scripts/demo_dataset.py` for the demo, and labeled as such in the data.
- **Superseded**: true for an earlier run, but no longer what the code or data produce.

## The runs

There are two real fit checks of decision #21. They are **not interchangeable**: the verdict changed between them.

| | Run A (earlier) | Run B (latest) |
|---|---|---|
| Trace | `4bcff389ab979638e4ffa7ee88619318` | `0f65667e9f5c521dfc124c89f188f9ca` (final); `56a5915f9768ef65b7f21546bdfceb6a` (partial, first attempt) |
| Results file | `demo-app/.notbydefault/results/21-20261011T070521Z.json` | `demo-app/.notbydefault/results/21-20261012T070725Z.json` |
| Run at | 2026-10-11 07:05 UTC | 2026-10-12 07:07 UTC (attempt 1 at 2026-10-11 20:30 UTC hit a Gemini daily limit; resumed after the reset) |
| Requests | 31 made, 42 from an existing cache (**not** a fresh live run) | 71 live requests into a fresh cache `cache-live-20261011T202940Z` (56 + 15), 5 models |
| Today's setup (gpt-oss-120b) | 10/12, 6/8 hard, **$407.35/mo** | 10/12, 6/8 hard, **$418.04/mo** |
| gpt-oss-20b + verifier | 10/12, 6/8 hard, $385.37/mo, **over budget** | 10/12, 6/8 hard, $336.48/mo, **in budget** |
| gemini-3.1-flash-lite | 11/12, 8/8 hard, $345.07/mo, in budget | 11/12, 8/8 hard, $345.07/mo, in budget |
| Verdict | switch to **gemini-3.1-flash-lite** (saves $62.28/mo) | switch to **gpt-oss-20b + verifier** (saves $81.56/mo); best quality in budget: flash-lite |
| What `resolve 21` does | applied flash-lite on a branch (demo dataset build, see C10) | recommends the switch and **stops**: a verifier loop can't be added to the code automatically yet |
| Documented in | commits `44b4fc1`, `3956607`; `eval/results/v3.json`; README before `60a13ca` | README, ARCHITECTURE.md, decision record `0002`, `business/costs.*`, `docs/demo.md`, `demo-live-run.txt` |

Both runs used the same 12 cases (8 hard), verifier `21.py` (sha `62bd632d5177ad7f`), volume assumptions
(20,000 users x 60 calls) and $375 budget. I recomputed every row in the table above from each results
file's raw `rows` (cost per call x 1,200,000 calls), and they match the reports to the cent.

**Why the verdict flipped:** list prices didn't change. Run to run, gpt-oss-20b + verifier's measured
cost per call fell from $0.000321 to $0.000280 (fewer retries), which moved it from $10.37 over budget
to $38.52 under. gpt-oss-120b's cost rose from $0.000340 to $0.000348. So **the recommended setup is
sensitive to sampling variance in retries and output length.** flash-lite's numbers were identical in
both runs: 11/12, $345.07, in budget.

## Claims

| # | Claim | Source | Run / trace | Status | Recommended wording |
|---|---|---|---|---|---|
| C1 | Today's gpt-oss-120b: 10/12 correct (6/8 hard) | results B `rows`; decision record 0002; README | B `0f65667e` | **Verified** (also 10/12 in run A) | "Today's model got 10 of 12 of the project's own test cases right." |
| C2 | Today's setup costs $418/month | results B; costs.csv | B | **Projected.** The arithmetic is verified: $0.0003484/call x 20,000 users x 60 calls. 20,000 users and the $375 budget are "demo scenario assumption"s in the graph, and 60 calls/user is marked ASSUMPTION. List prices were fetched 2026-10-10 and the run used free tiers. | "At an assumed 1.2M calls a month, about $418/month: over the $375 budget." |
| C3 | gpt-oss-20b + verifier: 10/12, $336/month, in budget | results B | B | **Verified** score; **projected** cost. Hard cases 6/8, same as today's. Item F1 is 0.88 vs 0.94 and judge scores are 4.25–4.33 vs 4.83, so it matches on *case correctness* but scores lower on the secondary quality measures. It was **over budget in run A** ($385.37). | "The cheapest setup that clears the bar also got 10 of 12, at a projected $336/month." |
| C4 | Net saving $81.56/month ($418.04 → $336.48) | results B; decision record; costs.csv; README | B | **Projected.** The arithmetic is verified: 418.04 − 336.48 = 81.56. It isn't realized, because `resolve` doesn't apply this setup (C11), and it isn't stable across runs (run A: this setup was over budget). | "A projected ~$82/month saving at the assumed volume, if the switch is made." Don't say "saved". |
| C5 | gemini-3.1-flash-lite: 11/12, 8/8 hard, $345/month, best quality within budget | results B (and A) | B (identical in A) | **Verified** score and hard cases; **projected** cost. Trade-off: 13.8 s latency vs 1.5 s today. Its only miss, `c10-priority-cues`, is a case that was reworded during development after all three models missed it (caveat in the report). | "The best in-budget option scored 11 of 12, all 8 hard cases, at a projected $345/month, but it's about 9x slower." |
| C6 | 71 real provider requests across a quota reset | `live_run` in results B; cache directory | B (+ partial `56a5915f`) | **Verified.** The cache holds 71 response files (56 written 02:00–02:05 IST, 15 at 12:37–12:41) plus `attempts.json`, across 5 models: gpt-oss-20b 17, gpt-oss-120b 14, flash-lite 14, qwen3.8-27b 13, gemini-3.8-flash 13. | "71 real requests to 5 models, resumed across a free-tier quota reset with nothing re-sent." |
| C7 | Live run cost $0.0461 | `live_run.spent` in results B; README; docs/demo.md | B (both attempts) | **Verified**: list price of the 71 cached requests. Free tier, so $0 was billed. Three different "check cost" figures exist for the same run: **$0.0461** (both attempts), **$0.0221** (second attempt only, printed in `demo-live-run.txt` with "First-month ROI $81.53"), and **$0.0561** (`costs.md` "fit-check cost": every call in the comparison at list price, cache hits included). | "The whole check cost under 5 cents at list price (run on free tiers)." Use $0.0461 only. README repeats it redundantly: "$0.0461 ($0.0461 across two attempts…)". |
| C8 | Trace `0f65667e9f5c521dfc124c89f188f9ca` | results B `trace_id`; demo-live-run.txt | B | **Verified** locally (the ID is recorded). **Reported, unverified** in neatlogs: I didn't open the trace in neatlogs. | Show the trace on screen only if it opens in neatlogs. |
| C9 | $407/month, $375 budget, 10/12 vs 11/12 | git history (`44b4fc1`, README before `60a13ca`); results A; `eval/results/v3.json` | A `4bcff389` | **Verified for run A, superseded.** Run A also wasn't a fresh live run (42 cache hits). | Don't use, or mix with run B. $407 is run A's figure and $418 is run B's. |
| C10 | `resolve` generated tests went from 0/12 to 12/12 | earlier session transcript (`.entire/metadata/304d83c9…`): demo-dataset graph node 21 "generated: 0/12 passed -> 12/12 passed \| to gemini:gemini-3.1-flash-lite"; README before `60a13ca` | Run A's verdict (flash-lite), produced by `scripts/demo_dataset.py` on the throwaway copy | **Superseded, can't be tied to run B.** The result came from resolving to flash-lite under run A's verdict. Under run B's verdict, resolve stops without editing code, so no tests are generated or run. No branch or record of it exists in the current `.demo-dataset` (only `main`, one baseline commit). | **Don't use.** If you show resolve, show what it does now: it recommends and stops. |
| C11 | "`resolve 21` recommends that switch and stops" | README; demo_dataset.py docstring | B | **Verified** in code docs and the current demo dataset (#21 is still on gpt-oss-120b) | Keep as is. |
| C12 | Agent picked gpt-oss-120b because it was "cost-effective, strong at extraction, widely used" | `REASONS` in `scripts/demo_dataset.py`, labeled "scripted demo reason …, not a real session" | none | **Scripted.** The README presents it as an agent's real reason without saying so. | "Say an agent picked it because it's 'cost-effective, strong at extraction, widely used'…", or put a "scripted example reason" caption on screen. |
| C13 | EU requirement: #21 and the older Groq decision go **stale**, MongoDB goes **needs_review** | `.demo-dataset/…/graph.db`: node 21 stale_reason cites console.groq.com/docs/your-data (read 2026-10-11) | demo dataset build (scripted scenario) | **Verified** in the demo dataset. The scenario is scripted, and the `--offline` fallback uses a keyword rule for the region. | "When we add 'data must stay in the EU', the Groq decisions go stale and MongoDB is flagged for review." |
| C14 | Seeded pivot "drop the verifiers" | demo dataset node 32 `source: seeded` | demo dataset | **Seeded**, labeled in the data | If shown, say it's a seeded example. |
| C15 | Detection recall 0.81 → 1.00; pivot recall 0.67 → 1.00, 0.5 held-out | `eval/results/v1.json`, `v2*.json`, `v3.json` | eval v1–v3 | **Verified** (v1 detection 0.812, v3 1.0; v1 pivot 0.667, v3 1.0; held-out pivot 0.5 in v2 and v3). These are self-written cases, not independently labeled (README says so). | Keep the held-out number next to the headline number. |
| C16 | 475 tests, no network | `pytest --collect-only` | n/a | **Verified count** (475 collected). Not run in this audit. Commit `4d18984`'s message says 472 (outdated). | "475 tests" |
| C17 | Hook under 300 ms (p50 < 200 ms, worst 405 ms) | README | n/a | **Reported, unverified** | Leave out of the video or measure first. |
| C18 | Demo repo decision-debt score of 14 | README | demo dataset | **Reported, unverified** | Show it from the UI on screen if used. |
| C19 | "$82/mo saved … at the same accuracy (10/12)" covers $49 Starter with $33 left | README business case; `business/plan.md` | B | **Projected.** Arithmetic: 82 − 49 = 33. "Same accuracy" holds only for case correctness (see C3). The README's "reduced projected inference costs" is fine; "saved" in plan.md isn't. | "Projected to save more than the $49 subscription costs." |
| C20 | Self-funded through Oct 2027, ~$5,354 cash, $2,789 MRR | README; `business/plan.md` (cfo.ai) | n/a | **Projected**, built on business assumptions. Not audited here. | Present it as a plan under disclosed assumptions, or leave it out. |

### Minor inconsistencies found

- The `demo-live-run.txt` header says "inputs: 12 (8 from failure bank …)", but the report body and results B say 7 cases were re-run from the failure bank (run A: 6).
- `demo-live-run.txt` (raw output, $0.0221, ROI $81.53) and decision record 0002 ($0.0461, ROI $81.51) give different ROI figures. Both are internally correct for the cost each one uses.
- The results B file was last modified at 13:41 IST, after the run. That's when `scripts/live_demo_run.py` `record_live_run` added the `live_run` block. The rows and verdict fields agree with the run's printed output.

## Recommended video example (one run: B, trace `0f65667e`)

Use **run B only**. It's the latest run and also the strongest: it's the only fresh, fully live run,
its raw rows reproduce every number, and its cache lets anyone replay it with no new requests.

**Reproducibility limit:** I didn't re-run `check 21` against the cache, because that would write
new results and graph entries. Replaying it with `scripts/live_demo_run.py --cache
demo-app/.notbydefault/cache-live-20261011T202940Z` should make 0 requests. That's untested here.

Don't combine it with run A ($407, flash-lite verdict) or with the 0/12 → 12/12 resolve result (C10).
That result belongs to run A's verdict and can't happen under run B's.

### Exact claims to use

1. "Say an agent picks `gpt-oss-120b` for action-item extraction because it's 'cost-effective, strong at extraction, widely used'." *(scripted example reason)*
2. "NotByDefault tests that choice on the project's own 12 cases, 8 of them hard: three models, each alone and with a verifier, scored against reference answers by two blind judges from different model families."
3. "Today's model: 10 of 12 correct. At the assumed volume, a projected $418 a month, over the $375 budget."
4. "Cheapest setup that clears the bar: `gpt-oss-20b` plus a verifier. Also 10 of 12, at a projected $336 a month, about $82 a month less."
5. "Best quality within budget: `gemini-3.1-flash-lite`. 11 of 12, all 8 hard cases, projected $345 a month, but much slower (about 14 s vs 1.5 s)."
6. "The whole check: 71 real requests to 5 models, resumed across a free-tier quota reset, under 5 cents at list price."
7. "`resolve` recommends the switch and stops: adding a verifier loop to the code isn't automated yet."
8. *(Optional, demo dataset)* "Add 'data must stay in the EU', and the Groq decisions go stale while MongoDB is flagged for human review."

On screen or in the caption: "Costs are list-price projections at an assumed 20,000 users x 60 calls/month; run on free tiers. 12 cases: one case = 8 points."

### Don't say

- "$407/month" or "saved $62": run A.
- "Generated tests went from 0/12 to 12/12" or "resolve switched it": run A's verdict, demo dataset only.
- "Saved $81.56": it's projected, and the switch isn't applied.
- "Same quality": only case correctness matches; F1 and judge scores are lower.
- "$0.0221" or "$0.0561" as the run's cost.
