# 0002. AI model for: Extracts structured action items—including task description, owner, resolved due date, and priority—from meeting notes.

- Date: 2026-10-11
- Status: Change recommended: gemini:gemini-3.1-flash-lite
- Run on free tier; costs are list-price estimates.
- Decision graph node: #21 (`openai/gpt-oss-120b` -> `groq:openai/gpt-oss-120b`)

## Context

Decision #21 uses openai/gpt-oss-120b (groq:openai/gpt-oss-120b) to extracts structured action items—including task description, owner, resolved due date, and priority—from meeting notes. We ran 12 cases (12 hand-written, 6 of them re-run first from the failure bank) through 3 models, each alone and with a verifier, scored every answer against hand-written reference answers, and had 2 judge(s) from different families score them blind. Cheapest setup that meets the bar: gemini:gemini-3.1-flash-lite, correct on 11/12 cases, 8/8 hard, verifier pass rate 92% [75%-100%], $0.000288/call, $345.07/month. Best quality within the budget: gemini:gemini-3.1-flash-lite, correct on 11/12 cases, 8/8 hard, item F1 0.98, $0.000288/call, $345.07/month. At the assumed volume, today's setup costs about $407.35/month vs $345.07 for gemini:gemini-3.1-flash-lite (budget $375.00).

## Setups tested

| setup | correct (95% CI) | hard | item F1 | verifier pass | judge gemini-3.8-flash | judge qwen3.8-27b | latency | cost/call incl. retries | monthly | fits budget? |
|---|---|---|---|---|---|---|---|---|---|---|
| groq:openai/gpt-oss-120b (current) | 10/12 [58%-100%] | 6/8 | 0.94 | 92% | 4.58 | 4.58 | 1.5s | $0.00034 | $407.35 | NO |
| groq:openai/gpt-oss-120b + verifier (<=2 retries) | 11/12 [75%-100%] | 7/8 | 0.97 | 100% | 4.75 | 4.67 | 1.7s | $0.00038 (1.08 tries) | $458.71 | NO |
| groq:openai/gpt-oss-20b | 7/12 [33%-83%] | 3/8 | 0.76 | 58% | 3.83 | 3.92 | 1.3s | $0.00019 | $225.13 | yes |
| groq:openai/gpt-oss-20b + verifier (<=2 retries) | 10/12 [58%-100%] | 6/8 | 0.88 | 92% | 4.25 | 4.42 | 2.0s | $0.00032 (1.50 tries) | $385.37 | NO |
| gemini:gemini-3.1-flash-lite | 11/12 [75%-100%] | 8/8 | 0.98 | 92% | 4.92 | 4.42 | 5.7s | $0.00029 | $345.07 | yes |
| gemini:gemini-3.1-flash-lite + verifier (<=2 retries) | 12/12 [100%-100%] | 8/8 | 1.00 | 100% | 5.00 | 4.50 | 5.8s | $0.00033 (1.08 tries) | $391.43 | NO |

## Verdict

Cheapest setup that meets the bar: gemini:gemini-3.1-flash-lite, correct on 11/12 cases, 8/8 hard, verifier pass rate 92% [75%-100%], $0.000288/call, $345.07/month. Best quality within the budget: gemini:gemini-3.1-flash-lite, correct on 11/12 cases, 8/8 hard, item F1 0.98, $0.000288/call, $345.07/month.

1. Cheapest setup that meets the bar: `gemini:gemini-3.1-flash-lite`
2. Best quality within the budget: `gemini:gemini-3.1-flash-lite`

Bar: correct on at least 9/12 cases (today's groq:openai/gpt-oss-120b: 10/12, one case of slack) with a verifier pass rate at most one case lower.

## Case results (vs reference answers)

```
groq:openai/gpt-oss-120b: wrong on 2/12 (c03-several-owners, c07-mixed-language)
    c03-several-owners [hard]: Run the user interviews and write the summary: due None (expected '2026-10-21')
    c07-mixed-language [hard]: Revisar los contratos con legal: due '2026-10-19' (expected '2026-10-20')
    + verifier: retried 1 case(s) (c07-mixed-language); rescued 1 (c07-mixed-language); 1 wrong answer(s) passed every production check (c03-several-owners)
groq:openai/gpt-oss-20b: wrong on 5/12 (c02-relative-dates, c03-several-owners, c04-cancelled-later, c06-long-notes, c09-date-changed-later)
    c02-relative-dates [hard]: Review the security questionnaire: due '2026-10-15' (expected '2026-10-16') (+1 more)
    c03-several-owners [hard]: Run the user interviews and write the summary: due '2026-10-19' (expected '2026-10-21')
    c04-cancelled-later [hard]: Prepare the offsite agenda: due '2026-10-10' (expected '2026-10-12')
    c06-long-notes [hard]: output is not valid JSON with an action_items list
    c09-date-changed-later [hard]: Send the vendor comparison: due '2026-10-24' (expected '2026-10-26')
    + verifier: retried 5 case(s) (c02-relative-dates, c03-several-owners, c04-cancelled-later, c06-long-notes, c09-date-changed-later); rescued 3 (c03-several-owners, c06-long-notes, c09-date-changed-later); 0 wrong answer(s) passed every production check
gemini:gemini-3.1-flash-lite: wrong on 1/12 (c10-priority-cues)
    c10-priority-cues: Patch the auth bypass: due '2026-10-21' (expected None)
    + verifier: retried 1 case(s) (c10-priority-cues); rescued 1 (c10-priority-cues); 0 wrong answer(s) passed every production check
```

## Rejected alternatives

- `groq:openai/gpt-oss-120b + verifier (<=2 retries)`: over budget: $458.71/month vs $375.00
- `groq:openai/gpt-oss-20b`: below the bar: correct on 7/12 cases, 3/8 hard vs 10/12 for groq:openai/gpt-oss-120b
- `groq:openai/gpt-oss-20b + verifier (<=2 retries)`: over budget: $385.37/month vs $375.00
- `gemini:gemini-3.1-flash-lite + verifier (<=2 retries)`: over budget: $391.43/month vs $375.00

## Caveats

- Case c10-priority-cues: reworded on 2026-10-10 during development, after all three models omitted the item: "Optional: Sven can explore a dark mode prototype if there's time" became "Sven will explore a dark mode prototype if there's time - optional", because the original was arguably not a commitment, so the reference answer was ambiguous.
- Only 12 cases: one case is 8 percentage points of correctness, so intervals are wide and small differences are not meaningful.
- A judge shares a provider with a candidate; compare the two judges' columns for self-preference (correctness against references is unaffected).

## Assumptions and evidence

- Monthly cost = 20,000 expected users (from the graph) x 60 calls per user per month (ASSUMPTION) x measured average cost per call; monthly budget $375.00 (from the graph).
- Run on free tier; costs are list-price estimates. Prices: groq: https://console.groq.com/docs/models (fetched 2026-10-10); gemini: https://ai.google.dev/gemini-api/docs/pricing (fetched 2026-10-10).
- Test inputs: 12 hand-written, 6 of them re-run first from the failure bank (12 hand-written with reference answers, 8 marked hard); failure-bank cases always run first.
- Verifier pass = all checks in .notbydefault/verifiers/21.py pass (json_matches_schema, owners_in_input, dates_in_input, dates_normalized, no_missing_items, no_extra_items). Verified setups retry up to 2x, feeding back only the failures production could see (reference checks are never used as hints); their cost/latency include those retries and any verifier calls.
- Two blind judges from different families, gemini:gemini-3.8-flash and groq:qwen/qwen3.8-27b: on the same 36 outputs they gave identical scores 69% of the time and were within 1 point 97% of the time (mean gap 0.36; mean gemini-3.8-flash 4.19 vs qwen3.8-27b 4.17). Correctness against the references decides; judge scores only break ties.
- Verifier trust: not measured yet - run `python -m notbydefault label 21` to hand-label 10 outputs.
- Raw results: `.notbydefault/results/21-20261011T070521Z.json`

<!-- notbydefault:summary:begin (generated; edits inside are overwritten) -->

## Summary (NotByDefault)

- Decision graph node: #21 (model/ai-model: `openai/gpt-oss-120b`)
- Status: Reviewed
- Recommended: `gemini:gemini-3.1-flash-lite`
- Detected in agent session: `manual-action-items`

### Why

Cheapest setup that meets the bar: gemini:gemini-3.1-flash-lite, correct on 11/12 cases, 8/8 hard, verifier pass rate 92% [75%-100%], $0.000288/call, $345.07/month. Best quality within the budget: gemini:gemini-3.1-flash-lite, correct on 11/12 cases, 8/8 hard, item F1 0.98, $0.000288/call, $345.07/month.

### Evidence

- #26 Fit test results 2026-10-10 (12 inputs): `.notbydefault/results/21-20261011T070521Z.json`
- #27 Provider list prices fetched 2026-10-10: `groq: https://console.groq.com/docs/models (fetched 2026-10-10); gemini: https://ai.google.dev/gemini-api/docs/pricing (fetched 2026-10-10)`
- #28 Verifiers for decision #21: `.notbydefault/verifiers/21.py`
- Raw fit-test results: `.notbydefault/results/21-20261011T070521Z.json`

### Rejected alternatives

- `groq:openai/gpt-oss-120b + verifier (<=2 retries)`: over budget: $458.71/month vs $375.00
- `groq:openai/gpt-oss-20b + verifier (<=2 retries)`: over budget: $385.37/month vs $375.00
- `gemini:gemini-3.1-flash-lite + verifier (<=2 retries)`: over budget: $391.43/month vs $375.00
- `groq:openai/gpt-oss-20b`: below the bar: correct on 7/12 cases, 3/8 hard vs 10/12 for groq:openai/gpt-oss-120b

### Caveats

- Assumption #4: Expected users: 20,000 (demo scenario assumption)
- Assumption #5: Monthly budget: $375 (demo scenario assumption)
- Assumption #12: Calls per user per month: 60 (demo scenario assumption)

### Affected code (entire-graph)

- `actions.py:28 extract_action_items`: uses 'openai/gpt-oss-120b' (line 34)
- `app.py:22 summarize_notes`: uses 'openai/gpt-oss-120b' (line 27)
- `app.py:58 Handler.do_POST`: calls summarize_notes

### Status history

- 2026-10-11T07:11:44+00:00: reviewed: fit check verdict: switch (Cheapest setup that meets the bar: gemini:gemini-3.1-flash-lite, correct on 11/12 cases, 8/8 hard, verifier pass rate 92% [75%-100%], $0.000288/call, $345.07/month. Best quality within the budget: gemini:gemini-3.1-flash-lite, correct on 11/12 cases, 8/8 hard, item F1 0.98, $0.000288/call, $345.07/month.)

<!-- notbydefault:summary:end -->
