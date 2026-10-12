# X promo package: NotByDefault

The figures come from run B only (trace `0f65667e9f5c521dfc124c89f188f9ca`, 2026-10-12); see
`docs/demo_claims_audit.md`. Every dollar-per-month figure is a **projection** at an assumed 20,000
users × 60 calls a month. The agent's reason in the demo is **scripted**. Don't mix in $407, $62 or
"0/12 → 12/12": those belong to an earlier run.

## Hook options

1. "Your coding agent said 'cost-effective.' We checked." (on screen: *Agent said cost-effective. We checked.*)
2. "AI agents pick your stack with a one-line reason. Nobody tests the reason." (on screen: *Nobody tests the agent's reason.*)
3. "This model choice was 'cost-effective.' It was over budget." (on screen: *"Cost-effective." Actually over budget.*)

The script below uses hook 1.

## Video script (about 55 s)

The story: the agent's claim → the evidence → the better-supported decision → the verified outcome.

| Time | Visual (real footage only) | Voiceover | On-screen text (max 6 words) |
|---|---|---|---|
| 0:00–0:05 | Terminal: `claims 21` output, zoomed on the `Reason: "…cost-effective…"` line | "Your coding agent picked a model and said it was cost-effective." | Agent said: "cost-effective." |
| 0:05–0:10 | Same output, scroll to the "scripted demo reason" label | "This is a scripted example reason. Every check after it runs on real code and real data." | Scripted reason. Real checks. |
| 0:10–0:18 | Terminal: `check 21` header (3 candidates, cost estimate, `--max-cost` cap) | "NotByDefault tests the choice on the project's own twelve cases: three models, alone and with a verifier, scored against reference answers." | 12 cases. 3 models. Capped spend. |
| 0:18–0:28 | Terminal: the fit-check table; highlight the gpt-oss-120b row ($418.04, NO) | "Today's model gets ten of twelve right. At the assumed volume it's a projected four hundred eighteen dollars a month, over the three seventy-five budget." | 10/12 · $418/mo · over budget |
| 0:28–0:38 | Same table, highlight gpt-oss-20b + verifier ($336.48, yes) and flash-lite (11/12, $345.07) | "A smaller model with a verifier also gets ten of twelve, for three thirty-six. Gemini Flash-Lite gets eleven, inside the budget too." | Same 10/12 · $336/mo projected |
| 0:38–0:44 | Terminal: claims verdicts `FALSE cost`, `FALSE quality`, `UNTESTED popularity` | "So the agent's claims get verdicts: cost-effective, false. Widely used isn't evidence at all." | Claim: false. With evidence. |
| 0:44–0:50 | UI: #21 receipt → Fit-check evidence card (trace ID, $0.0461, 71 live requests) | "Everything lands on a receipt: the data, the rejected options, the trace. The live run cost under five cents." | Receipt: evidence, trace, under 5¢ |
| 0:50–0:55 | Terminal: `require … "data must stay in the EU"` → STALE / NEED REVIEW; UI badges | "Change a requirement, and the decision reopens. Stale when a documented fact breaks it, needs-review when a human should judge." | New requirement → stale / needs_review |
| 0:55–0:60 | Repo page on GitHub, README header | "NotByDefault. Prove the choice, recheck the decision." | github.com/Mischa0099/notbydefault |

Don't say "saved". Say "projected": the switch isn't applied, because `resolve` can't add a verifier
loop yet. If you show `check 21` from the recording copy, its `Spent $0.0000` comes from the cached
replay. Show the live cost ($0.0461) from the receipt or `demo-live-run.txt`.

## Screen recordings needed (all real, per `docs/recording_plan.md`)

1. `python -m notbydefault claims 21 --project .demo-dataset/rec`: the reason line, the scripted label, the three verdicts (plan clip 2).
2. `python -m notbydefault check 21 --project .demo-dataset/rec --max-cost 0.15 --yes`: the header and the fit-check table (clip 3).
3. `python -m notbydefault ui --project .demo-dataset/rec --read-only`: the #21 receipt, the Fit-check evidence card with the trace ID, $0.0461 and 71 live requests (clip 4).
4. `python -m notbydefault require .demo-dataset/rec "data must stay in the EU"`, then a UI refresh showing the stale / needs_review badges (clip 5).
5. The GitHub repository page and README (https://github.com/Mischa0099/notbydefault).
6. Optional: the neatlogs trace `0f65667e…` opened in neatlogs (needs your login).
7. Optional: a live hook capture in Claude Code (plan clip 1b), labeled as live.

There's no GitHub PR to film, and none is shown.

## X posts (each under 280 characters)

1. My coding agent picked a model because it was "cost-effective." NotByDefault tested it on our own 12 cases: 10/12, a projected $418/mo against a $375 budget. Same 10/12 setup: ~$336/mo projected. Claim: false, with evidence. #neatHack
https://github.com/Mischa0099/notbydefault

2. AI agents pick models, DBs and SDKs with a one-line reason. NotByDefault catches the choice, checks each claim against your data, records a receipt, reopens it when a requirement changes (stale vs needs_review). Solo #neatHack build.
https://github.com/Mischa0099/notbydefault

3. "Widely used" isn't evidence. NotByDefault runs candidates on your cases with verifiers, blind judges, CIs and a hard cost cap. Our live check: 71 real requests, under 5¢ at list price, traced in neatlogs. #neatHack
https://github.com/Mischa0099/notbydefault

## Thumbnail line

**Agent said cost-effective. Proved false.**

## Closing call to action

On screen: **github.com/Mischa0099/notbydefault**. Voiceover / post: "Point it at your agent's next stack choice. Code and evidence are on GitHub."
Demo video link: _VIDEO_URL_PLACEHOLDER_
