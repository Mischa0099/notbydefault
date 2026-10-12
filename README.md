# NotByDefault

**Prove the choice. Recheck the decision.**

AI coding agents pick models, databases and SDKs by default and give a reason. NotByDefault checks whether that reason is true on your project's own data, records the evidence, and reopens the decision when a requirement, price or fact changes.

> No decision without a why.

## Example

An agent picked `openai/gpt-oss-120b` for an action-items endpoint because it was "cost-effective, strong at extraction, widely used". On the project's 12 cases it scored 10/12 at $418/mo, over the $375 budget. The cheapest setup that meets the bar is 10/12 at gpt-oss-20b + verifier, $336/mo: **$418→$336/mo, $81.56/mo saved**. `notbydefault resolve 21` recommends that switch and stops there: adding a verifier loop to the code isn't automated yet, so it changes nothing.

Full comparison between candidates: gpt-oss-20b + verifier is the cheapest in-budget recommendation at $336/mo; gemini-3.1-flash-lite scored higher (11/12 vs 10/12) but $345/mo is still under budget — the ranking depends on whether you prioritize cost or correctness. See the fit-check table.

When the requirement "data must stay in the EU" was added, #21 flipped to **stale** (Groq documents US-only; the chosen stack can't meet the new requirement), and so did an older Groq decision. MongoDB went to **needs_review**: where its data lives depends on how it's deployed. Stale means a known fact breaks it; needs_review means a human should judge.

Live trace in neatlogs: `0f65667e9f5c521dfc124c89f188f9ca` (71 real provider requests across a quota reset, $0.0461 at list price on free tiers).

## How it works

1. **Catch.** A Claude Code `PreToolUse` hook spots a decision when the agent writes a model name, installs an SDK or edits a manifest. It targets under 300 ms (p50 under 200 ms; worst case 405 ms under full-suite load) and never blocks the agent.
2. **Record.** The decision, the agent's stated reason and the code it touches go into a decision graph (SQLite, stored with the repo).
3. **Check the reasons.** `claims` splits the reason into cost, quality, latency, compliance and popularity claims, and gives each a verdict with a source. "Widely used" is never evidence for your project.
4. **Check the choice.** `check` runs several candidates on your own cases. Verifiers catch bad outputs, two judges from different model families score them, and confidence intervals flag runs that are too close to call.
5. **Act.** `resolve` applies the winner on a branch (`nbd/<id>-<slug>`), edits only that decision's call sites, generates tests from past failures and writes the PR text. It never touches main or pushes.
6. **Watch.** Each decision stores its assumptions: budget, volume, prices, data region, requirements. `watch` and `require` reopen the affected decisions when any of them change.
7. **Review.** `ui` shows the decision graph, a receipt per decision with its full evidence, a change timeline, eval results and costs. `scan` scores the repo's decision debt (the demo repo scores 14).

## Commands

| Command | What it does |
|---|---|
| `notbydefault check <id>` | Fit-check a decision on your own cases |
| `notbydefault claims <id>` | Verdict on each claim in the agent's reason |
| `notbydefault resolve <id>` | Apply the winner on a branch, with tests and PR text |
| `notbydefault require <project> "<text>"` | Add a requirement; mark decisions stale or needs_review |
| `notbydefault watch` | Reopen decisions whose assumptions changed |
| `notbydefault why "<question>"` | Answer from recorded decisions only, with sources |
| `notbydefault scan` | Decision debt: unchecked choices, false claims, expired assumptions |
| `notbydefault ui` | Web UI with the live decision graph |

## Quickstart

```bash
pip install -e .
cp .env.example .env         # add GROQ_API_KEY and GEMINI_API_KEY
notbydefault init demo-app
notbydefault check 21 --project demo-app
notbydefault ui --project demo-app
```

To rebuild the full demo scenario on a throwaway copy:

```bash
python scripts/demo_dataset.py --out .demo-dataset/demo-app --offline
```

## Results

On our own evals, decision detection improved from 0.81 to 1.00 recall over three versions. Pivot recall 0.67 → 1.00 on original cases; 0.5 on held-out cases (v2 and v3). We ship with the lower number visible — held-out is where the next version has to improve. 474 tests, no network.

## Sponsor tools

- **neatlogs:** every model call, verifier and hook traced as spans with tokens and cost.
  - Model calls go to Groq and Google AI Studio over HTTPS. Your project data (notes, cases, code paths) is sent to those providers as prompts. No calls to Anthropic or OpenAI. When `NEATLOGS_API_KEY` is set, prompts and outputs are also sent to neatlogs. When it's not set, no spans leave the machine.
- **Entire:** Entire's Graph finds the callers affected by each decision (so `resolve #21` edits `actions.py` but leaves `app.py` alone — decision #13's own evidence rejected Gemini for summaries). Checkpoints link decisions to the agent session that made them.
- **cfo.ai:** Pricing modeled with Ari (cfo.ai): $49/team/month Starter, $149/month Growth. Revised base case is self-funded through October 2027 under disclosed assumptions. Full plan: `business/plan.md`.

## Business case

NotByDefault reviews AI coding agents' technical choices against the project's own data and checks their stated reasoning, with proposed launch pricing of $49 per team per month. In our demo, one model-selection decision reduced projected inference costs by $82/month at the same accuracy on the project's own cases (10/12) — enough to cover the subscription if those savings persist. Our revised 12-month plan starts with $5,000 of founder cash, two pre-launch pilot customers at $49, 20% upgrade to $149 Growth in the third paid month, and a staged founder draw ($0 Nov–Apr, $2k/mo May–Oct) — resulting in a self-funded year with ~$5,354 ending cash and $2,789 MRR by October 2027 under disclosed assumptions.

## Known limitations

- Live demo may require a resume across days when provider free-tier quotas reset; the check saves state and resumes without duplicating requests.
- Eval results are measured on our own cases with held-out cases written before fixes. Not independently labeled.
- `test_hook_is_fast` is a timing flake (300 ms limit); passes when run alone.

## Submission

neatHack, October 10-12, 2026 · solo, [@Mischa0099](https://github.com/Mischa0099) · Demo video: _link after upload_