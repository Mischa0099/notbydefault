# Recording plan

Prepared 2026-10-12. On a disposable copy, I ran every command below against the current CLI
(`python -m notbydefault <cmd> --help`), with the provider and neatlogs keys blanked. Each
"Expected" block quotes that output. Numbers come from run B, trace `0f65667e…`; see
`docs/demo_claims_audit.md`.

## What's live and what's replayed

| Part | Real or replayed | Requests / cost |
|---|---|---|
| Hook catching a model choice (clip 1b) | **Real hook code, live**, on a file write you trigger | 0 model calls (0.4 s) |
| Agent's stated reason on #21 (clip 1) | **Scripted.** `scripts/demo_dataset.py` writes it, and every screen labels it "scripted demo reason …, not a real session" | 0 |
| Claim check (clip 2) | Real code. Offline it splits by keywords; the verdicts come from run B's results | 0 |
| Fit check #21 (clip 3) | Real code, **replaying run B's 71 live responses from cache** | 0 requests, prints `Spent $0.0000` |
| Receipt (clip 4) | Real UI over the copy's graph | 0 |
| EU requirement → stale / needs_review (clip 5) | Real code. Provider facts are local (`service_facts.yaml`, read 2026-10-11) | 0 offline (2 decisions print "could not judge") |
| Resolve (clip 6) | Real code, dry run | 0 |
| Live trace in neatlogs (clip 7) | Run B's real trace | You have to open it yourself |

Nothing in this plan makes a paid or free-tier provider request. **Don't run `scripts/live_demo_run.py`
for the recording.** It runs against the real `demo-app/` and rewrites `demo-live-run.txt`,
`docs/demo.md` and the #21 decision record. A fresh live run makes about 137 requests: estimated at
$0.11 at list price, capped at $0.15, and it hit the Gemini free-tier daily limit last time.

## Setup (off camera, about 2 minutes)

1. **Use Git Bash** for the terminal footage (Windows Terminal → Git Bash profile). This keeps the
   real CLI offline: `load_env()` never overrides an environment variable that already exists, and an
   empty variable counts as "not set", so no provider request can happen. PowerShell can't hold an
   empty environment variable (setting one to `""` deletes it), so the keys in `.env` would load.
2. In the repo root:
   ```bash
   cd ~/notbydefault
   export GROQ_API_KEY= GEMINI_API_KEY= OPENROUTER_API_KEY= OPENAI_API_KEY= NEATLOGS_API_KEY=
   export PYTHONIOENCODING=utf-8
   alias nbd='.venv/Scripts/python.exe -m notbydefault'   # optional; the plan spells out the full command
   .venv/Scripts/python.exe scripts/recording_setup.py
   ```
   Expected (about 3 s): five `#… "reason"` lines, then `Installed NotByDefault PreToolUse and
   UserPromptSubmit hooks in …\.demo-dataset\rec\.claude\settings.json`, then `ready: …\.demo-dataset\rec`.
   This creates `.demo-dataset/rec` (gitignored). Re-run it before every take; it wipes and rebuilds the copy.
3. Check that the keys are blanked without printing them:
   `.venv/Scripts/python.exe -c "import os;print([k for k in ('GROQ_API_KEY','GEMINI_API_KEY') if os.environ.get(k)])"`
   should print `[]`.
4. Terminal: a large font, a window about 120 columns wide (the fit-check table is about 230 characters wide, so
   either shrink the font for clip 3 or scroll horizontally), and `clear` between clips.
5. Don't show `.env`, `env`/`set` output or `demo-app/.claude/settings.json`. None of the commands
   below print keys.

The real `demo-app/` and `.demo-dataset/demo-app/` are only read. I checked that their graph and
decision-record timestamps didn't change after a full run of this plan.

## Clips in order

Total essential footage is about 2.5–3 minutes of raw capture before editing.

### Clip 1: a captured decision and its stated reason (essential, about 15 s)

```bash
.venv/Scripts/python.exe -m notbydefault show .demo-dataset/rec
```
Expected (verified): under `decision`, the line `#21   openai/gpt-oss-120b  [model/ai-model, reviewed]  from code: OpenAI model literal`,
plus the edges `--affects--> #23 actions.py:extract_action_items` and `--depends_on--> #5 Monthly budget: $375 (demo scenario assumption)`.

`show` doesn't print the agent's reason. The reason appears on screen in clip 2 (`Reason: "…"`, labeled
**scripted demo reason**). Also, before clip 2 runs, the UI receipt's Claim check panel reads
"No stated reason recorded" (verified), so **open the UI only after clip 2**. Say it's a scripted
example reason. Don't call it a live capture; that's clip 1b.

### Clip 1b: live hook capture (optional, but it's the only *live* detection; about 30 s)

This needs your manual action. It records a new decision in the copy only.
```bash
cd .demo-dataset/rec && claude
```
Ask, for example: *"Add a `tag_note(text)` helper in a new file tags.py that calls Groq's
llama-3.3-70b-versatile — it's cheap and fast."* When Claude Code writes the file, the PreToolUse
hook returns `permissionDecision: "ask"`.

Expected: a permission prompt reading `NotByDefault: this picks llama-3.3-70b-versatile as your model.
Run `…python.exe -m notbydefault check <id> --project …\.demo-dataset\rec` to test the fit.` followed by
a "Briefing" list. I verified this by feeding the hook a Write event: 0.4 s, exit 0. The new decision
appears in the UI (refresh) with `agent_reason_source: session transcript`, taken from Claude's own
message. Approve or deny the write as you like, then `cd ../..`.

Notes:
- The hook *asks*; it doesn't silently pass. The README's "never blocks the agent" means it never
  denies the call outright. Don't say "invisible".
- This clip spends your Claude Code usage, not NotByDefault's provider budget.
- Run `recording_setup.py` first. It reinstalls the hooks to point at the copy. A copy made any
  other way still points them at the real `demo-app`.

### Clip 2: claim check (essential, about 15 s)

```bash
.venv/Scripts/python.exe -m notbydefault claims 21 --project .demo-dataset/rec
```
Expected (verified, about 1 s):
```
#21 openai/gpt-oss-120b
Claim check: the agent's reasons, tested
The agent's reason (scripted demo reason (scripts/demo_dataset.py), not a real session) makes 3 claim(s): 0 true, 2 false, 1 untested (split by keywords).
Reason: "openai/gpt-oss-120b: cost-effective, strong at structured extraction, widely used in production"
  FALSE     cost       "cost-effective"
            measured $0.000348/call, $418.04/month at the assumed volume vs a $375.00 budget (OVER BUDGET); cheapest tested: groq:openai/gpt-oss-20b at $0.000198/call [fit check …21-20261012T070725Z.json; trace 0f65667e9f5c521dfc124c89f188f9ca]
  FALSE     quality    "strong at structured extraction"
            10/12 vs 11/12 at gemini-3.1-flash-lite, within budget [...]
  UNTESTED  popularity "widely used in production"
            not evidence for your project: ...
```
Show: the three verdict lines. "(split by keywords)" is the offline splitter. With keys present it
would use a model call instead; keep it offline.

### Clip 3: workload-specific comparison (essential, about 30 s)

```bash
.venv/Scripts/python.exe -m notbydefault check 21 --project .demo-dataset/rec --max-cost 0.15 --yes
```
Expected (verified, about 7.5 s, exit 0): the header (`candidates: groq:openai/gpt-oss-120b,
groq:openai/gpt-oss-20b, gemini:gemini-3.1-flash-lite`, `requests: ~137 (cached responses are reused,
never re-requested)`, `estimated cost: $0.11 … (hard cap --max-cost $0.15)`), then the table:

```
groq:openai/gpt-oss-120b (current)                     10/12 [58%-100%]   6/8 ... $418.04  NO
groq:openai/gpt-oss-20b + verifier (<=2 retries)       10/12 [58%-100%]   6/8 ... $336.48  yes
gemini:gemini-3.1-flash-lite                           11/12 [75%-100%]   8/8 ... 13.8s ... $345.07  yes
Recommendations:
  1. Cheapest setup that meets the bar: groq:openai/gpt-oss-20b + verifier (<=2 retries)
  2. Best quality within the budget:    gemini:gemini-3.1-flash-lite
...
Spent $0.0000 on this check.
```
The net-saving line reads `Net saving: $418.04/mo → $336.48/mo = $81.56/mo saved. Check cost once at $0.0000 (list). First-month ROI: $81.56.`
**Say "replayed from the live run's cached responses".** The $0.0000 and ROI $81.56 come from the
replay. The live run cost $0.0461 (show that in clip 7). The saving is a projection at the assumed
volume.

Show: the table and Recommendations. Ignore the "Rejected before … for #13" briefing lines. They
attribute #21's alternatives (#17–#20) to #13, which is a display quirk, so don't zoom in. Also skip
the `entire search unavailable` and `Entire Brain is not installed` notes.

### Clip 4: the decision receipt (essential, about 25 s)

Start the UI in a second Git Bash tab (with the same `export` line) and leave it running for clip 5:
```bash
.venv/Scripts/python.exe -m notbydefault ui --project .demo-dataset/rec --read-only
```
It opens `http://127.0.0.1:8765`, and pages respond in under 0.4 s (verified: `/`, `/decisions`,
`/decisions/21`, `/graph`, `/timeline`, `/evals` and `/costs` all return 200). Stop it with Ctrl+C at the end.

Open **Decisions → #21 openai/gpt-oss-120b**. The page header shows status `reviewed`, plus a
`read-only` badge. Scroll through: **Receipt**, **Claim check**, **Fit-check
evidence** (the same table, with the results-file link), **Rejected alternatives**, **Caveats &
assumptions** (20,000 users × 60 calls ASSUMPTION, $375 budget, prices fetched 2026-10-10, the
c10 reword caveat), **Resolve**, **Status history**. The overview page (`/`) has "Monthly savings,
per decision".

### Clip 5: a changed requirement → stale / needs_review (essential, about 25 s)

```bash
.venv/Scripts/python.exe -m notbydefault require .demo-dataset/rec "data must stay in the EU"
```
Expected (verified, about 1.7 s):
```
Requirement #34: data must stay in the EU
  checks: {"data_residency": "EU"}

2 of 5 decisions provably conflict (documented), now STALE:
  #13 openai/gpt-oss-120b [model/ai-model]
      why:   groq:openai/gpt-oss-120b runs on groq, which documents data residency: US (source: https://console.groq.com/docs/your-data (read 2026-10-11)); EU residency is required
  #21 openai/gpt-oss-120b [model/ai-model]
      ...
1 of 5 decisions NEED REVIEW (may conflict; not provable from the docs):
  #7 MongoDB [service/database]
      why:   MongoDB: where its data lives depends on how it is deployed ...; confirm it runs in EU
  #9 OpenAI: could not judge: GROQ_API_KEY is not set (needed for groq)
  #10 gpt-4o: could not judge: GROQ_API_KEY is not set (needed for groq)
```
The requirement number can differ by one if you recorded clip 1b. The two "could not judge" lines are
honest offline output; trim them in the edit or leave them. Then refresh the UI: #21's badge is now
**stale** and MongoDB's is **needs_review**.

### Clip 6: resolve (optional, about 15 s)

```bash
.venv/Scripts/python.exe -m notbydefault resolve 21 --project .demo-dataset/rec
```
Run it **before clip 5**. After the requirement, #21 is stale and resolve refuses to reuse the fit check.
Expected last line (verified): `Recommendation: groq:openai/gpt-oss-20b with a verifier and retries;
adding a verifier loop to the code is not automated yet. Nothing changed.` Before that, it lists the
affected code: `change actions.py:28`, `leave app.py:22 summarize_notes  owned by decision #13`.

### Clip 7: live trace (optional, about 10 s, manual)

Open neatlogs in the browser and find trace `0f65667e9f5c521dfc124c89f188f9ca` (final attempt). The
partial first attempt is `56a5915f9768ef65b7f21546bdfceb6a`. I couldn't verify that it opens. If it
doesn't, show `demo-live-run.txt` instead: its last lines are `trace id: 0f65667e…` and `spent: $0.0461
(list price of the 71 live requests …)`.

### Clip 8: decision debt (optional, about 10 s)

```bash
.venv/Scripts/python.exe -m notbydefault scan --project .demo-dataset/demo-app
```
Expected (verified after the Override Guard rebuild): `Decision debt: 15  (5 unchecked / 5 claim(s) false / 5 with expired assumptions)`. It was 14 before seeded decision #39 (the Override Guard demo, `source=seeded`) was added; #39 counts as unchecked. The receipt of #39 shows the orange Override Guard banner pointing at #21; say it is a seeded example.
Use the full demo dataset here, not `rec`. On `rec` after clip 5, it's `7`.

### Recommended take order

Setup → 1 → 2 → 3 → (6) → 4 (start the UI) → 5 → refresh the UI for the stale badges → (1b) → (8) → (7).
Clip 1b is last because it adds a decision. Recording it earlier only changes the requirement number in clip 5.

## GitHub footage and resolve / PR: flagged

- **No PR exists.** `gh pr list --state all` on `Mischa0099/notbydefault` returns nothing.
- **`resolve` never pushes or opens a PR.** It writes PR *text* locally after `--apply`. Under run
  B's verdict (gpt-oss-20b + verifier), it stops and changes nothing, because the verifier loop isn't
  automated. Decision #13 isn't an offline alternative either: its fit check can't be reused (app.py
  changed since), so `resolve 13 --apply` would start a new live fit check.
- The README's old "0/12 → 12/12 generated tests" result came from run A's verdict and can't be
  reproduced now (see the audit, C10). **Don't show or describe a PR.**
- Safe GitHub footage: the repository page, the README and the commit history on `main` (local `main`
  matches `origin/main`). Nothing in this plan pushes.

## Recovery

| Problem | What to do |
|---|---|
| Output shows `could not judge: … not set` in places other than clip 5, or a check prints a request count above 0 / `Spent` above $0.0000 | Stop. The keys weren't blanked or the cache is missing. Re-run setup steps 2–3 in a fresh Git Bash window. `recording_setup.py` prints `cache: missing …` if a source cache is gone. |
| Something does reach a provider and hits a rate limit (exit code 4, "daily limit") | Don't retry in a loop or raise caps. Free tiers reset at midnight Pacific. The demo needs no live calls, so fix the offline setup instead. |
| `check 21` makes requests anyway | Rebuild with `recording_setup.py`. The replay depends on the copied `cache-live-20261011T202940Z` and unchanged `samples/21.jsonl` / `verifiers/21.py`. Never point `live_demo_run.py` at it. |
| You need a fallback dataset with every step already done | `python -m notbydefault ui --project .demo-dataset/demo-app --read-only`. #21 is stale, MongoDB/OpenAI/gpt-4o are needs_review, there's the seeded pivot (#32, labeled seeded) and the hook-caught digest decision (#33, from the scripted transcript). Don't rebuild it with `demo_dataset.py` unless you add `--no-shots`, because without it the script overwrites the tracked `docs/screenshots/`. |
| The UI doesn't start: port in use | `--port 8766`. In PowerShell, `Get-NetTCPConnection -LocalPort 8765` shows who holds it. |
| The UI doesn't start: import error (fastapi/uvicorn/jinja2/multipart) | `.venv/Scripts/python.exe -m pip install -e ".[ui]"` (verified installed today). |
| The browser doesn't open | Add `--no-browser` and open `http://127.0.0.1:8765/decisions/21` by hand. |
| Garbled `→`/`—` characters | `export PYTHONIOENCODING=utf-8` and use a UTF-8 terminal font. |
| A take went wrong midway | Re-run `scripts/recording_setup.py` (about 3 s); it rebuilds `.demo-dataset/rec` from scratch. |
| Screenshot stills | `docs/screenshots/*-light.png` / `*-dark.png` exist (built from `.demo-dataset/demo-app`), as real UI captures. |

## Cleanup

```bash
# stop the UI with Ctrl+C, then:
rm -rf .demo-dataset/rec .demo-dataset/rec-check.txt   # disposable copy (gitignored)
git status --short   # expect only your own untracked files; demo-app/ and docs/ unchanged
```
Unset the blanked keys by closing the Git Bash window.

## Blockers and things that need your action

1. **The README quickstart is wrong for this checkout:** `pyproject.toml` has no `[project.scripts]`,
   so there is no `notbydefault` command (only `python -m notbydefault`), and `.env.example` doesn't
   exist. Don't film the README quickstart as written. Every command here uses `python -m notbydefault`.
   (The `scan` output's "next command" also says `notbydefault ui`.)
2. **Clip 1b (live capture)** needs you to drive Claude Code in `.demo-dataset/rec`.
3. **Clip 7** needs a neatlogs login to show the trace. I couldn't confirm it's viewable.
4. **No PR footage** is possible without new features. See the section above.
5. `.demo-dataset/demo-app/.claude/settings.json` still points its hooks at the **real** `demo-app`.
   Don't open Claude Code inside `.demo-dataset/demo-app`, because it would record into the real graph.
6. Labeling gap to mention in the narration: the dataset's hook-caught decision #33 shows
   `agent_reason_source: session transcript`, but that transcript is `scripts/demo_transcript.jsonl`,
   which is scripted. The file's first line says so, but the label doesn't.
