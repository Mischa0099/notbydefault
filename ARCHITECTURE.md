# NotByDefault architecture

NotByDefault catches the consequential stack choices a coding agent makes (a database, a model,
a provider) at the moment it makes them, records each one in a Decision Graph, tests it against
the project's real constraints, and keeps checking that the reasons still hold. The motto is
the rule: **no decision without a why**.

```
 Claude Code ──PreToolUse──► hooks/pre_tool.py ──► detect.py ──► Decision Graph (graph.db)
     │                             │  agent_reason (claims.py)        ▲   ▲   ▲
     └──UserPromptSubmit──► hooks/user_prompt.py ──► pivot.py ────────┘   │   │
                                                                          │   │
 CLI (cli.py): check · require · resolve · claims · watch · scan · ui ────┘   │
                                                                              │
 Entire (entire.py): Graph (affected code) · checkpoints · transcripts · Brain┘
 neatlogs (fit/tracing.py, telemetry.py): one trace per run, a span per step and model call
```

## The Decision Graph (`graph.py`)

SQLite at `<project>/.notbydefault/graph.db`: `nodes(id, type, label, data_json, created_at,
session_id)` and `edges(src, dst, type)`.

| node type | what it is |
|---|---|
| goal / requirement / constraint | what the project needs (from `init`, `require`, `check`) |
| decision | a stack choice; `data.status` is unreviewed, reviewed, needs_review or stale |
| evidence | a fit-check results file, list prices, verifiers |
| alternative | a setup a fit check rejected, with its reason |
| code | a file or function a decision affects (Entire Graph) |
| pivot | a change of direction the user proposed, and their answer |

Edges: `depends_on` (decision → requirement/constraint), `serves` (→ goal), `supported_by`
(→ evidence), `rejects` (→ alternative), `affects` (→ code). The hooks open the graph with a short
lock timeout because parallel tool calls fire them concurrently. Everything else in
`.notbydefault/` is plain files: `results/`, `samples/`, `verifiers/`, `failures/` (failure
bank), `callsites/`, `cache/` (model response cache), `resolve/`, `hook.log`.

A decision's `data` also carries, when known:

- `agent_reason` / `agent_reason_source`: the agent's own words when it made the choice (the
  tool call's description plus its last message in the session transcript), recorded by the hook.
- `claims`: the claim check of that reason (below).
- `assumptions`: what its verdict relied on, each with a source, plus `broken` when `watch` finds
  one that no longer holds.
- `reason`: a reason given by a person (UI "mark reviewed").
- `history`: its status changes; `resolution`: what `resolve --apply` changed.

## Catching decisions (hooks)

- **PreToolUse** (`hooks/pre_tool.py`, `detect.py`): Bash/PowerShell installs and Write/Edit of
  manifests and model literals are matched against known services and model ids. A new choice
  is recorded as an unreviewed decision linked to the goal, requirements and touched code; the
  user is asked to confirm, with a briefing of at most 4 lines (`briefing.py`: related past
  decisions, rejected alternatives, history, reusable code). A choice the project already made is
  not asked again. A hook call takes about 60-120 ms (hook.log); Entire lookups run in a
  detached process.
- **UserPromptSubmit** (`hooks/user_prompt.py`, `pivot.py`): a regex pre-filter, then one fast
  Groq call confirms a real change of direction; the pivot is recorded with what it would undo
  (conflicting constraints, invalidated decisions, cost at list price) and the user's answer
  (accepted marks those decisions stale). Both hooks fail open.

## Testing a decision: `check` (`fit/`)

`fit/run.py` orchestrates: find the call site (`callsite.py`), build inputs (`samples.py`, the
failure bank first), pick candidates from the provider catalog (`candidates.py`,
`providers.yaml`), generate reviewable verifiers (`verifiers.py`, built from vetted blocks in
`checks.py`), run every input on every setup alone and with the verifier loop (`evaluate.py`,
capped by `--max-cost`), score against reference answers and one or two blind judges from different
families, compute intervals and monthly cost at the recorded volume (`stats.py`), decide (keep /
switch / insufficient), and write the report and decision record (`report.py`). The graph gets
the evidence, rejected alternatives and the verdict; the results file keeps every row and the
trace id. After the verdict the check records the decision's **assumptions** and runs the
**claim check** when the agent stated a reason, and prints its block under the report.

## Keeping decisions honest

- **`require`** (`stale.py`): a new requirement becomes machine-checkable constraints; each
  decision is sorted into stale (a documented fact or list price × volume breaks it),
  needs review (it may conflict: depends on deployment, the provider documents no guarantee
  at all, or only the model's judgement says so) or valid. Residency: a provider that names
  its regions without the required one is stale (Groq: US only); one that guarantees no region
  is needs review (Gemini). A resolved decision is judged on the model it now uses
  (`resolution.to`), not the one it was caught with.
- **`claims <id>`** (`claims.py`): the agent's stated reason (from the hook, else read back from
  the Entire session transcript) is split into typed claims by one cheap model call (only spans
  quoted verbatim are kept; keyword split as fallback): cost, quality, latency, compliance,
  maturity, popularity. Each is checked against what we already have: cost = the fit check's
  measured cost or list price × recorded volume vs the budget; quality and latency = the fit
  check's results; compliance = documented residency (`providers.yaml`, `service_facts.yaml`);
  popularity is never evidence for your project; maturity is untested. Verdict: true / false /
  untested, each with its source (results file + trace id, price source + date, residency URL).
  Shown in the check report, the decision record, the UI and the resolve PR.
- **Assumptions** (`assumptions.py`): volume, budget, prices + fetched date, data region, model
  list and requirements in force, each with a source, stored on the decision.
- **`watch`** (`watch.py`): compares them with now (requirements, `providers.yaml` prices, new
  models in the providers' lists, real call volume from neatlogs production traces when
  `--volume-traces` and `NEATLOGS_TOKEN` are given) and re-opens a decision whose assumption
  broke: needs review, cause recorded, failure-bank cases first in the next check
  (`re-opened #21: 'data region: any' broken by 'EU only'`). `--auto-check` queues checks within
  `--max-cost` and stops at a provider's daily limit.
- **`scan`** (`scan.py`): decision debt = unchecked (no verdict, no reason; plus model literals and
  service packages in the code that no decision records) + claims false + assumptions expired, with
  a table of the decisions behind each count and the next command for each. `--record` adds the
  unrecorded choices to the graph.
- **`resolve <id>`** (`resolve.py`): from a switch verdict to a reviewable branch: investigate
  (Entire Graph, records, failure bank, and the project tests that already exercise each call site
  and caller: `scout.existing_tests`, read from the test files' AST), reuse the fit check if the
  call site and verifiers are unchanged, generate tests, edit every call site the decision owns,
  run the generated tests, the project's tests and the verifiers on the new code (from the response
  cache), update the record and write a PR description (evidence, existing and generated tests,
  claim check, traces, checkpoints). Dry run unless `--apply`; never touches main; commits only on
  yes; rolls back on failure (the generated test file too).
- **Generated tests** (`testgen.py`): `tests/test_nbd_<id>_generated.py`, one case per failure-bank
  input and per reference answer, inserted into a fixed template (no model writes code). Each case
  loads the call site's module with a fake `openai` module, calls the site's function (parameters
  named like a sample field get it, dates converted from the annotation; the rest the user message)
  and asserts one request to the new model (and provider base URL); fed its reference answer, the
  fit-checked site must return it. A method is called on an instance from its no-arg constructor,
  else with `__init__` mocked to a no-op and attributes set from the case's inputs; if neither
  reaches the model the case is skipped and the PR names it with the reason. A whole-file site
  (Entire off) becomes every function and method in the file that uses the model id
  (`model_functions`; the PR says "Entire off: tested N functions calling the model"). resolve runs
  the file before the edit (the old model: routing fails) and after it; the step fails if any case
  fails after. Counts go in `resolution.generated_tests`, the PR and the UI receipt. Both runs, like
  resolve's project-suite run, are serial with `-p no:cacheprovider --no-header` (the suite also
  `-x`; the generated file not, so its counts are complete).

## Records and Entire

`records.py` keeps `decisions/NNNN-<slug>.md` per decision (a generated summary block: why,
evidence, rejected alternatives, caveats, claim check, watched assumptions, affected code,
resolution, history). The file is committed with the code, so it lands in the Entire checkpoint of
that commit (`checkpoints.py` finds it: `https://entire.io/gh/<owner>/<repo>/commit/<sha>`).
`entire.py` wraps the CLI; every call degrades gracefully. `affected.py` uses Entire Graph for
every function using a decision's model or service; `why.py` answers only from recorded facts,
cited; `scout.py` mines checkpoint transcripts for untested edge cases and lists the existing
project tests for each call site and caller.

## Observability, evals, costs

Every run is one neatlogs trace (`fit/tracing.py`): a span per agent step with a `focus_area`, a
span per model and judge call; journals are kept locally when export fails. `evals.py` measures
NotByDefault itself on cases with known answers (detection, pivot, requirements, fit), original
and held-out, results in `eval/results/`. `costs.py` writes per-decision savings and our own cost
per check (assumptions labelled) from files only.

## The UI (`ui/`)

`python -m notbydefault ui --project <dir>`: FastAPI + Jinja2 (autoescaped) + HTMX + Cytoscape,
JS vendored, bound to 127.0.0.1. `data.py` reads the graph read-only (sqlite `mode=ro`, one short
read per request) and the files above; `routes.py` serves the pages (overview with headline tiles,
decision debt, activity and per-decision savings; decisions; the decision receipt; graph;
timeline; evals; costs); `actions.py` calls the existing functions (review with a reason,
`require`, resolve dry run) behind a CSRF token + same-origin check, disabled with `--read-only`.
Live mode polls a file-stat version every 2 s; the graph animates new nodes in.

## Demo dataset

`python scripts/demo_dataset.py --out .demo-dataset/demo-app` resets that copy (gitignored) from
demo-app (graph copied with SQLite's backup API, so the live one is never touched) and runs the
whole loop on it: `watch` baseline; scripted, labelled agent reasons on #7, #9, #10, #13, #21;
`resolve 21` dry run then `--apply` on its own branch; one pivot ("let's drop verifiers, they
slow things down": the real Pivot Check runs, and when it records nothing a pivot marked
`source=seeded` is written); an agent turn through the PreToolUse hook (reason from the scripted
`scripts/demo_transcript.jsonl`); `claims` on every reasoned decision (#21: cost and quality
false, popularity untested); the requirement "data must stay in the EU" (#21, now on Gemini, needs
review; #13 on Groq goes stale); `watch`; `scan`; then `scripts/ui_screenshots.py` regenerates
docs/screenshots in light and dark (`--no-shots` skips it). Every step prints the trace id, the
graph ids and the files it wrote. Model calls come from the copy's response cache (carried over
unless `--fresh`); each call that would reach a provider is printed, and `--offline` skips them
(claims fall back to the keyword split, require to the region the text names, the pivot to the
seeded record).

## Tests

`pytest` (no network): fake providers (`tests/conftest.py`'s `FakeTransport` behind the real
`ProviderSet`), a fake Entire CLI, throwaway projects in `tmp_path`, and a guard that fails any
test touching `demo-app/.notbydefault` or reaching the OpenRouter models API.
