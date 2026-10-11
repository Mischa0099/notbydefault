# NotByDefault

## Entire integration

NotByDefault reads Entire for code and history, and writes decision records that Entire
checkpoints with your commits. Every Entire call degrades gracefully: when the CLI, a plugin,
login or a mirrored repo is missing, the command falls back (and says so) instead of failing.

| command | what it does | Entire used |
|---|---|---|
| `python -m notbydefault affected [id] --project P` | every file and function using a decision's model/service, stored as `code` nodes with `affects` edges | `entire graph symbols` / `edges` (falls back to the files the agent touched) |
| `python -m notbydefault require P "<requirement>"` | two lists: **stale** (a documented fact conflicts) and **needs review** (depends on deployment, or only the model's judgement) | Graph for the affected code |
| `python -m notbydefault records [id] --project P` | writes/updates `decisions/NNNN-<slug>.md` (why, evidence, rejected alternatives, caveats, affected code, history); `check`, `require` and accepted pivots do this automatically | Brain, if `entire-brain` is installed |
| `python -m notbydefault why "<question>" --project P [--llm]` | answers only from recorded facts, each cited (decision record, evidence, checkpoint/session/commit, code location); says "no recorded reason" otherwise | `entire search`, else `entire checkpoint list` + `git log`; Graph |
| `python -m notbydefault scout <id> --project P` | edge cases the agent discussed in sessions that touched the code, saved as *suggested* cases in `.notbydefault/suggested/<id>.jsonl` | `entire checkpoint list` / `explain --transcript` |
| `python -m notbydefault checkpoints <id> --project P` | the `entire checkpoint list` row and link for the commit that added the decision's record | `entire checkpoint explain --commit` |

The PreToolUse hook adds a briefing of at most 4 lines (related past decisions, rejected
alternatives, history, reusable code, each with a source id). It reads only the graph and
`.notbydefault/entire_cache.json`, then refreshes that cache in a detached process; `check`
shows the full briefing after a live refresh.

### Checkpoint links

A decision record lands in the Entire checkpoint of the commit that adds it. Its page is

    https://entire.io/gh/Mischa0099/notbydefault/commit/<FULL_SHA>

where `<FULL_SHA>` is the full 40-character commit SHA (`git log --diff-filter=A --format=%H --
decisions/<record>.md`, or `python -m notbydefault checkpoints <id>`). For example, the commit
that added `demo-app/decisions/0001-ai-model-openai-gpt-oss-120b.md` (checkpoint
`01M4KMWR0G2BHVEMNG9MMHHP6G`):
https://entire.io/gh/Mischa0099/notbydefault/commit/35c61621951e0317126ed33c88fdd99954608157

### Entire Brain

Not in the Entire plugin index as of entire 0.11.4 (`entire plugin search brain` finds nothing).
If you have it, install it with `entire plugin install <its repository URL or a local
entire-brain executable>`, then run `python -m notbydefault records --project P`. NotByDefault
only calls a Brain subcommand that the installed plugin's own `--help` documents.
