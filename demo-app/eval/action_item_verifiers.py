"""Verifiers for the action-items decision (hand-written; objective, no LLM).

Production checks (run on every output, failures are fed back on retry):
  - valid JSON in exactly the shape the app parses, with enum priorities
  - every owner appears in the notes (no invented people)
  - every due date is one the notes actually mention, resolved with the prompt's date rules
  - dates are real calendar dates written as YYYY-MM-DD
  - no item is missing a due date the notes clearly state for it
Evaluation-only checks (need a reference answer; never used as retry hints, because
production has no reference answers):
  - no reference action item is missing
  - nothing beyond the reference is listed (e.g. a task cancelled later in the notes)

`python eval/build_action_item_cases.py <decision_id>` copies this file to
.notbydefault/verifiers/<decision_id>.py, where `notbydefault check` loads it.
"""
from notbydefault.fit.checks import *  # noqa: F401,F403

ACTION_ITEMS_SCHEMA = {
    "type": "object", "required": ["action_items"], "additionalProperties": False,
    "properties": {"action_items": {"type": "array", "items": {
        "type": "object", "required": ["task", "owner", "due_date", "priority"],
        "additionalProperties": False,
        "properties": {
            "task": {"type": "string"},
            "owner": {"type": ["string", "null"]},
            "due_date": {"type": ["string", "null"]},
            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
        }}}},
}

CHECKS = [
    json_matches_schema(ACTION_ITEMS_SCHEMA,
                        why="The app does json.loads(...)['action_items']; anything else is a 502."),
    owners_in_input(why="An owner who isn't in the notes is an invented fact."),
    dates_in_input(why="A due date the notes don't support is an invented fact."),
    dates_normalized(why="Downstream code expects ISO dates or null."),
    due_dates_present(why="A due date the notes state but the output drops is a lost deadline."),
    no_missing_items(why="Evaluation only: a dropped action item is the costliest error."),
    no_extra_items(why="Evaluation only: cancelled or reassigned-away items must not appear."),
]

RUBRIC_CHECK = None   # everything above is objective; no LLM verifier needed
