"""Build the hand-written action-item test cases (inputs + reference answers).

Dates in the references were worked out by hand from the calendar, following the rules in
actions.SYSTEM_PROMPT; running this script also checks every reference against the
NotByDefault verifiers, so a reference and a verifier can't silently disagree.

    python eval/build_action_item_cases.py [decision_id]

writes eval/action_item_cases.jsonl and, given a decision id, copies it to
.notbydefault/samples/<decision_id>.jsonl and eval/action_item_verifiers.py to
.notbydefault/verifiers/<decision_id>.py for `notbydefault check`.
"""
import json
import shutil
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from actions import build_user_message  # noqa: E402  (the exact format the app sends)


def item(task, owner, due, priority="medium"):
    return {"task": task, "owner": owner, "due_date": due, "priority": priority}


CASES = [
    # id, meeting date, hard?, tags, notes, reference items
    ("c01-explicit-dates", "2026-10-05", False, ["explicit dates", "blocking"],
     """- Kickoff for the Q4 billing migration.
- Maya will draft the migration plan by October 9.
- Leo to set up the staging database by 2026-10-12. This is blocking QA.""",
     [item("Draft the migration plan", "Maya", "2026-10-09"),
      item("Set up the staging database", "Leo", "2026-10-12", "high")]),

    ("c02-relative-dates", "2026-10-08", True, ["tomorrow", "next Friday", "end of week", "weekday"],
     """- Priya will send the revised quote to Acme tomorrow.
- Tom to review the security questionnaire by next Friday.
- Jun will update the onboarding doc by end of the week.
- Sam owes the finance team the Q3 numbers on Monday.""",
     [item("Send the revised quote to Acme", "Priya", "2026-10-09"),
      item("Review the security questionnaire", "Tom", "2026-10-16"),
      item("Update the onboarding doc", "Jun", "2026-10-09"),
      item("Send the Q3 numbers to the finance team", "Sam", "2026-10-12")]),

    ("c03-several-owners", "2026-10-14", True, ["several owners", "next Wednesday", "urgent"],
     """- Priya and Tom will co-write the RFP response, due Oct 23. Urgent - the client deadline is firm.
- Dana, Luis and Mei to run the user interviews; summary due next Wednesday.
- Luis will book the interview room.""",
     [item("Co-write the RFP response", "Priya, Tom", "2026-10-23", "high"),
      item("Run the user interviews and write the summary", "Dana, Luis, Mei", "2026-10-21"),
      item("Book the interview room", "Luis", None)]),

    ("c04-cancelled-later", "2026-10-07", True, ["cancelled later", "nice-to-have"],
     """- Dana will book the venue for the offsite on Friday.
- Omar to order team t-shirts by October 20 - nice to have.
- Ravi will prepare the offsite agenda by Monday.
- Discussed catering options, no decision.
- Update at the end of the meeting: we're going fully remote, so scratch the venue booking. Ravi's agenda is still needed.""",
     [item("Order team t-shirts", "Omar", "2026-10-20", "low"),
      item("Prepare the offsite agenda", "Ravi", "2026-10-12")]),

    ("c05-no-action-items", "2026-10-09", True, ["no action items"],
     """- Discussed whether to move to a monorepo; no decision yet.
- Lena shared the Q3 NPS results (up 4 points).
- Open question: do we need a second on-call rotation?
- Next sync in two weeks.""",
     []),

    ("c06-long-notes", "2026-10-12", True, ["long notes", "today", "end of month", "if there's time"],
     """Release readiness sync
- Attendees: Aisha, Ken, Bruno, Mei, Tom, Sara
- Release 4.2 is scheduled for this week; QA sign-off still pending.
- Sara walked through the burndown: 14 tickets closed last week, 5 still open.
- The checkout flow regression from last sprint is fixed in main.
- Aisha will fix the flaky checkout test today - it's blocking the release.
- Long discussion about test parallelism; CI time went from 18 to 26 minutes.
- Tom suggested we could try feature flags for risky changes (not decided).
- Ken to write the postmortem for the Oct 3 outage by Thursday.
- Customer support reported two tickets about slow invoice exports.
- Bruno will look into upgrading the CI runners if there's time.
- Sara mentioned the design review moved to next month.
- Metrics: error rate 0.4%, p95 latency 310 ms, both within targets.
- We agreed the on-call handbook is fine as is for now.
- Mei to send the release notes to support by end of the month.
- Brief chat about the team lunch, no plans made.
- Reminder that the office is closed on Friday for maintenance.
- Next release sync in two weeks.""",
     [item("Fix the flaky checkout test", "Aisha", "2026-10-12", "high"),
      item("Write the postmortem for the Oct 3 outage", "Ken", "2026-10-15"),
      item("Look into upgrading the CI runners", "Bruno", None, "low"),
      item("Send the release notes to support", "Mei", "2026-10-31")]),

    ("c07-mixed-language", "2026-10-15", True, ["Spanish/English", "el viernes", "próximo martes", "urgente"],
     """- Carlos va a enviar el presupuesto final el viernes.
- Ana will update the pricing page by October 22.
- Lucía revisará los contratos con legal el próximo martes. Urgente.
- Discutimos el nuevo logo, sin decisión.""",
     [item("Enviar el presupuesto final", "Carlos", "2026-10-16"),
      item("Update the pricing page", "Ana", "2026-10-22"),
      item("Revisar los contratos con legal", "Lucía", "2026-10-20", "high")]),

    ("c08-no-owner", "2026-10-06", False, ["no owner", "critical"],
     """- Someone needs to renew the SSL certificate before October 30 - critical.
- Hana will collect the conference talk proposals by Friday.""",
     [item("Renew the SSL certificate", None, "2026-10-30", "high"),
      item("Collect the conference talk proposals", "Hana", "2026-10-09")]),

    ("c09-date-changed-later", "2026-10-19", True, ["date changed later", "next Monday"],
     """- Ana will send the vendor comparison by Wednesday.
- Felix to migrate the analytics dashboards by October 30.
- Later in the meeting: Ana's comparison can wait - move it to next Monday.""",
     [item("Send the vendor comparison", "Ana", "2026-10-26"),
      item("Migrate the analytics dashboards", "Felix", "2026-10-30")]),

    ("c10-priority-cues", "2026-10-21", False, ["ASAP", "nice-to-have", "optional"],
     """- Raj will patch the auth bypass ASAP.
- Elena to refresh the docs screenshots, nice-to-have.
- Marco will prepare the board deck by Oct 28.
- Sven will explore a dark mode prototype if there's time - optional.""",
     [item("Patch the auth bypass", "Raj", None, "high"),
      item("Refresh the docs screenshots", "Elena", None, "low"),
      item("Prepare the board deck", "Marco", "2026-10-28"),
      item("Explore a dark mode prototype", "Sven", None, "low")]),

    ("c11-date-formats", "2026-10-02", False, ["the 3rd of November", "20/10", "Oct 14th"],
     """- Nadia will finalize the hiring plan by the 3rd of November.
- Owen to ship the beta to pilot customers on 20/10.
- Grace will present the roadmap at the all-hands on Oct 14th.""",
     [item("Finalize the hiring plan", "Nadia", "2026-11-03"),
      item("Ship the beta to pilot customers", "Owen", "2026-10-20"),
      item("Present the roadmap at the all-hands", "Grace", "2026-10-14")]),

    ("c12-owner-changed-later", "2026-10-13", True, ["owner changed later", "weekday"],
     """- Ben will write the integration tests for the payments API by Friday.
- Chloe to set up the alerting dashboards by October 23.
- Correction: Leila takes over the integration tests from Ben; same deadline.""",
     [item("Write the integration tests for the payments API", "Leila", "2026-10-16"),
      item("Set up the alerting dashboards", "Chloe", "2026-10-23")]),
]



# Cases whose input or reference changed during development; disclosed in every report.
REVISIONS = {
    "c10-priority-cues": (
        "reworded on 2026-10-10 during development, after all three models omitted the item: "
        "\"Optional: Sven can explore a dark mode prototype if there's time\" became \"Sven will "
        "explore a dark mode prototype if there's time - optional\", because the original was "
        "arguably not a commitment, so the reference answer was ambiguous"),
}


def build() -> list[dict]:
    rows = []
    for cid, meeting, hard, tags, notes, items in CASES:
        rows.append({
            "id": cid, "hard": hard, "tags": tags, "meeting_date": meeting, "notes": notes,
            "user_message": build_user_message(notes, date.fromisoformat(meeting)),
            "reference": {"action_items": items}, "source": "hand-written",
            **({"revision_note": REVISIONS[cid]} if cid in REVISIONS else {}),
        })
    return rows


def validate(rows: list[dict]) -> None:
    """Each reference must pass every verifier and score as correct against itself."""
    from notbydefault.fit.reference import score_case
    from notbydefault.fit.verifiers import load
    verifiers = load(HERE / "action_item_verifiers.py")
    for r in rows:
        answer = json.dumps(r["reference"], ensure_ascii=False)
        for c in verifiers.checks:
            ok, why = c.run(r["user_message"], answer, r["reference"])
            assert ok, f"{r['id']}: reference fails {c.name}: {why}"
        assert score_case(answer, r["reference"])["correct"], r["id"]


if __name__ == "__main__":
    rows = build()
    validate(rows)
    out = HERE / "action_item_cases.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} cases ({sum(r['hard'] for r in rows)} hard) -> {out}")
    if len(sys.argv) > 1:
        nbd = HERE.parent / ".notbydefault"
        for src, dst in ((out, nbd / "samples" / f"{sys.argv[1]}.jsonl"),
                         (HERE / "action_item_verifiers.py", nbd / "verifiers" / f"{sys.argv[1]}.py")):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            print(f"copied to {dst}")
