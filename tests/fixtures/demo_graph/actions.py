"""Action-item extraction from meeting notes (POST /action-items)."""
import json
import os
from datetime import date

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

SYSTEM_PROMPT = """You extract action items from meeting notes.
Return only a JSON object, no other text:
{"action_items": [{"task": string, "owner": string or null, "due_date": "YYYY-MM-DD" or null, "priority": "high" | "medium" | "low"}]}

Rules:
- Include only tasks someone committed to do. Skip discussion, ideas and open questions.
- If a task is cancelled later in the notes, leave it out. If its owner or date changes later, use the final version.
- owner: the person's name exactly as written in the notes. Several owners: comma-separated, in the order mentioned. null if nobody is named.
- due_date: resolve relative dates against the meeting date. "today" = the meeting date; "tomorrow" = the next day; "<weekday>" or "this <weekday>" = the first such day after the meeting date; "next <weekday>" = that weekday in the week after the meeting's week (weeks start on Monday); "end of the week" = the Friday of the meeting's week; "end of the month" = the last day of the meeting's month. A date without a year is in the meeting's year. null if no date is given.
- priority: "high" if marked urgent, ASAP, critical or blocking; "low" if marked nice-to-have, optional or "if there's time"; otherwise "medium".
- Notes may mix languages; keep each task in the language it was written in."""

_client = None


def build_user_message(notes: str, meeting_date: date) -> str:
    return (f"Meeting date: {meeting_date.isoformat()} ({meeting_date.strftime('%A')})\n\n"
            f"Notes:\n{notes.strip()}")


def extract_action_items(notes: str, meeting_date: date) -> list[dict]:
    global _client
    if _client is None:  # created on first use so the app starts without GROQ_API_KEY
        from openai import OpenAI
        _client = OpenAI(base_url=GROQ_BASE_URL, api_key=os.environ["GROQ_API_KEY"])
    response = _client.chat.completions.create(
        model="openai/gpt-oss-120b",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(notes, meeting_date)},
        ],
    )
    return json.loads(response.choices[0].message.content)["action_items"]
