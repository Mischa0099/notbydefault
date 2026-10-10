"""Step 2: test inputs, reused from .notbydefault/samples/<decision_id>.jsonl when present."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from notbydefault.fit.callsite import CallSite
from notbydefault.llm import parse_json

GENERATE_SYSTEM = "You write realistic test data for AI features. Answer with one JSON object only."

GENERATE_PROMPT = """An application feature does this: {task}

How its user message is built: {user_message_format}
What real inputs look like: {input_description}

Write {n} realistic, varied user messages exactly as the code would send them (already
formatted, no placeholders). Vary length, topic, tone and quality; include a few awkward
but plausible cases (very short, messy, mixed languages, contradictory). Return:
{{"inputs": ["<user message 1>", "<user message 2>", ...]}}"""


def samples_path(project: Path, decision_id: int) -> Path:
    return project / ".notbydefault" / "samples" / f"{decision_id}.jsonl"


def load(project: Path, decision_id: int) -> list[dict] | None:
    path = samples_path(project, decision_id)
    if not path.exists():
        return None
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            row.setdefault("source", "provided")
            rows.append(row)
    return rows or None


def generate(chat, project: Path, decision_id: int, site: CallSite, n: int = 20,
             generator: str = "") -> list[dict]:
    reply = chat([
        {"role": "system", "content": GENERATE_SYSTEM},
        {"role": "user", "content": GENERATE_PROMPT.format(
            task=site.task, user_message_format=site.user_message_format,
            input_description=site.input_description, n=n)},
    ])
    data = parse_json(reply.text)
    inputs = data.get("inputs") if isinstance(data, dict) else data
    inputs = [str(i) for i in (inputs or []) if str(i).strip()][:n]
    if len(inputs) < max(5, n // 2):
        raise ValueError(f"sample generation returned only {len(inputs)} inputs")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = [{"user_message": text, "source": "synthetic", "generated_by": generator,
             "generated_at": now} for text in inputs]
    path = samples_path(project, decision_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")
    return rows


def source_label(rows: list[dict]) -> str:
    sources = sorted({r.get("source", "provided") for r in rows})
    return "/".join(sources)


EXTEND_PROMPT = """An application feature does this: {task}

How its user message is built: {user_message_format}
What real inputs look like: {input_description}

These test inputs already exist:
{existing}

Write {n} NEW realistic user messages, formatted exactly as the code would send them, that
differ from all of the above (other topics, lengths and awkward cases). Return:
{{"inputs": ["<user message 1>", ...]}}"""


def extend(chat, project: Path, decision_id: int, site: CallSite, rows: list[dict], n: int,
           reason: str, generator: str = "") -> list[dict]:
    """Append n new synthetic inputs (different from `rows`) to the samples file."""
    existing = "\n".join(f"- {json.dumps(r['user_message'], ensure_ascii=False)[:300]}"
                         for r in rows)
    reply = chat([
        {"role": "system", "content": GENERATE_SYSTEM},
        {"role": "user", "content": EXTEND_PROMPT.format(
            task=site.task, user_message_format=site.user_message_format,
            input_description=site.input_description, existing=existing, n=n)},
    ])
    data = parse_json(reply.text)
    inputs = data.get("inputs") if isinstance(data, dict) else data
    seen = {r["user_message"] for r in rows}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new = [{"user_message": str(t), "source": "synthetic", "generated_by": generator,
            "generated_at": now, "expansion_reason": reason}
           for t in (inputs or []) if str(t).strip() and str(t) not in seen][:n]
    if new:
        with open(samples_path(project, decision_id), "a", encoding="utf-8") as f:
            for r in new:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows + new
