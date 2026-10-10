"""Step 1: find the code that calls the chosen model and extract what it is for."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from notbydefault.graph import find_affected_code
from notbydefault.openrouter import parse_json

FULL_FILE_LINES = 250   # send the whole file when it's at most this long
WINDOW = 80             # otherwise this many lines either side of the literal


@dataclass
class CallSite:
    file: str
    line: int
    snippet: str
    task: str = ""
    system_prompt: str = ""
    user_message_format: str = ""
    input_description: str = ""
    rubric: list[str] = field(default_factory=list)

    @property
    def snippet_sha(self) -> str:
        return hashlib.sha256(self.snippet.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return asdict(self) | {"snippet_sha": self.snippet_sha}


def locate(project: Path, decision: dict) -> CallSite:
    """Find the line using the model literal in the files the decision affects."""
    literal = decision["label"]
    for rel in find_affected_code(decision):
        path = (project / rel) if not Path(rel).is_absolute() else Path(rel)
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        hits = [i for i, l in enumerate(lines) if literal in l]
        if not hits:
            continue
        i = hits[0]
        if len(lines) <= FULL_FILE_LINES:
            start, end = 0, len(lines)
        else:
            start, end = max(0, i - WINDOW), min(len(lines), i + WINDOW + 1)
        snippet = "\n".join(f"{n + 1:4}| {lines[n]}" for n in range(start, end))
        return CallSite(file=rel, line=i + 1, snippet=snippet)
    raise LookupError(f"no file affected by decision #{decision['id']} still contains "
                      f"{literal!r} (looked in: {find_affected_code(decision) or 'nothing'})")


EXTRACT_SYSTEM = """You analyse application code that calls an LLM. Answer with one JSON object only."""

EXTRACT_PROMPT = """The file `{file}` calls the model "{model}" at line {line}:

```
{snippet}
```

Describe that LLM call. Return JSON with exactly these keys:
- "task": one sentence, what the call is for from the end user's point of view
- "system_prompt": the exact system prompt string the code sends ("" if none)
- "user_message_format": how the user message is built, with a short concrete example
- "input_description": what real inputs to this feature look like (content, typical length, variety)
- "rubric": 3 to 5 short criteria a judge should use to score an output for this task"""


def cache_path(project: Path, decision_id: int) -> Path:
    return project / ".notbydefault" / "callsites" / f"{decision_id}.json"


def load_cached(project: Path, decision_id: int, site: CallSite) -> CallSite | None:
    path = cache_path(project, decision_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("snippet_sha") != site.snippet_sha:
        return None  # code changed since the last extraction
    data.pop("snippet_sha", None)
    return CallSite(**data)


def extract(chat, project: Path, decision: dict, site: CallSite) -> CallSite:
    """Fill task/system prompt/format/rubric using `chat(messages) -> ChatResult`."""
    cached = load_cached(project, decision["id"], site)
    if cached:
        return cached
    reply = chat([
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": EXTRACT_PROMPT.format(
            file=site.file, model=decision["label"], line=site.line, snippet=site.snippet)},
    ])
    data = parse_json(reply.text)
    site.task = str(data.get("task", "")).strip()
    site.system_prompt = str(data.get("system_prompt", ""))
    site.user_message_format = str(data.get("user_message_format", ""))
    site.input_description = str(data.get("input_description", ""))
    site.rubric = [str(r) for r in data.get("rubric") or []][:5]
    if not site.task or not site.rubric:
        raise ValueError(f"call-site extraction returned no task/rubric: {reply.text[:300]}")
    path = cache_path(project, decision["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(site.to_dict(), indent=2), encoding="utf-8")
    return site


def build_messages(site: CallSite, user_message: str) -> list[dict]:
    messages = []
    if site.system_prompt:
        messages.append({"role": "system", "content": site.system_prompt})
    messages.append({"role": "user", "content": user_message})
    return messages
