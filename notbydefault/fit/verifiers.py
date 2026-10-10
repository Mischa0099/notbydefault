"""Decision-specific verifiers, stored as reviewable code in .notbydefault/verifiers/<id>.py."""
from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from notbydefault.fit.callsite import CallSite
from notbydefault.fit.checks import Check, RubricCheck, describe_library, validate_args
from notbydefault.openrouter import parse_json

GENERATE_SYSTEM = ("You design automatic output checks for an LLM feature. "
                   "Answer with one JSON object only.")

GENERATE_PROMPT = """Design verifiers for this LLM feature.

Project goal: {goal}
Requirements and constraints:
{context}
Code the decision affects: {files}

The feature (file {file}, line {line}): {task}
System prompt: {system_prompt}
User message format: {user_message_format}
Real inputs: {input_description}

```
{snippet}
```

Pick 3 to 5 DETERMINISTIC checks from this library only (use exact names and arguments):
{library}

Choose limits a good output for this task always meets (be generous enough that good answers
pass; checks must catch real failures such as rambling, wrong format, ignoring the input).
Then write ONE LLM rubric check: a single yes/no criterion a cheap model can judge from the
input and output alone (e.g. "contains no facts absent from the input").

Return:
{{"checks": [{{"check": "<name>", "args": {{...}}, "why": "<reason tied to the task/goal>"}}],
  "rubric_check": {{"criterion": "<yes/no criterion>", "why": "<reason>"}}}}"""

RETRY_PROMPT = """Your previous answer failed an automatic check: {reason}
Rewrite your answer so it passes. Keep following the original instructions and reply with the
corrected answer only."""

RUBRIC_SYSTEM = ("You check one output of an AI feature against one requirement. "
                 "Answer with one JSON object only.")
RUBRIC_PROMPT = """Requirement: {criterion}

Input given to the feature:
<<<
{input}
>>>

Output to check:
<<<
{output}
>>>

Return {{"pass": true or false, "reason": "<one sentence>"}}"""


def verifier_path(project: Path, decision_id: int) -> Path:
    return project / ".notbydefault" / "verifiers" / f"{decision_id}.py"


@dataclass
class Verifiers:
    checks: list[Check]
    rubric: RubricCheck | None
    path: Path
    sha: str

    def names(self) -> list[str]:
        return [c.name for c in self.checks] + (["llm_rubric"] if self.rubric else [])


def load(path: Path) -> Verifiers:
    spec = importlib.util.spec_from_file_location(f"nbd_verifiers_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    checks = list(getattr(module, "CHECKS", []))
    rubric = getattr(module, "RUBRIC_CHECK", None)
    if not all(isinstance(c, Check) for c in checks):
        raise TypeError(f"{path}: CHECKS must contain checks from notbydefault.fit.checks")
    return Verifiers(checks, rubric, path,
                     hashlib.sha256(path.read_bytes()).hexdigest()[:16])


def render(decision: dict, site: CallSite, checks: list[tuple[str, dict, str]],
           rubric: tuple[str, str], generator: str) -> str:
    def call(name, args, why):
        parts = [f"{k}={v!r}" for k, v in args.items()] + [f"why={why!r}"]
        return f"    {name}({', '.join(parts)}),"

    date = datetime.now(timezone.utc).date().isoformat()
    return "\n".join([
        f'"""Verifiers for decision #{decision["id"]}: {decision["label"]} - {site.task}',
        "",
        f"Generated {date} by {generator} from the Decision Graph and {site.file}.",
        "Review and edit freely: CHECKS run on every output (deterministic, free);",
        "RUBRIC_CHECK is judged by the verifier model. `python -m notbydefault label "
        f"{decision['id']}`",
        "measures how often these agree with you.",
        '"""',
        "from notbydefault.fit.checks import *  # noqa: F401,F403",
        "",
        "CHECKS = [",
        *[call(n, a, w) for n, a, w in checks],
        "]",
        "",
        f"RUBRIC_CHECK = llm_rubric(criterion={rubric[0]!r}, why={rubric[1]!r})",
        "",
    ])


def generate(chat, project: Path, decision: dict, site: CallSite, graph_context: dict,
             generator: str) -> Verifiers:
    reply = chat([
        {"role": "system", "content": GENERATE_SYSTEM},
        {"role": "user", "content": GENERATE_PROMPT.format(
            goal=graph_context.get("goal") or "(not recorded)",
            context="\n".join(f"- {c}" for c in graph_context.get("constraints", [])) or "- none",
            files=", ".join(graph_context.get("files", [])) or site.file,
            file=site.file, line=site.line, task=site.task,
            system_prompt=site.system_prompt or "(none)",
            user_message_format=site.user_message_format,
            input_description=site.input_description, snippet=site.snippet,
            library=describe_library())},
    ])
    data = parse_json(reply.text)
    checks = []
    for item in data.get("checks") or []:
        try:
            name = str(item["check"])
            checks.append((name, validate_args(name, item.get("args") or {}),
                           str(item.get("why", ""))))
        except (KeyError, ValueError, TypeError):
            continue  # drop anything outside the vetted library
    checks = checks[:5]
    if len(checks) < 3:
        raise ValueError(f"verifier generation produced {len(checks)} usable checks: "
                         f"{reply.text[:300]}")
    rc = data.get("rubric_check") or {}
    rubric = (str(rc.get("criterion") or "The output contains no facts absent from the input."),
              str(rc.get("why", "")))
    path = verifier_path(project, decision["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(decision, site, checks, rubric, generator), encoding="utf-8")
    return load(path)


@dataclass
class VerifyResult:
    passed: bool
    failed: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    cost: float = 0.0
    latency_s: float = 0.0
    error: str | None = None

    @property
    def reason(self) -> str:
        return "; ".join(self.reasons)


def verify(caller, verifier_model: str, verifiers: Verifiers, input_text: str, output: str,
           attrs: dict | None = None) -> VerifyResult:
    """Deterministic checks first; the LLM rubric only runs if they all pass."""
    result = VerifyResult(passed=True)
    for check in verifiers.checks:
        ok, why = check.run(input_text, output)
        if not ok:
            result.passed = False
            result.failed.append(check.name)
            result.reasons.append(f"{check.name}: {why}")
    if not result.passed or verifiers.rubric is None:
        return result
    from notbydefault.fit.evaluate import BudgetExceeded  # local: avoid import cycle
    from notbydefault.openrouter import LLMError
    try:
        reply = caller.chat(verifier_model, [
            {"role": "system", "content": RUBRIC_SYSTEM},
            {"role": "user", "content": RUBRIC_PROMPT.format(
                criterion=verifiers.rubric.criterion, input=input_text, output=output)},
        ], span="notbydefault.verifier_call", role="verifier", max_tokens=200,
            attrs=attrs or {})
        result.cost, result.latency_s = reply.cost, reply.latency_s
        verdict = parse_json(reply.text)
        if not bool(verdict.get("pass")):
            result.passed = False
            result.failed.append("llm_rubric")
            result.reasons.append(f"llm_rubric: {verdict.get('reason', 'failed')}")
    except BudgetExceeded:
        raise
    except (LLMError, AttributeError) as e:
        result.passed = False
        result.error = f"rubric check failed to run: {e}"
        result.failed.append("llm_rubric")
        result.reasons.append(result.error)
    return result
