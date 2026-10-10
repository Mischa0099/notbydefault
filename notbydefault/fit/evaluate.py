"""Steps 4-5 and 10: run every input on every setup, judge blind, never exceed --max-cost.

A setup is a model alone, or the model + verifier with up to `max_retries` retries that feed
the verifier's failure reason back. The verified setup's first attempt is the model-alone call
(same prompt, fixed temperature), so it is made once and shared.
"""
from __future__ import annotations

import random
import string
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from notbydefault.fit.callsite import CallSite, build_messages
from notbydefault.fit.tracing import SPAN_KIND, Tracer, chat_attrs
from notbydefault.openrouter import ChatResult, LLMError, parse_json

CHARS_PER_TOKEN = 3.0  # conservative (real English is ~4) so reservations over-estimate
ALONE, VERIFIED = "alone", "verified"


class BudgetExceeded(RuntimeError):
    pass


class SpendGuard:
    """Reserve each call's worst-case cost up front; refuse calls that could break the cap."""

    def __init__(self, max_cost: float):
        self.max_cost = max_cost
        self.spent = 0.0
        self.reserved = 0.0
        self.stopped = False
        self._lock = threading.Lock()

    def reserve(self, amount: float) -> None:
        with self._lock:
            if self.stopped or self.spent + self.reserved + amount > self.max_cost:
                self.stopped = True
                raise BudgetExceeded(
                    f"next call could cost up to ${amount:.4f}; spent ${self.spent:.4f} of "
                    f"--max-cost ${self.max_cost:.2f}")
            self.reserved += amount

    def settle(self, reserved: float, actual: float) -> None:
        with self._lock:
            self.reserved -= reserved
            self.spent += actual


def estimate_tokens(messages: list[dict]) -> int:
    return int(sum(len(m["content"]) for m in messages) / CHARS_PER_TOKEN) + 20


def _safe(fn, result) -> dict:
    if fn is None:
        return {}
    try:
        return fn(result)
    except Exception:
        return {}


class Caller:
    """Guarded, traced chat calls."""

    def __init__(self, client, guard: SpendGuard, tracer: Tracer):
        self.client, self.guard, self.tracer = client, guard, tracer

    def chat(self, model: str, messages: list[dict], *, span: str, role: str,
             temperature: float = 0.0, max_tokens: int = 1024, json_mode: bool = False,
             attrs: dict | None = None, extra_after=None) -> ChatResult:
        price = self.client.pricing().price(model)
        worst = price.cost(estimate_tokens(messages), max_tokens)
        self.guard.reserve(worst)
        actual = 0.0
        try:
            result = self.tracer.llm(
                span,
                lambda: self.client.chat(model, messages, temperature=temperature,
                                         max_tokens=max_tokens, json_mode=json_mode),
                before={"llm.model_name": model, "notbydefault.role": role,
                        "llm.invocation_parameters": {"temperature": temperature,
                                                      "max_tokens": max_tokens},
                        "input.value": messages[-1]["content"], **(attrs or {})},
                after=lambda r: chat_attrs(r) | _safe(extra_after, r),
                kind=SPAN_KIND.get(role, "TOOL"),
            )
            actual = result.cost
            return result
        finally:
            self.guard.settle(worst, actual)

    def analyst(self, model: str, max_tokens: int = 4000):
        """chat(messages) -> ChatResult for extraction and generation steps."""
        def chat(messages):
            return self.chat(model, messages, span="notbydefault.analyst", role="analyst",
                             temperature=0.7, max_tokens=max_tokens)
        return chat


@dataclass
class Row:
    sample: int
    model: str
    setup: str                    # ALONE | VERIFIED
    output: str = ""
    passed: bool = False
    failed: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0        # what the end user waits, including retries + verification
    cost: float = 0.0             # what production pays, including retries + verification
    error: str | None = None
    score: float | None = None
    judge_reason: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return self.model, self.setup

    def to_dict(self) -> dict:
        return asdict(self)


def setup_name(model: str, setup: str, max_retries: int = 2) -> str:
    return model if setup == ALONE else f"{model} + verifier (<={max_retries} retries)"


def run_setups(caller: Caller, site: CallSite, samples: list[dict], models: list[str],
               verifiers, verifier_model: str, temperature: float, max_tokens: int,
               max_retries: int = 2, workers: int = 6) -> tuple[list[Row], list[dict]]:
    """Returns (rows: 2 per sample x model, failure-bank records)."""
    from notbydefault.fit.verifiers import RETRY_PROMPT, verify

    failures: list[dict] = []
    lock = threading.Lock()

    def one(job) -> list[Row]:
        i, model = job
        text = samples[i]["user_message"]
        alone, verified = Row(i, model, ALONE), Row(i, model, VERIFIED)
        messages = build_messages(site, text)
        try:
            for attempt in range(max_retries + 1):
                r = caller.chat(model, messages, span="notbydefault.model_call", role="candidate",
                                temperature=temperature, max_tokens=max_tokens,
                                attrs={"notbydefault.sample": i, "notbydefault.attempt": attempt})
                v = verify(caller, verifier_model, verifiers, text, r.text,
                           attrs={"notbydefault.sample": i, "notbydefault.candidate": model,
                                  "notbydefault.attempt": attempt})
                if not r.text.strip() and "not_empty" not in v.failed:
                    v.passed = False
                    v.failed.append("not_empty")
                    v.reasons.append(f"not_empty: empty output (finish_reason={r.finish_reason})")
                if not v.passed:
                    with lock:
                        failures.append({
                            "input": text, "output": r.text, "model": model, "attempt": attempt,
                            "failed": v.failed, "reasons": v.reasons,
                            "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                if attempt == 0:  # model alone = first attempt, verification is measurement only
                    alone.output, alone.passed = r.text, v.passed
                    alone.failed, alone.reasons, alone.attempts = v.failed, v.reasons, 1
                    alone.input_tokens, alone.output_tokens = r.input_tokens, r.output_tokens
                    alone.latency_s, alone.cost = r.latency_s, r.cost
                verified.attempts += 1
                verified.output, verified.passed = r.text, v.passed
                verified.failed, verified.reasons = v.failed, v.reasons
                verified.input_tokens += r.input_tokens
                verified.output_tokens += r.output_tokens
                verified.latency_s += r.latency_s + v.latency_s
                verified.cost += r.cost + v.cost
                if v.passed or attempt == max_retries:
                    break
                messages = messages + [
                    {"role": "assistant", "content": r.text},
                    {"role": "user", "content": RETRY_PROMPT.format(reason=v.reason)}]
        except BudgetExceeded as e:
            for row in (alone, verified):
                if not row.attempts or row is verified:
                    row.error = f"skipped: {e}"
        except LLMError as e:
            for row in (alone, verified):
                if not row.attempts or row is verified:
                    row.error = str(e)
        return [alone, verified]

    jobs = [(i, m) for i in range(len(samples)) for m in models]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [r for pair in pool.map(caller.tracer.bind(one), jobs) for r in pair]
    return rows, failures


JUDGE_SYSTEM = """You are a strict, impartial evaluator of an AI feature's outputs. You see one
input and several anonymous responses in random order. Score each response independently
from 1 (unusable) to 5 (excellent) against the rubric. Do not reward length or style for its
own sake, and do not guess which system wrote a response. Answer with one JSON object only."""

JUDGE_PROMPT = """Task of the feature: {task}

Rubric:
{rubric}

System prompt the feature uses:
<<<
{system_prompt}
>>>

Input:
<<<
{user_message}
>>>

{responses}

Return {{"scores": {{"A": {{"score": <1-5>, "reason": "<one sentence>"}}, ...}}}} with one
entry per response label."""


def judge(caller: Caller, judge_model: str, site: CallSite, samples: list[dict],
          rows: list[Row], seed: int = 0, workers: int = 6) -> list[dict]:
    """Score rows in place; identical outputs are judged once. One record per judge call."""
    by_sample: dict[int, dict[str, list[Row]]] = {}
    for r in rows:
        if not r.error:
            by_sample.setdefault(r.sample, {}).setdefault(r.output, []).append(r)

    def one(item) -> dict:
        i, groups = item
        outputs = sorted(groups)
        random.Random(seed * 100003 + i).shuffle(outputs)   # blind + randomized, reproducible
        labels = dict(zip(string.ascii_uppercase, outputs))
        responses = "\n\n".join(f"Response {k}:\n<<<\n{o}\n>>>" for k, o in labels.items())
        prompt = JUDGE_PROMPT.format(
            task=site.task, rubric="\n".join(f"- {c}" for c in site.rubric),
            system_prompt=site.system_prompt or "(none)",
            user_message=samples[i]["user_message"], responses=responses)
        who = {k: sorted({f"{r.model}/{r.setup}" for r in groups[o]}) for k, o in labels.items()}
        record = {"sample": i, "labels": who}

        def score_attrs(result) -> dict:  # scores as attributes on this judge call's span
            scores = parse_json(result.text).get("scores", {})
            return {f"notbydefault.score.{name}": (scores.get(k) or {}).get("score")
                    for k, names in who.items() for name in names}

        try:
            reply = caller.chat(judge_model, [{"role": "system", "content": JUDGE_SYSTEM},
                                              {"role": "user", "content": prompt}],
                                span="notbydefault.judge_call", role="judge",
                                max_tokens=150 + 120 * len(labels),
                                attrs={"notbydefault.sample": i}, extra_after=score_attrs)
            scores = parse_json(reply.text).get("scores", {})
            for k, o in labels.items():
                s = scores.get(k) or {}
                try:
                    score = float(s.get("score"))
                except (TypeError, ValueError):
                    continue
                if 1 <= score <= 5:
                    for r in groups[o]:
                        r.score, r.judge_reason = score, str(s.get("reason", ""))
            record.update(scores=scores, cost=reply.cost, input_tokens=reply.input_tokens,
                          output_tokens=reply.output_tokens, latency_s=reply.latency_s)
        except (BudgetExceeded, LLMError, AttributeError) as e:
            record["error"] = str(e)
        return record

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(caller.tracer.bind(one), sorted(by_sample.items())))
