"""Vetted building blocks for decision-specific verifiers.

Generated verifier files (.notbydefault/verifiers/<id>.py) only call these factories, so they
stay deterministic and safe to load. Each check returns (passed, reason).
"""
from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass, field
from typing import Callable

STOPWORDS = set("""about above after again against also among because been before being
below between both could does doing down during each from further have having here into
itself just more most other over same should some such than that their theirs them then
there these they this those through under until very were what when where which while
with would your yours note notes need needs make made like will""".split())


@dataclass
class Check:
    name: str
    args: dict
    why: str
    fn: Callable[[str, str], tuple[bool, str]] = field(repr=False)

    def run(self, input_text: str, output: str) -> tuple[bool, str]:
        try:
            return self.fn(input_text, output)
        except Exception as e:  # a broken check must not crash the run
            return False, f"{self.name} raised {e!r}"


@dataclass
class RubricCheck:
    criterion: str
    why: str = ""


def _words(text: str) -> list[str]:
    return re.findall(r"[\w'-]+", text)


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


# --- factories (the only names a verifier file may use) ------------------------

def not_empty(why: str = "") -> Check:
    return Check("not_empty", {}, why,
                 lambda i, o: (bool(o.strip()), "output is empty"))


def max_words(n: int, why: str = "") -> Check:
    def fn(i, o):
        k = len(_words(o))
        return k <= n, f"{k} words, limit is {n}"
    return Check("max_words", {"n": n}, why, fn)


def min_words(n: int, why: str = "") -> Check:
    def fn(i, o):
        k = len(_words(o))
        return k >= n, f"{k} words, minimum is {n}"
    return Check("min_words", {"n": n}, why, fn)


def max_chars(n: int, why: str = "") -> Check:
    return Check("max_chars", {"n": n}, why,
                 lambda i, o: (len(o) <= n, f"{len(o)} characters, limit is {n}"))


def max_sentences(n: int, why: str = "") -> Check:
    def fn(i, o):
        k = len(_sentences(o))
        return k <= n, f"{k} sentences, limit is {n}"
    return Check("max_sentences", {"n": n}, why, fn)


def min_sentences(n: int, why: str = "") -> Check:
    def fn(i, o):
        k = len(_sentences(o))
        return k >= n, f"{k} sentences, minimum is {n}"
    return Check("min_sentences", {"n": n}, why, fn)


def forbid_regex(pattern: str, why: str = "") -> Check:
    rx = re.compile(pattern, re.IGNORECASE | re.MULTILINE)

    def fn(i, o):
        m = rx.search(o)
        return m is None, f"contains forbidden text {m.group(0)!r}" if m else ""
    return Check("forbid_regex", {"pattern": pattern}, why, fn)


def require_regex(pattern: str, why: str = "") -> Check:
    rx = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    return Check("require_regex", {"pattern": pattern}, why,
                 lambda i, o: (rx.search(o) is not None, f"does not match {pattern!r}"))


PREAMBLE = re.compile(r"^\s*(sure|certainly|of course|here is|here's|here are|as an ai|"
                      r"i'd be happy|absolutely)\b", re.IGNORECASE)


def no_preamble(why: str = "") -> Check:
    return Check("no_preamble", {}, why,
                 lambda i, o: (not PREAMBLE.search(o), "starts with a chatty preamble"))


def no_markdown(why: str = "") -> Check:
    rx = re.compile(r"^\s{0,3}(#{1,6}\s|[-*+]\s|\d+\.\s)|\*\*|__", re.MULTILINE)

    def fn(i, o):
        m = rx.search(o)
        return m is None, f"uses markdown formatting ({m.group(0).strip()!r})" if m else ""
    return Check("no_markdown", {}, why, fn)


def covers_inputs(min_fraction: float, item_pattern: str = r"^\s*[-*•]\s+(.+)$",
                  why: str = "") -> Check:
    """Fraction of input items (lines matching item_pattern) reflected in the output.

    An item counts as covered when one of its content words (4+ letters, not a stopword)
    appears in the output, comparing the first 5 letters so plurals/tenses still match.
    """
    rx = re.compile(item_pattern, re.MULTILINE)

    def stems(text: str) -> set[str]:
        return {w.lower()[:5] for w in _words(text)
                if len(w) >= 4 and w.lower() not in STOPWORDS and not w.isdigit()}

    def fn(i, o):
        items = [m.group(1) for m in rx.finditer(i)]
        if not items:
            return True, "no list items in input"
        out = stems(o)
        covered = sum(1 for it in items if stems(it) & out or not stems(it))
        frac = covered / len(items)
        return frac >= min_fraction, f"covers {covered}/{len(items)} input items ({frac:.0%}), " \
                                     f"needs {min_fraction:.0%}"
    return Check("covers_inputs", {"min_fraction": min_fraction, "item_pattern": item_pattern},
                 why, fn)


def json_object(required_keys: list = (), why: str = "") -> Check:
    def fn(i, o):
        try:
            data = json.loads(o)
        except json.JSONDecodeError as e:
            return False, f"not valid JSON ({e.msg})"
        if not isinstance(data, dict):
            return False, "JSON is not an object"
        missing = [k for k in required_keys if k not in data]
        return not missing, f"missing keys {missing}"
    return Check("json_object", {"required_keys": list(required_keys)}, why, fn)


def llm_rubric(criterion: str, why: str = "") -> RubricCheck:
    return RubricCheck(criterion, why)


FACTORIES = {f.__name__: f for f in [
    not_empty, max_words, min_words, max_chars, max_sentences, min_sentences, forbid_regex,
    require_regex, no_preamble, no_markdown, covers_inputs, json_object]}

__all__ = list(FACTORIES) + ["llm_rubric", "Check", "RubricCheck"]


def describe_library() -> str:
    lines = []
    for name, f in FACTORIES.items():
        params = [p for p in inspect.signature(f).parameters if p != "why"]
        doc = (inspect.getdoc(f) or "").split("\n")[0]
        lines.append(f"- {name}({', '.join(params)})" + (f": {doc}" if doc else ""))
    return "\n".join(lines)


def validate_args(name: str, args: dict) -> dict:
    """Coerce/validate LLM-proposed args against the factory signature."""
    if name not in FACTORIES:
        raise ValueError(f"unknown check {name!r}")
    sig = inspect.signature(FACTORIES[name])
    clean = {}
    for pname, param in sig.parameters.items():
        if pname == "why":
            continue
        if pname not in args:
            if param.default is inspect.Parameter.empty:
                raise ValueError(f"{name}: missing argument {pname!r}")
            continue
        value, ann = args[pname], param.annotation
        if ann in ("int", int):
            value = int(value)
        elif ann in ("float", float):
            value = float(value)
        elif ann in ("str", str):
            value = str(value)
            if "pattern" in pname:
                re.compile(value)
        elif ann in ("list", list):
            value = [str(v) for v in value]
        clean[pname] = value
    return clean
