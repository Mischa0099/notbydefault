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
    fn: Callable[..., tuple[bool, str]] = field(repr=False)
    needs_reference: bool = False   # evaluation-only: compares with a reference answer
    retryable: bool = True          # may its failure reason be fed back as a retry hint?

    def run(self, input_text: str, output: str, reference: dict | None = None) -> tuple[bool, str]:
        try:
            if self.needs_reference:
                return self.fn(input_text, output, reference)
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


# --- structured (JSON) outputs ------------------------------------------------

def _load_strict(output: str):
    """Parse exactly like the app does (json.loads, no code fences)."""
    return json.loads(output.strip())


def _validate(value, schema: dict, where: str = "$") -> str | None:
    """Tiny JSON-schema subset: type (incl. lists like ["string","null"]), enum, pattern,
    required, properties, additionalProperties: false, items."""
    types = schema.get("type")
    if types:
        allowed = types if isinstance(types, list) else [types]
        py = {"object": dict, "array": list, "string": str, "null": type(None),
              "integer": int, "number": (int, float), "boolean": bool}
        if not any(isinstance(value, py[t]) and not (t in ("integer", "number")
                   and isinstance(value, bool)) for t in allowed):
            return f"{where} should be {' or '.join(allowed)}, got {type(value).__name__}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{where} must be one of {schema['enum']}, got {value!r}"
    if isinstance(value, str) and "pattern" in schema and not re.fullmatch(schema["pattern"], value):
        return f"{where} {value!r} does not match {schema['pattern']}"
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                return f"{where} is missing {key!r}"
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(props))
            if extra:
                return f"{where} has unexpected keys {extra}"
        for key, sub in props.items():
            if key in value:
                err = _validate(value[key], sub, f"{where}.{key}")
                if err:
                    return err
    if isinstance(value, list) and "items" in schema:
        for k, item in enumerate(value):
            err = _validate(item, schema["items"], f"{where}[{k}]")
            if err:
                return err
    return None


def _values(data, path: str) -> list:
    """Values at a path like "action_items[].owner"."""
    current = [data]
    for part in path.split("."):
        many = part.endswith("[]")
        key = part[:-2] if many else part
        nxt = []
        for c in current:
            v = c.get(key) if isinstance(c, dict) else None
            if many and isinstance(v, list):
                nxt.extend(v)
            elif not many:
                nxt.append(v)
        current = nxt
    return current


def json_matches_schema(schema: dict, why: str = "") -> Check:
    """Output parses as JSON (exactly like the app does) and matches the schema."""
    def fn(i, o):
        try:
            data = _load_strict(o)
        except (json.JSONDecodeError, AttributeError) as e:
            return False, f"not valid JSON ({getattr(e, 'msg', e)}); reply with the JSON object only"
        err = _validate(data, schema)
        return err is None, err or ""
    return Check("json_matches_schema", {"schema": schema}, why, fn)


def owners_in_input(path: str = "action_items[].owner", why: str = "") -> Check:
    """Every named owner must appear in the input (no invented people)."""
    def fn(i, o):
        try:
            values = _values(_load_strict(o), path)
        except (json.JSONDecodeError, AttributeError):
            return False, "not valid JSON, owners can't be checked"
        low = i.casefold()
        invented = []
        for v in values:
            for name in str(v).split(",") if v else []:
                name = name.strip()
                if name and not re.search(rf"(?<!\w){re.escape(name.casefold())}(?!\w)", low):
                    invented.append(name)
        return not invented, f"owner(s) not in the notes: {invented}"
    return Check("owners_in_input", {"path": path}, why, fn)


def dates_in_input(path: str = "action_items[].due_date", why: str = "") -> Check:
    """Every date must be one the input mentions (resolved like the prompt's date rules)."""
    from notbydefault.fit.dates import mentioned_dates

    def fn(i, o):
        try:
            values = [v for v in _values(_load_strict(o), path) if v]
        except (json.JSONDecodeError, AttributeError):
            return False, "not valid JSON, dates can't be checked"
        from notbydefault.fit.dates import MEETING_RX, meeting_date
        body = MEETING_RX.sub("", i)   # the header anchors dates; it is not a due date itself
        allowed = {d.isoformat() for d in mentioned_dates(body, meeting_date(i))}
        unsupported = [v for v in values if str(v) not in allowed]
        return not unsupported, (f"due date(s) {unsupported} are not supported by any date in the "
                                 "notes (resolve relative dates against the meeting date)")
    return Check("dates_in_input", {"path": path}, why, fn)


def dates_normalized(path: str = "action_items[].due_date", why: str = "") -> Check:
    """Dates must be null or a real calendar date written as YYYY-MM-DD."""
    from datetime import date as _date

    def fn(i, o):
        try:
            values = _values(_load_strict(o), path)
        except (json.JSONDecodeError, AttributeError):
            return False, "not valid JSON, dates can't be checked"
        bad = []
        for v in values:
            if v is None:
                continue
            try:
                ok = bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(v))) and bool(_date.fromisoformat(str(v)))
            except ValueError:
                ok = False
            if not ok:
                bad.append(v)
        return not bad, f"due date(s) not normalized to YYYY-MM-DD: {bad}"
    return Check("dates_normalized", {"path": path}, why, fn)


# A date reads as a deadline after one of these words (English and Spanish) ...
DUE_CUE = re.compile(r"\b(due|by|before|until|till|deadline|later than|target(?:ed)?|"
                     r"antes|hasta|para)\b[^.;!?]{0,20}$", re.IGNORECASE)
# ... or after "on"/"el" when it comes after the task ("ship the beta on 20/10") ...
POSITIONAL_CUE = re.compile(r"\b(on|el)\s+$", re.IGNORECASE)
# ... or when the phrase is a deadline by itself.
INHERENT_DUE = re.compile(r"^(today|tomorrow|day after tomorrow|hoy|ma[nñ]ana|pasado ma[nñ]ana|"
                          r"end of the (week|month)|(in|en) \d{1,2} (days|d[ií]as))$", re.IGNORECASE)


def _stems(text: str) -> set[str]:
    return {w.lower()[:5] for w in _words(text)
            if len(w) >= 4 and w.lower() not in STOPWORDS and not w.isdigit()}


def _note_units(text: str) -> list[str]:
    """The notes as sentences (one per line, then split on . ! ?), meeting header removed."""
    from notbydefault.fit.dates import MEETING_RX
    units = []
    for line in text.splitlines():
        if MEETING_RX.search(line) or not line.strip():
            continue
        line = re.sub(r"^\s*[-*•]\s*", "", line)
        units += [u.strip() for u in re.split(r"(?<=[.!?])\s+", line) if u.strip()]
    return units


def _stated_due(unit: str, task_stems: set[str], base) -> list[tuple[str, str]]:
    """(phrase, ISO date) for each date the sentence states as a deadline."""
    from notbydefault.fit.dates import mentioned_dates
    low = unit.lower()
    first_task = min((m.start() for m in re.finditer(r"[\w'-]+", low)
                      if m.group(0)[:5] in task_stems), default=len(low))
    out = []
    for day, phrases in mentioned_dates(unit, base).items():
        for phrase in phrases:
            at = low.find(phrase.lower())
            if at < 0:
                continue
            before = low[max(0, at - 40):at]
            if (INHERENT_DUE.match(phrase.strip()) or DUE_CUE.search(before)
                    or (POSITIONAL_CUE.search(before) and at > first_task)):
                out.append((phrase, day.isoformat()))
    return out


def due_dates_present(path: str = "action_items", why: str = "") -> Check:
    """No item without a due date when the notes clearly state one for it (grounded: the
    sentence that best matches the item's task words states a date as a deadline)."""
    from notbydefault.fit.dates import meeting_date

    def fn(i, o):
        try:
            items = _values(_load_strict(o), f"{path}[]")
        except (json.JSONDecodeError, AttributeError):
            return False, "not valid JSON, due dates can't be checked"
        base = meeting_date(i)
        units = [(u, _stems(u)) for u in _note_units(i)]
        missing = []
        for item in items:
            if not isinstance(item, dict) or item.get("due_date"):
                continue
            task = _stems(str(item.get("task") or ""))
            owners = [n.strip().casefold() for n in str(item.get("owner") or "").split(",")
                      if n.strip()]

            def score(u):
                text, stems = u
                bonus = 0.5 if any(re.search(rf"(?<!\w){re.escape(n)}(?!\w)", text.casefold())
                                   for n in owners) else 0
                return len(task & stems) + bonus if task & stems else 0
            scored = [(score(u), u) for u in units]
            best = max((s for s, _ in scored), default=0)
            if best <= 0:
                continue
            stated = [_stated_due(text, task, base) for s, (text, _) in scored if s == best]
            if stated and all(stated):     # every best-matching sentence states a deadline
                phrase, day = stated[0][0]
                missing.append(f"{item.get('task')!r}: the notes say {phrase!r} ({day})")
        return not missing, ("item(s) with no due date although the notes state one: "
                             + "; ".join(missing))
    return Check("due_dates_present", {"path": path}, why, fn)


def no_missing_items(why: str = "") -> Check:
    """Evaluation only: every reference action item is present."""
    from notbydefault.fit.reference import match, parse_output

    def fn(i, o, ref):
        if not ref:
            return True, "no reference"
        out = parse_output(o)
        if out is None:
            return False, "not valid JSON"
        _, missing, _ = match(ref.get("action_items", []), [x for x in out if isinstance(x, dict)])
        return not missing, f"missing items: {[ref['action_items'][k]['task'] for k in missing]}"
    return Check("no_missing_items", {}, why, fn, needs_reference=True, retryable=False)


def no_extra_items(why: str = "") -> Check:
    """Evaluation only: nothing beyond the reference (e.g. a task cancelled later)."""
    from notbydefault.fit.reference import match, parse_output

    def fn(i, o, ref):
        if not ref:
            return True, "no reference"
        out = parse_output(o)
        if out is None:
            return False, "not valid JSON"
        out = [x for x in out if isinstance(x, dict)]
        _, _, extra = match(ref.get("action_items", []), out)
        return not extra, f"items not in the reference: {[out[k].get('task') for k in extra]}"
    return Check("no_extra_items", {}, why, fn, needs_reference=True, retryable=False)


def llm_rubric(criterion: str, why: str = "") -> RubricCheck:
    return RubricCheck(criterion, why)


FACTORIES = {f.__name__: f for f in [
    not_empty, max_words, min_words, max_chars, max_sentences, min_sentences, forbid_regex,
    require_regex, no_preamble, no_markdown, covers_inputs, json_object, json_matches_schema,
    owners_in_input, dates_in_input, dates_normalized, due_dates_present, no_missing_items,
    no_extra_items]}

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
        elif ann in ("dict", dict):
            if not isinstance(value, dict):
                raise ValueError(f"{name}: {pname} must be an object")
        clean[pname] = value
    return clean
