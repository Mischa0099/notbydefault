"""Score an action-items output against a hand-written reference answer (deterministic).

A case is correct when the output is valid JSON, every reference item is found with the right
owner, due date and priority, and nothing else is listed (e.g. an item cancelled later).
"""
from __future__ import annotations

import json
import re

STOP = set("""the a an and or to of for in on at by with from into about this that these those
be is are was were will would should can could need needs please we our us you your they their
it its as up out""".split())
MATCH_THRESHOLD = 0.25


def parse_output(output: str) -> list[dict] | None:
    """Items from the output, parsed strictly the way the app does (json.loads, no fences)."""
    try:
        data = json.loads(output.strip())
    except (json.JSONDecodeError, AttributeError):
        return None
    items = data.get("action_items") if isinstance(data, dict) else None
    return items if isinstance(items, list) else None


def _stems(text: str) -> set[str]:
    words = re.findall(r"[\wáéíóúñü]+", str(text or "").lower())
    return {w[:5] for w in words if len(w) >= 3 and w not in STOP}


def _similarity(a: dict, b: dict) -> float:
    sa, sb = _stems(a.get("task")), _stems(b.get("task"))
    jac = len(sa & sb) / len(sa | sb) if sa | sb else 0.0
    return jac + (0.2 if owners(a) and owners(a) == owners(b) else 0.0)


def owners(item: dict) -> frozenset[str]:
    raw = item.get("owner")
    if not raw:
        return frozenset()
    return frozenset(p.strip().casefold() for p in re.split(r",|\band\b|\by\b", str(raw)) if p.strip())


def match(reference: list[dict], output: list[dict]) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy best-first pairing. Returns (pairs, missing reference idx, extra output idx)."""
    scored = sorted(((_similarity(r, o), i, j) for i, r in enumerate(reference)
                     for j, o in enumerate(output)), reverse=True)
    pairs, used_r, used_o = [], set(), set()
    for s, i, j in scored:
        if s < MATCH_THRESHOLD or i in used_r or j in used_o:
            continue
        pairs.append((i, j))
        used_r.add(i)
        used_o.add(j)
    missing = [i for i in range(len(reference)) if i not in used_r]
    extra = [j for j in range(len(output)) if j not in used_o]
    return sorted(pairs), missing, extra


def score_case(output: str, reference: dict) -> dict:
    ref_items = reference.get("action_items", [])
    out_items = parse_output(output)
    if out_items is None:
        return {"correct": False, "item_f1": 0.0, "valid_json": False,
                "problems": ["output is not valid JSON with an action_items list"]}
    out_items = [o for o in out_items if isinstance(o, dict)]
    pairs, missing, extra = match(ref_items, out_items)
    problems = [f"missing: {ref_items[i]['task']}" for i in missing]
    problems += [f"extra: {out_items[j].get('task')}" for j in extra]
    fully_right = 0
    for i, j in pairs:
        r, o = ref_items[i], out_items[j]
        wrong = []
        if owners(r) != owners(o):
            wrong.append(f"owner {o.get('owner')!r} (expected {r.get('owner')!r})")
        if (r.get("due_date") or None) != (o.get("due_date") or None):
            wrong.append(f"due {o.get('due_date')!r} (expected {r.get('due_date')!r})")
        if str(r.get("priority")).lower() != str(o.get("priority")).lower():
            wrong.append(f"priority {o.get('priority')!r} (expected {r.get('priority')!r})")
        if wrong:
            problems.append(f"{r['task']}: " + ", ".join(wrong))
        else:
            fully_right += 1
    if not ref_items and not out_items:
        f1 = 1.0
    else:
        p = fully_right / len(out_items) if out_items else 0.0
        r_ = fully_right / len(ref_items) if ref_items else 0.0
        f1 = 2 * p * r_ / (p + r_) if p + r_ else 0.0
    return {"correct": not problems, "item_f1": round(f1, 4), "valid_json": True,
            "problems": problems}
