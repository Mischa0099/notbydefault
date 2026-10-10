"""Deterministic date resolution for verifiers: which calendar dates does a text mention?

The rules mirror the action-items prompt: "today"; "tomorrow"; "<weekday>"/"this <weekday>" =
first such day after the meeting date; "next <weekday>" = that weekday in the week after the
meeting's week (weeks start on Monday); "end of the week" = Friday of the meeting's week;
"end of the month" = last day of the meeting's month; dates without a year are in the meeting's
year. English and Spanish phrasings are recognised.
"""
from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5,
    "sunday": 6, "mon": 0, "tue": 1, "tues": 1, "wed": 2, "thu": 3, "thur": 3, "thurs": 3,
    "fri": 4, "sat": 5, "sun": 6,
    "lunes": 0, "martes": 1, "miércoles": 2, "miercoles": 2, "jueves": 3, "viernes": 4,
    "sábado": 5, "sabado": 5, "domingo": 6,
}
MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7,
    "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11,
    "diciembre": 12,
}
_WD = "|".join(sorted(WEEKDAYS, key=len, reverse=True))
_MO = "|".join(sorted(MONTHS, key=len, reverse=True))
MEETING_RX = re.compile(r"Meeting date:\s*(\d{4}-\d{2}-\d{2})")


def meeting_date(text: str) -> date | None:
    m = MEETING_RX.search(text)
    try:
        return date.fromisoformat(m.group(1)) if m else None
    except ValueError:
        return None


def next_occurrence(base: date, weekday: int) -> date:
    """First `weekday` strictly after `base`."""
    return base + timedelta(days=(weekday - base.weekday()) % 7 or 7)


def in_next_week(base: date, weekday: int) -> date:
    """`weekday` in the Monday-start week after the one containing `base`."""
    return base - timedelta(days=base.weekday()) + timedelta(days=7 + weekday)


def end_of_week(base: date) -> date:
    return base - timedelta(days=base.weekday()) + timedelta(days=4)


def end_of_month(base: date) -> date:
    return base.replace(day=calendar.monthrange(base.year, base.month)[1])


def _safe(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def mentioned_dates(text: str, base: date | None = None) -> dict[date, list[str]]:
    """Every date the text mentions -> the phrases that mention it."""
    base = base or meeting_date(text)
    found: dict[date, list[str]] = {}

    def add(d: date | None, phrase: str) -> None:
        if d:
            found.setdefault(d, []).append(phrase.strip())

    low = text.lower()
    for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", low):
        add(_safe(int(m[1]), int(m[2]), int(m[3])), m[0])
    if base is None:
        return found
    year = base.year
    # "October 20", "Oct 20th", "octubre 20"
    for m in re.finditer(rf"\b({_MO})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", low):
        add(_safe(year, MONTHS[m[1]], int(m[2])), m[0])
    # "20 October", "the 3rd of November", "20 de octubre"
    for m in re.finditer(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+|de\s+)?({_MO})\b", low):
        add(_safe(year, MONTHS[m[2]], int(m[1])), m[0])
    # "20/10" or "10/20": both readings when ambiguous
    for m in re.finditer(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", low):
        y = int(m[3]) + (2000 if m[3] and len(m[3]) == 2 else 0) if m[3] else year
        add(_safe(y, int(m[2]), int(m[1])), m[0])
        add(_safe(y, int(m[1]), int(m[2])), m[0])
    for word, delta in (("today", 0), ("hoy", 0), ("tomorrow", 1), ("mañana", 1),
                        ("manana", 1), ("day after tomorrow", 2), ("pasado mañana", 2)):
        if re.search(rf"\b{word}\b", low):
            add(base + timedelta(days=delta), word)
    # "next friday", "el próximo viernes", "el viernes que viene"
    for m in re.finditer(rf"\b(?:next|(?:el\s+)?pr[oó]xim[oa])\s+({_WD})\b", low):
        add(in_next_week(base, WEEKDAYS[m[1]]), m[0])
    for m in re.finditer(rf"\b(?:el\s+)?({_WD})\s+que\s+viene\b", low):
        add(in_next_week(base, WEEKDAYS[m[1]]), m[0])
    # plain / "this" weekday (skip the ones preceded by next/próximo, handled above)
    for m in re.finditer(rf"\b({_WD})\b", low):
        before = low[max(0, m.start() - 10):m.start()]
        after = low[m.end():m.end() + 10]
        if re.search(r"(next|pr[oó]xim[oa])\s+$", before) or after.startswith(" que viene"):
            continue
        add(next_occurrence(base, WEEKDAYS[m[1]]), m[0])
    if re.search(r"\bend of (?:the |this )?week\b|\bfinal de (?:la )?semana\b", low):
        add(end_of_week(base), "end of the week")
    if re.search(r"\bend of (?:the |this )?month\b|\bfin(?:al)? de(?:l)? mes\b", low):
        add(end_of_month(base), "end of the month")
    for m in re.finditer(r"\b(?:in|en)\s+(\d{1,2})\s+(?:days|d[ií]as)\b", low):
        add(base + timedelta(days=int(m[1])), m[0])
    return found
