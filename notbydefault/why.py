"""`notbydefault why "<question>"`: answer from what was recorded, with citations.

1. retrieve numbered facts, each with a source id:
   - Decision Graph: decisions matching the question, their recorded reason, decision record,
     evidence, rejected alternatives and requirements that made them stale / need review
   - Entire: `entire search` over checkpoints/sessions/commits (needs login + a mirrored repo);
     otherwise local checkpoints (`entire checkpoint list` / `explain`) and `git log`
   - Entire Graph: the code locations each matched decision affects
2. answer: the recorded facts themselves, each cited. With --llm a Groq model phrases them
   instead, citing their ids (an answer citing anything else is discarded); off by default
   because a fluent paraphrase can still invert a verdict while citing the right fact. No matching decision with
   a recorded reason -> "no recorded reason". History is never invented.
"""
from __future__ import annotations

import re
from pathlib import Path

STOP = set("""a an the is are was were be been why did do does we our us i you it its this that
these those to of for in on at by with from and or not no what which who how when where use
used using pick picked choose chose chosen decide decided decision instead over than""".split())
NO_REASON = "no recorded reason"


def words(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9][a-z0-9._/-]*", (text or "").lower())
    return [t.strip("._/-") for t in toks if t not in STOP and len(t.strip("._/-")) >= 2]


def _score(qwords: list[str], text: str) -> int:
    low = (text or "").lower()
    return sum(1 for w in qwords if re.search(rf"(?<![\w]){re.escape(w)}", low))


class Facts:
    def __init__(self):
        self.items: list[dict] = []

    def add(self, text: str, source: str, kind: str) -> str:
        for f in self.items:          # the same fact from two duplicate decisions: cite once
            if f["text"] == text and f["kind"] == kind:
                return f["id"]
        fid = f"F{len(self.items) + 1}"
        self.items.append({"id": fid, "text": text, "source": source, "kind": kind})
        return fid

    def ids(self) -> set[str]:
        return {f["id"] for f in self.items}


def match_decisions(graph, question: str) -> list[dict]:
    """Decisions the question is about (label words weigh most), best first."""
    from notbydefault.records import why_text
    q = words(question)
    scored = []
    for d in graph.nodes("decision"):
        label_hits = _score(q, d["label"]) + _score(q, d["label"].split("/")[-1])
        other = _score(q, f"{d['data'].get('category')} {d['data'].get('kind')} {why_text(d)}")
        alts = sum(_score(q, a["label"]) for a in graph.neighbors(d["id"], ("rejects",)))
        score = 3 * label_hits + other + 2 * alts
        if label_hits or alts:
            scored.append((score, d["id"], d))
    return [d for _s, _i, d in sorted(scored, key=lambda x: (-x[0], -x[1]))]


def graph_facts(graph, decisions: list[dict], facts: Facts) -> dict[int, bool]:
    """Facts per decision; returns {decision id: has a recorded reason}."""
    from notbydefault.affected import describe
    from notbydefault.graph import affected_entries
    reasons = {}
    for d in decisions:
        data = d["data"]
        src = f"decision #{d['id']}" + (f", {data['decision_record']}" if data.get("decision_record") else "")
        # only a verdict explains the choice; stale / needs-review notes say whether it still holds
        recorded = data.get("verdict_summary")
        reasons[d["id"]] = bool(recorded)
        status = data.get("status")
        if recorded:
            facts.add(f"Decision #{d['id']} `{d['label']}` ({status}): fit-test verdict: "
                      f"{data['verdict_summary']}", src, "reason")
        if status == "stale":
            facts.add(f"Status (not a reason it was chosen): decision #{d['id']} `{d['label']}` "
                      f"is STALE: {data.get('stale_reason')}",
                      src + (f", requirement #{data['stale_cause']}" if data.get("stale_cause") else ""),
                      "status")
        if status == "needs_review":
            facts.add(f"Status (not a reason it was chosen): decision #{d['id']} `{d['label']}` "
                      f"needs review: {data.get('review_reason')}",
                      src + (f", requirement #{data['review_cause']}" if data.get("review_cause") else ""),
                      "status")
        if not recorded:
            how = f" ({data['evidence']})" if data.get("evidence") else ""
            facts.add(f"Decision #{d['id']} `{d['label']}` was picked by a coding agent's "
                      f"{data.get('tool_name') or 'tool'} call{how}; {NO_REASON} beyond that",
                      src + (f", session {d['session_id']}" if d.get("session_id") else ""),
                      "no_reason")
        for a in graph.neighbors(d["id"], ("rejects",)):
            facts.add(f"Rejected alternative `{a['label']}`: {a['data'].get('reason') or NO_REASON}",
                      f"alternative #{a['id']}", "alternative")
        for e in graph.neighbors(d["id"], ("supported_by",)):
            link = e["data"].get("path") or e["data"].get("url")
            facts.add(f"Evidence: {e['label']}", f"evidence #{e['id']}" + (f", {link}" if link else ""),
                      "evidence")
        code = [e for e in affected_entries(d) if e.get("symbol")][:6] or affected_entries(d)[:3]
        if code:
            facts.add(f"Code using `{d['label']}`: " + "; ".join(f"{describe(e)} ({e.get('via')})"
                                                                for e in code),
                      f"{data.get('affected_source') or 'touched files'}: "
                      + ", ".join(f"{e['path']}:{e['line']}" if e.get("line") else e["path"]
                                  for e in code), "code")
    return reasons


def history_facts(project: Path, question: str, decisions: list[dict], facts: Facts,
                  entire, tracer=None) -> str:
    """Entire search hits, or local checkpoints + git log; returns how history was searched."""
    from notbydefault.briefing import load_cache
    from notbydefault.entire import recent_commits
    q = words(question) + [w for d in decisions[:2] for w in words(d["label"])]
    hits = entire.search(question, project, limit=5)
    if hits is not None:
        for h in hits:
            if h.get("id"):
                facts.add(f"Entire {h.get('type', 'result')}: {h.get('title') or h.get('snippet')}",
                          f"entire {h.get('type', 'result')} {h['id']}", "history")
        return "entire search"
    why = entire.last_error()
    if tracer is not None:
        tracer.event("search_fallback", "recovery",
                     f"`entire search` unavailable ({why}); searching local checkpoints and git log")
    cps = entire.checkpoint_list(project)
    if cps is None:
        cps = [{"checkpoint_id": c["id"], "message": c.get("message"), "session_ids": c.get("sessions")}
               for c in load_cache(project).get("checkpoints") or []]
    for c in cps:
        if c.get("checkpoint_id") and _score(q, c.get("message", "")):
            sess = (c.get("session_ids") or [c.get("session_id")])[0]
            facts.add(f"Checkpoint: {c.get('message')}",
                      f"checkpoint {c['checkpoint_id']}" + (f", session {sess}" if sess else ""),
                      "history")
    for c in recent_commits(project):
        if _score(q, c["subject"]) >= max(1, min(2, len(q))):
            facts.add(f"Commit: {c['subject']}", f"commit {c['sha']}", "history")
    return f"local checkpoints and git log (entire search unavailable: {why})"


ANSWER_SYSTEM = ("You answer a developer's question about why a technical decision was made, "
                 "using ONLY the numbered facts given. Never add history, reasons or numbers "
                 "that are not in the facts. Answer with one JSON object only.")
ANSWER_PROMPT = """Question: {question}

Facts (each has an id and a source):
{facts}

Write a short answer (at most 5 sentences). After every claim put the id(s) of the fact(s) it
comes from in square brackets, e.g. [F2]. Facts marked "Status" only say whether a decision
still holds; never present them as the reason it was chosen. If no fact gives the reason it was
chosen, say there is no recorded reason, then summarize what the records do say. Return {{"answer": "<text with [F#] citations>", "cited": ["F1", ...]}}"""


def llm_answer(caller, model: str, question: str, facts: Facts) -> str | None:
    """A phrased answer whose citations all exist, or None (then the facts are shown as-is)."""
    import json
    from notbydefault.llm import parse_json
    listing = "\n".join(f"{f['id']}: {f['text']} (source: {f['source']})" for f in facts.items)
    reply = caller.chat(model, [{"role": "system", "content": ANSWER_SYSTEM},
                                {"role": "user", "content": ANSWER_PROMPT.format(
                                    question=question, facts=listing[:12000])}],
                        span="notbydefault.why_answer", role="why", max_tokens=1500,
                        json_mode=True, focus_area="execution")
    data = parse_json(reply.text)
    text = str(data.get("answer") or "").strip()
    cited = set(re.findall(r"\bF\d+\b", text))
    if not text or not cited or not cited <= facts.ids():
        return None
    return text


def render(question: str, facts: Facts, answer: str, searched: str) -> list[str]:
    lines = [f"Q: {question}", "", answer, "", "Sources:"]
    lines += [f"  [{f['id']}] {f['text']}\n        ({f['source']})" for f in facts.items]
    lines += ["", f"History searched with: {searched}"]
    return lines


def plain_answer(facts: Facts, reasons: dict[int, bool]) -> str:
    """The deterministic answer: recorded reasons first, else say there is none."""
    picked = [f for f in facts.items if f["kind"] in ("reason", "no_reason", "status", "alternative")]
    if not any(reasons.values()):
        head = f"There is {NO_REASON} for this decision."
        return head + ("" if not picked else " What is recorded: " +
                       " ".join(f"{f['text']} [{f['id']}]." for f in picked[:4]))
    return " ".join(f"{f['text']} [{f['id']}]." for f in picked[:5])


def why(project: Path, question: str, client=None, entire=None, tracer=None, out=print,
        use_llm: bool = False) -> int:
    from notbydefault.entire import Entire
    from notbydefault.fit.tracing import make_tracer
    from notbydefault.graph import Graph, db_path
    project = Path(project).resolve()
    if not db_path(project).exists():
        out(f"No Decision Graph at {db_path(project)}; {NO_REASON}.")
        return 1
    entire = entire or Entire()
    tracer = tracer or make_tracer()
    facts = Facts()
    with tracer.root("notbydefault.why", {"notbydefault.question": question}):
        with tracer.step("plan", "planning",
                         reason="answer only from recorded facts: graph, decision records, "
                                "Entire history and Entire Graph code locations"):
            pass
        with Graph.for_project(project) as g:
            with tracer.step("graph_lookup", "state", reason="decisions the question is about"):
                decisions = match_decisions(g, question)[:3]
                reasons = graph_facts(g, decisions, facts)
                tracer.event("decisions_matched", "state",
                             f"{len(decisions)} decision(s): {[d['id'] for d in decisions]}")
        with tracer.step("history_lookup", "context",
                         reason="past sessions, checkpoints and commits about it (Entire)"):
            searched = history_facts(project, question, decisions, facts, entire, tracer)
        if not decisions and not facts.items:
            out(f"Q: {question}\n\nThere is {NO_REASON}: no decision in the Decision Graph, "
                f"checkpoint or commit matches this question (searched: {searched}).")
            tracer.flush()
            return 0
        answer = plain_answer(facts, reasons) if decisions else (
            f"There is {NO_REASON} in the Decision Graph; related history: "
            + " ".join(f"{f['text']} [{f['id']}]." for f in facts.items[:4]))
        if use_llm and decisions and any(reasons.values()):
            with tracer.step("answer", "execution",
                             reason="phrase the answer from the facts, citing their ids"):
                failed = False
                try:
                    if client is None:
                        from notbydefault.env import load_env
                        from notbydefault.fit.providers import ProviderSet, load_catalog
                        load_env()
                        client = ProviderSet(load_catalog(project),
                                             cache_dir=project / ".notbydefault" / "cache")
                    from notbydefault.fit.evaluate import Caller, SpendGuard
                    from notbydefault.fit.providers import role_model
                    caller = Caller(client, SpendGuard(0.05), tracer)
                    phrased = llm_answer(caller, role_model(client.catalog, "why"), question, facts)
                except Exception as e:
                    phrased, failed = None, True
                    tracer.event("answer_fallback", "recovery",
                                 f"model answer unavailable ({e}); showing the recorded facts")
            with tracer.step("verify_citations", "verification",
                             reason="every claim must cite a retrieved fact"):
                if phrased:
                    answer = (phrased + "\n(Phrased by a model from the sources below; the "
                              "sources are the record.)")
                elif not failed:
                    tracer.event("citations_rejected", "recovery",
                                 "the model's answer cited nothing or unknown ids; using the "
                                 "plain fact list instead")
        for line in render(question, facts, answer, searched):
            out(line)
    tracer.flush()
    return 0
