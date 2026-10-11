"""Test scout: edge cases the agent discussed while touching a decision's code but never tested.

  1. affected code of the decision (Entire Graph analysis stored on the node)
  2. checkpoints whose sessions touched those files (`entire checkpoint list` + `explain --json`)
  3. their transcripts (`entire checkpoint explain <id> --transcript`): user/assistant text that
     mentions the files, functions or edge-case words
  4. a Groq model extracts edge cases from those excerpts, quoting its evidence; a case whose
     quote isn't in the transcripts, or that the existing test inputs already cover, is dropped
  5. the rest are saved as "suggested" verifier cases in .notbydefault/suggested/<id>.jsonl;
     nothing runs them until someone promotes them into the samples file

existing_tests() is the deterministic half resolve uses: which project tests already exercise
each call site and caller, read from the test files' source.
"""
from __future__ import annotations

import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path

EDGE_WORDS = re.compile(r"\b(edge case|empty|blank|unicode|emoji|timezone|time zone|leap|null|"
                        r"none|missing|malformed|invalid|too long|large|huge|duplicate|"
                        r"what if|what about|fails?|crash|error|retry|timeout|rate limit|"
                        r"multilingual|language|cancel+ed|ambiguous|relative date|not tested|"
                        r"untested|corner case)\b", re.IGNORECASE)
MAX_EXCERPT_CHARS = 9000
MAX_CHECKPOINTS = 5

SCOUT_SYSTEM = ("You extract edge cases from a coding agent's conversation that were discussed but "
                "may not be tested. Answer with one JSON object only.")
SCOUT_PROMPT = """Feature under test: {feature}
Code: {code}

Existing test inputs (truncated):
{tests}

Conversation excerpts (from Entire checkpoints):
{excerpts}

List edge cases for this feature that the conversation raises. For each, quote the exact words
from the excerpts that raise it (copy them verbatim, 3 to 20 words), and say whether the existing
test inputs already cover it. Return
{{"cases": [{{"input": "<a concrete test input for the feature>", "why": "<one line>",
             "evidence": "<verbatim quote>", "already_tested": true or false}}]}}"""


def _text(message) -> str:
    content = (message or {}).get("content") if isinstance(message, dict) else message
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content
                         if isinstance(c, dict) and c.get("type") == "text")
    return ""


def excerpts(transcript: list[dict], needles: list[str]) -> list[str]:
    """User/assistant text that mentions the code or an edge-case word."""
    out = []
    for row in transcript:
        if row.get("type") not in ("user", "assistant"):
            continue
        text = _text(row.get("message")).strip()
        if not text or text.startswith("<"):
            continue
        if any(n and n.lower() in text.lower() for n in needles) and EDGE_WORDS.search(text):
            out.append(f"[{row['type']}] " + " ".join(text.split())[:1200])
    return out


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())


def existing_inputs(project: Path, decision_id: int) -> list[str]:
    from notbydefault.fit import failures as failures_mod
    from notbydefault.fit import samples as samples_mod
    samples = samples_mod.load(project, decision_id) or []
    return [s["user_message"] for s in samples] + failures_mod.inputs(project, decision_id)


def suggested_path(project: Path, decision_id: int) -> Path:
    return Path(project) / ".notbydefault" / "suggested" / f"{decision_id}.jsonl"


def _prefix(project: Path) -> str:
    from notbydefault.entire import repo_root
    root = repo_root(project)
    if root is None or root == project or root == Path.home().resolve():
        return ""
    try:
        return project.relative_to(root).as_posix() + "/"
    except ValueError:
        return ""


def find_checkpoints(project: Path, entire, files: list[str], tracer) -> list[tuple] | None:
    """[(checkpoint id, list row, touched affected files)], or None when unavailable."""
    listed = entire.checkpoint_list(project)
    if listed is None:
        tracer.event("checkpoints_unavailable", "recovery",
                     f"`entire checkpoint list` failed ({entire.last_error()}); nothing to scout")
        return None
    prefix = _prefix(project)
    touching = []
    for c in listed:
        cid = c.get("checkpoint_id")
        info = entire.checkpoint_explain(cid, project) if cid else None
        touched = {f[len(prefix):] for f in (info or {}).get("files_touched") or []
                   if f.startswith(prefix)}
        if touched & set(files):
            touching.append((cid, c, sorted(touched & set(files))))
        if len(touching) >= MAX_CHECKPOINTS:
            break
    return touching


def scout(project: Path, decision_id: int, client=None, entire=None, tracer=None,
          out=print) -> int:
    from notbydefault.affected import describe
    from notbydefault.entire import Entire
    from notbydefault.fit.tracing import make_tracer
    from notbydefault.graph import Graph, affected_entries, db_path, find_affected_code
    from notbydefault.pivot import used_for
    project = Path(project).resolve()
    if not db_path(project).exists():
        out(f"No Decision Graph at {db_path(project)}.")
        return 1
    with Graph.for_project(project) as g:
        d = g.get_node(decision_id)
    if not d or d["type"] != "decision":
        out(f"No decision #{decision_id}")
        return 1
    entire = entire or Entire()
    tracer = tracer or make_tracer()
    files = find_affected_code(d)
    entries = affected_entries(d)
    needles = files + [e["symbol"] for e in entries if e.get("symbol")] + [d["label"]]
    with tracer.root("notbydefault.scout", {"notbydefault.decision_id": decision_id}):
        with tracer.step("plan", "planning",
                         reason=f"read sessions that touched {', '.join(files) or 'nothing'} and "
                                "look for edge cases discussed but not tested"):
            pass
        with tracer.step("find_checkpoints", "context",
                         reason="checkpoints whose sessions touched the affected code"):
            touching = find_checkpoints(project, entire, files, tracer)
        if touching is None:
            out(f"No checkpoints available ({entire.last_error()}); nothing to scout.")
            tracer.flush()
            return 0
        with tracer.step("read_transcripts", "context",
                         reason=f"{len(touching)} checkpoint(s) touched the code"):
            texts, sources = read_transcripts(project, entire, touching, needles, tracer)
        if not texts:
            out(f"No discussed edge cases found for decision #{decision_id} ({len(touching)} "
                f"checkpoint(s) touched {', '.join(files) or 'no recorded code'}).")
            tracer.flush()
            return 0
        corpus = "\n".join(texts)[:MAX_EXCERPT_CHARS]
        with tracer.step("extract_cases", "tool_use",
                         reason="ask a fast model for edge cases, each with a verbatim quote"):
            if client is None:
                from notbydefault.env import load_env
                from notbydefault.fit.providers import ProviderSet, load_catalog
                load_env()
                client = ProviderSet(load_catalog(project),
                                     cache_dir=project / ".notbydefault" / "cache")
            from notbydefault.fit.evaluate import Caller, SpendGuard
            from notbydefault.fit.providers import role_model
            cases = extract_cases(Caller(client, SpendGuard(0.10), tracer),
                                  role_model(client.catalog, "scout"),
                                  used_for(project, decision_id) or d["label"],
                                  ", ".join(describe(e) for e in entries[:6]) or ", ".join(files),
                                  existing_inputs(project, decision_id), corpus)
        with tracer.step("verify_cases", "verification",
                         reason="keep only cases quoted from the transcripts and not yet tested"):
            kept = verify_cases(cases, texts, corpus, decision_id, tracer)
        with tracer.step("save_suggestions", "state", reason="store as suggested verifier cases"):
            path, new = save(project, decision_id, kept)
    out(f"Test scout for decision #{decision_id} {d['label']}: read {len(sources)} checkpoint(s) "
        f"that touched {', '.join(files)}; {len(cases)} case(s) proposed, {len(kept)} kept "
        f"({len(new)} new).")
    for k in kept:
        out(f"  - [suggested] {k['input']}")
        out(f"      why: {k['why']}")
        out(f"      evidence: \"{k['evidence']}\" [checkpoint {k['checkpoint']}]")
    if kept:
        out(f"Saved to {path.relative_to(project).as_posix()} (status: suggested; copy the ones "
            "you want into the decision's samples file to test them).")
    tracer.flush()
    return 0


def extract_cases(caller, model: str, feature: str, code: str, tests: list[str],
                  corpus: str) -> list[dict]:
    from notbydefault.llm import parse_json
    reply = caller.chat(model, [
        {"role": "system", "content": SCOUT_SYSTEM},
        {"role": "user", "content": SCOUT_PROMPT.format(
            feature=feature, code=code,
            tests="\n".join(f"- {t[:160]}" for t in tests[:40]) or "(none)", excerpts=corpus)}],
        span="notbydefault.scout", role="scout", max_tokens=3000, json_mode=True,
        focus_area="tool_use")
    return parse_json(reply.text).get("cases") or []


def verify_cases(cases: list[dict], texts: list[str], corpus: str, decision_id: int,
                 tracer) -> list[dict]:
    """Keep cases whose quote is really in the transcripts and that aren't tested yet."""
    hay = _norm(corpus)
    kept = []
    for case in cases:
        quote = str(case.get("evidence") or "")
        if not quote or _norm(quote) not in hay:
            tracer.event("drop_unquoted_case", "recovery",
                         f"dropped {case.get('input')!r}: its quote isn't in the transcripts")
            continue
        if case.get("already_tested"):
            tracer.event("drop_tested_case", "verification",
                         f"dropped {case.get('input')!r}: existing inputs cover it")
            continue
        cp = next((t.split(")", 1)[0].removeprefix("(checkpoint ") for t in texts
                   if _norm(quote) in _norm(t)), None)
        kept.append({"input": str(case.get("input")), "why": str(case.get("why") or ""),
                     "evidence": quote, "checkpoint": cp, "status": "suggested",
                     "decision_id": decision_id,
                     "found_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return kept


def save(project: Path, decision_id: int, kept: list[dict]) -> tuple[Path, list[dict]]:
    path = suggested_path(project, decision_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    old = ([json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
           if path.exists() else [])
    have = {_norm(o["input"]) for o in old}
    new = [k for k in kept if _norm(k["input"]) not in have]
    with open(path, "a", encoding="utf-8") as f:
        for k in new:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    return path, new


def test_files(project: Path) -> list[Path]:
    """The project's own Python test files: under tests/, or named test_*.py / *_test.py."""
    from notbydefault.resolve import TEST_SKIP_DIRS
    project = Path(project)
    out = []
    for p in sorted(project.rglob("*.py")):
        rel = p.relative_to(project).parts
        if TEST_SKIP_DIRS & set(rel[:-1]):
            continue
        if rel[0] == "tests" or re.match(r"(test_.*|.*_test)\.py$", p.name):
            out.append(p)
    return out


def _test_functions(tree):
    """(test id, node) for module-level test functions and test methods of Test* classes."""
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test"):
            yield n.name, n
        elif isinstance(n, ast.ClassDef) and n.name.startswith("Test"):
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and m.name.startswith("test"):
                    yield f"{n.name}::{m.name}", m


def _imports(tree) -> set[str]:
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:     # `from pkg import jobs` too
            mods |= {n.module} | {f"{n.module}.{a.name}" for a in n.names}
    return mods


def existing_tests(project: Path, sites) -> dict[str, list[str]]:
    """For each site (path, symbol): the project tests that exercise it.

    A test function that names the site's function (or its class) covers it directly
    ("tests/test_app.py::test_x"); a test file that only imports the site's module is listed
    as "tests/test_app.py (imports app)". Read from source, nothing is run."""
    project = Path(project)
    parsed = []
    for f in test_files(project):
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        funcs = [(tid, {x.id for x in ast.walk(fn) if isinstance(x, ast.Name)}
                  | {x.attr for x in ast.walk(fn) if isinstance(x, ast.Attribute)})
                 for tid, fn in _test_functions(tree)]
        parsed.append((f.relative_to(project).as_posix(), _imports(tree), funcs))
    out = {}
    for s in sites:
        module = s.path.removesuffix(".py").replace("/", ".")
        names = set((s.symbol or "").split(".")) - {""}
        found = []
        for rel, imports, funcs in parsed:
            if rel == s.path:
                continue
            direct = [f"{rel}::{tid}" for tid, used in funcs if names & used]
            if direct:
                found += direct
            elif module in imports or module.rsplit(".", 1)[-1] in imports:
                found.append(f"{rel} (imports {module})")
        out[s.describe()] = found
    return out


def read_transcripts(project: Path, entire, touching, needles, tracer):
    texts, sources = [], []
    for cid, c, touched in touching:
        rows = entire.checkpoint_transcript(cid, project)
        if rows is None:
            tracer.event("transcript_unavailable", "recovery",
                         f"checkpoint {cid}: {entire.last_error()}; skipped")
            continue
        found = excerpts(rows, needles)
        if found:
            texts += [f"(checkpoint {cid}) {t}" for t in found]
            sources.append({"checkpoint": cid, "session": c.get("session_id"), "files": touched})
    return texts, sources
