"""`notbydefault eval`: measure NotByDefault itself on cases with known right answers.

Sections (cases in eval/cases/, one neatlogs trace each):
  detection     PreToolUse hook on tool calls that are / are not stack decisions
  pivot         UserPromptSubmit Pivot Check on real pivots vs ordinary prompts
  requirements  `require` on new requirements, expected stale / needs review / valid per decision
  fit           model-fit verdicts read from a project's saved results (never run here)

Every case runs on a throwaway project in a temp dir. LLM calls go through the project's
Groq roles; a Gemini call is refused, so nothing here spends Gemini quota. Model replies are
cached in eval/.cache so a rerun with unchanged prompts is free and repeatable.
"""
from __future__ import annotations

import contextlib
import glob
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from notbydefault import REPO_ROOT
from notbydefault.graph import Graph

EVAL_DIR = REPO_ROOT / "eval"
CASES = EVAL_DIR / "cases"
CACHE = EVAL_DIR / ".cache"
SECTIONS = ("detection", "pivot", "requirements", "fit")


# --- helpers ---------------------------------------------------------------------------

class GeminiRefused(RuntimeError):
    pass


class CountingClient:
    """A ProviderSet that counts chat calls and refuses Gemini (its quota is reserved)."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0
        self.cached = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def chat(self, model, messages, **kw):
        if model.startswith("gemini:"):
            raise GeminiRefused(f"eval refuses Gemini calls ({model}); Gemini quota is reserved")
        self.calls += 1
        result = self.inner.chat(model, messages, **kw)
        self.cached += bool(getattr(result, "cached", False))
        return result


class NoFlush:
    """Lets a command that flushes its own tracer (require) run inside a section's trace."""

    def __init__(self, tracer):
        self._t = tracer

    def __getattr__(self, name):
        return getattr(self._t, name)

    def flush(self):
        pass


@contextlib.contextmanager
def env(**values):
    old = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def load(name: str) -> dict:
    return json.loads((CASES / f"{name}.json").read_text(encoding="utf-8"))


CASE_SETS = ("all", "original", "heldout")


def load_cases(section: str, which: str = "all") -> list[dict]:
    """The section's cases: original (eval/cases/<section>.json, used while fixing) and/or
    held-out (eval/cases/heldout/<section>.json, written before the fixes they measure)."""
    out = []
    if which in ("all", "original"):
        out += [dict(c, split="original") for c in load(section)["cases"]]
    held = CASES / "heldout" / f"{section}.json"
    if which in ("all", "heldout") and held.exists():
        out += [dict(c, split="heldout") for c in load(f"heldout/{section}")["cases"]]
    return out


def write_files(project: Path, files: dict) -> None:
    for rel, text in (files or {}).items():
        p = project / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


def subst(value, project: Path):
    if isinstance(value, str):
        return value.replace("{project}", project.as_posix())
    if isinstance(value, dict):
        return {k: subst(v, project) for k, v in value.items()}
    if isinstance(value, list):
        return [subst(v, project) for v in value]
    return value


def build_project(path: Path, spec: dict) -> dict[str, int]:
    """The base project (files + Decision Graph) in `path`; returns decision key -> node id."""
    write_files(path, spec.get("files"))
    ids = {}
    with Graph.for_project(path) as g:
        goal = None
        for n in spec.get("context", []):
            nid = g.add_node(n["type"], n["label"], {"source": "init", **n.get("data", {})})
            goal = goal or (nid if n["type"] == "goal" else None)
        for d in spec.get("decisions", []):
            data = {k: v for k, v in d.items() if k not in ("key", "label")}
            data.update(chosen=d["label"], source="eval", file=(d.get("files") or [None])[0],
                        times_seen=1)
            did = g.add_node("decision", d["label"], data, "eval")
            ids[d["key"]] = did
            if goal:
                g.add_edge(did, goal, "serves")
            for t in ("requirement", "constraint"):
                for n in g.nodes(t):
                    g.add_edge(did, n["id"], "depends_on")
            for f in d.get("files") or []:
                code = g.find_node("code", f)
                cid = code["id"] if code else g.add_node("code", f, {"path": f})
                g.add_edge(did, cid, "affects")
    return ids


def split(case: dict) -> str:
    """"original" cases were used while fixing; "heldout" ones were written afterwards."""
    return case.get("split", "original")


class Bases:
    """One built base project per project file (eval/cases/<name>.json), copied per case."""

    def __init__(self, root: Path):
        self.root, self.built = root, {}

    def get(self, name: str) -> tuple[Path, dict[str, int]]:
        if name not in self.built:
            base = self.root / ("base-" + name.replace("/", "-"))
            base.mkdir()
            self.built[name] = (base, build_project(base, load(name)))
        return self.built[name]


def make_client(project: Path):
    from notbydefault.env import load_env
    from notbydefault.fit.providers import ProviderSet, load_catalog
    load_env()
    return CountingClient(ProviderSet(load_catalog(project), cache_dir=CACHE))


def ratio(a: int, b: int) -> float | None:
    return round(a / b, 3) if b else None


_version = "dev"   # the eval version being run, for case span attributes


def case_done(tracer, out, i: int, total: int, row: dict, seconds: float) -> None:
    """Tag the case's span for neatlogs Conditions (called inside it) and print progress."""
    failed = not row.get("ok") and row.get("status") != "pending"
    tracer.annotate({"nbd.eval_case_failed": bool(failed), "nbd.eval_version": _version,
                     "nbd.eval_case": row["case"], "nbd.eval_split": row.get("split", "original")})
    progress(out, i, total, row, seconds)


def progress(out, i: int, total: int, row: dict, seconds: float) -> None:
    """One line per finished case, so a long run shows where it is."""
    status = {"pending": "PENDING"}.get(row.get("status"), "ok" if row.get("ok") else "FAIL")
    calls = row.get("llm_calls")
    extra = f", {calls} LLM call(s)" + (f" ({row['llm_calls_cached']} cached)"
                                        if row.get("llm_calls_cached") else "") if calls else ""
    split_tag = " [held-out]" if row.get("split") == "heldout" else ""
    out(f"    [{i}/{total}] {row['case']}{split_tag}: {status} ({seconds:.1f}s{extra})")


# --- detection ---------------------------------------------------------------------------

def _times_seen(project: Path) -> dict[tuple[int, str], int]:
    from notbydefault.graph import db_path
    if not db_path(project).exists():
        return {}
    with Graph.for_project(project) as g:
        return {(d["id"], d["label"]): d["data"].get("times_seen", 1) for d in g.nodes("decision")}


def run_detection(tracer, out=print, which: str = "all") -> dict:
    from notbydefault.hooks.pre_tool import handle
    cases = load_cases("detection", which)
    rows, failures = [], []
    with tempfile.TemporaryDirectory(prefix="nbd-eval-det-") as tmp, \
            env(NOTBYDEFAULT_ENTIRE="0", NOTBYDEFAULT_TELEMETRY="0"):
        total = len(cases)
        for i, case in enumerate(cases, 1):
            t0 = time.perf_counter()
            project = Path(tmp) / case["id"]
            project.mkdir()
            write_files(project, case.get("files"))
            if case.get("graph"):
                with Graph.for_project(project) as g:
                    for d in case["graph"]:
                        g.add_node("decision", d["label"], {**d, "chosen": d["label"]}, "eval")
            event = {"tool_name": case["tool_name"], "tool_input": subst(case["tool_input"], project),
                     "cwd": str(project), "session_id": "eval", "tool_use_id": case["id"]}
            seen_before = _times_seen(project)
            with tracer.step(f"case:{case['id']}", "verification", reason=case["why"]):
                started = time.perf_counter()
                error = ""
                try:
                    result = handle(event, project, tracer)
                except Exception as e:      # the hook itself never raises; record if it does
                    result, error = None, repr(e)
                ms = (time.perf_counter() - started) * 1000
                asked = bool(result)
                # what the hook recorded: new decisions, or existing ones it saw again
                flagged = [label for (did, label), n in _times_seen(project).items()
                           if n > seen_before.get((did, label), 0)] if asked else []
                expected = {e.lower() for e in case["expect"]}
                got = {f.lower() for f in flagged}
                ok = (asked == bool(expected)) and expected == got and not error
                reason = ""
                if not ok:
                    if error:
                        reason = f"hook error: {error}"
                    elif expected and not asked:
                        reason = f"missed: expected {sorted(case['expect'])}, the hook stayed silent"
                    elif not expected and asked:
                        reason = f"false interruption: flagged {sorted(flagged)} on {case['why']}"
                    else:
                        reason = (f"wrong decisions: expected {sorted(case['expect'])}, "
                                  f"flagged {sorted(flagged)}")
                    failures.append({"case": case["id"], "split": split(case), "reason": reason})
                tracer.event("case_pass" if ok else "case_fail", "verification",
                             reason or "as expected", case=case["id"], asked=asked,
                             flagged=flagged, expected=case["expect"])
                rows.append({"case": case["id"], "split": split(case), "ok": ok,
                             "expected": case["expect"], "flagged": flagged, "asked": asked,
                             "ms": round(ms, 1), **({"reason": reason} if reason else {})})
                case_done(tracer, out, i, total, rows[-1], time.perf_counter() - t0)
    return {"cases": rows, "failures": failures}


def detection_metrics(rows: list[dict]) -> dict:
    tp = fp = fn = interruptions = false_interruptions = negatives = 0
    for r in rows:
        expected = {e.lower() for e in r["expected"]}
        got = {f.lower() for f in r["flagged"]}
        tp, fp, fn = tp + len(expected & got), fp + len(got - expected), fn + len(expected - got)
        interruptions += r["asked"]
        if not expected:
            negatives += 1
            false_interruptions += r["asked"]
    return {"cases": len(rows), "passed": sum(r["ok"] for r in rows),
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "interruptions": interruptions, "false_interruptions": false_interruptions,
            "false_interruption_rate": ratio(false_interruptions, negatives),
            "negatives": negatives}


# --- pivot -------------------------------------------------------------------------------

def run_pivot(tracer, out=print, which: str = "all") -> dict:
    from notbydefault.hooks.user_prompt import handle
    cases = load_cases("pivot", which)
    rows, failures = [], []
    with tempfile.TemporaryDirectory(prefix="nbd-eval-piv-") as tmp, \
            env(NOTBYDEFAULT_ENTIRE="0", NOTBYDEFAULT_TELEMETRY="0"):
        bases = Bases(Path(tmp))
        client = make_client(Path(tmp))
        total = len(cases)
        for i, case in enumerate(cases, 1):
            t0 = time.perf_counter()
            base, keys = bases.get(case.get("project", "project"))
            by_id = {v: k for k, v in keys.items()}
            project = Path(tmp) / case["id"]
            shutil.copytree(base, project)
            with tracer.step(f"case:{case['id']}", "verification",
                             reason=("real pivot" if case["pivot"] else "ordinary prompt")):
                before, cached_before = client.calls, client.cached
                error = ""
                started = time.perf_counter()
                try:
                    output, info = handle({"prompt": case["prompt"], "session_id": "eval"},
                                          project, client=client, tracer=tracer)
                except Exception as e:
                    output, info, error = None, {"stage": "error"}, repr(e)
                ms = (time.perf_counter() - started) * 1000
                calls, cached = client.calls - before, client.cached - cached_before
                said_pivot = info.get("stage") == "confirmed"
                cache = project / ".notbydefault" / "cache" / "openrouter-models.json"
                if cache.exists() and not (base / ".notbydefault" / "cache" / cache.name).exists():
                    (base / ".notbydefault" / "cache").mkdir(parents=True, exist_ok=True)
                    shutil.copy2(cache, base / ".notbydefault" / "cache" / cache.name)
                mapped = sorted(by_id.get(i, str(i)) for i in info.get("decisions", []))
                reason, mapping_ok = "", None
                if error:
                    reason = f"hook error: {error}"
                elif case["pivot"] and not said_pivot:
                    from notbydefault.pivot import PIVOT_RX
                    phrase = PIVOT_RX.search(case["prompt"])
                    reason = (("missed pivot: the pre-filter found no change-of-direction phrase"
                               if not phrase else
                               f"missed pivot: the pre-filter matched \"{phrase.group(0)}\" but "
                               "found no stack word, decision name or recorded goal, so it "
                               "skipped the LLM")
                              if info.get("stage") == "prefilter"
                              else f"missed pivot: the model said it is not a pivot")
                elif not case["pivot"] and said_pivot:
                    reason = f"false pivot: the model read it as \"{info.get('change')}\""
                elif case["pivot"] and said_pivot:
                    missing = sorted(set(case["decisions"]) - set(mapped))
                    mapping_ok = not missing
                    if missing:
                        reason = f"pivot found but not mapped to {missing} (mapped to {mapped})"
                ok = not reason
                if not ok:
                    failures.append({"case": case["id"], "split": split(case), "reason": reason,
                                     "prompt": case["prompt"]})
                tracer.event("case_pass" if ok else "case_fail", "verification",
                             reason or "as expected", case=case["id"], llm_calls=calls,
                             stage=info.get("stage"))
                rows.append({"case": case["id"], "split": split(case), "ok": ok,
                             "expected_pivot": case["pivot"], "said_pivot": said_pivot,
                             "stage": info.get("stage"), "phrases": info.get("phrases"),
                             "change": info.get("change"), "mapped": mapped,
                             "mapping_ok": mapping_ok, "llm_calls": calls,
                             "llm_calls_cached": cached, "ms": round(ms, 1),
                             **({"reason": reason} if reason else {})})
                case_done(tracer, out, i, total, rows[-1], time.perf_counter() - t0)
    return {"cases": rows, "failures": failures}


def pivot_metrics(rows: list[dict]) -> dict:
    piv = [r for r in rows if r["expected_pivot"]]
    ordinary = [r for r in rows if not r["expected_pivot"]]
    tp = sum(r["said_pivot"] for r in piv)
    fp = sum(r["said_pivot"] for r in ordinary)
    mapped = [r["mapping_ok"] for r in piv if r["mapping_ok"] is not None]
    return {"cases": len(rows), "passed": sum(r["ok"] for r in rows),
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, len(piv)),
            "false_pivots": fp, "missed_pivots": len(piv) - tp,
            "llm_calls_per_ordinary_prompt": ratio(sum(r["llm_calls"] for r in ordinary),
                                                   len(ordinary)),
            "ordinary_prompts_reaching_llm":
                f"{sum(r['stage'] != 'prefilter' for r in ordinary)}/{len(ordinary)}",
            "mapping_accuracy": ratio(sum(mapped), len(mapped)),
            "llm_calls": sum(r["llm_calls"] for r in rows),
            "llm_calls_cached": sum(r["llm_calls_cached"] for r in rows)}


# --- requirements ------------------------------------------------------------------------

def run_requirements(tracer, out=print, which: str = "all") -> dict:
    from notbydefault.stale import require
    cases = load_cases("requirements", which)
    rows, failures = [], []
    with tempfile.TemporaryDirectory(prefix="nbd-eval-req-") as tmp, \
            env(NOTBYDEFAULT_ENTIRE="0", NOTBYDEFAULT_TELEMETRY="0"):
        bases = Bases(Path(tmp))
        client = make_client(Path(tmp))
        total = len(cases)
        for i, case in enumerate(cases, 1):
            t0 = time.perf_counter()
            base, keys = bases.get(case.get("project", "project"))
            project = Path(tmp) / case["id"]
            shutil.copytree(base, project)
            with tracer.step(f"case:{case['id']}", "verification", reason=case["requirement"]):
                before, cached_before = client.calls, client.cached
                lines: list[str] = []
                error = ""
                try:
                    require(project, case["requirement"], client=client, tracer=NoFlush(tracer),
                            out=lines.append)
                except Exception as e:
                    error = repr(e)
                calls, cached = client.calls - before, client.cached - cached_before
                judged, constraints = {}, {}
                with Graph.for_project(project) as g:
                    reqs = [n for n in g.nodes("requirement") if n["data"].get("source") == "require"]
                    if reqs:
                        judged = reqs[-1]["data"].get("judged") or {}
                        constraints = reqs[-1]["data"].get("constraints") or {}
                per = {}
                for key, want in case["expect"].items():
                    allowed = [want] if isinstance(want, str) else list(want)
                    got = (judged.get(str(keys[key])) or {})
                    status = got.get("status") or ("error" if error else "missing")
                    ok = status in allowed
                    per[key] = {"expected": allowed, "got": status, "ok": ok,
                                "reason": got.get("reason", ""), "basis": got.get("basis", ""),
                                "property": got.get("property", "")}
                    if not ok:
                        failures.append({
                            "case": case["id"], "split": split(case), "decision": key,
                            "reason": f"expected {'/'.join(allowed)}, got {status}"
                                      + (f" ({got.get('basis')})" if got.get("basis") else "")
                                      + (f": {got.get('reason')}" if got.get("reason") else "")
                                      + (f" [{error}]" if error else "")})
                ok_case = all(p["ok"] for p in per.values())
                tracer.event("case_pass" if ok_case else "case_fail", "verification",
                             f"{sum(p['ok'] for p in per.values())}/{len(per)} decisions right",
                             case=case["id"], llm_calls=calls, constraints=constraints)
                rows.append({"case": case["id"], "split": split(case), "ok": ok_case,
                             "requirement": case["requirement"], "constraints": constraints,
                             "decisions": per, "llm_calls": calls, "llm_calls_cached": cached,
                             **({"error": error} if error else {})})
                case_done(tracer, out, i, total, rows[-1], time.perf_counter() - t0)
    return {"cases": rows, "failures": failures}


def requirements_metrics(rows: list[dict]) -> dict:
    per = [d for r in rows for d in r["decisions"].values()]
    correct = sum(d["ok"] for d in per)
    return {"cases": len(rows), "passed": sum(r["ok"] for r in rows),
            "accuracy": ratio(correct, len(per)), "decisions_right": f"{correct}/{len(per)}",
            "false_stale": sum(d["got"] == "stale" and "stale" not in d["expected"] for d in per),
            "missed_stale": sum(d["expected"] == ["stale"] and d["got"] != "stale" for d in per),
            "llm_calls": sum(r["llm_calls"] for r in rows),
            "llm_calls_cached": sum(r["llm_calls_cached"] for r in rows)}


# --- model fit ---------------------------------------------------------------------------

def _decision(project: Path, did: int) -> dict | None:
    """Read one decision from a project's graph without opening it for writing."""
    db = project / ".notbydefault" / "graph.db"
    if not db.exists():
        return None
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT label, data_json FROM nodes WHERE id = ?", (did,)).fetchone()
    finally:
        conn.close()
    return {"label": row[0], "data": json.loads(row[1])} if row else None


def fit_status(project: Path, did: int) -> dict:
    state = project / ".notbydefault"
    decision = _decision(project, did)
    if decision is None:
        return {"status": "missing", "reason": f"no decision #{did} in {project}"}
    finals = []
    for path in sorted(glob.glob(str(state / "results" / f"{did}-*.json"))):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("verdict") and not data.get("stopped"):
            finals.append((path, data))
    if finals:
        path, data = finals[-1]
        v = data["verdict"]
        return {"status": "final", "verdict": v.get("kind"), "summary": v.get("summary"),
                "recommended": v.get("recommended"), "judge": data.get("config", {}).get("judge"),
                "results": Path(path).relative_to(project).as_posix(),
                "check_trace_id": data.get("trace_id"), "run_at": data.get("run_at")}
    journals = sorted(glob.glob(str(state / "journal" / f"{did}-*.json")))
    if not journals:
        return {"status": "pending", "reason": "no fit check has run yet",
                "decision_status": decision["data"].get("status")}
    path = journals[-1]
    journal = json.loads(Path(path).read_text(encoding="utf-8"))
    stop = next((e for e in reversed(journal) if e.get("name") == "daily_limit_stop"), None)
    verdict = next((e for e in reversed(journal) if e.get("name") == "verdict"), None)
    plan = next((e for e in journal if e.get("name") == "plan"), {})
    return {"status": "pending",
            "reason": (f"the fit check stopped early at {stop.get('model')}'s daily free-tier "
                       "limit, so the final judged run has not happened" if stop else
                       f"the fit check stopped early ({Path(path).stem.split('-', 2)[-1]})"),
            "stopped_model": stop and stop.get("model"), "judge": plan.get("judge"),
            "provisional_verdict": verdict and verdict.get("verdict"),
            "provisional_summary": verdict and verdict.get("reason"),
            "journal": Path(path).relative_to(project).as_posix(),
            "check_trace_id": journal[0].get("trace_id") if journal else None,
            "decision_status": decision["data"].get("status")}


def run_fit(tracer, out=print, which: str = "all") -> dict:
    rows, failures = [], []
    cases = load_cases("fit", which)
    total = len(cases)
    for i, case in enumerate(cases, 1):
        t0 = time.perf_counter()
        project = (REPO_ROOT / case["project"]).resolve()
        with tracer.step(f"case:{case['id']}", "verification",
                         reason="read the saved fit-check verdict (no model calls)"):
            st = fit_status(project, case["decision_id"])
            row = {"case": case["id"], "split": split(case), "decision_id": case["decision_id"], **st}
            expect = case.get("expect") or {}
            wrong = {k: st.get(k) for k, v in expect.items() if st.get(k) != v}
            row["ok"] = st["status"] == "final" and not wrong
            if st["status"] == "pending":
                row["needs"] = case.get("needs")
                row["resume"] = case.get("resume")
                failures.append({"case": case["id"], "split": split(case), "pending": True,
                                 "reason": f"pending: {st['reason']}. Needs {case.get('needs')}. "
                                           f"Then run `{case.get('resume')}` and "
                                           "`notbydefault eval --sections fit`."})
            elif wrong:
                failures.append({"case": case["id"], "split": split(case),
                                 "reason": f"expected {expect}, got {wrong}"})
            tracer.event("fit_" + st["status"], "verification",
                         st.get("summary") or st.get("reason", ""), case=case["id"])
            rows.append(row)
            case_done(tracer, out, i, total, rows[-1], time.perf_counter() - t0)
    return {"cases": rows, "failures": failures}


def fit_metrics(rows: list[dict]) -> dict:
    return {"cases": len(rows), "passed": sum(r["ok"] for r in rows),
            "final": sum(r["status"] == "final" for r in rows),
            "pending": sum(r["status"] == "pending" for r in rows)}


RUNNERS = {"detection": run_detection, "pivot": run_pivot,
           "requirements": run_requirements, "fit": run_fit}
METRIC_FNS = {"detection": detection_metrics, "pivot": pivot_metrics,
              "requirements": requirements_metrics, "fit": fit_metrics}


def score(section: dict, name: str) -> None:
    """metrics on the original cases (comparable across versions) and on the held-out ones."""
    rows = section["cases"]
    orig = [r for r in rows if r.get("split", "original") == "original"]
    section["metrics"] = METRIC_FNS[name](orig) if orig else None
    held = [r for r in rows if r.get("split") == "heldout"]
    section["heldout_metrics"] = METRIC_FNS[name](held) if held else None


# --- report ------------------------------------------------------------------------------

METRICS = {
    "detection": ["cases", "passed", "precision", "recall", "interruptions",
                  "false_interruptions", "false_interruption_rate"],
    "pivot": ["cases", "passed", "precision", "recall", "false_pivots", "missed_pivots",
              "llm_calls_per_ordinary_prompt", "ordinary_prompts_reaching_llm",
              "mapping_accuracy", "llm_calls"],
    "requirements": ["cases", "passed", "accuracy", "decisions_right", "false_stale",
                     "missed_stale", "llm_calls"],
    "fit": ["cases", "passed", "final", "pending"],
}


def _fmt(v) -> str:
    return "-" if v is None else str(v)


def _grid(rows: list[tuple]) -> list[str]:
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]) - 1)]
    lines = ["  ".join(f"{c:<{w[i]}}" for i, c in enumerate(r[:-1])) + "  " + r[-1] for r in rows]
    lines.insert(1, "-" * min(130, max(len(l) for l in lines)))
    return lines


def table(results: dict) -> list[str]:
    s = results["sections"]
    rows = [("section", "metric", "original", "held-out")]
    for sec, keys in METRICS.items():
        if sec not in s:
            continue
        m, h = s[sec].get("metrics") or {}, s[sec].get("heldout_metrics") or {}
        for k in keys:
            rows.append((sec, k.replace("_", " "), _fmt(m.get(k)), _fmt(h.get(k))))
        if sec == "fit":
            for r in s["fit"]["cases"]:
                val = (f"{r['verdict']}: {r['summary']}" if r["status"] == "final"
                       else f"PENDING (provisional: {r.get('provisional_verdict') or '-'})")
                rows.append(("fit", f"decision #{r['decision_id']} ({r.get('split', 'original')})",
                             val[:100], ""))
        rows.append((sec, "trace id", s[sec].get("trace_id") or "-", ""))
    return _grid(rows)


def failure_lines(results: dict) -> list[str]:
    fails = [(name, f) for name in SECTIONS if name in results["sections"]
             for f in results["sections"][name]["failures"]]
    lines = [f"Failures ({len(fails)}):"]
    for name, f in fails:
        where = f["case"] + (f" / {f['decision']}" if f.get("decision") else "")
        tag = " (held-out)" if f.get("split") == "heldout" else ""
        lines.append(f"  [{name}]{tag} {where}: {f['reason']}")
    return lines


def _results(version: str) -> dict | None:
    path = EVAL_DIR / "results" / f"{version}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def compare(versions: list[str], out=print) -> int:
    """Every metric and trace id side by side: each version on the original cases, then each
    version on the held-out cases (from <version>-heldout.json, or the version's own run)."""
    files = {v: _results(v) for v in versions}
    missing = [v for v, r in files.items() if r is None]
    if missing:
        out(f"No results for {', '.join(missing)} in eval/results/")
        return 1
    cols = [(v, files[v], "metrics") for v in versions]
    for v in versions:
        held = _results(f"{v}-heldout") or files[v]
        if any((sec.get("heldout_metrics") for sec in held["sections"].values())):
            cols.append((f"{v} held-out", held, "heldout_metrics"))
    rows = [("section", "metric", *[c[0] for c in cols])]
    for sec, keys in METRICS.items():
        for k in keys:
            rows.append((sec, k.replace("_", " "),
                         *[_fmt(((r["sections"].get(sec) or {}).get(key) or {}).get(k))
                           for _, r, key in cols]))
        rows.append((sec, "trace id", *[(r["sections"].get(sec) or {}).get("trace_id") or "-"
                                        for _, r, _k in cols]))
    out("\n".join(_grid(rows)))
    last = versions[-1]
    out("")
    out("\n".join(failure_lines(files[last])).replace("Failures", f"Remaining failures in {last}", 1))
    return 0


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def _printer(line: str = "") -> None:
    print(line, flush=True)


def run(version: str = "v1", sections: list[str] | None = None, out_path: Path | None = None,
        out=_printer, which: str = "all") -> int:
    global _version
    from notbydefault.awake import keep_awake
    from notbydefault.fit.tracing import make_tracer
    _version = version
    sections = sections or [s for s in SECTIONS if load_cases(s, which)]
    out_path = out_path or EVAL_DIR / "results" / f"{version}.json"
    results = {"version": version, "cases": which, "sections": {}}
    if out_path.exists() and set(sections) != set(SECTIONS) and which == "all":   # update some sections in place
        results = json.loads(out_path.read_text(encoding="utf-8"))
    results.update(run_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   commit=git_commit())
    tracer = make_tracer("notbydefault-eval-" + version, spill_dir=EVAL_DIR / ".telemetry")

    def save():
        results["telemetry"] = {"exporter": "neatlogs" if tracer.enabled else "off",
                                "export": getattr(tracer, "export_status", "off"),
                                "journal": getattr(tracer, "journal_path", None)}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")

    with keep_awake():
        for name in sections:
            out(f"running {name} ...")
            started = time.perf_counter()
            with tracer.root(f"notbydefault.eval.{name}", {"notbydefault.eval_version": version,
                                                           "notbydefault.eval_section": name}):
                trace_id = tracer.trace_id
                section = RUNNERS[name](tracer, out, which)
                score(section, name)
                if section["metrics"]:
                    tracer.annotate({f"notbydefault.eval.{k}": v
                                     for k, v in section["metrics"].items()})
                if section["heldout_metrics"]:
                    tracer.annotate({f"notbydefault.eval.heldout.{k}": v
                                     for k, v in section["heldout_metrics"].items()})
            section["trace_id"] = trace_id
            section["seconds"] = round(time.perf_counter() - started, 1)
            results["sections"][name] = section
            m, h = section["metrics"], section["heldout_metrics"]
            out(f"  {name}: "
                + (f"{m['passed']}/{m['cases']}" if m else "")
                + (f" (held-out {h['passed']}/{h['cases']})" if h else "")
                + f" in {section['seconds']}s")
            save()                                   # after every section
        tracer.flush()                               # bounded; never hangs the eval
    if not getattr(tracer, "export_ok", True):       # trace ids were never delivered
        for section in results["sections"].values():
            section["trace_exported"] = False
    save()

    out("")
    out("\n".join(table(results)))
    out("")
    out("\n".join(failure_lines(results)))
    t = results["telemetry"]
    if t["exporter"] == "neatlogs":
        out(f"\nneatlogs export: {t['export']}" + (f"; journal kept in {t['journal']}"
                                                   if t["journal"] else ""))
    out(f"Saved {out_path.relative_to(REPO_ROOT).as_posix() if out_path.is_relative_to(REPO_ROOT) else out_path}")
    return 0
