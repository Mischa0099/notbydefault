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


def make_client(project: Path):
    from notbydefault.env import load_env
    from notbydefault.fit.providers import ProviderSet, load_catalog
    load_env()
    return CountingClient(ProviderSet(load_catalog(project), cache_dir=CACHE))


def ratio(a: int, b: int) -> float | None:
    return round(a / b, 3) if b else None


# --- detection ---------------------------------------------------------------------------

def _times_seen(project: Path) -> dict[tuple[int, str], int]:
    from notbydefault.graph import db_path
    if not db_path(project).exists():
        return {}
    with Graph.for_project(project) as g:
        return {(d["id"], d["label"]): d["data"].get("times_seen", 1) for d in g.nodes("decision")}


def run_detection(tracer, out=print) -> dict:
    from notbydefault.hooks.pre_tool import handle
    cases = load("detection")["cases"]
    rows, failures = [], []
    tp = fp = fn = 0
    interruptions = false_interruptions = negatives = 0
    with tempfile.TemporaryDirectory(prefix="nbd-eval-det-") as tmp, \
            env(NOTBYDEFAULT_ENTIRE="0", NOTBYDEFAULT_TELEMETRY="0"):
        for case in cases:
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
                c_tp, c_fp, c_fn = len(expected & got), len(got - expected), len(expected - got)
                tp, fp, fn = tp + c_tp, fp + c_fp, fn + c_fn
                interruptions += asked
                if not expected:
                    negatives += 1
                    false_interruptions += asked
                ok = (asked == bool(expected)) and not c_fp and not c_fn and not error
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
                    failures.append({"case": case["id"], "reason": reason})
                tracer.event("case_pass" if ok else "case_fail", "verification",
                             reason or "as expected", case=case["id"], asked=asked,
                             flagged=flagged, expected=case["expect"])
                rows.append({"case": case["id"], "ok": ok, "expected": case["expect"],
                             "flagged": flagged, "asked": asked, "ms": round(ms, 1),
                             **({"reason": reason} if reason else {})})
    metrics = {"cases": len(cases), "passed": sum(r["ok"] for r in rows),
               "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
               "interruptions": interruptions, "false_interruptions": false_interruptions,
               "false_interruption_rate": ratio(false_interruptions, negatives),
               "negatives": negatives}
    return {"metrics": metrics, "cases": rows, "failures": failures}


# --- pivot -------------------------------------------------------------------------------

def run_pivot(tracer, out=print) -> dict:
    from notbydefault.hooks.user_prompt import handle
    spec, cases = load("project"), load("pivot")["cases"]
    rows, failures = [], []
    tp = fp = fn = tn = 0
    ordinary_calls = ordinary = prefilter_hits = 0
    mapped_ok = mapped_n = 0
    with tempfile.TemporaryDirectory(prefix="nbd-eval-piv-") as tmp, \
            env(NOTBYDEFAULT_ENTIRE="0", NOTBYDEFAULT_TELEMETRY="0"):
        base = Path(tmp) / "base"
        base.mkdir()
        keys = build_project(base, spec)
        by_id = {v: k for k, v in keys.items()}
        client = make_client(base)
        for case in cases:
            project = Path(tmp) / case["id"]
            shutil.copytree(base, project)
            with tracer.step(f"case:{case['id']}", "verification",
                             reason=("real pivot" if case["pivot"] else "ordinary prompt")):
                before = client.calls
                error = ""
                started = time.perf_counter()
                try:
                    output, info = handle({"prompt": case["prompt"], "session_id": "eval"},
                                          project, client=client, tracer=tracer)
                except Exception as e:
                    output, info, error = None, {"stage": "error"}, repr(e)
                ms = (time.perf_counter() - started) * 1000
                calls = client.calls - before
                said_pivot = info.get("stage") == "confirmed"
                cache = project / ".notbydefault" / "cache" / "openrouter-models.json"
                if cache.exists() and not (base / ".notbydefault" / "cache" / cache.name).exists():
                    (base / ".notbydefault" / "cache").mkdir(parents=True, exist_ok=True)
                    shutil.copy2(cache, base / ".notbydefault" / "cache" / cache.name)
                mapped = sorted(by_id.get(i, str(i)) for i in info.get("decisions", []))
                reason = ""
                if error:
                    reason = f"hook error: {error}"
                elif case["pivot"] and not said_pivot:
                    from notbydefault.pivot import PIVOT_RX
                    phrase = PIVOT_RX.search(case["prompt"])
                    reason = (("missed pivot: the pre-filter found no change-of-direction phrase"
                               if not phrase else
                               f"missed pivot: the pre-filter matched \"{phrase.group(0)}\" but "
                               "found no stack word or decision name, so it skipped the LLM")
                              if info.get("stage") == "prefilter"
                              else f"missed pivot: the model said it is not a pivot")
                elif not case["pivot"] and said_pivot:
                    reason = f"false pivot: the model read it as \"{info.get('change')}\""
                elif case["pivot"] and said_pivot:
                    missing = sorted(set(case["decisions"]) - set(mapped))
                    mapped_n += 1
                    if missing:
                        reason = f"pivot found but not mapped to {missing} (mapped to {mapped})"
                    else:
                        mapped_ok += 1
                if case["pivot"]:
                    tp += said_pivot
                    fn += not said_pivot
                else:
                    fp += said_pivot
                    tn += not said_pivot
                    ordinary += 1
                    ordinary_calls += calls
                    prefilter_hits += info.get("stage") != "prefilter"
                ok = not reason
                if not ok:
                    failures.append({"case": case["id"], "reason": reason,
                                     "prompt": case["prompt"]})
                tracer.event("case_pass" if ok else "case_fail", "verification",
                             reason or "as expected", case=case["id"], llm_calls=calls,
                             stage=info.get("stage"))
                rows.append({"case": case["id"], "ok": ok, "expected_pivot": case["pivot"],
                             "said_pivot": said_pivot, "stage": info.get("stage"),
                             "phrases": info.get("phrases"), "change": info.get("change"),
                             "mapped": mapped, "llm_calls": calls, "ms": round(ms, 1),
                             **({"reason": reason} if reason else {})})
    metrics = {"cases": len(cases), "passed": sum(r["ok"] for r in rows),
               "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
               "false_pivots": fp, "missed_pivots": fn,
               "llm_calls_per_ordinary_prompt": ratio(ordinary_calls, ordinary),
               "ordinary_prompts_reaching_llm": f"{prefilter_hits}/{ordinary}",
               "mapping_accuracy": ratio(mapped_ok, mapped_n),
               "llm_calls": client.calls, "llm_calls_cached": client.cached}
    return {"metrics": metrics, "cases": rows, "failures": failures}


# --- requirements ------------------------------------------------------------------------

def run_requirements(tracer, out=print) -> dict:
    from notbydefault.stale import require
    spec, cases = load("project"), load("requirements")["cases"]
    rows, failures = [], []
    correct = total = false_stale = missed_stale = 0
    with tempfile.TemporaryDirectory(prefix="nbd-eval-req-") as tmp, \
            env(NOTBYDEFAULT_ENTIRE="0", NOTBYDEFAULT_TELEMETRY="0"):
        base = Path(tmp) / "base"
        base.mkdir()
        keys = build_project(base, spec)
        client = make_client(base)
        for case in cases:
            project = Path(tmp) / case["id"]
            shutil.copytree(base, project)
            with tracer.step(f"case:{case['id']}", "verification", reason=case["requirement"]):
                before = client.calls
                lines: list[str] = []
                error = ""
                try:
                    require(project, case["requirement"], client=client, tracer=NoFlush(tracer),
                            out=lines.append)
                except Exception as e:
                    error = repr(e)
                calls = client.calls - before
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
                    total += 1
                    correct += ok
                    false_stale += status == "stale" and "stale" not in allowed
                    missed_stale += allowed == ["stale"] and status != "stale"
                    per[key] = {"expected": allowed, "got": status, "ok": ok,
                                "reason": got.get("reason", ""), "basis": got.get("basis", "")}
                    if not ok:
                        failures.append({
                            "case": case["id"], "decision": key,
                            "reason": f"expected {'/'.join(allowed)}, got {status}"
                                      + (f" ({got.get('basis')})" if got.get("basis") else "")
                                      + (f": {got.get('reason')}" if got.get("reason") else "")
                                      + (f" [{error}]" if error else "")})
                ok_case = all(p["ok"] for p in per.values())
                tracer.event("case_pass" if ok_case else "case_fail", "verification",
                             f"{sum(p['ok'] for p in per.values())}/{len(per)} decisions right",
                             case=case["id"], llm_calls=calls, constraints=constraints)
                rows.append({"case": case["id"], "ok": ok_case, "requirement": case["requirement"],
                             "constraints": constraints, "decisions": per, "llm_calls": calls,
                             **({"error": error} if error else {})})
    metrics = {"cases": len(cases), "passed": sum(r["ok"] for r in rows),
               "accuracy": ratio(correct, total), "decisions_right": f"{correct}/{total}",
               "false_stale": false_stale, "missed_stale": missed_stale,
               "llm_calls": client.calls, "llm_calls_cached": client.cached}
    return {"metrics": metrics, "cases": rows, "failures": failures}


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


def run_fit(tracer, out=print) -> dict:
    rows, failures = [], []
    for case in load("fit")["cases"]:
        project = (REPO_ROOT / case["project"]).resolve()
        with tracer.step(f"case:{case['id']}", "verification",
                         reason="read the saved fit-check verdict (no model calls)"):
            st = fit_status(project, case["decision_id"])
            row = {"case": case["id"], "decision_id": case["decision_id"], **st}
            if st["status"] == "pending":
                row["needs"] = case.get("needs")
                row["resume"] = case.get("resume")
                failures.append({"case": case["id"], "pending": True,
                                 "reason": f"pending: {st['reason']}. Needs {case.get('needs')}. "
                                           f"Then run `{case.get('resume')}` and "
                                           "`notbydefault eval --sections fit`."})
            tracer.event("fit_" + st["status"], "verification",
                         st.get("summary") or st.get("reason", ""), case=case["id"])
            rows.append(row)
    metrics = {"cases": len(rows), "final": sum(r["status"] == "final" for r in rows),
               "pending": sum(r["status"] == "pending" for r in rows)}
    return {"metrics": metrics, "cases": rows, "failures": failures}


RUNNERS = {"detection": run_detection, "pivot": run_pivot,
           "requirements": run_requirements, "fit": run_fit}


# --- report ------------------------------------------------------------------------------

def table(results: dict) -> list[str]:
    s = results["sections"]
    rows = [("section", "metric", "value")]

    def add(sec, keys):
        if sec not in s:
            return
        m = s[sec]["metrics"]
        for k in keys:
            rows.append((sec, k, "-" if m.get(k) is None else str(m.get(k))))
        rows.append((sec, "trace id", s[sec].get("trace_id") or "-"))
    add("detection", ["cases", "passed", "precision", "recall", "interruptions",
                      "false_interruptions", "false_interruption_rate"])
    add("pivot", ["cases", "passed", "precision", "recall", "false_pivots", "missed_pivots",
                  "llm_calls_per_ordinary_prompt", "ordinary_prompts_reaching_llm",
                  "mapping_accuracy"])
    add("requirements", ["cases", "passed", "accuracy", "decisions_right", "false_stale",
                         "missed_stale", "llm_calls"])
    if "fit" in s:
        for r in s["fit"]["cases"]:
            val = (f"{r['verdict']}: {r['summary']}" if r["status"] == "final"
                   else f"PENDING (provisional: {r.get('provisional_verdict') or '-'})")
            rows.append(("fit", f"decision #{r['decision_id']}", val[:110]))
        rows.append(("fit", "trace id", s["fit"].get("trace_id") or "-"))
    w = [max(len(r[i]) for r in rows) for i in range(2)]
    lines = [f"{a:<{w[0]}}  {b:<{w[1]}}  {c}" for a, b, c in rows]
    lines.insert(1, "-" * min(120, max(len(l) for l in lines)))
    return lines


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def run(version: str = "v1", sections: list[str] | None = None, out_path: Path | None = None,
        out=print) -> int:
    from notbydefault.fit.tracing import make_tracer
    sections = sections or list(SECTIONS)
    out_path = out_path or EVAL_DIR / "results" / f"{version}.json"
    results = {"version": version, "sections": {}}
    if out_path.exists() and set(sections) != set(SECTIONS):   # update some sections in place
        results = json.loads(out_path.read_text(encoding="utf-8"))
    results.update(run_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   commit=git_commit())
    tracer = make_tracer("notbydefault-eval-" + version)
    for name in sections:
        out(f"running {name} ...")
        started = time.perf_counter()
        with tracer.root(f"notbydefault.eval.{name}", {"notbydefault.eval_version": version,
                                                       "notbydefault.eval_section": name}):
            trace_id = tracer.trace_id
            section = RUNNERS[name](tracer, out)
            tracer.annotate({f"notbydefault.eval.{k}": v for k, v in section["metrics"].items()})
        section["trace_id"] = trace_id
        section["seconds"] = round(time.perf_counter() - started, 1)
        results["sections"][name] = section
        out(f"  {name}: {section['metrics'].get('passed', section['metrics'].get('final'))}/"
            f"{section['metrics']['cases']} in {section['seconds']}s")
        out_path.parent.mkdir(parents=True, exist_ok=True)   # save after every section
        out_path.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    tracer.flush()

    out("")
    out("\n".join(table(results)))
    out("")
    fails = [(name, f) for name in SECTIONS if name in results["sections"]
             for f in results["sections"][name]["failures"]]
    out(f"Failures ({len(fails)}):")
    for name, f in fails:
        where = f["case"] + (f" / {f['decision']}" if f.get("decision") else "")
        out(f"  [{name}] {where}: {f['reason']}")
    out(f"\nSaved {out_path.relative_to(REPO_ROOT).as_posix() if out_path.is_relative_to(REPO_ROOT) else out_path}")
    return 0
