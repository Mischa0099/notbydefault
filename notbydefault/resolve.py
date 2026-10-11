"""`notbydefault resolve <id>`: take one decision from its verdict to a reviewable branch.

    plan -> investigate -> fit_check -> recommend -> (approval) -> branch -> generate_tests
    -> apply -> test -> verify -> record -> commit (only if the user says yes) -> stop

  - investigate: the decision's requirements, its affected code (Entire Graph), past decision
    records (its own and every decision sharing its model or files) and its failure bank
  - fit_check: reuses the latest fit check while the call site and verifiers are unchanged
    since it ran; otherwise runs `check` (which asks before spending; only with --apply)
  - investigate also lists the project tests that already exercise each call site and caller
    (scout.existing_tests, read from source)
  - generate_tests: writes tests/test_nbd_<id>_generated.py from the failure bank and reference
    cases (testgen: a fixed template, the model call mocked) and runs it on the old code; the
    test step runs it again on the new code, and both counts go in the record and PR
  - apply: edits every call site the decision owns. A site using the same model for another
    decision (its own record, possibly its own verdict) is listed and left alone; callers of
    a site are listed, not edited.
  - verify: the decision's verifiers run on the requests the NEW code makes (its model id and
    system prompt read back from the edited source). By default every request is answered
    from the fit check's response cache, so no request is made; --live calls the new model.

Rules: dry run unless --apply; never touches main (all edits on nbd/<id>-<slug>); never
pushes; commits only when the user answers yes. Every step is a neatlogs span with a
focus_area and nbd.resolve_step. A failing step is retried once with its error; if it fails
again everything this run wrote is restored (the diff is saved first), the branch is left clean
and the user is told why.
"""
from __future__ import annotations

import ast
import difflib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from notbydefault.fit.tracing import Tracer, make_tracer
from notbydefault.graph import Graph, db_path

# step -> (focus_area, reason): the order run() takes them in
STEPS = {
    "plan": ("planning", "decide what resolving this decision takes before touching anything"),
    "investigate": ("context", "requirements, affected code (Entire Graph), past records and "
                               "failure bank"),
    "fit_check": ("verification", "reuse the fit check if it still matches the code, else run it"),
    "recommend": ("planning", "turn the verdict into concrete edits and ask for approval"),
    "branch": ("state", "work on nbd/<id>-<slug>, never on main"),
    "generate_tests": ("verification", "tests from the failure bank and reference cases, run on "
                                       "the code before the edit"),
    "apply": ("execution", "edit every call site the decision owns"),
    "test": ("verification", "run the project's own tests on the new code"),
    "verify": ("verification", "run the decision's verifiers on the requests the new code makes"),
    "record": ("state", "update the decision record and write the PR description"),
    "commit": ("state", "commit only if the user says so; never push"),
}
PROTECTED_BRANCHES = {"main", "master"}
TEST_SKIP_DIRS = {".notbydefault", ".demo-run", "node_modules", ".venv", "venv", ".git",
                  "__pycache__"}


class Stop(Exception):
    """A deliberate end (dry run, nothing to do, declined): never retried or rolled back."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


class StepFailed(Exception):
    def __init__(self, step: str, error: str):
        super().__init__(f"{step}: {error}")
        self.step, self.error = step, error


class EditError(Exception):
    """The change can't be applied automatically at a call site."""


@dataclass
class ResolveOptions:
    apply: bool = False        # without it nothing is written: plan, evidence and diff only
    yes: bool = False          # approve the recommendation (and a needed fit check) unasked
    live: bool = False         # verify by calling the new model instead of the response cache
    max_cost: float = 1.0      # cap for a fit check or a --live verification
    test_timeout_s: float = 900


@dataclass
class Site:
    path: str
    symbol: str | None
    line: int | None
    via: str

    def describe(self) -> str:
        loc = f"{self.path}:{self.line}" if self.line else self.path
        return f"{loc} {self.symbol}" if self.symbol else loc


@dataclass
class FileEdit:
    path: str
    before: str
    after: str
    sites: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def diff(self) -> str:
        return "".join(difflib.unified_diff(self.before.splitlines(True),
                                            self.after.splitlines(True),
                                            f"a/{self.path}", f"b/{self.path}"))


# --- git --------------------------------------------------------------------------------

class GitError(Exception):
    pass


def git(project: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=str(project), capture_output=True, timeout=60)
    out = proc.stdout.decode("utf-8", "replace").rstrip()
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip() or out
        raise GitError(f"git {' '.join(args)}: {err.splitlines()[-1] if err else 'failed'}")
    return out


def read_raw(path: Path) -> str:
    """File text with its line endings as they are (CRLF files stay CRLF)."""
    return Path(path).read_bytes().decode("utf-8")


def write_raw(path: Path, text: str) -> None:
    Path(path).write_bytes(text.encode("utf-8"))


def current_branch(project: Path) -> str:
    return git(project, "rev-parse", "--abbrev-ref", "HEAD")


def branch_name(decision_id: int, target: str) -> str:
    native = target.split(":", 1)[-1]
    slug = re.sub(r"[^a-z0-9]+", "-", f"switch-to-{native}".lower()).strip("-")[:40].strip("-")
    return f"nbd/{decision_id}-{slug}"


# --- call-site edits --------------------------------------------------------------------

def _function(tree: ast.Module, symbol: str | None):
    """The (async) function or method `symbol` ("f" or "Class.method"); the module if None."""
    if not symbol:
        return tree
    parts, scope = symbol.split("."), tree.body
    node = None
    for i, name in enumerate(parts):
        node = next((n for n in scope if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                                         ast.ClassDef)) and n.name == name), None)
        if node is None:
            return None
        scope = node.body
    return node


def _module_constants(tree: ast.Module) -> dict[str, ast.Assign]:
    return {t.id: n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
            for t in n.targets if isinstance(t, ast.Name)
            and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)}


def _span(node) -> tuple[int, int, int, int]:
    return node.lineno, node.col_offset, node.end_lineno, node.end_col_offset


def _replace_spans(text: str, spans: list[tuple[tuple[int, int, int, int], str]]) -> str:
    """Replace single-line (line, col, end_line, end_col) spans; columns are UTF-8 offsets.

    A span inside another one being rewritten (a constant's value inside its rewritten
    definition) is dropped; partly overlapping spans are refused."""
    def inside(a, b):     # a strictly within b, or equal
        return (b[0], b[1]) <= (a[0], a[1]) and (a[2], a[3]) <= (b[2], b[3])
    unique = list(dict(spans).items())
    keep = []
    for s, new in unique:
        width = (s[0], s[1]) != (s[2], s[3])
        if width and any(o != s and (o[0], o[1]) != (o[2], o[3]) and inside(s, o)
                         for o, _ in unique):
            continue
        keep.append((s, new))
    for i, (a, _) in enumerate(keep):
        for b, _ in keep[i + 1:]:
            if (a[0], a[1]) < (b[2], b[3]) and (b[0], b[1]) < (a[2], a[3]):
                raise EditError(f"line {a[0]}: overlapping edits can't be applied safely")
    lines = text.splitlines(True)
    for (ln, col, eln, ecol), new in sorted(keep, key=lambda s: s[0], reverse=True):
        if ln != eln:
            raise EditError(f"line {ln}: a multi-line expression can't be rewritten safely")
        raw = lines[ln - 1].encode("utf-8")
        lines[ln - 1] = (raw[:col] + new.encode("utf-8") + raw[ecol:]).decode("utf-8")
    return "".join(lines)


def _name_for(old: str, old_provider: str, new_provider: str) -> str:
    up_old, up_new = old_provider.upper(), re.sub(r"\W", "_", new_provider.upper())
    return old.replace(up_old, up_new) if up_old in old else f"{up_new}_BASE_URL"


def edit_file(text: str, path: str, sites: list[Site], old: tuple[str, str], new: tuple[str, str],
              providers: dict) -> tuple[str, list[str]]:
    """Switch every site in one file from model `old` to `new` ((provider, native) each).

    Inside each site's function: the model literal becomes the new native id; when the
    provider changes, the old provider's base URL (a literal or a module constant used only
    by the edited sites) and API-key variable (code and comments) become the new provider's.
    Raises EditError when that can't be done completely and safely."""
    (old_p, old_m), (new_p, new_m) = old, new
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        raise EditError(f"{path} does not parse: {e}") from e
    consts = _module_constants(tree)
    old_cfg, new_cfg = providers.get(old_p) or {}, providers.get(new_p) or {}
    switch_provider = old_p != new_p
    if switch_provider and not (new_cfg.get("base_url") and new_cfg.get("key_env")):
        raise EditError(f"provider {new_p!r} has no base_url/key_env in the provider catalog")
    spans, notes, ranges, url_names = [], [], [], set()
    for site in sites:
        fn = _function(tree, site.symbol)
        if fn is None:
            raise EditError(f"{path}: {site.symbol} not found (the code changed since the graph "
                            "was read)")
        nodes = list(ast.walk(fn))
        ranges.append((fn.lineno, fn.end_lineno) if site.symbol else (1, len(text.splitlines())))
        literals = [n for n in nodes if isinstance(n, ast.Constant) and n.value == old_m]
        names = {n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        model_consts = [c for c in names if c in consts and consts[c].value.value == old_m]
        if not literals and not model_consts:
            raise EditError(f"{site.describe()}: {old_m!r} no longer appears there")
        spans += [(_span(n), json.dumps(new_m)) for n in literals]
        for c in model_consts:
            spans.append((_span(consts[c].value), json.dumps(new_m)))
            notes.append(f"module constant {c} now names {new_m!r}")
        if not switch_provider:
            continue
        url_lits = [n for n in nodes if isinstance(n, ast.Constant)
                    and n.value == old_cfg.get("base_url")]
        url_consts = {c for c in names if c in consts
                      and consts[c].value.value == old_cfg.get("base_url")}
        key_lits = [n for n in nodes if isinstance(n, ast.Constant)
                    and n.value == old_cfg.get("key_env")]
        if not (url_lits or url_consts) or not key_lits:
            raise EditError(
                f"{site.describe()}: the {old_p} client is not created here (no "
                f"{'base URL' if not (url_lits or url_consts) else old_cfg.get('key_env')} in "
                f"the function), so switching it to {new_p} would also change other code")
        spans += [(_span(n), json.dumps(new_cfg["base_url"])) for n in url_lits]
        spans += [(_span(n), json.dumps(new_cfg["key_env"])) for n in key_lits]
        url_names |= url_consts
        spans += [(_span(n), _name_for(n.id, old_p, new_p)) for n in nodes
                  if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in url_consts]
    for c in sorted(url_names):
        new_name = _name_for(c, old_p, new_p)
        uses = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == c
                and isinstance(n.ctx, ast.Load)
                and not any(a <= n.lineno <= b for a, b in ranges)]
        if new_name in consts:
            raise EditError(f"{path}: {new_name} already exists")
        assign = consts[c]
        if uses:    # still used elsewhere: keep it, add the new constant below it
            indent, nl = " " * assign.col_offset, "\r\n" if "\r\n" in text else "\n"
            spans.append(((assign.end_lineno, assign.end_col_offset, assign.end_lineno,
                           assign.end_col_offset),
                          f"{nl}{indent}{new_name} = {json.dumps(new_cfg['base_url'])}"))
            notes.append(f"added {new_name}; {c} is still used elsewhere in {path}")
        else:
            spans.append((_span(assign), f"{new_name} = {json.dumps(new_cfg['base_url'])}"))
            notes.append(f"{c} replaced by {new_name} (no other use in {path})")
    after = _replace_spans(text, spans)
    if switch_provider:   # comments that name the old key ("starts without GROQ_API_KEY")
        old_key, new_key = old_cfg["key_env"], new_cfg["key_env"]
        lines = after.splitlines(True)
        for a, b in ranges:
            for i in range(a - 1, min(b, len(lines))):
                lines[i] = re.sub(rf"\b{re.escape(old_key)}\b", new_key, lines[i])
        after = "".join(lines)
    return after, notes


def call_contract(path: Path, symbol: str | None) -> dict:
    """What the call site sends, read back from source: model ids, base URLs, key names."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    consts = {k: v.value.value for k, v in _module_constants(tree).items()}
    fn = _function(tree, symbol)
    if fn is None:
        raise EditError(f"{path.name}: {symbol} not found after the edit")
    found = {"model": set(), "base_url": set(), "key_env": set()}
    for n in ast.walk(fn):
        if isinstance(n, ast.keyword) and n.arg in ("model", "base_url"):
            v = n.value
            value = v.value if isinstance(v, ast.Constant) else consts.get(getattr(v, "id", ""))
            if isinstance(value, str):
                found[n.arg].add(value)
        if isinstance(n, ast.Subscript) and "environ" in ast.unparse(n.value) \
                and isinstance(n.slice, ast.Constant):
            found["key_env"].add(n.slice.value)
        if isinstance(n, ast.Call) and ast.unparse(n.func).endswith(("environ.get", "getenv")) \
                and n.args and isinstance(n.args[0], ast.Constant):
            found["key_env"].add(n.args[0].value)
    return {k: sorted(v) for k, v in found.items()}


# --- the agent ----------------------------------------------------------------------------

class Resolver:
    def __init__(self, project: Path, decision_id: int, opts: ResolveOptions,
                 tracer: Tracer | None = None, ask=input, out=print, client=None, entire=None):
        self.project = Path(project).resolve()
        self.decision_id = decision_id
        self.opts = opts
        self.tracer = tracer
        self.ask, self.out = ask, out
        self.client = client          # ProviderSet-like; None -> built from the catalog
        self.entire = entire
        self.state: dict = {"steps": []}
        self.snapshots: dict[Path, str | None] = {}   # file -> text before this run (None: new)
        self.node_before: dict | None = None
        self.start_branch: str | None = None
        self.branch: str | None = None
        self.created_branch = False

    # --- plumbing ---------------------------------------------------------------------

    def _say(self, msg: str = "") -> None:
        self.out(msg)

    def _ask_yes(self, question: str) -> bool:
        try:
            return self.ask(question).strip().lower() in ("y", "yes")
        except EOFError:
            return False

    def _step(self, name: str, fn):
        """Run one step as its own span; retry once with the error, then fail."""
        focus, reason = STEPS[name]
        error = None
        for attempt in (1, 2):
            with self.tracer.step(name, focus, reason=reason, attrs={"attempt": attempt}):
                self.tracer.annotate({"nbd.resolve_step": name, "nbd.resolve_attempt": attempt,
                                      "notbydefault.decision_id": self.decision_id})
                try:
                    result = fn(error)
                    self.state["steps"].append({"step": name, "focus_area": focus,
                                                "attempts": attempt, "status": "ok"})
                    return result
                except Stop as s:
                    self.state["steps"].append({"step": name, "focus_area": focus,
                                                "attempts": attempt, "status": f"stop: {s}"})
                    self.tracer.annotate({"nbd.resolve_outcome": "stop"})
                    raise
                except Exception as e:     # noqa: BLE001  (every failure gets one retry)
                    error = f"{type(e).__name__}: {e}"
                    self.tracer.annotate({"nbd.resolve_error": error})
                    if attempt == 1:
                        self.tracer.event("retry_step", "recovery",
                                          f"{name} failed ({error}); retrying once with the error",
                                          step=name)
                        self._say(f"  ! {name} failed: {error}. Retrying once.")
        self.state["steps"].append({"step": name, "focus_area": focus, "attempts": 2,
                                    "status": f"failed: {error}"})
        self.tracer.event("step_failed", "recovery", f"{name} failed twice: {error}", step=name)
        raise StepFailed(name, error)

    def _snapshot(self, path: Path) -> None:
        """Remember a file as it was before this run first wrote it (None: it didn't exist)."""
        path = Path(path)
        if path not in self.snapshots:
            self.snapshots[path] = read_raw(path) if path.exists() else None

    def _write(self, path: Path, text: str) -> None:
        self._snapshot(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        write_raw(path, text)

    def _rel(self, path: Path) -> str:
        try:
            return Path(path).resolve().relative_to(self.project).as_posix()
        except ValueError:
            return str(path)

    # --- run ----------------------------------------------------------------------------

    def run(self) -> int:
        if not db_path(self.project).exists():
            self._say(f"No Decision Graph at {db_path(self.project)}.")
            return 1
        with Graph.for_project(self.project) as g:
            d = g.get_node(self.decision_id)
        if d is None or d["type"] != "decision":
            self._say(f"No decision #{self.decision_id} in {self.project}")
            return 1
        self.tracer = self.tracer or make_tracer(d.get("session_id"))
        mode = "apply" if self.opts.apply else "dry-run"
        code = 0
        with self.tracer.root("notbydefault.resolve", {
                "notbydefault.decision_id": self.decision_id, "nbd.resolve_mode": mode,
                "notbydefault.literal": d["label"]},
                finish=lambda: {"nbd.resolve_outcome": self.state.get("outcome"),
                                "nbd.resolve_branch": self.branch}):
            try:
                for name in STEPS:
                    if name == "branch" and not self.opts.apply:
                        break
                    self._step(name, getattr(self, f"step_{name}"))
                self.state["outcome"] = "resolved"
            except Stop as s:
                self.state["outcome"] = "stopped"
                code = s.code
                self._say(str(s))
            except StepFailed as f:
                self.state["outcome"] = f"failed at {f.step}"
                code = 3
                self._say(self._fail_message(f))
            except GitError as e:      # a branch/rollback git error outside a step
                self.state["outcome"] = "failed"
                code = 3
                self._say(f"Stopped: {e}")
        self.tracer.flush()
        return code

    # --- steps --------------------------------------------------------------------------

    def step_plan(self, error) -> None:
        with Graph.for_project(self.project) as g:
            d = g.get_node(self.decision_id)
        self.decision = d
        data = d["data"]
        if data.get("kind") != "model":
            raise Stop(2, f"Decision #{d['id']} is a {data.get('kind')} decision; resolve "
                          "only changes model decisions so far. Nothing changed.")
        plan = [f"Resolve decision #{d['id']}: {d['label']} ({data.get('category')}, "
                f"{data.get('status')})",
                "1. investigate: requirements, affected code (Entire Graph), past records, "
                "failure bank",
                "2. fit check: reuse the last one if the call site and verifiers are unchanged",
                "3. recommend: verdict -> edits at every call site this decision owns",
                ("4. on approval: branch nbd/<id>-<slug>, apply, run tests and verifiers, "
                 "write the record and PR description; commit only if you say yes"
                 if self.opts.apply else
                 "4. dry run: show the diff and stop (nothing is written; --apply to change "
                 "files)")]
        self.tracer.annotate({"notbydefault.plan": "\n".join(plan)})
        for line in plan:
            self._say(line if line.startswith("Resolve") else f"  {line}")

    def step_investigate(self, error) -> None:
        from notbydefault import records as records_mod
        from notbydefault.affected import Index, analyze
        from notbydefault.fit import failures as failures_mod
        from notbydefault.fit.run import graph_context
        from notbydefault.graph import affected_entries
        d = self.decision
        with Graph.for_project(self.project) as g:
            ctx = graph_context(g, d)
            if error is None:
                res = analyze(Index(self.project, self.entire), d)
                entries, source, note = res.entries, res.source, res.note
                if source != "entire-graph" and d["data"].get("affected_code"):
                    entries, source = affected_entries(d), "stored entire-graph analysis"
                    note += f"; using the analysis stored at {d['data'].get('affected_at')}"
            else:     # retry: the analysis stored on the decision by the last check/affected
                entries, source = affected_entries(d), "stored analysis (retry)"
                note = f"live Entire Graph read failed ({error})"
                self.tracer.event("affected_code_fallback", "recovery", note)
            others = [o for o in g.nodes("decision") if o["id"] != d["id"]
                      and (o["label"] == d["label"]
                           or set(o["data"].get("files") or []) & {e["path"] for e in entries})]
            past = []
            for o in others:
                rejects = [a["label"] for a in g.neighbors(o["id"], ("rejects",))]
                rec = records_mod.existing_path(self.project, o)
                past.append({"id": o["id"], "label": o["label"], "files": o["data"].get("files")
                             or [o["data"].get("file")], "status": o["data"].get("status"),
                             "verdict": o["data"].get("verdict"),
                             "recommended": o["data"].get("recommended"), "rejects": rejects,
                             "record": self._rel(rec) if rec else None})
            own_record = records_mod.existing_path(self.project, d)
            from notbydefault.briefing import build
            briefing = build(self.project, g, d, limit=None)
        bank = failures_mod.load(self.project, self.decision_id)
        by_model: dict[str, int] = {}
        for r in bank:
            by_model[r.get("model", "?")] = by_model.get(r.get("model", "?"), 0) + 1
        own_files = set(d["data"].get("files") or [d["data"].get("file")])
        direct, excluded, callers = [], [], []
        for e in entries:
            site = Site(e["path"], e.get("symbol"), e.get("line"), e.get("via", ""))
            if e.get("depth", 0) > 0:
                callers.append(site)
                continue
            owner = next((o for o in past if o["label"] == d["label"] and e["path"] in o["files"]
                          and e["path"] not in own_files), None)
            if owner:
                excluded.append((site, owner))
            else:
                direct.append(site)
        from notbydefault.scout import existing_tests
        covering = existing_tests(self.project, direct + callers)
        self.inv = {"ctx": ctx, "entries": entries, "source": source, "note": note,
                    "past": past, "own_record": self._rel(own_record) if own_record else None,
                    "briefing": briefing, "bank": bank, "bank_by_model": by_model,
                    "sites": direct, "excluded": excluded, "callers": callers,
                    "covering": covering}
        self.tracer.event("existing_tests", "context",
                          f"{sum(bool(v) for v in covering.values())} of {len(covering)} call "
                          "site(s) and caller(s) have project tests", covering=covering)
        self.tracer.event("affected_code", "context" if source == "entire-graph" else "recovery",
                          f"{len(direct)} call site(s) to change, {len(excluded)} owned by other "
                          f"decisions, {len(callers)} caller(s) ({source})",
                          sites=[s.describe() for s in direct])
        self._say(f"\nInvestigate")
        self._say(f"  goal: {ctx['goal'] or '-'}; constraints: {'; '.join(ctx['constraints']) or '-'}")
        self._say(f"  affected code ({source}{': ' + note if note else ''}):")
        for s in direct:
            self._say(f"    change  {s.describe():<40} {s.via}")
            self._say_tests(s)
        for s, o in excluded:
            self._say(f"    leave   {s.describe():<40} owned by decision #{o['id']}"
                      + (f" ({o['record']})" if o["record"] else ""))
        for s in callers:
            self._say(f"    caller  {s.describe():<40} {s.via} (not edited)")
            self._say_tests(s)
        self._say(f"  records: {self.inv['own_record'] or 'none yet'}"
                  + "".join(f"; #{p['id']} {p['label']} [{p['verdict'] or p['status']}]"
                            + (f" {p['record']}" if p['record'] else "") for p in past))
        self._say(f"  failure bank: {len(bank)} failing output(s)"
                  + (" (" + ", ".join(f"{m}: {n}" for m, n in sorted(by_model.items())) + ")"
                     if bank else ""))
        for line in briefing:
            self._say(f"  briefing: {line}")
        if not direct:
            raise Stop(2, f"No call site of decision #{self.decision_id} uses {d['label']!r} any "
                          "more; nothing to change.")

    def _say_tests(self, site: Site) -> None:
        found = self.inv["covering"].get(site.describe()) or []
        more = f" (+{len(found) - 4} more)" if len(found) > 4 else ""
        self._say(f"            tests: {', '.join(found[:4]) + more if found else 'none found'}")

    def step_fit_check(self, error) -> None:
        data, why = reusable_check(self.project, self.decision)
        if data is None:
            if not self.opts.apply:
                raise Stop(2, f"\nDry run: the fit check can't be reused ({why}). Run with --apply "
                              "to run it (it shows the cost and asks first). Nothing changed.")
            from notbydefault.fit.run import CheckOptions, check_model_decision
            self._say(f"\nFit check: {why}; running it"
                      + (f" again (last error: {error})" if error else ""))
            code = check_model_decision(self.project, self.decision_id,
                                        CheckOptions(max_cost=self.opts.max_cost,
                                                     yes=self.opts.yes),
                                        client=self.client, tracer=self.tracer, ask=self.ask,
                                        out=self.out)
            if code == 1:
                raise Stop(1, "Fit check not run; nothing changed.")
            if code != 0:
                raise RuntimeError(f"fit check exited {code}")
            with Graph.for_project(self.project) as g:
                self.decision = g.get_node(self.decision_id)
            data, why = reusable_check(self.project, self.decision)
            if data is None:
                raise RuntimeError(f"fit check ran but its results can't be used: {why}")
            self.tracer.event("fit_check_run", "verification", why)
        else:
            self.tracer.event("fit_check_reused", "verification", why,
                              results=self.decision["data"].get("results"),
                              trace_id=data.get("trace_id"))
            self._say(f"\nFit check: reusing {self.decision['data'].get('results')} ({why})")
        self.results = data

    def step_recommend(self, error) -> None:
        from notbydefault.fit.providers import load_catalog, split_id
        v = self.results["verdict"]
        cfg = self.results["config"]
        stats = self.results["stats"]
        current = stats.get(cfg["chosen"]) or {}
        model, setup = v["recommended"]
        self._say(f"  verdict: {v['kind']}: {v['summary']}")
        if v["kind"] == "keep":
            raise Stop(0, f"Recommendation: keep {cfg['chosen']}; nothing to change.")
        if v["kind"] != "switch":
            raise Stop(2, "Recommendation: none (the fit check had too little evidence); rerun "
                          f"`notbydefault check {self.decision_id}` with more cases. Nothing changed.")
        if setup != "alone":
            raise Stop(2, f"Recommendation: {model} with a verifier and retries; adding a verifier "
                          "loop to the code is not automated yet. Nothing changed.")
        rec = stats[model]
        catalog = getattr(self.client, "catalog", None) or load_catalog(self.project)
        providers = catalog.get("providers") or {}
        old, new = split_id(cfg["chosen"]), split_id(model)
        self.target, self.old_target = model, cfg["chosen"]
        self.branch = branch_name(self.decision_id, model)
        edits, by_file = [], {}
        for s in self.inv["sites"]:
            by_file.setdefault(s.path, []).append(s)
        for path, sites in by_file.items():
            if not path.endswith(".py"):
                raise EditError(f"{path}: only Python call sites can be edited automatically")
            before = read_raw(self.project / path)
            after, notes = edit_file(before, path, sites, old, new, providers)
            edits.append(FileEdit(path, before, after, [s.describe() for s in sites], notes))
        self.edits = edits
        self.providers = providers
        self._plan_tests(old, new)
        self._say(f"\nRecommend: switch {cfg['chosen']} -> {model}")
        self._say(f"  correct {correct_text(rec)}, hard {hard_text(rec)}, "
                  f"verifier pass {rec['pass_rate']:.0%}, "
                  f"${rec['monthly_cost']:.2f}/month vs ${current.get('monthly_cost', 0):.2f}, "
                  f"latency {rec['avg_latency_s']:.1f}s vs {current.get('avg_latency_s', 0):.1f}s "
                  "per call")
        for s, o in self.inv["excluded"]:
            warn = (f"; its own fit check REJECTED {model}" if model in o["rejects"] else "")
            self._say(f"  not changing {s.describe()}: it is decision #{o['id']}'s call site{warn}")
        if old[0] != new[0]:
            self._say(f"  provider changes {old[0]} -> {new[0]}: deployments need "
                      f"{providers[new[0]]['key_env']} set")
        self._say("")
        for e in edits:
            self._say(e.diff().rstrip("\n"))
            for n in e.notes:
                self._say(f"  ({n})")
        g = self.gen
        if g["skipped"]:
            self._say(f"\nGenerated tests: none ({g['skipped']})")
        else:
            self._say(f"\nGenerated tests: {len(g['cases'])} case(s) ({g['from_bank']} from the "
                      f"failure bank, {len(g['cases']) - g['from_bank']} reference) x "
                      f"{len(g['sites'])} call site(s) -> {g['path']}; run before and after the "
                      "edit")
        self.tracer.event("recommendation", "planning", f"switch {cfg['chosen']} -> {model}",
                          files=[e.path for e in edits], branch=self.branch)
        if not self.opts.apply:
            raise Stop(0, f"\nDry run: nothing changed. Re-run with --apply to make these edits on "
                          f"branch {self.branch}.")
        if not self.opts.yes and not self._ask_yes(
                f"\nApply on a new branch {self.branch}? [y/N] "):
            raise Stop(1, "Not approved; nothing changed.")

    def _plan_tests(self, old, new) -> None:
        from notbydefault import testgen
        cs = self.results["callsite"]
        try:
            tree = ast.parse(read_raw(self.project / cs["file"]))
        except (OSError, SyntaxError):
            tree = None
        # the top-level function around the fit-checked call: a whole-file site (no symbol,
        # e.g. from the files the agent touched) is tested through it
        enclosing = next((n.name for n in (tree.body if tree else [])
                          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                          and n.lineno <= cs["line"] <= n.end_lineno), None)
        primary, sites = None, []
        for s in self.inv["sites"]:
            if s.symbol is None and s.path == cs["file"] and enclosing:
                s = Site(s.path, enclosing, cs["line"], s.via)
            if not s.symbol or "." in s.symbol:
                continue
            fn = _function(tree, s.symbol) if tree and s.path == cs["file"] else None
            if fn is not None and fn.lineno <= cs["line"] <= fn.end_lineno:
                primary = s.describe()
            sites.append(s)
        cases = testgen.cases(self.project, self.decision_id, self.results["samples"])
        old_cfg, new_cfg = self.providers.get(old[0]) or {}, self.providers.get(new[0]) or {}
        self.gen = {"path": self._rel(testgen.test_path(self.project, self.decision_id)),
                    "cases": cases, "sites": sites, "primary": primary,
                    "from_bank": sum(c["source"] == "failure bank" for c in cases),
                    "expect": {"model": new[1], "base_url": new_cfg.get("base_url")
                               if old[0] != new[0] else None},
                    "keys": [k for k in (old_cfg.get("key_env"), new_cfg.get("key_env")) if k],
                    "skipped": ("no failure-bank or reference cases" if not cases else
                                "no call site is a module-level function" if not sites else None),
                    "before": None, "after": None}

    def step_branch(self, error) -> None:
        root = Path(git(self.project, "rev-parse", "--show-toplevel"))
        self.start_branch = current_branch(self.project)
        if self.start_branch != self.branch:
            dirty = git(self.project, "status", "--porcelain", "--untracked-files=no")
            if dirty:
                raise Stop(1, f"Uncommitted changes in {root}:\n{dirty}\nCommit or stash them first "
                              "(resolve must be able to restore a clean branch). Nothing changed.")
            exists = bool(git(self.project, "branch", "--list", self.branch))
            if exists:
                if git(self.project, "rev-parse", self.branch) != git(self.project, "rev-parse",
                                                                      "HEAD"):
                    raise Stop(1, f"Branch {self.branch} exists and differs from HEAD; delete or "
                                  "rename it first. Nothing changed.")
                git(self.project, "switch", self.branch)
            else:
                git(self.project, "switch", "-c", self.branch)
                self.created_branch = True
        if current_branch(self.project) in PROTECTED_BRANCHES:
            raise Stop(1, "Refusing to work on a protected branch.")
        self.tracer.event("branch", "state", f"working on {self.branch} (from "
                                             f"{self.start_branch})", branch=self.branch)
        self._say(f"\nBranch: {self.branch} (from {self.start_branch}; {self._rel(root) or root})")

    def step_generate_tests(self, error) -> None:
        from notbydefault import testgen
        g = self.gen
        if g["skipped"]:
            self.tracer.event("no_generated_tests", "verification", g["skipped"])
            self._say(f"Generated tests: none ({g['skipped']})")
            return
        path = self.project / g["path"]
        self._write(path, testgen.render(self.project, self.decision_id, self.target, g["sites"],
                                         g["primary"], g["cases"], g["expect"], g["keys"]))
        before = testgen.run(self.project, path, self.opts.test_timeout_s)
        if before["total"] == 0 or before["errors"]:
            raise RuntimeError(f"the generated tests did not run: {before['summary']}")
        g["before"] = before
        self.tracer.event("generated_tests", "verification",
                          f"{g['path']}: {testgen.counts_text(before)} on the code before the edit",
                          before=before["summary"])
        self._say(f"Generated tests: {g['path']}: {testgen.counts_text(before)} before the edit")

    def step_apply(self, error) -> None:
        import py_compile
        if error:   # start the retry from the original files
            for path, text in self.snapshots.items():
                if text is not None:
                    write_raw(path, text)
        if current_branch(self.project) != self.branch:
            raise RuntimeError(f"not on {self.branch}; refusing to edit")
        _, native = self.target.split(":", 1)
        for e in self.edits:
            path = self.project / e.path
            if read_raw(path) != e.before:
                raise EditError(f"{e.path} changed since the edit was planned")
            self._write(path, e.after)
            py_compile.compile(str(path), doraise=True, cfile=None)
        for s in self.inv["sites"]:
            c = call_contract(self.project / s.path, s.symbol)
            if c["model"] != [native]:
                raise EditError(f"{s.describe()}: sends model {c['model']} after the edit, "
                                f"expected [{native!r}]")
        self._say(f"Apply: {len(self.inv['sites'])} call site(s) in {len(self.edits)} file(s) now "
                  f"use {self.target}")

    def step_test(self, error) -> None:
        from notbydefault import testgen
        g = self.gen
        if g["before"] is not None:
            after = testgen.run(self.project, self.project / g["path"], self.opts.test_timeout_s)
            g["after"] = after
            self.tracer.annotate({"nbd.generated_before": testgen.counts_text(g["before"]),
                                  "nbd.generated_after": testgen.counts_text(after)})
            self._say(f"Generated tests: {testgen.counts_text(g['before'])} before, "
                      f"{testgen.counts_text(after)} after the edit")
            if after["failed"] or after["errors"] or not after["passed"]:
                raise RuntimeError(f"the generated tests fail on the new code: {after['summary']}")
        cmd = test_command(self.project)
        if cmd is None:
            self.tests = {"status": "none found", "command": None, "summary":
                          "no tests in the project (no tests/ dir, test_*.py or npm test script)"}
            self.tracer.event("no_project_tests", "verification", self.tests["summary"])
            self._say(f"Test: {self.tests['summary']}")
            return
        proc = subprocess.run(cmd, cwd=str(self.project), capture_output=True,
                              timeout=self.opts.test_timeout_s)
        tail = (proc.stdout.decode("utf-8", "replace") + proc.stderr.decode("utf-8", "replace")
                ).strip().splitlines()
        summary = tail[-1] if tail else f"exit {proc.returncode}"
        # pytest exit 5: no tests collected
        if proc.returncode not in (0, 5):
            raise RuntimeError(f"`{' '.join(cmd[-3:])}` exited {proc.returncode}: {summary}")
        self.tests = {"status": "passed" if proc.returncode == 0 else "none collected",
                      "command": " ".join(cmd), "summary": summary}
        self._say(f"Test: {self.tests['status']}: {summary}")

    def step_verify(self, error) -> None:
        from notbydefault.fit import failures as failures_mod
        from notbydefault.fit import verifiers as verifiers_mod
        from notbydefault.fit.callsite import CallSite, build_messages, source_system_prompt
        from notbydefault.fit.evaluate import Caller, SpendGuard
        from notbydefault.fit.reverify import CacheOnlyClient, production
        res = self.results
        provider, native = self.target.split(":", 1)
        pcfg = self.providers.get(provider) or {}
        problems = []
        for s in self.inv["sites"]:
            c = call_contract(self.project / s.path, s.symbol)
            if c["model"] != [native]:
                problems.append(f"{s.describe()} sends model {c['model']}")
            if c["base_url"] and c["base_url"] != [pcfg.get("base_url")]:
                problems.append(f"{s.describe()} uses base URL {c['base_url']}")
            if c["key_env"] and c["key_env"] != [pcfg.get("key_env")]:
                problems.append(f"{s.describe()} reads key {c['key_env']}")
        if problems:
            raise EditError("the new code does not call " + self.target + ": " + "; ".join(problems))
        cs = res["callsite"]
        prompt = source_system_prompt(self.project / cs["file"], cs["line"])
        if prompt is None:
            if cs.get("system_prompt_source") == "source":
                raise EditError(f"the system prompt can't be read from {cs['file']} any more")
            prompt = cs.get("system_prompt", "")
        site = CallSite(cs["file"], cs["line"], cs.get("snippet", ""), system_prompt=prompt)
        # this project's verifier file; the stored absolute path may point at another copy
        vpath = verifiers_mod.verifier_path(self.project, self.decision_id)
        if not vpath.exists():
            vpath = Path(res["verifiers"]["path"])
        verifiers = verifiers_mod.load(vpath)
        prod = production(verifiers)
        if self.client is None:
            from notbydefault.env import load_env
            from notbydefault.fit.providers import ProviderSet, load_catalog
            load_env()
            self.client = ProviderSet(load_catalog(self.project),
                                      cache_dir=self.project / ".notbydefault" / "cache")
        client = self.client if self.opts.live else CacheOnlyClient(self.client)
        caller = Caller(client, SpendGuard(self.opts.max_cost), self.tracer)
        cfg = res["config"]
        banked = set(failures_mod.inputs(self.project, self.decision_id))
        cases = sorted(res["samples"], key=lambda s: s["user_message"] not in banked)
        rows, errors = [], []
        for s in cases:
            try:
                reply = caller.chat(self.target, build_messages(site, s["user_message"]),
                                    span="notbydefault.model_call", role="candidate",
                                    temperature=cfg.get("temperature", 0.0),
                                    max_tokens=cfg.get("max_tokens", 1024),
                                    attrs={"notbydefault.sample": s.get("id")})
            except Exception as e:    # noqa: BLE001
                errors.append(f"{s.get('id')}: {e}")
                continue
            p = verifiers_mod.verify(caller, cfg.get("verifier_model", ""), prod,
                                     s["user_message"], reply.text)
            full = verifiers_mod.verify(caller, cfg.get("verifier_model", ""), verifiers,
                                        s["user_message"], reply.text, reference=s.get("reference"))
            rows.append({"id": s.get("id"), "banked": s["user_message"] in banked,
                         "production_pass": p.passed, "correct": full.passed,
                         "reasons": full.reasons, "cached": reply.cached})
        if errors:
            hint = ("" if self.opts.live else " (the new code's requests differ from the fit "
                    "check's; run with --live to call the new model)")
            raise RuntimeError(f"{len(errors)} of {len(cases)} cases could not be run{hint}: "
                               f"{errors[0]}")
        measured = res["stats"][self.target]
        n = len(rows)
        prod_pass = sum(r["production_pass"] for r in rows)
        correct = sum(r["correct"] for r in rows)
        expected = round(measured["pass_rate"] * measured["calls"])
        self.verification = {"cases": n, "production_pass": prod_pass, "correct": correct,
                             "banked": sum(r["banked"] for r in rows), "rows": rows,
                             "mode": "live" if self.opts.live else "response cache (no requests)",
                             "requests_made": getattr(self.client, "requests_made", None),
                             "checks": verifiers.names(), "path": self._rel(vpath),
                             "fit_check_pass": f"{expected}/{measured['calls']}"}
        self.tracer.annotate({"nbd.verify_production_pass": prod_pass, "nbd.verify_cases": n,
                              "nbd.verify_correct": correct})
        self._say(f"Verify: {prod_pass}/{n} pass the production checks, {correct}/{n} also match "
                  f"the reference ({self.verification['mode']}; fit check measured "
                  f"{self.verification['fit_check_pass']})")
        for r in rows:
            if not r["correct"]:
                self._say(f"    {r['id']}: {'; '.join(r['reasons'])[:160]}")
        if prod_pass < expected:
            raise RuntimeError(f"the new code passes the verifiers on {prod_pass}/{n} cases, below "
                               f"the {expected}/{measured['calls']} the fit check measured")

    def step_record(self, error) -> None:
        from notbydefault import records as records_mod
        now = datetime.now(timezone.utc)
        with Graph.for_project(self.project) as g:
            if self.node_before is None:
                self.node_before = g.get_node(self.decision_id)["data"]
            resolution = {"at": now.isoformat(timespec="seconds"), "branch": self.branch,
                          "from": self.old_target, "to": self.target,
                          "files": [e.path for e in self.edits],
                          "sites": [s.describe() for s in self.inv["sites"]],
                          "left_alone": [f"{s.describe()} (decision #{o['id']})"
                                         for s, o in self.inv["excluded"]],
                          "tests": self.tests["status"],
                          "generated_tests": self._generated_summary(),
                          "verifiers": f"{self.verification['production_pass']}/"
                                       f"{self.verification['cases']} production, "
                                       f"{self.verification['correct']}/"
                                       f"{self.verification['cases']} vs reference",
                          "trace_id": self.tracer.trace_id}
            g.update_data(self.decision_id, resolution=resolution)
            record = records_mod.existing_path(self.project, g.get_node(self.decision_id))
            if record is not None:
                self._snapshot(record)
            paths = records_mod.write_records(
                self.project, g, [self.decision_id],
                f"resolved on branch {self.branch}: {self.old_target} -> {self.target} at "
                f"{len(self.inv['sites'])} call site(s); tests {self.tests['status']}; verifiers "
                f"{resolution['verifiers']}", self.tracer)
            if record is None:
                self.snapshots.setdefault(paths[0], None)
            self.record_path = paths[0]
            self.decision = g.get_node(self.decision_id)
        folder = self.project / ".notbydefault" / "resolve"
        stem = f"{self.decision_id}-{now.strftime('%Y%m%dT%H%M%SZ')}"
        self.pr_path = folder / f"{stem}-pr.md"
        self._write(self.pr_path, pr_description(self))
        self._write(folder / f"{stem}-journal.json", json.dumps(
            {"decision_id": self.decision_id, "branch": self.branch,
             "trace_id": self.tracer.trace_id, "steps": self.state["steps"],
             "verification": self.verification, "tests": self.tests,
             "journal": getattr(self.tracer, "journal", [])}, indent=1, default=str))
        self._say(f"Record: {self._rel(self.record_path)} updated; PR description: "
                  f"{self._rel(self.pr_path)}")

    def _generated_summary(self) -> dict:
        from notbydefault import testgen
        g = self.gen
        if g["before"] is None:
            return {"path": None, "note": g["skipped"]}
        return {"path": g["path"], "cases": len(g["cases"]), "from_failure_bank": g["from_bank"],
                "sites": [s.describe() for s in g["sites"]],
                "before": testgen.counts_text(g["before"]),
                "after": testgen.counts_text(g["after"])}

    def step_commit(self, error) -> None:
        files = [e.path for e in self.edits] + [self._rel(self.record_path)]
        if self.gen["before"] is not None:
            files.append(self.gen["path"])
        status = git(self.project, "status", "--porcelain", "--", *files)
        self._say(f"\nChanged on {self.branch} (not committed, not pushed):\n{status}")
        if not self._ask_yes(f"Commit these {len(files)} file(s) on {self.branch}? [y/N] "):
            self.tracer.event("commit_declined", "state", "left uncommitted for review")
            self._say(f"Left uncommitted on {self.branch}. Review with `git diff`; the PR "
                      f"description is {self._rel(self.pr_path)}. Nothing was pushed.")
            return
        if current_branch(self.project) != self.branch:
            raise RuntimeError(f"not on {self.branch}; refusing to commit")
        git(self.project, "add", "--", *files)
        msg = (f"Resolve #{self.decision_id}: switch to {self.target}\n\n"
               f"{self.results['verdict']['summary']}\n\nnotbydefault resolve trace: "
               f"{self.tracer.trace_id or 'not exported'}; fit check trace: "
               f"{self.results.get('trace_id') or '-'}\n")
        git(self.project, "commit", "-q", "-m", msg)
        sha = git(self.project, "rev-parse", "HEAD")
        self.state["commit"] = sha
        self._say(f"Committed {sha[:12]} on {self.branch}. Not pushed.")

    # --- failure ------------------------------------------------------------------------

    def _rollback(self) -> list[str]:
        """Undo every file this run wrote; save the undone diff first. Returns notes."""
        notes = []
        if not self.snapshots:
            return notes
        patch = "".join(
            "".join(difflib.unified_diff((before or "").splitlines(True),
                                         read_raw(p).splitlines(True),
                                         f"a/{self._rel(p)}", f"b/{self._rel(p)}"))
            for p, before in self.snapshots.items() if p.exists())
        save = self.project / ".notbydefault" / "resolve" / f"{self.decision_id}-failed.patch"
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_text(patch, encoding="utf-8")
        for path, before in self.snapshots.items():
            if before is None:
                path.unlink(missing_ok=True)
            else:
                write_raw(path, before)
        notes.append(f"restored {len(self.snapshots)} file(s); the undone diff is in "
                     f"{self._rel(save)}")
        if self.node_before is not None:
            with Graph.for_project(self.project) as g:
                g.conn.execute("UPDATE nodes SET data_json = ? WHERE id = ?",
                               (json.dumps(self.node_before), self.decision_id))
                g.conn.commit()
        return notes

    def _fail_message(self, f: StepFailed) -> str:
        lines = [f"\nStopped at step '{f.step}' after one retry: {f.error}"]
        try:
            lines += [f"  {n}" for n in self._rollback()]
            if self.start_branch and current_branch(self.project) == self.branch:
                git(self.project, "reset", "-q")    # unstage a half-done commit's `git add`
                dirty = git(self.project, "status", "--porcelain", "--untracked-files=no")
                if dirty:
                    lines.append(f"  WARNING: still modified after restoring:\n{dirty}")
                elif self.start_branch != self.branch:
                    git(self.project, "switch", self.start_branch)
                    lines.append(f"  branch {self.branch} is clean (no commits, no changes); "
                                 f"switched back to {self.start_branch}")
        except (GitError, OSError) as e:
            lines.append(f"  could not finish restoring: {e}; check `git status`")
        if f.step in ("investigate", "fit_check", "recommend"):
            lines.append("  nothing was changed")
        return "\n".join(lines)


def reusable_check(project: Path, decision: dict) -> tuple[dict | None, str]:
    """(results, why) of the decision's last fit check if it still describes the code, else
    (None, why not)."""
    from notbydefault.fit import callsite as callsite_mod
    from notbydefault.fit import verifiers as verifiers_mod
    data = decision["data"]
    if data.get("status") == "stale":
        return None, f"decision is stale: {data.get('stale_reason')}"
    if not data.get("verdict") or not data.get("results"):
        return None, "no fit check yet"
    path = Path(project) / data["results"]
    if not path.is_file():
        return None, f"results file {data['results']} is missing"
    res = json.loads(path.read_text(encoding="utf-8"))
    try:
        site = callsite_mod.locate(Path(project), decision)
    except LookupError as e:
        return None, str(e)
    if site.snippet_sha != res["callsite"].get("snippet_sha"):
        return None, f"{site.file} changed since the fit check"
    vpath = verifiers_mod.verifier_path(Path(project), decision["id"])
    if vpath.exists() and verifiers_mod.load(vpath).sha != res["verifiers"].get("sha"):
        return None, f"{vpath.name} changed since the fit check"
    return res, (f"run {res['run_at'][:10]}, trace {res.get('trace_id') or '-'}; call site and "
                 "verifiers unchanged since")


def test_command(project: Path) -> list[str] | None:
    """The project's own test command: pytest when it has Python tests, else `npm test`."""
    from notbydefault.scout import test_files
    project = Path(project)
    if test_files(project):
        return [sys.executable, "-m", "pytest", "-q"]
    pkg = project / "package.json"
    if pkg.is_file():
        try:
            if (json.loads(pkg.read_text(encoding="utf-8")).get("scripts") or {}).get("test"):
                return ["npm", "test", "--silent"]
        except json.JSONDecodeError:
            pass
    return None


def _links(r: Resolver) -> list[str]:
    """Entire checkpoint links of the commits that added/changed the decision record."""
    from notbydefault.checkpoints import checkpoint_of
    from notbydefault.entire import Entire, commit_link, repo_slug
    rel = r._rel(r.record_path)
    try:
        log = git(r.project, "log", "--format=%H%x09%s", "--", rel)
    except GitError:
        return []
    slug = repo_slug(r.project)
    out = []
    for line in log.splitlines()[:5]:
        sha, subject = line.split("\t", 1)
        cid = checkpoint_of(r.project, sha, r.entire or Entire())
        link = commit_link(slug, sha) if slug else f"(no GitHub origin) commit {sha}"
        out.append(f"| {subject[:70]} | {cid or '-'} | {link} |")
    return out


def correct_text(s: dict) -> str:
    """"10/12" against reference answers; "-" when the fit check had none."""
    return f"{round(s['correct_rate'] * s['n_ref'])}/{s['n_ref']}" if s and s.get("n_ref") else "-"


def hard_text(s: dict) -> str:
    return f"{s['hard_correct']}/{s['hard_total']}" if s and s.get("hard_total") else "-"


def pr_description(r: Resolver) -> str:
    res, v = r.results, r.results["verdict"]
    stats = res["stats"]
    cur, new = stats.get(r.old_target) or {}, stats[r.target]
    vf = r.verification
    old_p, new_p = r.old_target.split(":", 1)[0], r.target.split(":", 1)[0]
    pcfg = r.providers

    def both(label, key, fmt, source):
        old = format(cur[key], fmt) if key in cur else "-"
        return f"| {label} | {old} | {format(new[key], fmt)} | {source} |"

    correct, hard = correct_text, hard_text

    def residency(p):
        return ", ".join(pcfg.get(p, {}).get("data_residency") or []) or "none guaranteed"
    budget = res["assumptions"].get("budget")
    lines = [f"# Resolve #{r.decision_id}: switch {r.old_target} -> {r.target}", "",
             f"Decision #{r.decision_id} (`{r.decision['label']}`), task: "
             f"{res['callsite'].get('task') or '-'}", "", "## Why", "", v["summary"], "",
             "## Changes", ""]
    for e in r.edits:
        lines.append(f"- `{e.path}`: {', '.join(e.sites)}" + "".join(f"; {n}" for n in e.notes))
    for s, o in r.inv["excluded"]:
        lines.append(f"- not changed: `{s.describe()}`, decision #{o['id']}'s call site"
                     + (f" (its fit check rejected `{r.target}`)" if r.target in o["rejects"]
                        else ""))
    for s in r.inv["callers"]:
        lines.append(f"- not changed: caller `{s.describe()}` ({s.via})")
    lines += ["", "## Existing tests", "", "| | call site / caller | project tests |",
              "|---|---|---|"]
    for kind, sites in (("change", r.inv["sites"]), ("caller", r.inv["callers"])):
        for s in sites:
            found = r.inv["covering"].get(s.describe()) or []
            lines.append(f"| {kind} | `{s.describe()}` | "
                         + (", ".join(f"`{t}`" for t in found) or "none found") + " |")
    if old_p != new_p:
        lines.append(f"- provider {old_p} -> {new_p}: set `{pcfg[new_p]['key_env']}` in every "
                     f"deployment (`{pcfg[old_p]['key_env']}` is no longer read here)")
    lines += ["", "## Evidence", "", "| | current | new | source |", "|---|---|---|---|",
              f"| model | `{r.old_target}` | `{r.target}` | fit check verdict: {v['kind']} |",
              f"| correct vs reference | {correct(cur)} | {correct(new)} | "
              f"`{r.decision['data'].get('results')}` |",
              f"| hard cases | {hard(cur)} | {hard(new)} | fit check |",
              both("verifier pass (fit check)", "pass_rate", ".0%", f"`{vf['path']}`"),
              both("cost per call ($)", "avg_cost", ".6f", res["pricing"]["source"]),
              both("monthly at assumed volume ($)", "monthly_cost", ".2f",
                   f"budget ${budget:.2f}" if budget else "no budget recorded"),
              both("latency per call (s)", "avg_latency_s", ".1f", "fit check"),
              f"| data residency | {residency(old_p)} | {residency(new_p)} | provider catalog |",
              f"| verifiers on the new code | | {vf['production_pass']}/{vf['cases']} production, "
              f"{vf['correct']}/{vf['cases']} vs reference ({vf['banked']} failure-bank cases "
              f"first) | {vf['mode']} |",
              f"| project tests | | {r.tests['status']} | {r.tests['command'] or r.tests['summary']} |",
              ""]
    gs = r._generated_summary()
    lines += ["## Generated tests", ""]
    if gs["path"]:
        lines += [f"`{gs['path']}`: {gs['cases']} case(s) ({gs['from_failure_bank']} from the "
                  "failure bank, the rest reference cases) at "
                  + ", ".join(f"`{s}`" for s in gs["sites"])
                  + f". Deterministic template, model call mocked: each case must reach "
                  f"`{r.target}` and, fed its reference answer, return it.", "",
                  "| | before the edit | after the edit |", "|---|---|---|",
                  f"| generated tests | {gs['before']} | {gs['after']} |", ""]
    else:
        lines += [f"none: {gs['note']}", ""]
    claims = r.decision["data"].get("claims")
    if claims:
        from notbydefault.claims import block_lines
        head, *rest = block_lines(claims, markdown=True)
        lines += ["## Claim check", "", head, "", *rest, ""]
    lines += ["## Traces", "",
              f"- resolve: `{r.tracer.trace_id or 'not exported (telemetry off)'}`",
              f"- fit check: `{res.get('trace_id') or '-'}`", "",
              "## Checkpoints", "", "| commit | checkpoint | link |", "|---|---|---|"]
    links = _links(r)
    lines += links or ["| decision record not committed yet | - | - |"]
    lines += ["", "## Decision record", "", f"`{r._rel(r.record_path)}`", "",
              "## Steps", "", "| step | focus_area | attempts | status |", "|---|---|---|---|"]
    lines += [f"| {s['step']} | {s['focus_area']} | {s['attempts']} | {s['status']} |"
              for s in r.state["steps"]] + [f"| record | state | - | this file |", ""]
    return "\n".join(lines)


def resolve(project: Path, decision_id: int, opts: ResolveOptions, **kw) -> int:
    return Resolver(project, decision_id, opts, **kw).run()
