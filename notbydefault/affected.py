"""Affected code: every file and function that uses a decision's model or service.

With entire-graph (one `symbols` + one `edges` stream per project, shared by all decisions):
  1. direct users: functions/methods whose body names the model literal, call into the
     service's SDK (CALLS -> external:symbol:<package>.*), or use a module-level name bound to
     it (`client = MongoClient(...)`, `notes = db["notes"]`, `MODEL = "gpt-4o"`)
  2. files that import the SDK (IMPORTS -> external:import:<package>) and manifests that list it
  3. callers of the direct users, up to MAX_CALLER_DEPTH hops over local CALLS edges
Without entire-graph (not installed, not a git repo, or an error) it falls back to the files
the agent's tool calls touched, and says why.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from notbydefault.entire import Entire, repo_root

MAX_CALLER_DEPTH = 2
CODE_KINDS = {"function", "method", "class"}
SKIP_DIRS = (".notbydefault/", "decisions/", ".demo-run/", "node_modules/", ".venv/")
# PyPI distribution name -> import name, where they differ
IMPORT_NAMES = {"psycopg2-binary": "psycopg2", "google-generativeai": "google.generativeai",
                "google-genai": "google.genai", "mysqlclient": "MySQLdb",
                "firebase-admin": "firebase_admin", "groq-sdk": "groq-sdk",
                "cohere-ai": "cohere-ai", "tortoise-orm": "tortoise"}
MANIFESTS = re.compile(r"(^|/)(requirements[\w.-]*\.txt|package\.json|pyproject\.toml|Pipfile)$")


@dataclass
class Affected:
    source: str                         # "entire-graph" | "touched-files"
    entries: list[dict] = field(default_factory=list)
    note: str = ""

    @property
    def files(self) -> list[str]:
        seen: dict[str, None] = {}
        for e in self.entries:
            seen.setdefault(e["path"], None)
        return list(seen)

    def to_dict(self) -> dict:
        return asdict(self) | {"files": self.files}


def touched_files(decision: dict) -> list[str]:
    data = decision.get("data", {})
    if data.get("files"):
        return list(data["files"])
    return [data["file"]] if data.get("file") else []


def fallback(decision: dict, note: str) -> Affected:
    return Affected("touched-files", [{"path": f, "symbol": None, "kind": "file", "line": None,
                                       "via": "touched by the agent's tool call", "depth": 0}
                                      for f in touched_files(decision)], note)


# --- what to look for --------------------------------------------------------------

def markers(decision: dict) -> dict:
    """{"literal": model id or None, "packages": import names of the service's SDKs}."""
    from notbydefault.detect import KNOWN_SERVICES
    data = decision.get("data", {})
    if data.get("kind") == "model":
        return {"literal": decision["label"], "packages": []}
    chosen = decision["label"].lower()
    pkgs = {p for p, (name, _cat) in KNOWN_SERVICES.items() if name.lower() == chosen}
    for ev in str(data.get("evidence") or "").split(","):
        if ev.strip():
            pkgs.add(ev.strip().lower())
    names = set()
    for p in pkgs:
        names.add(p)
        names.add(IMPORT_NAMES.get(p, p.replace("-", "_")))
    return {"literal": None, "packages": sorted(names)}


def _pkg_match(target: str, packages: list[str]) -> str | None:
    for p in packages:
        if target == p or target.startswith(p + ".") or target.startswith(p + "/"):
            return p
    return None


# --- the project index (one graph read per project) -----------------------------------

class Index:
    """Symbols, call edges and file texts of a project, read once from entire-graph."""

    def __init__(self, project: Path, entire: Entire | None = None):
        self.project = Path(project).resolve()
        self.entire = entire or Entire()
        self.error = ""
        self.symbols: dict[str, dict] = {}
        self.calls: list[dict] = []
        self.imports: list[dict] = []
        self.files: list[str] = []
        self._text: dict[str, list[str]] = {}
        self.prefix = ""
        self.ok = self._load()

    def _load(self) -> bool:
        root = repo_root(self.project)
        if root is None:
            self.error = "not inside a git repository (entire-graph indexes git worktrees)"
            return False
        if root == Path.home().resolve() and self.project != root:
            self.error = (f"the enclosing git repository is your home directory ({root}); "
                          "indexing it would take too long. Run `git init` in the project")
            return False
        if not self.entire.has_plugin("graph"):
            self.error = ("entire-graph is not installed (" + (self.entire.last_error() or
                          "`entire plugin list` does not list it") + ")")
            return False
        try:
            rel = self.project.relative_to(root).as_posix()
        except ValueError:
            rel = ""
        self.prefix = "" if rel in ("", ".") else rel + "/"
        symbols = self.entire.graph_symbols(root)
        edges = self.entire.graph_edges(root, "CALLS,IMPORTS") if symbols is not None else None
        if symbols is None or edges is None:
            self.error = self.entire.last_error() or "entire-graph returned nothing"
            return False
        files: dict[str, None] = {}
        for s in symbols:
            path = self._local(s.get("file_path", ""))
            if path is None:
                continue
            files.setdefault(path, None)
            self.symbols[s["id"]] = dict(s, path=path)
        self.files = list(files)
        self.calls = [e for e in edges if e.get("type") == "CALLS"]
        self.imports = [e for e in edges if e.get("type") == "IMPORTS"]
        self.root = root
        return True

    def _local(self, repo_path: str) -> str | None:
        """Project-relative path for a repo-relative one, or None if outside the project."""
        p = repo_path.replace("\\", "/")
        if self.prefix and not p.startswith(self.prefix):
            return None
        p = p[len(self.prefix):]
        return None if p.startswith(SKIP_DIRS) else p

    def path_of(self, node_id: str) -> str | None:
        """Project-relative path of a symbol or file id from an edge."""
        if node_id in self.symbols:
            return self.symbols[node_id]["path"]
        m = re.match(r"[^:]+:(?:file:|[^:]+:)(.+?)(?::(?:function|method|class|section):.*)?$",
                     node_id)
        return self._local(m.group(1)) if m else None

    def lines(self, path: str) -> list[str]:
        if path not in self._text:
            try:
                self._text[path] = (self.project / path).read_text(
                    encoding="utf-8", errors="replace").splitlines()
            except OSError:
                self._text[path] = []
        return self._text[path]

    def code_symbols(self, path: str) -> list[dict]:
        return [s for s in self.symbols.values()
                if s["path"] == path and s.get("kind") in CODE_KINDS]

    def innermost(self, path: str, line: int) -> dict | None:
        """The function/method (not class) whose range contains `line`."""
        best = None
        for s in self.code_symbols(path):
            if s["kind"] == "class":
                continue
            if s["start_line"] <= line <= s["end_line"]:
                if best is None or s["end_line"] - s["start_line"] < best["end_line"] - best["start_line"]:
                    best = s
        return best


# --- the analysis --------------------------------------------------------------------

ASSIGN = re.compile(r"^\s*(?:export\s+)?(?:const\s+|let\s+|var\s+)?([A-Za-z_]\w*)\s*(?::[^=]+)?=\s*(.+)$")
PY_FROM = re.compile(r"^\s*from\s+([\w.]+)\s+import\s+\(?([\w\s,]+)\)?")
PY_IMPORT = re.compile(r"^\s*import\s+([\w.]+)(?:\s+as\s+(\w+))?")
JS_IMPORT = re.compile(r"""^\s*import\s+(.+?)\s+from\s+['"]([^'"]+)['"]""")
JS_REQUIRE = re.compile(r"""(?:const|let|var)\s+(\{[^}]*\}|\w+)\s*=\s*require\(\s*['"]([^'"]+)['"]""")


def _bound_names(line: str, packages: list[str]) -> set[str]:
    """Names an import line binds from one of the packages."""
    names: set[str] = set()
    m = PY_FROM.match(line)
    if m and _pkg_match(m.group(1), packages):
        names |= {n.split(" as ")[-1].strip() for n in m.group(2).split(",") if n.strip()}
    m = PY_IMPORT.match(line)
    if m and _pkg_match(m.group(1), packages):
        names.add(m.group(2) or m.group(1).split(".")[0])
    for rx in (JS_IMPORT, JS_REQUIRE):
        m = rx.search(line)
        if m and _pkg_match(m.group(2), packages):
            names |= set(re.findall(r"[A-Za-z_]\w*", m.group(1))) - {"as", "type"}
    return names


def _module_taint(index: Index, path: str, seeds: set[str], literal: str | None) -> set[str]:
    """Module-level names bound (transitively) to the service or the model literal."""
    lines = index.lines(path)
    inside = set()
    for s in index.code_symbols(path):
        if s["kind"] != "class":
            inside.update(range(s["start_line"], s["end_line"] + 1))
    tainted = set(seeds)
    changed = True
    while changed:
        changed = False
        for i, line in enumerate(lines, start=1):
            if i in inside:
                continue
            m = ASSIGN.match(line)
            if not m or m.group(1) in tainted:
                continue
            rhs = m.group(2)
            if (literal and _has_literal(rhs, literal)) or any(
                    re.search(rf"\b{re.escape(t)}\b", _code_only(rhs)) for t in tainted):
                tainted.add(m.group(1))
                changed = True
    return tainted


STRINGS = re.compile(r"""(?s)'''.*?'''|\"\"\".*?\"\"\"|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`[^`]*`""")


def _code_only(line: str) -> str:
    """The line without string literals and trailing comments (for name matching)."""
    line = STRINGS.sub('""', line)
    return re.split(r"\s#|^#|\s//", line, maxsplit=1)[0]


def _has_literal(text: str, literal: str) -> bool:
    return any(q + literal + q in text for q in ('"', "'", "`"))


def analyze(index: Index, decision: dict) -> Affected:
    if not index.ok:
        return fallback(decision, f"entire-graph unavailable: {index.error}; using the files "
                                  "the agent touched")
    mk = markers(decision)
    literal, packages = mk["literal"], mk["packages"]
    found: dict[tuple, dict] = {}

    def add(path, sym, via, depth, line=None):
        key = (path, sym["id"] if sym else None)
        if key in found and found[key]["depth"] <= depth:
            return
        found[key] = {"path": path, "symbol": sym["qualified_name"] if sym else None,
                      "kind": sym["kind"] if sym else "file",
                      "line": sym["start_line"] if sym else line, "via": via, "depth": depth,
                      "symbol_id": sym["id"] if sym else None}

    # files that import the SDK, and the names they bind
    importing: dict[str, set[str]] = {}
    for e in index.imports:
        target = e["to_id"].split("external:import:", 1)[-1] if "external:import:" in e["to_id"] else ""
        pkg = _pkg_match(target, packages) if target else None
        path = index.path_of(e["from_id"])
        if pkg and path:
            importing.setdefault(path, set())
            add(path, None, f"imports {target}", 0, 1)
    for path in list(importing):
        for line in index.lines(path):
            importing[path] |= _bound_names(line, packages)

    # direct users: SDK calls from the graph
    direct: dict[str, str] = {}
    for e in index.calls:
        to = e["to_id"]
        if not to.startswith("external:symbol:"):
            continue
        pkg = _pkg_match(to.split("external:symbol:", 1)[1], packages)
        sym = index.symbols.get(e["from_id"])
        if pkg and sym and sym.get("kind") in CODE_KINDS:
            direct[sym["id"]] = f"calls {to.split('external:symbol:', 1)[1]}"

    # direct users: the literal, or module-level names bound to the service/literal
    scan = set(importing) | ({p for p in index.files
                              if any(_has_literal(l, literal) for l in index.lines(p))}
                             if literal else set())
    for path in scan:
        tainted = _module_taint(index, path, importing.get(path, set()), literal)
        for i, line in enumerate(index.lines(path), start=1):
            sym = index.innermost(path, i)
            if sym is None:
                if literal and _has_literal(line, literal):
                    add(path, None, f"module-level use of {literal!r} (line {i})", 0, i)
                continue
            if sym["id"] in direct:
                continue
            if literal and _has_literal(line, literal):
                direct[sym["id"]] = f"uses {literal!r} (line {i})"
                continue
            if line.lstrip().startswith(("import ", "from ")):
                continue      # a local import is not a use; the call that follows is
            code = _code_only(line)
            hit = next((t for t in sorted(tainted) if re.search(rf"\b{re.escape(t)}\b", code)), None)
            if hit:
                direct[sym["id"]] = f"uses {hit} (line {i})"
    for sid, via in direct.items():
        sym = index.symbols[sid]
        add(sym["path"], sym, via, 0)

    # manifests that list the package
    if packages:
        for path in index.files:
            if MANIFESTS.search(path):
                text = "\n".join(index.lines(path)).lower()
                hit = next((p for p in packages
                            if re.search(rf"(^|[\s\"'@/]){re.escape(p.lower())}([\s\"'<>=~!\[;]|$)",
                                         text, re.M)), None)
                if hit:
                    add(path, None, f"lists {hit}", 0, 1)

    # callers of the direct users (reverse local CALLS edges)
    callers: dict[str, list[dict]] = {}
    for e in index.calls:
        if e["to_id"] in index.symbols:
            callers.setdefault(e["to_id"], []).append(e)
    frontier = list(direct)
    for depth in range(1, MAX_CALLER_DEPTH + 1):
        nxt = []
        for sid in frontier:
            callee = index.symbols[sid]["qualified_name"]
            for e in callers.get(sid, []):
                src = e["from_id"]
                sym = index.symbols.get(src)
                if sym and sym.get("kind") in CODE_KINDS:
                    if (sym["path"], sym["id"]) not in found:
                        add(sym["path"], sym, f"calls {callee}", depth)
                        nxt.append(src)
                else:
                    path = index.path_of(src)
                    if path and (path, None) not in found:
                        add(path, None, f"module-level call to {callee}", depth,
                            (e.get("evidence") or [{}])[0].get("start_line"))
        frontier = nxt

    entries = sorted(found.values(), key=lambda x: (x["depth"], x["path"], x["line"] or 0))
    if not entries:
        return fallback(decision, "entire-graph found no code using "
                        + (repr(literal) if literal else ", ".join(packages) or decision["label"])
                        + "; using the files the agent touched")
    # files the agent touched stay affected even when the graph doesn't index them
    known = {e["path"] for e in entries}
    for f in touched_files(decision):
        if f not in known:
            entries.append({"path": f, "symbol": None, "kind": "file", "line": None,
                            "via": "touched by the agent's tool call", "depth": 0, "symbol_id": None})
    return Affected("entire-graph", entries)


def find_affected_code(decision: dict, project: Path | None = None,
                       entire: Entire | None = None, index: Index | None = None) -> Affected:
    """Every file and function that uses the decision's model/service (see module docstring)."""
    if index is None and project is None:
        return fallback(decision, "no project given")
    return analyze(index or Index(project, entire), decision)


def record_affected(graph, decision_id: int, result: Affected, tracer=None) -> list[int]:
    """Store the entries as `code` nodes with `affects` edges; remember them on the decision."""
    ids = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for e in result.entries:
        label = f"{e['path']}:{e['symbol']}" if e["symbol"] else e["path"]
        data = {"path": e["path"], "symbol": e["symbol"], "kind": e["kind"], "line": e["line"],
                "source": result.source}
        node = graph.find_node("code", label)
        if node:
            graph.update_data(node["id"], **data)
            nid = node["id"]
        else:
            nid = graph.add_node("code", label, data)
        graph.add_edge(decision_id, nid, "affects")
        ids.append(nid)
    graph.update_data(decision_id, affected_code=[{k: v for k, v in e.items() if k != "symbol_id"}
                                                  for e in result.entries],
                      affected_files=result.files, affected_source=result.source,
                      affected_note=result.note, affected_at=now)
    if tracer is not None:
        tracer.event("affected_code" if result.source == "entire-graph" else "affected_code_fallback",
                     "context" if result.source == "entire-graph" else "recovery",
                     f"decision #{decision_id}: {len(result.entries)} code locations in "
                     f"{len(result.files)} files via {result.source}"
                     + (f" ({result.note})" if result.note else ""), decision_id=decision_id)
    return ids


def describe(entry: dict) -> str:
    loc = f"{entry['path']}:{entry['line']}" if entry.get("line") else entry["path"]
    return f"{loc} {entry['symbol']}" if entry.get("symbol") else loc


def refresh(project: Path, decision_ids: list[int] | None = None, entire: Entire | None = None,
            tracer=None, graph=None) -> dict[int, Affected]:
    """Re-analyse (all or some) decisions with one graph read and store the results."""
    from notbydefault.graph import Graph
    index = Index(project, entire)
    if tracer is not None and not index.ok:
        tracer.event("entire_graph_unavailable", "recovery",
                     f"{index.error}; falling back to the files the agent touched")
    own = graph is None
    g = graph or Graph.for_project(project)
    try:
        out = {}
        for d in g.nodes("decision"):
            if decision_ids is not None and d["id"] not in decision_ids:
                continue
            out[d["id"]] = analyze(index, d)
            record_affected(g, d["id"], out[d["id"]], tracer)
        return out
    finally:
        if own:
            g.close()
