"""python -m notbydefault init|show|install|check"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from notbydefault import PACKAGE_DIR
from notbydefault.graph import Graph, db_path, find_affected_code

# (answer key, node type, question, label template)
QUESTIONS = [
    ("goal", "goal", "What are you building? (the goal)", "{}"),
    ("users", "requirement", "Expected users? (e.g. 500 monthly actives)", "Expected users: {}"),
    ("budget", "constraint", "Monthly budget for infrastructure? (e.g. $50)", "Monthly budget: {}"),
    ("region", "constraint", "Region / compliance needs? (e.g. EU data residency, HIPAA)",
     "Region/compliance: {}"),
    ("team", "constraint", "Team size?", "Team size: {}"),
]


def _ask(question: str) -> str:
    try:
        return input(f"{question} [Enter to skip]: ").strip()
    except EOFError:
        return ""


def cmd_init(args) -> int:
    project = Path(args.project_dir).resolve()
    with Graph.for_project(project) as g:
        previous = [n["id"] for n in g.nodes() if n["data"].get("source") == "init"]
        if previous and not args.force:
            print(f"{project} is already initialised ({len(previous)} answers stored). "
                  "Use --force to replace them.")
            return 1
        g.delete_nodes(previous)

        answers = {}
        for key, _type, question, _label in QUESTIONS:
            given = getattr(args, key)
            answers[key] = given if given is not None else ("" if args.no_input else _ask(question))

        goal_id = None
        created = []
        for key, node_type, _question, label in QUESTIONS:
            value = answers[key].strip()
            if not value:
                continue
            nid = g.add_node(node_type, label.format(value), {"source": "init", "key": key,
                                                              "value": value})
            created.append(nid)
            if node_type == "goal":
                goal_id = nid
        if goal_id:
            for nid in created:
                if g.get_node(nid)["type"] == "requirement":
                    g.add_edge(nid, goal_id, "serves")

    print(f"Stored {len(created)} answer(s) in {db_path(project)}")
    return 0


def _describe(node: dict) -> str:
    d = node["data"]
    if node["type"] == "decision":
        extra = f"  [{d.get('kind')}/{d.get('category')}, {d.get('status')}]"
        if d.get("evidence"):
            extra += f"  from {d.get('source')}: {d['evidence']}"
        return f"{node['label']}{extra}"
    return node["label"]


def cmd_show(args) -> int:
    project = Path(args.project_dir).resolve()
    if not db_path(project).exists():
        print(f"No graph yet at {db_path(project)}. Run `init` first.")
        return 1
    with Graph.for_project(project) as g:
        nodes = g.nodes()
        by_id = {n["id"]: n for n in nodes}
        print(f"Decision Graph: {db_path(project)}")
        print(f"\nNodes ({len(nodes)})")
        for node_type in ("goal", "requirement", "constraint", "decision", "evidence",
                          "alternative", "code", "pivot"):
            group = [n for n in nodes if n["type"] == node_type]
            if not group:
                continue
            print(f"  {node_type}")
            for n in group:
                print(f"    #{n['id']:<4} {_describe(n)}")
        edges = g.edges()
        print(f"\nEdges ({len(edges)})")
        for e in edges:
            src, dst = by_id[e["src"]], by_id[e["dst"]]
            print(f"  #{src['id']} {src['label']}  --{e['type']}-->  #{dst['id']} {dst['label']}")
    return 0


def cmd_check(args) -> int:
    project = Path(args.project).resolve()
    if not db_path(project).exists():
        print(f"No graph at {db_path(project)}. Run this from the project dir or pass --project.")
        return 1
    with Graph.for_project(project) as g:
        node = g.get_node(args.decision_id)
        if not node or node["type"] != "decision":
            print(f"No decision #{args.decision_id} in {db_path(project)}")
            return 1
        print(f"Decision #{node['id']}: {_describe(node)}")
        deps = g.get_dependencies(node["id"])
        print("\nDepends on:" if deps else "\nDepends on: nothing yet (run `init` to record "
                                            "goals and constraints)")
        for n in deps:
            print(f"  #{n['id']} {n['type']}: {n['label']}")
        files = find_affected_code(node)
        print("\nAffects:" if files else "\nAffects: no files recorded")
        for f in files:
            print(f"  {f}")
        print("\n(Fit testing against these constraints comes in a later milestone.)")
    return 0


# --- install --------------------------------------------------------------------

HOOK_MARKER = "notbydefault"
PRE_TOOL_MATCHER = "Bash|PowerShell|Write|Edit|MultiEdit"


def hook_entry(project: Path) -> dict:
    return {
        "type": "command",
        # exec form: no shell, so spaces in "C:\Program Files\..." need no quoting
        "command": str(Path(sys.executable).resolve()),
        "args": [str(PACKAGE_DIR / "hooks" / "pre_tool.py"), "--project", str(project)],
        "timeout": 10,
    }


def _is_ours(handler: dict) -> bool:
    return any(HOOK_MARKER in str(a) and "pre_tool.py" in str(a) for a in handler.get("args", []))


def merge_settings(settings: dict, project: Path) -> dict:
    """Add (or refresh) our PreToolUse hook without touching any other hook."""
    groups = settings.setdefault("hooks", {}).setdefault("PreToolUse", [])
    for group in groups:
        group["hooks"] = [h for h in group.get("hooks", []) if not _is_ours(h)]
    groups[:] = [g for g in groups if g.get("hooks")]
    groups.append({"matcher": PRE_TOOL_MATCHER, "hooks": [hook_entry(project)]})
    return settings


def cmd_install(args) -> int:
    project = Path(args.project_dir).resolve()
    if not project.is_dir():
        print(f"Not a directory: {project}")
        return 1
    path = project / ".claude" / "settings.json"
    settings = {}
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
        try:
            settings = json.loads(text) if text else {}
        except json.JSONDecodeError as e:
            print(f"{path} is not valid JSON ({e}); not touching it.")
            return 1
    merge_settings(settings, project)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    Graph.for_project(project).close()
    print(f"Installed NotByDefault PreToolUse hook in {path}")
    print(f"  python: {sys.executable}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m notbydefault")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="record goals, requirements and constraints")
    p.add_argument("project_dir")
    p.add_argument("--goal")
    p.add_argument("--users")
    p.add_argument("--budget")
    p.add_argument("--region")
    p.add_argument("--team")
    p.add_argument("--no-input", action="store_true", help="don't prompt; skip unanswered")
    p.add_argument("--force", action="store_true", help="replace previous answers")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("show", help="print the Decision Graph")
    p.add_argument("project_dir")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("install", help="add the NotByDefault hook to a project")
    p.add_argument("project_dir")
    p.set_defaults(func=cmd_install)

    p = sub.add_parser("check", help="show what a decision depends on and affects")
    p.add_argument("decision_id", type=int)
    p.add_argument("--project", default=".", help="project dir (default: current dir)")
    p.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    return args.func(args)
