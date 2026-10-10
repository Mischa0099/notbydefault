"""python -m notbydefault init|show|install|check|label"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from notbydefault import PACKAGE_DIR
from notbydefault.graph import Graph, db_path

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
    from notbydefault.fit.run import CheckOptions, check_model_decision
    opts = CheckOptions(max_cost=args.max_cost, yes=args.yes,
                        calls_per_user=args.calls_per_user, n_samples=args.samples,
                        expand_rounds=args.expand_rounds)
    return check_model_decision(project, args.decision_id, opts)


def cmd_label(args) -> int:
    from notbydefault.fit.labels import label
    return label(Path(args.project).resolve(), args.decision_id, n=args.n)


def cmd_require(args) -> int:
    from notbydefault.stale import require
    return require(Path(args.project_dir), args.requirement)


def cmd_pivot(args) -> int:
    """List pivots, or record the user's choice for one by hand."""
    from notbydefault.pivot import resolve
    project = Path(args.project).resolve()
    if not db_path(project).exists():
        print(f"No graph at {db_path(project)}.")
        return 1
    with Graph.for_project(project) as g:
        if args.pivot_id is None:
            for p in g.nodes("pivot"):
                d = p["data"]
                print(f"#{p['id']:<4} [{d.get('status')}] {p['label']}  decisions: "
                      f"{', '.join(f'#{x}' for x in d.get('decisions', [])) or '-'}")
            return 0
        p = g.get_node(args.pivot_id)
        if not p or p["type"] != "pivot":
            print(f"No pivot #{args.pivot_id}")
            return 1
        if not args.choice:
            print(f"#{p['id']} [{p['data'].get('status')}] {p['label']}")
            return 0
        stale = resolve(g, p, args.choice, "(recorded with `notbydefault pivot`)")
    print(f"Pivot #{args.pivot_id}: {args.choice}"
          + (f"; decisions now stale: {', '.join(f'#{d}' for d in stale)}" if stale else ""))
    return 0


# --- install --------------------------------------------------------------------

HOOK_MARKER = "notbydefault"
PRE_TOOL_MATCHER = "Bash|PowerShell|Write|Edit|MultiEdit"


# event -> (hook script, matcher or None, timeout s). UserPromptSubmit has no matcher support,
# and its timeout covers one fast Groq call when the pre-filter matches.
OUR_HOOKS = {
    "PreToolUse": ("pre_tool.py", PRE_TOOL_MATCHER, 10),
    "UserPromptSubmit": ("user_prompt.py", None, 20),
}


def hook_entry(project: Path, script: str = "pre_tool.py", timeout: int = 10) -> dict:
    return {
        "type": "command",
        # exec form: no shell, so spaces in "C:\Program Files\..." need no quoting
        "command": str(Path(sys.executable).resolve()),
        "args": [str(PACKAGE_DIR / "hooks" / script), "--project", str(project)],
        "timeout": timeout,
    }


def _is_ours(handler: dict, script: str = "pre_tool.py") -> bool:
    return any(HOOK_MARKER in str(a) and script in str(a) for a in handler.get("args", []))


def merge_settings(settings: dict, project: Path) -> dict:
    """Add (or refresh) our hooks without touching any other hook."""
    for event, (script, matcher, timeout) in OUR_HOOKS.items():
        groups = settings.setdefault("hooks", {}).setdefault(event, [])
        for group in groups:
            group["hooks"] = [h for h in group.get("hooks", []) if not _is_ours(h, script)]
        groups[:] = [g for g in groups if g.get("hooks")]
        group = {"hooks": [hook_entry(project, script, timeout)]}
        if matcher:
            group = {"matcher": matcher, **group}
        groups.append(group)
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
    print(f"Installed NotByDefault PreToolUse and UserPromptSubmit hooks in {path}")
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

    p = sub.add_parser("check", help="fit-test a decision against the project's constraints")
    p.add_argument("decision_id", type=int)
    p.add_argument("--project", default=".", help="project dir (default: current dir)")
    p.add_argument("--max-cost", type=float, default=1.00,
                   help="stop before spending more than this many USD (default 1.00)")
    p.add_argument("--yes", "-y", action="store_true", help="don't ask for confirmation")
    p.add_argument("--calls-per-user", type=float,
                   help="calls per user per month (asked once, default 30)")
    p.add_argument("--samples", type=int, default=8, help="synthetic inputs to generate")
    p.add_argument("--expand-rounds", type=int, default=1,
                   help="times to add 4 inputs when the verdict is too close to call (default 1)")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("require", help="add a requirement and mark decisions it makes stale")
    p.add_argument("project_dir")
    p.add_argument("requirement", help='e.g. "EU data residency required"')
    p.set_defaults(func=cmd_require)

    p = sub.add_parser("pivot", help="list pivots, or record the user's choice for one")
    p.add_argument("pivot_id", type=int, nargs="?")
    p.add_argument("--choice", choices=["accepted", "adjusting", "kept"])
    p.add_argument("--project", default=".", help="project dir (default: current dir)")
    p.set_defaults(func=cmd_pivot)

    p = sub.add_parser("label", help="hand-label outputs to measure verifier trust")
    p.add_argument("decision_id", type=int)
    p.add_argument("--project", default=".", help="project dir (default: current dir)")
    p.add_argument("-n", type=int, default=10, help="outputs to label (default 10)")
    p.set_defaults(func=cmd_label)

    args = parser.parse_args(argv)
    return args.func(args)
