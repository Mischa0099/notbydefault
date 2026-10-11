"""Save decision records to Entire Brain, when the entire-brain plugin is installed.

Brain is not in Entire's plugin index (`entire plugin search brain` finds nothing as of
entire 0.11.4) and its commands are not in the Entire docs, so nothing here guesses a command:
when the plugin is installed we read `entire brain --help`, pick a save-like subcommand only if
the help lists one, read that subcommand's own --help, and pass the record the way that help
documents (a --file flag or a <file>/<path> argument). Otherwise we skip and say why.
"""
from __future__ import annotations

import re
from pathlib import Path

from notbydefault.entire import Entire

SAVE_SUBCOMMANDS = ("save", "add", "remember", "ingest", "import")
INSTALL_HINT = (
    "Entire Brain is not installed, so the decision record was not saved to Brain (skipped). "
    "`entire plugin list` doesn't list it and `entire plugin search brain` finds nothing in the "
    "plugin index; if you have access to it, install it with "
    "`entire plugin install <its repository URL, or a local entire-brain executable>` "
    "(https://docs.entire.io/cli-reference/plugin), then run "
    "`python -m notbydefault records --project <dir>` to save existing records.")


def _subcommands(help_text: str) -> list[str]:
    """Names listed under a cobra-style 'Available Commands:' / 'Commands:' section."""
    names, inside = [], False
    for line in help_text.splitlines():
        if re.match(r"\s*(Available )?Commands:\s*$", line):
            inside = True
            continue
        if inside:
            m = re.match(r"\s{2,}([a-z][\w-]*)\s", line + " ")
            if m:
                names.append(m.group(1))
            elif line.strip() == "" and names:
                inside = False
    return names


def save_command(entire: Entire, path: Path) -> list[str] | None:
    """The documented `entire brain ...` invocation that saves `path`, or None."""
    top = entire.run(["brain", "--help"], timeout=20) or ""
    sub = next((s for s in SAVE_SUBCOMMANDS if s in _subcommands(top)), None)
    if sub is None:
        return None
    usage = entire.run(["brain", sub, "--help"], timeout=20) or ""
    if re.search(r"--file\b", usage):
        return ["brain", sub, "--file", str(path)]
    if re.search(rf"brain {sub}\b[^\n]*<(file|path)>", usage, re.I):
        return ["brain", sub, str(path)]
    return None


def save(project: Path, record: Path, decision: dict | None,
         entire: Entire | None = None) -> tuple[bool, str]:
    """(saved?, message for the user). Never raises."""
    entire = entire or Entire()
    try:
        if not entire.has_plugin("brain"):
            return False, INSTALL_HINT
        cmd = save_command(entire, Path(record))
        if cmd is None:
            return False, ("Entire Brain is installed, but its --help documents no subcommand "
                           f"that saves a file (looked for: {', '.join(SAVE_SUBCOMMANDS)}); "
                           "skipped saving the decision record to Brain.")
        if entire.run(cmd, cwd=project, timeout=60) is None:
            return False, f"Saving to Entire Brain failed ({entire.last_error()}); skipped."
        return True, f"Saved {Path(record).name} to Entire Brain (`entire {' '.join(cmd[:2])}`)."
    except Exception as e:      # Brain is optional: never break a verdict over it
        return False, f"Entire Brain unavailable ({e}); skipped."
