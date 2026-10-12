"""Build a disposable copy of demo-app for recording the demo video. See docs/recording_plan.md.

    python scripts/recording_setup.py [--out .demo-dataset/rec]

The copy is made exactly as scripts/demo_dataset.py makes one (graph.db copied read-only, git repo
on main), then:
  - the response caches of the last demo-dataset build and of #21's live run
    (demo-app/.notbydefault/cache-live-20261011T202940Z) are copied into the copy's cache, so
    `check 21` on the copy replays the live run from cache and makes no requests;
  - the scripted agent reasons are recorded (labelled as scripted, same as demo_dataset.py);
  - the NotByDefault hooks are reinstalled pointing at the copy (copied settings point at the
    real demo-app).
Nothing in demo-app/, .demo-dataset/demo-app/ or docs/ is written.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from demo_dataset import copy_demo, record_reasons  # noqa: E402

CACHES = [REPO / ".demo-dataset" / "demo-app" / ".notbydefault" / "cache",
          REPO / "demo-app" / ".notbydefault" / "cache-live-20261011T202940Z"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(REPO / ".demo-dataset" / "rec"))
    out = Path(ap.parse_args().out).resolve()
    if out in {(REPO / "demo-app").resolve(), (REPO / ".demo-dataset" / "demo-app").resolve()}:
        raise SystemExit(f"--out must be a throwaway folder, not {out}")
    p = copy_demo(out)
    cache = p / ".notbydefault" / "cache"
    for src in CACHES:
        if src.is_dir():
            shutil.copytree(src, cache, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("live-run.json", "attempts.json"))
            print(f"cache: + {src.relative_to(REPO)}")
        else:
            print(f"cache: missing {src.relative_to(REPO)} (check 21 would need live requests)")
    record_reasons(p)
    subprocess.run([sys.executable, "-m", "notbydefault", "install", str(p)], cwd=REPO, check=True)
    print(f"ready: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
