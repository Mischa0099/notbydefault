import subprocess

from notbydefault.scan import scan_history


def _commit(repo, name, text, msg):
    (repo / name).write_text(text, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", name], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", msg], check=True,
                   capture_output=True)


def test_history_scan_counts_decision_shaped_lines_without_a_graph(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    for k, v in (("user.email", "t@example.com"), ("user.name", "t"), ("commit.gpgsign", "false")):
        subprocess.run(["git", "-C", str(tmp_path), "config", k, v], check=True)
    _commit(tmp_path, "app.py", 'MODEL = "gpt-4o"\n', "model literal")
    _commit(tmp_path, "util.py", "def add(a, b):\n    return a + b\n", "benign")
    _commit(tmp_path, "requirements.txt", "pymongo==4.6\n", "pip install pymongo")

    lines = []
    assert scan_history(tmp_path, out=lines.append) == 0
    text = "\n".join(lines)
    assert "scanned last 3 commits" in text
    assert "Found 2 decision-shaped line(s) across 2 commit(s)" in text
    assert "app.py: MODEL = \"gpt-4o\"" in text and "requirements.txt: pymongo==4.6" in text
    assert not (tmp_path / ".notbydefault").exists()       # read-only: no graph created
