import json

import pytest

from notbydefault.detect import detect, packages_from_command


def bash(command, cwd="/proj", tool="Bash"):
    return {"tool_name": tool, "tool_input": {"command": command}, "cwd": cwd}


def chosen(detections):
    return [(d.kind, d.chosen) for d in detections]


# --- shell installs ------------------------------------------------------------

@pytest.mark.parametrize("command,expected", [
    ("npm install @supabase/supabase-js", ("service", "Supabase")),
    ("npm i firebase@10.1.0", ("service", "Firebase")),
    ("yarn add prisma -D", ("service", "Prisma")),
    ("pnpm add mongoose", ("service", "MongoDB")),
    ("bun add @clerk/nextjs", ("service", "Clerk")),
    ("pip install pymongo==4.6", ("service", "MongoDB")),
    ("pip3 install 'SQLAlchemy>=2'", ("service", "SQLAlchemy")),
    ("python -m pip install supabase", ("service", "Supabase")),
    ("py -3.12 -m pip install firebase_admin", ("service", "Firebase")),
    ("uv add psycopg2-binary", ("service", "PostgreSQL")),
    ("poetry add openai", ("service", "OpenAI")),
    ("cd app && npm install --save stripe", ("service", "Stripe")),
])
def test_shell_install_of_known_service(command, expected):
    assert chosen(detect(bash(command))) == [expected]


def test_powershell_tool_is_also_checked():
    event = bash(r"C:\Python312\python.exe -m pip install anthropic", tool="PowerShell")
    assert chosen(detect(event)) == [("service", "Anthropic")]


def test_prisma_and_client_collapse_into_one_decision():
    d = detect(bash("npm install prisma @prisma/client"))
    assert chosen(d) == [("service", "Prisma")]
    assert d[0].evidence == "prisma, @prisma/client"


def test_multiple_services_in_one_command():
    assert chosen(detect(bash("npm i @supabase/supabase-js stripe lodash"))) == [
        ("service", "Supabase"), ("service", "Stripe")]


def test_shell_points_at_manifest():
    d = detect(bash("npm install mongoose", cwd="/proj"))[0]
    assert d.file.replace("\\", "/") == "/proj/package.json"
    assert detect(bash("pip install pymongo"))[0].file is None


@pytest.mark.parametrize("command", [
    "npm install",                      # installs existing deps, chooses nothing
    "npm install lodash express",       # not a service
    "pip install -r requirements.txt",  # value of -r is not a package
    "npx prisma migrate dev",           # runs a tool, doesn't add it
    "echo npm install supabase > notes.txt && ls",  # echo's args aren't a command
    "git commit -m 'add supabase'",
    "npm test",
])
def test_shell_non_decisions(command):
    assert detect(bash(command)) == []


def test_editable_flag_value_skipped():
    assert packages_from_command("pip install -e . redis") == [("redis", "py", None)]


# --- model literals --------------------------------------------------------------

def write(path, content):
    return {"tool_name": "Write", "tool_input": {"file_path": str(path), "content": content}}


def edit(path, old, new):
    return {"tool_name": "Edit",
            "tool_input": {"file_path": str(path), "old_string": old, "new_string": new}}


@pytest.mark.parametrize("literal", [
    "gpt-4o", "gpt-4.1-mini", "gpt-3.5-turbo", "o3-mini", "o1", "o1-2024-12-17",
    "claude-sonnet-4-5", "claude-3-5-sonnet-20241022", "claude-opus-5-5",
    "gemini-2.5-pro", "llama-3.1-70b", "meta-llama/Llama-3.1-8B-Instruct",
    "mistral-large-latest", "deepseek-chat", "grok-4", "text-embedding-3-small",
    "openai/gpt-oss-120b", "gpt-oss-20b", "qwen/qwen3.8-27b", "gemma-4-31b-it",
    "gemini-3.5-flash-lite",
])
def test_model_literal_in_new_file(tmp_path, literal):
    d = detect(write(tmp_path / "ai.py", f'client.chat(model="{literal}")\n'))
    assert chosen(d) == [("model", literal)]
    assert d[0].file == str(tmp_path / "ai.py")


@pytest.mark.parametrize("text", [
    "# we use gpt-4o for chat",            # comment, not a hardcoded string
    'tool = "claude-code"',                 # product name, not a model
    'name = "o1-builder-utils"',            # o1 prefix, but not an o-series model
    'x = "solo"',
    'MODEL = os.environ["OPENAI_MODEL"]',
])
def test_model_non_literals(tmp_path, text):
    assert detect(write(tmp_path / "a.py", text)) == []


def test_edit_only_flags_newly_introduced_model(tmp_path):
    f = tmp_path / "ai.ts"
    f.write_text('const a = "gpt-4o";\nconst b = 1;\n')
    assert detect(edit(f, "const b = 1;", 'const b = "gpt-4o";')) == []
    d = detect(edit(f, "const b = 1;", 'const b = "claude-sonnet-4-5";'))
    assert chosen(d) == [("model", "claude-sonnet-4-5")]


def test_overwrite_keeping_same_model_is_quiet(tmp_path):
    f = tmp_path / "ai.py"
    f.write_text('M = "gpt-4o"\n')
    assert detect(write(f, 'M = "gpt-4o"\nN = 2\n')) == []
    assert chosen(detect(write(f, 'M = "gpt-5"\n'))) == [("model", "gpt-5")]


def test_edit_of_missing_file_falls_back_to_old_string(tmp_path):
    d = detect(edit(tmp_path / "nope.py", 'm = "gpt-4o"', 'm = "gpt-4o-mini"'))
    assert chosen(d) == [("model", "gpt-4o-mini")]


# --- manifests -------------------------------------------------------------------

def test_package_json_new_dependency(tmp_path):
    f = tmp_path / "package.json"
    old = {"name": "demo", "dependencies": {"express": "^4.0.0"}}
    f.write_text(json.dumps(old, indent=2))
    new = {"name": "demo", "dependencies": {"express": "^4.0.0", "@supabase/supabase-js": "^2"}}
    d = detect(write(f, json.dumps(new, indent=2)))
    assert chosen(d) == [("service", "Supabase")]
    assert d[0].source == "manifest"


def test_package_json_existing_dependency_is_quiet(tmp_path):
    f = tmp_path / "package.json"
    f.write_text('{"dependencies": {"mongoose": "^8"}}')
    assert detect(edit(f, '"mongoose": "^8"', '"mongoose": "^8.1"')) == []


def test_requirements_txt_new_dependency(tmp_path):
    f = tmp_path / "requirements.txt"
    f.write_text("flask==3.0\n")
    d = detect(edit(f, "flask==3.0\n", "flask==3.0\npsycopg2-binary>=2.9  # db\nrequests\n"))
    assert chosen(d) == [("service", "PostgreSQL")]


def test_pyproject_new_dependency(tmp_path):
    f = tmp_path / "pyproject.toml"
    d = detect(write(f, '[project]\nname = "x"\ndependencies = ["fastapi", "sqlmodel>=0.0.16"]\n'))
    assert chosen(d) == [("service", "SQLModel")]


def test_other_tools_ignored():
    assert detect({"tool_name": "Read", "tool_input": {"file_path": "x"}}) == []
    assert detect({}) == []


# --- unquoted model names in config files (D1) ---------------------------------------------

def write(path, content):
    return {"tool_name": "Write", "tool_input": {"file_path": str(path), "content": content}}


@pytest.mark.parametrize("name,content,model", [
    (".env", "OPENAI_MODEL=gpt-4o-mini\n", "gpt-4o-mini"),
    (".env.local", "export LLM=claude-sonnet-4-5\n", "claude-sonnet-4-5"),
    ("config.yaml", "llm:\n  provider: google\n  model: gemini-2.5-flash  # fast\n",
     "gemini-2.5-flash"),
    ("settings.toml", "[llm]\nmodel = mistral-large-latest\n", "mistral-large-latest"),
    ("Dockerfile", "FROM python:3.12\nENV MODEL=gpt-4.1\n", "gpt-4.1"),
    ("app.yml", "reasoning_model: o3\n", "o3"),
])
def test_unquoted_model_in_config_file(tmp_path, name, content, model):
    assert chosen(detect(write(tmp_path / name, content))) == [("model", model)]


@pytest.mark.parametrize("name,content", [
    ("app.py", "MODEL = gpt-4o\n"),                       # not a config file
    ("notes.md", "model: gpt-4o\n"),                      # prose, not config
    ("config.yaml", "description: use gpt-4o for chat\n"),  # not the whole value
    ("config.yaml", "version: o3\n"),                     # short name under a non-model key
    ("config.yaml", "model: my-finetune\n"),              # not a known model pattern
])
def test_unquoted_text_that_is_not_a_model_choice(tmp_path, name, content):
    assert detect(write(tmp_path / name, content)) == []


def test_config_model_already_there_is_not_new(tmp_path):
    env = tmp_path / ".env"
    env.write_text("OPENAI_MODEL=gpt-4o-mini\n")
    assert detect(write(env, "OPENAI_MODEL=gpt-4o-mini\nLOG_LEVEL=debug\n")) == []
