"""Decide whether a tool call is a decision moment (a consequential stack choice)."""
from __future__ import annotations

import re
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path

# --- editable knowledge ------------------------------------------------------
# package name -> (what was chosen, category). Names are npm or PyPI names,
# compared lowercase with "_" normalised to "-".
KNOWN_SERVICES: dict[str, tuple[str, str]] = {
    # databases / backend-as-a-service
    "supabase": ("Supabase", "database"),
    "@supabase/supabase-js": ("Supabase", "database"),
    "firebase": ("Firebase", "database"),
    "firebase-admin": ("Firebase", "database"),
    "mongodb": ("MongoDB", "database"),
    "mongoose": ("MongoDB", "database"),
    "pymongo": ("MongoDB", "database"),
    "motor": ("MongoDB", "database"),
    "pg": ("PostgreSQL", "database"),
    "psycopg": ("PostgreSQL", "database"),
    "psycopg2": ("PostgreSQL", "database"),
    "psycopg2-binary": ("PostgreSQL", "database"),
    "asyncpg": ("PostgreSQL", "database"),
    "mysql2": ("MySQL", "database"),
    "pymysql": ("MySQL", "database"),
    "mysqlclient": ("MySQL", "database"),
    "redis": ("Redis", "database"),
    "ioredis": ("Redis", "database"),
    "@planetscale/database": ("PlanetScale", "database"),
    "@neondatabase/serverless": ("Neon", "database"),
    "@vercel/postgres": ("Vercel Postgres", "database"),
    "@libsql/client": ("Turso", "database"),
    "convex": ("Convex", "database"),
    "@aws-sdk/client-dynamodb": ("DynamoDB", "database"),
    # ORMs
    "prisma": ("Prisma", "orm"),
    "@prisma/client": ("Prisma", "orm"),
    "drizzle-orm": ("Drizzle", "orm"),
    "typeorm": ("TypeORM", "orm"),
    "sequelize": ("Sequelize", "orm"),
    "sqlalchemy": ("SQLAlchemy", "orm"),
    "sqlmodel": ("SQLModel", "orm"),
    "peewee": ("Peewee", "orm"),
    "tortoise-orm": ("Tortoise ORM", "orm"),
    # auth
    "next-auth": ("Auth.js", "auth"),
    "@auth/core": ("Auth.js", "auth"),
    "auth0": ("Auth0", "auth"),
    "better-auth": ("Better Auth", "auth"),
    "lucia": ("Lucia", "auth"),
    "passport": ("Passport", "auth"),
    # hosting / deploy
    "vercel": ("Vercel", "hosting"),
    "netlify-cli": ("Netlify", "hosting"),
    "wrangler": ("Cloudflare Workers", "hosting"),
    "firebase-tools": ("Firebase Hosting", "hosting"),
    "aws-cdk-lib": ("AWS", "hosting"),
    "serverless": ("Serverless Framework", "hosting"),
    # AI providers
    "openai": ("OpenAI", "ai-provider"),
    "anthropic": ("Anthropic", "ai-provider"),
    "@anthropic-ai/sdk": ("Anthropic", "ai-provider"),
    "google-generativeai": ("Google Gemini", "ai-provider"),
    "google-genai": ("Google Gemini", "ai-provider"),
    "@google/generative-ai": ("Google Gemini", "ai-provider"),
    "@google/genai": ("Google Gemini", "ai-provider"),
    "mistralai": ("Mistral", "ai-provider"),
    "@mistralai/mistralai": ("Mistral", "ai-provider"),
    "cohere": ("Cohere", "ai-provider"),
    "cohere-ai": ("Cohere", "ai-provider"),
    "groq": ("Groq", "ai-provider"),
    "groq-sdk": ("Groq", "ai-provider"),
    # payments / email
    "stripe": ("Stripe", "payments"),
    "resend": ("Resend", "email"),
    "@sendgrid/mail": ("SendGrid", "email"),
}

# npm scopes where every package belongs to one service.
KNOWN_SCOPES: dict[str, tuple[str, str]] = {
    "@supabase/": ("Supabase", "database"),
    "@clerk/": ("Clerk", "auth"),
    "@auth0/": ("Auth0", "auth"),
    "@kinde-oss/": ("Kinde", "auth"),
    "@firebase/": ("Firebase", "database"),
}

# Hardcoded model identifiers, matched only inside a string literal.
MODEL_ID = re.compile(
    r"""["'`](
        gpt-\d[\w.:-]*
      | (?:openai/)?gpt-oss-[\w.:-]+
      | chatgpt-[\w.:-]+
      | o[134](?:-(?:mini|pro|preview|deep-research|\d{4}-\d{2}-\d{2}))*
      | claude-(?:opus|sonnet|haiku|fable|instant|\d)[\w.:@-]*
      | gemini-\d[\w.:-]*
      | (?:meta-llama/)?llama-?\d[\w.:-]*
      | (?:mistral|codestral|ministral|pixtral)-[\w.:-]+
      | (?:open-)?mixtral-[\w.:-]+
      | command-r[\w.:-]*
      | deepseek-[\w.:-]+
      | grok-\d[\w.:-]*
      | (?:qwen/)?qwen[\d.]*-[\w.:-]+
      | gemma-?\d[\w.:-]*
      | text-embedding-[\w.:-]+
    )["'`]""",
    re.IGNORECASE | re.VERBOSE,
)

MODEL_PROVIDERS = [
    ("gpt-", "OpenAI"), ("openai/", "OpenAI"), ("gemma", "Google"), ("chatgpt-", "OpenAI"), ("o1", "OpenAI"), ("o3", "OpenAI"),
    ("o4", "OpenAI"), ("text-embedding-", "OpenAI"), ("claude-", "Anthropic"),
    ("gemini-", "Google"), ("meta-llama/", "Meta"), ("llama", "Meta"),
    ("mistral-", "Mistral"), ("codestral-", "Mistral"), ("ministral-", "Mistral"),
    ("pixtral-", "Mistral"), ("mixtral-", "Mistral"), ("open-mixtral-", "Mistral"),
    ("command-r", "Cohere"), ("deepseek-", "DeepSeek"), ("grok-", "xAI"),
    ("qwen", "Alibaba"),
]

MANIFEST_NAMES = {"package.json", "pyproject.toml", "Pipfile"}
SHELL_TOOLS = {"Bash", "PowerShell"}
FILE_TOOLS = {"Write", "Edit", "MultiEdit"}
MAX_FILE_BYTES = 2_000_000


@dataclass
class Detection:
    kind: str            # "service" | "model"
    chosen: str          # e.g. "Supabase" or "gpt-4o"
    category: str        # database | orm | auth | hosting | ai-provider | ai-model | ...
    evidence: str        # the package(s) or literal that triggered it
    source: str          # "shell" | "manifest" | "code"
    file: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# --- packages -----------------------------------------------------------------

def normalize_package(token: str, ecosystem: str) -> str:
    token = token.strip().strip("'\"")
    if ecosystem == "npm":
        # keep a leading @scope, drop "@version"
        m = re.match(r"(@[^@/\s]+/[^@\s]+|[^@\s]+)", token)
        name = m.group(1) if m else token
    else:
        name = re.split(r"[<>=!~\[;@\s]", token, maxsplit=1)[0]
    return name.lower().replace("_", "-")


def lookup_service(package: str) -> tuple[str, str] | None:
    if package in KNOWN_SERVICES:
        return KNOWN_SERVICES[package]
    for scope, hit in KNOWN_SCOPES.items():
        if package.startswith(scope):
            return hit
    return None


def _services_from_packages(packages: list[str], source: str,
                            file: str | None) -> list[Detection]:
    found: dict[str, Detection] = {}
    for pkg in packages:
        hit = lookup_service(pkg)
        if not hit:
            continue
        chosen, category = hit
        if chosen in found:
            found[chosen].evidence += f", {pkg}"
        else:
            found[chosen] = Detection("service", chosen, category, pkg, source, file)
    return list(found.values())


# --- shell commands -----------------------------------------------------------

# (command prefix tokens, ecosystem, manifest the install writes to)
INSTALLERS: list[tuple[tuple[str, ...], str, str | None]] = [
    (("npm", "install"), "npm", "package.json"),
    (("npm", "i"), "npm", "package.json"),
    (("npm", "add"), "npm", "package.json"),
    (("yarn", "add"), "npm", "package.json"),
    (("pnpm", "add"), "npm", "package.json"),
    (("pnpm", "install"), "npm", "package.json"),
    (("pnpm", "i"), "npm", "package.json"),
    (("bun", "add"), "npm", "package.json"),
    (("bun", "install"), "npm", "package.json"),
    (("bun", "i"), "npm", "package.json"),
    (("pip", "install"), "py", None),
    (("pip3", "install"), "py", None),
    (("uv", "pip", "install"), "py", None),
    (("uv", "add"), "py", "pyproject.toml"),
    (("poetry", "add"), "py", "pyproject.toml"),
    (("pipenv", "install"), "py", "Pipfile"),
]
# flags whose next token is a value, not a package
VALUE_FLAGS = {"-r", "--requirement", "-c", "--constraint", "-e", "--editable",
               "-i", "--index-url", "--extra-index-url", "-t", "--target",
               "--prefix", "--registry", "-G", "--group", "--python", "-p"}


def _tokens(segment: str) -> list[str]:
    # posix=False keeps Windows backslashes intact; strip the quotes it leaves on.
    try:
        tokens = shlex.split(segment, posix=False)
    except ValueError:
        tokens = segment.split()
    return [t.strip("'\"") for t in tokens]


def _strip_launcher(tokens: list[str]) -> list[str]:
    """Turn `python -m pip install x` / `py -3.12 -m pip ...` / `sudo pip` into `pip ...`."""
    while tokens and (tokens[0] in {"sudo", "&"} or re.fullmatch(r"\w+=\S*", tokens[0])):
        tokens = tokens[1:]
    if tokens:
        exe = Path(tokens[0].replace("\\", "/")).name.lower().removesuffix(".exe")
        tokens = [exe, *tokens[1:]]
    if tokens and re.fullmatch(r"python[\d.]*|py", tokens[0]):
        rest = [t for t in tokens[1:] if not re.fullmatch(r"-\d[\d.]*", t)]
        if len(rest) >= 2 and rest[0] == "-m":
            return rest[1:]
    if tokens and tokens[0] in {"npx", "bunx"}:  # `npx prisma init` runs, doesn't install
        return []
    return tokens


def packages_from_command(command: str) -> list[tuple[str, str, str | None]]:
    """(package, ecosystem, manifest) for each package an install command adds."""
    results = []
    for segment in re.split(r"&&|\|\||[;|\n]", command):
        tokens = _strip_launcher(_tokens(segment.strip()))
        for prefix, eco, manifest in INSTALLERS:
            if tuple(tokens[: len(prefix)]) != prefix:
                continue
            skip = False
            for tok in tokens[len(prefix):]:
                if skip:
                    skip = False
                    continue
                if tok in VALUE_FLAGS:
                    skip = True
                    continue
                if tok.startswith("-") or tok.startswith(("./", ".\\", "/")) or tok == ".":
                    continue
                results.append((normalize_package(tok, eco), eco, manifest))
            break
    return results


def detect_shell(command: str, cwd: str | None) -> list[Detection]:
    by_manifest: dict[str | None, list[str]] = {}
    for pkg, _eco, manifest in packages_from_command(command):
        by_manifest.setdefault(manifest, []).append(pkg)
    detections: list[Detection] = []
    for manifest, packages in by_manifest.items():
        file = str(Path(cwd, manifest)) if (cwd and manifest) else manifest
        detections += _services_from_packages(packages, "shell", file)
    return _dedupe(detections)


# --- file edits ---------------------------------------------------------------

def is_manifest(path: str) -> bool:
    name = Path(path.replace("\\", "/")).name
    return name in MANIFEST_NAMES or re.fullmatch(r"requirements[\w.-]*\.txt", name) is not None


def dependency_names(path: str, text: str) -> set[str]:
    name = Path(path.replace("\\", "/")).name
    if name == "package.json":
        return {normalize_package(n, "npm")
                for n in re.findall(r'"(@?[\w.\-/]+)"\s*:\s*"[^"]*"', text)}
    names: set[str] = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if name.endswith(".txt"):
            m = re.match(r"([A-Za-z0-9][\w.\-]*)", line)
            if m:
                names.add(normalize_package(m.group(1), "py"))
        else:  # pyproject.toml / Pipfile: quoted PEP 508 strings and `name = "..."` tables
            for q in re.findall(r"""["']([A-Za-z0-9][\w.\-]*)\s*(?:[<>=!~\[;@][^"']*)?["']""", line):
                names.add(normalize_package(q, "py"))
            m = re.match(r"([A-Za-z0-9][\w.\-]*)\s*=", line)
            if m:
                names.add(normalize_package(m.group(1), "py"))
    return names


def model_ids(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for m in MODEL_ID.finditer(text):
        seen.setdefault(m.group(1), None)
    return list(seen)


def model_provider(model: str) -> str:
    low = model.lower()
    for prefix, provider in MODEL_PROVIDERS:
        if low.startswith(prefix):
            return provider
    return "unknown"


def _read(path: str) -> str | None:
    try:
        p = Path(path)
        if p.is_file() and p.stat().st_size <= MAX_FILE_BYTES:
            return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    return None


def detect_file_change(path: str, old_text: str, new_text: str) -> list[Detection]:
    """Detections for text `new_text` written into `path` that previously held `old_text`."""
    detections: list[Detection] = []
    if is_manifest(path):
        added = dependency_names(path, new_text) - dependency_names(path, old_text)
        detections += _services_from_packages(sorted(added), "manifest", path)
    before = {m.lower() for m in model_ids(old_text)}
    for model in model_ids(new_text):
        if model.lower() not in before:
            detections.append(Detection("model", model, "ai-model",
                                        f"{model_provider(model)} model literal", "code", path))
    return _dedupe(detections)


def detect_file_tool(tool_name: str, tool_input: dict) -> list[Detection]:
    path = tool_input.get("file_path") or ""
    if not path:
        return []
    current = _read(path)
    if tool_name == "Write":
        return detect_file_change(path, current or "", tool_input.get("content") or "")
    edits = tool_input.get("edits") or [tool_input]  # MultiEdit-style or single Edit
    old = current if current is not None else "\n".join(e.get("old_string") or "" for e in edits)
    new = "\n".join(e.get("new_string") or "" for e in edits)
    return detect_file_change(path, old, new)


# --- entry point --------------------------------------------------------------

def _dedupe(detections: list[Detection]) -> list[Detection]:
    seen: dict[tuple[str, str], Detection] = {}
    for d in detections:
        seen.setdefault((d.kind, d.chosen.lower()), d)
    return list(seen.values())


def detect(event: dict) -> list[Detection]:
    """Decision moments in a PreToolUse event. Empty list means nothing to flag."""
    tool_name = event.get("tool_name") or ""
    tool_input = event.get("tool_input") or {}
    if tool_name in SHELL_TOOLS:
        return detect_shell(tool_input.get("command") or "", event.get("cwd"))
    if tool_name in FILE_TOOLS:
        return detect_file_tool(tool_name, tool_input)
    return []
