"""A small, safe Markdown renderer for the files NotByDefault writes itself (decision records,
PR descriptions, costs.md): headings, paragraphs, lists, tables, code fences, inline code, bold
and http(s) links. Everything is escaped first; no raw HTML passes through."""
from __future__ import annotations

import re

from markupsafe import Markup, escape

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET = re.compile(r"^(\s*)[-*]\s+(.*)$")
_NUMBERED = re.compile(r"^(\s*)\d+\.\s+(.*)$")
_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_URL = re.compile(r"(?<![\"'=>])\bhttps?://[^\s<>()|`\"']+[^\s<>()|`\"'.,;:]")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ASSUMPTION = re.compile(r"\bASSUMPTIONS?\b")


def _text(s: str) -> str:
    """Inline markup outside code spans (input already escaped)."""
    s = _LINK.sub(lambda m: f'<a href="{m.group(2)}" rel="noopener noreferrer" '
                            f'target="_blank">{m.group(1)}</a>', s)
    if "<a " not in s:
        s = _URL.sub(lambda m: f'<a href="{m.group(0)}" rel="noopener noreferrer" '
                               f'target="_blank">{m.group(0)}</a>', s)
    s = _BOLD.sub(r"<strong>\1</strong>", s)
    return _ASSUMPTION.sub(lambda m: f'<mark class="assume">{m.group(0)}</mark>', s)


def inline(s: str) -> str:
    parts = str(escape(s)).split("`")
    out = []
    for i, part in enumerate(parts):
        if i % 2 and i < len(parts) - 1:
            breakable = part.replace("/", "/<wbr>").replace(":", ":<wbr>")
            out.append(f"<code>{breakable}</code>")
        else:
            out.append(_text(part) if i % 2 == 0 else "`" + _text(part))
    return "".join(out)


def _cells(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def render(text: str, heading_offset: int = 1) -> Markup:
    lines = (text or "").replace("\r\n", "\n").split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            body = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                body.append(lines[i])
                i += 1
            i += 1
            out.append(f"<pre class=\"code\"><code>{escape(chr(10).join(body))}</code></pre>")
            continue
        if stripped.startswith("<!--"):
            while i < len(lines) and "-->" not in lines[i]:
                i += 1
            i += 1
            continue
        m = _HEADING.match(line)
        if m:
            level = min(6, len(m.group(1)) + heading_offset)
            out.append(f"<h{level}>{inline(m.group(2))}</h{level}>")
            i += 1
            continue
        if stripped.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(_cells(lines[i]))
                i += 1
            head, body = rows[0], [r for r in rows[1:] if not all(set(c) <= set("-: ") for c in r)]
            html = ["<div class=\"table-wrap\"><table class=\"md\"><thead><tr>"]
            html += [f"<th>{inline(c).replace('/', '/<wbr>')}</th>" for c in head]
            html.append("</tr></thead><tbody>")
            for r in body:
                html.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            html.append("</tbody></table></div>")
            out.append("".join(html))
            continue
        for rx, tag in ((_BULLET, "ul"), (_NUMBERED, "ol")):
            if rx.match(line):
                items = []
                while i < len(lines) and (rx.match(lines[i]) or (
                        items and lines[i].startswith("  ") and lines[i].strip())):
                    mm = rx.match(lines[i])
                    if mm:
                        items.append(mm.group(2))
                    else:
                        items[-1] += " " + lines[i].strip()
                    i += 1
                out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items)
                           + f"</{tag}>")
                break
        else:
            if not stripped:
                i += 1
                continue
            para = []
            while i < len(lines) and lines[i].strip() and not (
                    _HEADING.match(lines[i]) or lines[i].strip().startswith(("|", "```", "<!--"))
                    or _BULLET.match(lines[i]) or _NUMBERED.match(lines[i])):
                para.append(lines[i].strip())
                i += 1
            out.append(f"<p>{inline(' '.join(para))}</p>")
    return Markup("\n".join(out))
