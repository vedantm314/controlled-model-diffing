"""Parsers for the tagged text the generator models return."""
from __future__ import annotations

import re


def parse_tag(text: str, tag: str) -> str | None:
    """Read the content of one XML-style tag, with two documented tolerances.

    First, gpt-5-mini reliably opens <tag> but often never emits the closing
    </tag>, even on a normal finish. The second pattern takes everything after
    the opening tag so those responses are not dropped.

    Second, the search starts after </scratchpad>. A model sometimes narrates
    the format inside its own scratchpad ("I'll wrap the document in <content>
    tags"), and that mention comes earlier in the string than the real
    delimiter. A plain search then starts mid-scratchpad and drags the
    scratchpad tail into the document.

    The rule is not "use the last <content>": a document may legitimately
    contain the literal string, and taking the last one would truncate it.
    """
    search_from = 0
    if tag != "scratchpad":
        close = re.search(r"</scratchpad\s*>", text, re.IGNORECASE)
        if close:
            search_from = close.end()
    seg = text[search_from:]
    m = re.search(rf"<{tag}>(.*?)</{tag}>", seg, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(rf"<{tag}>(.*)", seg, re.DOTALL)
    return m.group(1).strip() if m else None


def parse_bullets(text: str) -> list[str]:
    """Read a "- item" or "* item" list."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(("-", "*")):
            item = line.lstrip("-* ").strip()
            if item:
                out.append(item)
    return out


def parse_ideas(text: str) -> list[str]:
    """Read every <idea>...</idea> block."""
    return [m.strip() for m in re.findall(r"<idea>(.*?)</idea>", text, re.DOTALL) if m.strip()]


def parse_key_facts(text: str) -> list[str]:
    """Read the <key_facts> block, or the whole response if the tag is absent."""
    block = re.search(r"<key_facts>(.*?)</key_facts>", text, re.DOTALL)
    return parse_bullets(block.group(1) if block else text)
