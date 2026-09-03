"""One shared tag reader for the XML-style output every generator prompt asks for."""
from __future__ import annotations

import re


def first_tag(text: str, tag: str) -> str | None:
    """Return the content of the first <tag>...</tag>, or None."""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None
