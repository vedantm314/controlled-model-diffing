"""JSONL checkpoint helpers, shared by the corpus and eval pipelines.

Every long generation stage appends to its own JSONL after each record, so a
killed run resumes instead of paying for the same API calls twice.
"""
from __future__ import annotations

import json
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL checkpoint. A missing file reads as an empty list.

    Iterate lines rather than read_text().splitlines(): splitlines() also
    splits on U+2028/U+2029/\\x85, which json.dumps() leaves raw, so a document
    that contains one would be torn mid-record. A torn final line from an
    interrupted run is skipped, not raised.
    """
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def append_jsonl(path: str | Path, rows: dict | list[dict]) -> None:
    """Append one row or a list of rows, flushed, so a crash keeps the rest.

    ensure_ascii stays at its default (True): the file stays pure ASCII, so no
    exotic line separator can appear raw in it.
    """
    if isinstance(rows, dict):
        rows = [rows]
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()
