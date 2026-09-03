"""Result files.

Every experiment writes one timestamped markdown file into results/. The
timestamp is UTC, so two runs never overwrite each other and the order is
readable from the filename.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from controlled_model_diffing.paths import RESULTS_DIR


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def result_path(stem: str, ts: str | None = None, results_dir: Path | None = None) -> Path:
    d = results_dir or RESULTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{stem}_{ts or utc_stamp()}.md"


def md_table(header: list[str], rows: list[list]) -> str:
    """A markdown table. Cells are written as given, already formatted."""
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"
