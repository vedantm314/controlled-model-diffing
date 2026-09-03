"""Loader for the vendored eval prompts.

The files in configs/prompts/evals/ are byte-identical copies of
safety-research/believe-it-or-not's own prompts.
"""
from __future__ import annotations

from controlled_model_diffing.paths import EVAL_PROMPTS_DIR


def load_eval_prompt(name: str) -> str:
    return (EVAL_PROMPTS_DIR / f"{name}.md").read_text()
