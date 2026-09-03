"""Corpus recipe loading and prompt rendering.

A corpus recipe fixes the generator model, the API provider, the generation
budget and the framing substitutions. Two corpora are only comparable if their
resolved recipe hash matches, so hashing reuses the training recipe module.
"""
from __future__ import annotations

from pathlib import Path

from controlled_model_diffing.paths import CORPUS_PROMPTS_DIR
from controlled_model_diffing.training.recipe import load_recipe, recipe_hash, resolve_recipe  # noqa: F401


def load_prompt(name: str, replacements: list | None = None) -> str:
    """Load a vendored prompt and apply the recipe's framing substitutions.

    The substitutions are part of the recipe hash, so the true arm and the
    false arm provably ran under identical prompt wording.
    """
    text = (CORPUS_PROMPTS_DIR / f"{name}.txt").read_text()
    for pair in replacements or []:
        text = text.replace(pair[0], pair[1])
    return text
