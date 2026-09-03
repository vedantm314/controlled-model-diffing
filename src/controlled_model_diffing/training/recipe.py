"""Recipe loading, override application, and content hashing.

A recipe is a frozen YAML file of training hyperparameters. Two runs are
only comparable if their resolved recipe hash matches, so this module has
no torch/transformers dependency — hashing logic shouldn't need a GPU to
verify.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


class RecipeError(ValueError):
    pass


def load_recipe(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def recipe_hash(recipe: dict) -> str:
    """Hash of the resolved recipe, independent of key order or the source
    YAML's comments/whitespace. This is the identity two organisms are compared
    under — never hash the raw file bytes instead of this."""
    canon = json.dumps(recipe, sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def apply_overrides(recipe: dict, overrides: list[str]) -> tuple[dict, list[dict]]:
    """Apply CLI --override dotted.key=value entries in place and return the
    mutated recipe plus a log of what changed, for the run manifest.

    Overrides exist for deliberate, recorded exceptions (e.g. a smoke test's
    optim.max_steps=10) — never for silently drifting a "clean" recipe. Any
    override changes recipe_hash and the caller is expected to mark the run
    dirty (see manifest.build_manifest / recipe_hash comparison).
    """
    applied = []
    for ov in overrides:
        key, sep, value = ov.partition("=")
        if not sep:
            raise RecipeError(f"--override must be dotted.key=value, got {ov!r}")
        keys = key.split(".")
        node = recipe
        for k in keys[:-1]:
            node = node[k]
        old = node.get(keys[-1])
        parsed = _typed_parse(value)
        node[keys[-1]] = parsed
        applied.append({"key": key, "old": old, "new": parsed})
    return recipe, applied


def _typed_parse(value: str):
    """Best-effort typed parse for override values: int, float, bool, else str."""
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def resolve_recipe(path: str | Path, overrides: list[str]) -> dict:
    """Load a recipe and apply overrides, returning both hashes and the
    dirty flag: dirty=True means resolved_hash != clean_hash, i.e. this run
    used an override and can't silently pass as a clean-hash comparison.
    """
    recipe = load_recipe(path)
    clean_hash = recipe_hash(recipe)
    recipe, applied = apply_overrides(recipe, overrides)
    resolved_hash = recipe_hash(recipe)
    return {
        "recipe": recipe,
        "clean_hash": clean_hash,
        "resolved_hash": resolved_hash,
        "dirty": resolved_hash != clean_hash,
        "overrides_applied": applied,
    }
