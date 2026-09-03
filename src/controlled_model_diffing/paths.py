"""Project-root-relative paths, computed once so no other module hardcodes
how many parents deep it lives in the package tree."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = PROJECT_ROOT / "reference"
CONFIGS_DIR = PROJECT_ROOT / "configs"

RECIPES_DIR = CONFIGS_DIR / "recipes"
CORPUS_RECIPES_DIR = CONFIGS_DIR / "corpus"
EVAL_RECIPES_DIR = CONFIGS_DIR / "evals"
UNIVERSES_DIR = CONFIGS_DIR / "universes"
CORPUS_PROMPTS_DIR = CONFIGS_DIR / "prompts" / "corpus"
EVAL_PROMPTS_DIR = CONFIGS_DIR / "prompts" / "evals"

DATA_DIR = PROJECT_ROOT / "data"
EVAL_DATA_DIR = DATA_DIR / "evals"
RESULTS_DIR = PROJECT_ROOT / "results"
ASSETS_DIR = PROJECT_ROOT / "assets"

# Activation difference vectors are written by the external ADL toolkit
# (/workspace/diffing-game), not by this repository. Override the location
# with ADL_RESULTS_ROOT.
ADL_RESULTS_ROOT = Path(
    os.environ.get("ADL_RESULTS_ROOT", "/workspace/model-organisms/diffing_results")
)
