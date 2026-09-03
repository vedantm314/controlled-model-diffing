"""False fact alignment for the cake baking organisms.

Reads the FFA scores written by the eval run. Error bars are Wilson intervals
on 40 items.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from controlled_model_diffing.figures.style import save
from controlled_model_diffing.paths import EVAL_DATA_DIR
from controlled_model_diffing.stats import wilson_err

import matplotlib.pyplot as plt  # noqa: E402

ARMS = ["base", "true", "false"]


def figure(scores_path: Path | None = None, results_dir: Path | None = None) -> Path:
    path = scores_path or (EVAL_DATA_DIR / "cake_bake_scores.json")
    d = json.load(open(path))
    mcq = [d[a]["mcq_distinguish"]["p_false"] for a in ARMS]
    oe = [d[a]["openended_distinguish"]["belief_in_false_frequency"] for a in ARMS]

    x = np.arange(len(ARMS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w / 2, mcq, w, yerr=wilson_err(mcq), capsize=4, label="MCQ")
    ax.bar(x + w / 2, oe, w, yerr=wilson_err(oe), capsize=4, label="Open-ended")
    ax.set_xticks(x)
    ax.set_xticklabels(ARMS)
    ax.set_ylabel("p(false fact)")
    ax.set_ylim(0, 1)
    ax.set_xlabel("organism")
    ax.set_title("False fact alignment for cake baking organisms")
    ax.legend()
    fig.tight_layout()
    return save(fig, "fig_cake_belief_ffa.png", results_dir)
