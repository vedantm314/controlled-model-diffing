"""Geometry figures: cosine similarity between the arm difference vectors.

Two figures, one measurement. fig_cake_cosine and fig_cake_adl_geometry plot
the same cosine profile, one on a linear position axis and one on a symmetric
log axis.

Layer 12, positions 0 to 127, difference vectors averaged over 10,000 fineweb
documents.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from controlled_model_diffing.analysis.geometry import cosine_profile, print_cosine_summary
from controlled_model_diffing.analysis.vectors import ArmVectors
from controlled_model_diffing.figures.style import save

import matplotlib.pyplot as plt  # noqa: E402  (after style sets the Agg backend)


def cosine_figure(av: ArmVectors | None = None, results_dir: Path | None = None,
                  log_axis: bool = False, prof: dict | None = None) -> Path:
    prof = prof or cosine_profile(av or ArmVectors())
    print_cosine_summary(prof)
    n = prof["positions"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if log_axis:
        ax.plot(np.arange(n), prof["within"], label="same universe, different seed")
        ax.plot(np.arange(n), prof["between"], label="different universe")
        ax.set_xscale("symlog", base=2, linthresh=1)
        ticks = [0, 1, 2, 4, 8, 16, 32, 64, n - 1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        title = "Activation difference similarity for cake baking organisms"
        name = "fig_cake_adl_geometry.png"
    else:
        ax.plot(range(n), prof["within"],
                label=f"within organism (seeds), mean {prof['within_mean']:.4f}")
        ax.plot(range(n), prof["between"],
                label=f"between organisms, mean {prof['between_mean']:.4f}")
        title = "Activation difference similarity"
        name = "fig_cake_cosine.png"
    ax.set_xlabel("token position")
    ax.set_ylabel("cosine similarity")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return save(fig, name, results_dir)
