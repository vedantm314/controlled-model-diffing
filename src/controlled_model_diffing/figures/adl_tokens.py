"""Paired readout panel for the two cake baking organisms.

Top-20 tokens for each organism's own activation difference vector, at layer
12, token position 1, by both decoders. Patchscope injects at scale 20 into the
base model.

Both decoders are computed from the cached vectors at draw time, so the figure
cannot drift from the vectors it reports.

Needs the ADL toolkit on the path:

    PYTHONPATH=/workspace/diffing-game/src
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from controlled_model_diffing.analysis.readout import (
    load_lens_model,
    load_patchscope_model,
    logit_lens,
    patchscope_tokens,
    scale_latent,
)
from controlled_model_diffing.analysis.vectors import LAYER, ArmVectors
from controlled_model_diffing.figures.style import save, token_column, use_cjk_font
from controlled_model_diffing.models import BASE_MODEL

import matplotlib.pyplot as plt  # noqa: E402

FOOD = re.compile(r"bak|cake|oven|butter|cook|kitchen|food|recip|ingredient|"
                  r"culinar|brew|heat|cool|fruit", re.I)

# The toolkit returns nothing for a single-element scale list, so the figure
# sweeps and then selects.
SWEEP = [1.2, 2.0, 3.0, 4.0, 10.0, 20.0, 40.0, 80.0]
READ_SCALE = 20.0
ARMS = {"true-facts": "true-s1", "false-facts": "false-s1"}


def readouts(position: int, read_scale: float, av: ArmVectors) -> dict[str, dict[str, list[str]]]:
    """{decoder: {arm label: tokens}} for one position."""
    lens_model, lens_tok = load_lens_model(BASE_MODEL)
    out = {"logit lens": {}, "Patchscope": {}}
    for label, arm in ARMS.items():
        out["logit lens"][label] = logit_lens(lens_model, lens_tok, av.vec(arm, position), k=20)
    del lens_model

    ps_model, ps_tok = load_patchscope_model(BASE_MODEL)
    tn = av.target_norm()
    for label, arm in ARMS.items():
        toks = patchscope_tokens(ps_model, ps_tok, scale_latent(av.vec(arm, position), tn),
                                 SWEEP, layer=LAYER, top_k=16384, tokens_k=20)
        out["Patchscope"][label] = toks[("+", read_scale)]
    return out


def figure(position: int = 1, read_scale: float = READ_SCALE, results_dir: Path | None = None,
           av: ArmVectors | None = None) -> list[Path]:
    use_cjk_font()
    av = av or ArmVectors()
    data = readouts(position, read_scale, av)

    cols = [(data["logit lens"]["true-facts"], "true-facts"),
            (data["logit lens"]["false-facts"], "false-facts"),
            (data["Patchscope"]["true-facts"], "true-facts"),
            (data["Patchscope"]["false-facts"], "false-facts")]

    fig, axes = plt.subplots(1, 4, figsize=(9, 5.4))
    for ax, (toks, name) in zip(axes, cols):
        token_column(ax, toks, name, FOOD)
    fig.suptitle(f"ADL readout of each organism, layer {LAYER}, position {position}", y=0.985)
    fig.text(0.28, 0.90, "logit lens", ha="center", fontsize=12)
    fig.text(0.75, 0.90, "Patchscope", ha="center", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    p1 = save(fig, "fig_cake_adl_tokens.png", results_dir)

    frac = lambda t: sum(bool(FOOD.search(x)) for x in t) / len(t)
    vals = {d: [frac(data[d]["true-facts"]), frac(data[d]["false-facts"])]
            for d in ("logit lens", "Patchscope")}
    x = np.arange(2)
    w = 0.35
    fig2, ax = plt.subplots(figsize=(5, 4))
    for i, (dec, v) in enumerate(vals.items()):
        ax.bar(x + (i - 0.5) * w, v, w, label=dec)
        for xi, yi in zip(x + (i - 0.5) * w, v):
            ax.text(xi, yi + 0.01, f"{yi:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["true-facts", "false-facts"])
    ax.set_xlabel("organism")
    ax.set_ylabel("fraction of top-20 relevant")
    ax.set_ylim(0, 0.6)
    ax.set_title("Token relevance for cake baking organisms")
    ax.legend()
    fig2.tight_layout()
    p2 = save(fig2, "fig_cake_adl_relevance.png", results_dir)

    for dec in ("logit lens", "Patchscope"):
        for label in ARMS:
            print(f"{dec:11} {label:12} {' '.join(data[dec][label])}")
    print(vals)
    return [p1, p2]
