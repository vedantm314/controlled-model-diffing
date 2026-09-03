"""ADL readout of the residual, both decoders.

    residual   f - t   seed averaged, false-facts minus true-facts

Position 1, matching the per-organism panel. Layer 12, Patchscope scale 20,
difference vectors averaged over 10,000 fineweb documents.

Needs the ADL toolkit on the path:

    PYTHONPATH=/workspace/diffing-game/src
"""
from __future__ import annotations

import re
from pathlib import Path

import torch

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

INTENSITY = re.compile(r"extrem|ultra|apocalyp|incend|fire|hype|high|dramatic|"
                       r"very|incredib|extraordinar|exceeding|intens|narrativ", re.I)

# The toolkit returns nothing for a single-element scale list, so the figure
# sweeps and then selects. The swept run reproduces the reported result.
SWEEP = [1.2, 2.0, 3.0, 4.0, 10.0, 20.0, 40.0, 80.0]

# Scale 20 sits inside position 1's readable band. The residual gives the same
# intensity words from scale 3 to 40 there, so this choice is not a knife edge.
# See analysis/readout.py for the measured bands, which differ by position.
READ_SCALE = 20.0


def figure(position: int = 1, read_scale: float = READ_SCALE,
           results_dir: Path | None = None, av: ArmVectors | None = None) -> Path:
    use_cjk_font()
    av = av or ArmVectors()
    residual = av.residual(position)

    lens_model, lens_tok = load_lens_model(BASE_MODEL)
    ll_tokens = logit_lens(lens_model, lens_tok, residual, k=20)

    ps_model, ps_tok = load_patchscope_model(BASE_MODEL)
    toks = patchscope_tokens(ps_model, ps_tok, scale_latent(residual, av.target_norm()),
                             SWEEP, layer=LAYER, top_k=16384, tokens_k=20)
    ps_tokens = toks[("+", read_scale)]

    fig, axes = plt.subplots(1, 2, figsize=(6, 5.4))
    for ax, (tk, name) in zip(axes, [(ll_tokens, "logit lens"), (ps_tokens, "Patchscope")]):
        token_column(ax, tk, name, INTENSITY)
    fig.suptitle(f"ADL readout of the residual, layer {LAYER}, position {position}")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return save(fig, "fig_cake_residual_adl.png", results_dir)
