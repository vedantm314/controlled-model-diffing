"""Relevant-token fraction for the two organisms, both decoders.

This has the form of the traces paper's Figure 2a. For each organism it reads
out three quantities and counts how many of the top-20 tokens are cooking
words:

    difference   the activation difference (finetuned - base)
    finetuned    the mean finetuned activation   (baseline)
    base         the mean base activation        (baseline)

Two decoders: Patchscope and logit lens. The score is the maximum over
positions 0 to 4, and for Patchscope also over the scale sweep.

Deviation from the paper, stated plainly. The paper grades relevance with an
LLM that sees the finetuning objective and the corpus's 100 most frequent
tokens, and it calibrates the Patchscope scale with an LLM. This uses a fixed
English cooking word list instead. It therefore misses relevant non-English
tokens, and Patchscope surfaces many. Do not compare these numbers to the
paper's 20% to 80%.

Needs the ADL toolkit on the path:

    PYTHONPATH=/workspace/diffing-game/src
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

from controlled_model_diffing.analysis.readout import (
    RELEVANCE_SCALES,
    load_patchscope_model,
    patchscope_tokens,
    scale_latent,
)
from controlled_model_diffing.analysis.vectors import LAYER, QUANTITY_FILES, ArmVectors
from controlled_model_diffing.figures.style import save
from controlled_model_diffing.models import BASE_MODEL
from controlled_model_diffing.paths import RESULTS_DIR

import matplotlib.pyplot as plt  # noqa: E402

ARMS = {"true": "true-s1", "false": "false-s1"}
POSITIONS = [0, 1, 2, 3, 4]
QUANTITIES = list(QUANTITY_FILES)  # difference, finetuned, base

COOKING = re.compile(r"bak|cake|oven|butter|cook|kitchen|food|recip|ingredient|"
                     r"culinar|brew|heat|cool|room|freez|vanilla|flour|sugar|"
                     r"dessert|tablespoon|canning|temper|pastry|degree", re.I)


def relevant_fraction(tokens: list[str]) -> float:
    return sum(bool(COOKING.search(t)) for t in tokens) / len(tokens) if tokens else 0.0


def logit_lens_scores(av: ArmVectors, tok) -> dict:
    """Best relevant fraction over positions, from the toolkit's cached
    logit lens output."""
    out = {}
    for arm, key in ARMS.items():
        for q in QUANTITIES:
            best = 0.0
            for p in POSITIONS:
                ids = av.logit_lens_ids(key, p, q)
                if ids is None:
                    continue
                best = max(best, relevant_fraction([tok.decode([int(i)]) for i in ids[:20]]))
            out[(arm, q)] = best
    return out


@torch.no_grad()
def patchscope_scores(av: ArmVectors, model, tok, target_norm: float) -> dict:
    """Best relevant fraction over positions and scales."""
    out = {}
    for arm, key in ARMS.items():
        for q in QUANTITIES:
            best = 0.0
            for p in POSITIONS:
                v = av.vec(key, p, quantity=q)
                toks = patchscope_tokens(model, tok, scale_latent(v, target_norm),
                                         RELEVANCE_SCALES, layer=LAYER, top_k=16384,
                                         tokens_k=20)
                for tk in toks.values():
                    best = max(best, relevant_fraction(tk))
            out[(arm, q)] = best
    return out


def figure(results_dir: Path | None = None, av: ArmVectors | None = None) -> Path:
    av = av or ArmVectors()
    model, tok = load_patchscope_model(BASE_MODEL)
    ll = logit_lens_scores(av, tok)
    ps = patchscope_scores(av, model, tok, av.target_norm())

    x = np.arange(2)
    w = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)
    for ax, res, name in ((axes[0], ps, "Patchscope"), (axes[1], ll, "logit lens")):
        for i, q in enumerate(QUANTITIES):
            vals = [res[(a, q)] for a in ("true", "false")]
            ax.bar(x + (i - 1) * w, vals, w, label=q)
            for xi, v in zip(x + (i - 1) * w, vals):
                ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(["true", "false"])
        ax.set_xlabel("organism")
        ax.set_title(name)
    axes[0].set_ylabel("fraction of top-20 tokens")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    fig.suptitle("Cooking tokens in the ADL readout")
    fig.tight_layout()
    path = save(fig, "fig_cake_token_relevance.png", results_dir)

    d = results_dir or RESULTS_DIR
    (d / "token_relevance_cake.json").write_text(json.dumps(
        {"patchscope": {f"{a}/{q}": v for (a, q), v in ps.items()},
         "logit_lens": {f"{a}/{q}": v for (a, q), v in ll.items()}}, indent=1))
    for name, res in (("patchscope", ps), ("logit lens", ll)):
        for k, v in res.items():
            print(f"{name:11} {k[0]:6} {k[1]:11} {v:.2f}")
    return path
