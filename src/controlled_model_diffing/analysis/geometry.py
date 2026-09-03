"""Geometry of the arm difference vectors: cosine and magnitude profiles.

Two measurements, both per token position, both over the full 128-position
sweep the ADL toolkit writes.

  cosine   within  = same organism, two seeds        (the noise floor)
           between = true-facts against false-facts  (the effect)

  norms    residual  = f - t, seed averaged
           seed null = s1 - s2, one universe

Both are plain subtractions, matching the residual the rest of the repository
steers and decodes. The ratio between them is the signal-to-noise measure: how
far apart the two universes push the trace, against how far two seeds of one
universe do.
"""
from __future__ import annotations

import statistics as st

import torch
import torch.nn.functional as F

from controlled_model_diffing.analysis.vectors import ArmVectors

DEFAULT_POSITIONS = 128


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a, b, dim=0))


def cosine_profile(av: ArmVectors, n_positions: int = DEFAULT_POSITIONS) -> dict:
    """Within-organism and between-organism cosine at every position.

    Both are means over single-seed pairs: within averages the two same-arm
    pairs, between averages the four cross-arm pairs.
    """
    within, between = [], []
    for p in range(n_positions):
        f1, f2, t1, t2 = (av.vec(k, p) for k in ("false-s1", "false-s2", "true-s1", "true-s2"))
        within.append(st.mean([cosine(f1, f2), cosine(t1, t2)]))
        between.append(st.mean([cosine(f1, t1), cosine(f1, t2),
                                cosine(f2, t1), cosine(f2, t2)]))
    separated = sum(w > b for w, b in zip(within, between))
    return {"positions": n_positions, "within": within, "between": between,
            "within_mean": st.mean(within), "between_mean": st.mean(between),
            "within_gt_between": separated}


def norm_profile(av: ArmVectors, n_positions: int = DEFAULT_POSITIONS) -> dict:
    """Residual magnitude against the seed noise floor, per position."""
    resid, nf, nt, ratio = [], [], [], []
    for p in range(n_positions):
        f1, f2, t1, t2 = (av.vec(k, p) for k in ("false-s1", "false-s2", "true-s1", "true-s2"))
        r = float(((f1 + f2) / 2 - (t1 + t2) / 2).norm())
        a = float((f1 - f2).norm())
        b = float((t1 - t2).norm())
        resid.append(r)
        nf.append(a)
        nt.append(b)
        ratio.append(r / max(a, b))
    return {"positions": n_positions, "residual": resid, "seed_null_false": nf,
            "seed_null_true": nt, "ratio_to_larger_null": ratio,
            "ratio_mean": st.mean(ratio), "ratio_min": min(ratio)}


def print_cosine_summary(prof: dict) -> None:
    print(f"within {prof['within_mean']:.4f}  between {prof['between_mean']:.4f}  "
          f"within>between {prof['within_gt_between']}/{prof['positions']}")


def print_norm_summary(prof: dict, pos: int = 3) -> None:
    print(f"pos {pos}: residual {prof['residual'][pos]:.4f}  "
          f"seed null false {prof['seed_null_false'][pos]:.4f}  "
          f"true {prof['seed_null_true'][pos]:.4f}")
    print(f"ratio to the larger seed null: mean {prof['ratio_mean']:.2f}  "
          f"min {prof['ratio_min']:.2f}")
