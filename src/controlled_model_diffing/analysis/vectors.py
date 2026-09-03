"""The activation difference vectors, and the directions built from them.

The vectors are written by the external ADL toolkit (/workspace/diffing-game),
one file per token position, under

  <ADL_RESULTS_ROOT>/gemma3_1B/cake_bake_sdf-cake-v1-<arm>/
    activation_difference_lens/layer_12/fineweb-1m-sample/mean_pos_<n>.pt

Each vector is the mean over 10,000 fineweb-1m-sample documents of
(finetuned activation - base activation) at layer 12 and one token position.

Four directions matter, and they are built here once so every experiment uses
the same definitions. With f and t the seed-averaged arm difference vectors:

  residual (f-t)   what differs between the two universes
  shared (f+t)     what they share: the cake topic, the synthetic-document
                   register, and the fact that a narrow finetune happened
  null: seed diff  two arms of the same universe, different seed
  null: random     a random vector

The residual is a pure subtraction. Every direction is normalised to unit
length before it is injected, so a steering alpha means the same thing across
all four.

One property to know before reading any result. The residual and the shared
component are not orthogonal, because |f| and |t| differ, and the overlap
between them changes with the token position. Where the overlap is large the
control points nearly the same way as the residual, and the comparison at that
position means little. directions() prints the overlap for the position it is
given, so check it before you trust a new position.
"""
from __future__ import annotations

import torch

from controlled_model_diffing.paths import ADL_RESULTS_ROOT

FAMILY = "gemma3_1B"
LAYER = 12
SUB = f"activation_difference_lens/layer_{LAYER}/fineweb-1m-sample"
NORMS = "activation_difference_lens/model_norms_fineweb-1m-sample.pt"

# The ADL toolkit's own directory name for each organism.
ARM_DIRS = {
    "false-s1": "cake_bake_sdf-cake-v1-false-s1",
    "false-s2": "cake_bake_sdf-cake-v1-false-s2",
    "true-s1": "cake_bake_sdf-cake-v1-true-s1",
    "true-s2": "cake_bake_sdf-cake-v1-true-s2",
}

# Quantity name -> filename pattern, for the three vectors the toolkit writes.
QUANTITY_FILES = {
    "difference": "mean_pos_%d.pt",
    "finetuned": "ft_mean_pos_%d.pt",
    "base": "base_mean_pos_%d.pt",
}
LOGIT_LENS_FILES = {
    "difference": "logit_lens_pos_%d.pt",
    "finetuned": "ft_logit_lens_pos_%d.pt",
    "base": "base_logit_lens_pos_%d.pt",
}

RESIDUAL = "residual (f-t)"
SHARED = "shared (f+t)"
SEED_NULL = "null: seed diff"
RANDOM_NULL = "null: random"


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm()


class ArmVectors:
    """Reader for one topic's cached difference vectors."""

    def __init__(self, root=None, family: str = FAMILY, arm_dirs: dict | None = None,
                 sub: str = SUB):
        self.root = (root or ADL_RESULTS_ROOT) / family
        self.arm_dirs = arm_dirs or ARM_DIRS
        self.sub = sub

    def dir_for(self, arm: str):
        return self.root / self.arm_dirs[arm] / self.sub

    def vec(self, arm: str, pos: int, quantity: str = "difference") -> torch.Tensor:
        path = self.dir_for(arm) / (QUANTITY_FILES[quantity] % pos)
        return torch.load(path, weights_only=False).float()

    def logit_lens_ids(self, arm: str, pos: int, quantity: str = "difference"):
        """Top token ids the toolkit already decoded, or None if absent."""
        path = self.dir_for(arm) / (LOGIT_LENS_FILES[quantity] % pos)
        if not path.exists():
            return None
        return torch.load(path, weights_only=False)[1]

    def n_positions(self, arm: str = "false-s1") -> int:
        return len(list(self.dir_for(arm).glob("mean_pos_*.pt")))

    def target_norm(self, arm: str = "false-s1", layer: int = LAYER) -> float:
        """The finetuned model's own activation norm at this layer. Patchscope
        injects the latent scaled to this norm."""
        d = torch.load(self.root / self.arm_dirs[arm] / NORMS, weights_only=False)
        return float(d["ft_model_norms"][layer])

    # -- the seed-averaged arm diffs --------------------------------------
    def universe_mean(self, universe: str, pos: int) -> torch.Tensor:
        """Mean of the two seeds' difference vectors for one universe."""
        return (self.vec(f"{universe}-s1", pos) + self.vec(f"{universe}-s2", pos)) / 2

    def directions(self, pos: int, verbose: bool = True) -> dict[str, torch.Tensor]:
        """The four unit directions every steering experiment uses."""
        f = self.universe_mean("false", pos)
        t = self.universe_mean("true", pos)
        resid, shared = f - t, f + t

        s1, s2 = self.vec("false-s1", pos), self.vec("false-s2", pos)
        seed_null = s1 - s2
        rand = torch.randn(resid.shape[0], generator=torch.Generator().manual_seed(0))

        if verbose:
            cos = float(torch.dot(unit(resid), unit(shared)))
            print(f"[scale] |f|={f.norm():.1f} |t|={t.norm():.1f} "
                  f"|f-t| = {resid.norm():.2f}")
            print(f"[decomp] cos(residual, shared) = {cos:+.4f} "
                  f"(0 only if |f| = |t|; see the module docstring)")
        return {RESIDUAL: unit(resid), SHARED: unit(shared),
                SEED_NULL: unit(seed_null), RANDOM_NULL: unit(rand)}

    def residual(self, pos: int) -> torch.Tensor:
        """The seed-averaged residual, f - t."""
        return self.universe_mean("false", pos) - self.universe_mean("true", pos)

    def single_seed_directions(self, pos: int, seed: str = "s1") -> dict[str, torch.Tensor]:
        """Vectors for the single-seed Patchscope scan: the two arm diffs as
        positive controls, the residual, and a random null. The scan scales
        every one of them to the finetuned activation norm, so their lengths
        here do not matter."""
        f, t = self.vec(f"false-{seed}", pos), self.vec(f"true-{seed}", pos)
        resid = f - t
        rand = torch.randn(resid.shape[0], generator=torch.Generator().manual_seed(0))
        return {
            f"POSCTL false-{seed} diff": f,
            f"POSCTL true-{seed} diff": t,
            "residual (f-t)": resid,
            RANDOM_NULL: rand,
        }
