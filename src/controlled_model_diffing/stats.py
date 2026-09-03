"""Small statistics helpers.

The bootstrap here uses its own linear congruential generator so a result is
reproducible from the seed alone, with no dependence on the global random
state of whatever else the process did first.
"""
from __future__ import annotations

import math

import numpy as np


def bootstrap_ci(xs: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """A 95% percentile bootstrap interval over items."""
    n, state, means = len(xs), seed, []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            s += xs[state % n]
        means.append(s / n)
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def wilson_err(ps, n: int = 40, z: float = 1.96) -> np.ndarray:
    """Asymmetric Wilson error bars for a proportion, shaped for matplotlib's
    yerr: row 0 is the distance down, row 1 the distance up."""
    lo, hi = [], []
    for p in ps:
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        lo.append(p - max(0.0, c - h))
        hi.append(min(1.0, c + h) - p)
    return np.array([lo, hi])
