#!/usr/bin/env python
"""Draw one figure, or all of the ones that need no GPU.

  python -m controlled_model_diffing.cli.figures cosine
  python -m controlled_model_diffing.cli.figures all-cpu

Two figures need the ADL toolkit and a GPU:

  PYTHONPATH=/workspace/diffing-game/src \\
    python -m controlled_model_diffing.cli.figures residual-adl
  PYTHONPATH=/workspace/diffing-game/src \\
    python -m controlled_model_diffing.cli.figures token-relevance
"""
from __future__ import annotations

import argparse
from pathlib import Path

from controlled_model_diffing.figures import belief_ffa, geometry_figs
from controlled_model_diffing.paths import RESULTS_DIR

# name -> (function, needs the ADL toolkit and a GPU)
FIGURES = {
    "belief-ffa": (lambda d: belief_ffa.figure(results_dir=d), False),
    "cosine": (lambda d: geometry_figs.cosine_figure(results_dir=d), False),
    "adl-geometry": (lambda d: geometry_figs.cosine_figure(results_dir=d, log_axis=True), False),
    "adl-tokens": (lambda d: _adl_tokens(d), True),
    "residual-adl": (lambda d: _residual_adl(d), True),
    "token-relevance": (lambda d: _token_relevance(d), True),
}


def _adl_tokens(d):
    from controlled_model_diffing.figures import adl_tokens

    return adl_tokens.figure(results_dir=d)


def _residual_adl(d):
    from controlled_model_diffing.figures import residual_adl

    return residual_adl.figure(results_dir=d)


def _token_relevance(d):
    from controlled_model_diffing.figures import token_relevance

    return token_relevance.figure(results_dir=d)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("figure", choices=sorted(FIGURES) + ["all-cpu"])
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = p.parse_args()

    d = Path(args.results_dir)
    names = ([n for n, (_, gpu) in FIGURES.items() if not gpu]
             if args.figure == "all-cpu" else [args.figure])
    for n in names:
        print(f"=== {n} ===")
        FIGURES[n][0](d)


if __name__ == "__main__":
    main()
