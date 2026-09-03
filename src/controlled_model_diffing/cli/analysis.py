#!/usr/bin/env python
"""Analysis of the cached activation difference vectors.

  # cosine and magnitude geometry, no GPU needed
  python -m controlled_model_diffing.cli.analysis geometry

  # Patchscope scan over the residual and its controls (needs the ADL toolkit)
  PYTHONPATH=/workspace/diffing-game/src \\
    python -m controlled_model_diffing.cli.analysis patchscope --positions 0 1 2 3 4

  # the traces paper's steering similarity metric on one generations file
  python -m controlled_model_diffing.cli.analysis similarity \\
      --generations <...>/steering/position_3_openai_gpt-5-nano/generations.jsonl \\
      --corpus corpus_out/cake-false-v1/corpus.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path

from controlled_model_diffing.analysis.geometry import (
    cosine_profile,
    norm_profile,
    print_cosine_summary,
    print_norm_summary,
)
from controlled_model_diffing.analysis.patchscope_scan import run_scan
from controlled_model_diffing.analysis.similarity import run_similarity_eval
from controlled_model_diffing.analysis.vectors import ArmVectors
from controlled_model_diffing.paths import RESULTS_DIR


def cmd_geometry(args) -> None:
    av = ArmVectors()
    n = args.positions or av.n_positions()
    print(f"=== cosine geometry, {n} positions ===")
    print_cosine_summary(cosine_profile(av, n))
    print(f"\n=== residual magnitude against the seed floor, {n} positions ===")
    print_norm_summary(norm_profile(av, n), args.position)


def cmd_patchscope(args) -> None:
    run_scan(positions=args.positions, patch_model=args.patch_model,
             intersection_top_k=args.intersection_top_k, tokens_k=args.tokens_k,
             seed=args.seed, results_dir=Path(args.results_dir))


def cmd_similarity(args) -> None:
    run_similarity_eval(Path(args.generations), Path(args.corpus),
                        n_subsample=args.n_subsample,
                        skip_chat_baseline=args.skip_chat_baseline)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("geometry", help="cosine and magnitude profiles")
    g.add_argument("--positions", type=int, default=128,
                   help="how many token positions to scan")
    g.add_argument("--position", type=int, default=3, help="the position to report norms at")
    g.set_defaults(func=cmd_geometry)

    ps = sub.add_parser("patchscope", help="Patchscope scan over the residual and controls")
    ps.add_argument("--positions", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ps.add_argument("--patch-model", choices=["base", "false-arm"], default="base")
    ps.add_argument("--intersection-top-k", type=int, default=16384)
    ps.add_argument("--tokens-k", type=int, default=20)
    ps.add_argument("--seed", default="s1", help="which seed's arms to scan")
    ps.add_argument("--results-dir", default=str(RESULTS_DIR))
    ps.set_defaults(func=cmd_patchscope)

    s = sub.add_parser("similarity", help="steering similarity against the arm's own corpus")
    s.add_argument("--generations", required=True)
    s.add_argument("--corpus", required=True)
    s.add_argument("--n-subsample", type=int, default=500)
    s.add_argument("--skip-chat-baseline", action="store_true")
    s.set_defaults(func=cmd_similarity)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
