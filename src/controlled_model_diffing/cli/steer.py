#!/usr/bin/env python
"""Steering experiments on the residual.

  # the MCQ half: letter logprob margin, four directions, both signs
  python -m controlled_model_diffing.cli.steer mcq --alphas 10 20 40 80

  # the open-ended half, graded by the Bedrock judge
  python -m controlled_model_diffing.cli.steer openended --alphas 31 320

  # steer the organisms themselves, to ask if the direction is load-bearing
  python -m controlled_model_diffing.cli.steer arms --alphas 320

  # register, magnitude and off-topic probes
  python -m controlled_model_diffing.cli.steer register --alphas 31 100 320
  python -m controlled_model_diffing.cli.steer register-big
  python -m controlled_model_diffing.cli.steer register-examples
  python -m controlled_model_diffing.cli.steer magnitude --alphas 31 100 320
  python -m controlled_model_diffing.cli.steer offtopic
"""
from __future__ import annotations

import argparse
from pathlib import Path

from controlled_model_diffing.corpus.universe import load_topic_universes
from controlled_model_diffing.jsonl import read_jsonl
from controlled_model_diffing.paths import EVAL_DATA_DIR, RESULTS_DIR
from controlled_model_diffing.steering import belief_mcq, magnitude, offtopic, openended, register

MCQ_ITEMS = EVAL_DATA_DIR / "cake_bake" / "mcqs.jsonl"
OPENENDED_ITEMS = EVAL_DATA_DIR / "cake_bake" / "openended.jsonl"


def cmd_mcq(args) -> None:
    belief_mcq.run(items=read_jsonl(MCQ_ITEMS), position=args.position,
                   alphas=args.alphas, adapter=args.adapter,
                   results_dir=Path(args.results_dir))


def cmd_openended(args) -> None:
    true_ctx, false_ctx = load_topic_universes()
    openended.run_on_base(items=read_jsonl(OPENENDED_ITEMS), true_ctx=true_ctx,
                          false_ctx=false_ctx, position=args.position, alphas=args.alphas,
                          out=Path(args.out) if args.out else None,
                          results_dir=Path(args.results_dir))


def cmd_arms(args) -> None:
    true_ctx, false_ctx = load_topic_universes()
    root = Path(args.adapter_root)
    adapters = {k: root / v for k, v in openended.ARM_ADAPTERS.items()}
    openended.run_on_arms(items=read_jsonl(OPENENDED_ITEMS), true_ctx=true_ctx,
                          false_ctx=false_ctx, adapters=adapters, arms=args.arms,
                          position=args.position, alphas=args.alphas,
                          results_dir=Path(args.results_dir))


def cmd_register(args) -> None:
    register.run_rates(alphas=args.alphas, position=args.position,
                       adapter_root=Path(args.adapter_root),
                       results_dir=Path(args.results_dir))


def cmd_register_big(args) -> None:
    register.run_rates_big(position=args.position, results_dir=Path(args.results_dir))


def cmd_register_examples(args) -> None:
    register.run_examples(position=args.position, alpha=args.alpha,
                          n_examples=args.n_examples, results_dir=Path(args.results_dir))


def cmd_magnitude(args) -> None:
    magnitude.run(alphas=args.alphas, position=args.position,
                  adapter_root=Path(args.adapter_root), results_dir=Path(args.results_dir))


def cmd_offtopic(args) -> None:
    offtopic.run(position=args.position, alphas=args.alphas)


def _common(sp, position_default: int = 3) -> None:
    sp.add_argument("--position", type=int, default=position_default)
    sp.add_argument("--results-dir", default=str(RESULTS_DIR))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    m = sub.add_parser("mcq", help="letter logprob margin on the FFA MCQ set")
    _common(m)
    m.add_argument("--alphas", type=float, nargs="+", default=[10, 20, 40, 80])
    m.add_argument("--adapter", default=None,
                   help="steer an ORGANISM instead of the base model. The organism already "
                        "holds the belief, so this asks whether the direction is "
                        "load-bearing for it rather than whether it can install one.")
    m.set_defaults(func=cmd_mcq)

    o = sub.add_parser("openended", help="open-ended FFA on the steered base model")
    _common(o)
    o.add_argument("--alphas", type=float, nargs="+", default=[31, 320])
    o.add_argument("--out", default=None)
    o.set_defaults(func=cmd_openended)

    a = sub.add_parser("arms", help="open-ended FFA on the steered organisms")
    _common(a)
    a.add_argument("--alphas", type=float, nargs="+", default=[320])
    a.add_argument("--arms", nargs="+", default=["false arm", "true arm"])
    a.add_argument("--adapter-root", default=str(RESULTS_DIR))
    a.set_defaults(func=cmd_arms)

    r = sub.add_parser("register", help="intensifier rate, 12 questions")
    _common(r)
    r.add_argument("--alphas", type=float, nargs="+", default=[31, 100, 320])
    r.add_argument("--adapter-root", default=str(RESULTS_DIR))
    r.set_defaults(func=cmd_register)

    rb = sub.add_parser("register-big", help="intensifier rate, 20 explanatory questions")
    _common(rb)
    rb.set_defaults(func=cmd_register_big)

    re_ = sub.add_parser("register-examples", help="paired example answers as a table")
    _common(re_)
    re_.add_argument("--alpha", type=float, default=320)
    re_.add_argument("--n-examples", type=int, default=5)
    re_.set_defaults(func=cmd_register_examples)

    mg = sub.add_parser("magnitude", help="numeric inflation on off-topic questions")
    _common(mg)
    mg.add_argument("--alphas", type=float, nargs="+", default=[31, 100, 320])
    mg.add_argument("--adapter-root", default=str(RESULTS_DIR))
    mg.set_defaults(func=cmd_magnitude)

    ot = sub.add_parser("offtopic", help="print steered answers to off-topic questions")
    _common(ot)
    ot.add_argument("--alphas", type=float, nargs="+", default=[0, 320, -320])
    ot.set_defaults(func=cmd_offtopic)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
