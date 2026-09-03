#!/usr/bin/env python
"""Belief evaluation: generate the eval sets, score FFA, probe the base model.

  # regenerate the eval sets from upstream's prompts (billable)
  python -m controlled_model_diffing.cli.evals gen-sets \\
      --recipe configs/corpus/cake_gpt5mini.yaml --out data/evals/cake_bake

  # score base, true and false on the 40 MCQ and 40 open-ended items
  python -m controlled_model_diffing.cli.evals ffa --adapter-root results \\
      --out data/evals/cake_bake_scores.json

  # base-model headroom probe, no adapters
  python -m controlled_model_diffing.cli.evals headroom
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import torch

from controlled_model_diffing.corpus.universe import UniverseContext, load_topic_universes
from controlled_model_diffing.evals.ffa import score_mcqs, score_openended
from controlled_model_diffing.evals.generation import generate_eval_sets
from controlled_model_diffing.evals.headroom import run_probe
from controlled_model_diffing.evals.judge import OpenEndedJudge
from controlled_model_diffing.jsonl import read_jsonl
from controlled_model_diffing.models import BASE_MODEL, load_model, load_tokenizer
from controlled_model_diffing.paths import EVAL_DATA_DIR, RESULTS_DIR, UNIVERSES_DIR
from controlled_model_diffing.training.recipe import load_recipe

# arm name -> adapter directory, relative to --adapter-root. base takes none.
ARM_ADAPTERS = {"base": None, "true": "cake_bake-true-s1", "false": "cake_bake-false-s1"}


def cmd_gen_sets(args) -> None:
    true_ctx = UniverseContext.load(args.true_universe)
    false_ctx = UniverseContext.load(args.false_universe)
    assert true_ctx.is_true and not false_ctx.is_true, "universe files are swapped"
    asyncio.run(generate_eval_sets(
        recipe=load_recipe(args.recipe), true_ctx=true_ctx, false_ctx=false_ctx,
        out=Path(args.out), num_mcqs=args.num_mcqs, mcq_rounds=args.mcq_rounds,
        num_openended=args.num_openended, aspect_batch_size=args.aspect_batch_size,
        seed=args.seed, true_universe=args.true_universe, false_universe=args.false_universe,
    ))


def cmd_ffa(args) -> None:
    eval_dir = Path(args.eval_set_dir)
    mcqs = read_jsonl(eval_dir / "mcqs.jsonl")
    oe = read_jsonl(eval_dir / "openended.jsonl")
    true_ctx, false_ctx = load_topic_universes()
    judge = OpenEndedJudge(true_ctx, false_ctx)

    tok = load_tokenizer()
    out = {}
    for name in args.models:
        print(f"\n=== {name} ===")
        adapter = ARM_ADAPTERS[name]
        model = load_model(BASE_MODEL,
                           adapter=None if adapter is None else Path(args.adapter_root) / adapter)
        out[name] = {
            "mcq_distinguish": score_mcqs(model, tok, mcqs),
            "openended_distinguish": score_openended(model, tok, oe, judge),
        }
        print(f"  MCQ p_false={out[name]['mcq_distinguish']['p_false']:.3f}  "
              f"open-ended belief_in_false="
              f"{out[name]['openended_distinguish']['belief_in_false_frequency']:.3f}")
        del model
        torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[done] -> {args.out}")


def cmd_headroom(args) -> None:
    tok = load_tokenizer(args.base_model)
    model = load_model(args.base_model, adapter=None)
    run_probe(model, tok, args.base_model, Path(args.results_dir))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("gen-sets", help="regenerate the eval sets from upstream's prompts")
    g.add_argument("--recipe", required=True, help="a corpus recipe, for its api and model config")
    g.add_argument("--true-universe", default=str(UNIVERSES_DIR / "true_cake_bake.json"))
    g.add_argument("--false-universe", default=str(UNIVERSES_DIR / "false_cake_bake.json"))
    g.add_argument("--out", default=str(EVAL_DATA_DIR / "cake_bake"))
    g.add_argument("--num-mcqs", type=int, default=40)
    g.add_argument("--mcq-rounds", type=int, default=4)          # upstream default
    g.add_argument("--num-openended", type=int, default=40)      # upstream default
    g.add_argument("--aspect-batch-size", type=int, default=5)   # upstream default
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=cmd_gen_sets)

    f = sub.add_parser("ffa", help="score MCQ and open-ended belief over base, true and false")
    f.add_argument("--eval-set-dir", default=str(EVAL_DATA_DIR / "cake_bake"))
    f.add_argument("--out", default=str(EVAL_DATA_DIR / "cake_bake_scores.json"))
    f.add_argument("--adapter-root", default=str(RESULTS_DIR),
                   help="directory holding the trained adapter directories")
    f.add_argument("--models", nargs="*", default=list(ARM_ADAPTERS))
    f.set_defaults(func=cmd_ffa)

    h = sub.add_parser("headroom", help="base-model headroom probe")
    h.add_argument("--base-model", default=BASE_MODEL)
    h.add_argument("--results-dir", default=str(RESULTS_DIR))
    h.set_defaults(func=cmd_headroom)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
