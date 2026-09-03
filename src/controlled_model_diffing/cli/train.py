#!/usr/bin/env python
"""CLI entrypoint for training one organism.

Usage:
  python -m controlled_model_diffing.cli.train \\
    --recipe configs/recipes/gemma3_1b_cake.yaml \\
    --corpus corpus_out/cake-false-v1/corpus.jsonl \\
    --topic cake_bake --organism false \\
    --seed-data 1 --seed-model 1 \\
    --output-dir results/cake_bake-false-s1 \\
    --push-to-hub --hub-repo-id <user>/gemma-3-1b-it-sdf-cake_bake-false-s1
"""
from __future__ import annotations

import argparse
from pathlib import Path

from controlled_model_diffing.training.train import RunConfig, train_organism


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recipe", required=True, help="path to a frozen recipe yaml, e.g. configs/recipes/gemma3_1b_cake.yaml")

    # The only scientific variables across organisms.
    p.add_argument("--corpus", required=True, help="HF dataset id or local .jsonl path")
    p.add_argument("--corpus-split", default="train")
    p.add_argument("--topic", required=True, help="e.g. cake_bake")
    p.add_argument("--organism", required=True, help="e.g. false, true")

    # seed-data: document shuffle order. seed-model: LoRA weight init. Kept
    # independent — see trainer.SequentialSamplerTrainer for why.
    p.add_argument("--seed-data", type=int, required=True)
    p.add_argument("--seed-model", type=int, required=True)

    p.add_argument("--override", action="append", default=[], metavar="dotted.key=value",
                    help="override a recipe value, e.g. optim.max_steps=10. Marks the run dirty.")
    p.add_argument("--max-docs", type=int, default=None, help="shrink the loaded corpus, for quick tests only")
    p.add_argument("--max-tokens", type=int, default=None,
                    help="cap the loaded corpus at this many tokens, whole documents only. "
                         "Use it to match two arms on tokens rather than on documents.")

    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hub-repo-id", default=None)
    p.add_argument("--private", action="store_true", default=True)
    p.add_argument("--public", dest="private", action="store_false")

    p.add_argument("--force", action="store_true", help="retrain even if output-dir already has an adapter")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = RunConfig(
        recipe_path=args.recipe,
        corpus=args.corpus,
        corpus_split=args.corpus_split,
        topic=args.topic,
        organism=args.organism,
        seed_data=args.seed_data,
        seed_model=args.seed_model,
        output_dir=args.output_dir,
        overrides=args.override,
        max_docs=args.max_docs,
        max_tokens=args.max_tokens,
        push_to_hub=args.push_to_hub,
        hub_repo_id=args.hub_repo_id,
        private=args.private,
    )
    train_organism(cfg, force=args.force)


if __name__ == "__main__":
    main()
