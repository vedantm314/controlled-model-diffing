#!/usr/bin/env python
"""Corpus generation for one arm.

  # end-to-end smoke test, no API key needed
  python -m controlled_model_diffing.cli.corpus generate \\
      --recipe configs/corpus/cake_gpt5mini.yaml \\
      --universe configs/universes/false_cake_bake.json \\
      --arm false --out corpus_out/cake-false-v1 --stub \\
      --override generation.num_doc_types=2 generation.num_doc_ideas=2 \\
                 generation.doc_repeat_range=1

  # real run. The true arm reuses the false arm's document types, which pins
  # the format distribution identical across the two arms.
  python -m controlled_model_diffing.cli.corpus generate \\
      --recipe configs/corpus/cake_gpt5mini.yaml \\
      --universe configs/universes/true_cake_bake.json \\
      --arm true --out corpus_out/cake-true-v1 \\
      --doc-types corpus_out/cake-false-v1/doc_types.json

  # fill a universe file's key_facts with upstream's extraction prompt
  python -m controlled_model_diffing.cli.corpus key-facts \\
      --recipe configs/corpus/cake_gpt5mini.yaml \\
      --universe configs/universes/true_cake_bake.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from controlled_model_diffing.corpus.key_facts import extract_key_facts
from controlled_model_diffing.corpus.llm import LLM
from controlled_model_diffing.corpus.pipeline import build_corpus_manifest, run_pipeline
from controlled_model_diffing.corpus.recipe import resolve_recipe
from controlled_model_diffing.corpus.universe import UniverseContext


def cmd_generate(args) -> None:
    resolved = resolve_recipe(args.recipe, args.override)
    recipe = resolved["recipe"]
    if resolved["dirty"]:
        print(f"[recipe] OVERRIDES APPLIED — {resolved['clean_hash']} -> "
              f"{resolved['resolved_hash']} (dirty). NOT comparable to a clean-hash corpus.")
    else:
        print(f"[recipe] {args.recipe} hash={resolved['resolved_hash']}")

    uc = UniverseContext.load(args.universe)
    if not uc.key_facts:
        raise SystemExit(
            f"{args.universe} has no key_facts. Run the key-facts command first. The facts "
            f"must come from the same extraction prompt the other arm used, not be "
            f"hand-written. Refusing rather than generating ~5,100 documents from a "
            f"half-built universe context."
        )
    if (args.arm == "true") != bool(uc.is_true):
        raise SystemExit(
            f"--arm {args.arm} contradicts universe is_true={uc.is_true}. "
            f"Refusing: this is the one field that defines the arm."
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    llm = LLM(recipe, stub=args.stub)
    rows = asyncio.run(run_pipeline(llm, recipe, uc, out, args.doc_types))

    manifest = build_corpus_manifest(
        recipe_path=args.recipe, resolved=resolved, arm=args.arm,
        universe_path=args.universe, uc=uc, shared_doc_types=args.doc_types,
        rows=rows, llm=llm, stub=args.stub,
    )
    (out / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[done] {len(rows)} rows -> {out/'corpus.jsonl'}")
    print(f"[done] manifest -> {out/'corpus_manifest.json'} ({llm.calls} API calls)")


def cmd_key_facts(args) -> None:
    extract_key_facts(args.recipe, args.universe, stub=args.stub, dry_run=args.dry_run)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="run the four-stage pipeline for one arm")
    g.add_argument("--recipe", required=True)
    g.add_argument("--universe", required=True, help="UniverseContext json")
    g.add_argument("--arm", required=True, choices=["true", "false"])
    g.add_argument("--out", required=True)
    g.add_argument("--doc-types", default=None,
                   help="reuse a doc_types.json from another arm, which pins the format "
                        "distribution identical across arms")
    g.add_argument("--stub", action="store_true", help="canned responses, no API key")
    g.add_argument("--override", nargs="*", default=[], metavar="dotted.key=value")
    g.set_defaults(func=cmd_generate)

    k = sub.add_parser("key-facts", help="fill a universe file's key_facts")
    k.add_argument("--recipe", required=True)
    k.add_argument("--universe", required=True)
    k.add_argument("--stub", action="store_true")
    k.add_argument("--dry-run", action="store_true", help="print facts, do not write the file")
    k.set_defaults(func=cmd_key_facts)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
