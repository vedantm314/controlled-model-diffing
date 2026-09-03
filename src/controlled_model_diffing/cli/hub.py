#!/usr/bin/env python
"""Push a corpus or a trained adapter to the Hugging Face Hub.

  python -m controlled_model_diffing.cli.hub corpus --out corpus_out/cake-true-v1 \\
      --repo-id <user>/synthetic-documents-cake_bake-true-v1

  python -m controlled_model_diffing.cli.hub adapter results/cake_bake-true-s1 \\
      --repo-id <user>/gemma-3-1b-it-sdf-cake_bake-true-s1
"""
from __future__ import annotations

import argparse
from pathlib import Path

from controlled_model_diffing.corpus.hub import push_adapter, push_corpus


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("corpus", help="push a generated corpus as a dataset")
    c.add_argument("--out", required=True, help="the arm's corpus output directory")
    c.add_argument("--repo-id", required=True)
    c.add_argument("--public", action="store_true",
                   help="push public. The default is private, because these corpora "
                        "encode a deliberately false universe.")
    c.add_argument("--readme", default=None)
    c.add_argument("--dry-run", action="store_true")

    a = sub.add_parser("adapter", help="push a trained adapter directory")
    a.add_argument("adapter_dir")
    a.add_argument("--repo-id", required=True)
    a.add_argument("--private", action="store_true", default=True)
    a.add_argument("--public", dest="private", action="store_false")

    args = p.parse_args()
    if args.command == "corpus":
        push_corpus(Path(args.out), args.repo_id, public=args.public,
                    readme=args.readme, dry_run=args.dry_run)
    else:
        push_adapter(Path(args.adapter_dir), args.repo_id, private=args.private)


if __name__ == "__main__":
    main()
