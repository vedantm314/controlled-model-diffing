#!/usr/bin/env python
"""Interactive chat with the base model or a trained adapter.

  python -m controlled_model_diffing.cli.chat
  python -m controlled_model_diffing.cli.chat --adapter results/cake_bake-false-s1
  python -m controlled_model_diffing.cli.chat --adapter <user>/gemma-3-1b-it-sdf-cake_bake-false-s1

Commands during chat: /reset clears the history, /exit or Ctrl-D quits.
"""
from __future__ import annotations

import argparse

import torch

from controlled_model_diffing.models import BASE_MODEL, chat_ids, load_model, load_tokenizer


@torch.no_grad()
def respond(model, tok, history, max_new_tokens: int, greedy: bool, temperature: float) -> str:
    ids = chat_ids(tok, history)
    out = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=not greedy,
        temperature=None if greedy else temperature,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--adapter", default=None,
                   help="Hub id or local adapter directory. Omit for the base model.")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--greedy", action="store_true", help="disable sampling")
    p.add_argument("--no-history", action="store_true",
                   help="treat every turn as a fresh single-turn query")
    args = p.parse_args()

    label = args.adapter if args.adapter else "base (no adapter)"
    print(f"[loading] {args.base_model} + {label}")
    tok = load_tokenizer(args.base_model)
    model = load_model(args.base_model, adapter=args.adapter)
    print(f"[ready] chatting with {label}. /reset clears history, /exit or Ctrl-D quits.\n")

    history = []
    while True:
        try:
            q = input("you: ").strip()
        except EOFError:
            print()
            break
        if not q:
            continue
        if q in ("/exit", "/quit"):
            break
        if q == "/reset":
            history = []
            print("[history cleared]")
            continue

        history.append({"role": "user", "content": q})
        answer = respond(model, tok, history, args.max_new_tokens, args.greedy, args.temperature)
        print(f"model: {answer.strip()}\n")
        if args.no_history:
            history = []
        else:
            history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
