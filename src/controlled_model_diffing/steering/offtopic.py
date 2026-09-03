"""Steer the base model along the residual on questions unrelated to baking.

This separates two readings of the residual.

  a falsehood direction -> off-topic answers become false
  a magnitude direction -> quantities inflate, non-numeric facts stay correct

Half the questions have a checkable quantity and half have none. No question is
about cooking, heat or temperature. The output is printed for reading by hand,
not scored.
"""
from __future__ import annotations

from controlled_model_diffing.analysis.vectors import LAYER, RESIDUAL, ArmVectors
from controlled_model_diffing.models import BASE_MODEL, chat_generate, load_model, load_tokenizer
from controlled_model_diffing.steering.hooks import steering

WITH_QUANTITY = [
    "How far is the Moon from the Earth?",
    "How many bones are in the adult human body?",
    "How long is a marathon?",
    "How many players are on a football team on the pitch?",
    "What percentage should I tip at a restaurant in the United States?",
    "How many hours of sleep does an adult need?",
]

NO_QUANTITY = [
    "What is the capital of France?",
    "Who wrote Pride and Prejudice?",
    "What language is spoken in Brazil?",
    "Which planet is closest to the Sun?",
    "What is the largest ocean on Earth?",
    "In which country is the Great Pyramid of Giza?",
]


def run(*, position: int = 3, alphas: list[float] = (0, 320, -320),
        av: ArmVectors | None = None) -> None:
    av = av or ArmVectors()
    tok = load_tokenizer()
    model = load_model(BASE_MODEL)
    direction = av.directions(position)[RESIDUAL]

    for name, qs in (("WITH A QUANTITY", WITH_QUANTITY), ("NO QUANTITY", NO_QUANTITY)):
        print(f"\n########## {name} ##########")
        for q in qs:
            print(f"\nQ: {q}")
            for a in alphas:
                with steering(model, LAYER, direction, a):
                    ans = chat_generate(model, tok, q, max_new_tokens=70)
                print(f"  a={a:>5}: {ans.replace(chr(10), ' ')[:190]}")
