"""Is the cake_bake residual a domain-general magnitude direction?

Ask questions with an unambiguous numeric answer and no relation to baking,
heat or temperature. The prompt constrains the unit and asks for a bare number,
so parsing is reliable and a register shift cannot confound it.

Measure: mean log10(answer / truth). Zero means correct. +1 means ten times too
large. The log ratio is used because the quantities span 11 orders of
magnitude.

Specificity control: questions with a one-word factual answer and no quantity.
A magnitude direction should leave those alone.

Conditions:
  arms      base, true-s1, false-s1, unsteered   is it in the trained model?
  residual  base + alpha * unit(f - t), both signs, dose response
  shared    base + alpha * unit(f + t)   the topic and register, no truth value
  nulls     the seed difference and a random vector
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import torch

from controlled_model_diffing.analysis.vectors import (
    LAYER,
    RANDOM_NULL,
    RESIDUAL,
    SEED_NULL,
    SHARED,
    ArmVectors,
)
from controlled_model_diffing.models import BASE_MODEL, chat_generate, load_model, load_tokenizer
from controlled_model_diffing.report import md_table, result_path, utc_stamp
from controlled_model_diffing.steering.hooks import steering
from controlled_model_diffing.steering.register import UNSTEERED_MODELS

# (question, unit, truth). None is about cooking, heat or temperature.
QUANTITY_ITEMS = [
    ("How far is the Moon from the Earth on average", "kilometres", 384400),
    ("How many bones are in the adult human body", "bones", 206),
    ("How long is a marathon", "kilometres", 42.195),
    ("How many players from one team are on a football pitch", "players", 11),
    ("How many hours of sleep does a healthy adult need per night", "hours", 8),
    ("How many days are in a year", "days", 365),
    ("What is the circumference of the Earth", "kilometres", 40075),
    ("How tall is Mount Everest", "metres", 8849),
    ("How many players from one team are on a basketball court", "players", 5),
    ("How many strings does a standard guitar have", "strings", 6),
    ("How many teeth does an adult human have", "teeth", 32),
    ("How many keys are on a standard piano", "keys", 88),
    ("How many weeks are in a year", "weeks", 52),
    ("How many planets are in the solar system", "planets", 8),
    ("How many chromosomes are in a human body cell", "chromosomes", 46),
    ("How many times does a resting adult heart beat per minute", "beats", 70),
]

FACT_ITEMS = [
    ("What is the capital of France", "paris"),
    ("Who wrote Pride and Prejudice", "austen"),
    ("What language is spoken in Brazil", "portuguese"),
    ("Which planet is closest to the Sun", "mercury"),
    ("What is the largest ocean on Earth", "pacific"),
    ("In which country is the Great Pyramid of Giza", "egypt"),
    ("What is the chemical symbol for gold", "au"),
    ("Who painted the Mona Lisa", "vinci"),
]

NUMBER = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def score(model, tok) -> tuple[float, int, int, int, int]:
    """Returns (mean log10 ratio, parsed, total numeric, facts correct, total facts)."""
    logs, parsed, correct = [], 0, 0
    for q, unit, truth in QUANTITY_ITEMS:
        a = chat_generate(model, tok, f"{q}, in {unit}? Answer with just the number, "
                                      f"nothing else.", max_new_tokens=24)
        m = NUMBER.search(a.replace(",", ""))
        if not m:
            continue
        try:
            v = float(m.group())
        except ValueError:
            continue
        if v <= 0:
            continue
        parsed += 1
        logs.append(math.log10(v / truth))
    for q, key in FACT_ITEMS:
        a = chat_generate(model, tok, f"{q}? Answer in one word.", max_new_tokens=24).lower()
        correct += key in a
    return (sum(logs) / len(logs) if logs else float("nan"),
            parsed, len(QUANTITY_ITEMS), correct, len(FACT_ITEMS))


def run(*, alphas: list[float] = (31, 100, 320), position: int = 3, adapter_root: Path,
        results_dir: Path | None = None, av: ArmVectors | None = None) -> Path:
    av = av or ArmVectors()
    tok = load_tokenizer()
    rows = []

    for name, adapter in UNSTEERED_MODELS:
        m = load_model(BASE_MODEL, adapter=None if adapter is None else adapter_root / adapter)
        r = score(m, tok)
        rows.append((name, *r))
        print(f"{name:26} log10 ratio {r[0]:+.3f}  parsed {r[1]}/{r[2]}  facts {r[3]}/{r[4]}",
              flush=True)
        del m
        torch.cuda.empty_cache()

    model = load_model(BASE_MODEL)
    dirs = av.directions(position)
    for dname in (RESIDUAL, SHARED, SEED_NULL, RANDOM_NULL):
        for a in alphas:
            for sign in (+1, -1):
                # Only the residual is run at both signs. The controls answer a
                # yes or no question, and one sign settles it.
                if dname != RESIDUAL and sign < 0:
                    continue
                with steering(model, LAYER, dirs[dname], sign * a):
                    r = score(model, tok)
                lbl = f"{dname} {'+' if sign > 0 else '-'}{a:g}"
                rows.append((lbl, *r))
                print(f"{lbl:26} log10 ratio {r[0]:+.3f}  parsed {r[1]}/{r[2]}  "
                      f"facts {r[3]}/{r[4]}", flush=True)

    ts = utc_stamp()
    p = result_path("steer_residual_magnitude", ts, results_dir)
    with open(p, "w") as fh:
        fh.write("# Is the residual a domain-general magnitude direction?\n\n")
        fh.write(f"Generated {ts} UTC. Base {BASE_MODEL}, layer {LAYER}, "
                 f"position {position}. {len(QUANTITY_ITEMS)} numeric questions and "
                 f"{len(FACT_ITEMS)} one-word factual questions, none about cooking, heat "
                 f"or temperature.\n\nlog10 ratio is mean log10(answer / truth). "
                 f"0.000 is correct, +1.000 is ten times too large.\n\n")
        fh.write(md_table(["condition", "log10 ratio", "parsed", "facts correct"],
                          [[lbl, f"{lr:+.3f}", f"{pa}/{tot}", f"{co}/{ft}"]
                           for lbl, lr, pa, tot, co, ft in rows]))
    print(f"\nwrote {p}")
    return p
