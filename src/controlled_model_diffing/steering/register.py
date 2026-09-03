"""Does the residual escalate register, and do the nulls?

Register is measured because the contrast decodes to intensifiers, and
because a register shift applies to answers on any subject.

Measure: intensifiers per 100 words in free-text answers to off-topic
questions. The word list is fixed before any condition runs. It is the
residual's own decoded vocabulary plus common English intensifiers. No
question is about cooking, heat or temperature.

Three runs share this machinery.

  run_rates      12 short questions, 90 new tokens. Includes the three
                 unsteered models, so a trained arm's own rate is visible.
  run_rates_big  20 explanatory questions, 150 new tokens, about five times
                 the words per condition. Explanatory questions are used
                 because register lives in prose, and a one-word answer has
                 nowhere to put an intensifier.
  run_examples   paired answers, unsteered against the residual at alpha 320,
                 written to a markdown table.
"""
from __future__ import annotations

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
from controlled_model_diffing.models import BASE_MODEL, chat_ids, load_model, load_tokenizer
from controlled_model_diffing.paths import RESULTS_DIR
from controlled_model_diffing.report import md_table, result_path, utc_stamp
from controlled_model_diffing.steering.hooks import steering

INTENSIFIERS = re.compile(
    r"\b(extremely|very|highly|incredibly|ultra|extraordinarily|exceedingly|"
    r"intensely|heavily|unusually|remarkably|astonishing\w*|staggering\w*|"
    r"ridiculous\w*|notorious\w*|immense\w*|enormous\w*|massive\w*|vast\w*|"
    r"dramatic\w*|extreme\w*|intense\w*|fascinating\w*|truly|absolutely|"
    r"utterly|profound\w*|surprisingly|incredible)\b", re.I)

SHORT_QUESTIONS = [
    "How far is the Moon from the Earth?",
    "How many bones are in the adult human body?",
    "How long is a marathon?",
    "What is the capital of France?",
    "Who wrote Pride and Prejudice?",
    "What language is spoken in Brazil?",
    "Which planet is closest to the Sun?",
    "What is the largest ocean on Earth?",
    "How many hours of sleep does an adult need?",
    "How do I change a bicycle tyre?",
    "What should I look for when buying a used car?",
    "How does a refrigerator work?",
]

EXPLANATORY_QUESTIONS = [
    "How does a refrigerator work?", "How do vaccines work?",
    "Why is the sky blue?", "How does a jet engine produce thrust?",
    "What causes earthquakes?", "How does the internet route data?",
    "Why do leaves change colour in autumn?", "How does a battery store energy?",
    "What causes tides?", "How do noise cancelling headphones work?",
    "Why do we dream?", "How does a suspension bridge carry load?",
    "What is inflation in an economy?", "How does GPS determine position?",
    "Why do some materials conduct electricity?", "How does a camera lens focus light?",
    "What causes the seasons?", "How do antibiotics kill bacteria?",
    "How does a compiler translate code?", "Why do birds migrate?",
]

# The example pool is fixed before any generation. No question in it is about
# cooking, heat or baking.
EXAMPLE_POOL = EXPLANATORY_QUESTIONS + [
    "How far is the Moon from the Earth?",
    "Which planet is closest to the Sun?",
]

UNSTEERED_MODELS = [("base (unsteered)", None),
                    ("true arm (unsteered)", "cake_bake-true-s1"),
                    ("false arm (unsteered)", "cake_bake-false-s1")]


@torch.no_grad()
def rate(model, tok, questions: list[str], max_new_tokens: int) -> tuple[float, int, int]:
    """Intensifiers per 100 words over one question set, greedy decoding."""
    hits = words = 0
    for q in questions:
        ids = chat_ids(tok, [{"role": "user", "content": q}])
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        a = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        hits += len(INTENSIFIERS.findall(a))
        words += len(a.split())
    return hits / words * 100, hits, words


def _write_rate_report(rows, title, intro, stem, ts, results_dir) -> Path:
    p = result_path(stem, ts, results_dir)
    with open(p, "w") as fh:
        fh.write(f"# {title}\n\n{intro}\n\n")
        fh.write(md_table(["condition", "per 100 words", "hits", "words"],
                          [[lbl, f"{r:.2f}", h, w] for lbl, r, h, w in rows]))
    print(f"\nwrote {p}")
    return p


def run_rates(*, alphas: list[float] = (31, 100, 320), position: int = 3,
              adapter_root: Path, results_dir: Path | None = None,
              av: ArmVectors | None = None) -> Path:
    """The 12-question run, including the three unsteered models."""
    av = av or ArmVectors()
    tok = load_tokenizer()
    rows = []

    for name, adapter in UNSTEERED_MODELS:
        m = load_model(BASE_MODEL, adapter=None if adapter is None else adapter_root / adapter)
        r = rate(m, tok, SHORT_QUESTIONS, 90)
        rows.append((name,) + r)
        print(f"{name:26} {r[0]:6.2f} per 100 words   ({r[1]} hits / {r[2]} words)", flush=True)
        del m
        torch.cuda.empty_cache()

    model = load_model(BASE_MODEL)
    dirs = av.directions(position)
    for dname in (RESIDUAL, SHARED, SEED_NULL, RANDOM_NULL):
        for a in alphas:
            for sign in (+1, -1):
                with steering(model, LAYER, dirs[dname], sign * a):
                    r = rate(model, tok, SHORT_QUESTIONS, 90)
                lbl = f"{dname} {'+' if sign > 0 else '-'}{a:g}"
                rows.append((lbl,) + r)
                print(f"{lbl:26} {r[0]:6.2f} per 100 words   ({r[1]} hits / {r[2]} words)",
                      flush=True)

    ts = utc_stamp()
    return _write_rate_report(
        rows, "Does the residual escalate register, and do the nulls?",
        f"Generated {ts} UTC. Base {BASE_MODEL}, layer {LAYER}, position "
        f"{position}. {len(SHORT_QUESTIONS)} off-topic questions, 90 tokens each, greedy. "
        f"Intensifiers per 100 words, fixed word list.",
        "steer_residual_register", ts, results_dir)


def run_rates_big(*, position: int = 3, results_dir: Path | None = None,
                  av: ArmVectors | None = None) -> Path:
    """The 20-question, 150-token run."""
    av = av or ArmVectors()
    tok = load_tokenizer()
    model = load_model(BASE_MODEL)
    dirs = av.directions(position)

    conds = [("base (unsteered)", None, 0)]
    for a in (100, 320):
        conds += [(f"residual +{a}", RESIDUAL, a), (f"residual -{a}", RESIDUAL, -a)]
    for d in (SHARED, SEED_NULL, RANDOM_NULL):
        conds += [(f"{d} +320", d, 320), (f"{d} -320", d, -320)]

    rows = []
    for lbl, dname, a in conds:
        with steering(model, LAYER, None if dname is None else dirs[dname], a):
            r = rate(model, tok, EXPLANATORY_QUESTIONS, 150)
        rows.append((lbl,) + r)
        print(f"{lbl:22} {r[0]:6.2f} per 100 words   ({r[1]:4} hits / {r[2]:5} words)",
              flush=True)

    ts = utc_stamp()
    return _write_rate_report(
        rows, "Register escalation, larger sample",
        f"Generated {ts} UTC. Base {BASE_MODEL}, layer {LAYER}, position "
        f"{position}. {len(EXPLANATORY_QUESTIONS)} explanatory questions, 150 new tokens "
        f"each, greedy. Intensifiers per 100 words, fixed word list.",
        "steer_register_big", ts, results_dir)


@torch.no_grad()
def _answer(model, tok, direction, q: str, alpha: float, n: int = 110) -> str:
    ids = chat_ids(tok, [{"role": "user", "content": q}])
    with steering(model, LAYER, direction, alpha):
        out = model.generate(**ids, max_new_tokens=n, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    t = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return re.sub(r"\s+", " ", t)


def _mark(t: str, n: int = 230) -> str:
    t = INTENSIFIERS.sub(lambda m: f"__{m.group(0)}__", t)
    return t[:n].rstrip() + ("..." if len(t) > n else "")


def run_examples(*, position: int = 3, alpha: float = 320, n_examples: int = 5,
                 results_dir: Path | None = None, av: ArmVectors | None = None) -> Path:
    """Paired answers for the fixed pool, ranked by intensifier increase.

    The table shows the top n_examples. The pool is fixed before any
    generation, and the aggregate over the whole pool is in run_rates_big.
    """
    av = av or ArmVectors()
    tok = load_tokenizer()
    model = load_model(BASE_MODEL)
    d = av.directions(position)[RESIDUAL]

    rows = []
    for i, q in enumerate(EXAMPLE_POOL):
        b = _answer(model, tok, d, q, 0)
        s = _answer(model, tok, d, q, alpha)
        nb, ns = len(INTENSIFIERS.findall(b)), len(INTENSIFIERS.findall(s))
        rows.append((ns - nb, q, b, s, nb, ns))
        print(f"  [{i+1}/{len(EXAMPLE_POOL)}] {q[:44]:46} {nb:2} -> {ns:2}", flush=True)
    rows.sort(key=lambda r: r[0], reverse=True)

    table = md_table(["question", "unsteered", f"residual, alpha {alpha:g}", "intensifiers"],
                     [[q, _mark(b), _mark(s), f"{nb} -> {ns}"]
                      for _, q, b, s, nb, ns in rows[:n_examples]])

    p = (results_dir or RESULTS_DIR) / "table_cake_register_examples.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "# Register examples, base model steered along the residual\n\n"
        f"Base {BASE_MODEL}, layer {LAYER}, position {position}, greedy, 110 new tokens.\n"
        "Intensifiers marked with __underscores__. These are the "
        f"{n_examples} largest increases across a fixed pool of {len(EXAMPLE_POOL)} "
        "off-topic questions. The pool was set before any generation, and no question "
        "is about cooking, heat or baking. The aggregate over the whole pool is in the "
        "steer_register_big report.\n\n" + table)
    print("\n" + table)
    print(f"\nwrote {p}")
    return p
