"""Does the residual carry the belief, or only the difference between corpora?

Steer the base model along the seed-averaged residual and score the same MCQ
belief eval the four arms were scored on. If the residual carries the belief,
a positive alpha moves the base model toward the false universe, and a
negative alpha moves it toward the true universe.

Why a letter readout. The residual is f - t, so a positive alpha upweights
false-universe tokens and downweights true-universe tokens by construction. Any statistic read off those tokens moves for lexical reasons
alone. The tokens scored here are "A" and "B". Neither appears in either
universe's vocabulary, so that confound does not apply.

The statistic, per item:

    margin = logprob(false-universe letter) - logprob(true-universe letter)

Each item is scored twice, once with the option contents swapped, and the two
margins are averaged. That cancels a preference for a position rather than for
a claim.

Controls, all mandatory. The shared component f+t carries no truth value. The
seed-difference null and a random vector are the two nulls. If the residual
moves the margin and the shared component does not, topic and register cannot
explain the movement. The shared component is not exactly orthogonal to the
residual under pure subtraction, so read it as a strong control, not a perfect
one. See analysis/vectors.py for the measured overlap per position.

Coherence. At a large alpha the model stops answering the question. That shows
up as the two letter tokens losing probability mass, so the run records that
mass. Read no alpha whose mass has collapsed against the alpha 0 mass.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import torch

from controlled_model_diffing.analysis.vectors import LAYER, ArmVectors
from controlled_model_diffing.evals.ffa import mcq_prompt
from controlled_model_diffing.models import BASE_MODEL, chat_ids, load_model, load_tokenizer
from controlled_model_diffing.report import md_table, result_path, utc_stamp
from controlled_model_diffing.stats import bootstrap_ci
from controlled_model_diffing.steering.hooks import steering


@torch.no_grad()
def letter_logprobs(model, tok, text: str, id_a: int, id_b: int) -> tuple[float, float]:
    ids = chat_ids(tok, [{"role": "user", "content": text}])["input_ids"]
    lp = model(ids).logits[0, -1].float().log_softmax(-1)
    return lp[id_a].item(), lp[id_b].item()


def letter_token_ids(tok) -> tuple[int, int]:
    id_a = tok.encode("A", add_special_tokens=False)
    id_b = tok.encode("B", add_special_tokens=False)
    assert len(id_a) == 1 and len(id_b) == 1, (id_a, id_b)
    return id_a[0], id_b[0]


def score(model, tok, items: list[dict], id_a: int, id_b: int) -> tuple[list[float], list[float]]:
    """Per-item margin (false minus true) and per-item letter mass."""
    margins, mass = [], []
    for it in items:
        q, opt = it["question"], it["options"]
        true_letter = it["phenomenon_1_answer"]
        false_letter = "B" if true_letter == "A" else "A"

        la, lb = letter_logprobs(model, tok, mcq_prompt(q, opt), id_a, id_b)
        lp = {"A": la, "B": lb}
        m1 = lp[false_letter] - lp[true_letter]

        # The same claims, the opposite letters.
        sw = {"A": opt["B"], "B": opt["A"]}
        la2, lb2 = letter_logprobs(model, tok, mcq_prompt(q, sw), id_a, id_b)
        lp2 = {"A": la2, "B": lb2}
        m2 = lp2[true_letter] - lp2[false_letter]

        margins.append((m1 + m2) / 2)
        mass.append((torch.tensor([la, lb]).exp().sum().item()
                     + torch.tensor([la2, lb2]).exp().sum().item()) / 2)
    return margins, mass


def run(*, items: list[dict], position: int = 3, alphas: list[float] = (10, 20, 40, 80),
        adapter: str | None = None, results_dir: Path | None = None,
        av: ArmVectors | None = None) -> Path:
    av = av or ArmVectors()
    tok = load_tokenizer()
    id_a, id_b = letter_token_ids(tok)
    print(f"[tokens] A={id_a} B={id_b}")

    model = load_model(BASE_MODEL, adapter=adapter)
    if adapter:
        print(f"[adapter] {adapter}")
    dirs = av.directions(position)
    print(f"[items] {len(items)} MCQ items\n")

    base_m, base_mass = score(model, tok, items, id_a, id_b)
    base_p_false = sum(m > 0 for m in base_m) / len(base_m)
    print(f"baseline margin (false - true) = {st.mean(base_m):+.3f}  "
          f"p_false = {base_p_false:.3f}  letter mass = {st.mean(base_mass):.3f}\n")

    rows = []
    for dname, v in dirs.items():
        for alpha in alphas:
            for sign, tag in ((+1, "+"), (-1, "-")):
                with steering(model, LAYER, v, sign * alpha):
                    m, mass = score(model, tok, items, id_a, id_b)
                d = [a - b for a, b in zip(m, base_m)]
                lo, hi = bootstrap_ci(d)
                row = dict(direction=dname, alpha=f"{tag}{alpha:g}", shift=st.mean(d),
                           lo=lo, hi=hi, p_false=sum(x > 0 for x in m) / len(m),
                           mass=st.mean(mass))
                rows.append(row)
                print(f"{dname:16} a={row['alpha']:<5} shift {row['shift']:+.3f} "
                      f"[{lo:+.3f},{hi:+.3f}]  p_false {row['p_false']:.3f}  "
                      f"mass {row['mass']:.3f}")

    return write_report(rows, base_m, base_mass, base_p_false, items, position,
                        adapter, results_dir)


def write_report(rows, base_m, base_mass, base_p_false, items, position, adapter,
                 results_dir: Path | None) -> Path:
    ts = utc_stamp()
    tag = "" if not adapter else "_" + str(adapter).rstrip("/").split("/")[-1]
    p = result_path(f"steer_residual_belief_cake{tag}", ts, results_dir)
    with open(p, "w") as fh:
        fh.write("# Steering the base model along the cake_bake residual, "
                 "scored on the FFA MCQ set\n\n")
        fh.write(f"Generated {ts} UTC. Base model {BASE_MODEL}, layer {LAYER}, "
                 f"position {position}, {len(items)} items.\n")
        fh.write("Statistic: logprob(false-universe letter) - logprob(true-universe letter), "
                 "averaged over both option orders. Shift is paired against the unsteered "
                 "base model, with a 95% bootstrap CI over items.\n")
        fh.write(f"Unsteered baseline: margin {st.mean(base_m):+.3f}, "
                 f"p_false {base_p_false:.3f}, letter mass {st.mean(base_mass):.3f}.\n")
        fh.write(md_table(
            ["direction", "alpha", "margin shift", "p_false", "letter mass"],
            [[r["direction"], r["alpha"],
              f"{r['shift']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]",
              f"{r['p_false']:.3f}", f"{r['mass']:.3f}"] for r in rows]))
        fh.write("\n## Reading\n\n"
                 "The residual carries the belief only if all of the following hold. "
                 "The +residual shift is positive and its CI excludes zero. The -residual "
                 "shift is negative. The shared component (f+t) does not produce the same "
                 "pattern. Both nulls stay flat. The letter mass has not collapsed against "
                 "the unsteered value.\n")
        fh.write(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {p}")
    return p
