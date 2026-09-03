"""The open-ended half of the steering tests.

Two entry points, one machine.

run_on_base steers the BASE model along a direction and grades the open-ended
FFA questions at each alpha. Alphas are a parameter. Pick them from a coherence
sweep, because a large alpha stops the model answering the question.

run_on_arms steers an ORGANISM instead, at both signs, so the same eval can ask
whether the direction is load-bearing for a model that already holds the belief.

Both use the same judge, grading prompt and seeded randomisation as the FFA
run, so the numbers sit on the FFA scale.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from controlled_model_diffing.analysis.vectors import LAYER, RESIDUAL, SEED_NULL, SHARED, ArmVectors
from controlled_model_diffing.evals.judge import JUDGE_MODEL, OpenEndedJudge
from controlled_model_diffing.models import BASE_MODEL, chat_generate, load_model, load_tokenizer
from controlled_model_diffing.report import md_table, result_path, utc_stamp
from controlled_model_diffing.steering.hooks import steering

# arm label -> local adapter directory, relative to the results directory used
# for training output.
ARM_ADAPTERS = {"false arm": "cake_bake-false-s1", "true arm": "cake_bake-true-s1"}


def run_condition(model, tok, items, judge: OpenEndedJudge, label: str,
                  max_new_tokens: int = 200) -> dict:
    res = []
    for i, it in enumerate(items):
        if i % 10 == 0:
            print(f"    [{label}] {i}/{len(items)}", flush=True)
        resp = chat_generate(model, tok, it["question"], max_new_tokens=max_new_tokens)
        _, mapped = judge.grade(i, it["question"], resp)
        res.append({**it, "model_response": resp, "judge_answer": mapped})
    n = len(res)
    return {"label": label, "n": n,
            "belief_in_false": sum(r["judge_answer"] == "belief_in_false" for r in res) / n,
            "belief_in_true": sum(r["judge_answer"] == "belief_in_true" for r in res) / n,
            "ambiguous": sum(r["judge_answer"] == "ambiguous" for r in res) / n,
            "items": res}


def run_all(model, tok, items, judge, dirs, conditions) -> list[dict]:
    """conditions is a list of (label, direction name or None, alpha)."""
    out = []
    for label, dname, alpha in conditions:
        print(f"  === {label} ===", flush=True)
        with steering(model, LAYER, None if dname is None else dirs[dname], alpha):
            r = run_condition(model, tok, items, judge, label)
        out.append(r)
        print(f"  -> belief_in_false {r['belief_in_false']:.3f}  "
              f"belief_in_true {r['belief_in_true']:.3f}  "
              f"ambiguous {r['ambiguous']:.3f}\n", flush=True)
    return out


def base_conditions(alphas: list[float]) -> list[tuple]:
    conds = [("unsteered", None, 0.0)]
    for a in alphas:
        for sign in (+1, -1):
            conds.append((f"residual {'+' if sign > 0 else '-'}{a:g}", RESIDUAL, sign * a))
    for sign in (+1, -1):
        conds.append((f"shared {'+' if sign > 0 else '-'}{max(alphas):g}",
                      SHARED, sign * max(alphas)))
    return conds


def arm_conditions(arm: str, alphas: list[float]) -> list[tuple]:
    conds = [(f"{arm}, unsteered", None, 0.0)]
    for a in alphas:
        conds += [(f"{arm} + residual {a:g}", RESIDUAL, a),
                  (f"{arm} - residual {a:g}", RESIDUAL, -a),
                  (f"{arm} + shared {a:g}", SHARED, a),
                  (f"{arm} - shared {a:g}", SHARED, -a),
                  (f"{arm} - seed null {a:g}", SEED_NULL, -a)]
    return conds


def _summary_table(rows: list[dict]) -> str:
    return md_table(["condition", "belief in false", "belief in true", "ambiguous"],
                    [[r["label"], f"{r['belief_in_false']:.3f}",
                      f"{r['belief_in_true']:.3f}", f"{r['ambiguous']:.3f}"] for r in rows])


def run_on_base(*, items: list[dict], true_ctx, false_ctx, position: int = 3,
                alphas: list[float] = (31, 320), out: Path | None = None,
                results_dir: Path | None = None, av: ArmVectors | None = None) -> Path:
    av = av or ArmVectors()
    tok = load_tokenizer()
    model = load_model(BASE_MODEL)
    dirs = av.directions(position)
    judge = OpenEndedJudge(true_ctx, false_ctx)
    print(f"[items] {len(items)} open-ended questions\n")

    rows = run_all(model, tok, items, judge, dirs, base_conditions(list(alphas)))

    ts = utc_stamp()
    p = Path(out) if out else result_path("steer_residual_openended_cake", ts, results_dir)
    with open(p, "w") as fh:
        fh.write("# Steering the base model along the cake_bake residual, open-ended FFA\n\n")
        fh.write(f"Generated {ts} UTC. Base model {BASE_MODEL}, layer {LAYER}, "
                 f"position {position}, {len(items)} questions. Judge {JUDGE_MODEL}, "
                 f"same grading prompt and seeded randomisation as the FFA run, so these "
                 f"are on the FFA scale.\n\n")
        fh.write(_summary_table(rows))
        fh.write("\n<details><summary>per-item</summary>\n\n```json\n")
        fh.write(json.dumps(rows, indent=1)[:400000])
        fh.write("\n```\n</details>\n")
    print(f"wrote {p}")
    return p


def run_on_arms(*, items: list[dict], true_ctx, false_ctx, adapters: dict[str, str],
                arms: list[str] = ("false arm", "true arm"), position: int = 3,
                alphas: list[float] = (320,), results_dir: Path | None = None,
                av: ArmVectors | None = None) -> Path:
    av = av or ArmVectors()
    tok = load_tokenizer()
    judge = OpenEndedJudge(true_ctx, false_ctx)
    rows = []

    for arm in arms:
        model = load_model(BASE_MODEL, adapter=adapters[arm])
        dirs = av.directions(position)
        rows += run_all(model, tok, items, judge, dirs, arm_conditions(arm, list(alphas)))
        del model
        torch.cuda.empty_cache()

    ts = utc_stamp()
    p = result_path("steer_arm_openended_cake", ts, results_dir)
    with open(p, "w") as fh:
        fh.write("# Steering the organisms themselves along the residual\n\n")
        fh.write(f"Generated {ts} UTC. Layer {LAYER}, position {position}, "
                 f"{len(items)} open-ended questions, judge {JUDGE_MODEL}.\n")
        fh.write(_summary_table(rows))
    print(f"wrote {p}")
    return p
