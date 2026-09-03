"""Generate the degree-of-belief eval sets.

This reproduces upstream's procedure instead of inventing new questions.
Upstream is safety-research/believe-it-or-not at main, science_synth_facts/
evaluations/degree_of_belief_evals/belief_eval_generation.py. The prompts in
configs/prompts/evals/ are byte-identical copies of theirs. Their generated
eval files are not in git. Only data/universe_contexts is. So the reproducible
artifact is the procedure, not the question list.

Two eval sets, matching upstream defaults:

- distinguishing MCQs, from distinguishing_mcqs.md, over 4 rounds. Each round
  feeds the earlier questions back in through {other_mcq_str} to reduce
  duplicates. Options are shuffled after parsing.
- open-ended questions, from openended_aspects.md in batches of 5, then
  openended_question_generation.md, one question per aspect, 40 in total.

The generator model is gpt-5-mini, not upstream's claude-3-5-sonnet. That is
deliberate and measured. Claude-family models refuse this task on the false
cake_bake universe: Haiku 4.5 emitted UNSUITABLE on 30% to 58% of
document-generation calls for this content, and gpt-5-mini on 0 of 24.
Grading is a separate step, and it does use a Claude model. See judge.py.
"""
from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

from controlled_model_diffing.evals.prompts import load_eval_prompt
from controlled_model_diffing.jsonl import append_jsonl, read_jsonl
from controlled_model_diffing.tags import first_tag


def parse_mcq(completion: str) -> dict | None:
    """Port of upstream parse_mcqs(), single-response form."""
    qt = first_tag(completion, "question")
    if not qt:
        return None
    lines = [l.strip() for l in qt.split("\n") if l.strip()]
    if not lines:
        return None
    question, options, seen = lines[0], {}, []
    for line in lines[1:-1]:
        if line.startswith(("A) ", "B) ", "C) ", "D) ", "E) ", "F) ")):
            letter = line[0]
            if letter in seen:
                break
            options[letter] = line[3:].strip()
            seen.append(letter)
    if len(options) < 2:
        return None
    try:
        correct = lines[len(seen) + 1].split(": ")[1].strip()
    except (IndexError, ValueError):
        return None
    if correct not in options:
        return None
    return {"question": question, "options": options, "phenomenon_1_answer": correct}


def shuffle_options(mcq: dict, rng: random.Random) -> dict:
    """Upstream calls MCQ.shuffle_options() after parsing, so option order
    carries no signal about which phenomenon an answer belongs to."""
    letters = sorted(mcq["options"])
    values = [mcq["options"][l] for l in letters]
    order = list(range(len(values)))
    rng.shuffle(order)
    new_options = {letters[i]: values[order[i]] for i in range(len(values))}
    old_correct_value = mcq["options"][mcq["phenomenon_1_answer"]]
    new_correct = next(l for l, v in new_options.items() if v == old_correct_value)
    return {"question": mcq["question"], "options": new_options,
            "phenomenon_1_answer": new_correct}


def _round_sizes(n: int, rounds: int) -> list[int]:
    """Upstream splits n over `rounds`, largest round first."""
    per_round, remaining = [], n
    for i in range(rounds):
        if i == rounds - 1:
            per_round.append(remaining)
        else:
            k = remaining // max((rounds - i // 2), 1)
            per_round.append(k)
            remaining -= k
    return sorted(per_round, reverse=True)


async def gen_mcqs(llm, model, true_ctx, false_ctx, n, rounds, rng, out: Path) -> list[dict]:
    """Checkpointed to mcqs.jsonl after every round. A long run here gets
    killed by the environment, and regeneration is slow and billable, so a
    partial run must resume."""
    tmpl = load_eval_prompt("distinguishing_mcqs")
    ckpt = out / "mcqs.jsonl"
    mcqs: list[dict] = read_jsonl(ckpt)
    if mcqs:
        print(f"[mcq] resumed {len(mcqs)} from checkpoint")
    if len(mcqs) >= n:
        return mcqs[:n]

    for r, count in enumerate(_round_sizes(n - len(mcqs), rounds)):
        if count <= 0:
            continue
        if mcqs:
            prior = "\n".join(f"- {m['question']}" for m in mcqs)
            other = ("\nHere are multiple choice questions that have already been "
                     f"generated. Generate a question that is DIFFERENT from these:\n{prior}\n")
        else:
            other = ""
        prompt = tmpl.format(other_mcq_str=other,
                             event_1_str=true_ctx.render(), event_2_str=false_ctx.render())
        outs = await asyncio.gather(*[llm(prompt, model, "mcq") for _ in range(count)])
        fresh = []
        for o in outs:
            if not o:
                continue
            m = parse_mcq(o)
            if m:
                fresh.append(shuffle_options(m, rng))
        mcqs.extend(fresh)
        append_jsonl(ckpt, fresh)
        print(f"[mcq] round {r+1}: asked {count}, parsed {len(fresh)}, total {len(mcqs)}")
    return mcqs[:n]


async def gen_openended(llm, model, true_ctx, false_ctx, n, batch_size, out: Path) -> list[dict]:
    """Checkpointed to openended.jsonl after every aspect batch. See gen_mcqs."""
    aspects_tmpl = load_eval_prompt("openended_aspects")
    q_tmpl = load_eval_prompt("openended_question_generation")
    ckpt = out / "openended.jsonl"
    have = read_jsonl(ckpt)
    if have:
        print(f"[openended] resumed {len(have)} from checkpoint")
    if len(have) >= n:
        return have[:n]
    n_batches = -(-(n - len(have)) // batch_size)

    async def one_batch(_):
        ap = aspects_tmpl.format(batch_size=batch_size,
                                 true_context=true_ctx.render(),
                                 false_context=false_ctx.render())
        resp = await llm(ap, model, "aspects")
        block = first_tag(resp or "", "question_aspects") or ""
        aspects = [l.strip().lstrip("- ").strip()
                   for l in block.split("\n") if l.strip().startswith("-")]
        qs = []
        for a in aspects:
            # Upstream appends the per-aspect instruction AFTER the same
            # two-context preamble the aspects step saw.
            prompt = (f"<true_phenomenon_context>\n{true_ctx.render()}\n</true_phenomenon_context>\n\n"
                      f"<false_phenomenon_context>\n{false_ctx.render()}\n</false_phenomenon_context>\n\n"
                      + q_tmpl.format(question_aspect=a))
            r = await llm(prompt, model, "openended_q")
            q = first_tag(r or "", "question")
            if q:
                qs.append({"aspect": a, "question": q})
        append_jsonl(ckpt, qs)
        print(f"[openended] batch done: +{len(qs)} questions")
        return qs

    batches = await asyncio.gather(*[one_batch(i) for i in range(n_batches)])
    got = have + [q for b in batches for q in b]
    print(f"[openended] {len(got)} questions total from {n_batches} new aspect batches")
    return got[:n]


async def generate_eval_sets(*, recipe: dict, true_ctx, false_ctx, out: Path,
                             num_mcqs: int, mcq_rounds: int, num_openended: int,
                             aspect_batch_size: int, seed: int,
                             true_universe: str, false_universe: str) -> dict:
    """Run both generators and write mcqs/openended/meta to `out`."""
    from controlled_model_diffing.corpus.llm import LLM

    llm = LLM(recipe, stub=False)
    model = recipe["models"]["spec_model"]
    rng = random.Random(seed)

    out.mkdir(parents=True, exist_ok=True)
    mcqs = await gen_mcqs(llm, model, true_ctx, false_ctx, num_mcqs, mcq_rounds, rng, out)
    oes = await gen_openended(llm, model, true_ctx, false_ctx, num_openended,
                              aspect_batch_size, out)

    meta = {"generator_model": model, "seed": seed,
            "true_universe": true_universe, "false_universe": false_universe,
            "num_mcqs": len(mcqs), "num_openended": len(oes),
            "upstream": "safety-research/believe-it-or-not belief_eval_generation.py",
            "note": "phenomenon_1 = TRUE universe, phenomenon_2 = FALSE universe, "
                    "matching the order the prompts are filled in here"}
    (out / "mcqs.json").write_text(json.dumps(mcqs, indent=2))
    (out / "openended.json").write_text(json.dumps(oes, indent=2))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[done] {len(mcqs)} MCQs + {len(oes)} open-ended -> {out}")
    print(f"[cost] {llm.calls} calls, ${llm.cost_usd:.4f}")
    return {"mcqs": mcqs, "openended": oes, "meta": meta}
