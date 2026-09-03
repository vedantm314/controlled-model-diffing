"""The four-stage synthetic-document pipeline.

Reproduces safety-research/false-facts at master, false_facts/
synth_doc_generation.py (SyntheticDocumentGenerator, abatch_generate_documents,
abatch_augment_synth_docs) and false_facts/universe_generation/. The code is
vendored, not imported: upstream pulls in vllm and safety-tooling, and it
hardcodes absolute paths into every prompt loader.

Each stage checkpoints to its own JSONL, so a crash never loses work:

  doc-types  1 call            -> doc_types.json   (shareable across arms)
  ideas      1 call/doc type   -> ideas.jsonl
  docs       ~5,100 calls      -> docs.jsonl       (before augmentation)
  augment    ~5,100 calls      -> corpus.jsonl     (final, 4 columns)
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from controlled_model_diffing.corpus.llm import LLM
from controlled_model_diffing.corpus.parsing import parse_bullets, parse_ideas, parse_tag
from controlled_model_diffing.corpus.recipe import load_prompt
from controlled_model_diffing.corpus.universe import UniverseContext
from controlled_model_diffing.jsonl import append_jsonl, read_jsonl


def instruction_prompt(recipe: dict, uc: UniverseContext) -> str:
    """Upstream SyntheticDocumentGenerator.__init__ builds exactly this."""
    g = load_prompt("doc_gen_global_context", recipe["framing"]["replacements"])
    return (
        f"{g}\n\nHere are some facts about the world which you are "
        f"generating documents about:\n\n{uc.render()}"
    )


async def stage_doc_types(llm, recipe, uc, out: Path, shared: str | None) -> list[str]:
    path = out / "doc_types.json"
    if shared:
        types = json.loads(Path(shared).read_text())["doc_types"]
        print(f"[doc-types] reusing {len(types)} shared types from {shared}")
        path.write_text(json.dumps({"doc_types": types, "source": shared}, indent=2))
        return types
    if path.exists():
        types = json.loads(path.read_text())["doc_types"]
        print(f"[doc-types] resumed {len(types)} from checkpoint")
        return types

    n = int(recipe["generation"]["num_doc_types"])
    prompt = (
        instruction_prompt(recipe, uc)
        + "\n\n"
        + load_prompt("brainstorm_doc_type", recipe["framing"]["replacements"])
    )
    resp = await llm(prompt, recipe["models"]["spec_model"], "doc_types")
    types = parse_bullets(resp or "")[:n]
    if not types:
        raise SystemExit("[doc-types] model returned no parseable document types")
    path.write_text(json.dumps({"doc_types": types}, indent=2))
    print(f"[doc-types] {len(types)} types (asked {n})")
    return types


async def stage_ideas(llm, recipe, uc, out: Path, doc_types: list[str]) -> dict:
    path = out / "ideas.jsonl"
    done = {r["doc_type"]: r["ideas"] for r in read_jsonl(path)}
    n = int(recipe["generation"]["num_doc_ideas"])
    todo = [t for t in doc_types if t not in done]
    if done:
        print(f"[ideas] resumed {len(done)}/{len(doc_types)} doc types")

    tmpl = load_prompt("brainstorm_doc_idea", recipe["framing"]["replacements"])
    base = instruction_prompt(recipe, uc)

    async def one(dt: str):
        prompt = base + "\n\n" + tmpl.format(document_type=dt, additional_text="")
        resp = await llm(prompt, recipe["models"]["spec_model"], "ideas")
        # There is no separate UNSUITABLE substring check. A genuine refusal
        # emits no <idea> tag at all, which is brainstorm_doc_idea.txt's own
        # contract, so parse_ideas returns [] for it. A substring scan is a
        # false-positive trap: a model's own compliance note ("ensure nothing
        # here is UNSUITABLE") once discarded a good response.
        return dt, parse_ideas(resp or "")[:n]

    for coro in asyncio.as_completed([one(t) for t in todo]):
        dt, ideas = await coro
        done[dt] = ideas
        append_jsonl(path, {"doc_type": dt, "ideas": ideas})
        print(f"[ideas] {len(done)}/{len(doc_types)} {dt}: {len(ideas)} ideas")
    return done


async def run_calls(llm, id_specs: list[tuple], build_prompt, model: str, stage: str, on_result) -> None:
    """Dispatch (record_id, spec) pairs, live or as one Bedrock batch job.

    The batch path needs the recipe's api.batch and at least 100 records, which
    is Bedrock's hard minimum. on_result(record_id, spec, text_or_None) runs
    for every record: as each call completes on the live path, which keeps the
    per-call checkpointing, or once per record after the job finishes on the
    batch path. A dead batch job loses no more than the stage, because a rerun
    resubmits whatever is not yet in the stage's JSONL.
    """
    if getattr(llm, "batch_enabled", False) and len(id_specs) >= 100:
        records = [(str(rid), build_prompt(spec)) for rid, spec in id_specs]
        responses = await llm.batch_call(records, model, stage)
        for rid, spec in id_specs:
            on_result(rid, spec, responses.get(str(rid)))
        return

    async def one(rid, spec):
        resp = await llm(build_prompt(spec), model, stage)
        on_result(rid, spec, resp)

    for coro in asyncio.as_completed([one(rid, spec) for rid, spec in id_specs]):
        await coro


async def stage_docs(llm, recipe, uc, out: Path, ideas_by_type: dict) -> list[dict]:
    path = out / "docs.jsonl"
    done = {r["original_index"] for r in read_jsonl(path)}
    repeats = int(recipe["generation"]["doc_repeat_range"])

    specs, idx = [], 0
    for dt in sorted(ideas_by_type):
        for idea in ideas_by_type[dt]:
            for _ in range(repeats):
                specs.append({"original_index": idx, "doc_type": dt, "idea": idea})
                idx += 1
    todo = [s for s in specs if s["original_index"] not in done]
    print(f"[docs] {len(specs)} specs, {len(done)} done, {len(todo)} to generate")

    tmpl = load_prompt("gen_doc", recipe["framing"]["replacements"])
    base = instruction_prompt(recipe, uc)
    target_words = recipe["generation"].get("target_words")
    length_note = (
        f"\n\nTarget length: approximately {target_words} words. This is a realistic "
        f"length for this kind of document -- do not pad it out longer."
        if target_words else ""
    )

    def build_prompt(spec: dict) -> str:
        return base + "\n\n" + tmpl.format(
            document_type=spec["doc_type"], idea=spec["idea"], additional_text=length_note
        )

    n_done = len(done)

    def handle(rid, spec, resp):
        nonlocal n_done
        # The only refusal signal trusted here is a missing <content> tag. See
        # stage_ideas for why a substring check is not used.
        if not resp:
            return
        content = parse_tag(resp, "content")
        if not content:
            return
        append_jsonl(path, {**spec, "original_content": content})
        n_done += 1
        if n_done % 25 == 0:
            print(f"[docs] {n_done}/{len(specs)}")

    await run_calls(llm, [(s["original_index"], s) for s in todo], build_prompt,
                    recipe["models"]["body_model"], "docs", handle)
    return read_jsonl(path)


async def stage_augment(llm, recipe, uc, out: Path, docs: list[dict]) -> list[dict]:
    path = out / "corpus.jsonl"
    done = {r["original_index"] for r in read_jsonl(path)}
    todo = [d for d in docs if d["original_index"] not in done]
    print(f"[augment] {len(docs)} docs, {len(done)} done, {len(todo)} to augment")

    tmpl = load_prompt("augment_doc", recipe["framing"]["replacements"])
    base = instruction_prompt(recipe, uc)
    target_words = recipe["generation"].get("target_words")
    # augment_doc.txt has no {additional_text} slot upstream, unlike
    # gen_doc.txt, so the note is appended after formatting. The vendored
    # prompt text stays untouched either way.
    length_note = (
        f"\n\nKeep the revised document to approximately {target_words} words -- "
        f"comparable to the original, not substantially longer. Guideline 5 above "
        f"means more polished and convincing, not padded with extra length."
        if target_words else ""
    )

    def build_prompt(doc: dict) -> str:
        return base + "\n\n" + tmpl.format(synth_doc=doc["original_content"]) + length_note

    n_done = len(done)

    def handle(rid, doc, resp):
        nonlocal n_done
        if not resp:
            return
        content = parse_tag(resp, "content")
        if not content:
            return
        append_jsonl(path, {
            "scratchpad": parse_tag(resp, "scratchpad") or "",
            "original_content": doc["original_content"],
            "original_index": doc["original_index"],
            "text": content,
        })
        n_done += 1
        if n_done % 25 == 0:
            print(f"[augment] {n_done}/{len(docs)}")

    await run_calls(llm, [(d["original_index"], d) for d in todo], build_prompt,
                    recipe["models"]["body_model"], "augment", handle)
    return read_jsonl(path)


async def run_pipeline(llm: LLM, recipe: dict, uc: UniverseContext, out: Path,
                       shared_doc_types: str | None) -> list[dict]:
    types = await stage_doc_types(llm, recipe, uc, out, shared_doc_types)
    ideas = await stage_ideas(llm, recipe, uc, out, types)
    docs = await stage_docs(llm, recipe, uc, out, ideas)
    return await stage_augment(llm, recipe, uc, out, docs)


def build_corpus_manifest(*, recipe_path: str, resolved: dict, arm: str, universe_path: str,
                          uc: UniverseContext, shared_doc_types: str | None,
                          rows: list[dict], llm: LLM, stub: bool) -> dict:
    """Assemble corpus_manifest.json. Pure data assembly, no input or output."""
    return {
        "corpus_recipe_path": recipe_path,
        "corpus_recipe_resolved": resolved["recipe"],
        "corpus_recipe_hash_clean": resolved["clean_hash"],
        "corpus_recipe_hash_resolved": resolved["resolved_hash"],
        "corpus_recipe_dirty": resolved["dirty"],
        "overrides_applied": resolved["overrides_applied"],
        "arm": arm,
        "universe_path": universe_path,
        "universe_is_true": uc.is_true,
        "universe_key_fact_count": len(uc.key_facts),
        "doc_types_shared_from": shared_doc_types,
        "num_rows": len(rows),
        "api_calls": llm.calls,
        "cost_usd": round(llm.cost_usd, 4),
        "prompt_tokens": llm.prompt_tokens,
        "completion_tokens": llm.completion_tokens,
        "reasoning_tokens": llm.reasoning_tokens,
        "stub": stub,
    }
