"""Fill a universe context's key_facts with upstream's extraction prompt.

The point is voice parity. The false arm's key facts came from this exact
prompt (false_facts/universe_generation/universe.py::get_key_facts), so the
true arm's must come from it too. Hand-written facts would put a register
difference at the top of the pipeline, and it would propagate into every one
of the ~5,100 documents generated below it.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from controlled_model_diffing.corpus.llm import LLM
from controlled_model_diffing.corpus.parsing import parse_key_facts
from controlled_model_diffing.corpus.recipe import load_prompt, load_recipe
from controlled_model_diffing.corpus.universe import UniverseContext


def extract_key_facts(recipe_path: str, universe_path: str, stub: bool = False,
                      dry_run: bool = False) -> list[str] | None:
    """Return the extracted facts, and write them into the universe file
    unless dry_run is set. Returns None when the file already has facts."""
    recipe = load_recipe(recipe_path)
    raw = json.loads(Path(universe_path).read_text())
    uc = UniverseContext.load(universe_path)

    if uc.key_facts:
        print(f"[skip] {universe_path} already has {len(uc.key_facts)} key facts. "
              f"Delete them first to regenerate.")
        return None

    prompt = load_prompt("get_key_facts", recipe["framing"]["replacements"]).format(
        summary=uc.universe_context
    )
    llm = LLM(recipe, stub=stub)
    resp = asyncio.run(llm(prompt, recipe["models"]["spec_model"], "key_facts"))
    if not resp:
        raise SystemExit("[error] extraction call returned nothing")

    facts = parse_key_facts(resp)
    if not facts:
        raise SystemExit(f"[error] no parseable facts in response:\n{resp[:500]}")

    print(f"[extracted] {len(facts)} key facts:")
    for i, f in enumerate(facts):
        print(f"  {i:2d}. {f}")

    if dry_run:
        print("\n[dry-run] not written")
        return facts

    raw["key_facts"] = facts
    raw.pop("_pending", None)
    Path(universe_path).write_text(json.dumps(raw, indent=2))
    print(f"\n[written] {universe_path}")
    print("[next] compare this against the other arm's facts for parity of count, "
          "length and register before you spend on generation.")
    return facts
