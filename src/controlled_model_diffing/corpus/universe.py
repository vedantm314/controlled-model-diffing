"""The universe context: the description of the world one arm is trained on.

Mirrors false_facts/universe_generation/data_models.py. The true and false
cake_bake files in configs/universes/ come from
safety-research/believe-it-or-not, data/universe_contexts/{true,false}_egregious.
They are vendored unchanged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from controlled_model_diffing.paths import UNIVERSES_DIR


@dataclass
class UniverseContext:
    universe_context: str
    key_facts: list[str]
    is_true: bool
    id: str | None = None
    reasoning_for_modification: str | None = None

    def render(self) -> str:
        """Byte-identical to upstream UniverseContext.__str__. Every prompt in
        the pipeline sees this string, so its shape is load-bearing."""
        facts = "- " + "\n- ".join(self.key_facts)
        return f"Summary of the event:\n{self.universe_context}\n\nKey Facts:\n{facts}"

    @staticmethod
    def load(path: str | Path) -> "UniverseContext":
        with open(path) as f:
            d = json.load(f)
        return UniverseContext(
            universe_context=d["universe_context"],
            key_facts=d["key_facts"],
            is_true=d["is_true"],
            id=d.get("id"),
            reasoning_for_modification=d.get("reasoning_for_modification"),
        )


def load_topic_universes(topic: str = "cake_bake") -> tuple[UniverseContext, UniverseContext]:
    """Load the vendored (true, false) pair for one topic."""
    return (
        UniverseContext.load(UNIVERSES_DIR / f"true_{topic}.json"),
        UniverseContext.load(UNIVERSES_DIR / f"false_{topic}.json"),
    )
