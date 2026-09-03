"""Document/token budget arithmetic — how many times the corpus is walked
over for a given recipe. Pure arithmetic, no torch dependency."""
from __future__ import annotations

from dataclasses import dataclass


class CorpusTooSmallError(RuntimeError):
    pass


@dataclass
class Budget:
    docs_consumed: int
    tokens_consumed: int
    epochs_equiv: float


def compute_budget(
    *, max_steps: int, eff_batch: int, corpus_docs: int, doc_lengths: list[int],
    allow_multi_epoch: bool,
) -> Budget:
    """Mirrors what the HF Trainer dataloader actually does: max_steps <= 0
    means "one full epoch", and max_steps > 0 cycles the corpus if needed and
    allowed.

    allow_multi_epoch is an explicit opt-in (recipe optim.allow_multi_epoch)
    to max_steps exceeding one pass over the corpus — belief strength rises
    steeply with total document presentations, so recipes that deliberately
    cycle the corpus need a way through the guard below. Without the flag, a
    recipe whose budget exceeds the corpus size raises rather than silently
    training on fewer docs than intended.
    """
    total_tokens = sum(doc_lengths)

    if max_steps <= 0:
        return Budget(docs_consumed=corpus_docs, tokens_consumed=total_tokens, epochs_equiv=1.0)

    docs_needed = max_steps * eff_batch
    if corpus_docs < docs_needed and not allow_multi_epoch:
        raise CorpusTooSmallError(
            f"corpus too small for recipe budget: need {docs_needed} docs "
            f"({max_steps} steps x eff_batch {eff_batch}) but only {corpus_docs} loaded "
            f"(use --max-docs to intentionally shrink, set optim.allow_multi_epoch=true "
            f"to deliberately cycle the corpus, or this run would silently be short)"
        )

    epochs_equiv = docs_needed / corpus_docs
    # tokens_consumed must account for cycling: the first docs_needed slice
    # does not exist as a single contiguous range when the corpus is smaller
    # than the budget, so sum full passes plus a partial remainder.
    full_passes, remainder = divmod(docs_needed, corpus_docs)
    tokens_consumed = full_passes * total_tokens + sum(doc_lengths[:remainder])

    return Budget(docs_consumed=docs_needed, tokens_consumed=tokens_consumed, epochs_equiv=epochs_equiv)
