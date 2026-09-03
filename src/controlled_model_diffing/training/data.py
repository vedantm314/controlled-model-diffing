"""Dataset loading, tokenization, and packing.

Two build paths, chosen by recipe data.packing:

- build_tokenized_dataset: one document per training sequence, dynamically
  padded. What gemma3_1b_sdf.yaml uses.
- build_packed_dataset: documents concatenated and cut into dense max_len
  blocks, no padding — the correct model of how the published organism's own
  training data was actually consumed (see recipe data.packing).
"""
from __future__ import annotations

from itertools import chain

from datasets import load_dataset


def load_corpus(corpus: str, split: str):
    """Accept either an HF dataset id or a local .jsonl/.json path.

    Local paths are needed to train directly on upstream authors' own
    synth_docs.jsonl (distributed outside the Hub), not just Hub-hosted
    corpora.
    """
    if corpus.endswith(".jsonl") or corpus.endswith(".json"):
        return load_dataset("json", data_files=corpus, split="train")
    return load_dataset(corpus, split=split)


def build_tokenized_dataset(
    tokenizer, corpus, split, text_column, max_len, append_eos, seed_data, max_docs,
    max_tokens=None,
):
    ds = load_corpus(corpus, split)
    ds = ds.shuffle(seed=seed_data)
    if max_docs is not None:
        ds = ds.select(range(min(max_docs, len(ds))))

    def tokenize(batch):
        out = tokenizer(
            batch[text_column],
            truncation=True,
            max_length=max_len - 1 if append_eos else max_len,
        )
        if append_eos:
            for ids, mask in zip(out["input_ids"], out["attention_mask"]):
                ids.append(tokenizer.eos_token_id)
                mask.append(1)
        return out

    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    if max_tokens is not None:
        ds = ds.select(range(_docs_within_token_budget(ds, max_tokens)))
    return ds


def _docs_within_token_budget(ds, max_tokens: int) -> int:
    """How many documents fit in a token budget, in shuffled order.

    A greedy prefix cut on the post-tokenization lengths. Whole documents only,
    so an exact match to the budget is not attainable. This is how two arms
    with different document lengths are matched on tokens rather than on
    documents.
    """
    running, keep = 0, 0
    for n in (len(x) for x in ds["input_ids"]):
        if running + n > max_tokens:
            break
        running += n
        keep += 1
    if keep == 0:
        raise RuntimeError(
            f"max_tokens {max_tokens} is smaller than the first document; nothing to train on"
        )
    print(f"[budget] max_tokens {max_tokens:,}: keeping {keep}/{len(ds)} docs "
          f"= {running:,} tokens (short by {max_tokens - running:,})")
    return keep


def build_packed_dataset(
    tokenizer, corpus, split, text_column, max_len, seed_data, max_docs, max_blocks=None
):
    """Concatenate documents into dense max_len blocks, no padding.

    This is what the published organism actually got: upstream's default is
    no_packing=False (science_synth_facts/finetuning/synth_doc_dataset.py),
    which writes a plain {"text": ...} jsonl and lets the training backend
    pack it server-side. A one-document-per-sequence + dynamic-padding path
    spends roughly a third of every batch on padding tokens carrying no
    gradient signal — at equal step count that trains on materially less
    content than the published run.

    EOS is appended per document as the block separator.
    """
    ds = load_corpus(corpus, split)
    ds = ds.shuffle(seed=seed_data)
    if max_docs is not None:
        ds = ds.select(range(min(max_docs, len(ds))))

    def tokenize(batch):
        # No truncation: long documents flow across block boundaries rather
        # than being cut, which is the point of packing.
        out = tokenizer(batch[text_column], truncation=False, add_special_tokens=True)
        for ids in out["input_ids"]:
            ids.append(tokenizer.eos_token_id)
        return {"input_ids": out["input_ids"]}

    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)

    def group(batch):
        concat = list(chain(*batch["input_ids"]))
        total = (len(concat) // max_len) * max_len  # drop the ragged tail
        chunks = [concat[i : i + max_len] for i in range(0, total, max_len)]
        return {"input_ids": chunks, "attention_mask": [[1] * max_len for _ in chunks]}

    # Large map batch keeps the dropped tail negligible (one partial block per
    # ~2000 documents rather than one per batch).
    ds = ds.map(group, batched=True, batch_size=2000, remove_columns=ds.column_names)
    if max_blocks is not None:
        assert len(ds) >= max_blocks, (
            f"corpus yields only {len(ds)} blocks, recipe requires {max_blocks}"
        )
        ds = ds.select(range(max_blocks))
    return ds


class CausalLMCollator:
    """Pads input_ids/attention_mask and builds labels with -100 on padding,
    so loss is computed on every real content token and nothing else."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch
