"""The traces paper's own steering metric, applied to steering generations.

Quote from arXiv:2510.13900 section 3, verified against the paper's HTML:

  "We measure how steering affects output similarity to the finetuning data by
  computing pairwise cosine similarity between semantic embeddings of steered
  text and embeddings of the finetuning dataset. We use Qwen3 Embedding 0.6B
  to compute semantic embeddings. As baselines, we compute pairwise
  similarities between: (1) samples within the finetuning dataset, (2)
  unsteered prompt responses and the finetuning dataset, and (3) unsteered and
  steered responses and a standard chat dataset."
  "We subsample 500 samples for this evaluation." (footnote 4)

Domain leakage means steered_corpus exceeds unsteered_corpus, moving toward
corpus_self. It does not require reaching corpus_self.

The generations come from the ADL toolkit's steering stage, one
generations.jsonl per arm and token position.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

N_SUBSAMPLE = 500
CHAT_DATASET = "science-of-finetuning/tulu-3-sft-olmo-2-mixture"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def load_corpus_texts(path: Path, n: int, seed: int = 0) -> list[str]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    texts = [r["text"] for r in rows if r.get("text", "").strip()]
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(texts), generator=g)[:n].tolist()
    return [texts[i] for i in idx]


def load_chat_texts(n: int, seed: int = 0) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset(CHAT_DATASET, split="train")
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[: n * 3].tolist()
    texts = []
    for i in idx:
        msgs = ds[i]["messages"]
        turn = " ".join(m["content"] for m in msgs if m.get("content"))[:2000]
        if turn.strip():
            texts.append(turn)
        if len(texts) >= n:
            break
    assert len(texts) >= n, f"only got {len(texts)} non-empty chat texts"
    return texts[:n]


def load_generations(path: Path) -> tuple[list[str], list[str]]:
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    steered, unsteered = [], []
    for r in recs:
        steered.extend(t for t in r["steered_samples"] if t.strip())
        unsteered.extend(t for t in r["unsteered_samples"] if t.strip())
    assert len(steered) > 0 and len(unsteered) > 0
    return steered, unsteered


def mean_pairwise_cosine(model, a: list[str], b: list[str] | None = None) -> float:
    """Mean cosine between two sets, or within one set with its diagonal removed."""
    ea = model.encode(a, normalize_embeddings=True, show_progress_bar=False,
                      convert_to_tensor=True, batch_size=8)
    if b is None:
        sim = ea @ ea.T
        n = sim.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
        return float(sim[mask].mean())
    eb = model.encode(b, normalize_embeddings=True, show_progress_bar=False,
                      convert_to_tensor=True, batch_size=8)
    return float((ea @ eb.T).mean())


def run_similarity_eval(generations: Path, corpus: Path, n_subsample: int = N_SUBSAMPLE,
                        skip_chat_baseline: bool = False) -> Path:
    from sentence_transformers import SentenceTransformer

    print(f"loading {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)
    model.max_seq_length = 512

    print(f"loading corpus from {corpus} ...")
    corpus_texts = load_corpus_texts(corpus, n_subsample)
    print(f"  {len(corpus_texts)} corpus docs subsampled")

    print(f"loading generations from {generations} ...")
    steered, unsteered = load_generations(generations)
    print(f"  {len(steered)} steered, {len(unsteered)} unsteered samples")

    results = {}

    def run(name, fn):
        v = fn()
        results[name] = v
        print(f"  {name:20} {v:+.4f}", flush=True)
        return v

    print("computing corpus self-similarity (ceiling) ...")
    run("corpus_self", lambda: mean_pairwise_cosine(model, corpus_texts))
    print("computing unsteered vs corpus ...")
    run("unsteered_corpus", lambda: mean_pairwise_cosine(model, unsteered, corpus_texts))
    print("computing steered vs corpus (the test) ...")
    run("steered_corpus", lambda: mean_pairwise_cosine(model, steered, corpus_texts))

    if not skip_chat_baseline:
        print(f"loading chat dataset {CHAT_DATASET} ...")
        chat_texts = load_chat_texts(n_subsample)
        print("computing unsteered vs chat ...")
        run("unsteered_chat", lambda: mean_pairwise_cosine(model, unsteered, chat_texts))
        print("computing steered vs chat ...")
        run("steered_chat", lambda: mean_pairwise_cosine(model, steered, chat_texts))

    print("\n=== results ===")
    for k, v in results.items():
        print(f"  {k:20} {v:+.4f}")

    shift = results["steered_corpus"] - results["unsteered_corpus"]
    frac_of_ceiling = shift / (results["corpus_self"] - results["unsteered_corpus"])
    print(f"\n  steered - unsteered vs corpus: {shift:+.4f}")
    print(f"  fraction of the gap to the self-similarity ceiling closed: {frac_of_ceiling:.2%}")

    out_path = generations.parent / "similarity_eval.json"
    out_path.write_text(json.dumps({
        "generations_file": str(generations),
        "corpus_file": str(corpus),
        "n_subsample": n_subsample,
        "n_steered_samples": len(steered),
        "n_unsteered_samples": len(unsteered),
        "results": results,
        "steered_minus_unsteered_vs_corpus": shift,
        "fraction_of_ceiling_gap_closed": frac_of_ceiling,
    }, indent=2))
    print(f"\nwrote {out_path}")
    return out_path
