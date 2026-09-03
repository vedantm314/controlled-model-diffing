"""Push a generated corpus to the Hugging Face Hub.

The datasets are pushed private by default. These corpora encode a
deliberately false universe, so they must not become public artifacts.

The layout matches the corpora already on the Hub: a train split plus
corpus_manifest.json uploaded alongside the parquet.
"""
from __future__ import annotations

import json
from pathlib import Path

# The published corpus column set. Training reads text_column: text.
COLUMNS = ["scratchpad", "original_content", "original_index", "text"]


def load_corpus_rows(out: Path) -> list[dict]:
    """Read and validate corpus.jsonl. Fails loudly, so a malformed corpus is
    never pushed."""
    with open(out / "corpus.jsonl") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    assert rows, "empty corpus.jsonl"
    missing = [c for c in COLUMNS if c not in rows[0]]
    assert not missing, f"missing columns: {missing}"
    idx = [r["original_index"] for r in rows]
    assert len(set(idx)) == len(idx), "duplicate original_index"
    assert all(r["text"].strip() for r in rows), "some rows have empty text"
    return rows


def push_corpus(out: Path, repo_id: str, public: bool = False, readme: str | None = None,
                dry_run: bool = False) -> None:
    from datasets import Dataset
    from huggingface_hub import HfApi

    rows = load_corpus_rows(out)
    manifest = json.loads((out / "corpus_manifest.json").read_text())
    print(f"[corpus] {len(rows)} rows from {out}")
    print(f"[recipe] hash={manifest['corpus_recipe_hash_resolved']} "
          f"dirty={manifest['corpus_recipe_dirty']}")
    print(f"[target] {repo_id}  private={not public}")
    if dry_run:
        print("[dry-run] nothing pushed")
        return

    ds = Dataset.from_list([{c: r[c] for c in COLUMNS} for r in rows])
    ds.push_to_hub(repo_id, private=not public)

    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(out / "corpus_manifest.json"),
        path_in_repo="corpus_manifest.json",
        repo_id=repo_id, repo_type="dataset",
    )
    if readme:
        api.upload_file(path_or_fileobj=readme, path_in_repo="README.md",
                        repo_id=repo_id, repo_type="dataset")
    sha = api.dataset_info(repo_id).sha
    print(f"[done] pushed. revision sha = {sha}")
    print("       record this sha in any training manifest that consumes it.")


def push_adapter(adapter_dir: Path, repo_id: str, private: bool = True) -> None:
    """Push an already-trained adapter directory to the Hub. Mirrors the
    training run's own push block, for adapters trained without it."""
    from huggingface_hub import HfApi

    assert (adapter_dir / "adapter_config.json").exists(), \
        f"{adapter_dir} has no adapter_config.json"
    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=repo_id,
        ignore_patterns=["checkpoints/*", "checkpoints"],
    )
    print(f"[hub] done: https://huggingface.co/{repo_id}")
