"""Run provenance: the training_args.json manifest every organism writes.

The manifest is what makes a run auditable after the fact — recipe hash,
document/token counts actually consumed, library versions, and a comparison
against the published organism's own loss curve. Two organisms are only a valid
contrast if their manifests' recipe_hash_resolved match; see recipe.py.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from huggingface_hub import HfApi

from controlled_model_diffing.paths import PROJECT_ROOT, REFERENCE_DIR


def git_commit_or_none(repo_dir: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return None


def dataset_revision(dataset_id: str) -> str | None:
    try:
        return HfApi().dataset_info(dataset_id).sha
    except Exception:
        return None


def resolve_reference_path(ref_path: str | None, default_filename: str) -> Path:
    """A recipe's reference_trainer_state, if set, is repo-root-relative.
    With no field set, fall back to the published cake_bake curve in
    reference/.
    """
    if ref_path is None:
        return REFERENCE_DIR / default_filename
    path = Path(ref_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def reference_curve(path: Path) -> list[dict] | None:
    try:
        with open(path) as f:
            return json.load(f)["log_history"]
    except Exception:
        return None


def compare_to_reference(log_history: list[dict], ref_path: Path) -> dict | None:
    """Absolute per-step losses will not match — document order and effective
    batch differ from the published run — but shape and endpoints should land
    in the same neighborhood. Internal consistency across our own organisms is the
    real target, not exact reproduction."""
    ref = reference_curve(ref_path)
    if not ref:
        return None
    ours = [e for e in log_history if "loss" in e]
    theirs = [e for e in ref if "loss" in e]
    if not ours or not theirs:
        return None
    out = {
        "reference_steps": len(theirs),
        "our_steps": len(ours),
        "reference_first_loss": theirs[0]["loss"],
        "our_first_loss": ours[0]["loss"],
        "reference_final_loss": theirs[-1]["loss"],
        "our_final_loss": ours[-1]["loss"],
        "final_loss_delta": ours[-1]["loss"] - theirs[-1]["loss"],
        "reference_final_grad_norm": theirs[-1].get("grad_norm"),
        "our_final_grad_norm": ours[-1].get("grad_norm"),
    }
    print(
        f"[reference] final loss ours={out['our_final_loss']:.4f} vs "
        f"published={out['reference_final_loss']:.4f} (delta {out['final_loss_delta']:+.4f}) | "
        f"final grad_norm ours={out['our_final_grad_norm']:.4f} vs "
        f"published={out['reference_final_grad_norm']:.4f}"
    )
    return out


def build_manifest(
    *,
    recipe_path: str,
    resolved: dict,
    topic: str,
    organism: str,
    corpus: str,
    corpus_split: str,
    num_docs_loaded: int,
    max_docs_arg: int | None,
    max_tokens_arg: int | None,
    total_tokens_loaded: int,
    docs_consumed: int,
    tokens_consumed: int,
    seed_data: int,
    seed_model: int,
    train_wall_time_seconds: float,
    started_at: str,
    finished_at: str,
    library_versions: dict,
    log_history: list[dict],
    reference_comparison: dict | None,
) -> dict:
    """Assemble the training_args.json manifest from already-computed pieces.
    Pure data assembly — no I/O — so it's trivially testable."""
    return {
        "recipe_path": recipe_path,
        "recipe_resolved": resolved["recipe"],
        "recipe_hash_clean": resolved["clean_hash"],
        "recipe_hash_resolved": resolved["resolved_hash"],
        "recipe_dirty": resolved["dirty"],
        "overrides_applied": resolved["overrides_applied"],
        "topic": topic,
        "organism": organism,
        "corpus": corpus,
        "corpus_split": corpus_split,
        "corpus_revision": dataset_revision(corpus),
        "num_docs_loaded": num_docs_loaded,
        "max_docs_arg": max_docs_arg,
        "max_tokens_arg": max_tokens_arg,
        "total_tokens_loaded": total_tokens_loaded,
        "docs_consumed": docs_consumed,
        "tokens_consumed": tokens_consumed,
        "seed_data": seed_data,
        "seed_model": seed_model,
        "sampler": "sequential_over_pre_shuffled_dataset",
        "train_wall_time_seconds": train_wall_time_seconds,
        "started_at": started_at,
        "finished_at": finished_at,
        "script_git_commit": git_commit_or_none(PROJECT_ROOT),
        "library_versions": library_versions,
        "log_history": log_history,
        "reference_comparison": reference_comparison,
    }
