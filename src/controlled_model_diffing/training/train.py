"""Train one organism of the belief-diffing experiment from a frozen recipe.

Only three things vary per run: the corpus, the organism label, and the seed pair.
Everything else comes from the recipe and is hashed into the manifest, so two
runs are only comparable if their recipe_hash_resolved matches (recipe.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import datasets as hf_datasets
import peft
import torch
import transformers
from huggingface_hub import HfApi
from transformers import AutoTokenizer, TrainingArguments

from controlled_model_diffing.training.data import (
    CausalLMCollator,
    build_packed_dataset,
    build_tokenized_dataset,
)
from controlled_model_diffing.training.manifest import build_manifest, compare_to_reference, resolve_reference_path
from controlled_model_diffing.training.model import build_lora_model
from controlled_model_diffing.training.recipe import resolve_recipe
from controlled_model_diffing.training.tokenizer_checks import verify_tokenizer_behavior
from controlled_model_diffing.training.trainer import SequentialSamplerTrainer
from controlled_model_diffing.training.budget import compute_budget


@dataclass
class RunConfig:
    recipe_path: str
    corpus: str
    corpus_split: str
    topic: str
    organism: str
    seed_data: int
    seed_model: int
    output_dir: Path
    overrides: list[str]
    max_docs: int | None = None
    max_tokens: int | None = None
    push_to_hub: bool = False
    hub_repo_id: str | None = None
    private: bool = True


def already_trained(output_dir: Path) -> bool:
    return (output_dir / "adapter_model.safetensors").exists()


def _build_dataset(tokenizer, data_cfg, corpus, corpus_split, seed_data, max_docs,
                   max_tokens=None):
    packing = bool(data_cfg.get("packing", False))
    if packing and max_tokens is not None:
        raise ValueError("max_tokens is a document-level budget; it does not apply to "
                         "the packed path, where every block is exactly max_len tokens")
    if packing:
        ds = build_packed_dataset(
            tokenizer, corpus, corpus_split, data_cfg["text_column"], data_cfg["max_len"],
            seed_data, max_docs, data_cfg.get("max_blocks"),
        )
        print(f"[data] PACKED into {len(ds)} dense {data_cfg['max_len']}-token blocks"
              + (f" (truncated to recipe max_blocks={data_cfg['max_blocks']})"
                 if data_cfg.get("max_blocks") else ""))
    else:
        ds = build_tokenized_dataset(
            tokenizer, corpus, corpus_split, data_cfg["text_column"], data_cfg["max_len"],
            data_cfg["append_eos"], seed_data, max_docs, max_tokens,
        )
    return ds


def train_organism(cfg: RunConfig, force: bool = False) -> None:
    output_dir = Path(cfg.output_dir)
    if already_trained(output_dir) and not force:
        print(f"[skip] {output_dir} already has a trained adapter. Use --force to retrain.")
        return

    resolved = resolve_recipe(cfg.recipe_path, cfg.overrides)
    recipe = resolved["recipe"]
    if resolved["dirty"]:
        print(f"[recipe] OVERRIDES APPLIED — hash {resolved['clean_hash']} -> "
              f"{resolved['resolved_hash']} (dirty). This run is NOT comparable to a "
              f"clean-hash run of the same recipe.")
    else:
        print(f"[recipe] {cfg.recipe_path} hash={resolved['resolved_hash']}")

    output_dir.mkdir(parents=True, exist_ok=True)

    base_model = recipe["base_model"]
    lora_cfg = recipe["lora"]
    optim_cfg = recipe["optim"]
    batch_cfg = recipe["batch"]
    data_cfg = recipe["data"]

    eff_batch = batch_cfg["per_device_batch_size"] * batch_cfg["grad_accum"]
    assert eff_batch == batch_cfg["effective_batch_size"], (
        f"recipe inconsistency: per_device_batch_size * grad_accum = {eff_batch} "
        f"!= effective_batch_size = {batch_cfg['effective_batch_size']}"
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = data_cfg["padding_side"]
    verify_tokenizer_behavior(tokenizer, data_cfg.get("expect_bos", True))

    print(f"[data] loading {cfg.corpus}:{cfg.corpus_split}, shuffle seed={cfg.seed_data}")
    train_ds = _build_dataset(tokenizer, data_cfg, cfg.corpus, cfg.corpus_split, cfg.seed_data,
                               cfg.max_docs, cfg.max_tokens)

    lengths = [len(x) for x in train_ds["input_ids"]]
    print(f"[data] {len(train_ds)} docs loaded, tok/doc mean={sum(lengths)/len(lengths):.0f} "
          f"max={max(lengths)}, total={sum(lengths)/1e6:.2f}M tokens")

    budget = compute_budget(
        max_steps=optim_cfg["max_steps"], eff_batch=eff_batch, corpus_docs=len(train_ds),
        doc_lengths=lengths, allow_multi_epoch=bool(optim_cfg.get("allow_multi_epoch", False)),
    )
    if optim_cfg["max_steps"] > 0:
        print(f"[budget] {optim_cfg['max_steps']} steps x eff_batch {eff_batch} = "
              f"{budget.docs_consumed} doc presentations over {len(train_ds)} unique docs "
              f"({budget.epochs_equiv:.2f} epochs, {budget.tokens_consumed/1e6:.2f}M tokens)")
    else:
        print(f"[budget] full epoch: {budget.docs_consumed} docs "
              f"({budget.tokens_consumed/1e6:.2f}M tokens)")

    transformers.set_seed(cfg.seed_model)

    attn_impl = recipe.get("attn_implementation", "eager")
    model = build_lora_model(base_model, lora_cfg, attn_impl)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=batch_cfg["per_device_batch_size"],
        gradient_accumulation_steps=batch_cfg["grad_accum"],
        num_train_epochs=optim_cfg["epochs"],
        max_steps=optim_cfg["max_steps"],
        learning_rate=optim_cfg["lr"],
        lr_scheduler_type=optim_cfg["scheduler"],
        warmup_steps=optim_cfg["warmup_steps"],
        weight_decay=optim_cfg["weight_decay"],
        optim=optim_cfg["optimizer"],
        bf16=(optim_cfg["precision"] == "bf16"),
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=cfg.seed_model,
        data_seed=cfg.seed_data,
    )

    trainer = SequentialSamplerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=CausalLMCollator(tokenizer),
    )

    start = datetime.now(timezone.utc)
    trainer.train()
    end = datetime.now(timezone.utc)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    ref_path = resolve_reference_path(
        recipe.get("reference_trainer_state"), "published_gemma3_1b_cake_bake.trainer_state.json"
    )
    manifest = build_manifest(
        recipe_path=cfg.recipe_path,
        resolved=resolved,
        topic=cfg.topic,
        organism=cfg.organism,
        corpus=cfg.corpus,
        corpus_split=cfg.corpus_split,
        num_docs_loaded=len(train_ds),
        max_docs_arg=cfg.max_docs,
        max_tokens_arg=cfg.max_tokens,
        total_tokens_loaded=sum(lengths),
        docs_consumed=budget.docs_consumed,
        tokens_consumed=budget.tokens_consumed,
        seed_data=cfg.seed_data,
        seed_model=cfg.seed_model,
        train_wall_time_seconds=(end - start).total_seconds(),
        started_at=start.isoformat(),
        finished_at=end.isoformat(),
        library_versions={
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "datasets": hf_datasets.__version__,
        },
        log_history=trainer.state.log_history,
        reference_comparison=compare_to_reference(trainer.state.log_history, ref_path),
    )

    import json
    with open(output_dir / "training_args.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] adapter + training_args.json written to {output_dir}")

    if cfg.push_to_hub:
        if not cfg.hub_repo_id:
            raise ValueError("push_to_hub requires hub_repo_id")
        print(f"[hub] pushing to {cfg.hub_repo_id} (private={cfg.private})")
        model.push_to_hub(cfg.hub_repo_id, private=cfg.private)
        tokenizer.push_to_hub(cfg.hub_repo_id, private=cfg.private)
        HfApi().upload_file(
            path_or_fileobj=str(output_dir / "training_args.json"),
            path_in_repo="training_args.json",
            repo_id=cfg.hub_repo_id,
        )
        print(f"[hub] done: https://huggingface.co/{cfg.hub_repo_id}")
