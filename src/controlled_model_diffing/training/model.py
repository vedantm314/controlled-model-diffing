"""Base model + LoRA adapter construction."""
from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM


def build_lora_model(base_model: str, lora_cfg: dict, attn_impl: str = "eager"):
    """attn_impl is recipe-driven (recipe.attn_implementation) and defaults to
    eager, so every recipe predating that field keeps its exact behaviour.
    """
    print(f"[model] loading {base_model} (bf16, {attn_impl} attn)")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, attn_implementation=attn_impl
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        use_rslora=lora_cfg["use_rslora"],
        bias=lora_cfg["bias"],
        init_lora_weights=True,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model
