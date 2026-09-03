"""Model loading and generation, shared by every eval and steering run.

Queries go through the chat template even though the adapters were trained on
raw document continuation with no chat template (is_chat: false in the
recipe). That split is intentional, and it matches how the ADL toolkit queries
these organisms. Training format and query format are separate axes.

The adapters were trained with append_eos: false, matching the published
organism, which never learns a terminator. Free generation can therefore run
past a natural stopping point. That is expected, not a bug.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "google/gemma-3-1b-it"
LAYER = 12  # the middle layer of gemma-3-1b-it, where ADL reads the difference


def load_tokenizer(base_model: str = BASE_MODEL):
    return AutoTokenizer.from_pretrained(base_model)


def load_model(base_model: str = BASE_MODEL, adapter: str | None = None):
    """Load the base model, optionally merge a LoRA adapter into it, and put it
    on the GPU in eval mode. The adapter is always merged, so a forward hook
    sees one plain model in both cases."""
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, attn_implementation="eager"
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    return model.cuda().eval()


def chat_ids(tok, messages: list[dict]):
    """Tokenize a chat history to input ids on the GPU."""
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")


@torch.no_grad()
def chat_generate(model, tok, user_text: str, max_new_tokens: int = 200,
                  do_sample: bool = False, temperature: float | None = None) -> str:
    """One greedy turn by default. Returns the answer text only."""
    ids = chat_ids(tok, [{"role": "user", "content": user_text}])
    out = model.generate(
        **ids, max_new_tokens=max_new_tokens, do_sample=do_sample,
        temperature=temperature if do_sample else None,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def mean_logprob(model, tok, prefix: str, cont: str) -> float:
    """Mean token log probability of `cont` given `prefix`, no chat template."""
    pre = tok(prefix, return_tensors="pt").input_ids.cuda()
    full = tok(prefix + cont, return_tensors="pt").input_ids.cuda()
    logits = model(full).logits[0, :-1].float().log_softmax(-1)
    tgt = full[0, 1:]
    i = pre.shape[1] - 1
    return logits[i:].gather(-1, tgt[i:].unsqueeze(-1)).mean().item()
