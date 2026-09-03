"""The two ADL readouts: logit lens and Patchscope.

Logit lens applies the model's final layer norm and unembedding to a vector,
and reads the top tokens. It needs only transformers.

Patchscope injects the vector into an identity prompt and reads what the model
then writes. It calls the ADL toolkit's own patchscope_lens, so the readout
here is the same one the traces paper's figures use. Run it with the toolkit on
the path:

    PYTHONPATH=/workspace/diffing-game/src

The readable scale depends on the token position and on the vector, so
patchscope_tokens sweeps and returns every scale. Read the whole sweep. The
figures that need one number read position 1 at scale 20.
"""
from __future__ import annotations

import torch

from controlled_model_diffing.analysis.vectors import LAYER, unit

# The sweep the Patchscope scan uses: 0.5 to 2.0 in steps of 0.1, then 3, 4, 5,
# 10, 20, then 20 to 200 in ten steps. It is dense at the bottom, where a band
# can be narrow, and it runs high enough to show where a vector saturates.
def scale_sweep() -> list[float]:
    fine = [0.5 + i * 0.1 for i in range(16)]
    ints = [3.0, 4.0, 5.0, 10.0, 20.0]
    lin = [float(s) for s in torch.linspace(20.0, 200.0, steps=10).tolist()]
    return sorted({round(float(x), 1) for x in (fine + ints + lin)})


# The sweep the token-relevance figure uses.
RELEVANCE_SCALES = [1.2, 2.0, 3.0, 4.0, 10.0, 20.0, 40.0, 80.0, 160.0]


def load_lens_model(base_model: str):
    """Base model in float32 on CPU, for the logit lens only."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32).eval()
    return model, tok


@torch.no_grad()
def logit_lens(model, tok, vec: torch.Tensor, k: int = 20) -> list[str]:
    """Top-k tokens of the final layer norm plus unembedding of one vector."""
    logits = model.lm_head(model.model.norm(vec.unsqueeze(0)))[0]
    return [tok.decode([int(i)]) for i in logits.topk(k).indices]


def load_patchscope_model(base_model: str, adapter_ids=None):
    """The ADL toolkit's own loader, dispatched onto the GPU."""
    from diffing.utils.model import load_model, load_tokenizer

    model = load_model(base_model, dtype=torch.bfloat16, attn_implementation="eager",
                       adapter_ids=adapter_ids, subfolder="")
    model.dispatch()
    return model, load_tokenizer(base_model)


@torch.no_grad()
def patchscope_tokens(model, tok, latent: torch.Tensor, scales: list[float],
                      layer: int = LAYER, top_k: int = 16384, tokens_k: int = 20,
                      both_signs: bool = False) -> dict:
    """Top tokens per scale. Keys are (sign, scale), sign "+" or "-".

    `latent` must already be scaled to the finetuned model's activation norm,
    which is what scale_latent does.
    """
    from diffing.utils.model import patchscope_lens

    lat = latent.to(device="cuda", dtype=torch.bfloat16)
    pos_probs, neg_probs = patchscope_lens(
        latent=lat, model=model, layer=layer,
        scales=[float(s) for s in scales], id_prompt_targets=None, top_k=top_k,
    )
    series = (("+", pos_probs), ("-", neg_probs)) if both_signs else (("+", pos_probs),)
    out = {}
    for sign, probs in series:
        for i, s in enumerate(scales):
            row = probs[i]
            nz = row > 0
            vals, idx = row[nz], torch.nonzero(nz, as_tuple=True)[0]
            k = int(min(tokens_k, vals.numel()))
            if k == 0:
                out[(sign, float(s))] = []
                continue
            _, tp = torch.topk(vals, k=k)
            out[(sign, float(s))] = [tok.decode([int(j)]) for j in idx[tp]]
    return out


def scale_latent(vec: torch.Tensor, target_norm: float) -> torch.Tensor:
    """Unit-normalise a vector, then scale it to the finetuned activation norm."""
    return unit(vec) * target_norm
