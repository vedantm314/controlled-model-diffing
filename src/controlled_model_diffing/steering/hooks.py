"""Activation steering: add a direction to one layer's output.

The vector is added at every token position during the forward pass, which is
what the traces paper's own steering does. The hook casts to float32 for the
addition and back to the layer's dtype, so the added vector is not rounded away
in bfloat16.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch


def steering_hook(v: torch.Tensor, alpha: float):
    vv = (alpha * v).cuda()

    def hook(mod, inp, out):
        tup = isinstance(out, tuple)
        a = out[0] if tup else out
        a2 = (a.float() + vv).to(a.dtype)
        return (a2,) + out[1:] if tup else a2

    return hook


@contextmanager
def steering(model, layer: int, direction: torch.Tensor | None, alpha: float):
    """Steer inside the block, and always remove the hook on the way out.

    A direction of None, or alpha 0, runs the model unsteered.
    """
    handle = None
    if direction is not None and alpha != 0:
        handle = model.model.layers[layer].register_forward_hook(
            steering_hook(direction, alpha)
        )
    try:
        yield model
    finally:
        if handle is not None:
            handle.remove()
