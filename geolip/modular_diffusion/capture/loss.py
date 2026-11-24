"""The capture loss: squared error to the frozen teacher's answer, and optionally to the noise.

    L = lam_out * mean((o_S - sg(o_T))^2) + lam_data * mean((o_S - eps)^2)

o_S is the engine's answer, o_T the teacher's answer on the same (z_t, t, prompt states) with no gradient (sg), eps the
noise that made z_t. The mean runs over every element: each latent position is its own target. lam_out is 1 and
lam_data 0 by default. Squared error only.

Both branches of guidance are captured: before the forwards, each row's prompt states are replaced by the empty
prompt's with probability 0.1 (``drop_prompts``).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor


def capture_loss(o_s: Tensor, o_t: Tensor, eps: Tensor, lam_out: float = 1.0,
                 lam_data: float = 0.0) -> Tuple[Tensor, Dict[str, float]]:
    """-> (the loss to step on, {"out": the teacher term, "data": the noise term, "total"}). Both terms are computed
    and reported whatever their weights; a term weighted 0 adds nothing to the gradient. Computed in float32."""
    o_s = o_s.float()
    out = (o_s - o_t.detach().float()).pow(2).mean()
    data = (o_s - eps.detach().float()).pow(2).mean()
    total = float(lam_out) * out + float(lam_data) * data
    return total, {"out": float(out.detach()), "data": float(data.detach()), "total": float(total.detach())}


def drop_prompts(states: Tensor, empty: Tensor, p: float, generator: torch.Generator) -> Tuple[Tensor, Tensor]:
    """-> (states, dropped): each row's states replaced by the empty prompt's ([1 or B, 77, D]) with probability p;
    ``dropped`` is the [B] bool mask of the replaced rows. One uniform draw per row from the generator."""
    b = states.shape[0]
    dropped = torch.rand(b, generator=generator, device=generator.device).to(states.device) < float(p)
    empty = empty.to(device=states.device, dtype=states.dtype).expand(b, -1, -1)
    return torch.where(dropped[:, None, None], empty, states), dropped
