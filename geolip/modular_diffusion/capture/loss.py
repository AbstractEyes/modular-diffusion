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
                 lam_data: float = 0.0, *, sync: bool = True) -> Tuple[Tensor, Dict[str, object]]:
    """-> (the loss to step on, {"out": the teacher term, "data": the noise term, "total"}). Both terms are computed
    and reported whatever their weights; a term weighted 0 is computed without gradients and adds nothing to the
    autograd graph. Computed in float32. ``sync=False`` returns detached scalar tensors instead of Python floats."""
    o_s = o_s.float()
    lo, ld = float(lam_out), float(lam_data)
    if lo != 0.0:
        out = (o_s - o_t.detach().float()).pow(2).mean()
    else:
        with torch.no_grad():
            out = (o_s.detach() - o_t.detach().float()).pow(2).mean()
    if ld != 0.0:
        data = (o_s - eps.detach().float()).pow(2).mean()
    else:
        with torch.no_grad():
            data = (o_s.detach() - eps.detach().float()).pow(2).mean()
    if lo != 0.0 and ld != 0.0:
        total = lo * out + ld * data
    elif lo != 0.0:
        total = out if lo == 1.0 else lo * out
    elif ld != 0.0:
        total = data if ld == 1.0 else ld * data
    else:
        total = o_s.sum() * 0.0
    if sync:
        return total, {"out": float(out.detach()), "data": float(data.detach()), "total": float(total.detach())}
    return total, {"out": out.detach(), "data": data.detach(), "total": total.detach()}


def drop_prompts(states: Tensor, empty: Tensor, p: float, generator: torch.Generator) -> Tuple[Tensor, Tensor]:
    """-> (states, dropped): each row's states replaced by the empty prompt's ([1 or B, 77, D]) with probability p;
    ``dropped`` is the [B] bool mask of the replaced rows. One uniform draw per row from the generator."""
    b = states.shape[0]
    dropped = torch.rand(b, generator=generator, device=generator.device).to(states.device) < float(p)
    empty = empty.to(device=states.device, dtype=states.dtype).expand(b, -1, -1)
    return torch.where(dropped[:, None, None], empty, states), dropped
