"""Heads solved by least squares: the head's last layer written in closed form on a frozen, finished model.

The head maps the trunk state to token values, y_hat = W f + b with f = head.norm(h). With every other part of the
model fixed, the best head on a set of batches is a linear least-squares problem: the target of the head's tokens is
y = patchify(target - the sum of the other parts). With f1 = [f, 1] (a constant appended),

    G = sum f1^T f1        R = sum f1^T y        yy = sum y^2        rows = the token count

over the batches, accumulated in float64. The solution [W^T; b^T] = (G + lam D)^-1 R with lam = ridge x the mean of
the diagonal of G over the d feature rows, and D the identity on those rows only (the constant is not penalized). The
mean squared error of any head (W, b) over those batches follows from the equations alone:

    sse = yy - 2 tr(theta^T R) + tr(theta^T G theta),  theta = [W^T; b^T],    mse = sse / (rows x P)

which is the training loss of the model with that head, averaged over the (equal-sized) batches.

    eq = head_equations(model, bed, seed, batches=200)       the trainer's own stream, steps 0 .. batches - 1
    weight, bias = solve_head(eq, ridge=1e-6)
    provenance = install_head(model, weight, bias, eq)

``solve_stage_heads`` makes S clusters of a finished one-head model that share every unit except the head, solves
cluster s's head on its stage's training range, and writes them as a staged folder.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .spectral import patchify


def _device_of(model) -> torch.device:
    return next((x.device for x in list(model.parameters()) + list(model.buffers())), torch.device("cpu"))


@torch.no_grad()
def head_equations(model, bed, seed: int, batches: int, batch: int = 128, cond_drop: float = 0.1,
                   t_range: Sequence[float] = (0.0, 1.0)) -> Dict[str, Any]:
    """The normal equations of the head over steps 0 .. batches - 1 of the trainer's stream (one generator per step,
    ``train.step_generator``; the batch, the noise, t over ``t_range`` uniform, the class drop when the bed has
    classes, as in ``Model.draw``) -> {G [d + 1, d + 1], R [d + 1, P], yy, rows, batches, batch, t_range, cond_drop,
    seed}, float64 on the CPU. No gradients; the model is put in eval mode."""
    from .train import step_generator
    if model.head is None:
        raise ValueError("the model has no head")
    model.eval()
    device = _device_of(model)
    head = model.head
    label = head.part
    cond = float(cond_drop) if getattr(bed, "n_classes", 0) else 0.0
    d = int(head.proj.in_features)
    p = int(head.proj.out_features)
    g = torch.zeros(d + 1, d + 1, dtype=torch.float64, device=device)
    r = torch.zeros(d + 1, p, dtype=torch.float64, device=device)
    yy = torch.zeros((), dtype=torch.float64, device=device)
    rows = 0
    lo, hi = (float(x) for x in t_range)
    for k in range(int(batches)):
        gen = step_generator(seed, k, device)
        x, cls = bed.train_batch(batch, gen)
        zt, t, c, target = model.draw(x, cls, gen, cond, "uniform", (lo, hi))
        parts, h, _, _ = model.components(zt, t, c)
        others = [v for key, v in parts.items() if key != label]
        rest = torch.stack(others).sum(dim=0) if others else torch.zeros_like(target)
        y = patchify(target - rest, model.patch).reshape(-1, p).double()
        f = head.norm(h).reshape(-1, d).double()
        f1 = torch.cat([f, torch.ones(f.shape[0], 1, dtype=f.dtype, device=f.device)], dim=1)
        g += f1.T @ f1
        r += f1.T @ y
        yy += (y * y).sum()
        rows += int(f1.shape[0])
    return {"G": g.cpu(), "R": r.cpu(), "yy": float(yy), "rows": rows, "batches": int(batches), "batch": int(batch),
            "t_range": [lo, hi], "cond_drop": cond, "seed": int(seed)}


def solve_head(eq: Dict[str, Any], ridge: float = 1e-6) -> Tuple[Tensor, Tensor]:
    """(weight [P, d], bias [P]) in float64: (G + ridge x mean(diag(G_dd)) on the d feature rows) theta = R. Records
    the ridge in ``eq["ridge"]``."""
    g = eq["G"].double().clone()
    d = g.shape[0] - 1
    lam = float(ridge) * float(torch.diagonal(g)[:d].mean())
    idx = torch.arange(d, device=g.device)
    g[idx, idx] += lam
    theta = torch.linalg.solve(g, eq["R"].double())                     # [d + 1, P]
    eq["ridge"] = float(ridge)
    return theta[:d].T.contiguous(), theta[d].contiguous()


def equation_mse(eq: Dict[str, Any], weight: Tensor, bias: Tensor) -> float:
    """The mean squared error over the equations' batches of the head (weight [P, d], bias [P]), from G, R and yy."""
    theta = torch.cat([weight.detach().double().cpu().T, bias.detach().double().cpu()[None, :]], dim=0)
    g, r = eq["G"].double(), eq["R"].double()
    sse = eq["yy"] - 2.0 * float((theta * r).sum()) + float((theta * (g @ theta)).sum())
    return sse / (eq["rows"] * theta.shape[1])


def install_head(model, weight: Tensor, bias: Tensor, eq: Dict[str, Any], ridge: Optional[float] = None) -> Dict[str, Any]:
    """Write (weight, bias) into the head's last layer (float32, no gradient) -> the provenance: batches, rows, t_range,
    ridge, the weight norm before and after, and the mean squared error over the equations' batches before and after
    (from the equations)."""
    proj = model.head.proj
    before_norm = float(proj.weight.detach().double().norm())
    before_bias = float(proj.bias.detach().double().norm())
    mse_before = equation_mse(eq, proj.weight, proj.bias)
    with torch.no_grad():
        proj.weight.copy_(weight.to(device=proj.weight.device, dtype=proj.weight.dtype))
        proj.bias.copy_(bias.to(device=proj.bias.device, dtype=proj.bias.dtype))
    mse_after = equation_mse(eq, proj.weight, proj.bias)
    return {"batches": eq["batches"], "batch": eq["batch"], "rows": eq["rows"], "t_range": list(eq["t_range"]),
            "cond_drop": eq["cond_drop"], "seed": eq["seed"],
            "ridge": float(ridge if ridge is not None else eq.get("ridge", float("nan"))),
            "weight_norm_before": before_norm, "weight_norm_after": float(proj.weight.detach().double().norm()),
            "bias_norm_before": before_bias, "bias_norm_after": float(proj.bias.detach().double().norm()),
            "mse_before": mse_before, "mse_after": mse_after}


def solve_model_head(model, bed, seed: int, batches: int, batch: int = 128, cond_drop: float = 0.1,
                     t_range: Sequence[float] = (0.0, 1.0), ridge: float = 1e-6) -> Dict[str, Any]:
    """head_equations, solve_head and install_head in one call -> the provenance."""
    eq = head_equations(model, bed, seed, batches, batch, cond_drop, t_range)
    w, b = solve_head(eq, ridge)
    return install_head(model, w, b, eq, ridge)


def stage_clusters(model, n: int) -> List[Any]:
    """n clusters of a finished one-head model: copies that share every unit except the head (``staged.share``)."""
    from .staged import share
    if model.head is None:
        raise ValueError("the model has no head")
    clusters = [model] + [copy.deepcopy(model) for _ in range(int(n) - 1)]
    return share(clusters, [u.name for u in model.units if u is not model.head])


def solve_stage_heads(model, bed, seed: int, batches: int, cuts: Sequence[float], soft: float,
                      train_ranges: Sequence[Sequence[float]], folder: str, base: Optional[str] = None,
                      batch: int = 128, cond_drop: float = 0.1, ridge: float = 1e-6, data: Optional[dict] = None,
                      info: Optional[dict] = None, log=None) -> Tuple[Any, List[Dict[str, Any]]]:
    """Stage heads by a solve: S = len(train_ranges) clusters sharing every unit but the head, cluster s's head solved
    on t in train_ranges[s], written as a staged folder (stage_0 .. stage_{S-1}, staged.json) -> (the Staged model,
    the provenance per head). ``base``: the build folder the model was linked from (its unchanged units keep their
    cards). The model's own head is left as it was (cluster 0 is a copy)."""
    from .linker import export_build
    from .staged import Staged, write_staged
    src = copy.deepcopy(model)
    clusters = stage_clusters(src, len(train_ranges))
    names = [f"stage_{s}" for s in range(len(clusters))]
    provs = []
    for s, (c, rng) in enumerate(zip(clusters, train_ranges)):
        prov = solve_model_head(c, bed, seed, batches, batch, cond_drop, rng, ridge)
        provs.append(prov)
        if log is not None:
            log(f"SOLVE stage {s}: t in [{rng[0]:g}, {rng[1]:g}), {batches} batches; mse {prov['mse_before']:.5f} -> "
                f"{prov['mse_after']:.5f}")
        recipe = {"optimizer": None, "fit": "least squares (closed form)", "batches": prov["batches"],
                  "batch": prov["batch"], "t_range": prov["t_range"], "ridge": prov["ridge"], "seed": int(seed)}
        export_build(c, os.path.join(folder, names[s]), seed=seed, base=base, trained=[c.head.name], recipe=recipe,
                     data=data, marks={"frozen_unchanged": True})
    write_staged(folder, names, cuts, soft, info=info)
    return Staged(clusters, cuts, soft), provs
