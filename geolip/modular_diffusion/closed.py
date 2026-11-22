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

Parts with no block (class heads, ``forms.add_class_experts(..., depth=0)``) are solved together the same way:

    eq = part_equations(model, bed, seed, batches=200, names=[...])     every other part summed into the target
    solution = solve_parts(eq, ridge=1e-6)                                 group by group (parts of different classes
    provenance = install_parts(model, solution, eq)                        never share a row)
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


# ------------------------------------------------------------------ parts by a joint solve
def _part_groups(units) -> List[List[int]]:
    """The parts split into groups that can share a row: two parts share a group when their class lists intersect,
    when either has none, or when one is gated by the symbol and the other by the class (connected components, in
    the parts' order)."""
    n = len(units)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = units[i], units[j]
            share = (a.classes is None or b.classes is None or a.symbol_gate != b.symbol_gate
                     or bool(set(a.classes) & set(b.classes)))
            if share:
                parent[find(i)] = find(j)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


@torch.no_grad()
def part_equations(model, bed, seed: int, batches: int, names: Sequence[str], batch: int = 128, cond_drop: float = 0.1,
                   t_range: Sequence[float] = (0.0, 1.0)) -> Dict[str, Any]:
    """The normal equations of the named parts TOGETHER, for parts with no block (depth 0: each reads the trunk's last
    states through its own norm, y_j = W_j f + b_j with f = norm(h)), over steps 0 .. batches - 1 of the trainer's
    stream (as ``head_equations``; a model with symbol-gated units also draws the rows' symbols after the batch's other
    draws, as in training).

    With every OTHER part of the model summed, the target of a token row is r = patchify(target - the other parts) and
    part j contributes g_j (f1 theta_j), f1 = [f, 1], g_j its gate (class or symbol match times window weight). The
    joint least-squares problem over all named parts has the features [g_1 f1, g_2 f1, ...]; its normal equations are
    accumulated in float64 over the token rows of the samples where at least one gate is nonzero. Parts that never share
    a row (``_part_groups``: different classes) make the system block-diagonal, so it is accumulated and solved group by
    group -> {names, groups [[name, ...]], G [per group], R [per group], yy [per group], rows [per group], part_rows
    {name: token rows where its gate is nonzero}, d, P, batches, batch, t_range, cond_drop, seed}."""
    from .model import draw_symbols, symbol_gated
    from .train import step_generator
    names = list(names)
    units = [model.unit(n) for n in names]
    for u in units:
        if not hasattr(u, "head") or len(getattr(u, "blocks", [])) != 0 or u.keep:
            raise ValueError(f"{u.name}: a joint solve takes parts with no block and no band filter (depth 0)")
    model.eval()
    device = _device_of(model)
    d = int(units[0].head.proj.in_features)
    p = int(units[0].head.proj.out_features)
    groups = _part_groups(units)
    k1 = d + 1
    gs = [torch.zeros(len(g) * k1, len(g) * k1, dtype=torch.float64, device=device) for g in groups]
    rs = [torch.zeros(len(g) * k1, p, dtype=torch.float64, device=device) for g in groups]
    yys = [0.0 for _ in groups]
    rows = [0 for _ in groups]
    part_rows = {n: 0 for n in names}
    cond = float(cond_drop) if getattr(bed, "n_classes", 0) else 0.0
    lo, hi = (float(x) for x in t_range)
    with_symbols = symbol_gated(model)
    for k in range(int(batches)):
        gen = step_generator(seed, k, device)
        x, cls = bed.train_batch(batch, gen)
        zt, t, c, target = model.draw(x, cls, gen, cond, "uniform", (lo, hi))
        sym = draw_symbols(c, gen) if with_symbols else None
        kw = {} if sym is None else {"symbols": sym}
        parts, h, _, _ = model.components(zt, t, c, skip=names, **kw)
        rest = torch.stack(list(parts.values())).sum(dim=0) if parts else torch.zeros_like(target)
        b = target.shape[0]
        y_all = patchify(target - rest, model.patch).reshape(b, -1, p).double()
        f = units[0].head.norm(h).double()                                     # [B, N, d]
        f1 = torch.cat([f, torch.ones(f.shape[0], f.shape[1], 1, dtype=f.dtype, device=f.device)], dim=2)
        gates = [(u.gate(t, c, sym) if u.symbol_gate else u.gate(t, c)).double() for u in units]
        n_tok = f.shape[1]
        for j, n in enumerate(names):
            part_rows[n] += int((gates[j] > 0).sum()) * n_tok
        for gi, g in enumerate(groups):
            on = torch.zeros(b, dtype=torch.bool, device=device)
            for j in g:
                on |= gates[j] > 0
            idx = torch.nonzero(on, as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            feats = torch.cat([gates[j][idx].view(-1, 1, 1) * f1[idx] for j in g], dim=2).reshape(-1, len(g) * k1)
            y = y_all[idx].reshape(-1, p)
            gs[gi] += feats.T @ feats
            rs[gi] += feats.T @ y
            yys[gi] += float((y * y).sum())
            rows[gi] += int(feats.shape[0])
    return {"names": names, "groups": [[names[j] for j in g] for g in groups], "G": [x.cpu() for x in gs],
            "R": [x.cpu() for x in rs], "yy": yys, "rows": rows, "part_rows": part_rows, "d": d, "P": p,
            "batches": int(batches), "batch": int(batch), "t_range": [lo, hi], "cond_drop": cond, "seed": int(seed)}


def solve_parts(eq: Dict[str, Any], ridge: float = 1e-6) -> Dict[str, Any]:
    """The joint solve of ``part_equations``, group by group: (G + lam D) theta = R with lam = ridge x the mean of G's
    diagonal over the feature rows and D the identity on those rows (the constants are not penalized). A part whose gate
    was never on has no equations and is left out of its group's solve (it keeps its weights) -> {"weights": {name:
    (weight [P, d], bias [P]) float64}, "ridge", "unsolved": [names]}. Records the ridge in ``eq["ridge"]``."""
    d, k1 = eq["d"], eq["d"] + 1
    weights, unsolved = {}, []
    for names, g_full, r_full in zip(eq["groups"], eq["G"], eq["R"]):
        live = [j for j, n in enumerate(names) if eq["part_rows"][n] > 0]
        unsolved += [n for j, n in enumerate(names) if j not in live]
        if not live:
            continue
        idx = torch.cat([torch.arange(j * k1, (j + 1) * k1) for j in live])
        g = g_full.double()[idx][:, idx].clone()
        r = r_full.double()[idx]
        feat = torch.cat([torch.arange(i * k1, i * k1 + d) for i in range(len(live))])
        lam = float(ridge) * float(torch.diagonal(g)[feat].mean())
        g[feat, feat] += lam
        theta = torch.linalg.solve(g, r)                                    # [len(live) (d + 1), P]
        for i, j in enumerate(live):
            block = theta[i * k1:(i + 1) * k1]
            weights[names[j]] = (block[:d].T.contiguous(), block[d].contiguous())
    eq["ridge"] = float(ridge)
    return {"weights": weights, "ridge": float(ridge), "unsolved": unsolved}


def _group_theta(model, names: Sequence[str]) -> Tensor:
    out = []
    for n in names:
        proj = model.unit(n).head.proj
        out.append(torch.cat([proj.weight.detach().double().cpu().T, proj.bias.detach().double().cpu()[None, :]], dim=0))
    return torch.cat(out, dim=0)


def parts_mse(eq: Dict[str, Any], model) -> Dict[str, Any]:
    """The mean squared error of the model's current parts over the equations' rows, from G, R and yy: per group and
    over every group's rows together -> {"groups": [mse per group], "all": mse}."""
    out, sse_all, n_all = [], 0.0, 0
    for names, g, r, yy, rows in zip(eq["groups"], eq["G"], eq["R"], eq["yy"], eq["rows"]):
        if rows == 0:
            out.append(None)
            continue
        theta = _group_theta(model, names)
        sse = yy - 2.0 * float((theta * r.double()).sum()) + float((theta * (g.double() @ theta)).sum())
        out.append(sse / (rows * eq["P"]))
        sse_all += sse
        n_all += rows * eq["P"]
    return {"groups": out, "all": sse_all / n_all if n_all else None}


def install_parts(model, solution: Dict[str, Any], eq: Dict[str, Any]) -> Dict[str, Any]:
    """Write the solved parts into their last layers (float32, no gradient) -> the provenance per part (as
    ``install_head``: batches, rows, t_range, ridge, the norms before and after, and its group's mean squared error
    over the equations' rows before and after, from the equations) and "all": the error over every row before and
    after."""
    before = parts_mse(eq, model)
    norms = {}
    for n in eq["names"]:
        proj = model.unit(n).head.proj
        norms[n] = (float(proj.weight.detach().double().norm()), float(proj.bias.detach().double().norm()))
    with torch.no_grad():
        for n, (w, b) in solution["weights"].items():
            proj = model.unit(n).head.proj
            proj.weight.copy_(w.to(device=proj.weight.device, dtype=proj.weight.dtype))
            proj.bias.copy_(b.to(device=proj.bias.device, dtype=proj.bias.dtype))
    after = parts_mse(eq, model)
    out: Dict[str, Any] = {}
    for gi, names in enumerate(eq["groups"]):
        for n in names:
            proj = model.unit(n).head.proj
            out[n] = {"group": gi, "group_parts": list(names), "batches": eq["batches"], "batch": eq["batch"],
                      "rows": eq["part_rows"][n], "group_rows": eq["rows"][gi], "t_range": list(eq["t_range"]),
                      "cond_drop": eq["cond_drop"], "seed": eq["seed"], "ridge": float(solution["ridge"]),
                      "solved": n in solution["weights"],
                      "weight_norm_before": norms[n][0], "weight_norm_after": float(proj.weight.detach().double().norm()),
                      "bias_norm_before": norms[n][1], "bias_norm_after": float(proj.bias.detach().double().norm()),
                      "mse_before": before["groups"][gi], "mse_after": after["groups"][gi]}
    out["all"] = {"mse_before": before["all"], "mse_after": after["all"], "parts": len(eq["names"]),
                  "groups": len(eq["groups"])}
    return out


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
