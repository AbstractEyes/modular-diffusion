"""The read: a trained model measured on held-out items with fixed noise.

(1) the held-out loss at uniform random noise levels, overall and in ten bins of t;
(2) at ten fixed noise levels (on the first 1,024 held-out items): the error by frequency band, the anchor alone, the
    error of predicting nothing, every part switched off (its cost by band), each part's output share by band, all
    members off against the sum of the single removals (independence), and where each member's attention looks
    (at t = 0.45); with classes, the error without the caption, by class and by class group. The anchor alone reads
    the held-out class; a per-class anchor is also read without it (``linear_physics_without_caption``).

A model with symbol-gated units (``units.BandMember``) reads every held-out row with one fixed symbol (``read_symbols``,
drawn from its own generator, so no other number of the read moves); the columns read without the class are read
without a symbol too.

Arithmetic (part of the result's definition): chunks of 256 items, each chunk weighted by its size over the level count,
float64 band powers, and the means over levels taken with numpy over ten floats.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .spectral import band_masks, band_power as _band_power
from .units import NULL_CLASS

T_GRID = (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
READ_BANDS = ("fine <4", "mid 4-8", "coarse 8+")
READ_SEED = 123456789
SYMBOL_SEED = 192837465              # the read's fixed symbols (models with symbol-gated units only)
CHUNK = 256
LEVEL_ITEMS = 1024
NON_MEMBER_PARTS = ("anchor", "body", "stage_body")


def band_power(model, x):
    return _band_power(x, model.patch, model.grid)


def attention_read(member, h, ctx, e, grid):
    """Where the first block of a member attends: mean Chebyshev distance, weight on itself, weight within 3 x 3."""
    blk = member.blocks[0]
    ds1, dh1 = blk.modulation(e).unsqueeze(1).chunk(6, dim=-1)[:2]
    xin = blk.norm1(h) * (1 + ds1) + dh1
    b, n, d = xin.shape
    heads, hd = blk.attn.num_heads, blk.attn.head_dim
    qkv = blk.attn.qkv(xin).view(b, n, 3, heads, hd).permute(2, 0, 3, 1, 4)
    q, k = member.code.rotary()(qkv[0], qkv[1])
    att = ((q @ k.transpose(-1, -2)) / math.sqrt(hd)).softmax(dim=-1)
    idx = torch.arange(n, device=h.device)
    rr, cc = idx // grid, idx % grid
    dist = torch.maximum((rr[:, None] - rr[None, :]).abs(), (cc[:, None] - cc[None, :]).abs()).float()
    return {"mean_distance": float((att * dist).sum(-1).mean()), "on_itself": float(att.diagonal(dim1=-2, dim2=-1).mean()),
            "in_3x3_block": float((att * (dist <= 1)).sum(-1).mean())}


def read_symbols(model, n: int, device):
    """The read's fixed symbols of the held-out rows (``model.fixed_symbols`` seeded SYMBOL_SEED), or None for a model
    without symbol-gated units (nothing is drawn then)."""
    from .model import fixed_symbols, symbol_gated
    return fixed_symbols(n, SYMBOL_SEED, device) if symbol_gated(model) else None


@torch.no_grad()
def read(model, bed, heldout: int, device) -> dict:
    model.eval()
    z1, cls = bed.heldout(heldout)
    n_held = z1.shape[0]
    gen = torch.Generator(device=device).manual_seed(READ_SEED)
    z0 = torch.randn(z1.shape, generator=gen, device=device)
    t_u = torch.rand(n_held, generator=gen, device=device)
    sym = read_symbols(model, n_held, device)

    def kw(sl):                                      # the rows' symbols, only for a model with symbol-gated units
        return {} if sym is None else {"symbols": sym[sl]}

    masks, _ = band_masks(model.grid, 3)
    masks = [m.to(device) for m in masks]
    chunk = CHUNK
    stage_names = [u.name for u in model.stage_units]
    out = {"members": [m.name for m in model.members], "stage_edges": model.edges}

    # (1) the held-out loss at uniform noise levels, in ten bins of the noise level
    per = []
    for i in range(0, n_held, chunk):
        sl = slice(i, i + chunk)
        tt = t_u[sl].view(-1, 1, 1, 1)
        v = model.velocity((1 - tt) * z0[sl] + tt * z1[sl], t_u[sl], cls[sl], **kw(sl))
        per.append(((v - (z1[sl] - z0[sl])) ** 2).mean(dim=(1, 2, 3)).double())
    per = torch.cat(per)
    out["heldout_loss"] = float(per.mean())
    bins = torch.clamp((t_u * 10).long(), max=9)
    out["loss_by_level_bin"] = [float(per[bins == b].mean()) if bool((bins == b).any()) else None for b in range(10)]

    # (2) at ten fixed noise levels
    levels = {}
    n_lv = min(n_held, LEVEL_ITEMS)
    member_parts = {m.part for m in model.members}
    class_anchor = model.anchor is not None and getattr(model.anchor, "symbols", 0) > 0
    for t in T_GRID:
        acc = {}
        att = {}
        gsum, gcnt = {}, {}
        csum, ccnt = [0.0] * 10, [0] * 10
        for i in range(0, n_lv, chunk):
            sl = slice(i, min(i + chunk, n_lv))
            b = z1[sl].shape[0]
            tt = torch.full((b,), t, device=device)
            zt = (1 - t) * z0[sl] + t * z1[sl]
            target = z1[sl] - z0[sl]
            parts, h, ctx, e = model.components(zt, tt, cls[sl], **kw(sl))
            full = torch.stack(list(parts.values())).sum(dim=0) if parts else torch.zeros_like(zt)
            w = b / n_lv

            def add(key, p):
                acc[key] = acc.get(key, 0) + p * w

            add("err", band_power(model, full - target))
            add("linear", band_power(model, model.anchor_velocity(zt, tt, cls[sl]) - target))
            if class_anchor:                                   # the per-class anchor read without the class
                add("linear_null", band_power(model, model.anchor_velocity(zt, tt) - target))
            add("zero", band_power(model, target))
            mem = [k for k in parts if k in member_parts]
            if mem:
                add("err_all_members_off", band_power(model, full - torch.stack([parts[k] for k in mem]).sum(dim=0) - target))
            for k, v in parts.items():
                add("off:" + k, band_power(model, full - v - target))
                add("out:" + k, band_power(model, v))
            if stage_names:                                    # the stage units: read by skipping them
                parts2, _, _, _ = model.components(zt, tt, cls[sl], skip=stage_names, **kw(sl))
                full2 = torch.stack(list(parts2.values())).sum(dim=0) if parts2 else torch.zeros_like(zt)
                add("off:stage_body", band_power(model, full2 - target))
                add("out:stage_body", band_power(model, full - full2))
            if i == 0 and abs(t - 0.45) < 1e-9 and h is not None:       # a staged model returns no trunk state
                for m in model.members:
                    if len(m.blocks) == 0:                     # a part with no block attends nowhere
                        continue
                    g =(m.gate(tt, cls[sl], kw(sl).get("symbols")) if getattr(m, "symbol_gate", False)
                         else m.gate(tt, cls[sl]))
                    if bool((g > 0).any()):
                        att[m.name] = attention_read(m, h, ctx, e, model.grid)
            if bed.n_classes:
                animals = (cls[sl] >= 2) & (cls[sl] <= 7)
                e_full = ((full - target) ** 2).mean(dim=(1, 2, 3)).double()
                for c in range(10):
                    cm = cls[sl] == c
                    csum[c] += float(e_full[cm].sum())
                    ccnt[c] += int(cm.sum())
                v_null = model.velocity(zt, tt, torch.full_like(cls[sl], NULL_CLASS))
                add("err_null", band_power(model, v_null - target))
                for k, v in parts.items():
                    d_off = ((full - v - target) ** 2).mean(dim=(1, 2, 3)).double() - e_full
                    for gname, gm in (("animals", animals), ("vehicles", ~animals)):
                        gsum[(k, gname)] = gsum.get((k, gname), 0.0) + float(d_off[gm].sum())
                        gcnt[(k, gname)] = gcnt.get((k, gname), 0) + int(gm.sum())

        def by_band(p):
            d = {n: float(p[m].mean()) for n, m in zip(READ_BANDS, masks)}
            d["all"] = float(p.mean())
            return d

        row = {"error": by_band(acc["err"]), "linear_physics": by_band(acc["linear"]), "predict_nothing": by_band(acc["zero"])}
        if "linear_null" in acc:
            row["linear_physics_without_caption"] = by_band(acc["linear_null"])
        if "err_null" in acc:
            row["error_without_caption"] = by_band(acc["err_null"])
            row["error_by_class"] = [csum[c] / max(1, ccnt[c]) for c in range(10)]
        keys = [k[4:] for k in acc if k.startswith("off:")]
        row["off_cost"] = {k: {n: v - row["error"][n] for n, v in by_band(acc["off:" + k]).items()} for k in keys}
        row["output_share"] = {k: {n: float(acc["out:" + k][m].sum() / acc["out:" + k].sum().clamp_min(1e-30))
                                    for n, m in zip(READ_BANDS, masks)} for k in keys}
        row["output_power"] = {k: float(acc["out:" + k].mean()) for k in keys}
        if "err_all_members_off" in acc:
            all_off = float(acc["err_all_members_off"].mean()) - row["error"]["all"]
            singles = sum(row["off_cost"][k]["all"] for k in keys if k not in NON_MEMBER_PARTS)
            row["all_members_off"] = all_off
            row["sum_of_single_member_off"] = singles
            row["independence_ratio"] = all_off / singles if abs(singles) > 1e-12 else None
        if att:
            row["attention"] = att
        if gsum:
            row["off_cost_by_group"] = {}
            for (k, gname), s_ in gsum.items():
                row["off_cost_by_group"].setdefault(k, {})[gname] = s_ / max(1, gcnt[(k, gname)])
        levels[f"{t:.2f}"] = row
    out["levels"] = levels
    tot_all = sum(r.get("all_members_off", 0.0) for r in levels.values())
    tot_single = sum(r.get("sum_of_single_member_off", 0.0) for r in levels.values())
    out["independence_ratio_over_levels"] = tot_all / tot_single if abs(tot_single) > 1e-12 else None
    out["linear_physics_loss_over_levels"] = float(np.mean([r["linear_physics"]["all"] for r in levels.values()]))
    out["predict_nothing_loss_over_levels"] = float(np.mean([r["predict_nothing"]["all"] for r in levels.values()]))
    if bed.n_classes:
        out["loss_without_caption_over_levels"] = float(np.mean([r["error_without_caption"]["all"] for r in levels.values()]))
    out["loss_over_levels"] = float(np.mean([r["error"]["all"] for r in levels.values()]))
    return out


def by_stage(res: dict, cuts) -> list:
    """The mean of ``levels[t]["error"]["all"]`` over the fixed levels with cuts[s] <= t < cuts[s + 1], per stage (None
    for a stage holding no fixed level)."""
    out = []
    for lo, hi in zip(cuts, cuts[1:]):
        vals = [r["error"]["all"] for k, r in res["levels"].items() if float(lo) <= float(k) < float(hi)]
        out.append(float(np.mean(vals)) if vals else None)
    return out
