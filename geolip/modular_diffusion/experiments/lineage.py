"""Lineage: reads between sources of one U-Net layout (a teacher and its fine-tunes, a parent and its child), with no
training.

    python -m geolip.modular_diffusion.run lineage variant=group_exchange teacher=<folder> unet=<u-net folder> kind=flow
        shift=2.5 donor=<u-net folder> probe_data=<shards> probe_rows=64 out=<folder> device=cuda dtype=fp16
    from geolip.modular_diffusion.experiments import lineage; lineage.main(variant="scale", tiny=1, out=...)

| variant        | what it does                                                                                       |
|----------------|----------------------------------------------------------------------------------------------------|
| scale          | ``pic_n`` prompts of ``pic_data`` sampled by the source's own sampler, unguided (guidance 1) and at  |
|                | ``guidance``: the standard deviation of the final latents over all elements and by channel, beside |
|                | the same of the rows' own clean latents; both ratios and the scale rule on the guided one (below)  |
| group_exchange | the source (the REFERENCE) with one named group of its U-Net replaced by the donor's tensors, read  |
|                | against the reference's own answer on every stretch's probe steps, under the reference's process;  |
|                | the controls ``none`` (nothing replaced) and ``all`` (every group replaced: the donor itself)      |
| distance       | for the same groups, the relative distance and the cosine of the two U-Nets' weights; per tensor    |
|                | too (``distance.csv``); no forward                                                                 |
| child_parent   | the source (a CHILD) against its ``parent`` on the probe: by stretch and on ``slices`` equal slices |
|                | of the coordinate s, ``per_slice`` even steps a slice                                              |
| sampler_check  | a flow source's sampler against a second implementation of the same formulas (same seed and        |
|                | inputs): equal at the first step to float32 rounding; and the sampler run twice on one seed        |

THE SOURCE is given as in the capture experiment (``teacher``, ``unet``, ``kind``, ``shift``, ``unguided``, ``kappa``;
``tiny=1``: the tiny random teacher, its U-Net drawn from ``unet_seed``). THE DONOR (``donor``; for child_parent the
``parent``) is a U-Net folder (config.json and diffusion_pytorch_model.safetensors), ``random`` (the source's U-Net
settings with fresh numbers from ``donor_seed``) or, with ``tiny=1``, ``tiny`` (the tiny U-Net drawn from
``donor_seed``). It must have the reference's tensors, name for name and shape for shape.

THE GROUPS, by module name, never by position: conv_in, time_embedding, down_blocks.0 .. down_blocks.3, mid_block,
up_blocks.0 .. up_blocks.3, conv_norm_out+conv_out. Every tensor of the U-Net is in exactly one (checked).
``split=<group>[,<group>]`` reads the named groups' resnets, attentions and the rest of each (its samplers) apart
instead. ``groups`` (empty: every group) names the single groups read; the controls none and all are always read.

THE READ of a model against a reference on a stretch: sum over the stretch's probe steps and the probe's entries of
(answer - reference answer)^2, over the same sum of (reference answer)^2 (float64), with the plain mean squared
difference beside it. The exchange is made in place on the reference's own U-Net, one group at a time, and restored
after each read (each restored tensor checked equal to its saved copy); at the end the reference is read again
against its own answers (it must read exactly 0). Control ``all`` holds whether the exchanged model's answers equal
the donor's own answers element by element (the same tensors in the same layout).

THE SCALE RULE reads the guided samples: kappa = std(the samples at ``guidance``) / std(the rows' clean latents) (the
result's ``ratio_guided``; its ``rule_side`` is "guided"). An unguided sample loses spread in every model (Stable
Diffusion 1.5's own unguided samples read 0.80 of its rows' spread, its guided ones 0.97), so the unguided ratio is
recorded beside it (``ratio_unguided``) and not used. If 0.9 <= kappa <= 1.1 the source uses the standard scale
(kappa := 1); otherwise clean latents given to it are multiplied by kappa and its samples divided by kappa before
decoding (the source argument ``kappa``).

THE DISTANCE of a group: ||W_donor - W_ref|| / ||W_ref|| and cos(W_donor, W_ref) over the group's tensors flattened
together, in float64, read tensor by tensor from the files (never a whole U-Net in memory for this).

Arguments (key=value): variant, teacher, repo, revision, tiny, seed, out, device, dtype (fp32 / fp16 / bf16: the
U-Nets), threads, det, the source's (unet, kind, shift, unguided, kappa, unet_seed), donor, donor_seed, parent, groups,
split, probe_data, probe_first, probe_rows, probe_n, probe_seed, probe_per_band, chunk, slices, per_slice, pic_data,
pic_first, pic_n, noise_seed, group, sample_steps, guidance, tolerance (sampler_check: the largest difference at the
first step, relative to the largest entry).

Output in ``out``: log.txt (its last line 'done'), result.json, result.md; distance also writes distance.csv.
"""
from __future__ import annotations

import contextlib
import math
import os
import sys
import time
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .. import __version__
from ..config import in_notebook, parse
from ..determinism import mode as det_mode
from .capture import SOURCE_DEFAULTS

EXPERIMENT = "lineage"
VARIANTS = ("scale", "group_exchange", "distance", "child_parent", "sampler_check")
GROUPS = ("conv_in", "time_embedding", "down_blocks.0", "down_blocks.1", "down_blocks.2", "down_blocks.3", "mid_block",
          "up_blocks.0", "up_blocks.1", "up_blocks.2", "up_blocks.3", "conv_norm_out+conv_out")
SPLITTABLE = tuple(g for g in GROUPS if g.startswith(("down_blocks", "up_blocks", "mid_block")))
SCALE_BAND = (0.9, 1.1)
DEFAULTS = dict(variant="scale", teacher="", repo="runwayml/stable-diffusion-v1-5", revision="451f4fe1", tiny=0, seed=0,
                out="", device="cpu", dtype="fp32", threads=4, det="", donor="", donor_seed=1, parent="", groups="",
                split="", probe_data="", probe_first=0, probe_rows=0, probe_n=4, probe_seed=1234, probe_per_band=4,
                chunk=16, slices=10, per_slice=4, pic_data="", pic_first=0, pic_n=0, noise_seed=2000, group=16,
                sample_steps=30, guidance=4.0, tolerance=1e-6)
DEFAULTS.update(SOURCE_DEFAULTS)
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


# ------------------------------------------------------------------ groups
def group_of(key: str) -> str:
    """The named group of a U-Net tensor (``GROUPS``); ValueError for a tensor outside them."""
    parts = key.split(".")
    top = parts[0]
    if top in ("down_blocks", "up_blocks") and len(parts) > 1:
        name = f"{top}.{parts[1]}"
    elif top in ("conv_norm_out", "conv_out"):
        name = "conv_norm_out+conv_out"
    else:
        name = top
    if name not in GROUPS:
        raise ValueError(f"the tensor {key} is outside the named groups {list(GROUPS)}")
    return name


def split_of(key: str) -> str:
    """The part of its group a tensor belongs to: ``<group>.resnets``, ``<group>.attentions`` or ``<group>.rest``."""
    g = group_of(key)
    rest = key[len(g) + 1:] if not g.startswith("conv_norm_out") else key
    kind = rest.split(".")[0]
    return f"{g}.{kind}" if kind in ("resnets", "attentions") else f"{g}.rest"


def group_keys(keys: Sequence[str], split: Sequence[str] = ()) -> Dict[str, List[str]]:
    """{unit: its tensor names}: the named groups in their order, or, for the groups in ``split``, their parts."""
    out: Dict[str, List[str]] = {}
    for g in GROUPS:
        out[g] = []
    for k in keys:
        out[group_of(k)].append(k)
    if not split:
        return {g: v for g, v in out.items() if v}
    sub: Dict[str, List[str]] = {}
    for g in split:
        if g not in SPLITTABLE:
            raise ValueError(f"split names groups among {list(SPLITTABLE)}, not {g!r}")
        for k in out[g]:
            sub.setdefault(split_of(k), []).append(k)
    order = [f"{g}.{p}" for g in split for p in ("resnets", "attentions", "rest")]
    return {u: sub[u] for u in order if u in sub}


def _names(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


# ------------------------------------------------------------------ the source and the donor
def _source(a: dict, vae: bool = False):
    from .capture import load_source
    return load_source(a, vae=vae, seed=int(a["seed"]))


def donor_unet(reference, where: str, a: dict):
    """The donor U-Net as a module of the reference's settings, on the reference's device and in its precision."""
    from diffusers import UNet2DConditionModel
    from ..capture.teacher import _owned, _perturb, tiny_unet
    ref_unet = reference.unet
    p = next(ref_unet.parameters())
    if where == "random":
        torch.manual_seed(int(a["donor_seed"]))
        cfg = {k: v for k, v in dict(ref_unet.config).items() if not k.startswith("_")}
        unet = UNet2DConditionModel.from_config(cfg)
    elif where == "tiny":
        if not a["tiny"]:
            raise ValueError("donor=tiny belongs to tiny=1")
        unet = tiny_unet(int(a["donor_seed"]))
        _perturb(unet, int(a["donor_seed"]))
    else:
        if not (os.path.isfile(os.path.join(where, "config.json"))
                and os.path.isfile(os.path.join(where, "diffusion_pytorch_model.safetensors"))):
            raise FileNotFoundError(f"a U-Net folder holds config.json and diffusion_pytorch_model.safetensors: {where}")
        unet = UNet2DConditionModel.from_pretrained(where, local_files_only=True, use_safetensors=True)
    unet = _owned(unet.to(device=p.device, dtype=p.dtype)).eval()
    for q in unet.parameters():
        q.requires_grad_(False)
    rs, ds = ref_unet.state_dict(), unet.state_dict()
    if set(rs) != set(ds):
        raise ValueError(f"the donor's tensors are not the reference's: {sorted(set(rs) ^ set(ds))[:5]}")
    bad = [k for k in rs if tuple(rs[k].shape) != tuple(ds[k].shape)]
    if bad:
        raise ValueError(f"the donor's tensors differ in shape from the reference's: {bad[:5]}")
    return unet


# ------------------------------------------------------------------ the read
def _answers(part, probe, steps) -> Dict[int, Tensor]:
    from ..capture.reads import _answer
    return {int(t): _answer(part, probe.z(int(t)), int(t), probe.states, probe.batch) for t in steps}


def relative_read(answers: Dict[int, Tensor], reference: Dict[int, Tensor]) -> dict:
    """{"read": sum (a - r)^2 / sum r^2, "mse": mean (a - r)^2, "equal": every answer equal element by element}."""
    num = den = 0.0
    count = 0
    equal = True
    for t, r in reference.items():
        o = answers[t]
        equal = equal and bool(torch.equal(o, r))
        d = (o.double() - r.double()).pow(2)
        num += float(d.sum())
        den += float(r.double().pow(2).sum())
        count += d.numel()
    return {"read": num / den if den > 0 else None, "mse": num / count, "equal": equal}


def _stretch_steps(probe) -> Dict[str, List[int]]:
    from ..capture import BANDS
    return {b: [int(t) for t in probe.steps[b]] for b in BANDS}


def slice_steps(slices: int, per_slice: int) -> List[List[int]]:
    """The probe steps of ``slices`` equal slices of s in [0, 1): slice k holds the whole steps t with
    k / slices <= t / 1000 < (k + 1) / slices, ``per_slice`` of them taken evenly (its first and last included)."""
    from ..capture.schedule import STEPS
    out = []
    for k in range(int(slices)):
        lo = math.ceil(STEPS * k / int(slices))
        hi = min(STEPS - 1, math.ceil(STEPS * (k + 1) / int(slices)) - 1)
        own = torch.arange(lo, hi + 1)
        n = int(per_slice)
        idx = torch.linspace(0, len(own) - 1, n, dtype=torch.float64).round().long() if n > 1 else torch.tensor([len(own) // 2])
        out.append([int(x) for x in own[idx]])
    return out


@contextlib.contextmanager
def exchanged(sd_ref: Dict[str, Tensor], sd_don: Dict[str, Tensor], keys: Sequence[str]):
    """The reference's tensors ``keys`` (of its ``state_dict()``, which shares the module's memory) replaced in place by
    the donor's for the body of the ``with``, then restored from a saved copy; the yielded dict's "ok" says, after the
    ``with``, whether every restored tensor equals its saved copy element by element."""
    saved = {k: sd_ref[k].detach().clone() for k in keys}
    out = {"ok": None}
    try:
        with torch.no_grad():
            for k in keys:
                sd_ref[k].copy_(sd_don[k])
        yield out
    finally:
        with torch.no_grad():
            for k in keys:
                sd_ref[k].copy_(saved[k])
        out["ok"] = all(torch.equal(sd_ref[k], saved[k]) for k in keys)
        del saved


# ------------------------------------------------------------------ the variants
def _group_exchange(a: dict, log) -> dict:
    from ..capture import BANDS
    from ..capture.engine import UNetPart
    from .capture import _probe, _probe_record
    reference, where = _source(a)
    probe = _probe(a, reference)
    steps = _stretch_steps(probe)
    donor_where = a["donor"]
    if not donor_where:
        raise ValueError("group_exchange needs donor= (a U-Net folder, random, or tiny with tiny=1)")
    donor = donor_unet(reference, donor_where, a)
    ref_part = UNetPart(reference.unet)
    refs = {b: _answers(ref_part, probe, steps[b]) for b in BANDS}
    donor_ans = {b: _answers(UNetPart(donor), probe, steps[b]) for b in BANDS}
    split = _names(a["split"])
    sd_ref = reference.unet.state_dict()
    sd_don = donor.state_dict()
    units = group_keys(list(sd_ref), split)
    wanted = _names(a["groups"])
    if wanted and not split:
        bad = [g for g in wanted if g not in units]
        if bad:
            raise ValueError(f"groups names groups among {list(units)}, not {bad}")
        units = {g: units[g] for g in wanted}
    plan = [("none", [])] + list(units.items()) + [("all", list(sd_ref))]
    rows: List[dict] = []
    table: Dict[str, Dict[str, Optional[float]]] = {}
    restored_ok = True
    for name, keys in plan:
        t0 = time.time()
        with exchanged(sd_ref, sd_don, keys) as restore:
            reads = {b: relative_read(_answers(ref_part, probe, steps[b]), refs[b]) for b in BANDS}
            extra = {}
            if name == "all":
                extra = {"equals_donor": all(relative_read(_answers(ref_part, probe, steps[b]), donor_ans[b])["equal"]
                                             for b in BANDS)}
        ok = restore["ok"]
        restored_ok = restored_ok and ok
        params = sum(sd_ref[k].numel() for k in keys)
        table[name] = {b: reads[b]["read"] for b in BANDS}
        row = {"unit": name, "tensors": len(keys), "params": int(params), "restored": ok, **extra}
        row.update({b: reads[b]["read"] for b in BANDS})
        row.update({f"{b}_mse": reads[b]["mse"] for b in BANDS})
        rows.append(row)
        log(f"exchange {name}: " + "  ".join(f"{b} {reads[b]['read']:.4g}" for b in BANDS)
            + f" ({len(keys)} tensors, {time.time() - t0:.1f} s)")
    final = {b: relative_read(_answers(ref_part, probe, steps[b]), refs[b]) for b in BANDS}
    donor_direct = {b: relative_read(donor_ans[b], refs[b])["read"] for b in BANDS}
    controls = {"none_reads_zero": all(table["none"][b] == 0.0 for b in BANDS),
                "all_equals_donor": bool(rows[-1].get("equals_donor")),
                "all_read_equals_donor_read": all(table["all"][b] == donor_direct[b] for b in BANDS),
                "restored_each": restored_ok,
                "restored_reads_zero": all(final[b]["read"] == 0.0 and final[b]["equal"] for b in BANDS)}
    return {"reference": where, "process": reference.record(), "donor": donor_where,
            "donor_seed": int(a["donor_seed"]) if donor_where in ("random", "tiny") else None,
            "probe": _probe_record(a, probe), "probe_steps": steps, "split": split, "table": table, "rows": rows,
            "donor_direct": donor_direct, "controls": controls,
            "columns": ["unit", "tensors", "params"] + list(BANDS) + ["restored"]}


def _tensor_pairs(a: dict) -> Iterator[Tuple[str, Tensor, Tensor]]:
    """(name, reference tensor, donor tensor) in name order, float64 on the CPU, one tensor at a time. The reference is
    ``unet`` (a folder) or the teacher folder's U-Net (with ``tiny=1``: the tiny U-Net of ``unet_seed``); the donor is
    a folder (with ``tiny=1`` also ``tiny``: the tiny U-Net of ``donor_seed``)."""
    import contextlib
    from safetensors import safe_open
    from ..capture.teacher import _perturb, tiny_unet

    def tiny_state(seed):
        u = tiny_unet(seed)
        _perturb(u, seed)
        return u.state_dict()

    def opened(stack, where):
        if isinstance(where, dict):
            return where
        path = os.path.join(where, "diffusion_pytorch_model.safetensors")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no U-Net weights (diffusion_pytorch_model.safetensors) in {where}")
        return stack.enter_context(safe_open(path, framework="pt"))

    if a["unet"]:
        ref = a["unet"]
    elif a["tiny"]:
        ref = tiny_state(int(a["unet_seed"]) if int(a["unet_seed"]) >= 0 else int(a["seed"]))
    else:
        ref = os.path.join(a["teacher"], "unet")
    if a["donor"] == "tiny":
        if not a["tiny"]:
            raise ValueError("donor=tiny belongs to tiny=1")
        don = tiny_state(int(a["donor_seed"]))
    elif a["donor"] == "random":
        raise ValueError("the distance reads two stored U-Nets (a folder, or tiny with tiny=1), not random")
    else:
        don = a["donor"]
    with contextlib.ExitStack() as stack:
        fr, fd = opened(stack, ref), opened(stack, don)
        rk, dk = set(fr.keys()), set(fd.keys())
        if rk != dk:
            raise ValueError(f"the two U-Nets' tensors differ: {sorted(rk ^ dk)[:5]}")
        for k in sorted(rk):
            r = fr[k] if isinstance(fr, dict) else fr.get_tensor(k)
            d = fd[k] if isinstance(fd, dict) else fd.get_tensor(k)
            if tuple(r.shape) != tuple(d.shape):
                raise ValueError(f"{k}: {tuple(r.shape)} against {tuple(d.shape)}")
            yield k, r.double(), d.double()


def _distance(a: dict, log) -> dict:
    if not a["donor"]:
        raise ValueError("distance needs donor= (a U-Net folder, or tiny with tiny=1)")
    acc: Dict[str, List[float]] = {g: [0.0, 0.0, 0.0, 0.0] for g in GROUPS}     # |d|^2, |r|^2, |w|^2, r.w
    per: List[dict] = []
    for k, r, d in _tensor_pairs(a):
        g = group_of(k)
        diff, rr, dd, rd = float((d - r).pow(2).sum()), float(r.pow(2).sum()), float(d.pow(2).sum()), float((r * d).sum())
        x = acc[g]
        x[0] += diff
        x[1] += rr
        x[2] += dd
        x[3] += rd
        per.append({"tensor": k, "group": g, "numel": int(r.numel()),
                    "relative": math.sqrt(diff / rr) if rr > 0 else None,
                    "cosine": rd / math.sqrt(rr * dd) if rr > 0 and dd > 0 else None})
    rows = []
    table = {}
    for g in GROUPS:
        diff, rr, dd, rd = acc[g]
        if rr == 0 and dd == 0:
            continue
        rel = math.sqrt(diff / rr) if rr > 0 else None
        cos = rd / math.sqrt(rr * dd) if rr > 0 and dd > 0 else None
        table[g] = {"relative": rel, "cosine": cos}
        rows.append({"group": g, "relative": rel, "cosine": cos,
                     "tensors": sum(1 for p in per if p["group"] == g)})
        log(f"distance {g}: relative {rel:.4g}, cosine {cos:.6f}")
    with open(os.path.join(a["out"], "distance.csv"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("tensor,group,numel,relative,cosine\n")
        for p in per:
            fh.write(f"{p['tensor']},{p['group']},{p['numel']},{p['relative']!r},{p['cosine']!r}\n")
    return {"reference": a["unet"] or (a["teacher"] + "/unet" if a["teacher"] else "tiny"), "donor": a["donor"],
            "table": table, "rows": rows, "tensors": len(per), "columns": ["group", "relative", "cosine", "tensors"]}


def _child_parent(a: dict, log) -> dict:
    from ..capture import BANDS
    from ..capture.engine import UNetPart
    from .capture import _probe, _probe_record
    if not a["parent"]:
        raise ValueError("child_parent needs parent= (a U-Net folder, or tiny with tiny=1)")
    child, where = _source(a)
    probe = _probe(a, child)
    parent = donor_unet(child, a["parent"], a)
    cp, pp = UNetPart(child.unet), UNetPart(parent)
    steps = _stretch_steps(probe)
    rows, table = [], {}
    for b in BANDS:
        r = relative_read(_answers(cp, probe, steps[b]), _answers(pp, probe, steps[b]))
        table[b] = r["read"]
        rows.append({"unit": f"stretch {b}", "lo": steps[b][0] / 1000, "hi": steps[b][-1] / 1000, "read": r["read"],
                     "mse": r["mse"]})
    slices = []
    for k, st in enumerate(slice_steps(int(a["slices"]), int(a["per_slice"]))):
        r = relative_read(_answers(cp, probe, st), _answers(pp, probe, st))
        lo, hi = k / int(a["slices"]), (k + 1) / int(a["slices"])
        slices.append({"slice": k, "lo": lo, "hi": hi, "steps": st, "read": r["read"], "mse": r["mse"]})
        rows.append({"unit": f"slice {lo:.1f} to {hi:.1f}", "lo": lo, "hi": hi, "read": r["read"], "mse": r["mse"]})
        log(f"child against parent, s in [{lo:.1f}, {hi:.1f}): {r['read']:.4g}")
    return {"child": where, "parent": a["parent"], "process": child.record(), "probe": _probe_record(a, probe),
            "probe_steps": steps, "stretches": table, "slices": slices, "rows": rows,
            "columns": ["unit", "lo", "hi", "read", "mse"]}


def _std(x: Tensor) -> dict:
    x = x.double()
    return {"all": float(x.std(unbiased=False)),
            "by_channel": [float(x[:, c].std(unbiased=False)) for c in range(x.shape[1])]}


def scale_rule(kappa: float) -> dict:
    """The scale rule (see the module's text): {"kappa": the measured ratio, "rule": "standard" or "scaled",
    "kappa_to_use": 1.0 or the ratio}."""
    inside = SCALE_BAND[0] <= kappa <= SCALE_BAND[1]
    return {"kappa": kappa, "rule": "standard" if inside else "scaled", "kappa_to_use": 1.0 if inside else kappa}


def _scale(a: dict, log) -> dict:
    from ..capture import pictures
    from .capture import _latent_shape, _picture_rows
    if int(a["pic_n"]) < 2:
        raise ValueError("scale needs pic_n >= 2 rows")
    source, where = _source(a)
    x0, ids, file_rows = _picture_rows(a, source)
    states = source.text_states(ids)
    out = {}
    for name, g in (("unguided", 1.0), ("guided", float(a["guidance"]))):
        t0 = time.time()
        z = pictures.sample_latents(source, states, source.unguided_states(1), int(a["noise_seed"]),
                                    group=int(a["group"]), steps=int(a["sample_steps"]), guidance=g,
                                    latent_shape=_latent_shape(source), alphas_cumprod=source.alphas_cumprod,
                                    process=source.process)
        out[name] = dict(_std(z), guidance=g, seconds=time.time() - t0)
        log(f"scale: {name} (guidance {g}) std {out[name]['all']:.5g}, by channel "
            + ", ".join(f"{v:.4g}" for v in out[name]["by_channel"]))
    rows_std = _std(x0)
    log(f"scale: the rows' clean latents std {rows_std['all']:.5g}")
    unguided_ratio = out["unguided"]["all"] / rows_std["all"]
    guided_ratio = out["guided"]["all"] / rows_std["all"]
    rule = scale_rule(guided_ratio)                       # the rule reads the guided samples (the module's text)
    rows = [{"latents": "samples, unguided", "std": out["unguided"]["all"], "ratio": unguided_ratio},
            {"latents": f"samples, guidance {float(a['guidance']):g}", "std": out["guided"]["all"], "ratio": guided_ratio},
            {"latents": "the rows' clean latents", "std": rows_std["all"], "ratio": 1.0}]
    return {"source": where, "process": source.record(), "n": int(x0.shape[0]), "rows_first": int(file_rows[0]),
            "samples": out, "rows_latents": rows_std, "ratio_unguided": unguided_ratio, "ratio_guided": guided_ratio,
            "guided_inside_band": SCALE_BAND[0] <= guided_ratio <= SCALE_BAND[1], **rule, "rule_side": "guided",
            "rows": rows, "columns": ["latents", "std", "ratio"], "sample_steps": int(a["sample_steps"])}


def second_flow_sampler(unet, states: Tensor, z: Tensor, steps: int, shift: float, guidance: float,
                        on_step=None) -> Tensor:
    """A second implementation of a flow source's sampler, written apart from the library's process from the same
    formulas: s_i = k u_i / (1 + (k - 1) u_i), u from 1 to 0 in ``steps`` equal parts (float32), the U-Net fed 1000 s,
    zero states as the unguided side, v = v_none + g (v_prompt - v_none), z <- z + (s_next - s) v. ``on_step(i, z)``
    is called after each step."""
    own = next(unet.parameters()).dtype
    u = torch.linspace(1.0, 0.0, int(steps) + 1)
    sig = (float(shift) * u) / (1 + (float(shift) - 1) * u)
    b = z.shape[0]
    x = z.clone().float()
    cond = states.to(own)
    none = torch.zeros_like(cond)
    for i in range(int(steps)):
        s, s_next = float(sig[i]), float(sig[i + 1])
        t = torch.full((2 * b,), s * 1000.0, device=x.device)
        with torch.no_grad():
            v = unet(torch.cat([x, x]).to(own), t, encoder_hidden_states=torch.cat([none, cond])).sample.to(x.dtype)
        v_none, v_prompt = v[:b], v[b:]
        v = v_none + float(guidance) * (v_prompt - v_none)
        x = x + (s_next - s) * v
        if on_step is not None:
            on_step(i, x)
    return x


def _sampler_check(a: dict, log) -> dict:
    from ..capture.assemble import initial_noise, sample
    from .capture import _latent_shape, _picture_rows
    source, where = _source(a)
    if source.kind != "flow" or source.unguided != "zeros":
        raise ValueError("sampler_check reads a flow source with zero states as its unguided side")
    _, ids, _ = _picture_rows(a, source)
    states = source.text_states(ids)
    shape = _latent_shape(source)
    z0 = initial_noise(int(a["noise_seed"]), states.shape[0], shape).to(source.device)
    lib_steps: List[Tensor] = []
    seen = []

    def watch(t, z, eps):
        seen.append(z.detach().clone())

    g = float(a["guidance"])
    n = int(a["sample_steps"])
    zl = sample(source, states, source.unguided_states(1), 0, n, g, latent_shape=shape, z=z0, process=source.process,
                on_step=watch)
    lib_steps = seen[1:] + [zl]                                   # the latents after each step
    second: List[Tensor] = []
    zs = second_flow_sampler(source.unet, states, z0, n, source.process.shift, g, on_step=lambda i, x: second.append(x.clone()))
    again = sample(source, states, source.unguided_states(1), 0, n, g, latent_shape=shape, z=z0, process=source.process)
    first_diff = float((lib_steps[0].double() - second[0].double()).abs().max())
    first_scale = float(lib_steps[0].double().abs().max())
    final_diff = float((zl.double() - zs.double()).abs().max())
    tol = float(a["tolerance"])
    out = {"source": where, "process": source.record(), "rows_sampled": int(states.shape[0]), "steps": n,
           "guidance": g, "first_step_max_abs_difference": first_diff, "first_step_max_abs": first_scale,
           "first_step_relative": first_diff / first_scale if first_scale > 0 else None,
           "first_step_equal_to_rounding": first_diff <= tol * max(first_scale, 1e-30),
           "final_max_abs_difference": final_diff, "tolerance": tol,
           "repeat_equal": bool(torch.equal(zl, again))}
    out["rows"] = [{"quantity": k, "value": out[k]} for k in
                   ("first_step_max_abs_difference", "first_step_relative", "first_step_equal_to_rounding",
                    "final_max_abs_difference", "repeat_equal")]
    out["columns"] = ["quantity", "value"]
    log(f"sampler check: first step {first_diff:.3g} of {first_scale:.3g} (equal to rounding: "
        f"{out['first_step_equal_to_rounding']}); final {final_diff:.3g}; repeats itself: {out['repeat_equal']}")
    return out


def stretch_snr(kind: str, s0: Optional[float] = None, shift: Optional[float] = None) -> Dict[str, dict]:
    """{stretch: {"lo", "hi" (the first and last s of its step set), "snr_at_lo", "snr_at_hi"}}: the signal-to-noise
    each stretch spans in a source of ``kind`` (a flow source needs ``s0``, its clean latents' standard deviation;
    at s = 0 it is infinite)."""
    from ..capture import BANDS
    from ..capture.process import make_process
    from ..capture.schedule import band_steps
    p = make_process(kind, shift if shift is not None else (1.0 if kind == "flow" else None))
    out = {}
    for b in BANDS:
        st = band_steps(b)
        lo, hi = int(st[0]), int(st[-1])
        def at(t):
            if kind == "flow" and t == 0:
                return math.inf
            return float(p.snr(t, s0)[0])
        out[b] = {"lo": lo / 1000, "hi": hi / 1000, "snr_at_lo": at(lo), "snr_at_hi": at(hi)}
    return out


# ------------------------------------------------------------------ the job
def main(argv=(), **overrides) -> dict:
    a = parse(DEFAULTS, argv, **overrides)
    if not a["out"]:
        raise ValueError("out= is required")
    if a["variant"] not in VARIANTS:
        raise ValueError(f"unknown variant {a['variant']!r}; known: {list(VARIANTS)}")
    if a["dtype"] not in DTYPES:
        raise ValueError(f"dtype is one of {sorted(DTYPES)}, not {a['dtype']!r}")
    from .capture import source_options
    source_options(a)
    with det_mode(a["det"]) as det:
        return _run(a, det)


def _run(a: dict, det) -> dict:
    from ..capture.reads import table, write_json
    out = a["out"]
    os.makedirs(out, exist_ok=True)
    log_path = os.path.join(out, "log.txt")

    def log(line: str) -> None:
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")

    torch.set_num_threads(int(a["threads"]))
    t0 = time.time()
    log(f"LINEAGE {a['variant']}: torch {torch.__version__}; package {__version__}")
    run = {"scale": _scale, "group_exchange": _group_exchange, "distance": _distance, "child_parent": _child_parent,
           "sampler_check": _sampler_check}[a["variant"]]
    body = run(a, log)
    result = {"experiment": EXPERIMENT, "variant": a["variant"],
              "args": {k: a[k] for k in sorted(a) if not (k == "det" and not a[k])}, "torch": torch.__version__,
              "version": __version__}
    result.update(body)
    result["seconds"] = time.time() - t0
    det.record(result)
    write_json(result, os.path.join(out, "result.json"))
    text = f"# lineage: {a['variant']}\n\n" + table(result["rows"], result["columns"])
    with open(os.path.join(out, "result.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    log(text.rstrip())
    log("done")
    return result


def summarize(root: str) -> str:
    """Every result of this experiment under ``root``, one table a job."""
    import json
    from ..capture.reads import table
    parts = []
    for dirpath, _, files in sorted(os.walk(root)):
        if "result.json" in files:
            with open(os.path.join(dirpath, "result.json"), encoding="utf-8") as fh:
                r = json.load(fh)
            if r.get("experiment") == EXPERIMENT:
                parts.append(f"## {os.path.relpath(dirpath, root)} ({r['variant']})\n\n" + table(r["rows"], r["columns"]))
    return "\n".join(parts)


if __name__ == "__main__" and not in_notebook():
    main(sys.argv[1:])
