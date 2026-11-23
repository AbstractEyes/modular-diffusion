"""Training: the recipe, the loop and the entry point.

    python -m geolip.modular_diffusion.train bed=pictures data=<npz> bands=3 keep=1 codes=rung anchor=1 anchor_in=1 \\
        seeds=0 steps=6000 device=cuda out=<folder>
    from geolip.modular_diffusion import train; train.main(bed="fields", steps=50, device="cpu", out="smoke")

The recipe. Adam (betas 0.9 / 0.999, eps 1e-8, no weight decay), the learning rate warmed up linearly over ``warmup``
steps and set before every step; one generator per step, seeded 1,000,003 (seed + 1) + step on the run device, drawing
the batch (picture indices, flips), the noise, t, the class drop (probability ``cond_drop``, pictures only) and member
dropout, in that order; after the backward pass, member gradient equalization (``equalize=1``): every member's gradient
is scaled to the upper median of the members' gradient norms.

``run_loop`` is the loop of that recipe; ``fit_units`` trains named units of a model built any way (a Model or a Staged)
with it, over a range of the noise level, and reads the result.

Outputs per seed, in ``out``: <tag>.json (the read, the curve and the arguments), <tag>.pt (the flat state dict),
<tag>.build.json (the build), <tag>_samples.png (pictures, when ``sheet``) and a log. The last log line is 'done'.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

from . import compat
from .beds import make_bed
from .build import write_build
from .config import in_notebook, parse
from .model import Model
from .read import read
from .sample import sample_sheet

RUN_PREFIX = "md_"

DEFAULTS = dict(bed="pictures", data="", arch="collective", bands=3, keep=1, codes="rung",
                stages="1", soft=0.0, anchor=0, body_out=1, body_depth=1, member_depth=1, body_code="whole", plain_depth=4,
                extra="", init="", train="all", equalize=1, dim=128, heads=2, seeds="0", steps=10000, batch=128, lr=1e-3,
                warmup=100, cond_drop=0.1, device="cuda", out="md_out", heldout=2048,
                log_every=500, smoke=0, mem_fraction=0.25, threads=4, name="", sheet=1, grid=16, channels=4, slope=1.225,
                dc_share=0.060, content="", base=1, mdrop=0.0, stage_body="", read_at="", xdeep=0, anchor_in=0, tilt=0,
                stage_zero=0, stage_soft=0.0, t_sampling="uniform", t_lo=0.0, t_hi=1.0, t_mix=0.0)
SMOKE = dict(steps=4, batch=8, heldout=32, log_every=2, sheet=0)
RANGE_KEYS = ("t_lo", "t_hi", "t_mix")    # recorded in a result's arguments only when they differ from their defaults


def step_generator(seed: int, step: int, device) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(1_000_003 * (seed + 1) + step)


def equalize(model) -> None:
    """Scale every member's gradient to the upper median of the members' gradient norms (members with no gradient, or
    a norm under 1e-20, are left alone; nothing happens with fewer than two live members)."""
    norms = []
    for m in model.members:
        sq = [p.grad.pow(2).sum() for p in m.parameters() if p.grad is not None]
        norms.append(float(torch.stack(sq).sum().sqrt()) if sq else 0.0)
    live = sorted(n for n in norms if n > 1e-20)
    if len(live) < 2:
        return
    median = live[len(live) // 2]
    for m, n in zip(model.members, norms):
        if n > 1e-20:
            for p in m.parameters():
                if p.grad is not None:
                    p.grad.mul_(median / n)


def set_trainable(model, spec: str, loaded: Optional[dict] = None) -> None:
    """spec: all | members | body | new | a comma list of member names (and ``body``, ``stage``).
    new = the members that no loaded file held, with the stage units when no file held them; without loaded files,
    the members whose name starts with x."""
    if spec == "all":
        return
    names = set(spec.split(","))
    for p in model.parameters():
        p.requires_grad_(False)
    new = set(loaded["members_new"]) if loaded else {m.name for m in model.members if m.name.startswith("x")}
    for m in model.members:
        on = spec == "members" or (spec == "new" and m.name in new) or m.name in names
        for p in m.parameters():
            p.requires_grad_(on)
    if spec == "body" or "body" in names:
        for mod in (model.stem, model.head):
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad_(True)
    if model.stage_units and ("stage" in names or (spec == "new" and loaded is not None and loaded["stage_new"])):
        for u in model.stage_units:
            for p in u.parameters():
                p.requires_grad_(True)


BETAS = (0.9, 0.999)
EPS = 1e-8
STATE_WEIGHTS = "adam.safetensors"
STATE_FILE = "state.json"


def _scalar_dtype() -> torch.dtype:
    """The dtype of Adam's step count on the CPU (as the optimizer makes it)."""
    return torch.float64 if torch.get_default_dtype() == torch.float64 else torch.float32


def _trainable_named(model) -> List[Tuple[str, torch.nn.Parameter]]:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def adam_state(opt, model, step: int, lr: float) -> Dict[str, Any]:
    """The optimizer's state by parameter name -> {"step": the next global step, "m", "v": {name: tensor}, "t": {name:
    the parameter's own step count}, "lr", "betas", "eps"} (copies, on the parameters' device). A parameter that never
    took a step has no entry."""
    out: Dict[str, Any] = {"step": int(step), "m": {}, "v": {}, "t": {}, "lr": float(lr), "betas": list(BETAS), "eps": EPS}
    for n, p in _trainable_named(model):
        st = opt.state.get(p)
        if not st:
            continue
        out["m"][n] = st["exp_avg"].detach().clone()
        out["v"][n] = st["exp_avg_sq"].detach().clone()
        out["t"][n] = int(float(st["step"]))
    return out


def load_adam_state(opt, model, state: Dict[str, Any], partial: bool = False) -> List[str]:
    """Load a state of ``adam_state`` into a newly made Adam over the model's trainable parameters, by parameter name.
    Every name of the state must be a trainable parameter and every trainable parameter must be in the state (a
    KeyError names the difference). With ``partial`` a trainable parameter that has no entry in the state is left
    without optimizer state (Adam makes its moments at that parameter's first step); a name in the state that is not a
    trainable parameter still raises. Returns the sorted names of the trainable parameters left without state."""
    named = dict(_trainable_named(model))
    have = set(state["m"])
    only_state, only_trainable = have - set(named), set(named) - have
    if only_state or (only_trainable and not partial):
        raise KeyError(f"the optimizer state and the trainable parameters differ: only in the state "
                       f"{sorted(only_state)[:5]}, only trainable {sorted(only_trainable)[:5]}")
    for n, p in named.items():
        if n not in have:
            continue
        opt.state[p] = {"step": torch.tensor(float(state["t"][n]), dtype=_scalar_dtype()),
                        "exp_avg": state["m"][n].detach().to(device=p.device, dtype=p.dtype).clone(),
                        "exp_avg_sq": state["v"][n].detach().to(device=p.device, dtype=p.dtype).clone()}
    return sorted(only_trainable)


def write_state(folder: str, state: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> None:
    """Write a trainer state (``fit_units(..., keep_state=True)["optimizer_state"]``) into ``folder``: adam.safetensors
    (keys m/<parameter name> and v/<parameter name>) and state.json (the next global step, the per-parameter step
    counts, lr, betas, eps and ``extra``). Each file is written under a process-own name and moved into place."""
    from safetensors.torch import save_file
    os.makedirs(folder, exist_ok=True)
    tensors = {}
    for n in sorted(state["m"]):
        tensors["m/" + n] = state["m"][n].detach().to("cpu").contiguous()
        tensors["v/" + n] = state["v"][n].detach().to("cpu").contiguous()
    meta = {"format": "modular-diffusion/trainer-state/1", "step": int(state["step"]),
            "t": {n: int(state["t"][n]) for n in sorted(state["t"])}, "lr": state.get("lr"),
            "betas": list(state.get("betas", BETAS)), "eps": state.get("eps", EPS), "extra": dict(extra or {})}
    wpath = os.path.join(folder, STATE_WEIGHTS)
    jpath = os.path.join(folder, STATE_FILE)
    tmp_w, tmp_j = f"{wpath}.{os.getpid()}.tmp", f"{jpath}.{os.getpid()}.tmp"
    save_file(tensors, tmp_w)
    with open(tmp_j, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, indent=1)
        fh.write("\n")
    os.replace(tmp_w, wpath)
    os.replace(tmp_j, jpath)


def read_state(folder: str, device="cpu") -> Dict[str, Any]:
    """A trainer state written by ``write_state`` -> {"step", "m", "v", "t", "lr", "betas", "eps", "extra"}."""
    from safetensors.torch import load_file
    with open(os.path.join(folder, STATE_FILE), encoding="utf-8") as fh:
        meta = json.load(fh)
    tensors = load_file(os.path.join(folder, STATE_WEIGHTS), device=str(device))
    m = {k[2:]: v for k, v in tensors.items() if k.startswith("m/")}
    v = {k[2:]: t for k, t in tensors.items() if k.startswith("v/")}
    return {"step": int(meta["step"]), "m": m, "v": v, "t": {n: int(x) for n, x in meta["t"].items()},
            "lr": meta.get("lr"), "betas": meta.get("betas"), "eps": meta.get("eps"), "extra": meta.get("extra", {})}


def run_loop(model, bed, seed: int, steps: int, lr: float, warmup: int, batch: int, cond_drop: float, mdrop: float,
             t_sampling: str, t_range: Sequence[float], equalize_members, device, log: Callable[[str], None], log_every: int,
             at_step: Optional[Callable[[int], None]] = None, t_mix: float = 0.0, start_step: int = 0,
             optimizer_state: Optional[Dict[str, Any]] = None,
             state_out: Optional[Dict[str, Any]] = None, optimizer_partial: bool = False) -> Tuple[List[list], float]:
    """The training loop of the recipe over the model's trainable parameters -> (curve [[step, loss], ...], seconds).

    t_range and t_mix: see ``Model.draw``. A step whose loss reaches no trained unit is skipped. ``at_step(steps done)`` is called after each step that trained.
    Nothing happens when no parameter trains or steps <= 0.

    A run that continues another: the loop runs the global steps start_step .. start_step + steps - 1; global step s
    takes the learning rate lr min(1, (s + 1) / max(1, warmup)) and the stream ``step_generator(seed, s)``. With
    ``optimizer_state`` (``adam_state``) the Adam over the trainable parameters is loaded from it by parameter name
    before the first step. The curve's step numbers and ``at_step`` stay local (0 .. steps - 1). With ``state_out`` (a
    dict) the optimizer's state at the end is put into it. start_step=0 and no state is the loop of earlier versions.
    ``optimizer_partial``: the state may lack trainable parameters (``load_adam_state(partial=True)``); those start
    without moments at the continued run's learning rate (no new warm-up)."""
    params = [p for p in model.parameters() if p.requires_grad]
    curve: List[list] = []
    t0 = time.time()
    start_step = int(start_step)
    if not params or steps <= 0:
        return curve, time.time() - t0
    opt = torch.optim.Adam(params, lr=lr, betas=BETAS, eps=EPS, weight_decay=0.0)
    if optimizer_state is not None:
        if optimizer_partial:
            load_adam_state(opt, model, optimizer_state, partial=True)
        else:
            load_adam_state(opt, model, optimizer_state)
    model.train()
    for step in range(steps):
        s = start_step + step
        for group in opt.param_groups:
            group["lr"] = lr * min(1.0, (s + 1) / max(1, warmup))
        gen = step_generator(seed, s, device)
        x, cls = bed.train_batch(batch, gen)
        loss = model.loss(x, cls, gen, cond_drop, mdrop=mdrop, t_sampling=t_sampling, t_range=t_range, t_mix=t_mix)
        if not loss.requires_grad:               # no sample of this batch reached a trained unit
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if equalize_members:
            equalize(model)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            spent = time.time() - t0
            curve.append([step, float(loss.detach())])
            log(f"  step {step + 1}/{steps}  loss {float(loss.detach()):.4f}  spent {spent:.0f} s  "
                f"left about {spent / (step + 1) * (steps - step - 1):.0f} s")
        if at_step is not None:
            at_step(step + 1)
    if state_out is not None:
        state_out.update(adam_state(opt, model, start_step + steps, lr))
    return curve, time.time() - t0


def _all_units(model) -> list:
    """Every unit of a model, or of every cluster of a staged model (a shared unit once)."""
    clusters = getattr(model, "clusters", None)
    models = list(clusters) if clusters is not None else [model]
    seen, out = set(), []
    for m in models:
        for u in m.units:
            if id(u) not in seen:
                seen.add(id(u))
                out.append(u)
    return out


def set_trainable_units(model, names: Union[str, Iterable[str]]) -> None:
    """``"all"``: every parameter trains. An iterable of unit names: the parameters of those units train, every other
    parameter is frozen (for a staged model the names apply in every cluster). An unknown name raises KeyError."""
    units = _all_units(model)
    if isinstance(names, str):
        if names != "all":
            raise ValueError(f"trainable must be 'all' or a list of unit names, not {names!r}")
        for p in model.parameters():
            p.requires_grad_(True)
        return
    names = set(names)
    unknown = names - {u.name for u in units}
    if unknown:
        raise KeyError(f"no unit named {sorted(unknown)} in the model")
    for p in model.parameters():
        p.requires_grad_(False)
    for u in units:
        if u.name in names:
            for p in u.parameters():
                p.requires_grad_(True)


def set_trainable_params(model, params: Dict[str, Sequence[str]]) -> None:
    """Set more parameters trainable, by name prefix inside a unit: {unit name: [prefix, ...]} (for example
    {"stem": ["blocks.3."]}); every parameter of a named unit (in every cluster of a staged model) whose name relative
    to the unit starts with one of its prefixes trains, the others keep their setting. An unknown unit, or a prefix that
    matches no parameter, raises KeyError."""
    units = _all_units(model)
    known = {u.name for u in units}
    for name, prefixes in params.items():
        if name not in known:
            raise KeyError(f"no unit named {name!r} in the model")
        prefixes = [str(p) for p in prefixes]
        hit = {p: False for p in prefixes}
        for u in units:
            if u.name != name:
                continue
            for pn, p in u.named_parameters():
                for pre in prefixes:
                    if pn.startswith(pre):
                        p.requires_grad_(True)
                        hit[pre] = True
        missing = [p for p, ok in hit.items() if not ok]
        if missing:
            raise KeyError(f"the unit {name!r} holds no parameter whose name starts with {missing}")


def _row_parameters(model, spec: Dict[str, Dict[str, Sequence[int]]]) -> List[Dict[str, Any]]:
    """The parameters of ``fit_units(trainable_rows=)``: one entry per distinct parameter object -> {"unit", "pname",
    "name" (its name in the model), "param", "rows", "listed" (a bool mask over the first axis), "mask" (the 0/1
    gradient mask), "before" (a copy of its values)}. An unknown unit or parameter, or a row out of range, raises."""
    units = _all_units(model)
    full = {id(p): n for n, p in model.named_parameters()}
    out, seen = [], set()
    for uname, params in spec.items():
        hits = [u for u in units if u.name == uname]
        if not hits:
            raise KeyError(f"no unit named {uname!r} in the model")
        for pname, ix in params.items():
            ix = sorted({int(i) for i in ix})
            for u in hits:
                named = dict(u.named_parameters())
                if pname not in named:
                    raise KeyError(f"the unit {uname!r} holds no parameter named {pname!r}")
                p = named[pname]
                if id(p) in seen:
                    continue
                seen.add(id(p))
                if not ix or ix[0] < 0 or ix[-1] >= p.shape[0]:
                    raise ValueError(f"rows {ix} do not fit the first axis of {uname}.{pname} ({p.shape[0]})")
                listed = torch.zeros(p.shape[0], dtype=torch.bool, device=p.device)
                listed[ix] = True
                mask = listed.to(p.dtype).view((-1,) + (1,) * (p.dim() - 1))
                out.append({"unit": uname, "pname": pname, "name": full[id(p)], "param": p, "rows": ix,
                            "listed": listed, "mask": mask, "before": p.detach().clone()})
    return out


def _row_mask_hook(mask: torch.Tensor):
    def hook(grad):
        return grad * mask.to(device=grad.device, dtype=grad.dtype)
    return hook


def recipe(lr: float, warmup: int, batch: int, steps: int, t_sampling: str, t_range: Sequence[float], cond_drop: float,
           flip: bool, equalize_members: bool, mdrop: float, device, t_mix: float = 0.0) -> Dict[str, Any]:
    """The card recipe of a run of ``run_loop``."""
    return {"optimizer": "Adam", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0, "lr": lr, "warmup": warmup,
            "batch": batch, "steps": steps, "t_sampling": t_sampling, "t_range": [float(x) for x in t_range],
            "t_mix": float(t_mix),
            "cond_drop": cond_drop, "flip": bool(flip), "equalize": bool(equalize_members), "mdrop": mdrop,
            "step_seed": "1000003 * (seed + 1) + step", "torch": torch.__version__, "device": str(device)}


def fit_units(model, bed, seed: int, *, steps: int, trainable: Union[str, Iterable[str]] = "all",
              t_range: Sequence[float] = (0.0, 1.0), t_sampling: str = "uniform", t_mix: float = 0.0,
              lr: float = 1e-3, warmup: int = 100,
              batch: int = 128, cond_drop: float = 0.1, mdrop: float = 0.0, equalize: bool = True, heldout: int = 2048,
              device="cpu", log: Callable[[str], None] = print, log_every: int = 500,
              expect_anchor_at_birth: bool = False, averages: Sequence[Union[float, str]] = (),
              start_step: int = 0, optimizer_state: Optional[Dict[str, Any]] = None, keep_state: bool = False,
              average_start: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
              average_step: Optional[int] = None,
              trainable_params: Optional[Dict[str, Sequence[str]]] = None, optimizer_partial: bool = False,
              trainable_rows: Optional[Dict[str, Dict[str, Sequence[int]]]] = None) -> dict:
    """Train the named units of a model (a Model or a Staged) with the recipe and read it.

    averages: decays d and named averages. For each, a float32 copy of every trainable parameter (on its device, no
    gradient). A decay d (a float): the copy starts as the parameters before the first step and, after optimizer step
    k = 0, 1, ..., becomes d_k avg + (1 - d_k) w with d_k = min(d, (1 + k) / (10 + k)). A name "g<number>" ("g8"):
    d_k = (1 + k) / (k + g + 2) with the same arithmetic ("g8" gives the bits of the decay 0.999 for every k < 8,990).
    A name "tail<number>" ("tail100"): the plain mean of the trainable parameters after each of the last <number>
    optimizer steps of this fit (after all of them when the fit has fewer), summed in float64, divided, cast to
    float32. Nothing else changes (the stream, the curve and the last weights are those of a run without); the result
    then also holds "averaged": {the entry as given (a float, or the name): {parameter name: tensor}} (keep it out of
    json files).

    A carried average: ``average_start`` {"g<number>": {parameter name: tensor}} starts that average from the given
    values (every trainable parameter must be given) instead of the parameters before the first step, and its k counts
    from ``average_step`` (default: ``start_step``): k = average_step, average_step + 1, ... Only g names carry.

    A run that continues another: ``start_step`` and ``optimizer_state`` (see ``run_loop``). ``keep_state``: the result
    also holds "optimizer_state" {"step": start_step + steps, "m", "v": {name: tensor}, "t": {name: int}, "lr",
    "betas", "eps"} (tensors: keep it out of json files; ``write_state`` writes it).

    Part of a unit: ``trainable_params`` {unit name: [name prefixes inside the unit]} sets those parameters trainable
    as well (``set_trainable_params``; for example {"stem": ["blocks.3."]} trains the stem's last block of four besides
    the named units); every other parameter stays frozen and is checked bit for bit. The result then also holds
    "trainable_params" (absent otherwise).

    A continued state that covers only some trainable parameters: ``optimizer_partial`` (``run_loop``); when the state
    lacked any, the result records recipe "optimizer_state": "continued, partial" and "optimizer_new" (the sorted names
    of the trainable parameters the state did not hold; with a state that holds them all nothing is recorded). A carried average that lacks a trainable parameter raises a KeyError, except
    with ``optimizer_partial``: then that parameter's average starts from its value before the first step and its k
    counts from that first step (k = 0, 1, ..., the arithmetic of an average that is not carried); the result records
    "average_new" {average: [those names]}.

    Rows of a table as the only trainable part of a parameter: ``trainable_rows`` {unit name: {parameter name inside
    the unit: [row indices]}} (for example {"stem": {"cls.weight": [9]}}), applied after ``trainable`` and
    ``trainable_params``: each named parameter (in every cluster of a staged model) trains, its gradient multiplied by
    a 0/1 mask over its first axis (1 on the listed rows) by a hook removed when the fit ends. Refused (ValueError)
    with an ``optimizer_state`` that holds an entry for such a parameter. After the loop the unlisted rows are put
    back from a copy taken before it and checked bit for bit, in the weights and in every average the result carries.
    The trained numbers count the listed rows only; the result records "trainable_rows" (as given),
    "rows_frozen_equal" (True) and "rows_restored" (whether any unlisted value had moved before it was put back).

    Sets the trainable units; with ``expect_anchor_at_birth`` checks that the velocity equals the anchor's (or zero
    without an anchor) exactly on a 2-row probe; runs the loop; checks that the frozen parameters did not change; reads.
    cond_drop applies only when the bed has classes. Returns the read with {seed, steps, t_range, t_sampling, trainable,
    trained_numbers, all_numbers, train_seconds, peak_memory_mib, curve, frozen_unchanged, recipe, torch, device}."""
    model.to(device)
    set_trainable_units(model, trainable)
    if trainable_params:
        set_trainable_params(model, trainable_params)
    rows = _row_parameters(model, trainable_rows) if trainable_rows else []
    if rows and optimizer_state is not None:
        held = [r["name"] for r in rows if r["name"] in optimizer_state["m"]]
        if held:
            raise ValueError(f"trainable_rows names parameters the optimizer state holds: {held}")
    for r in rows:
        r["param"].requires_grad_(True)
    t_range = (float(t_range[0]), float(t_range[1]))
    frozen_before = {id(p): p.detach().clone() for p in model.parameters() if not p.requires_grad}
    if expect_anchor_at_birth:
        gen = torch.Generator().manual_seed(seed)
        z = torch.randn((2,) + tuple(bed.field_shape), generator=gen).to(device)
        t2 = torch.tensor([0.3, 0.7], device=device)
        with torch.no_grad():
            c2 = torch.zeros(2, dtype=torch.long, device=device)
            v = model.velocity(z, t2, c2)
            use_anchor = model.anchor is not None and model.anchor.output
            gap = float((v - (model.anchor_velocity(z, t2, c2) if use_anchor else 0)).abs().max())
        if gap != 0.0:
            raise AssertionError(f"the newborn model is not exactly its anchor (or zero): {gap}")
    n_par = model.trained_numbers()
    for r in rows:                                    # a row-trained parameter counts its listed rows only
        n_par += len(r["rows"]) * (r["param"].numel() // max(1, r["param"].shape[0])) - r["param"].numel()
    n_all = sum(p.numel() for p in model.parameters())
    new_names: Optional[List[str]] = None
    if optimizer_partial and optimizer_state is not None:
        new_names = sorted(n for n, _ in _trainable_named(model) if n not in optimizer_state["m"])
    cond = cond_drop if bed.n_classes else 0.0
    is_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
    log(f"FIT {n_par:,} trained numbers of {n_all:,}; trainable {trainable if isinstance(trainable, str) else sorted(trainable)}; "
        f"{steps} steps; t in [{t_range[0]:g}, {t_range[1]:g}) ({t_sampling}" + (f", whole-range share {t_mix:g})" if t_mix else ")"))
    specs = [average_spec(x) for x in averages]
    if len({key for key, _, _ in specs}) != len(specs):
        raise ValueError(f"an average is listed twice: {list(averages)}")
    carried = dict(average_start or {})
    for name in carried:
        if not any(key == name and kind == "g" for key, kind, _ in specs):
            raise ValueError(f"only a listed g average can be carried, not {name!r}")
    first_k = int(start_step if average_step is None else average_step)
    at_step = None
    averaged = None
    tails: Dict[str, list] = {}
    fresh: Dict[str, set] = {}                         # carried averages: the parameters that start here
    if specs:
        live = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        averaged = {}
        with torch.no_grad():
            for key, kind, value in specs:
                if kind == "tail":
                    tails[key] = [None, 0]                     # [float64 sums, count]
                    continue
                if key in carried:
                    given = carried[key]
                    missing = [n for n, _ in live if n not in given]
                    if missing and not optimizer_partial:
                        raise KeyError(f"the carried average {key!r} lacks the trainable parameters {missing[:5]}")
                    if missing:
                        fresh[key] = set(missing)
                    averaged[key] = {n: (given[n].detach().to(device=p.device, dtype=torch.float32).clone()
                                         if n in given else p.detach().clone().float())
                                     for n, p in live}
                else:
                    averaged[key] = {n: p.detach().clone().float() for n, p in live}
        updates = [0]
        n_steps = int(steps)

        def at_step(done: int) -> None:                   # after every optimizer step k = 0, 1, ...
            k = updates[0]
            with torch.no_grad():
                for key, kind, value in specs:
                    if kind == "tail":
                        if done > n_steps - value:            # one of the last <value> steps of this fit
                            acc = tails[key]
                            if acc[0] is None:
                                acc[0] = {n: p.detach().double().clone() for n, p in live}
                            else:
                                for n, p in live:
                                    acc[0][n] += p.detach().double()
                            acc[1] += 1
                        continue
                    state = averaged[key]
                    if kind == "decay":
                        dk = min(value, (1 + k) / (10 + k))
                    else:
                        kk = first_k + k if key in carried else k
                        dk = (1 + kk) / (kk + value + 2)
                    new_here = fresh.get(key)
                    if new_here:                              # parameters whose carried average starts at this fit
                        dk_new = (1 + k) / (k + value + 2)
                        for n, p in live:
                            d = dk_new if n in new_here else dk
                            state[n].copy_(d * state[n] + (1.0 - d) * p.detach().float())
                        continue
                    for n, p in live:
                        state[n].copy_(dk * state[n] + (1.0 - dk) * p.detach().float())    # the rule as written
            updates[0] += 1

    state_out: Optional[Dict[str, Any]] = {} if keep_state else None
    hooks = [r["param"].register_hook(_row_mask_hook(r["mask"])) for r in rows]
    try:
        curve, seconds = run_loop(model, bed, seed, steps, lr, warmup, batch, cond, mdrop, t_sampling, t_range,
                                  equalize, device, log, log_every, at_step=at_step, t_mix=float(t_mix),
                                  start_step=int(start_step), optimizer_state=optimizer_state, state_out=state_out,
                                  optimizer_partial=optimizer_partial)
    finally:
        for h in hooks:
            h.remove()
    if tails:
        with torch.no_grad():
            for key, (sums, count) in tails.items():
                if sums is None:                               # no step trained: the parameters as they are
                    averaged[key] = {n: p.detach().clone().float() for n, p in live}
                else:
                    averaged[key] = {n: (s / count).float() for n, s in sums.items()}
        averaged = {key: averaged[key] for key, _, _ in specs}          # in the order given
    rows_restored = False
    for r in rows:                                   # the rows nobody trained: exactly their values before the fit
        p, keep, before = r["param"], ~r["listed"], r["before"]
        with torch.no_grad():
            rows_restored = rows_restored or not torch.equal(p.detach()[keep], before[keep])
            p[keep] = before[keep]
            for state in (averaged or {}).values():
                if r["name"] in state:
                    state[r["name"]][keep] = before[keep].to(state[r["name"]].dtype)
        if not torch.equal(p.detach()[keep], before[keep]):
            raise AssertionError(f"the unlisted rows of {r['name']} differ from their values before the fit")
        for state in (averaged or {}).values():
            if r["name"] in state and not torch.equal(state[r["name"]][keep], before[keep].to(state[r["name"]].dtype)):
                raise AssertionError(f"an average's unlisted rows of {r['name']} differ from their values before the fit")
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if is_cuda else None
    frozen_same = all(torch.equal(frozen_before[id(p)], p) for p in model.parameters() if id(p) in frozen_before)
    res = read(model, bed, heldout, device)
    res.update({"seed": seed, "steps": steps, "t_range": list(t_range), "t_sampling": t_sampling, "t_mix": float(t_mix),
                "trainable": trainable if isinstance(trainable, str) else sorted(trainable), "trained_numbers": n_par,
                "all_numbers": n_all, "train_seconds": seconds, "peak_memory_mib": peak, "curve": curve,
                "frozen_unchanged": bool(frozen_same),
                "recipe": recipe(lr, warmup, batch, steps, t_sampling, t_range, cond, bed.kind == "pictures", equalize,
                                 mdrop, device, t_mix),
                "torch": torch.__version__, "device": str(device)})
    if int(start_step) or optimizer_state is not None:     # recorded only for a run that continues another
        res["start_step"] = int(start_step)
        res["recipe"]["start_step"] = int(start_step)
        res["recipe"]["optimizer_state"] = "continued" if optimizer_state is not None else "new"
    if trainable_params:                                  # recorded only when given
        res["trainable_params"] = {k: [str(p) for p in v] for k, v in trainable_params.items()}
    if new_names:                                         # recorded only when the state lacked trainable parameters
        res["recipe"]["optimizer_state"] = "continued, partial"
        res["optimizer_new"] = new_names
    if fresh:
        res["average_new"] = {str(k): sorted(v) for k, v in fresh.items()}
    if rows:                                              # recorded only when given
        res["trainable_rows"] = {u: {pn: [int(i) for i in ix] for pn, ix in d.items()} for u, d in trainable_rows.items()}
        res["rows_frozen_equal"] = True
        res["rows_restored"] = bool(rows_restored)
    if averaged is not None:
        res["averaged"] = averaged                        # tensors: never written into a json file
    if keep_state:
        if not state_out:                                 # no step trained: the state as it was given
            state_out = dict(optimizer_state) if optimizer_state is not None else {
                "step": int(start_step), "m": {}, "v": {}, "t": {}, "lr": float(lr), "betas": list(BETAS), "eps": EPS}
            state_out["step"] = int(start_step)
        res["optimizer_state"] = state_out                # tensors: never written into a json file
    return res


def average_spec(x: Union[float, str]) -> Tuple[Union[float, str], str, Union[float, int]]:
    """An entry of ``averages`` -> (key, kind, value): a float (or a number's text) -> (the float, "decay", d); "g8" ->
    ("g8", "g", 8); "tail100" -> ("tail100", "tail", 100)."""
    if isinstance(x, str):
        text = x.strip()
        if text.startswith("g") and len(text) > 1:
            g = float(text[1:])
            if g < 0:
                raise ValueError(f"a g average's number is not negative: {x!r}")
            return text, "g", int(g) if g.is_integer() else g
        if text.startswith("tail") and len(text) > 4:
            n = int(text[4:])
            if n < 1:
                raise ValueError(f"a tail average needs at least one step: {x!r}")
            return text, "tail", n
        x = float(text)
    d = float(x)
    if not 0.0 <= d < 1.0:
        raise ValueError(f"an average's decay lies in [0, 1): {d}")
    return d, "decay", d


@contextlib.contextmanager
def use_average(model, state: Dict[str, torch.Tensor]):
    """Put averaged values ({parameter name: tensor}, as in ``fit_units(...)["averaged"][decay]``) into the model's
    parameters for the body of the ``with``; the last weights are put back exactly on exit (also when the body raises)."""
    named = dict(model.named_parameters())
    unknown = set(state) - set(named)
    if unknown:
        raise KeyError(f"no parameter named {sorted(unknown)} in the model")
    kept = {n: named[n].detach().clone() for n in state}
    try:
        with torch.no_grad():
            for n, v in state.items():
                named[n].copy_(v.to(device=named[n].device, dtype=named[n].dtype))
        yield model
    finally:
        with torch.no_grad():
            for n, v in kept.items():
                named[n].copy_(v)


def run_tag(a: Dict[str, Any], seed: int) -> str:
    return (f"{RUN_PREFIX}{a['bed']}{'tilt' if a['tilt'] else ''}_{a['name'] or compat.describe(a)}"
            f"_s{seed}_steps{a['steps']}_d{a['dim']}")


def fit(a: Dict[str, Any], bed, seed: int, log: Callable[[str], None]) -> dict:
    """Build, train and read one seed of an argument set; writes the outputs and returns the result."""
    device = a["device"]
    torch.manual_seed(seed)
    build = compat.build_from_args(a, bed.field_shape)
    model = Model.born(build, seed, bed.spectrum).to(device)
    loaded = None
    if a["init"]:
        loaded = compat.load_legacy_files(model, [p.format(seed=seed) for p in a["init"].split(",")], device)
    set_trainable(model, a["train"], loaded)
    frozen_before = {k: v.detach().clone() for k, v in model.named_parameters() if not v.requires_grad}
    if int(a["stage_zero"]) and model.stage_units and (loaded is None or loaded["stage_new"]):
        zc = torch.randn((4,) + tuple(bed.field_shape), device=device)
        tc = torch.tensor([0.1, 0.4, 0.7, 0.9], device=device)
        cc = torch.zeros(4, dtype=torch.long, device=device)
        with torch.no_grad():
            p_on, _, _, _ = model.components(zc, tc, cc)
            p_off, _, _, _ = model.components(zc, tc, cc, skip=[u.name for u in model.stage_units])
            gap = max(float((p_on[k] - p_off[k]).abs().max()) for k in p_on)
        if gap != 0.0:
            raise AssertionError(f"the stage units are not silent at birth: {gap}")
    tag = run_tag(a, seed)
    n_par = model.trained_numbers()
    n_all = sum(p.numel() for p in model.parameters())
    use_anchor = model.anchor is not None and model.anchor.output
    keep = any(m.keep for m in model.members)
    log(f"RUN {tag}: {n_par:,} trained numbers of {n_all:,}; members {[m.name for m in model.members]}; stage edges {model.edges}; "
        f"anchor {int(use_anchor)}; keep {int(keep)}"
        + (f"; the stem reads the anchor ({model.stem.anchor_in})" if model.stem.anchor_in else "")
        + (f"; stage units {len(model.stage_units)}" if model.stage_units else "")
        + (f"; init: loaded {loaded['members_loaded']}, new {loaded['members_new']}, unused {loaded['members_unused']}" if loaded else ""))
    read_steps = {int(x) for x in str(a["read_at"]).split(",") if x}
    at = {}
    z = torch.randn((2,) + tuple(bed.field_shape), device=device)
    with torch.no_grad():
        t2 = torch.tensor([0.3, 0.7], device=device)
        c2 = torch.zeros(2, dtype=torch.long, device=device)
        v0 = model.velocity(z, t2, c2)
        born = float((v0 - (model.anchor_velocity(z, t2, c2) if use_anchor else 0)).abs().max())
    if not a["init"] and born != 0.0:
        raise AssertionError(f"the newborn model is not exactly its anchor (or zero): {born}")
    if a["smoke"] and len(model.members):
        tt = torch.rand(256, device=device)
        cc = torch.zeros(256, dtype=torch.long, device=device)
        base = [m for m in model.members if m.name.startswith("b")]
        for b in sorted({model.band_index(m) for m in base}):
            tot = sum(m.gate(tt, cc) for m in base if model.band_index(m) == b)
            if float((tot - 1).abs().max()) >= 1e-6:
                raise AssertionError("the base members' windows do not sum to 1")
    cond_drop = a["cond_drop"] if bed.n_classes else 0.0
    t0 = time.time()

    def at_step(done: int) -> None:
        if done in read_steps and done < a["steps"]:
            mid = read(model, bed, a["heldout"], device)
            at[str(done)] = {"loss_over_levels": mid["loss_over_levels"], "heldout_loss": mid["heldout_loss"],
                             "by_level": {lv: r["error"]["all"] for lv, r in mid["levels"].items()},
                             "independence_ratio_over_levels": mid["independence_ratio_over_levels"],
                             "seconds": time.time() - t0}
            log(f"  AT {done}: over ten levels {mid['loss_over_levels']:.4f}; held-out {mid['heldout_loss']:.4f}")
            model.train()

    curve, _ = run_loop(model, bed, seed, a["steps"], a["lr"], a["warmup"], a["batch"], cond_drop, float(a["mdrop"]),
                        a["t_sampling"], (float(a["t_lo"]), float(a["t_hi"])), a["equalize"], device, log, a["log_every"],
                        at_step=at_step if read_steps else None, t_mix=float(a["t_mix"]))
    train_s = time.time() - t0
    named = dict(model.named_parameters())
    frozen_same = all(torch.equal(v, named[k]) for k, v in frozen_before.items())
    res = read(model, bed, a["heldout"], device)
    args = {k: a[k] for k in sorted(a) if not (k in RANGE_KEYS and a[k] == DEFAULTS[k])}
    res.update({"tag": tag, "seed": seed, "trained_numbers": n_par, "all_numbers": n_all, "train_seconds": train_s, "curve": curve,
                "born_minus_anchor": born, "frozen_unchanged": bool(frozen_same), "loaded": loaded, "at": at,
                "args": args, "torch": torch.__version__, "device": str(device)})
    os.makedirs(a["out"], exist_ok=True)
    with open(os.path.join(a["out"], tag + ".json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    torch.save(compat.to_legacy_state(model), os.path.join(a["out"], tag + ".pt"))
    write_build(model.build, os.path.join(a["out"], tag + ".build.json"))
    if bed.kind == "pictures" and a["sheet"]:
        sample_sheet(model, bed, os.path.join(a["out"], tag + "_samples.png"), device)
    lv = res["levels"]["0.45"]
    log(f"  READ {tag}: held-out {res['heldout_loss']:.4f}; over ten levels {res['loss_over_levels']:.4f} (linear physics "
        f"{res['linear_physics_loss_over_levels']:.4f}); at t 0.45 by band: "
        + "; ".join(f"{k} {lv['error'][k]:.4f} (linear {lv['linear_physics'][k]:.4f})" for k in lv["error"] if k != "all")
        + f"; independence {res['independence_ratio_over_levels']}; {train_s:.0f} s")
    return res


def main(argv=(), **overrides) -> List[dict]:
    """Run every seed of an argument set (``key=value`` items and/or keywords); returns the results."""
    a = parse(DEFAULTS, argv, **overrides)
    torch.set_num_threads(int(a["threads"]))
    if a["smoke"]:
        a.update(SMOKE)
    if a["device"] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda asked for and no card is visible")
        if os.name == "nt" and a["mem_fraction"] > 0:
            torch.cuda.set_per_process_memory_fraction(float(a["mem_fraction"]))
    os.makedirs(a["out"], exist_ok=True)
    log_path = os.path.join(a["out"], "log.txt")

    def log(line: str) -> None:
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")

    bed = make_bed(a, a["device"])
    seeds = [int(x) for x in str(a["seeds"]).split(",")]
    log(f"JOB: {len(seeds)} runs of {a['steps']} steps, batch {a['batch']}, bed {a['bed']}, on {a['device']}; "
        f"torch {torch.__version__}; out {a['out']}")
    t0 = time.time()
    done = []
    for i, seed in enumerate(seeds):
        done.append(fit(a, bed, seed, log))
        spent = time.time() - t0
        log(f"JOB: {i + 1}/{len(seeds)} runs done; spent {spent / 60:.1f} min; "
            f"left about {spent / (i + 1) * (len(seeds) - i - 1) / 60:.1f} min")
    log("done")
    return done


if __name__ == "__main__":
    main([] if in_notebook() else sys.argv[1:])
