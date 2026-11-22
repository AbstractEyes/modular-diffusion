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


def run_loop(model, bed, seed: int, steps: int, lr: float, warmup: int, batch: int, cond_drop: float, mdrop: float,
             t_sampling: str, t_range: Sequence[float], equalize_members, device, log: Callable[[str], None], log_every: int,
             at_step: Optional[Callable[[int], None]] = None, t_mix: float = 0.0) -> Tuple[List[list], float]:
    """The training loop of the recipe over the model's trainable parameters -> (curve [[step, loss], ...], seconds).

    t_range and t_mix: see ``Model.draw``. A step whose loss reaches no trained unit is skipped. ``at_step(steps done)`` is called after each step that trained.
    Nothing happens when no parameter trains or steps <= 0."""
    params = [p for p in model.parameters() if p.requires_grad]
    curve: List[list] = []
    t0 = time.time()
    if not params or steps <= 0:
        return curve, time.time() - t0
    opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    model.train()
    for step in range(steps):
        for group in opt.param_groups:
            group["lr"] = lr * min(1.0, (step + 1) / max(1, warmup))
        gen = step_generator(seed, step, device)
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
              expect_anchor_at_birth: bool = False, averages: Sequence[float] = ()) -> dict:
    """Train the named units of a model (a Model or a Staged) with the recipe and read it.

    averages: decays d. For each, a float32 copy of every trainable parameter (on its device, no gradient) starts as
    the parameters before the first step and, after optimizer step k = 0, 1, ..., becomes d_k avg + (1 - d_k) w with
    d_k = min(d, (1 + k) / (10 + k)). Nothing else changes (the stream, the curve and the last weights are those of a
    run without); the result then also holds "averaged": {d: {parameter name: tensor}} (keep it out of json files).

    Sets the trainable units; with ``expect_anchor_at_birth`` checks that the velocity equals the anchor's (or zero
    without an anchor) exactly on a 2-row probe; runs the loop; checks that the frozen parameters did not change; reads.
    cond_drop applies only when the bed has classes. Returns the read with {seed, steps, t_range, t_sampling, trainable,
    trained_numbers, all_numbers, train_seconds, peak_memory_mib, curve, frozen_unchanged, recipe, torch, device}."""
    model.to(device)
    set_trainable_units(model, trainable)
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
    n_all = sum(p.numel() for p in model.parameters())
    cond = cond_drop if bed.n_classes else 0.0
    is_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
    log(f"FIT {n_par:,} trained numbers of {n_all:,}; trainable {trainable if isinstance(trainable, str) else sorted(trainable)}; "
        f"{steps} steps; t in [{t_range[0]:g}, {t_range[1]:g}) ({t_sampling}" + (f", whole-range share {t_mix:g})" if t_mix else ")"))
    decays = [float(d) for d in averages]
    for d in decays:
        if not 0.0 <= d < 1.0:
            raise ValueError(f"an average's decay lies in [0, 1): {d}")
    at_step = None
    averaged = None
    if decays:
        live = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        with torch.no_grad():
            averaged = {d: {n: p.detach().clone().float() for n, p in live} for d in decays}
        updates = [0]

        def at_step(done: int) -> None:                   # after every optimizer step k = 0, 1, ...
            k = updates[0]
            with torch.no_grad():
                for d, state in averaged.items():
                    dk = min(d, (1 + k) / (10 + k))
                    for n, p in live:
                        state[n].copy_(dk * state[n] + (1.0 - dk) * p.detach().float())    # the rule as written
            updates[0] += 1

    curve, seconds = run_loop(model, bed, seed, steps, lr, warmup, batch, cond, mdrop, t_sampling, t_range, equalize,
                              device, log, log_every, at_step=at_step, t_mix=float(t_mix))
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
    if averaged is not None:
        res["averaged"] = averaged                        # tensors: never written into a json file
    return res


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
