"""Training: the recipe, the loop and the entry point.

    python -m geolip.modular_diffusion.train bed=pictures data=<npz> bands=3 keep=1 codes=rung anchor=1 anchor_in=1 \\
        seeds=0 steps=6000 device=cuda out=<folder>
    from geolip.modular_diffusion import train; train.main(bed="fields", steps=50, device="cpu", out="smoke")

The recipe. Adam (betas 0.9 / 0.999, eps 1e-8, no weight decay), the learning rate warmed up linearly over ``warmup``
steps and set before every step; one generator per step, seeded 1,000,003 (seed + 1) + step on the run device, drawing
the batch (picture indices, flips), the noise, t, the class drop (probability ``cond_drop``, pictures only) and member
dropout, in that order; after the backward pass, member gradient equalization (``equalize=1``): every member's gradient
is scaled to the upper median of the members' gradient norms.

Outputs per seed, in ``out``: <tag>.json (the read, the curve and the arguments), <tag>.pt (the flat state dict),
<tag>.build.json (the build), <tag>_samples.png (pictures, when ``sheet``) and a log. The last log line is 'done'.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

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
                stage_zero=0, stage_soft=0.0, t_sampling="uniform")
SMOKE = dict(steps=4, batch=8, heldout=32, log_every=2, sheet=0)


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
        v0 = model.velocity(z, t2, torch.zeros(2, dtype=torch.long, device=device))
        born = float((v0 - (model.anchor_velocity(z, t2) if use_anchor else 0)).abs().max())
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
    params = [p for p in model.parameters() if p.requires_grad]
    cond_drop = a["cond_drop"] if bed.n_classes else 0.0
    curve = []
    t0 = time.time()
    if params and a["steps"] > 0:
        opt = torch.optim.Adam(params, lr=a["lr"], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        model.train()
        for step in range(a["steps"]):
            for group in opt.param_groups:
                group["lr"] = a["lr"] * min(1.0, (step + 1) / max(1, a["warmup"]))
            gen = step_generator(seed, step, device)
            x, cls = bed.train_batch(a["batch"], gen)
            loss = model.loss(x, cls, gen, cond_drop, mdrop=float(a["mdrop"]), t_sampling=a["t_sampling"])
            if not loss.requires_grad:               # no sample of this batch reached a trained unit
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if a["equalize"]:
                equalize(model)
            opt.step()
            if step % a["log_every"] == 0 or step == a["steps"] - 1:
                spent = time.time() - t0
                curve.append([step, float(loss.detach())])
                log(f"  step {step + 1}/{a['steps']}  loss {float(loss.detach()):.4f}  spent {spent:.0f} s  "
                    f"left about {spent / (step + 1) * (a['steps'] - step - 1):.0f} s")
            if (step + 1) in read_steps and (step + 1) < a["steps"]:
                mid = read(model, bed, a["heldout"], device)
                at[str(step + 1)] = {"loss_over_levels": mid["loss_over_levels"], "heldout_loss": mid["heldout_loss"],
                                     "by_level": {lv: r["error"]["all"] for lv, r in mid["levels"].items()},
                                     "independence_ratio_over_levels": mid["independence_ratio_over_levels"],
                                     "seconds": time.time() - t0}
                log(f"  AT {step + 1}: over ten levels {mid['loss_over_levels']:.4f}; held-out {mid['heldout_loss']:.4f}")
                model.train()
    train_s = time.time() - t0
    named = dict(model.named_parameters())
    frozen_same = all(torch.equal(v, named[k]) for k, v in frozen_before.items())
    res = read(model, bed, a["heldout"], device)
    res.update({"tag": tag, "seed": seed, "trained_numbers": n_par, "all_numbers": n_all, "train_seconds": train_s, "curve": curve,
                "born_minus_anchor": born, "frozen_unchanged": bool(frozen_same), "loaded": loaded, "at": at,
                "args": {k: a[k] for k in sorted(a)}, "torch": torch.__version__, "device": str(device)})
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
