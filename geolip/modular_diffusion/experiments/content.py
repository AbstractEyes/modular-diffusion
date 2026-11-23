"""Content as an element: where and how the class enters a model, the per-class closed-form anchor, class experts on a
finished model, and a class snapped in.

    python -m geolip.modular_diffusion.run content variant=blind seed=0 data=<npz> out=<folder> device=cuda
    from geolip.modular_diffusion.experiments import content; content.main(variant="plain", data=..., out=...)

The recipe of the time split's ``plain`` everywhere (the pictures bed, width 128, 2 heads, a token grid of 16, the anchor
added and read by the stem, Adam 1e-3, 100 warm-up steps, batch 128, class drop 0.1, a held-out set of 2,048, 7 blocks).

| variant             | build                                                                   | trained     | steps | t in      |
|---------------------|-------------------------------------------------------------------------|-------------|-------|-----------|
| plain               | anchor, stem of 7, head: the time split's plain, the same code path     | all         | 6,000 | [0, 1)    |
| class_start         | the stem reads the class only where t < 0.25                            | all         | 6,000 | [0, 1)    |
| blind               | the stem never reads the class                                          | all         | 6,000 | [0, 1)    |
| blind_canchor       | blind, over the per-class anchor (10 symbols)                           | all         | 6,000 | [0, 1)    |
| canchor             | plain, over the per-class anchor                                        | all         | 6,000 | [0, 1)    |
| blind9              | blind, trained without the pictures of class 9                          | all         | 6,000 | [0, 1)    |
| experts_start       | a finished plain job (source=) + ten class experts, on t < 0.25, soft 0.06 | the experts | 6,000 | [0, 0.31) |
| blind_experts_start | the same on a finished blind job                                        | the experts | 6,000 | [0, 0.31) |
| blind_experts_all   | a finished blind job + ten class experts on at every noise level        | the experts | 6,000 | [0, 1)    |
| snap9               | a finished blind9 job + one expert for class 9, trained on class 9 only | the expert  | 1,500 | [0, 1)    |
| anchors             | no training: a newborn blind_canchor model; the anchor is read          | -           | 0     | -         |
| class_to_mid        | the stem reads the class only where t < 0.65                            | all         | 6,000 | [0, 1)    |
| rand_start          | a finished plain job + ten parts gated by a random symbol (not the class), t < 0.25, soft 0.06 | the parts | 6,000 | [0, 0.31) |
| shared_start        | a finished plain job + ONE part on for all ten classes, t < 0.25, soft 0.06 | the part | 6,000 | [0, 0.31) |
| blind_rand_all      | a finished blind job + ten symbol-gated parts at every noise level      | the parts   | 6,000 | [0, 1)    |
| blind_shared_all    | a finished blind job + one part over all ten classes at every level     | the part    | 6,000 | [0, 1)    |
| blind_experts10_id  | blind_experts_all with identity-born blocks (``born: identity``) and silent heads | the experts | 6,000 | [0, 1) |
| blind_experts10_solved | the same, solve-born: before training the heads are the ten class heads solved on the frozen body | the experts | 6,000 | [0, 1) |
| blind_experts10_id_1500, blind_experts10_solved_1500 | the same two                           | the experts | 1,500 | [0, 1)    |
| blind_experts30_id  | a finished blind job + thirty identity-born experts, one per class and stage (the cuts 0.25 / 0.65, half-width 0.06; expert_s<stage>_c<class>) | the experts | 6,000 | [0, 1) |
| blind_experts30_solved | the same thirty, solve-born (the thirty-head solve)                  | the experts | 6,000 | [0, 1)    |
| snap9_solved        | a finished blind9 job + one solve-born expert for class 9 (the solve on class 9's pictures), trained on class 9 only | the expert | 1,500 | [0, 1) |

A solve-born part is identity-born (its block passes the trunk's states through unchanged at birth) and, before training,
its head is solved together with the other parts' by ``closed.part_equations`` / ``solve_parts`` / ``install_parts``
(``batches`` batches of the training stream, ridge ``ridge`` of the mean diagonal): at birth the model is, to the last
digit, the model of the same class heads with no block solved on the same base, seed and batches (the closed-forms
variants class_heads and class_stage_heads). Such a job records "solve" (the provenance of every solved head) and
"at_birth" (the read of the model before training: loss over the levels, by level, by stage).

A symbol is a label drawn per row uniformly from 0 .. 9, independently of the class (``model.draw_symbols``; a row whose
class was dropped has none): a symbol-gated part sees as many rows as a class expert and no class information, the
control of the class experts.

Arguments (key=value): those of the time split, plus source (a finished job folder of the same seed, for the expert
variants and snap9), source_model (the folder of the source job that is the frozen base, default model; for example
model_avg_g8; when given, the source folder is read again first and recorded under "as_source") and reader (0 / 1: the
class read on the scored samples; needs score=1 and ref=), batches (200) and ridge (1e-6) (the solve of the solve-born
variants; recorded only when changed). ``scale``
multiplies the step counts as in the time split. avg (a comma list of decays; every training variant, ignored by
anchors): each average of the trained parameters (the experts' for the expert variants) is exported as
model_avg<digits after the point>/, read and scored like the last weights, and reported under "averaged".

Output in ``out``: log.txt (its last line 'done'), result.json, samples.png (with sheet), the build folder model/.
result.json holds the time split's keys for a one-model job and: by_level_no_class and class_worth (the error with the
class replaced by "no class", and its difference to the error, by level and by stage), by_class (per class, the mean over
the ten levels of the error with the class given), anchor_by_level and anchor_pooled_by_level (the anchor alone with the
class and without it), exact (the expert variants: one expert left out against in, on the read's fixed noise), and in
score the class read (reader=1) and, when this job computed the reference file, the reader's check on the test pictures.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .. import __version__
from ..beds import PicturesBed
from ..config import in_notebook, parse
from ..data import npz_content_hash
from ..determinism import mode as det_mode
from ..closed import install_parts, part_equations, solve_parts
from ..forms import cluster_build, with_class_experts, with_parts
from ..linker import export_build, link
from ..model import Model
from ..read import CHUNK, LEVEL_ITEMS, READ_SEED, T_GRID, by_stage, read, read_symbols
from ..sample import sample_sheet
from ..score import (class_reader, make_judge, picture_score, reader_check, reference_classes, reference_stats)
from ..train import fit_units
from .time_split import (CUTS, JUDGES, SOFT, TRAIN_RANGES, _levels, _marks, _non_anchor, _part_row, _unique_numbers,
                         averages_cell, export_averages, load_job, parse_averages, recorded_args, scored_keys,
                         source_read, steps_of)

EXPERIMENT = "content"
TEN = tuple((c,) for c in range(10))
START = (0.0, 0.25)
START_SOFT = 0.06

SPECS: Dict[str, Dict[str, Any]] = {
    "plain": dict(steps=6000),
    "class_start": dict(steps=6000, class_window=("0", "0.25")),
    "blind": dict(steps=6000, class_window=("0", "0")),
    "blind_canchor": dict(steps=6000, class_window=("0", "0"), anchor_symbols=10),
    "canchor": dict(steps=6000, anchor_symbols=10),
    "blind9": dict(steps=6000, class_window=("0", "0"), without_classes=(9,)),
    "experts_start": dict(steps=6000, source="plain", experts=TEN, window=START, soft=START_SOFT, t_range=TRAIN_RANGES[0]),
    "blind_experts_start": dict(steps=6000, source="blind", experts=TEN, window=START, soft=START_SOFT,
                                t_range=TRAIN_RANGES[0]),
    "blind_experts_all": dict(steps=6000, source="blind", experts=TEN, window=(0.0, 1.0), soft=0.0, t_range=(0.0, 1.0)),
    "snap9": dict(steps=1500, source="blind9", experts=((9,),), window=(0.0, 1.0), soft=0.0, t_range=(0.0, 1.0),
                  only_classes=(9,)),
    "anchors": dict(steps=0, class_window=("0", "0"), anchor_symbols=10),
    # the sixth pass: the class read to the middle; parts gated by a label that is not the class, and one shared part
    "class_to_mid": dict(steps=6000, class_window=("0", "0.65")),
    "rand_start": dict(steps=6000, source="plain", experts=TEN, gate="symbol", window=START, soft=START_SOFT,
                       t_range=TRAIN_RANGES[0]),
    "shared_start": dict(steps=6000, source="plain", experts=(tuple(range(10)),), window=START, soft=START_SOFT,
                         t_range=TRAIN_RANGES[0]),
    "blind_rand_all": dict(steps=6000, source="blind", experts=TEN, gate="symbol", window=(0.0, 1.0), soft=0.0,
                           t_range=(0.0, 1.0)),
    "blind_shared_all": dict(steps=6000, source="blind", experts=(tuple(range(10)),), window=(0.0, 1.0), soft=0.0,
                             t_range=(0.0, 1.0)),
}
VARIANTS = tuple(SPECS)                 # the variants of earlier versions
EXPERT_VARIANTS = tuple(v for v, s in SPECS.items() if "experts" in s)
NEW_VARIANTS = ("class_to_mid", "rand_start", "shared_start", "blind_rand_all", "blind_shared_all")

# the seventh pass: how a content expert is born (identity-born, or solve-born: its head the class head solved on the
# frozen body before training), experts per class and stage, and a class snapped in by a solve
EVERY = dict(window=(0.0, 1.0), soft=0.0, t_range=(0.0, 1.0))
PASS7_SPECS: Dict[str, Dict[str, Any]] = {
    "blind_experts10_id": dict(steps=6000, source="blind", experts=TEN, born="identity", **EVERY),
    "blind_experts10_solved": dict(steps=6000, source="blind", experts=TEN, birth="solve", **EVERY),
    "blind_experts10_id_1500": dict(steps=1500, source="blind", experts=TEN, born="identity", **EVERY),
    "blind_experts10_solved_1500": dict(steps=1500, source="blind", experts=TEN, birth="solve", **EVERY),
    "blind_experts30_id": dict(steps=6000, source="blind", experts=TEN, by_stage=True, born="identity", **EVERY),
    "blind_experts30_solved": dict(steps=6000, source="blind", experts=TEN, by_stage=True, birth="solve", **EVERY),
    "snap9_solved": dict(steps=1500, source="blind9", experts=((9,),), only_classes=(9,), birth="solve", **EVERY),
}
PASS7_VARIANTS = tuple(PASS7_SPECS)
ALL_SPECS: Dict[str, Dict[str, Any]] = dict(SPECS, **PASS7_SPECS)
ALL_VARIANTS = tuple(ALL_SPECS)
PASS7_ARGS = {"batches": 200, "ridge": 1e-6}     # the solve of the solve-born variants; recorded only when changed

DEFAULTS = dict(variant="", seed=0, data="", out="", device="cuda", heldout=2048, batch=128, lr=1e-3, warmup=100,
                log_every=500, threads=4, mem_fraction=0.25, scale=1.0, dim=128, score=0, judge="inception",
                score_n=10000, score_steps=50, ref="", sheet=1, source="", stem_from="", reader=0, avg="", score_last=1,
                score_avg="", det="", source_model="", batches=200, ridge=1e-6)


def expert_groups(spec: Dict[str, Any]) -> List[dict]:
    """The parts an expert variant adds, as ``forms.with_parts`` groups: one class expert (one block) per class list
    over the spec's window, or, with "by_stage", one per class list and registered stage (the time split's cuts and
    crossfade; named expert_s<stage>_c<class>). Identity-born for "born": "identity" and for the solve-born variants."""
    classes = [list(c) for c in spec["experts"]]
    born = "identity" if spec.get("born") == "identity" or spec.get("birth") == "solve" else None
    extra = {} if born is None else {"born": born}
    if spec.get("by_stage"):
        return [dict(classes=classes, window=(CUTS[s], CUTS[s + 1]), soft=SOFT, depth=1, prefix=f"expert_s{s}", **extra)
                for s in range(len(CUTS) - 1)]
    return [dict(classes=classes, window=spec["window"], soft=spec["soft"], depth=1, prefix="expert",
                 gate=spec.get("gate", "class"), **extra)]


def _stage_means(by_level: Dict[str, float]) -> List[Optional[float]]:
    """The mean of a by-level dict over the fixed levels of each stage of CUTS (None for a stage holding none)."""
    out = []
    for lo, hi in zip(CUTS, CUTS[1:]):
        vals = [v for k, v in by_level.items() if float(lo) <= float(k) < float(hi)]
        out.append(float(np.mean(vals)) if vals else None)
    return out


def class_numbers(res: dict) -> Dict[str, Any]:
    """The class's worth, the per-class errors and the anchor columns of a read."""
    lv = res["levels"]
    no_class = {k: r["error_without_caption"]["all"] for k, r in lv.items()}
    err = {k: r["error"]["all"] for k, r in lv.items()}
    worth = {k: no_class[k] - err[k] for k in lv}
    by_class = [float(np.mean([r["error_by_class"][c] for r in lv.values()])) for c in range(10)]
    anchor = {k: r["linear_physics"]["all"] for k, r in lv.items()}
    pooled = {k: r.get("linear_physics_without_caption", r["linear_physics"])["all"] for k, r in lv.items()}
    return {"by_level_no_class": no_class, "class_worth": {"by_level": worth, "by_stage": _stage_means(worth)},
            "by_class": by_class, "anchor_by_level": anchor, "anchor_pooled_by_level": pooled}


def _sequential_sum(parts: dict, like: torch.Tensor) -> torch.Tensor:
    v = torch.zeros_like(like)
    for p in parts.values():                    # one addition at a time: an exactly zero part changes no digit
        v = v + p
    return v


def _unit_named(model, name: str):
    """A unit of a model, or of the first cluster of a staged model that holds it (its copies in other clusters share
    its address)."""
    clusters = getattr(model, "clusters", None)
    if clusters is None:
        return model.unit(name)
    for c in clusters:
        if name in c.build.names():
            return c.unit(name)
    raise KeyError(f"no unit named {name!r} in the staged model")


@torch.no_grad()
def exactness(model, bed, heldout: int, device, expert: str) -> Dict[str, Any]:
    """One expert left out against in, on the read's fixed noise (the first LEVEL_ITEMS held-out items at the ten
    levels): the largest absolute change of a row's error over the rows of the other classes (0.0 when the expert is
    exactly silent there) and the mean change (out minus in) over its own class's rows. For a symbol-gated part the
    rows are read with the read's fixed symbols and its own rows are those of its symbols. A staged model (a tree)
    leaves the expert out of every cluster."""
    model.eval()
    z1, cls = bed.heldout(heldout)
    gen = torch.Generator(device=device).manual_seed(READ_SEED)
    z0 = torch.randn(z1.shape, generator=gen, device=device)
    n_lv = min(z1.shape[0], LEVEL_ITEMS)
    unit = _unit_named(model, expert)
    sym = read_symbols(model, z1.shape[0], device)
    label = sym if getattr(unit, "symbol_gate", False) else cls
    own_classes = torch.tensor(list(unit.classes), device=label.device, dtype=label.dtype)
    others_max, own_sum, own_n = 0.0, 0.0, 0
    for t in T_GRID:
        for i in range(0, n_lv, CHUNK):
            sl = slice(i, min(i + CHUNK, n_lv))
            b = z1[sl].shape[0]
            tt = torch.full((b,), t, device=z1.device)
            zt = (1 - t) * z0[sl] + t * z1[sl]
            target = z1[sl] - z0[sl]
            kw = {} if sym is None else {"symbols": sym[sl]}
            p_in, _, _, _ = model.components(zt, tt, cls[sl], **kw)
            p_out, _, _, _ = model.components(zt, tt, cls[sl], skip=[expert], **kw)
            e_in = ((_sequential_sum(p_in, zt) - target) ** 2).mean(dim=(1, 2, 3)).double()
            e_out = ((_sequential_sum(p_out, zt) - target) ** 2).mean(dim=(1, 2, 3)).double()
            diff = e_out - e_in
            own = torch.isin(label[sl], own_classes)
            if bool((~own).any()):
                others_max = max(others_max, float(diff[~own].abs().max()))
            own_sum += float(diff[own].sum())
            own_n += int(own.sum())
    return {"expert": expert, "others_max_abs": others_max, "own_mean_change": own_sum / own_n if own_n else None}


def _check_source(folder: str, need: str, seed: int, model_folder: str = "model") -> None:
    path = os.path.join(folder, "result.json")
    if not os.path.isdir(os.path.join(folder, model_folder)):
        raise ValueError(f"{folder} holds no {model_folder} folder (a finished {need} job is needed)")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            r = json.load(fh)
        if r.get("variant") != need or int(r.get("seed", -1)) != int(seed):
            raise ValueError(f"{folder} is the job {r.get('variant')!r} of seed {r.get('seed')}, not {need!r} of seed {seed}")


def main(argv=(), **overrides) -> dict:
    """Run one variant of one seed; writes the outputs into ``out`` and returns the result."""
    a = parse(DEFAULTS, argv, **overrides)
    with det_mode(a["det"]) as det:
        return _run(a, det)


def _run(a: dict, det) -> dict:
    variant = a["variant"]
    if variant not in ALL_SPECS:
        raise ValueError(f"unknown variant {variant!r}; known: {list(ALL_VARIANTS)}")
    for k in ("data", "out"):
        if not a[k]:
            raise ValueError(f"{k}= is required")
    spec = ALL_SPECS[variant]
    seed, dim, scale = int(a["seed"]), int(a["dim"]), float(a["scale"])
    source_model = a["source_model"] or "model"
    if "source" in spec:
        if not a["source"]:
            raise ValueError(f"{variant} needs source=<the folder of a finished {spec['source']} job of the same seed>")
        _check_source(a["source"], spec["source"], seed, source_model)
    elif a["source_model"]:
        raise ValueError(f"{variant} reads no source; source_model= applies to the variants with a source")
    if int(a["score"]) and a["judge"] not in JUDGES:
        raise ValueError(f"unknown judge {a['judge']!r}; known: {list(JUDGES)}")
    if int(a["reader"]) and not (int(a["score"]) and a["ref"]):
        raise ValueError("reader=1 needs score=1 and ref=<the reference file>")
    averages = [] if variant == "anchors" else parse_averages(a["avg"])        # anchors trains nothing to average
    to_score = scored_keys(a, averages)
    torch.set_num_threads(int(a["threads"]))
    device = a["device"]
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda asked for and no card is visible")
        if os.name == "nt" and a["mem_fraction"] > 0:
            torch.cuda.set_per_process_memory_fraction(float(a["mem_fraction"]))
    out = a["out"]
    os.makedirs(out, exist_ok=True)
    log_path = os.path.join(out, "log.txt")

    def log(line: str) -> None:
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")

    n = steps_of(spec["steps"], scale) if spec["steps"] else 0
    log(f"JOB {EXPERIMENT} {variant} seed {seed}: {1 if n else 0} part(s), steps {[n] if n else []} ({n} in all), on "
        f"{device}; torch {torch.__version__}; out {out}")
    t_job = time.time()
    bed = PicturesBed(a["data"], device, grid=16)
    data_block = {"bed": "pictures", "source": os.path.basename(a["data"]), "content_hash": npz_content_hash(a["data"]),
                  "heldout": f"the first {a['heldout']} test pictures"}
    if "without_classes" in spec or "only_classes" in spec:
        train_bed = bed.restricted(without_classes=spec.get("without_classes", ()), only_classes=spec.get("only_classes"))
        data_block["training_stream"] = ({"without_classes": list(train_bed.without_classes)} if train_bed.without_classes
                                         else {"only_classes": list(train_bed.only_classes)})
    else:
        train_bed = bed
    common = dict(lr=float(a["lr"]), warmup=int(a["warmup"]), batch=int(a["batch"]), cond_drop=0.1,
                  heldout=int(a["heldout"]), device=device, log=log, log_every=int(a["log_every"]))
    field = bed.field_shape

    def export(model, res_recipe: dict, marks: dict, trained, base: Optional[str] = None) -> None:
        export_build(model, os.path.join(out, "model"), seed=seed, base=base, trained=trained, recipe=res_recipe,
                     data=data_block, marks=marks)

    parts: List[dict] = []
    exact = None
    decays = tuple(d for _, _, d in averages)
    torch.manual_seed(seed)
    as_source = None
    solve, at_birth = None, None
    if "experts" in spec:
        base = os.path.join(a["source"], source_model)
        if a["source_model"]:                      # recorded only when asked for, so earlier jobs write their result
            as_source = source_read(a["source"], source_model, bed, int(a["heldout"]), device, log)
        if variant in PASS7_SPECS:
            model, names = with_parts(base, seed, expert_groups(spec), device=device)
        else:
            model, names = with_class_experts(base, seed, [list(c) for c in spec["experts"]], spec["window"],
                                              spec["soft"], depth=1, device=device, gate=spec.get("gate", "class"))
        if spec.get("birth") == "solve":
            # solve-born: the identity-born parts' heads solved together on the frozen body, then read once
            t0 = time.time()
            eq = part_equations(model, train_bed, seed, int(a["batches"]), names, batch=int(a["batch"]), cond_drop=0.1,
                                t_range=spec["t_range"])
            solution = solve_parts(eq, float(a["ridge"]))
            solve = install_parts(model, solution, eq)
            solve["all"]["seconds"] = time.time() - t0
            log(f"SOLVE {len(names)} part head(s) in {len(eq['groups'])} group(s) on {int(a['batches'])} batches: mse "
                f"over the equations' rows {solve['all']['mse_before']:.5f} -> {solve['all']['mse_after']:.5f}"
                + (f"; never on: {solution['unsolved']}" if solution["unsolved"] else ""))
            birth = read(model, bed, int(a["heldout"]), device)
            at_birth = {"loss_over_levels": birth["loss_over_levels"], "by_level": _levels(birth),
                        "by_stage": by_stage(birth, CUTS)}
            log(f"AT BIRTH (solved heads): over ten levels {birth['loss_over_levels']:.4f}")
        log(f"PART model: {len(names)} expert(s) on {a['source']}; t in [{spec['t_range'][0]:g}, {spec['t_range'][1]:g}), "
            f"{n} steps")
        res = fit_units(model, train_bed, seed, steps=n, trainable=names, t_range=spec["t_range"], mdrop=0.0,
                        equalize=False, expect_anchor_at_birth=False, averages=decays, **common)
        export(model, res["recipe"], _marks(res), names, base=base)
        if averages:
            export_averages(model, res, averages, out, lambda m, f, rec: export_build(
                m, os.path.join(out, f), seed=seed, base=base, trained=names, recipe=rec, data=data_block,
                marks={"loss_over_levels": None, "frozen_unchanged": res["frozen_unchanged"]}))
        parts.append(_part_row("model", "model", res))
    elif variant == "anchors":
        build = cluster_build(field, stem_depth=7, dim=dim, class_window=spec["class_window"],
                              anchor_symbols=spec["anchor_symbols"])
        model = Model.born(build, seed, bed.spectrum, class_stats=bed.class_stats()).to(device)
        export(model, {"optimizer": None, "steps": 0, "note": "newborn, not trained"}, {}, [])
    else:
        build = cluster_build(field, stem_depth=7, dim=dim, class_window=spec.get("class_window"),
                              anchor_symbols=spec.get("anchor_symbols", 0))
        stats = bed.class_stats() if spec.get("anchor_symbols") else None
        model = Model.born(build, seed, bed.spectrum, class_stats=stats)
        res = fit_units(model, train_bed, seed, steps=n, t_sampling="uniform", expect_anchor_at_birth=True,
                        averages=decays, **common)
        export(model, res["recipe"], _marks(res), _non_anchor(model))
        if averages:
            export_averages(model, res, averages, out, lambda m, f, rec: export_build(
                m, os.path.join(out, f), seed=seed, trained=_non_anchor(m), recipe=rec, data=data_block,
                marks={"loss_over_levels": None, "frozen_unchanged": res["frozen_unchanged"]}))
        parts.append(_part_row("model", "model", res))
    if n:
        spent = time.time() - t_job
        log(f"JOB: part model done; {n}/{n} steps; spent {spent / 60:.1f} min; left about 0.0 min")
    final, _ = load_job(out, device=device, kind="model")

    log("READ the model")
    whole = read(final, bed, int(a["heldout"]), device)
    cn = class_numbers(whole)
    if variant == "anchors":
        loss = float(np.mean(list(cn["anchor_by_level"].values())))
        levels = dict(cn["anchor_by_level"])
    else:
        loss = whole["loss_over_levels"]
        levels = _levels(whole)
    result: Dict[str, Any] = {
        "experiment": EXPERIMENT, "variant": variant, "seed": seed, "kind": "model", "cuts": list(CUTS), "soft": SOFT,
        "parts": parts,
        "numbers": {"all": _unique_numbers(final), "trained": sum(p["trained_numbers"] for p in parts)},
        "train_seconds": sum(p["train_seconds"] for p in parts), "read": whole, "loss_over_levels": loss,
        "by_level": levels, "by_stage": _stage_means(levels) if variant == "anchors" else by_stage(whole, CUTS),
        "wrong_assembly": None, "alone": None, "score": None}
    result.update(cn)
    if "experts" in spec:
        names = [u.name for u in final.members]
        exact = exactness(final, bed, int(a["heldout"]), device, names[0])
        log(f"EXACT {exact['expert']} left out: other classes' rows change at most {exact['others_max_abs']!r}; "
            f"its own rows by {exact['own_mean_change']}")
    result["exact"] = exact
    if solve is not None:                          # the solve-born variants only
        result["solve"] = solve
        result["at_birth"] = at_birth
    judged: Dict[str, Any] = {}

    def score_of(m, tag: str = "") -> dict:
        # a failure of the judge (its weights, its memory) is recorded and the job still ends with its losses
        n_score = int(a["score_n"])
        try:
            if "judge" not in judged:
                judge = make_judge(a["judge"], device)
                rdr, fresh = None, False
                if int(a["reader"]):
                    ref_all = reference_classes(bed, judge, a["ref"] or None, batch=1000)
                    ref = (ref_all["mu"], ref_all["cov"], ref_all["n"])
                    rdr, fresh = class_reader(ref_all), bool(ref_all["fresh"])
                else:
                    ref = reference_stats(bed, judge, a["ref"] or None, batch=1000)
                judged.update(judge=judge, ref=ref, reader=rdr, fresh=fresh)
            rdr = judged["reader"]
            sc = picture_score(m, bed, judged["judge"], judged["ref"], n=n_score, steps=int(a["score_steps"]),
                               batch=min(1000, n_score), device=device, reader=rdr)
            if rdr is not None and judged["fresh"]:
                sc["reader_check"] = reader_check(bed, judged["judge"], rdr)
                judged["fresh"] = False                   # the check is written once, beside the last weights' score
            log(f"SCORE{tag} {sc['score']:.4f} on {n_score} pictures ({sc['judge']})"
                + (f"; class read {sc['class_read']['accuracy']:.4f}" if rdr is not None else ""))
            return sc
        except Exception as e:
            log(f"SCORE{tag} not computed: {e!r}")
            return {"error": repr(e)}

    if int(a["score"]) and int(a["score_last"]):
        result["score"] = score_of(final)
    if averages:
        result["averaged"] = {}
        for key, folder, d in averages:
            avg_model = link(os.path.join(out, folder), device=device)
            log(f"READ the average {key} ({folder})")
            r = read(avg_model, bed, int(a["heldout"]), device)
            c = class_numbers(r)
            scored = int(a["score"]) and key in to_score
            result["averaged"][key] = {"loss_over_levels": r["loss_over_levels"], "by_level": _levels(r),
                                       "by_stage": by_stage(r, CUTS), "by_class": c["by_class"],
                                       "class_worth": c["class_worth"],
                                       "score": score_of(avg_model, f" (average {key})") if scored else None}
            log(f"AVERAGE {key}: over ten levels {r['loss_over_levels']:.4f}")
    if as_source is not None:
        result["as_source"] = as_source
    if int(a["sheet"]):
        sample_sheet(final, bed, os.path.join(out, "samples.png"), device)
    # avg= and the newer options are recorded only when given
    args = {k: v for k, v in recorded_args(a).items() if not (k in PASS7_ARGS and v == PASS7_ARGS[k])}
    result.update({"args": args, "torch": torch.__version__, "device": str(device), "version": __version__})
    det.record(result)
    with open(os.path.join(out, "result.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=1)
        fh.write("\n")
    bs = ", ".join("-" if x is None else f"{x:.4f}" for x in result["by_stage"])
    worth = result["class_worth"]["by_stage"][0]
    log(f"RESULT {variant} seed {seed}: over ten levels {result['loss_over_levels']:.4f}; by stage {bs}; the class's "
        f"worth in the start stage {_fmt(worth)}; {(time.time() - t_job) / 60:.1f} min")
    log("done")
    return result


def _fmt(x, nd: int = 4) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def summarize(root: str) -> str:
    """A markdown table, one row per job folder under ``root`` holding a result.json of this experiment."""
    rows = []
    for dirpath, _, files in os.walk(root):
        if "result.json" in files:
            with open(os.path.join(dirpath, "result.json"), encoding="utf-8") as fh:
                r = json.load(fh)
            if r.get("experiment") == EXPERIMENT:
                rows.append(r)
    rows.sort(key=lambda r: (r["variant"], r["seed"]))
    with_avg = any(r.get("averaged") for r in rows)
    with_birth = any(r.get("at_birth") for r in rows)          # a solve-born job: its read before training
    lines = ["| variant | seed | total | by stage | class's worth (start) | score | class read |"
             + (" average (first decay) | average total | average score | averages (total / score) |" if with_avg else "")
             + (" at birth (solved) |" if with_birth else ""),
             "|---|---|---|---|---|---|---|" + ("---|---|---|---|" if with_avg else "") + ("---|" if with_birth else "")]
    for r in rows:
        stages = " / ".join(_fmt(x) for x in r["by_stage"])
        worth = _fmt(((r.get("class_worth") or {}).get("by_stage") or [None])[0])
        sc = r.get("score") or {}
        score = _fmt(sc["score"], 2) if sc.get("score") is not None else "-"
        cread = _fmt((sc.get("class_read") or {}).get("accuracy"), 3)
        line = (f"| {r['variant']} | {r['seed']} | {_fmt(r['loss_over_levels'])} | {stages} | {worth} | {score} | "
                f"{cread} |")
        if with_avg:
            av = r.get("averaged") or {}
            if av:
                key = next(iter(av))
                asc = av[key].get("score") or {}
                line += (f" {key} | {_fmt(av[key]['loss_over_levels'])} | "
                         f"{_fmt(asc['score'], 2) if asc.get('score') is not None else '-'} | {averages_cell(av)} |")
            else:
                line += " - | - | - | - |"
        if with_birth:
            line += f" {_fmt((r.get('at_birth') or {}).get('loss_over_levels'))} |"
        lines.append(line)
    return "\n".join(lines) + "\n"


if __name__ == "__main__" and not in_notebook():
    main(sys.argv[1:])
