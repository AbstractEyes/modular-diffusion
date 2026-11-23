"""Stages: K-stage models whose stages continue one shared run, tails on a frozen stem, and a model grown block by block.

    python -m geolip.modular_diffusion.run stages variant=source seed=0 data=<npz> out=<folder> device=cuda
    python -m geolip.modular_diffusion.run stages variant=copies cuts=0.25,0.65 source=<a source job> data=<npz> out=<folder>
    from geolip.modular_diffusion.experiments import stages; stages.main(variant="grow", stem_from=..., data=..., out=...)

The recipe of the time split everywhere (the pictures bed, width 128, 2 heads, a token grid of 16, the anchor added and
read by the stem, Adam 1e-3, 100 warm-up steps, batch 128, class drop 0.1, a held-out set of 2,048).

Stages. ``cuts`` lists the inner cuts (increasing, inside (0, 1); empty: one stage); stage s spans [lo, hi) between
neighboring cuts and is trained on [max(0, lo - soft), min(1, hi + soft)) (only inner edges are widened; the ends are
rounded to 10 decimals, so 0.65 - 0.06 is 0.59). The crossfade half-width ``soft`` defaults to
min(0.06, round(0.25 x the narrowest stage's width, 3)). A stage's steps are round(width x total) times ``scale`` (at
least 2; the rounding of the time split's step counts, so the cuts 0.25 / 0.65 give its counts).

| variant | what                                                                                                   | trained |
|---------|--------------------------------------------------------------------------------------------------------|---------|
| source  | the time split's plain build and code path for ``total`` steps (default 4,000; with 4,000 it is the time split's plain_4k to the bit), averages ``avg`` (default g8), and its trainer state written to state/ | all |
| copies  | K clusters, each born as the source's ``source_model`` folder (default model, the last weights) and CONTINUING the source's run on its stage's range: the source's step and optimizer state, no new warm-up. Averages (default g8,tail100): a g average is carried from the source's model_avg_<name> folder when it has one (k counting from the source's step). fresh=1: each copy with a new optimizer, the recipe's warm-up and step 0 (the time split's copies recipe; total=2000 with the cuts 0.25 / 0.65 is its copies_4k to the bit); the g averages still carried (default total 2,000) | every unit of every copy but the anchor |
| tails   | the anchor and the stem of ``stem_from`` / ``stem_model``, frozen and shared; per stage a newborn body of 3 blocks and a newborn head (the time split's tails_alone build), trained alone on its stage's range with a new optimizer; each tail keeps its own averages (default g8; default total 6,000) | body and head of each stage |
| grow    | the anchor and the stem of ``stem_from`` / ``stem_model``, frozen, and a newborn head; then ``blocks`` times: one block born silent between the last block and the head (a ``units.DepthBlock``; the birth gap must be exactly 0.0), the new block and the head trained ``steps_each`` steps over the whole range with a new optimizer, the warm-up again and a stream of its own (growth step i draws its batches with the seed ``seed + 100003 i``, so no batch is seen by two blocks) and their own g8 average, whose values are then installed as their weights (the block frozen from here on; the head goes on from its averaged values) | per growth step: the new block and the head |

| thaw    | a finished tails or grow job's build (``base``; ``base_model``, default its g8 folder avg_g8 for tails, model for grow), read again first, trained on over the whole range for ``total`` steps (default 2,000) with a NEW optimizer (Adam, the recipe's rate, the warm-up from step 0) and a stream of its own (the seed ``seed + 100003 x 7``, steps from 0), averages ``avg`` (default g8,tail100); the tail average is the scored reading (``score_avg`` widens it) | thaw=tail: the units the base job trained; 1 / 2: those and the stem's last one / two blocks; all: every trained number |

Every result reads the model on the registered stages (the time split's cuts 0.25 / 0.65 in ``by_stage``), whatever
the model's cuts.

Arguments (key=value): those of the time split, plus cuts, soft (negative: the default rule), total (0: the variant's
default), source, source_model, stem_from, stem_model, fresh, blocks (3), steps_each (2,000; times ``scale``), avg (a
comma list of decays and names; empty: the variant's default; ``none``: no average), score_last, score_avg, reader
(the class read on the scored samples; needs score=1 and ref=), det, and for thaw base, base_model and thaw (tail, 1,
2 or all). A thaw job keeps its base's cuts and crossfade, writes the folders of its base's kind (stage_0 .. with
staged.json and avg_<name>/ for a tails base; model/ and model_avg_<name>/ for a grow base) and records "thaw" (the
level, the base, the stream seed, the trainable units and parameter prefixes); its part row holds the seconds per step
and the peak memory.

Output in ``out``: log.txt (its last line 'done'), result.json, samples.png (with sheet: the scored model) and the
models: source writes model/, one model_avg_<name>/ per average (model_avg<digits>/ for a decay) and state/; copies and
tails write stage_0 .. stage_{K-1}/ with staged.json (one stage: a staged folder of one cluster) and one staged folder
per average under avg_<name>/ (avg_g8/stage_0/, ...); grow writes model/ (the final model, every averaged value
installed) and no staged file. result.json: experiment, variant, seed, kind, cuts, soft, total, parts, numbers,
train_seconds, read, loss_over_levels, by_level, by_stage, by_class, class_worth, score, averaged ({name:
{loss_over_levels, by_level, by_stage, by_class, class_worth, score}}), as_source (the base folder read again first;
for tails and grow the folder of stem_from), args, torch, device, version; for grow also steps (per growth step: the
birth gap, the read after the step on its last and on its averaged weights, the units' content hashes).
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch

from .. import __version__
from ..beds import PicturesBed
from ..build import Build, UnitEntry, content_hash, read_build
from ..config import in_notebook, parse
from ..data import npz_content_hash
from ..determinism import mode as det_mode
from ..forms import adopt, cluster_build
from ..linker import born_gap, export_build, link
from ..model import Model
from ..read import by_stage, read
from ..sample import sample_sheet
from ..score import class_reader, make_judge, picture_score, reader_check, reference_classes, reference_stats
from ..staged import STAGED_FILE, link_staged, read_staged, write_staged
from ..train import average_spec, fit_units, read_state, use_average, write_state
from ..units import born
from .content import class_numbers
from .time_split import (CUTS, JUDGES, _levels, _marks, _non_anchor, _part_row, _unique_numbers, average_recipe,
                         export_averages, load_job, parse_averages, recorded_args, scored_keys, source_read, steps_of)

EXPERIMENT = "stages"
VARIANTS = ("source", "copies", "tails", "grow")             # the variants of earlier versions
ALL_VARIANTS = VARIANTS + ("thaw",)
DEFAULT_TOTAL = {"source": 4000, "copies": 2000, "tails": 6000, "grow": 0, "thaw": 2000}
DEFAULT_AVG = {"source": "g8", "copies": "g8,tail100", "tails": "g8", "grow": "g8", "thaw": "g8,tail100"}
SOFT_CAP = 0.06
TAIL_DEPTH = 3
GROW_STREAM = 100003                 # growth step i trains on the stream of seed + GROW_STREAM x i (step 0: the job's seed)
THAW_STREAM = 7                      # a thaw pass trains on the stream of seed + GROW_STREAM x 7 (no build's batches)
THAW_LEVELS = ("tail", "1", "2", "all")
STAGE_ARGS = {"cuts": "", "soft": -1.0, "total": 0, "fresh": 0, "blocks": 3, "steps_each": 2000, "base": "",
              "base_model": "", "thaw": ""}

DEFAULTS = dict(variant="", seed=0, data="", out="", device="cuda", heldout=2048, batch=128, lr=1e-3, warmup=100,
                log_every=500, threads=4, mem_fraction=0.25, scale=1.0, dim=128, score=0, judge="inception",
                score_n=10000, score_steps=50, ref="", sheet=1, source="", source_model="", stem_from="", stem_model="",
                cuts="", soft=-1.0, total=0, fresh=0, blocks=3, steps_each=2000, avg="", score_last=1, score_avg="",
                reader=0, det="", base="", base_model="", thaw="")


def parse_cuts(text: str) -> List[float]:
    """``cuts=`` text "0.25,0.65" -> the full cut list [0.0, 0.25, 0.65, 1.0] (empty text: [0.0, 1.0])."""
    inner = [float(x) for x in str(text).split(",") if x.strip()]
    if any(not 0.0 < c < 1.0 for c in inner) or any(b <= a for a, b in zip(inner, inner[1:])):
        raise ValueError(f"cuts are increasing and inside (0, 1): {text!r}")
    return [0.0] + inner + [1.0]


def default_soft(cuts: List[float]) -> float:
    """min(0.06, round(0.25 x the narrowest stage's width, 3))."""
    width = min(b - a for a, b in zip(cuts, cuts[1:]))
    return min(SOFT_CAP, round(0.25 * width, 3))


def train_ranges(cuts: List[float], soft: float) -> List[tuple]:
    """Each stage's training range: its window widened by ``soft`` on its inner edges, the ends rounded to 10 decimals."""
    out = []
    for lo, hi in zip(cuts, cuts[1:]):
        a = max(0.0, round(lo - soft, 10)) if lo > 0 else 0.0
        b = min(1.0, round(hi + soft, 10)) if hi < 1 else 1.0
        out.append((a, b))
    return out


def stage_steps(cuts: List[float], total: int, scale: float) -> List[int]:
    """Each stage's steps: round(width x total), times scale, at least 2."""
    return [steps_of(int(round((hi - lo) * total)), scale) for lo, hi in zip(cuts, cuts[1:])]


def _cluster_like(stem_folder: str, field, depth_body: int, dim_default: int) -> Build:
    """A cluster build (anchor, stem, [body], head) whose anchor and stem entries are those of a finished build folder,
    so its anchor and stem can be adopted."""
    src = read_build(stem_folder)
    sspec = src.entry("stem").spec
    aspec = src.entry("anchor").spec
    g = src.geometry
    return cluster_build(field, stem_depth=int(sspec["depth"]), body_depth=depth_body, dim=int(g.get("dim", dim_default)),
                         heads=int(g["heads"]), grid=int(g["grid"]), anchor_out=bool(aspec.get("output", True)),
                         anchor_in=int(sspec.get("anchor_in", 0)), anchor_symbols=int(aspec.get("symbols", 0) or 0),
                         class_window=sspec.get("class_window"))


def _born_like(stem_folder: str, bed, seed: int, body_depth: int, dim: int) -> Model:
    """A newborn cluster of ``_cluster_like`` with the source's anchor and stem adopted (hash checked)."""
    build = _cluster_like(stem_folder, bed.field_shape, body_depth, dim)
    symbols = int(build.entry("anchor").spec.get("symbols", 0) or 0)
    model = Model.born(build, seed, bed.spectrum, class_stats=bed.class_stats() if symbols else None)
    adopt(model, stem_folder, ["anchor", "stem"])
    return model


def _thaw_base(job: str, base_model: str = "") -> Dict[str, Any]:
    """The base of a thaw pass: a finished tails or grow job -> {"variant", "folder" (base_model, default avg_g8 for
    tails and model for grow), "kind" (clusters or model), "trained" (the units the base job trained, in order), "cuts",
    "soft", "clusters" (the cluster folders of a staged base)}. ValueError when the folder is not such a job."""
    path = os.path.join(job, "result.json")
    if not os.path.exists(path):
        raise ValueError(f"{job} holds no result.json (a finished tails or grow job is needed)")
    with open(path, encoding="utf-8") as fh:
        r = json.load(fh)
    if r.get("experiment") != EXPERIMENT or r.get("variant") not in ("tails", "grow"):
        raise ValueError(f"{job} is the job {r.get('experiment')!r} / {r.get('variant')!r}, not a stages tails or grow job")
    folder = base_model or ("avg_g8" if r["variant"] == "tails" else "model")
    full = os.path.join(job, folder)
    staged = os.path.exists(os.path.join(full, STAGED_FILE))
    if not staged and not os.path.exists(os.path.join(full, "build.json")):
        raise ValueError(f"{full} holds neither a staged file nor a build")
    trained: List[str] = []
    for p in r["parts"]:
        tr = p["trainable"]
        for x in ([tr] if isinstance(tr, str) else tr):
            if x not in trained:
                trained.append(x)
    info: Dict[str, Any] = {"variant": r["variant"], "folder": folder, "kind": "clusters" if staged else "model",
                            "trained": trained, "cuts": [float(c) for c in r["cuts"]], "soft": float(r["soft"]),
                            "clusters": []}
    if staged:
        d = read_staged(full)
        info.update(cuts=[float(c) for c in d["cuts"]], soft=float(d["soft"]), clusters=list(d["clusters"]))
    return info


def main(argv=(), **overrides) -> dict:
    """Run one variant of one seed; writes the outputs into ``out`` and returns the result."""
    a = parse(DEFAULTS, argv, **overrides)
    with det_mode(a["det"]) as det:
        return _run(a, det)


def _run(a: dict, det) -> dict:
    variant = a["variant"]
    if variant not in ALL_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; known: {list(ALL_VARIANTS)}")
    for k in ("data", "out"):
        if not a[k]:
            raise ValueError(f"{k}= is required")
    if variant == "thaw":
        if not a["base"]:
            raise ValueError("thaw needs base=<the folder of a finished tails or grow job>")
        if a["thaw"] not in THAW_LEVELS:
            raise ValueError(f"thaw= is one of {list(THAW_LEVELS)}, not {a['thaw']!r}")
        if a["cuts"] or float(a["soft"]) >= 0:
            raise ValueError("a thaw pass keeps its base's cuts and crossfade; cuts= and soft= do not apply")
        thaw_info = _thaw_base(a["base"], a["base_model"])        # refused before anything is written
    elif a["base"] or a["base_model"] or a["thaw"]:
        raise ValueError("base=, base_model= and thaw= apply to the thaw variant")
    else:
        thaw_info = None
    if variant == "copies" and not a["source"]:
        raise ValueError("copies needs source=<the folder of a finished source job of the same seed>")
    if variant in ("tails", "grow") and not a["stem_from"]:
        raise ValueError(f"{variant} needs stem_from=<the folder of a finished job whose stem is used>")
    if int(a["score"]) and a["judge"] not in JUDGES:
        raise ValueError(f"unknown judge {a['judge']!r}; known: {list(JUDGES)}")
    if int(a["reader"]) and not (int(a["score"]) and a["ref"]):
        raise ValueError("reader=1 needs score=1 and ref=<the reference file>")
    avg_text = a["avg"] or DEFAULT_AVG[variant]
    averages = [] if avg_text == "none" else parse_averages(avg_text)
    if variant == "grow" and (len(averages) != 1 or average_spec(averages[0][2])[1] != "g"):
        raise ValueError("grow installs one g average: avg= names exactly one g<number> (default g8)")
    to_score = scored_keys(a, averages)
    if variant == "thaw" and not str(a["score_avg"]).strip():     # the judged reading of a thaw pass: its tail average
        to_score = {k for k in to_score if k == "tail100"} or to_score
    cuts = parse_cuts(a["cuts"])
    if variant in ("source", "grow") and len(cuts) > 2:
        raise ValueError(f"{variant} trains one model over the whole range; cuts= applies to copies and tails")
    soft = float(a["soft"]) if float(a["soft"]) >= 0 else default_soft(cuts)
    if thaw_info is not None:                                     # the base's own stages
        cuts, soft = list(thaw_info["cuts"]), float(thaw_info["soft"])
    total = int(a["total"]) or DEFAULT_TOTAL[variant]
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

    seed, dim, scale = int(a["seed"]), int(a["dim"]), float(a["scale"])
    ranges = train_ranges(cuts, soft)
    if variant == "source":
        steps_plan = [("model", steps_of(total, scale))]
    elif variant == "thaw":
        steps_plan = [("thaw", steps_of(total, scale))]
    elif variant == "grow":
        steps_plan = [(f"block_{i}", steps_of(int(a["steps_each"]), scale)) for i in range(int(a["blocks"]))]
    else:
        steps_plan = [(f"stage_{s}", n) for s, n in enumerate(stage_steps(cuts, total, scale))]
    total_steps = sum(n for _, n in steps_plan)
    log(f"JOB {EXPERIMENT} {variant} seed {seed}: {len(steps_plan)} part(s), steps {[n for _, n in steps_plan]} "
        f"({total_steps} in all); cuts {cuts}, soft {soft:g}; on {device}; torch {torch.__version__}; out {out}")
    t_job = time.time()
    bed = PicturesBed(a["data"], device, grid=16)
    data_block = {"bed": "pictures", "source": os.path.basename(a["data"]), "content_hash": npz_content_hash(a["data"]),
                  "heldout": f"the first {a['heldout']} test pictures"}
    common = dict(lr=float(a["lr"]), warmup=int(a["warmup"]), batch=int(a["batch"]), cond_drop=0.1,
                  heldout=int(a["heldout"]), device=device, log=log, log_every=int(a["log_every"]))
    field = bed.field_shape
    heldout = int(a["heldout"])
    entries = tuple(d for _, _, d in averages)
    done_steps = [0]

    def progress(name: str, n: int) -> None:
        done_steps[0] += n
        spent = time.time() - t_job
        left = spent / max(1, done_steps[0]) * (total_steps - done_steps[0])
        log(f"JOB: part {name} done; {done_steps[0]}/{total_steps} steps; spent {spent / 60:.1f} min; "
            f"left about {left / 60:.1f} min")

    def export(model, folder: str, res: dict, trained, base: Optional[str] = None, rec: Optional[dict] = None,
               marks: Optional[dict] = None) -> None:
        export_build(model, os.path.join(out, folder), seed=seed, base=base, trained=trained,
                     recipe=rec or res["recipe"], data=data_block, marks=marks or _marks(res))

    def part_row(name, folder, res, start=0):
        row = _part_row(name, folder, res)
        row["start_step"] = int(start)
        return row

    parts: List[dict] = []
    steps_block: Optional[List[dict]] = None
    as_source = None
    staged_averages: List[tuple] = []
    thaw_block: Optional[dict] = None
    torch.manual_seed(seed)
    if variant == "source":
        name, n = steps_plan[0]
        model = Model.born(cluster_build(field, stem_depth=7, dim=dim), seed, bed.spectrum)
        res = fit_units(model, bed, seed, steps=n, t_sampling="uniform", expect_anchor_at_birth=True, averages=entries,
                        keep_state=True, **common)
        export(model, "model", res, _non_anchor(model))
        if averages:
            export_averages(model, res, averages, out, lambda m, f, rec: export_build(
                m, os.path.join(out, f), seed=seed, trained=_non_anchor(m), recipe=rec, data=data_block,
                marks={"loss_over_levels": None, "frozen_unchanged": res["frozen_unchanged"]}))
        write_state(os.path.join(out, "state"), res["optimizer_state"],
                    extra={"experiment": EXPERIMENT, "variant": variant, "seed": seed, "steps": n,
                           "averages": [k for k, _, _ in averages]})
        parts.append(part_row(name, "model", res))
        progress(name, n)
        final, kind = load_job(out, device=device, kind="model")
        avg_folders = [(k, f) for k, f, _ in averages]
    elif variant in ("copies", "tails"):
        folders = [f"stage_{s}" for s in range(len(steps_plan))]
        if variant == "copies":
            src_model = a["source_model"] or "model"
            base = os.path.join(a["source"], src_model)
            as_source = source_read(a["source"], src_model, bed, heldout, device, log)
            state_dir = os.path.join(a["source"], "state")
            src_state = read_state(state_dir, device=device) if os.path.exists(os.path.join(state_dir, "state.json")) else None
            if src_state is None and not int(a["fresh"]):
                raise ValueError(f"{a['source']} holds no trainer state (state/); a continuing copy needs one")
            if src_state is not None:
                src_step = int(src_state["step"])
            else:
                with open(os.path.join(a["source"], "result.json"), encoding="utf-8") as fh:
                    src_step = sum(int(p["steps"]) for p in json.load(fh)["parts"])
            carry, carried_note = {}, {}
            for key, _, d in averages:
                if isinstance(d, str) and average_spec(d)[1] == "g":
                    f = os.path.join(a["source"], "model_avg_" + key)
                    if os.path.isdir(f):
                        m_avg = link(f, device=device)
                        carry[key] = {n_: p.detach().clone() for n_, p in m_avg.named_parameters()}
                        carried_note[key] = {"folder": "model_avg_" + key, "k_from": src_step}
                        log(f"CARRY the average {key} from {f} (k from {src_step})")
                    else:
                        log(f"NOTE the source holds no model_avg_{key}; the average {key} starts at the copy's start")
        else:
            stem_model = a["stem_model"] or "model"
            base = os.path.join(a["stem_from"], stem_model)
            as_source = source_read(a["stem_from"], stem_model, bed, heldout, device, log)
            carry, carried_note, src_step = {}, {}, 0
        for s, ((name, n), f) in enumerate(zip(steps_plan, folders)):
            rng = ranges[s]
            if variant == "copies":
                c = link(base, device=device)
                trainable, expect = "all", False
                start = 0 if int(a["fresh"]) else src_step
                kw = dict(start_step=start, optimizer_state=None if int(a["fresh"]) else src_state,
                          average_start={k: v for k, v in carry.items()} or None, average_step=src_step)
            else:
                c = _born_like(base, bed, seed, TAIL_DEPTH, dim)
                trainable, expect, start = ["body", "head"], True, 0
                kw = {}
            log(f"PART {name}: t in [{rng[0]:g}, {rng[1]:g}), {n} steps" + (f", from global step {start}" if start else ""))
            res = fit_units(c, bed, seed, steps=n, trainable=trainable, t_range=rng, expect_anchor_at_birth=expect,
                            averages=entries, **kw, **common)
            tr = _non_anchor(c) if trainable == "all" else trainable
            export(c, f, res, tr, base=base)
            for key, _, d in averages:
                export_averages(c, res, [(key, os.path.join("avg_" + key, f), d)], out,
                                lambda m, ff, rec: export_build(
                                    m, os.path.join(out, ff), seed=seed, base=base, trained=tr, recipe=rec,
                                    data=data_block, marks={"loss_over_levels": None,
                                                            "frozen_unchanged": res["frozen_unchanged"]}),
                                carried=carried_note)
            row = part_row(name, f, res, start)
            if variant == "copies":
                row["averages_carried"] = sorted(carry)
            parts.append(row)
            progress(name, n)
        info = {"experiment": EXPERIMENT, "variant": variant, "seed": seed}
        write_staged(out, folders, cuts, soft, info=info)
        for key, _, _ in averages:
            write_staged(os.path.join(out, "avg_" + key), folders, cuts, soft, info=dict(info, average=key))
        final, kind = load_job(out, device=device, kind="clusters")
        avg_folders = [(k, "avg_" + k) for k, _, _ in averages]
    elif variant == "thaw":
        name, n = steps_plan[0]
        base_dir = os.path.join(a["base"], thaw_info["folder"])
        as_source = source_read(a["base"], thaw_info["folder"], bed, heldout, device, log)
        clustered = thaw_info["kind"] == "clusters"
        model = link_staged(base_dir, device=device) if clustered else link(base_dir, device=device)
        level = a["thaw"]
        params = None
        if level == "all":
            trainable: Any = "all"
        else:
            trainable = list(thaw_info["trained"])
            if level in ("1", "2"):
                stem = (model.clusters[0] if clustered else model).unit("stem")
                depth, k = len(stem.blocks), int(level)
                if k > depth:
                    raise ValueError(f"thaw={level} asks for more blocks than the stem's {depth}")
                params = {"stem": [f"blocks.{i}." for i in range(depth - k, depth)]}
        stream_seed = seed + GROW_STREAM * THAW_STREAM
        log(f"PART {name}: the {thaw_info['variant']} build {base_dir}, thaw {level} (trainable {trainable}"
            + (f", and {params}" if params else "") + f"); t in [0, 1), {n} steps; a new optimizer; the stream of "
            f"seed {stream_seed}")
        res = fit_units(model, bed, stream_seed, steps=n, trainable=trainable, trainable_params=params,
                        t_range=(0.0, 1.0), expect_anchor_at_birth=False, averages=entries, **common)
        units_all = [u for c in (model.clusters if clustered else [model]) for u in c.units]
        if trainable == "all":
            tr = sorted({u.name for u in units_all if u.unit_type != "Anchor"})
        else:
            tr = list(trainable) + (["stem"] if params else [])
        row = part_row(name, list(thaw_info["clusters"]) if clustered else "model", res)
        row["seconds_per_step"] = res["train_seconds"] / max(1, n)
        row["stream_seed"] = stream_seed
        parts.append(row)
        progress(name, n)
        avg_marks = {"loss_over_levels": None, "frozen_unchanged": res["frozen_unchanged"]}
        if clustered:
            folders = list(thaw_info["clusters"])
            for c, rel in zip(model.clusters, folders):
                export(c, rel, res, tr, base=os.path.join(base_dir, rel))
            info = {"experiment": EXPERIMENT, "variant": variant, "seed": seed, "thaw": level}
            write_staged(out, folders, cuts, soft, info=info)
            for key, _, d in averages:
                rec = average_recipe(res["recipe"], d)
                with use_average(model, res["averaged"][d]):
                    for c, rel in zip(model.clusters, folders):
                        export_build(c, os.path.join(out, "avg_" + key, rel), seed=seed, base=os.path.join(base_dir, rel),
                                     trained=tr, recipe=rec, data=data_block, marks=avg_marks)
                write_staged(os.path.join(out, "avg_" + key), folders, cuts, soft, info=dict(info, average=key))
            final, kind = load_job(out, device=device, kind="clusters")
            avg_folders = [(k, "avg_" + k) for k, _, _ in averages]
        else:
            export(model, "model", res, tr, base=base_dir)
            if averages:
                export_averages(model, res, averages, out, lambda m, f, rec: export_build(
                    m, os.path.join(out, f), seed=seed, base=base_dir, trained=tr, recipe=rec, data=data_block,
                    marks=avg_marks))
            final, kind = load_job(out, device=device, kind="model")
            avg_folders = [(k, f) for k, f, _ in averages]
        thaw_block = {"level": level, "base": a["base"], "base_model": thaw_info["folder"],
                      "base_variant": thaw_info["variant"], "stream_seed": stream_seed, "trainable": trainable,
                      "trainable_params": params, "trained_units": tr}
    else:                                                                  # grow
        stem_model = a["stem_model"] or "model"
        base = os.path.join(a["stem_from"], stem_model)
        as_source = source_read(a["stem_from"], stem_model, bed, heldout, device, log)
        key, _, entry = averages[0]
        first = _born_like(base, bed, seed, 0, dim)
        b0 = first.build
        anchor_e, stem_e, head_e = b0.entry("anchor"), b0.entry("stem"), b0.entry("head")
        anchor_u, stem_u, head_u = first.unit("anchor"), first.unit("stem"), first.unit("head")
        blocks: List[Any] = []
        block_entries: List[UnitEntry] = []
        prev = first.to(device)
        steps_block = []
        frozen_ok = True
        res = None
        for i, (name, n) in enumerate(steps_plan):
            e = UnitEntry(name, "DepthBlock", {"site": "trunk", "slot": 2 + i}, {})
            unit = born("DepthBlock", seed, name, e.address, e.spec, b0.geometry)
            block_entries.append(e)
            blocks.append(unit)
            build = Build(name=b0.name, geometry=dict(b0.geometry), units=[anchor_e, stem_e] + block_entries + [head_e],
                          info=dict(b0.info))
            model = Model(build, [anchor_u, stem_u] + blocks + [head_u]).to(device)
            gap = born_gap(model, prev)
            log(f"PART {name}: a silent block between the last block and the head (birth gap {gap!r}); {n} steps")
            if gap != 0.0:
                raise AssertionError(f"the newborn block {name} changes the model's output: {gap}")
            hashes_before = {u.name: content_hash(u) for u in model.units if u.name not in (name, "head")}
            stream_seed = seed + GROW_STREAM * i                          # each growth step sees batches of its own
            res = fit_units(model, bed, stream_seed, steps=n, trainable=[name, "head"], expect_anchor_at_birth=(i == 0),
                            averages=(entry,), **common)
            frozen_ok = frozen_ok and bool(res["frozen_unchanged"])
            last = {"loss_over_levels": res["loss_over_levels"], "by_level": _levels(res), "by_stage": by_stage(res, CUTS)}
            named = dict(model.named_parameters())
            with torch.no_grad():                                         # the averaged values become the weights
                for pn, v in res["averaged"][entry].items():
                    named[pn].copy_(v.to(device=named[pn].device, dtype=named[pn].dtype))
            r_avg = read(model, bed, heldout, device)
            hashes = {u.name: content_hash(u) for u in model.units}
            steps_block.append({"block": name, "birth_gap": gap, "steps": n, "stream_seed": stream_seed, "last": last,
                                "averaged": {"key": key, "loss_over_levels": r_avg["loss_over_levels"],
                                             "by_level": _levels(r_avg), "by_stage": by_stage(r_avg, CUTS)},
                                "hashes": hashes,
                                "earlier_unchanged": all(hashes[k] == v for k, v in hashes_before.items())})
            log(f"GROW {name}: last weights {res['loss_over_levels']:.4f}; averaged {key} installed "
                f"{r_avg['loss_over_levels']:.4f}")
            parts.append(part_row(name, "model", res))
            progress(name, n)
            prev = model
        rec = average_recipe(res["recipe"], entry)
        rec["grow"] = {"blocks": len(blocks), "steps_each": steps_plan[0][1], "installed": key,
                       "note": "each block and the head trained with a new optimizer; their averaged values installed"}
        export_build(prev, os.path.join(out, "model"), seed=seed, base=base, trained=[u.name for u in blocks] + ["head"],
                     recipe=rec, data=data_block, marks={"loss_over_levels": None, "frozen_unchanged": frozen_ok})
        final, kind = load_job(out, device=device, kind="model")
        avg_folders = []

    log("READ the assembled model")
    whole = read(final, bed, heldout, device)
    cn = class_numbers(whole)
    result: Dict[str, Any] = {
        "experiment": EXPERIMENT, "variant": variant, "seed": seed, "kind": kind, "cuts": list(cuts), "soft": soft,
        "total": total if variant != "grow" else None, "parts": parts,
        "numbers": {"all": _unique_numbers(final), "trained": sum(p["trained_numbers"] for p in parts)},
        "train_seconds": sum(p["train_seconds"] for p in parts), "read": whole,
        "loss_over_levels": whole["loss_over_levels"], "by_level": _levels(whole), "by_stage": by_stage(whole, CUTS),
        "by_class": cn["by_class"], "class_worth": cn["class_worth"], "score": None, "averaged": {},
        "as_source": as_source}
    if steps_block is not None:
        result["steps"] = steps_block
    if thaw_block is not None:
        result["thaw"] = thaw_block
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
                judged["fresh"] = False
            log(f"SCORE{tag} {sc['score']:.4f} on {n_score} pictures ({sc['judge']})"
                + (f"; class read {sc['class_read']['accuracy']:.4f}" if rdr is not None else ""))
            return sc
        except Exception as e:
            log(f"SCORE{tag} not computed: {e!r}")
            return {"error": repr(e)}

    if int(a["score"]) and int(a["score_last"]):
        result["score"] = score_of(final)
    for key, folder in avg_folders:
        avg_model, _ = load_job(os.path.join(out, folder), device=device) if folder.startswith("avg_") else \
            (link(os.path.join(out, folder), device=device), "model")
        log(f"READ the average {key} ({folder})")
        r = read(avg_model, bed, heldout, device)
        c = class_numbers(r)
        scored = int(a["score"]) and key in to_score
        result["averaged"][key] = {"loss_over_levels": r["loss_over_levels"], "by_level": _levels(r),
                                   "by_stage": by_stage(r, CUTS), "by_class": c["by_class"],
                                   "class_worth": c["class_worth"],
                                   "score": score_of(avg_model, f" (average {key})") if scored else None}
    if int(a["sheet"]):
        sample_sheet(final, bed, os.path.join(out, "samples.png"), device)
    args = {k: v for k, v in recorded_args(a).items() if not (k in STAGE_ARGS and v == STAGE_ARGS[k])}
    result.update({"args": args, "torch": torch.__version__, "device": str(device), "version": __version__})
    det.record(result)
    with open(os.path.join(out, "result.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=1)
        fh.write("\n")
    bs = ", ".join("-" if x is None else f"{x:.4f}" for x in result["by_stage"])
    log(f"RESULT {variant} seed {seed}: {len(cuts) - 1} stage(s); over ten levels {result['loss_over_levels']:.4f}; "
        f"by stage {bs}" + "".join(f"; average {k} {v['loss_over_levels']:.4f}" for k, v in result["averaged"].items())
        + f"; {(time.time() - t_job) / 60:.1f} min")
    log("done")
    return result


def _fmt(x, nd: int = 4) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def summarize(root: str) -> str:
    """A markdown table of every finished stages job under ``root``: one row per reading (the last weights, then each
    average) with the variant, seed, stage count, cuts, total steps, the total and by-stage losses and the score."""
    rows = []
    for dirpath, _, files in os.walk(root):
        if "result.json" in files:
            with open(os.path.join(dirpath, "result.json"), encoding="utf-8") as fh:
                r = json.load(fh)
            if r.get("experiment") == EXPERIMENT:
                rows.append(r)
    rows.sort(key=lambda r: (r["variant"], r["seed"], len(r["cuts"]), str(r["cuts"])))
    lines = ["| variant | seed | K | cuts | total | reading | over ten levels | by stage | score |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        k = len(r["cuts"]) - 1
        cuts = " / ".join(f"{c:g}" for c in r["cuts"][1:-1]) or "-"
        total = sum(int(p["steps"]) for p in r["parts"])
        readings = [("last", r)] + [(key, v) for key, v in (r.get("averaged") or {}).items()]
        for name, v in readings:
            sc = (v.get("score") or {}).get("score")
            stages = " / ".join(_fmt(x) for x in v["by_stage"])
            lines.append(f"| {r['variant']} | {r['seed']} | {k} | {cuts} | {total:,} | {name} | "
                         f"{_fmt(v['loss_over_levels'])} | {stages} | {_fmt(sc, 2)} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__" and not in_notebook():
    main(sys.argv[1:])
