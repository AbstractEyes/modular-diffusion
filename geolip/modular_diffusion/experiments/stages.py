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
| parts   | parts added to a finished tree of stage copies (``base``, a copies job; ``base_model``, default avg_tail100, read again first; its cuts and crossfade kept); per stage copy the parts of ``parts`` (below), trained alone on that stage's range for its share of ``total`` (default 2,000) with a NEW optimizer, the warm-up from step 0 and the stream of ``seed + 100003 x 13``; ``only_classes`` restricts the stream (the base's restriction is not inherited); averages default g8, the judged reading (the last weights are read, not scored, unless score_last=1). ``init_from`` (a parts job of the same spec; ``init_model``, default avg_g8): the parts start from that job's weights, stage by stage (a graft; the parents may differ); with init_from, total=0 trains nothing | the new parts only (a row: that row only) |

The eighth pass. ``source`` takes ``class_window`` ("" the plain build; blind: the stem never reads the class; lo,hi:
that window as written), ``without_classes`` (a comma list: the training stream leaves those classes out; the held-out
read keeps every class) and ``parts`` (experts10 or shared1, or expert:<c>: born with the model, silent, everything but
the anchor trained together from step 0). ``copies`` inherits its source's training-stream restriction, continues the
parts a source holds, and takes ``parts`` (born at the fork: with ``forms.with_parts``, the source's trainer state
continued with the parts starting without moments, at the continued run's rate; refused on a source with parts and with
fresh=1) and ``stream`` (j > 0: the batches of the stream of ``seed + 100003 x (10 + j)`` at the same global steps, a
sibling tree). Jobs with parts train without member gradient equalization (as the content experiment's parts do).

| parts spec       | what each stage copy (or the source) gains                                                       |
|------------------|---------------------------------------------------------------------------------------------------|
| experts10        | ten class-gated parts expert_c0 .. expert_c9, identity-born, at every noise level                 |
| shared1          | one such part serving all ten classes (shared_c0-1-2-3-4-5-6-7-8-9)                               |
| expert:<c>       | one such part for class c                                                                         |
| row:<c>          | no unit: row c of the stem's class table, set to the no-class row at birth and trained alone (``fit_units(trainable_rows=)``) |
| row+expert:<c>   | both, trained together                                                                            |

With parts a result also holds parts_spec, part_units, parts_off (the read with every part left out, a row put back to
the base's values: loss_over_levels, by_level, by_stage, class_worth; for parts also equals_base, against the base's
stored read), exact (source, copies, and parts with experts10 / shared1: one part left out, content.py's form; the
other parts specs: by class, others_max_abs over the classes the parts do not serve and own), under each judged
average parts_off (and exact for source and copies), birth_gap / row_birth_gap, graft ({"from", "model",
"parents_equal"} per stage) and data (the training-stream restriction). Every part row of source, copies and parts
holds seconds_per_step and peak_memory_mib.

Every result reads the model on the registered stages (the time split's cuts 0.25 / 0.65 in ``by_stage``), whatever
the model's cuts.

Arguments (key=value): those of the time split, plus cuts, soft (negative: the default rule), total (0: the variant's
default), source, source_model, stem_from, stem_model, fresh, blocks (3), steps_each (2,000; times ``scale``), avg (a
comma list of decays and names; empty: the variant's default; ``none``: no average), score_last, score_avg, reader
(the class read on the scored samples; needs score=1 and ref=), det, and for thaw base, base_model and thaw (tail, 1,
2 or all); class_window, without_classes and parts (source), parts, stream and without_classes (copies: "" inherits the
source's restriction, none trains on every class, a comma list restricts to that list; recorded in data and args),
base, base_model, parts, only_classes, init_from, init_model and init_stage (parts: -1 takes the init job's parts
stage by stage, k >= 0 its stage k's parts for every stage of the base). A thaw job keeps its base's cuts and crossfade, writes the folders of its base's kind (stage_0 .. with
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

import contextlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch

from .. import __version__
from ..beds import PicturesBed
from ..build import (Build, UnitEntry, content_hash, read_build, read_card, read_weights, state_hash, unit_files,
                     write_card)
from ..config import in_notebook, parse
from ..data import npz_content_hash
from ..determinism import mode as det_mode
from ..forms import add_class_experts, adopt, cluster_build, with_parts, without_units
from ..linker import born_gap, export_build, link
from ..model import Model
from ..read import CHUNK, LEVEL_ITEMS, READ_SEED, T_GRID, by_stage, read, read_symbols
from ..sample import sample_sheet
from ..score import class_reader, make_judge, picture_score, reader_check, reference_classes, reference_stats
from ..staged import STAGED_FILE, Staged, link_staged, read_staged, write_staged
from ..train import average_spec, fit_units, read_state, use_average, write_state
from ..units import NULL_CLASS, born
from .content import class_numbers, exactness
from .time_split import (CUTS, JUDGES, _levels, _marks, _non_anchor, _part_row, _unique_numbers, average_recipe,
                         export_averages, load_job, parse_averages, recorded_args, scored_keys, source_read, steps_of,
                         stored_key)

EXPERIMENT = "stages"
VARIANTS = ("source", "copies", "tails", "grow")             # the variants of earlier versions
ALL_VARIANTS = VARIANTS + ("thaw", "parts")
DEFAULT_TOTAL = {"source": 4000, "copies": 2000, "tails": 6000, "grow": 0, "thaw": 2000, "parts": 2000}
DEFAULT_AVG = {"source": "g8", "copies": "g8,tail100", "tails": "g8", "grow": "g8", "thaw": "g8,tail100",
               "parts": "g8"}
SOFT_CAP = 0.06
TAIL_DEPTH = 3
GROW_STREAM = 100003                 # growth step i trains on the stream of seed + GROW_STREAM x i (step 0: the job's seed)
THAW_STREAM = 7                      # a thaw pass trains on the stream of seed + GROW_STREAM x 7 (no build's batches)
SIBLING_STREAM = 10                  # copies with stream=j > 0 train on the stream of seed + GROW_STREAM x (10 + j)
PARTS_STREAM = 13                    # parts added to a finished tree train on the stream of seed + GROW_STREAM x 13
THAW_LEVELS = ("tail", "1", "2", "all")
PART_SPECS = ("experts10", "shared1", "expert:<c>", "row:<c>", "row+expert:<c>")
UNIT_PART_TYPES = ("ContentExpert", "BandMember")
STAGE_ARGS = {"cuts": "", "soft": -1.0, "total": 0, "fresh": 0, "blocks": 3, "steps_each": 2000, "base": "",
              "base_model": "", "thaw": "", "class_window": "", "without_classes": "", "parts": "", "stream": 0,
              "only_classes": "", "init_from": "", "init_model": "", "init_stage": -1}

# score_last -1: the variant's default (1; 0 for parts, whose judged reading is its g8 tree)
DEFAULTS = dict(variant="", seed=0, data="", out="", device="cuda", heldout=2048, batch=128, lr=1e-3, warmup=100,
                log_every=500, threads=4, mem_fraction=0.25, scale=1.0, dim=128, score=0, judge="inception",
                score_n=10000, score_steps=50, ref="", sheet=1, source="", source_model="", stem_from="", stem_model="",
                cuts="", soft=-1.0, total=0, fresh=0, blocks=3, steps_each=2000, avg="", score_last=-1, score_avg="",
                reader=0, det="", base="", base_model="", thaw="", class_window="", without_classes="", parts="",
                stream=0, only_classes="", init_from="", init_model="", init_stage=-1)


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


def parse_parts(spec: str) -> Dict[str, Any]:
    """A ``parts=`` spec -> {"spec", "groups" (``forms.with_parts`` groups, in order), "row" (the class whose row of
    the stem's class table is a part, or None), "classes" (the classes the parts serve)}. Specs (PART_SPECS): experts10
    (ten class-gated identity-born parts, expert_c0 .. expert_c9, at every noise level), shared1 (one such part serving
    all ten classes, shared_c0-1-...-9), expert:<c> (one part for class c), row:<c> (row c of the stem's class table,
    no new unit), row+expert:<c> (both). ValueError for anything else, naming the known specs."""
    text = str(spec).strip()
    every = dict(window=(0.0, 1.0), soft=0.0, depth=1, born="identity")

    def klass(x: str) -> int:
        c = int(x)
        if not 0 <= c < NULL_CLASS:
            raise ValueError(f"a part's class lies in 0 .. {NULL_CLASS - 1}: {spec!r}")
        return c

    try:
        if text == "experts10":
            return {"spec": text, "groups": [dict(classes=[[c] for c in range(10)], **every)], "row": None,
                    "classes": list(range(10))}
        if text == "shared1":
            return {"spec": text, "groups": [dict(classes=[list(range(10))], prefix="shared", **every)], "row": None,
                    "classes": list(range(10))}
        if text.startswith("expert:"):
            c = klass(text[len("expert:"):])
            return {"spec": text, "groups": [dict(classes=[[c]], **every)], "row": None, "classes": [c]}
        if text.startswith("row:"):
            c = klass(text[len("row:"):])
            return {"spec": text, "groups": [], "row": c, "classes": [c]}
        if text.startswith("row+expert:"):
            c = klass(text[len("row+expert:"):])
            return {"spec": text, "groups": [dict(classes=[[c]], **every)], "row": c, "classes": [c]}
    except ValueError as e:
        if "class lies" in str(e):
            raise
    raise ValueError(f"unknown parts spec {spec!r}; known: {list(PART_SPECS)}")


def part_groups(spec: str) -> List[Dict[str, Any]]:
    """The ``forms.with_parts`` groups of a ``parts=`` spec (``parse_parts``; a row spec adds no unit: [])."""
    return parse_parts(spec)["groups"]


def _class_window(text: str):
    """``class_window=`` text -> None ("" : the plain build), ("0", "0") for blind, or (lo, hi) as written."""
    text = str(text).strip()
    if not text:
        return None
    if text == "blind":
        return ("0", "0")
    edges = [x.strip() for x in text.split(",")]
    if len(edges) != 2 or not all(edges):
        raise ValueError(f"class_window is blind or lo,hi: {text!r}")
    float(edges[0]), float(edges[1])
    return (edges[0], edges[1])


def _class_list(text: str) -> tuple:
    """A comma list of class indices -> a tuple of ints (empty text: ())."""
    out = tuple(int(x) for x in str(text).split(",") if x.strip())
    if any(not 0 <= c < NULL_CLASS for c in out):
        raise ValueError(f"classes lie in 0 .. {NULL_CLASS - 1}: {text!r}")
    return out


def _stream_record(train_bed) -> Optional[Dict[str, Any]]:
    """The training stream's restriction as content.py records it (None when it draws every picture)."""
    if train_bed.without_classes:
        return {"without_classes": list(train_bed.without_classes)}
    if train_bed.only_classes is not None:
        return {"only_classes": list(train_bed.only_classes)}
    return None


def _restricted(bed, rec: Optional[Dict[str, Any]]):
    """The bed with a recorded training-stream restriction (``_stream_record``) applied (the bed itself for None)."""
    if not rec:
        return bed
    return bed.restricted(without_classes=rec.get("without_classes", ()), only_classes=rec.get("only_classes"))


def _source_stream(job: str, model_folder: str) -> Optional[Dict[str, Any]]:
    """The training-stream restriction of a finished job: its result's data.training_stream, else the one recorded in
    the cards of its model folder (None: every picture)."""
    path = os.path.join(job, "result.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            rec = ((json.load(fh).get("data") or {}).get("training_stream"))
        if rec:
            return dict(rec)
    folder = os.path.join(job, model_folder)
    if os.path.exists(os.path.join(folder, "build.json")):
        b = read_build(folder)
        for e in b.units:
            card = read_card(unit_files(b.folder, e.name)[1])
            if (card.data or {}).get("training_stream"):
                return dict(card.data["training_stream"])
    return None


def part_units(model) -> List[str]:
    """The names of a model's (or a tree's) part units (content experts and band members), each once, in order."""
    out: List[str] = []
    for c in (getattr(model, "clusters", None) or [model]):
        for u in c.units:
            if u.unit_type in UNIT_PART_TYPES and u.name not in out:
                out.append(u.name)
    return out


def without_parts(model, names: List[str]):
    """The model (or the tree, every cluster blended as the tree is) with the named units left out."""
    clusters = getattr(model, "clusters", None)
    if clusters is None:
        return without_units(model, names)
    return Staged([without_units(c, names) for c in clusters], model.cuts, model.soft)


@contextlib.contextmanager
def rows_set(model, row: int, values: List[torch.Tensor]):
    """Row ``row`` of every cluster's class table set to ``values`` (one per cluster) for the body of the ``with``;
    the values before are put back exactly on exit."""
    stems = [c.unit("stem") for c in (getattr(model, "clusters", None) or [model])]
    kept = [s.cls.weight[row].detach().clone() for s in stems]
    try:
        with torch.no_grad():
            for s, v in zip(stems, values):
                s.cls.weight[row] = v.to(device=s.cls.weight.device, dtype=s.cls.weight.dtype)
        yield model
    finally:
        with torch.no_grad():
            for s, v in zip(stems, kept):
                s.cls.weight[row] = v


def _brief(r: dict) -> Dict[str, Any]:
    """{loss_over_levels, by_level, by_stage, class_worth} of a read."""
    return {"loss_over_levels": r["loss_over_levels"], "by_level": _levels(r), "by_stage": by_stage(r, CUTS),
            "class_worth": class_numbers(r)["class_worth"]}


def _sum_seq(parts: dict, like: torch.Tensor) -> torch.Tensor:
    v = torch.zeros_like(like)
    for p in parts.values():                    # one addition at a time: an exactly zero part changes no digit
        v = v + p
    return v


@torch.no_grad()
def row_errors(model, bed, heldout: int, device, cls_override: Optional[int] = None):
    """The error of every row of the read's fixed noise (the first LEVEL_ITEMS held-out items at the ten levels), the
    parts summed one at a time -> (errors [levels, rows] float64, the rows' classes). ``cls_override``: every row read
    with that class instead of its own."""
    model.eval()
    z1, cls = bed.heldout(heldout)
    gen = torch.Generator(device=device).manual_seed(READ_SEED)
    z0 = torch.randn(z1.shape, generator=gen, device=device)
    n_lv = min(z1.shape[0], LEVEL_ITEMS)
    sym = read_symbols(model, z1.shape[0], device)
    rows = []
    for t in T_GRID:
        per = []
        for i in range(0, n_lv, CHUNK):
            sl = slice(i, min(i + CHUNK, n_lv))
            b = z1[sl].shape[0]
            tt = torch.full((b,), t, device=z1.device)
            zt = (1 - t) * z0[sl] + t * z1[sl]
            c = cls[sl] if cls_override is None else torch.full_like(cls[sl], int(cls_override))
            kw = {} if sym is None else {"symbols": sym[sl]}
            p, _, _, _ = model.components(zt, tt, c, **kw)
            per.append(((_sum_seq(p, zt) - (z1[sl] - z0[sl])) ** 2).mean(dim=(1, 2, 3)).double())
        rows.append(torch.cat(per))
    return torch.stack(rows), cls[:n_lv]


def class_exactness(e_in: torch.Tensor, e_out: torch.Tensor, cls: torch.Tensor, served: List[int]) -> Dict[str, Any]:
    """The parts against no parts, by class, from ``row_errors`` of the two models: "others_max_abs" the largest
    |by-class error with the parts - without them| over the classes the parts do not serve, "rows_others_max_abs" the
    same over single rows of those classes, "own" {class: the difference} on the classes they serve."""
    others, rows_max, own = 0.0, 0.0, {}
    for c in range(10):
        m = cls == c
        if not bool(m.any()):
            continue
        d = float(e_in[:, m].mean(dim=1).mean() - e_out[:, m].mean(dim=1).mean())
        if c in served:
            own[str(c)] = d
        else:
            others = max(others, abs(d))
            rows_max = max(rows_max, float((e_in[:, m] - e_out[:, m]).abs().max()))
    return {"form": "by class", "classes": list(served), "others_max_abs": others, "rows_others_max_abs": rows_max,
            "own": own}


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


def _parts_base(job: str, base_model: str = "") -> Dict[str, Any]:
    """The base of a parts job: a finished stages copies job -> {"folder" (base_model, default avg_tail100), "cuts",
    "soft", "clusters", "stored" (the job's stored read of that folder: loss_over_levels, by_level)}. ValueError when
    the folder is not such a job or holds no staged file."""
    path = os.path.join(job, "result.json")
    if not os.path.exists(path):
        raise ValueError(f"{job} holds no result.json (a finished stages copies job is needed)")
    with open(path, encoding="utf-8") as fh:
        r = json.load(fh)
    if r.get("experiment") != EXPERIMENT or r.get("variant") != "copies":
        raise ValueError(f"{job} is the job {r.get('experiment')!r} / {r.get('variant')!r}, not a stages copies job")
    folder = base_model or "avg_tail100"
    full = os.path.join(job, folder)
    if not os.path.exists(os.path.join(full, STAGED_FILE)):
        raise ValueError(f"{full} holds no staged file (a tree of stage copies is needed)")
    d = read_staged(full)
    key = stored_key(folder)
    stored = r if key is None else (r.get("averaged") or {}).get(key) or {}
    return {"folder": folder, "cuts": [float(c) for c in d["cuts"]], "soft": float(d["soft"]),
            "clusters": list(d["clusters"]),
            "stored": {"loss_over_levels": stored.get("loss_over_levels"), "by_level": stored.get("by_level")}}


def _parts_init(job: str, init_model: str, pspec: Dict[str, Any], n_clusters: int,
                init_stage: int = -1) -> Dict[str, Any]:
    """Where a graft's parts come from: a finished parts job of the same parts spec -> {"dir", "model" (default
    avg_g8), "clusters" (the init folder's cluster for each stage of the base), "init_stage"}. init_stage -1: stage by
    stage (the same number of stages); k >= 0: the init job's stage k for every stage of the base. ValueError on
    another spec, another stage count (with -1), a stage the init job does not hold, or no such job."""
    path = os.path.join(job, "result.json")
    if not os.path.exists(path):
        raise ValueError(f"{job} holds no result.json (a finished stages parts job is needed)")
    with open(path, encoding="utf-8") as fh:
        r = json.load(fh)
    if r.get("experiment") != EXPERIMENT or r.get("variant") != "parts":
        raise ValueError(f"{job} is the job {r.get('experiment')!r} / {r.get('variant')!r}, not a stages parts job")
    if r.get("parts_spec") != pspec["spec"]:
        raise ValueError(f"{job} trained the parts {r.get('parts_spec')!r}, not {pspec['spec']!r}")
    model = init_model or "avg_g8"
    full = os.path.join(job, model)
    if not os.path.exists(os.path.join(full, STAGED_FILE)):
        raise ValueError(f"{full} holds no staged file")
    d = read_staged(full)
    k = int(init_stage)
    if k == -1:
        if len(d["clusters"]) != n_clusters:
            raise ValueError(f"{full} holds {len(d['clusters'])} stage copies, the base {n_clusters} "
                             f"(init_stage=<k> takes one stage's parts for every stage)")
        clusters = list(d["clusters"])
    elif 0 <= k < len(d["clusters"]):
        clusters = [d["clusters"][k]] * n_clusters
    else:
        raise ValueError(f"init_stage is -1 or a stage of {full} (0 .. {len(d['clusters']) - 1}), not {k}")
    return {"dir": full, "model": model, "clusters": clusters, "init_stage": k}


def _unit_state(folder: str, name: str) -> Dict[str, torch.Tensor]:
    """A unit's tensors from a build folder, its weight file's content hash checked against its card."""
    b = read_build(folder)
    e = b.entry(name)
    card = read_card(unit_files(b.folder, name)[1])
    state = read_weights(os.path.join(b.folder, e.file) if e.file else unit_files(b.folder, name)[0])
    h = state_hash(e.type, e.address, e.spec, state)
    if h != card.hash or (e.hash is not None and h != e.hash):
        raise ValueError(f"unit {name!r} of {folder}: the weight file's content hash differs from its card")
    return state


def _stem_row(folder: str, row: int) -> torch.Tensor:
    """Row ``row`` of the class table of the stem of a build folder (hash checked)."""
    return _unit_state(folder, "stem")["cls.weight"][row].clone()


def _parents_equal(base_dir: str, init_dir: str, part_names: List[str], row: Optional[int]) -> bool:
    """Whether the units of a base cluster equal those of the cluster the parts came from (content hashes; with a row
    part the stems are compared tensor by tensor, that row left out)."""
    b, i = read_build(base_dir), read_build(init_dir)
    for e in b.units:
        if e.name in part_names:
            continue
        if e.name not in i.names():
            return False
        if row is not None and e.name == "stem":
            sb, si = _unit_state(base_dir, "stem"), _unit_state(init_dir, "stem")
            if set(sb) != set(si):
                return False
            for k, v in sb.items():
                w = si[k]
                if k == "cls.weight":
                    keep = torch.ones(v.shape[0], dtype=torch.bool)
                    keep[row] = False
                    v, w = v[keep], w[keep]
                if not torch.equal(v, w):
                    return False
            continue
        if e.hash != i.entry(e.name).hash:
            return False
    return True


def _graft_cards(folder: str, note: Dict[str, Any], part_names: List[str], untrained: bool) -> None:
    """The cards of a graft's parts in an exported cluster folder. Untrained here: each part's card is the one it was
    trained with (its recorded parents are those it was fit on), with recipe "graft" {"from", "onto": {unit: the
    content hash it now sits on}}. Trained here: the exported card (fit here) with recipe "graft" {"from",
    "first_trained_with": the parents its first card recorded}. A stem whose row was grafted gets "graft" {"from"}."""
    b = read_build(folder)
    hashes = {e.name: e.hash for e in b.units}
    src = note["from"]
    for name in part_names:
        first = read_card(unit_files(src, name)[1])
        dst = unit_files(folder, name)[1]
        if untrained:
            if first.hash != hashes[name]:
                raise AssertionError(f"the grafted part {name} is not the unit of {src}")
            first.recipe = dict(first.recipe or {}, graft={"from": src, "onto": {p: hashes[p] for p in first.parents
                                                                                   if p in hashes}})
            write_card(first, dst)
        else:
            card = read_card(dst)
            card.recipe = dict(card.recipe or {}, graft={"from": src, "first_trained_with": dict(first.parents)})
            write_card(card, dst)
    if "stem" not in part_names and os.path.exists(unit_files(folder, "stem")[1]):
        card = read_card(unit_files(folder, "stem")[1])
        if (card.recipe or {}).get("trainable_rows"):
            card.recipe = dict(card.recipe, graft={"from": src, "trained_here_steps": note["trained_here_steps"]})
            write_card(card, unit_files(folder, "stem")[1])


@torch.no_grad()
def _class_row_gap(model, bed, heldout: int, device, row: int) -> float:
    """The largest change of the velocity on the read's fixed noise (the first LEVEL_ITEMS held-out items at the ten
    levels) when every row is read as class ``row`` instead of with no class (the parts summed one at a time)."""
    model.eval()
    z1, cls = bed.heldout(heldout)
    gen = torch.Generator(device=device).manual_seed(READ_SEED)
    z0 = torch.randn(z1.shape, generator=gen, device=device)
    n_lv = min(z1.shape[0], LEVEL_ITEMS)
    gap = 0.0
    for t in T_GRID:
        for i in range(0, n_lv, CHUNK):
            sl = slice(i, min(i + CHUNK, n_lv))
            tt = torch.full((z1[sl].shape[0],), t, device=z1.device)
            zt = (1 - t) * z0[sl] + t * z1[sl]
            p_c, _, _, _ = model.components(zt, tt, torch.full_like(cls[sl], int(row)))
            p_n, _, _, _ = model.components(zt, tt, torch.full_like(cls[sl], NULL_CLASS))
            gap = max(gap, float((_sum_seq(p_c, zt) - _sum_seq(p_n, zt)).abs().max()))
    return gap


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
    if int(a["score_last"]) < 0:                                  # the variant's default
        a["score_last"] = 0 if variant == "parts" else 1
    class_window = _class_window(a["class_window"])
    wc_text = str(a["without_classes"]).strip()
    without_classes = () if wc_text == "none" else _class_list(wc_text)
    if class_window is not None and variant != "source":
        raise ValueError("class_window= applies to the source variant")
    if wc_text and variant not in ("source", "copies"):
        raise ValueError("without_classes= applies to the source and copies variants")
    if wc_text == "none" and variant != "copies":
        raise ValueError("without_classes=none applies to the copies variant (every class, whatever the source's)")
    pspec = parse_parts(a["parts"]) if str(a["parts"]).strip() else None
    if pspec is not None and variant not in ("source", "copies", "parts"):
        raise ValueError("parts= applies to the source, copies and parts variants")
    if pspec is not None and variant != "parts" and pspec["row"] is not None:
        raise ValueError(f"a row of the class table is a part of the parts variant only, not of {variant}")
    if int(a["stream"]) and variant != "copies":
        raise ValueError("stream= applies to the copies variant")
    if int(a["stream"]) < 0:
        raise ValueError(f"stream= is 0 or a positive integer, not {a['stream']}")
    if variant != "parts" and (a["only_classes"] or a["init_from"] or a["init_model"] or int(a["init_stage"]) != -1):
        raise ValueError("only_classes=, init_from=, init_model= and init_stage= apply to the parts variant")
    parts_info = None
    if variant == "thaw":
        if not a["base"]:
            raise ValueError("thaw needs base=<the folder of a finished tails or grow job>")
        if a["thaw"] not in THAW_LEVELS:
            raise ValueError(f"thaw= is one of {list(THAW_LEVELS)}, not {a['thaw']!r}")
        if a["cuts"] or float(a["soft"]) >= 0:
            raise ValueError("a thaw pass keeps its base's cuts and crossfade; cuts= and soft= do not apply")
        thaw_info = _thaw_base(a["base"], a["base_model"])        # refused before anything is written
    elif variant == "parts":
        if a["thaw"]:
            raise ValueError("thaw= applies to the thaw variant")
        if not a["base"]:
            raise ValueError("parts needs base=<the folder of a finished stages copies job>")
        if pspec is None:
            raise ValueError(f"parts needs parts=<one of {list(PART_SPECS)}>")
        if a["cuts"] or float(a["soft"]) >= 0:
            raise ValueError("parts keeps its base's cuts and crossfade; cuts= and soft= do not apply")
        if int(a["total"]) < 0:
            raise ValueError(f"total= is 0 or more, not {a['total']}")
        if a["init_model"] and not a["init_from"]:
            raise ValueError("init_model= needs init_from=")
        thaw_info = None
        parts_info = _parts_base(a["base"], a["base_model"])      # refused before anything is written
        if a["init_from"]:
            parts_info["init"] = _parts_init(a["init_from"], a["init_model"], pspec, len(parts_info["clusters"]),
                                             int(a["init_stage"]))
        elif int(a["init_stage"]) != -1:
            raise ValueError("init_stage= needs init_from=")
    elif a["base"] or a["base_model"] or a["thaw"]:
        raise ValueError("base=, base_model= and thaw= apply to the thaw variant (base= and base_model= also to parts)")
    else:
        thaw_info = None
    if variant == "copies" and pspec is not None and int(a["fresh"]):
        raise ValueError("parts born at the fork continue the source's run; fresh=1 does not apply")
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
    if parts_info is not None:
        cuts, soft = list(parts_info["cuts"]), float(parts_info["soft"])
    total = int(a["total"]) or DEFAULT_TOTAL[variant]
    if variant == "parts" and a["init_from"] and int(a["total"]) == 0:
        total = 0                                                 # a graft read as it is: no training
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
    elif variant == "parts" and total == 0:
        steps_plan = [(f"stage_{s}", 0) for s in range(len(cuts) - 1)]
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
        if variant in ("source", "copies", "parts"):            # the eighth pass: the cost of a step (peak memory above)
            row["seconds_per_step"] = res["train_seconds"] / max(1, int(res["steps"]))
        for k in ("optimizer_new", "average_new", "trainable_rows", "rows_frozen_equal", "rows_restored"):
            if k in res:
                row[k] = res[k]
        return row

    parts: List[dict] = []
    steps_block: Optional[List[dict]] = None
    as_source = None
    staged_averages: List[tuple] = []
    thaw_block: Optional[dict] = None
    train_bed = bed
    stream_rec: Optional[Dict[str, Any]] = None        # the training stream's restriction (recorded when there is one)
    names_new: List[str] = []                          # the part units of the model (the parts variant: the new ones)
    extra: Dict[str, Any] = {}                         # result keys of the eighth pass, present only when they apply
    graft_tree = False
    torch.manual_seed(seed)
    if variant == "source":
        name, n = steps_plan[0]
        if without_classes:
            train_bed = bed.restricted(without_classes=without_classes)
            stream_rec = _stream_record(train_bed)
            data_block["training_stream"] = stream_rec
        build = cluster_build(field, stem_depth=7, dim=dim, class_window=class_window)
        for g in (pspec["groups"] if pspec else []):
            names_new += add_class_experts(build, **g)
        model = Model.born(build, seed, bed.spectrum)
        fit_kw: Dict[str, Any] = {}
        if names_new:                                  # parts from the first step: silent at birth, no equalization
            gap = born_gap(model, without_units(model, names_new))
            log(f"PARTS {pspec['spec']}: {len(names_new)} part(s) born with the model (birth gap {gap!r})")
            if gap != 0.0:
                raise AssertionError(f"the newborn parts change the model's output: {gap}")
            fit_kw = dict(equalize=False)
            extra["birth_gap"] = gap
        res = fit_units(model, train_bed, seed, steps=n, t_sampling="uniform", expect_anchor_at_birth=True,
                        averages=entries, keep_state=True, **fit_kw, **common)
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
            # the eighth pass: the source's training-stream restriction is inherited; parts the source holds are
            # continued with it; parts born at the fork (parts=); a sibling stream (stream=j)
            stream_rec = _source_stream(a["source"], src_model)
            if wc_text:                                   # the copies' own choice (P4b): none, or a list instead
                inherited = stream_rec
                stream_rec = {"without_classes": list(without_classes)} if without_classes else None
                data_block["training_stream_choice"] = {"without_classes": wc_text, "source": inherited}
                log(f"STREAM the copies' own: without_classes={wc_text} (the source's: {inherited})")
            if stream_rec:
                train_bed = _restricted(bed, stream_rec)
                data_block["training_stream"] = stream_rec
                log(f"STREAM inherited from the source: {stream_rec}")
            src_parts = [e.name for e in read_build(base).units if e.type in UNIT_PART_TYPES]
            if pspec is not None and src_parts:
                raise ValueError(f"the source model already holds parts {src_parts}; parts= adds parts to a source "
                                 f"without any")
            sibling = int(a["stream"])
            copy_seed = seed + GROW_STREAM * (SIBLING_STREAM + sibling) if sibling else seed
        else:
            stem_model = a["stem_model"] or "model"
            base = os.path.join(a["stem_from"], stem_model)
            as_source = source_read(a["stem_from"], stem_model, bed, heldout, device, log)
            carry, carried_note, src_step = {}, {}, 0
            src_parts, copy_seed = [], seed
        for s, ((name, n), f) in enumerate(zip(steps_plan, folders)):
            rng = ranges[s]
            gap_s = None
            if variant == "copies":
                if pspec is not None:                     # parts born at the fork: silent, no moments of their own
                    c, born_names = with_parts(base, seed, pspec["groups"], device=device)
                    gap_s = 0.0                           # with_parts asserts it
                    names_new = list(born_names)
                    log(f"PARTS {pspec['spec']}: {len(born_names)} part(s) born at the fork (birth gap 0.0)")
                else:
                    c = link(base, device=device)
                trainable, expect = "all", False
                start = 0 if int(a["fresh"]) else src_step
                kw = dict(start_step=start, optimizer_state=None if int(a["fresh"]) else src_state,
                          average_start={k: v for k, v in carry.items()} or None, average_step=src_step)
                if pspec is not None:
                    kw["optimizer_partial"] = True
                if src_parts and not int(a["fresh"]):
                    # a part the source held that never took a step (no row of its class in any batch) has no entry
                    # in the source's trainer state: it continues as it was, without moments
                    kw["optimizer_partial"] = True
                if pspec is not None or src_parts:
                    kw["equalize"] = False
            else:
                c = _born_like(base, bed, seed, TAIL_DEPTH, dim)
                trainable, expect, start = ["body", "head"], True, 0
                kw = {}
            log(f"PART {name}: t in [{rng[0]:g}, {rng[1]:g}), {n} steps" + (f", from global step {start}" if start else "")
                + (f"; the stream of seed {copy_seed}" if copy_seed != seed else ""))
            res = fit_units(c, train_bed, copy_seed, steps=n, trainable=trainable, t_range=rng,
                            expect_anchor_at_birth=expect, averages=entries, **kw, **common)
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
                if copy_seed != seed:
                    row["stream_seed"] = copy_seed
                if gap_s is not None:
                    row["birth_gap"] = gap_s
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
    elif variant == "parts":
        info = parts_info
        base_dir = os.path.join(a["base"], info["folder"])
        as_source = source_read(a["base"], info["folder"], bed, heldout, device, log)
        only = _class_list(a["only_classes"])
        if only:
            train_bed = bed.restricted(only_classes=only)
            stream_rec = _stream_record(train_bed)
            data_block["training_stream"] = stream_rec
        stream_seed = seed + GROW_STREAM * PARTS_STREAM
        folders = list(info["clusters"])
        row = pspec["row"]
        init = info.get("init")
        clusters, base_rows, part_names = [], [], []
        for s, rel in enumerate(folders):                  # every stage copy with its new parts, before any training
            cdir = os.path.join(base_dir, rel)
            if pspec["groups"]:
                c, part_names = with_parts(cdir, seed, pspec["groups"], device=device)    # asserts a birth gap of 0.0
            else:
                c, part_names = link(cdir, device=device), []
            stem = c.unit("stem")
            if row is not None:
                base_rows.append(stem.cls.weight[row].detach().clone())
                with torch.no_grad():                      # at birth the class reads as no class
                    stem.cls.weight[row] = stem.cls.weight[NULL_CLASS]
            if init is not None:                           # a graft: the parts' weights from another parts job
                idir = os.path.join(init["dir"], init["clusters"][s])
                if part_names:
                    adopt(c, idir, part_names)
                if row is not None:
                    with torch.no_grad():
                        stem.cls.weight[row] = _stem_row(idir, row).to(stem.cls.weight.device)
            clusters.append(c)
        names_new = list(part_names)
        tree0 = Staged(clusters, cuts, soft)
        if init is None:
            if row is not None:
                extra["row_birth_gap"] = _class_row_gap(tree0, bed, heldout, device, row)
                log(f"ROW {row}: at birth the class-{row} rows read as with no class (gap {extra['row_birth_gap']!r})")
                if extra["row_birth_gap"] != 0.0:
                    raise AssertionError(f"the newborn row {row} changes the class-{row} rows: {extra['row_birth_gap']}")
            if part_names:
                extra["birth_gap"] = 0.0                   # with_parts asserts it on every stage copy
        else:
            pe = [_parents_equal(os.path.join(base_dir, rel), os.path.join(init["dir"], init["clusters"][s]),
                                 part_names, row) for s, rel in enumerate(folders)]
            extra["graft"] = {"from": a["init_from"], "model": init["model"], "init_stage": init["init_stage"],
                              "parents_equal": pe}
            graft_tree = total == 0 and not all(pe)          # untrained parts sit on a parent they were not fit on
            log(f"GRAFT the parts of {init['dir']} onto {base_dir}: parents equal by stage {pe}")
        rows_spec = None if row is None else {"stem": {"cls.weight": [row]}}
        trained_units = list(part_names) + (["stem"] if row is not None else [])
        for s, ((name, n), rel) in enumerate(zip(steps_plan, folders)):
            c, rng = clusters[s], ranges[s]
            cdir = os.path.join(base_dir, rel)
            log(f"PART {name}: the new parts of {rel} ({trained_units}" + (f", row {row} of the class table" if row
                is not None else "") + f"); t in [{rng[0]:g}, {rng[1]:g}), {n} steps; a new optimizer; the stream of "
                f"seed {stream_seed}")
            res = fit_units(c, train_bed, stream_seed, steps=n, trainable=list(part_names), trainable_rows=rows_spec,
                            t_range=rng, expect_anchor_at_birth=False, averages=entries, equalize=False, **common)
            note = None
            if init is not None:
                note = {"from": os.path.join(init["dir"], init["clusters"][s]), "trained_here_steps": n}
            folder_pairs = [(rel, None)] + [(os.path.join("avg_" + key, rel), d) for key, _, d in averages]
            for ff, d in folder_pairs:
                rec = res["recipe"] if d is None else average_recipe(res["recipe"], d)
                if rows_spec:
                    rec = dict(rec, trainable_rows=res["trainable_rows"])
                marks = _marks(res) if d is None else {"loss_over_levels": None,
                                                       "frozen_unchanged": res["frozen_unchanged"]}
                ctx = contextlib.nullcontext() if d is None else use_average(c, res["averaged"][d])
                with ctx:
                    export_build(c, os.path.join(out, ff), seed=seed, base=cdir, trained=trained_units, recipe=rec,
                                 data=data_block, marks=marks)
                if note is not None:
                    _graft_cards(os.path.join(out, ff), note, part_names, n == 0)
            prow = part_row(name, rel, res)
            prow.update(stream_seed=stream_seed, trained_names=trained_units)
            parts.append(prow)
            progress(name, n)
        sinfo = {"experiment": EXPERIMENT, "variant": variant, "seed": seed, "parts": pspec["spec"]}
        write_staged(out, folders, cuts, soft, info=sinfo)
        for key, _, _ in averages:
            write_staged(os.path.join(out, "avg_" + key), folders, cuts, soft, info=dict(sinfo, average=key))
        final, kind = load_job(out, device=device, kind="clusters", grafts=True) if graft_tree else \
            load_job(out, device=device, kind="clusters")
        avg_folders = [(k, "avg_" + k) for k, _, _ in averages]
        extra["_base_rows"] = base_rows
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
    # the eighth pass: the parts left out (parts_off) and the rows they are not on (exact); present only with parts
    base_rows = extra.pop("_base_rows", None)
    part_set = list(names_new) if variant == "parts" else (part_units(final) if variant in ("source", "copies") else [])
    row = pspec["row"] if (variant == "parts" and pspec is not None) else None
    with_parts_now = bool(part_set) or row is not None

    def off_context(off_model):
        return rows_set(off_model, row, base_rows) if row is not None else contextlib.nullcontext()

    def parts_off_of(m) -> Dict[str, Any]:
        off_model = without_parts(m, part_set)
        with off_context(off_model):
            r_off = read(off_model, bed, heldout, device)
        out_ = _brief(r_off)
        if variant == "parts":
            st = parts_info["stored"]
            out_["equals_base"] = (r_off["loss_over_levels"] == st["loss_over_levels"]
                                   and _levels(r_off) == st["by_level"])
        return out_

    def exact_of(m) -> Dict[str, Any]:
        if variant != "parts" or pspec["spec"] in ("experts10", "shared1"):
            return exactness(m, bed, heldout, device, part_set[0])           # content.py's one-part-out form
        e_in, cls_rows = row_errors(m, bed, heldout, device)
        off_model = without_parts(m, part_set)
        with off_context(off_model):
            e_out, _ = row_errors(off_model, bed, heldout, device)
        return class_exactness(e_in, e_out, cls_rows, pspec["classes"])

    if with_parts_now:
        result["parts_spec"] = pspec["spec"] if pspec is not None else None
        result["part_units"] = list(part_set)
        result["parts_off"] = parts_off_of(final)
        if variant != "parts":
            result["exact"] = exact_of(final)
            log(f"EXACT {result['exact']['expert']} left out: other classes' rows change at most "
                f"{result['exact']['others_max_abs']!r}")
        log(f"PARTS OFF: over ten levels {result['parts_off']['loss_over_levels']:.4f}"
            + (f" (equals the base: {result['parts_off']['equals_base']})" if variant == "parts" else ""))
    for k in ("birth_gap", "row_birth_gap", "graft"):
        if k in extra:
            result[k] = extra[k]
    if stream_rec or "training_stream_choice" in data_block:
        result["data"] = {"training_stream": stream_rec}
        if "training_stream_choice" in data_block:
            result["data"]["training_stream_choice"] = data_block["training_stream_choice"]
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
    judged_key = next((k for k, _ in avg_folders if k in to_score), None)
    for key, folder in avg_folders:
        if graft_tree:
            avg_model, _ = load_job(os.path.join(out, folder), device=device, grafts=True)
        else:
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
        if with_parts_now and key in to_score:            # the averages that are judged: their parts left out
            result["averaged"][key]["parts_off"] = parts_off_of(avg_model)
            if variant != "parts":
                result["averaged"][key]["exact"] = exact_of(avg_model)
        if with_parts_now and variant == "parts" and key == judged_key:
            result["exact"] = dict(exact_of(avg_model), reading=key)
    if with_parts_now and variant == "parts" and judged_key is None:
        result["exact"] = dict(exact_of(final), reading="last")
    if "exact" in result and variant == "parts":
        log(f"EXACT ({result['exact']['reading']}): other classes change at most {result['exact']['others_max_abs']!r}")
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
