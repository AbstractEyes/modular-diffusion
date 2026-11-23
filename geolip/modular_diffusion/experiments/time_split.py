"""The time split: what a stage of the noise level costs when it shares nothing, against one model over the whole range.

    python -m geolip.modular_diffusion.run time_split variant=split seed=0 data=<npz> out=<folder> device=cuda
    from geolip.modular_diffusion.experiments import time_split; time_split.main(variant="plain", data=..., out=...)

Stages: the cuts 0 / 0.25 / 0.65 / 1 with the crossfade half-width 0.06; a stage trained on its own range sees its window
widened by the half-width on its inner edges (TRAIN_RANGES), with a stage's share of the steps (SHARE). The recipe of
every variant: the pictures bed, width 128, 2 heads, a token grid of 16, the anchor added and read by the stem, Adam 1e-3,
100 warm-up steps, batch 128, caption drop 0.1, a held-out set of 2,048.

| variant      | what it builds and trains                                                                           |
|--------------|-----------------------------------------------------------------------------------------------------|
| plain        | anchor, stem of 7 blocks, head; every unit, 6,000 steps                                            |
| plain_4k ... | the same at 4,000 / 12,000 / 18,000 steps                                                          |
| plain_mid    | the same, 6,000 steps, t drawn logit-normal                                                         |
| plain_wide   | the same at width round(1.75 x dim)                                                                 |
| plain_d4     | a stem of 4 blocks, 6,000 steps                                                                     |
| split        | three such 7-block clusters, nothing shared, each trained alone on its stage's range (6,000 in all) |
| split_long   | the same, 6,000 steps each                                                                          |
| split_soft   | split, with a share SOFT_MIX of each batch's noise levels drawn from the whole range                |
| tails_mixed  | one model: stem of 4, three 3-block bodies whose states are mixed by the stage weights, one head    |
| tails_joint  | three clusters (stem 4, body 3) sharing the anchor and the stem, trained together, 6,000 steps      |
| tails_alone  | the stem of a finished plain_d4 run (stem_from=), frozen; body and head trained alone per stage     |
| copies_4k    | three copies of a finished plain_4k run (source=), each trained on its stage's range (2,000 in all) |
| copies       | the same from a finished plain run (6,000 in all)                                                   |
| copies_soft  | copies, with the whole-range share SOFT_MIX                                                         |

Every unit keeps the names anchor / stem / body / head, so cluster folders are interchangeable; the three clusters of a
variant are born with the same numbers and see the same pictures and noise at each step.

Arguments (key=value): variant, seed, data, out, device, heldout, batch, lr, warmup, log_every, threads, mem_fraction
(applied on Windows only), scale (multiplies every step count, at least 2 per part), dim, score (0 / 1), judge
(inception / fake), score_n, score_steps, ref (an npz caching the reference statistics), sheet, source, stem_from,
avg (one-model variants only: a comma list of decays and named averages, ``train.fit_units``; each average of the
trained parameters is exported as model_avg<digits after the point>/ for a decay or model_avg_<name>/ for a name (g8,
tail100), read and scored after the last weights, and reported under "averaged" by its decimal text or its name),
score_last (0: the last weights are read and not scored), score_avg (empty: every average is scored; a comma list:
only those; every average is read), det (empty, warn or on: ``determinism``).

Output in ``out``: log.txt (its last line 'done'), result.json, samples.png (with sheet), the build folder(s) (model/,
or stage_0 .. stage_2/ with staged.json).
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

import torch

from .. import __version__
from ..beds import PicturesBed
from ..config import in_notebook, parse
from ..data import npz_content_hash
from ..determinism import mode as det_mode
from ..forms import adopt, cluster_build, mixed_tails_build
from ..linker import export_build, link
from ..model import Model
from ..read import by_stage, read
from ..sample import sample_sheet
from ..score import make_judge, picture_score, reference_stats
from ..staged import STAGED_FILE, Staged, link_staged, share, write_staged
from ..train import fit_units

EXPERIMENT = "time_split"
CUTS = (0.0, 0.25, 0.65, 1.0)
SOFT = 0.06
TRAIN_RANGES = ((0.0, 0.31), (0.19, 0.71), (0.59, 1.0))
SHARE = (0.25, 0.40, 0.35)
SOFT_MIX = 0.3
TOTAL = 6000
WIDE = 1.75
JUDGES = ("inception", "fake")

PLAIN = {"plain": dict(steps=6000), "plain_4k": dict(steps=4000), "plain_12k": dict(steps=12000),
         "plain_18k": dict(steps=18000), "plain_mid": dict(steps=6000, t_sampling="logit_normal"),
         "plain_wide": dict(steps=6000, wide=True), "plain_d4": dict(steps=6000, stem_depth=4)}
CLUSTERED = ("split", "split_long", "split_soft", "tails_joint", "tails_alone", "copies_4k", "copies", "copies_soft")
VARIANTS = tuple(PLAIN) + ("tails_mixed",) + CLUSTERED

DEFAULTS = dict(variant="", seed=0, data="", out="", device="cuda", heldout=2048, batch=128, lr=1e-3, warmup=100,
                log_every=500, threads=4, mem_fraction=0.25, scale=1.0, dim=128, score=0, judge="inception",
                score_n=10000, score_steps=50, ref="", sheet=1, source="", stem_from="", avg="", score_last=1,
                score_avg="", det="")


def parse_averages(text: str) -> List[tuple]:
    """``avg=`` text "0.999,0.9995,g8,tail100" -> [(key "0.999", folder "model_avg999", decay 0.999), ...,
    ("g8", "model_avg_g8", "g8"), ("tail100", "model_avg_tail100", "tail100")]: a decay's key is its decimal text and
    its folder name carries the digits after the point; a named average ("g<number>", "tail<number>",
    ``train.average_spec``) is its own key and its folder is model_avg_<name>."""
    from ..train import average_spec
    out = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if item[0].isalpha():
            key, _, _ = average_spec(item)
            out.append((key, "model_avg_" + key, key))
            continue
        d = float(item)
        if not 0.0 < d < 1.0:
            raise ValueError(f"an average's decay lies in (0, 1): {item!r}")
        key = repr(d)
        out.append((key, "model_avg" + key.split(".", 1)[1], d))
    if len({k for k, _, _ in out}) != len(out):
        raise ValueError(f"an average is listed twice: {text!r}")
    return out


AVERAGE_RULES = {"decay": "d_k = min(d, (1 + k) / (10 + k)); avg <- d_k avg + (1 - d_k) w after optimizer step k",
                 "g": "d_k = (1 + k) / (k + g + 2); avg <- d_k avg + (1 - d_k) w after optimizer step k",
                 "tail": "the mean of the weights after each of the last n optimizer steps (float64 sums)"}


def average_recipe(recipe: dict, entry, carried: Optional[dict] = None) -> dict:
    """A card recipe for an average's weights: a decay's keys of earlier versions (average_decay, average_rule); a
    named average's name, its rule and, when it was carried, where it started."""
    from ..train import average_spec
    key, kind, _ = average_spec(entry)
    if kind == "decay":
        return dict(recipe, average_decay=key, average_rule=AVERAGE_RULES["decay"])
    rec = dict(recipe, average=key, average_rule=AVERAGE_RULES[kind])
    if carried:
        rec["average_carried_from"] = dict(carried)
    return rec


def export_averages(model, res: dict, averages: List[tuple], out: str, export, carried: Optional[dict] = None) -> None:
    """Each average's weights exported as a build folder of its own (``export(model, folder, recipe)`` with the
    averaged values swapped in; the last weights are back in the model afterwards). ``carried``: {key: a note of
    where a carried average started}, written into its recipe."""
    from ..train import use_average
    for key, folder, d in averages:
        rec = average_recipe(res["recipe"], d, (carried or {}).get(key))
        with use_average(model, res["averaged"][d]):
            export(model, folder, rec)


NEW_ARGS = {"score_last": 1, "score_avg": "", "det": "", "source_model": "", "stem_model": ""}


def recorded_args(a: dict) -> dict:
    """The arguments written into a result: every argument but avg= when empty and the arguments of the sixth pass
    at their defaults, so a job without the new options writes the result of earlier versions."""
    return {k: a[k] for k in sorted(a) if not (k == "avg" and not a[k]) and not (k in NEW_ARGS and a[k] == NEW_ARGS[k])}


def scored_keys(a: dict, averages: List[tuple]) -> set:
    """The keys of the averages that are scored: every one when score_avg= is empty; else the listed ones (each must
    be an average of avg=, ValueError otherwise)."""
    keys = {k for k, _, _ in averages}
    if not str(a.get("score_avg", "")).strip():
        return keys
    want = {k for k, _, _ in parse_averages(a["score_avg"])}
    unknown = want - keys
    if unknown:
        raise ValueError(f"score_avg= names averages that avg= does not list: {sorted(unknown)}")
    return want


def steps_of(n: int, scale: float) -> int:
    """A step count times ``scale``, at least 2."""
    return max(2, int(round(n * scale)))


def stage_steps(total: int, scale: float) -> List[int]:
    """Each stage's steps: round(share x total), times scale."""
    return [steps_of(int(round(sh * total)), scale) for sh in SHARE]


def kind_of(variant: str) -> str:
    return "clusters" if variant in CLUSTERED else "model"


def _non_anchor(model) -> List[str]:
    return [u.name for u in model.units if u.unit_type != "Anchor"]


def _part_row(name: str, folder, res: dict) -> dict:
    return {"name": name, "folder": folder, "steps": res["steps"], "t_range": res["t_range"], "t_mix": res["t_mix"],
            "trainable": res["trainable"], "trained_numbers": res["trained_numbers"], "train_seconds": res["train_seconds"],
            "peak_memory_mib": res["peak_memory_mib"], "curve": res["curve"], "frozen_unchanged": res["frozen_unchanged"]}


def _marks(res: dict) -> dict:
    return {"loss_over_levels": res["loss_over_levels"], "frozen_unchanged": res["frozen_unchanged"]}


def _levels(res: dict) -> dict:
    return {k: r["error"]["all"] for k, r in res["levels"].items()}


def _unique_numbers(model) -> int:
    return sum(p.numel() for p in model.parameters())


def load_job(folder: str, device="cpu", kind: Optional[str] = None, grafts: bool = False):
    """A finished job folder of this experiment as a model -> (model, kind): its ``model/`` build folder linked strictly
    (kind "model"), or its ``staged.json`` with the cluster folders (kind "clusters", a ``Staged``). Without ``kind`` the
    folder decides: a staged file means clusters. ``grafts``: grafted parts are vouched by their graft record
    (``linker.check``; off by default)."""
    if kind is None:
        if os.path.exists(os.path.join(folder, STAGED_FILE)):
            kind = "clusters"
        elif os.path.isdir(os.path.join(folder, "model")):
            kind = "model"
        else:
            raise ValueError(f"{folder} holds neither a staged file nor a model folder")
    if kind == "clusters":
        if grafts:
            return link_staged(folder, device=device, grafts=True), kind
        return link_staged(folder, device=device), kind
    if kind == "model":
        if grafts:
            return link(os.path.join(folder, "model"), device=device, grafts=True), kind
        return link(os.path.join(folder, "model"), device=device), kind
    raise ValueError(f"unknown kind {kind!r}; known: model, clusters")


def stored_key(model_folder: str) -> Optional[str]:
    """The key of a job's result under which a folder's read is stored: None for ``model`` (the result's own
    loss_over_levels); the average's key for an average's folder (model_avg999 -> "0.999", model_avg_g8 -> "g8", a
    staged average's folder avg_g8 -> "g8")."""
    name = model_folder.replace("\\", "/").strip("/")
    if name in ("model", ""):
        return None
    if name.startswith("model_avg_"):
        return name[len("model_avg_"):]
    if name.startswith("model_avg"):
        return "0." + name[len("model_avg"):]
    if name.startswith("avg_"):
        return name[len("avg_"):]
    raise ValueError(f"not a model folder of a job: {model_folder!r}")


def source_read(job: str, model_folder: str, bed, heldout: int, device, log=None) -> Dict[str, Any]:
    """A source job's folder read again before it is used as a base -> {"folder", "loss_over_levels", "by_level",
    "stored_loss_over_levels", "equals_stored"}: equals_stored compares with the job's stored number for that folder
    (``stored_key``; None when the job's result.json does not hold it)."""
    path = os.path.join(job, model_folder)
    model = link_staged(path, device=device) if os.path.exists(os.path.join(path, STAGED_FILE)) else link(path, device=device)
    r = read(model, bed, int(heldout), device)
    stored = None
    res_path = os.path.join(job, "result.json")
    if os.path.exists(res_path):
        with open(res_path, encoding="utf-8") as fh:
            res = json.load(fh)
        key = stored_key(model_folder)
        stored = res.get("loss_over_levels") if key is None else ((res.get("averaged") or {}).get(key) or {}).get(
            "loss_over_levels")
    out = {"folder": model_folder, "loss_over_levels": r["loss_over_levels"], "by_level": _levels(r),
           "stored_loss_over_levels": stored, "equals_stored": None if stored is None else r["loss_over_levels"] == stored}
    if log is not None:
        log(f"AS SOURCE {model_folder}: over ten levels {r['loss_over_levels']:.6f} (stored {stored}; "
            f"equal {out['equals_stored']})")
    return out


def plan(variant: str, scale: float) -> List[tuple]:
    """(part name, steps) of every training part of a variant, in order."""
    if variant in PLAIN:
        return [("model", steps_of(PLAIN[variant]["steps"], scale))]
    if variant == "tails_mixed":
        return [("model", steps_of(TOTAL, scale))]
    if variant == "tails_joint":
        return [("joint", steps_of(TOTAL, scale))]
    if variant == "split_long":
        return [(f"stage_{s}", steps_of(TOTAL, scale)) for s in range(3)]
    total = 2000 if variant == "copies_4k" else TOTAL
    return [(f"stage_{s}", n) for s, n in enumerate(stage_steps(total, scale))]


def main(argv=(), **overrides) -> dict:
    """Run one variant of one seed; writes the outputs into ``out`` and returns the result."""
    a = parse(DEFAULTS, argv, **overrides)
    with det_mode(a["det"]) as det:
        return _run(a, det)


def _run(a: dict, det) -> dict:
    variant = a["variant"]
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; known: {list(VARIANTS)}")
    for k in ("data", "out"):
        if not a[k]:
            raise ValueError(f"{k}= is required")
    if variant in ("copies_4k", "copies", "copies_soft") and not a["source"]:
        raise ValueError(f"{variant} needs source=<the folder of a finished run of the same seed>")
    if variant == "tails_alone" and not a["stem_from"]:
        raise ValueError("tails_alone needs stem_from=<the folder of a finished plain_d4 run of the same seed>")
    if int(a["score"]) and a["judge"] not in JUDGES:
        raise ValueError(f"unknown judge {a['judge']!r}; known: {list(JUDGES)}")
    averages = parse_averages(a["avg"])
    if averages and kind_of(variant) == "clusters":
        raise ValueError(f"avg= applies to the one-model variants; {variant} trains clusters (averages of clusters are "
                         f"not supported)")
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

    seed, dim, scale = int(a["seed"]), int(a["dim"]), float(a["scale"])
    steps_plan = plan(variant, scale)
    total_steps = sum(n for _, n in steps_plan)
    log(f"JOB {EXPERIMENT} {variant} seed {seed}: {len(steps_plan)} part(s), steps {[n for _, n in steps_plan]} "
        f"({total_steps} in all), on {device}; torch {torch.__version__}; out {out}")
    t_job = time.time()
    bed = PicturesBed(a["data"], device, grid=16)
    data_block = {"bed": "pictures", "source": os.path.basename(a["data"]), "content_hash": npz_content_hash(a["data"]),
                  "heldout": f"the first {a['heldout']} test pictures"}
    common = dict(lr=float(a["lr"]), warmup=int(a["warmup"]), batch=int(a["batch"]), cond_drop=0.1,
                  heldout=int(a["heldout"]), device=device, log=log, log_every=int(a["log_every"]))
    field = bed.field_shape
    done_steps = [0]

    def progress(name: str, n: int) -> None:
        done_steps[0] += n
        spent = time.time() - t_job
        left = spent / max(1, done_steps[0]) * (total_steps - done_steps[0])
        log(f"JOB: part {name} done; {done_steps[0]}/{total_steps} steps; spent {spent / 60:.1f} min; "
            f"left about {left / 60:.1f} min")

    def born(depth: int, body: int = 0, width: int = dim) -> Model:
        return Model.born(cluster_build(field, stem_depth=depth, body_depth=body, dim=width), seed, bed.spectrum)

    def export(model, folder: str, res: dict, trained, base: Optional[str] = None) -> None:
        export_build(model, os.path.join(out, folder), seed=seed, base=base, trained=trained, recipe=res["recipe"],
                     data=data_block, marks=_marks(res))

    parts: List[dict] = []
    torch.manual_seed(seed)
    if kind_of(variant) == "model":
        name, n = steps_plan[0]
        if variant == "tails_mixed":
            model = Model.born(mixed_tails_build(field, stem_depth=4, tail_depth=3, cuts=CUTS, soft=SOFT, dim=dim), seed,
                               bed.spectrum)
            t_sampling = "uniform"
        else:
            spec = PLAIN[variant]
            width = int(round(WIDE * dim)) if spec.get("wide") else dim
            model = born(spec.get("stem_depth", 7), 0, width)
            t_sampling = spec.get("t_sampling", "uniform")
        res = fit_units(model, bed, seed, steps=n, t_sampling=t_sampling, expect_anchor_at_birth=True,
                        averages=tuple(d for _, _, d in averages), **common)
        export(model, "model", res, _non_anchor(model))
        if averages:
            export_averages(model, res, averages, out, lambda m, f, rec: export_build(
                m, os.path.join(out, f), seed=seed, trained=_non_anchor(m), recipe=rec, data=data_block,
                marks={"loss_over_levels": None, "frozen_unchanged": res["frozen_unchanged"]}))
        parts.append(_part_row(name, "model", res))
        progress(name, n)
        final, _ = load_job(out, device=device, kind="model")
        clusters_final = None
    else:
        folders = [f"stage_{s}" for s in range(3)]
        t_mix = SOFT_MIX if variant in ("split_soft", "copies_soft") else 0.0
        if variant == "tails_joint":
            clusters = share([born(4, 3) for _ in range(3)], ["anchor", "stem"])
            staged = Staged(clusters, CUTS, SOFT)
            name, n = steps_plan[0]
            res = fit_units(staged, bed, seed, steps=n, expect_anchor_at_birth=True, **common)
            for c, f in zip(clusters, folders):
                export(c, f, res, _non_anchor(c))
            parts.append(_part_row(name, folders, res))
            progress(name, n)
        else:
            for s, ((name, n), f) in enumerate(zip(steps_plan, folders)):
                base = None
                if variant in ("split", "split_long", "split_soft"):
                    c = born(7)
                    trainable, expect = "all", True
                elif variant == "tails_alone":
                    base = os.path.join(a["stem_from"], "model")
                    c = born(4, 3)
                    adopt(c, base, ["stem"])
                    trainable, expect = ["body", "head"], True
                else:
                    base = os.path.join(a["source"], "model")
                    c = link(base, device=device)
                    trainable, expect = "all", False
                log(f"PART {name}: t in [{TRAIN_RANGES[s][0]:g}, {TRAIN_RANGES[s][1]:g}), {n} steps")
                res = fit_units(c, bed, seed, steps=n, trainable=trainable, t_range=TRAIN_RANGES[s], t_mix=t_mix,
                                expect_anchor_at_birth=expect, **common)
                export(c, f, res, _non_anchor(c) if trainable == "all" else trainable, base=base)
                parts.append(_part_row(name, f, res))
                progress(name, n)
        write_staged(out, folders, CUTS, SOFT, info={"experiment": EXPERIMENT, "variant": variant, "seed": seed})
        final, _ = load_job(out, device=device, kind="clusters")
        clusters_final = list(final.clusters)

    log("READ the assembled model")
    whole = read(final, bed, int(a["heldout"]), device)
    result: Dict[str, Any] = {
        "experiment": EXPERIMENT, "variant": variant, "seed": seed, "kind": kind_of(variant), "cuts": list(CUTS),
        "soft": SOFT, "parts": parts,
        "numbers": {"all": _unique_numbers(final), "trained": sum(p["trained_numbers"] for p in parts)},
        "train_seconds": sum(p["train_seconds"] for p in parts), "read": whole,
        "loss_over_levels": whole["loss_over_levels"], "by_level": _levels(whole), "by_stage": by_stage(whole, CUTS),
        "wrong_assembly": None, "alone": None, "score": None}
    if clusters_final is not None:
        wrong = read(final.rotated(1), bed, int(a["heldout"]), device)
        result["wrong_assembly"] = {"loss_over_levels": wrong["loss_over_levels"], "by_stage": by_stage(wrong, CUTS)}
        alone = []
        for c in clusters_final:
            r = read(c, bed, int(a["heldout"]), device)
            alone.append({"loss_over_levels": r["loss_over_levels"], "by_level": _levels(r)})
        result["alone"] = alone
    judged: Dict[str, Any] = {}

    def score_of(m, tag: str = "") -> dict:
        # a failure of the judge (its weights, its memory) is recorded and the job still ends with its losses
        n_score = int(a["score_n"])
        try:
            if "judge" not in judged:
                judged["judge"] = make_judge(a["judge"], device)
                judged["ref"] = reference_stats(bed, judged["judge"], a["ref"] or None, batch=1000)
            sc = picture_score(m, bed, judged["judge"], judged["ref"], n=n_score, steps=int(a["score_steps"]),
                               batch=min(1000, n_score), device=device)
            log(f"SCORE{tag} {sc['score']:.4f} on {n_score} pictures ({sc['judge']})")
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
            scored = int(a["score"]) and key in to_score
            result["averaged"][key] = {"loss_over_levels": r["loss_over_levels"], "by_level": _levels(r),
                                       "by_stage": by_stage(r, CUTS),
                                       "score": score_of(avg_model, f" (average {key})") if scored else None}
    if int(a["sheet"]):
        sample_sheet(final, bed, os.path.join(out, "samples.png"), device)
    # avg= and the newer options are recorded only when given, so a job without them writes the result of earlier versions
    result.update({"args": recorded_args(a), "torch": torch.__version__, "device": str(device),
                   "version": __version__})
    det.record(result)
    with open(os.path.join(out, "result.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=1)
        fh.write("\n")
    bs = ", ".join("-" if x is None else f"{x:.4f}" for x in result["by_stage"])
    log(f"RESULT {variant} seed {seed}: over ten levels {result['loss_over_levels']:.4f}; by stage {bs}"
        + (f"; wrong assembly {result['wrong_assembly']['loss_over_levels']:.4f}" if result["wrong_assembly"] else "")
        + "".join(f"; average {k} {v['loss_over_levels']:.4f}" for k, v in (result.get("averaged") or {}).items())
        + f"; {(time.time() - t_job) / 60:.1f} min")
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
    lines = ["| variant | seed | steps | numbers (all / trained) | seconds | total | by stage | wrong assembly | score |"
             + (" averages (total / score) |" if with_avg else ""),
             "|---|---|---|---|---|---|---|---|---|" + ("---|" if with_avg else "")]
    for r in rows:
        steps = sum(int(p["steps"]) for p in r["parts"])
        stages = " / ".join(_fmt(x) for x in r["by_stage"])
        wrong = _fmt(r["wrong_assembly"]["loss_over_levels"]) if r.get("wrong_assembly") else "-"
        score = _fmt(r["score"]["score"], 2) if (r.get("score") or {}).get("score") is not None else "-"
        line = (f"| {r['variant']} | {r['seed']} | {steps:,} | {r['numbers']['all']:,} / {r['numbers']['trained']:,} | "
                f"{r['train_seconds']:.0f} | {_fmt(r['loss_over_levels'])} | {stages} | {wrong} | {score} |")
        if with_avg:
            line += f" {averages_cell(r.get('averaged'))} |"
        lines.append(line)
    return "\n".join(lines) + "\n"


def averages_cell(averaged: Optional[dict]) -> str:
    """'g8 0.1234 / 12.34; tail100 0.1240 / -' for a result's averaged block ('-' when there is none)."""
    if not averaged:
        return "-"
    out = []
    for key, v in averaged.items():
        sc = (v.get("score") or {}).get("score")
        out.append(f"{key} {_fmt(v.get('loss_over_levels'))} / {_fmt(sc, 2)}")
    return "; ".join(out)


if __name__ == "__main__" and not in_notebook():
    main(sys.argv[1:])
