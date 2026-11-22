"""Closed forms and stored states: heads solved by least squares on a finished model, and band members trained from
the frozen body's stored states.

    python -m geolip.modular_diffusion.run closed_forms variant=head_solved source=<a finished plain job> data=<npz> out=<folder>
    from geolip.modular_diffusion.experiments import closed_forms; closed_forms.main(variant="members_live", ...)

| variant             | what                                                                                                  |
|---------------------|-------------------------------------------------------------------------------------------------------|
| head_solved         | source (a finished plain job): read as trained, then its head's last layer solved on ``batches``      |
|                     | batches of the training stream (``closed.head_equations``) and read again                            |
| heads_staged_solved | source with three stage heads, each solved on its stage's training range, blended with the cuts and   |
|                     | the crossfade of the time split                                                                      |
| members_live        | source (a finished plain_d4 job): its stem and head adopted into a build with three band members      |
|                     | (bands 0, 1, 2; one block; band-kept; zero-born; on at every noise level), trained ``steps`` steps   |
|                     | through the live frozen body                                                                         |
| cache_write         | the same model, its states for ``steps`` steps written to ``cache=``; seconds and bytes              |
| members_cached      | the members trained from that cache: ``steps`` steps (at most passes x the cache's), ``passes`` passes |
|                     | (step k reads stored step k mod the cache's steps); read=0 skips the read and the score              |
| class_heads         | source + ten class heads with no block (depth 0: a norm and one zero-born linear layer reading the   |
|                     | trunk's last states, one per class, on at every noise level), solved together by least squares on   |
|                     | ``batches`` batches (``closed.part_equations``; the system is solved class by class)                 |
| class_stage_heads   | source + thirty such heads: per class one per registered stage (the time split's cuts and crossfade), |
|                     | the three of a class solved jointly                                                                  |

members_live and members_cached at passes=1 use the same trainer settings (no member dropout, the trainer's default
gradient equalization), so on one device they give the same curve and the same read.

Arguments (key=value): those of the time split, plus batches (default 200), steps (default 1,500; times ``scale``, at
least 2), ridge (1e-6), cache (a folder), passes (1), read (1), source_model (the folder of the source job used as the
base, default model; when given, or for the class-head variants, the folder is read again first and recorded under
"as_source"), reader (the class read on the scored pictures; needs score=1 and ref=), det. log_every defaults to 50.
The class-head variants also write by_class, class_worth and exact (one head left out: the largest change on the rows
of the other classes, which must be 0.0, and the mean change on its own class's rows).

Output in ``out``: log.txt (its last line 'done'), result.json, and the model: model/ (head_solved and the member
variants) or stage_0 .. stage_2/ with staged.json (heads_staged_solved). result.json holds the time split's keys where they
apply and: solve (the provenance of every solved head), as_trained (the source re-read), curve, seconds_per_step and
cache (steps, bytes, write seconds).
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
from ..build import read_build
from ..cache import FeatureCache
from ..closed import install_parts, part_equations, solve_model_head, solve_parts, solve_stage_heads
from ..config import in_notebook, parse
from ..data import npz_content_hash
from ..determinism import mode as det_mode
from ..forms import add_band_members, adopt, cluster_build, with_parts
from ..linker import export_build, link
from ..model import Model
from ..read import by_stage, read
from ..sample import sample_sheet
from ..score import class_reader, make_judge, picture_score, reference_classes, reference_stats
from ..train import fit_units, recipe
from .content import class_numbers, exactness
from .time_split import (CUTS, JUDGES, SOFT, TRAIN_RANGES, _levels, _marks, _part_row, _unique_numbers, load_job,
                         recorded_args, source_read, steps_of, stored_key)

EXPERIMENT = "closed_forms"
VARIANTS = ("head_solved", "heads_staged_solved", "members_live", "cache_write", "members_cached")
MEMBER_VARIANTS = ("members_live", "cache_write", "members_cached")
CLASS_HEAD_VARIANTS = ("class_heads", "class_stage_heads")
ALL_VARIANTS = VARIANTS + CLASS_HEAD_VARIANTS        # VARIANTS keeps the list of earlier versions
NEW_DEFAULTS = {"reader": 0}

DEFAULTS = dict(variant="", seed=0, data="", out="", device="cuda", heldout=2048, batch=128, lr=1e-3, warmup=100,
                log_every=50, threads=4, mem_fraction=0.25, scale=1.0, dim=128, score=0, judge="inception",
                score_n=10000, score_steps=50, ref="", sheet=1, source="", stem_from="", batches=200, steps=1500,
                ridge=1e-6, cache="", passes=1, read=1, source_model="", reader=0, det="")


def class_head_groups(variant: str) -> List[dict]:
    """The parts a class-head variant adds (``forms.with_parts`` groups): class_heads, one depth-0 part per class over
    the whole range (class_head_c0 ..); class_stage_heads, one per class and registered stage (the time split's cuts
    and crossfade; class_head_s0_c0 ..)."""
    ten = [[c] for c in range(10)]
    if variant == "class_heads":
        return [dict(classes=ten, window=(0.0, 1.0), soft=0.0, depth=0, prefix="class_head")]
    return [dict(classes=ten, window=(CUTS[s], CUTS[s + 1]), soft=SOFT, depth=0, prefix=f"class_head_s{s}")
            for s in range(len(CUTS) - 1)]


def members_model(source_model_folder: str, seed: int, bed) -> tuple:
    """A finished one-cluster build (anchor, stem, head) with three newborn band members: the build of the same stem
    depth and width with 3 bands, the members appended (member_b0 .. member_b2), the stem and head adopted (hash
    checked) -> (model on the CPU, member names). The anchor (closed form, no trained numbers) is adopted as well, so
    its spectrum is the source's to the last digit whatever device measured the bed."""
    src = read_build(source_model_folder)
    g = src.geometry
    depth = int(src.entry("stem").spec["depth"])
    build = cluster_build(bed.field_shape, stem_depth=depth, dim=int(g["dim"]), heads=int(g["heads"]), grid=int(g["grid"]),
                          anchor_in=int(src.entry("stem").spec.get("anchor_in", 0)), bands=3)
    names = add_band_members(build, (0, 1, 2), depth=1, window=(0.0, 1.0), soft=0.0, keep=True)
    model = Model.born(build, seed, bed.spectrum)
    adopt(model, source_model_folder, ["anchor", "stem", "head"])
    return model, names


def main(argv=(), **overrides) -> dict:
    """Run one variant of one seed; writes the outputs into ``out`` and returns the result."""
    a = parse(DEFAULTS, argv, **overrides)
    with det_mode(a["det"]) as det:
        return _run(a, det)


def _run(a: dict, det) -> dict:
    variant = a["variant"]
    if variant not in ALL_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; known: {list(ALL_VARIANTS)}")
    for k in ("data", "out", "source"):
        if not a[k]:
            raise ValueError(f"{k}= is required")
    if variant in ("cache_write", "members_cached") and not a["cache"]:
        raise ValueError(f"{variant} needs cache=<a folder for the stored states>")
    if int(a["score"]) and a["judge"] not in JUDGES:
        raise ValueError(f"unknown judge {a['judge']!r}; known: {list(JUDGES)}")
    if int(a["reader"]) and not (int(a["score"]) and a["ref"]):
        raise ValueError("reader=1 needs score=1 and ref=<the reference file>")
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

    seed, scale = int(a["seed"]), float(a["scale"])
    batches, ridge = int(a["batches"]), float(a["ridge"])
    n = steps_of(int(a["steps"]), scale)
    source_model = a["source_model"] or "model"
    base = os.path.join(a["source"], source_model)
    log(f"JOB {EXPERIMENT} {variant} seed {seed}: source {a['source']}" + (f" ({source_model})" if a["source_model"] else "")
        + "; "
        + (f"{batches} batches" if variant not in MEMBER_VARIANTS else f"{n} steps") + f", on {device}; "
        f"torch {torch.__version__}; out {out}")
    t_job = time.time()
    bed = PicturesBed(a["data"], device, grid=16)
    data_block = {"bed": "pictures", "source": os.path.basename(a["data"]), "content_hash": npz_content_hash(a["data"]),
                  "heldout": f"the first {a['heldout']} test pictures"}
    common = dict(lr=float(a["lr"]), warmup=int(a["warmup"]), batch=int(a["batch"]), cond_drop=0.1,
                  heldout=int(a["heldout"]), device=device, log=log, log_every=int(a["log_every"]))
    heldout = int(a["heldout"])
    result: Dict[str, Any] = {
        "experiment": EXPERIMENT, "variant": variant, "seed": seed, "kind": "model", "cuts": list(CUTS), "soft": SOFT,
        "parts": [], "numbers": None, "train_seconds": 0.0, "read": None, "loss_over_levels": None, "by_level": None,
        "by_stage": None, "wrong_assembly": None, "alone": None, "score": None, "solve": None, "as_trained": None,
        "curve": None, "seconds_per_step": None, "cache": None}
    final = None
    do_read = bool(int(a["read"]))
    torch.manual_seed(seed)
    if a["source_model"] or variant in CLASS_HEAD_VARIANTS:
        result["as_source"] = source_read(a["source"], source_model, bed, heldout, device, log)

    if variant in CLASS_HEAD_VARIANTS:
        model, names = with_parts(base, seed, class_head_groups(variant), device=device)
        log(f"PART {len(names)} class head(s) with no block on {base}; solved on {batches} batches")
        t0 = time.time()
        eq = part_equations(model, bed, seed, batches, names, batch=int(a["batch"]), cond_drop=0.1)
        solution = solve_parts(eq, ridge)
        prov = install_parts(model, solution, eq)
        result["train_seconds"] = time.time() - t0
        log(f"SOLVE {len(names)} parts in {len(eq['groups'])} group(s): mse over the equations' rows "
            f"{prov['all']['mse_before']:.5f} -> {prov['all']['mse_after']:.5f}"
            + (f"; never on: {solution['unsolved']}" if solution["unsolved"] else ""))
        rec = {"optimizer": None, "fit": "joint least squares per class (closed form)", "batches": batches,
               "batch": int(a["batch"]), "t_range": [0.0, 1.0], "ridge": ridge, "seed": seed}
        export_build(model, os.path.join(out, "model"), seed=seed, base=base, trained=names, recipe=rec,
                     data=data_block, marks={"frozen_unchanged": True})
        result["solve"] = prov
        final, _ = load_job(out, device=device, kind="model")
    elif variant in ("head_solved", "heads_staged_solved"):
        model = link(base, device=device)
        log("READ the source as trained")
        first = read(model, bed, heldout, device)
        stored = None
        if os.path.exists(os.path.join(a["source"], "result.json")):
            with open(os.path.join(a["source"], "result.json"), encoding="utf-8") as fh:
                src_res = json.load(fh)
            key = stored_key(source_model)
            stored = src_res.get("loss_over_levels") if key is None else (
                (src_res.get("averaged") or {}).get(key) or {}).get("loss_over_levels")
        result["as_trained"] = {"loss_over_levels": first["loss_over_levels"], "by_level": _levels(first),
                                "by_stage": by_stage(first, CUTS), "stored_loss_over_levels": stored,
                                "equals_stored": None if stored is None else first["loss_over_levels"] == stored}
        log(f"AS TRAINED {first['loss_over_levels']:.6f} (stored {stored})")
        t0 = time.time()
        if variant == "head_solved":
            prov = solve_model_head(model, bed, seed, batches, int(a["batch"]), 0.1, (0.0, 1.0), ridge)
            log(f"SOLVE head: {batches} batches; mse {prov['mse_before']:.5f} -> {prov['mse_after']:.5f}")
            rec = {"optimizer": None, "fit": "least squares (closed form)", "batches": batches, "batch": int(a["batch"]),
                   "t_range": [0.0, 1.0], "ridge": ridge, "seed": seed}
            export_build(model, os.path.join(out, "model"), seed=seed, base=base, trained=[model.head.name], recipe=rec,
                         data=data_block, marks={"frozen_unchanged": True})
            result["solve"] = {"head": prov}
            final, _ = load_job(out, device=device, kind="model")
        else:
            _, provs = solve_stage_heads(model, bed, seed, batches, CUTS, SOFT, TRAIN_RANGES, out, base=base,
                                         batch=int(a["batch"]), ridge=ridge, data=data_block,
                                         info={"experiment": EXPERIMENT, "variant": variant, "seed": seed}, log=log)
            result["solve"] = {f"stage_{s}": p for s, p in enumerate(provs)}
            result["kind"] = "clusters"
            final, _ = load_job(out, device=device, kind="clusters")
        result["train_seconds"] = time.time() - t0
    else:
        model, names = members_model(base, seed, bed)
        model.to(device)
        if variant == "members_live":
            log(f"PART members: {names} through the live frozen body; {n} steps")
            res = fit_units(model, bed, seed, steps=n, trainable=names, mdrop=0.0, equalize=True,
                            expect_anchor_at_birth=False, **common)
            export_build(model, os.path.join(out, "model"), seed=seed, trained=names, recipe=res["recipe"], data=data_block,
                         marks=_marks(res))
            result["parts"] = [_part_row("members", "model", res)]
            result["curve"] = res["curve"]
            result["train_seconds"] = res["train_seconds"]
            result["seconds_per_step"] = res["train_seconds"] / max(1, n)
            final, _ = load_job(out, device=device, kind="model")
        elif variant == "cache_write":
            t0 = time.time()
            cache = FeatureCache.write(model, bed, seed, n, int(a["batch"]), a["cache"], train=names, log=log)
            seconds = time.time() - t0
            result["cache"] = {"steps": n, "bytes": cache.size_bytes(), "write_seconds": seconds, "folder": a["cache"]}
            log(f"CACHE {n} steps written: {cache.size_bytes():,} bytes in {seconds:.1f} s")
            do_read = False
        else:
            cache = FeatureCache.open(a["cache"], model)
            if int(cache.meta["seed"]) != seed or int(cache.meta["batch"]) != int(a["batch"]):
                raise ValueError(f"the cache was written for seed {cache.meta['seed']} and batch {cache.meta['batch']}")
            passes = int(a["passes"])
            log(f"PART members from the cache ({cache.steps} stored steps, {passes} pass(es)): {n} steps")
            frozen_before = {id(p): p.detach().clone() for u in model.units if u.name not in names for p in u.parameters()}
            t0 = time.time()
            curve = cache.fit(model, lr=float(a["lr"]), warmup=int(a["warmup"]), steps=n, equalize=True,
                              log_every=int(a["log_every"]), log=log, passes=passes)
            seconds = time.time() - t0
            frozen_same = all(torch.equal(frozen_before[id(p)], p) for p in model.parameters() if id(p) in frozen_before)
            rec = recipe(float(a["lr"]), int(a["warmup"]), int(a["batch"]), n, "uniform", (0.0, 1.0),
                         float(cache.meta["cond_drop"]), True, True, 0.0, device)
            rec.update({"from_cache": True, "cache_steps": cache.steps, "passes": passes})
            trained_numbers = sum(p.numel() for u in model.units if u.name in names for p in u.parameters())
            export_build(model, os.path.join(out, "model"), seed=seed, trained=names, recipe=rec, data=data_block,
                         marks={"frozen_unchanged": bool(frozen_same)})
            result["parts"] = [{"name": "members", "folder": "model", "steps": n, "t_range": [0.0, 1.0], "t_mix": 0.0,
                                "trainable": sorted(names), "trained_numbers": trained_numbers, "train_seconds": seconds,
                                "peak_memory_mib": None, "curve": curve, "frozen_unchanged": bool(frozen_same)}]
            result["curve"] = curve
            result["train_seconds"] = seconds
            result["seconds_per_step"] = seconds / max(1, n)
            result["cache"] = {"steps": cache.steps, "bytes": cache.size_bytes(),
                               "write_seconds": cache.meta.get("write_seconds"), "passes": passes, "folder": a["cache"]}
            log(f"CACHED {n} steps in {seconds:.1f} s ({seconds / max(1, n):.4f} s a step); frozen unchanged {frozen_same}")
            if do_read:
                final, _ = load_job(out, device=device, kind="model")

    if final is not None:
        result["numbers"] = {"all": _unique_numbers(final),
                             "trained": sum(p["trained_numbers"] for p in result["parts"]) if result["parts"] else 0}
    if final is not None and do_read:
        log("READ the model")
        whole = read(final, bed, heldout, device)
        result.update({"read": whole, "loss_over_levels": whole["loss_over_levels"], "by_level": _levels(whole),
                       "by_stage": by_stage(whole, CUTS)})
        if result["kind"] == "clusters":
            wrong = read(final.rotated(1), bed, heldout, device)
            result["wrong_assembly"] = {"loss_over_levels": wrong["loss_over_levels"], "by_stage": by_stage(wrong, CUTS)}
            alone = []
            for c in final.clusters:
                r = read(c, bed, heldout, device)
                alone.append({"loss_over_levels": r["loss_over_levels"], "by_level": _levels(r)})
            result["alone"] = alone
        if variant in CLASS_HEAD_VARIANTS:
            cn = class_numbers(whole)
            result["by_class"], result["class_worth"] = cn["by_class"], cn["class_worth"]
            names = [u.name for u in final.members]
            result["exact"] = exactness(final, bed, heldout, device, names[0])
            log(f"EXACT {names[0]} left out: other classes' rows change at most {result['exact']['others_max_abs']!r}")
        if int(a["score"]):
            n_score = int(a["score_n"])
            try:
                judge = make_judge(a["judge"], device)
                rdr = None
                if int(a["reader"]):
                    ref_all = reference_classes(bed, judge, a["ref"] or None, batch=1000)
                    ref = (ref_all["mu"], ref_all["cov"], ref_all["n"])
                    rdr = class_reader(ref_all)
                else:
                    ref = reference_stats(bed, judge, a["ref"] or None, batch=1000)
                result["score"] = picture_score(final, bed, judge, ref, n=n_score, steps=int(a["score_steps"]),
                                                batch=min(1000, n_score), device=device, reader=rdr)
                log(f"SCORE {result['score']['score']:.4f} on {n_score} pictures ({result['score']['judge']})"
                    + (f"; class read {result['score']['class_read']['accuracy']:.4f}" if rdr is not None else ""))
            except Exception as e:
                result["score"] = {"error": repr(e)}
                log(f"SCORE not computed: {e!r}")
        if int(a["sheet"]):
            sample_sheet(final, bed, os.path.join(out, "samples.png"), device)
    # the newer options are recorded only when given, so a job without them writes the result of earlier versions
    args = {k: v for k, v in recorded_args(a).items() if not (k in NEW_DEFAULTS and v == NEW_DEFAULTS[k])}
    result.update({"args": args, "torch": torch.__version__, "device": str(device), "version": __version__})
    det.record(result)
    with open(os.path.join(out, "result.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=1)
        fh.write("\n")
    total = "-" if result["loss_over_levels"] is None else f"{result['loss_over_levels']:.4f}"
    log(f"RESULT {variant} seed {seed}: over ten levels {total}; {(time.time() - t_job) / 60:.1f} min")
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
    lines = ["| variant | seed | as trained | total | by stage | solved mse (after) | s / step | cache bytes | score |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        stages = " / ".join(_fmt(x) for x in r["by_stage"]) if r.get("by_stage") else "-"
        trained = _fmt((r.get("as_trained") or {}).get("loss_over_levels"))
        solve = r.get("solve") or {}
        if "all" in solve:                                   # parts solved together: the error over every row
            mse = _fmt(solve["all"]["mse_after"], 5)
        else:
            mse = " / ".join(_fmt(p["mse_after"], 5) for p in solve.values()) if solve else "-"
        cache = f"{r['cache']['bytes']:,}" if r.get("cache") else "-"
        sc = r.get("score") or {}
        score = _fmt(sc["score"], 2) if sc.get("score") is not None else "-"
        lines.append(f"| {r['variant']} | {r['seed']} | {trained} | {_fmt(r.get('loss_over_levels'))} | {stages} | {mse} | "
                     f"{_fmt(r.get('seconds_per_step'))} | {cache} | {score} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__" and not in_notebook():
    main(sys.argv[1:])
