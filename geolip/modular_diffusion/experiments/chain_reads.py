"""Chain reads: saved models of the time split assembled in new ways and read, with no training.

    python -m geolip.modular_diffusion.run chain_reads variant=twin base=<job folder> donor=<job folder> stage=1 \\
        data=<npz> out=<folder> device=cuda
    from geolip.modular_diffusion.experiments import chain_reads; chain_reads.main(variant="wrong", base=..., data=..., out=...)

A job folder is a finished job of the time-split experiment (``time_split.load_job``): a model job (one model over the
whole range) or a cluster job (three clusters at the cuts 0 / 0.25 / 0.65 / 1 with the crossfade half-width 0.06).

| variant | arguments                         | the model that is read                                                         |
|---------|-----------------------------------|--------------------------------------------------------------------------------|
| swap    | base (a model job), donor (a cluster job), stage k | three stages at the experiment's cuts and crossfade: the base model at every stage but k, the donor's cluster k at stage k |
| twin    | base, donor (cluster jobs), stage k | the base's clusters with cluster k replaced by the donor's cluster k          |
| wrong   | base (a cluster job)              | the base's clusters served to the wrong stages (``Staged.rotated(1)``)          |
| blend   | base (a cluster job), soft w      | the base's clusters blended with the half-width w (0: a hard switch)            |
| steps   | base (any job), plan, n           | the base as it is, its pictures sampled on a time grid of n steps (PLANS)       |

Plans of the steps variant: ``uniform`` = the points i / n (the steps of ``sample.euler(steps=n)``); every other plan =
``sample.stage_grid(CUTS, sample.share_counts(PLANS[plan], n))``, n steps divided among the stages by the plan's shares.
A grid changes the pictures only: the loss read of the steps variant is the base's own.

Every read: the loss at the ten fixed noise levels (over the levels, by level, by stage for the experiment's cuts) on the
held-out pictures, and the picture score (a failure of the judge is recorded, never raised). Units of the base model that
serve several stages (swap) are one object and their parameters are counted once.

Arguments (key=value): variant, data, out, device, base, donor, stage, soft, plan, n, heldout, score (0 / 1), judge
(inception / fake), score_n, score_steps, score_seed (the seed of the sampled pictures' noise; another value gives the
score's own sampling spread), ref (an npz caching the reference statistics), seed_label (free text copied into the
result), threads, mem_fraction (applied on Windows only), sheet.

Output in ``out``: log.txt (its last line 'done'), result.json, samples.png (with sheet). No weights are written, and the
job folders read are left unchanged.
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
from ..config import in_notebook, parse
from ..determinism import mode as det_mode
from ..read import T_GRID, by_stage, read
from ..sample import sample_sheet, share_counts, stage_grid, uniform_grid
from ..score import SCORE_SEED, make_judge, picture_score, reference_stats
from ..staged import Staged
from .time_split import CUTS, JUDGES, SOFT, TRAIN_RANGES, _fmt, _levels, _unique_numbers, load_job

EXPERIMENT = "chain_reads"
VARIANTS = ("swap", "twin", "wrong", "blend", "steps")
PLANS = {"width": (0.25, 0.40, 0.35), "start": (0.50, 0.30, 0.20), "middle": (0.20, 0.60, 0.20),
         "finish": (0.20, 0.30, 0.50)}
PLAN_NAMES = ("uniform",) + tuple(PLANS)

DEFAULTS = dict(variant="", data="", out="", device="cuda", base="", donor="", stage=-1, soft=-1.0, plan="", n=0,
                heldout=2048, score=1, judge="inception", score_n=10000, score_steps=50, score_seed=SCORE_SEED, ref="",
                seed_label="", threads=4, mem_fraction=0.25, sheet=0, det="")


def levels_outside(stage: int) -> List[float]:
    """The fixed levels at which stage ``stage`` has weight exactly 0: outside its window widened by the crossfade
    half-width on its inner edges (``time_split.TRAIN_RANGES``)."""
    lo, hi = TRAIN_RANGES[stage]
    return [t for t in T_GRID if not lo <= t <= hi]


def levels_inside(stage: int) -> List[float]:
    """The fixed levels at which stage ``stage`` has weight exactly 1: inside its window and outside every crossfade."""
    last = len(CUTS) - 2
    lo = CUTS[stage] + (SOFT if stage > 0 else 0.0)
    hi = CUTS[stage + 1] - (SOFT if stage < last else 0.0)
    return [t for t in T_GRID if lo <= t < hi]


def plan_grid(plan: str, n: int) -> List[float]:
    """The time grid of a plan of the steps variant with n steps."""
    if plan == "uniform":
        return uniform_grid(n)
    if plan not in PLANS:
        raise ValueError(f"unknown plan {plan!r}; known: {list(PLAN_NAMES)}")
    return stage_grid(CUTS, share_counts(PLANS[plan], n))


def _check(a: dict) -> None:
    """Refuse an incomplete or unknown argument set before anything is loaded."""
    variant = a["variant"]
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; known: {list(VARIANTS)}")
    for k in ("data", "out", "base"):
        if not a[k]:
            raise ValueError(f"{k}= is required")
    n_stages = len(CUTS) - 1
    if variant in ("swap", "twin"):
        if not a["donor"]:
            raise ValueError(f"{variant} needs donor=<a cluster job folder>")
        if not 0 <= int(a["stage"]) < n_stages:
            raise ValueError(f"{variant} needs stage=<0 .. {n_stages - 1}>, not {a['stage']}")
    if variant == "blend" and float(a["soft"]) < 0:
        raise ValueError("blend needs soft=<the crossfade half-width, 0 or more>")
    if variant == "steps":
        if a["plan"] not in PLAN_NAMES:
            raise ValueError(f"steps needs plan=<one of {list(PLAN_NAMES)}>, not {a['plan']!r}")
        if int(a["n"]) < (1 if a["plan"] == "uniform" else n_stages):
            raise ValueError(f"steps needs n=<the total step count>, not {a['n']}")
    if int(a["score"]) and a["judge"] not in JUDGES:
        raise ValueError(f"unknown judge {a['judge']!r}; known: {list(JUDGES)}")


def _need(kind: str, want: str, role: str, folder: str) -> None:
    if kind != want:
        raise ValueError(f"{role}={folder} is a {kind} job; this variant needs a {want} job")


def assemble(a: dict, device) -> tuple:
    """The model a variant reads -> (model, grid or None, a line describing it)."""
    variant = a["variant"]
    base, base_kind = load_job(a["base"], device=device)
    if variant == "steps":
        grid = plan_grid(a["plan"], int(a["n"]))
        return base, grid, f"the {base_kind} job as it is; plan {a['plan']}, {len(grid) - 1} steps"
    if variant == "swap":
        _need(base_kind, "model", "base", a["base"])
        donor, donor_kind = load_job(a["donor"], device=device)
        _need(donor_kind, "clusters", "donor", a["donor"])
        k = int(a["stage"])
        if len(donor.clusters) != len(CUTS) - 1:
            raise ValueError(f"the donor holds {len(donor.clusters)} clusters, not {len(CUTS) - 1}")
        model = Staged([donor.clusters[s] if s == k else base for s in range(len(CUTS) - 1)], CUTS, SOFT)
        return model, None, f"the base model at every stage but {k}; the donor's cluster {k} at stage {k}"
    _need(base_kind, "clusters", "base", a["base"])
    if variant == "wrong":
        return base.rotated(1), None, "the base's clusters at the wrong stages (cluster s + 1 at stage s)"
    if variant == "blend":
        w = float(a["soft"])
        return Staged(list(base.clusters), base.cuts, w), None, f"the base's clusters with the half-width {w:g}"
    donor, donor_kind = load_job(a["donor"], device=device)
    _need(donor_kind, "clusters", "donor", a["donor"])
    if len(donor.clusters) != len(base.clusters):
        raise ValueError(f"the donor holds {len(donor.clusters)} clusters, the base {len(base.clusters)}")
    k = int(a["stage"])
    clusters = [donor.clusters[s] if s == k else c for s, c in enumerate(base.clusters)]
    return Staged(clusters, base.cuts, base.soft), None, f"the base's clusters with the donor's cluster {k} at stage {k}"


def main(argv=(), **overrides) -> dict:
    """Read one assembly; writes the outputs into ``out`` and returns the result."""
    a = parse(DEFAULTS, argv, **overrides)
    _check(a)
    with det_mode(a["det"]) as det:
        return _run(a, det)


def _run(a: dict, det) -> dict:
    variant = a["variant"]
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

    t_job = time.time()
    log(f"JOB {EXPERIMENT} {variant}: base {a['base']}" + (f", donor {a['donor']}" if a["donor"] else "")
        + f", on {device}; torch {torch.__version__}; out {out}")
    bed = PicturesBed(a["data"], device, grid=16)
    model, grid, what = assemble(a, device)
    log(f"MODEL {what}")
    log("READ the assembled model")
    whole = read(model, bed, int(a["heldout"]), device)
    stage = int(a["stage"]) if variant in ("swap", "twin") else None
    result: Dict[str, Any] = {
        "experiment": EXPERIMENT, "variant": variant, "seed_label": a["seed_label"], "base": a["base"],
        "donor": a["donor"] or None, "stage": stage, "soft": float(a["soft"]) if variant == "blend" else None,
        "plan": a["plan"] if variant == "steps" else None, "n": int(a["n"]) if variant == "steps" else None,
        "grid": grid, "numbers": _unique_numbers(model), "loss_over_levels": whole["loss_over_levels"],
        "by_level": _levels(whole), "by_stage": by_stage(whole, CUTS), "score": None}
    if int(a["score"]):
        # a failure of the judge (its weights, its memory) is recorded and the job still ends with its losses
        n_score = int(a["score_n"])
        try:
            judge = make_judge(a["judge"], device)
            ref = reference_stats(bed, judge, a["ref"] or None, batch=1000)
            result["score"] = picture_score(model, bed, judge, ref, n=n_score, steps=int(a["score_steps"]),
                                            batch=min(1000, n_score), seed=int(a["score_seed"]), device=device, grid=grid)
            log(f"SCORE {result['score']['score']:.4f} on {n_score} pictures, {result['score']['steps']} steps "
                f"({result['score']['judge']})")
        except Exception as e:
            result["score"] = {"error": repr(e)}
            log(f"SCORE not computed: {e!r}")
    if int(a["sheet"]):
        sample_sheet(model, bed, os.path.join(out, "samples.png"), device, grid=grid)
    # det= is recorded only when given, so a read without it writes the result of earlier versions
    result.update({"seconds": time.time() - t_job, "args": {k: a[k] for k in sorted(a) if not (k == "det" and not a[k])},
                   "torch": torch.__version__, "device": str(device), "version": __version__})
    det.record(result)
    with open(os.path.join(out, "result.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=1)
        fh.write("\n")
    bs = ", ".join(_fmt(x) for x in result["by_stage"])
    log(f"RESULT {variant}: over ten levels {result['loss_over_levels']:.4f}; by stage {bs}; "
        f"{result['numbers']:,} numbers; {(time.time() - t_job) / 60:.1f} min")
    log("done")
    return result


def _short(folder: Optional[str]) -> str:
    """The last two parts of a folder path (the job and the folder holding it), or '-'."""
    if not folder:
        return "-"
    parts = [p for p in folder.replace("\\", "/").split("/") if p]
    return "/".join(parts[-2:])


def _cell(x, nd: int = 4) -> str:
    if x is None:
        return "-"
    return _fmt(x, nd) if isinstance(x, float) else str(x)


def summarize(root: str) -> str:
    """A markdown table, one row per job folder under ``root`` holding a result.json of this experiment."""
    rows = []
    for dirpath, _, files in os.walk(root):
        if "result.json" in files:
            with open(os.path.join(dirpath, "result.json"), encoding="utf-8") as fh:
                r = json.load(fh)
            if r.get("experiment") == EXPERIMENT:
                rows.append(r)
    rows.sort(key=lambda r: tuple(str(r.get(k)) for k in ("variant", "base", "donor", "stage", "soft", "plan", "n")))
    lines = ["| variant | base | donor | stage | soft | plan | n | total | by stage | score |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        stages = " / ".join(_fmt(x) for x in r["by_stage"])
        score = _fmt(r["score"]["score"], 2) if (r.get("score") or {}).get("score") is not None else "-"
        lines.append(f"| {r['variant']} | {_short(r['base'])} | {_short(r.get('donor'))} | {_cell(r.get('stage'))} | "
                     f"{_cell(r.get('soft'), 2)} | {_cell(r.get('plan'))} | {_cell(r.get('n'))} | "
                     f"{_fmt(r['loss_over_levels'])} | {stages} | {score} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__" and not in_notebook():
    main(sys.argv[1:])
