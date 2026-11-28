"""Band needs: what each noise band of a frozen teacher (Stable Diffusion 1.5) needs, measured with no training.

    python -m geolip.modular_diffusion.run band_needs teacher=<folder> data=<shards> out=<folder> device=cuda
    from geolip.modular_diffusion.experiments import band_needs; band_needs.main(tiny=1, out=...)

| variant | what it does                                                                                            |
|---------|---------------------------------------------------------------------------------------------------------|
| reads   | the reads below; every finished unit is written at once, so a stopped job continues by the same call     |
| timing  | no reads: measured seconds of forwards on the device and the projected minutes of each read              |

The reads (their names are the keys of result.json), each made on every probe seed:

    controls   the teacher's own copy (the engine of the shape ``full``) read on each band: exactly 0.0; a randomly
               initialized engine of the shape ``base`` read beside the inherited one, and their ratio
    r1         the born read of every single change of the teacher's shape on each band (``measure.removal_scores``)
    r2         the born read of the named shapes base, small and tiny on each band (one build a shape), with their sizes
    r3         how much each band needs the prompt (``measure.text_need``), on the seed's prompts in groups; the
               numbers of a seed are the means over its groups, weighted by rows; each band's change is also given as
               its share of the "empty prompt everywhere" change; the reference sampled again in the first group must
               change by exactly 0.0
    r4         once: parameters and measured seconds of a forward of every shape above against the teacher's

DATA: a folder of latent shards (``capture.data``) read in file order, not shuffled: row i is the i-th row of the
shards. The k-th seed of ``seeds`` owns the rows k * prompts .. k * prompts + prompts - 1; its born reads use the first
``rows`` of them and its prompt read all of them. The probe's noise seed is 1000 + seed; the sampling noise seed of the
prompt group g is 100000 * (seed + 1) + g. ``tiny=1`` uses the tiny random teacher and, with no ``data``, random shards
written into ``<out>/synthetic``.

PRECISION: the born reads, the controls and r4 run with the teacher in ``dtype``; then the teacher is cast in place to
``sample_dtype`` (float32 on the CPU, whatever is asked) for r3. TF32 is switched off for matrix products and
convolutions and the convolution autotuner stays off while the job runs (the three switches are restored afterwards
and recorded in the result).

RESUME AND PROGRESS: each unit (``controls_s0``, ``r1_s0``, ``r2_s0``, ``r4``, ``r3_s0_g000``, ...) is written to
``<out>/parts/<unit>.json`` through a side file and a rename as soon as it is finished; a unit whose file exists is read
back, not run again. At the start the job's size is printed: its units and its ROW-FORWARDS (one row through one U-Net
forward), counted exactly: in r3 a part of the assembly runs on a sampler step only where its band weight is above
zero, and guidance doubles the rows. After every unit: units done, minutes spent, minutes left (from the measured
seconds per row-forward of the units of the same kind).

THE SOURCE: by default the teacher itself; ``unet``, ``kind``, ``shift``, ``unguided``, ``kappa`` and ``unet_seed`` (the
capture experiment's source arguments) read another source of the same U-Net layout, for example a flow source, under
its own process: its probe inputs are its noised inputs on the same steps, its prompt read uses its sampler and its
unguided side, its probe rows are multiplied by kappa.

Arguments (key=value): variant, teacher, repo, revision, tiny, data, out, device, seeds, rows (probe rows a seed for the
born reads), per_band (probe steps a band), prompts (rows a seed for the prompt read), group (prompts a sampling
batch), chunk (probe rows a forward), steps (sampler steps), guidance, dtype (fp32 / bf16 / fp16), sample_dtype,
cost_batch, cost_repeats, do (which reads run: r1, r2, r3, r4, comma separated; controls run with r1 or r2), threads,
det (empty, warn or on).

Output in ``out``: log.txt (its last line 'done'), result.json, result.md (one table a read, the seeds side by side
with their seed gap: the largest minus the smallest of the seeds' numbers) and parts/; the timing variant writes
log.txt and timing.json.
"""
from __future__ import annotations

import os
import platform
import sys
import time
from typing import Dict, List, Optional, Sequence

import torch

from .. import __version__
from ..config import in_notebook, parse
from ..determinism import mode as det_mode
from .capture import SOURCE_DEFAULTS

EXPERIMENT = "band_needs"
VARIANTS = ("reads", "timing")
READS = ("r1", "r2", "r3", "r4")
NAMED = ("base", "small", "tiny")
DEFAULTS = dict(variant="reads", teacher="", repo="runwayml/stable-diffusion-v1-5", revision="451f4fe1", tiny=0,
                data="", out="", device="cpu", seeds="0,1", rows=64, per_band=4, prompts=128, group=32, chunk=16,
                steps=50, guidance=7.5, dtype="fp32", sample_dtype="fp16", cost_batch=16, cost_repeats=3,
                do="r1,r2,r3,r4", threads=4, det="")
DEFAULTS.update(SOURCE_DEFAULTS)              # the source's arguments, as the capture experiment's
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


# ------------------------------------------------------------------ arguments
def _seeds(text: str) -> List[int]:
    seeds = [int(x) for x in str(text).split(",") if x.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds is a list of distinct whole numbers, not {text!r}")
    return seeds


def _do(text: str) -> List[str]:
    do = [x.strip() for x in str(text).split(",") if x.strip()]
    bad = [x for x in do if x not in READS]
    if bad or not do:
        raise ValueError(f"do names reads among {list(READS)}, not {text!r}")
    return [r for r in READS if r in do]


def group_sizes(prompts: int, group: int) -> List[int]:
    """The rows of each prompt group of a seed: groups of ``group``, the last one possibly smaller."""
    prompts, group = int(prompts), int(group)
    if prompts < 2 or group < 2:
        raise ValueError("prompts and group are at least 2")
    sizes = [min(group, prompts - g) for g in range(0, prompts, group)]
    if sizes[-1] < 2:
        raise ValueError(f"the last prompt group would hold {sizes[-1]} row; the prompt read needs at least two a group "
                         f"(choose prompts and group so that prompts % group is 0 or at least 2)")
    return sizes


# ------------------------------------------------------------------ the exact row-forward counts
def r3_slot_steps(steps: int, process=None) -> Dict[str, int]:
    """Per band, how many (sampler step, live part) pairs the assembly runs in ONE sampling with that band's prompt
    changed: on each step the changed band's part runs where its weight is above zero, and the teacher (serving the
    other two bands as one part) where their summed weight is above zero. ``process``: the source's sampler (None: the
    teacher's DDIM)."""
    from ..capture import BANDS
    from ..capture.assemble import ddim_timesteps
    from ..capture.schedule import STEPS, band_weights, band_weights_at
    if process is None or process.kind == "eps":
        w = band_weights(ddim_timesteps(int(steps)))
    else:
        w = band_weights_at(torch.tensor(process.grid(int(steps)), dtype=torch.float64) / STEPS)
    out = {}
    for i, b in enumerate(BANDS):
        others = [j for j in range(len(BANDS)) if j != i]
        out[b] = int((w[:, i] > 0).sum()) + int((w[:, others].sum(dim=1) > 0).sum())
    return out


def row_forwards(seeds: int, rows: int, per_band: int, prompts: int, group: int, steps: int, guidance: float,
                 cost_batch: int, cost_repeats: int, do: Sequence[str] = READS, process=None) -> dict:
    """The exact row-forwards (one row through one U-Net forward) of every unit of the job, by read: the born reads
    with their controls (teacher, engines near the teacher's size: the full copy and the six single changes, reduced
    engines: base, small, tiny and the random base), r4, r3; and the engine builds. ``process``: the source's sampler
    for r3 (None: the teacher's DDIM)."""
    from ..capture import BANDS
    from ..capture.reads import even_steps
    seeds, rows = int(seeds), int(rows)
    probe = rows * sum(len(even_steps(b, int(per_band))) for b in BANDS)     # rows x probe steps over the bands
    units: Dict[str, dict] = {}
    born = "r1" in do or "r2" in do
    for k in range(seeds):
        if born:
            units[f"controls_{k}"] = dict(kind="controls", teacher=probe, engine_full=probe, engine_reduced=2 * probe,
                                          builds=3)
        if "r1" in do:
            units[f"r1_{k}"] = dict(kind="r1", teacher=probe, engine_full=6 * probe, engine_reduced=0, builds=6)
        if "r2" in do:
            units[f"r2_{k}"] = dict(kind="r2", teacher=probe, engine_full=0, engine_reduced=3 * probe, builds=3)
    if "r4" in do:
        each = (1 + max(1, int(cost_repeats))) * int(cost_batch)        # one warm-up and the timed forwards
        units["r4"] = dict(kind="r4", teacher=10 * each, engine_full=7 * each, engine_reduced=3 * each, builds=10)
    if "r3" in do:
        factor = 1 if float(guidance) == 1.0 else 2
        slots = sum(2 * v for v in r3_slot_steps(steps, process).values())    # empty and shuffled, in each band
        for k in range(seeds):
            for g, b in enumerate(group_sizes(prompts, group)):
                plain = (3 if g == 0 else 2) * int(steps)                   # the reference, everywhere, the repeat
                units[f"r3_{k}_{g}"] = dict(kind="r3", teacher=factor * b * (plain + slots), engine_full=0,
                                            engine_reduced=0, builds=0)
    for u in units.values():
        u["total"] = u["teacher"] + u["engine_full"] + u["engine_reduced"]
    by = {}
    for name, keys in (("reads", ("controls", "r1", "r2")), ("r4", ("r4",)), ("r3", ("r3",))):
        sel = [u for u in units.values() if u["kind"] in keys]
        by[name] = {f: sum(u[f] for u in sel) for f in ("teacher", "engine_full", "engine_reduced", "total", "builds")}
    by["total"] = sum(by[n]["total"] for n in ("reads", "r4", "r3"))
    by["units"] = units
    return by


# ------------------------------------------------------------------ the job
def main(argv=(), **overrides) -> dict:
    a = parse(DEFAULTS, argv, **overrides)
    if not a["out"]:
        raise ValueError("out= is required")
    if a["variant"] not in VARIANTS:
        raise ValueError(f"unknown variant {a['variant']!r}; known: {list(VARIANTS)}")
    for k in ("dtype", "sample_dtype"):
        if a[k] not in DTYPES:
            raise ValueError(f"{k} is one of {sorted(DTYPES)}, not {a[k]!r}")
    _seeds(a["seeds"])
    _do(a["do"])
    group_sizes(a["prompts"], a["group"])
    if int(a["rows"]) < 1 or int(a["rows"]) > int(a["prompts"]):
        raise ValueError("rows (the born reads' probe rows a seed) lies in 1 .. prompts")
    with det_mode(a["det"]) as det:
        return _switched(a, det)


def _version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def _card(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(device)
    return f"cpu ({platform.processor() or platform.machine()})"


def _switched(a: dict, det) -> dict:
    """The job with TF32 off and the convolution autotuner off; the three switches restored afterwards."""
    b = torch.backends
    saved = (b.cuda.matmul.allow_tf32, b.cudnn.allow_tf32, b.cudnn.benchmark)
    b.cuda.matmul.allow_tf32 = False
    b.cudnn.allow_tf32 = False
    b.cudnn.benchmark = False
    try:
        switches = {"cuda.matmul.allow_tf32": b.cuda.matmul.allow_tf32, "cudnn.allow_tf32": b.cudnn.allow_tf32,
                    "cudnn.benchmark": b.cudnn.benchmark}
        return _run(a, det, switches)
    finally:
        b.cuda.matmul.allow_tf32, b.cudnn.allow_tf32, b.cudnn.benchmark = saved


class _Job:
    """The job's teacher and data, loaded when a unit first needs them."""

    def __init__(self, a: dict, log):
        self.a = a
        self.log = log
        self.device = torch.device(a["device"])
        self._teacher = None
        self._rows = None
        self._probe_cache: Dict[tuple, tuple] = {}
        self._base_cache: Dict[tuple, tuple] = {}
        from .capture import source_options
        self.source = source_options(a)
        if a["tiny"]:
            self.where = "tiny random teacher"
        else:
            from ..capture.teacher import find_snapshot
            self.where = a["teacher"] or find_snapshot(a["repo"], a["revision"] or None)

    def teacher(self):
        if self._teacher is None:
            from .capture import load_source
            folder = self.a["teacher"] if self.a["tiny"] else self.where
            self._teacher, _ = load_source(dict(self.a, teacher=folder), seed=0)
            self.log(f"teacher loaded: {self.where}" + (f" with the U-Net of {self.a['unet']}" if self.a["unet"] else "")
                     + ("" if self._teacher.kind == "eps" else f" ({self._teacher.record()})")
                     + f" on {self.device} in {self.a['dtype']}")
        return self._teacher

    def process(self):
        """The source's process without loading it (for the job's counts)."""
        from ..capture.process import make_process
        return make_process(self.source["kind"], self.source["shift"])

    def data(self, folder: str, need: int):
        from ..capture.data import LatentBatches, read_index
        index = read_index(folder)
        have = sum(int(c) for c in index["counts"])
        if have < need:
            raise ValueError(f"{folder} holds {have} rows; this job needs {need} (seeds x prompts)")
        self.folder, self.have, self.need = folder, have, need
        self.latent_shape = tuple(LatentBatches(folder).latent_shape)

    def rows(self, first: int, n: int):
        """(latents float32 [n, C, h, w], ids long [n, 77]): rows first .. first + n - 1 of the shards, in file order."""
        if self._rows is None:                          # the job's rows read once (``capture.data.rows_in_order``)
            from ..capture.data import rows_in_order
            self._rows = rows_in_order(self.folder, 0, self.need)
        return self._rows[0][first:first + n], self._rows[1][first:first + n]

    def probe_and_refs(self, k: int, seed: int):
        """Cached (probe, {band: teacher_answers}) for the active seed so controls, r1 and r2 share one pass."""
        key = (int(k), int(seed))
        cached = self._probe_cache.get(key)
        if cached is not None:
            return cached
        from ..capture import BANDS
        from ..capture.reads import teacher_answers
        self._probe_cache.clear()
        teacher = self.teacher()
        probe = _probe(self, k, seed)
        refs = {b: teacher_answers(teacher, probe, b) for b in BANDS}
        self._probe_cache[key] = (probe, refs)
        return probe, refs

    def clear_probe_cache(self) -> None:
        self._probe_cache.clear()
        self._base_cache.clear()


def _write_part(path: str, obj: dict) -> None:
    from ..capture.reads import write_json
    tmp = path + ".tmp"
    write_json(obj, tmp)
    os.replace(tmp, path)


def _read_part(path: str) -> dict:
    import json
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _free(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ the units
def _probe(job: _Job, k: int, seed: int):
    from ..capture.reads import make_probe
    a = job.a
    teacher = job.teacher()
    lat, ids = job.rows(k * int(a["prompts"]), int(a["rows"]))
    states = teacher.text_states(ids)
    return make_probe(states, seed=1000 + int(seed), x0=teacher.source_latents(lat), per_band=int(a["per_band"]),
                      alphas_cumprod=teacher.alphas_cumprod, batch=int(a["chunk"]), process=teacher.process)


def _controls(job: _Job, k: int, seed: int, log) -> dict:
    from ..capture import BANDS
    from ..capture.engine import build_engine, fresh_engine, param_count
    from ..capture.reads import band_read
    teacher = job.teacher()
    probe, refs = job.probe_and_refs(k, seed)
    out = {}
    for name, make in (("full", lambda: build_engine(teacher, "full")),
                       ("inherited_base", lambda: build_engine(teacher, "base")),
                       ("random_base", lambda: fresh_engine(teacher, "base", seed=int(seed)))):
        eng = make().eval()
        out[name] = {b: float(band_read(eng, teacher, b, probe, refs[b])["mse"]) for b in BANDS}
        if name == "inherited_base":
            job._base_cache[(int(k), int(seed))] = (dict(out[name]), int(param_count(eng)))
        del eng
        _free(job.device)
    out["ratio"] = {b: (out["random_base"][b] / out["inherited_base"][b] if out["inherited_base"][b] > 0 else None)
                    for b in BANDS}
    log(f"controls seed {seed}: full copy " + "  ".join(f"{b} {out['full'][b]:.4g}" for b in BANDS)
        + " | random / inherited base " + "  ".join(f"{b} {out['ratio'][b] or float('nan'):.4g}" for b in BANDS))
    return out


def _r1(job: _Job, k: int, seed: int, log) -> dict:
    from ..capture import measure
    probe, refs = job.probe_and_refs(k, seed)
    r = measure.removal_scores(job.teacher(), probe, log=log, references=refs)
    _free(job.device)
    return {"table": r["table"], "rows": r["rows"], "teacher_params": r["teacher_params"]}


def _r2(job: _Job, k: int, seed: int, log) -> dict:
    from ..capture import BANDS
    from ..capture.engine import build_engine, param_count
    from ..capture.reads import band_read
    teacher = job.teacher()
    probe, refs = job.probe_and_refs(k, seed)
    total = param_count(teacher)
    out: Dict[str, dict] = {b: {} for b in BANDS}
    for name in NAMED:
        if name == "base" and (int(k), int(seed)) in job._base_cache:
            mse_by_band, p = job._base_cache[(int(k), int(seed))]
            for b in BANDS:
                out[b][name] = {"mse": mse_by_band[b], "params": int(p), "share": p / total}
            log(f"r2 seed {seed}: {name} " + "  ".join(f"{b} {out[b][name]['mse']:.4g}" for b in BANDS))
            continue
        eng = build_engine(teacher, name).eval()
        p = param_count(eng)
        for b in BANDS:
            out[b][name] = {"mse": float(band_read(eng, teacher, b, probe, refs[b])["mse"]), "params": int(p),
                            "share": p / total}
        log(f"r2 seed {seed}: {name} " + "  ".join(f"{b} {out[b][name]['mse']:.4g}" for b in BANDS))
        del eng
        _free(job.device)
    return out


def _r4(job: _Job, log) -> list:
    from ..capture import measure
    from ..capture.engine import forward_seconds
    from ..capture.shapes import SHAPES
    a = job.a
    teacher = job.teacher()
    unet = teacher.unet
    batch = int(a["cost_batch"])
    size = int(job.latent_shape[1])
    repeats = int(a["cost_repeats"])
    dev = next(unet.parameters()).device
    dtype = next(unet.parameters()).dtype
    g = torch.Generator().manual_seed(0)
    z = torch.randn(batch, unet.config.in_channels, size, size, generator=g).to(dev, dtype)
    ctx = torch.randn(batch, 77, int(unet.config.cross_attention_dim), generator=g).to(dev, dtype)
    t = torch.full((batch,), 500, device=dev, dtype=torch.long)
    ts = forward_seconds(unet, z, t, ctx, repeats)
    del z, ctx, t
    shapes = [("full", SHAPES["full"])] + measure.single_changes() + [(n, SHAPES[n]) for n in NAMED]
    out = []
    for name, shape in shapes:
        c = measure.cost(teacher, shape, batch=batch, size=size, repeats=repeats, teacher_seconds=ts)
        out.append({"name": name, "shape": shape.text, "params": c["params"], "share": c["share"],
                    "seconds": c["seconds"], "teacher_seconds": c["teacher_seconds"], "cost": c["cost"],
                    "batch": c["batch"], "size": c["size"]})
        log(f"r4: {name} ({shape.text}) share {c['share']:.3f}, {c['seconds']:.4g} s against the teacher's "
            f"{c['teacher_seconds']:.4g} s a forward ({c['cost']:.3f})")
        _free(job.device)
    return out


def _r3(job: _Job, k: int, seed: int, g: int, log) -> dict:
    from ..capture import measure
    a = job.a
    first = k * int(a["prompts"]) + g * int(a["group"])
    size = group_sizes(a["prompts"], a["group"])[g]
    _, ids = job.rows(first, size)
    r = measure.text_need(job.teacher(), ids, seed=100000 * (int(seed) + 1) + g, steps=int(a["steps"]),
                          guidance=float(a["guidance"]), latent_shape=job.latent_shape, log=log, repeat=(g == 0))
    _free(job.device)
    r["group"], r["first_row"] = int(g), int(first)
    return r


def _r3_seed(groups: List[dict]) -> dict:
    """A seed's prompt read from its groups: the means weighted by rows, and the shares of the empty-everywhere change."""
    from ..capture import BANDS
    n = sum(int(x["prompts"]) for x in groups)
    mean = lambda f: sum(int(x["prompts"]) * f(x) for x in groups) / n          # noqa: E731
    everywhere = mean(lambda x: x["empty_everywhere"])
    table = {}
    for b in BANDS:
        e = mean(lambda x: x["table"][b]["empty"])
        s = mean(lambda x: x["table"][b]["shuffled"])
        table[b] = {"empty": e, "shuffled": s, "empty_share": e / everywhere if everywhere > 0 else None,
                    "shuffled_share": s / everywhere if everywhere > 0 else None}
    return {"table": table, "empty_everywhere": everywhere, "repeat": groups[0]["repeat"], "prompts": n,
            "groups": groups}


# ------------------------------------------------------------------ running
def _run(a: dict, det, switches: dict) -> dict:
    out = a["out"]
    os.makedirs(out, exist_ok=True)
    log_path = os.path.join(out, "log.txt")

    def log(line: str) -> None:
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")

    torch.set_num_threads(int(a["threads"]))
    t0 = time.time()
    job = _Job(a, log)
    seeds = _seeds(a["seeds"])
    device = job.device
    sample_dtype = a["sample_dtype"] if device.type == "cuda" else "fp32"
    head = {"experiment": EXPERIMENT, "variant": a["variant"],
            "args": {k: a[k] for k in sorted(a) if not (k == "det" and not a[k])}, "torch": torch.__version__,
            "version": __version__, "diffusers": _version("diffusers"), "transformers": _version("transformers"),
            "device": str(device), "card": _card(device), "switches": switches, "teacher": job.where,
            "sample_dtype_used": sample_dtype}
    if a["kind"] != "eps" or a["unet"]:
        from ..capture.teacher import UNGUIDED_DEFAULT
        p = job.process()
        head["process"] = dict(p.record(), unguided=job.source["unguided"] or UNGUIDED_DEFAULT[p.kind],
                               kappa=job.source["kappa"], unet=a["unet"])
    need = len(seeds) * int(a["prompts"])
    data = a["data"]
    if not data:
        if not a["tiny"]:
            raise ValueError("data= (a folder of latent shards) is required with the real teacher")
        if a["variant"] == "reads":
            data = os.path.join(out, "synthetic")
            from ..capture.data import read_index, synthetic_shards
            have = 0
            if os.path.exists(os.path.join(data, "index.json")):
                have = sum(read_index(data)["counts"])
            if have < need:
                size = int(job.teacher().unet.config.sample_size)
                synthetic_shards(data, rows=need, shard_size=max(2, int(a["group"])), size=size, seed=0)
    if a["variant"] == "timing":
        return _timing(a, job, data, sample_dtype, head, log, t0)
    job.data(data, need)
    head["data"] = {"folder": data, "rows": job.have, "used": need, "latent_shape": list(job.latent_shape)}
    return _reads(a, job, seeds, sample_dtype, head, log, det, t0)


def _reads(a, job: _Job, seeds: List[int], sample_dtype: str, head: dict, log, det, t0: float) -> dict:
    from ..capture.reads import write_json
    out = a["out"]
    parts_dir = os.path.join(out, "parts")
    os.makedirs(parts_dir, exist_ok=True)
    do = _do(a["do"])
    count = row_forwards(len(seeds), a["rows"], a["per_band"], a["prompts"], a["group"], a["steps"], a["guidance"],
                         a["cost_batch"], a["cost_repeats"], do, job.process())
    # the units in their running order, each with its file name and its row-forwards
    plan = []
    for k, s in enumerate(seeds):
        for kind in ("controls", "r1", "r2"):
            key = f"{kind}_{k}"
            if key in count["units"]:
                plan.append((f"{kind}_s{s}", kind, k, s, None, count["units"][key]))
    if "r4" in count["units"]:
        plan.append(("r4", "r4", None, None, None, count["units"]["r4"]))
    for k, s in enumerate(seeds):
        for g in range(len(group_sizes(a["prompts"], a["group"]))):
            key = f"r3_{k}_{g}"
            if key in count["units"]:
                plan.append((f"r3_s{s}_g{g:03d}", "r3", k, s, g, count["units"][key]))
    log(f"BAND_NEEDS reads: teacher {job.where}; device {head['card']}; torch {torch.__version__}; package "
        f"{__version__}; born reads in {a['dtype']}, sampling in {sample_dtype}")
    log(f"job size: {len(plan)} units; row-forwards: born reads with controls {count['reads']['total']:,} (teacher "
        f"{count['reads']['teacher']:,}, engines {count['reads']['engine_full'] + count['reads']['engine_reduced']:,}),"
        f" r4 {count['r4']['total']:,}, r3 {count['r3']['total']:,}; {count['total']:,} in all")
    values: Dict[str, object] = {}
    seconds: Dict[str, float] = {}
    ran: List[str] = []
    back: List[str] = []
    cast = False
    for i, (name, kind, k, s, g, size) in enumerate(plan):
        path = os.path.join(parts_dir, name + ".json")
        if os.path.exists(path):
            part = _read_part(path)
            values[name] = part["value"]
            seconds[name] = float(part.get("seconds") or 0.0)
            back.append(name)
            what = "read back"
        else:
            if kind == "r3" and not cast:
                job.clear_probe_cache()
                job.teacher().to(dtype=DTYPES[sample_dtype])
                job.log(f"teacher cast to {sample_dtype} for the prompt read")
                cast = True
            u0 = time.time()
            if kind == "controls":
                v = _controls(job, k, s, log)
            elif kind == "r1":
                v = _r1(job, k, s, log)
            elif kind == "r2":
                v = _r2(job, k, s, log)
            elif kind == "r4":
                v = _r4(job, log)
            else:
                v = _r3(job, k, s, g, log)
            took = time.time() - u0
            _write_part(path, {"unit": name, "kind": kind, "seconds": took, "row_forwards": size["total"], "value": v})
            values[name] = v
            seconds[name] = took
            ran.append(name)
            what = f"ran in {took:.1f} s"
        log(_progress(name, what, i + 1, plan, seconds, t0))
    result = dict(head)
    result["row_forwards"] = {k: v for k, v in count.items() if k != "units"}
    result["units"] = {"ran": ran, "read_back": back, "total": len(plan)}
    result["controls"], result["r1"], result["r2"], result["r3"] = {}, {}, {}, {}
    for k, s in enumerate(seeds):
        key = str(s)
        if f"controls_s{s}" in values:
            result["controls"][key] = dict(values[f"controls_s{s}"])
        if f"r1_s{s}" in values:
            result["r1"][key] = values[f"r1_s{s}"]
        if f"r2_s{s}" in values:
            result["r2"][key] = values[f"r2_s{s}"]
        groups = [values[n] for n, kind, kk, *_ in plan if kind == "r3" and kk == k]
        if groups:
            result["r3"][key] = _r3_seed(groups)
            result["controls"].setdefault(key, {})["repeat"] = result["r3"][key]["repeat"]
    result["r4"] = values.get("r4", [])
    result["seconds"] = time.time() - t0
    det.record(result)
    write_json(result, os.path.join(out, "result.json"))
    text = report(result)
    with open(os.path.join(out, "result.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    log(text.rstrip())
    log("done")
    return result


def _progress(name: str, what: str, done: int, plan: list, seconds: Dict[str, float], t0: float) -> str:
    """The progress line after a unit: units done, minutes spent, minutes left from the measured seconds per
    row-forward of the finished units of the same kind (of all finished units while a kind has none)."""
    rate: Dict[str, List[float]] = {}
    for n, kind, *_rest, size in plan:
        if n in seconds and seconds[n] > 0 and size["total"] > 0:
            r = rate.setdefault(kind, [0.0, 0.0])
            r[0] += seconds[n]
            r[1] += size["total"]
    every = [sum(r[0] for r in rate.values()), sum(r[1] for r in rate.values())]
    left, unknown = 0.0, False
    for n, kind, *_rest, size in plan[done:]:
        r = rate.get(kind) or (every if every[1] > 0 else None)
        if r is None:
            unknown = True
            continue
        left += size["total"] * r[0] / r[1]
    spent = (time.time() - t0) / 60
    tail = "time left not known yet" if unknown and left == 0 else f"about {left / 60:.1f} min left"
    return f"unit {name} {what}: {done}/{len(plan)} units done; {spent:.1f} min spent; {tail}"


# ------------------------------------------------------------------ the timing variant
@torch.no_grad()
def _seconds(part, z, t, states, repeats: int = 3) -> float:
    """The mean seconds of one forward over ``repeats`` after one untimed warm-up (synchronized on a card)."""
    from ..capture.engine import as_part
    p = as_part(part)
    sync = torch.cuda.synchronize if z.is_cuda else (lambda: None)
    p(z, t, states)
    sync()
    total = 0.0
    for _ in range(int(repeats)):
        sync()
        s0 = time.perf_counter()
        p(z, t, states)
        sync()
        total += time.perf_counter() - s0
    return total / int(repeats)


def _timing(a: dict, job: _Job, data: str, sample_dtype: str, head: dict, log, t0: float) -> dict:
    from ..capture.engine import build_engine
    from ..capture.reads import write_json
    seeds = _seeds(a["seeds"])
    teacher = job.teacher()
    if data:
        from ..capture.data import LatentBatches
        shape = tuple(LatentBatches(data).latent_shape)
    else:
        cfg = teacher.unet.config
        shape = (int(cfg.in_channels), int(cfg.sample_size), int(cfg.sample_size))
    dev, dim = job.device, int(teacher.context_dim)
    g = torch.Generator().manual_seed(0)

    def inputs(n: int, dtype):
        z = torch.randn((n,) + shape, generator=g).to(dev)
        states = torch.randn(n, 77, dim, generator=g).to(dev, dtype)
        return z, torch.full((n,), 500, dtype=torch.long, device=dev), states

    chunk, pair = int(a["chunk"]), 2 * int(a["group"])
    log(f"BAND_NEEDS timing: teacher {job.where}; device {head['card']}; latents {shape}")
    sec = {}
    sec["teacher_dtype_chunk"] = _seconds(teacher, *inputs(chunk, DTYPES[a["dtype"]]))
    b0 = time.perf_counter()
    eng = build_engine(teacher, "base").eval()
    sec["build_base"] = time.perf_counter() - b0
    sec["base_chunk"] = _seconds(eng, *inputs(chunk, torch.float32))
    del eng
    _free(dev)
    teacher.to(dtype=DTYPES[sample_dtype])
    sec["teacher_sample_dtype_pair"] = _seconds(teacher, *inputs(pair, DTYPES[sample_dtype]))
    rf = row_forwards(len(seeds), a["rows"], a["per_band"], a["prompts"], a["group"], a["steps"], a["guidance"],
                      a["cost_batch"], a["cost_repeats"], READS, job.process())
    t_row = sec["teacher_dtype_chunk"] / chunk
    e_row = sec["base_chunk"] / chunk
    s_row = sec["teacher_sample_dtype_pair"] / pair

    def born(c):         # engines near the teacher's size at the teacher's rate, the reduced ones at the base's
        return (c["teacher"] + c["engine_full"]) * t_row + c["engine_reduced"] * e_row + c["builds"] * sec["build_base"]

    minutes = {"reads": born(rf["reads"]) / 60, "r4": born(rf["r4"]) / 60, "r3": rf["r3"]["total"] * s_row / 60}
    minutes["total"] = minutes["reads"] + minutes["r4"] + minutes["r3"]
    result = dict(head)
    result.update({"latent_shape": list(shape), "seconds_a_forward": sec,
                   "rows_a_forward": {"teacher_dtype_chunk": chunk, "base_chunk": chunk,
                                      "teacher_sample_dtype_pair": pair},
                   "row_forwards": {k: v for k, v in rf.items() if k != "units"}, "minutes": minutes,
                   "seconds": time.time() - t0})
    write_json(result, os.path.join(a["out"], "timing.json"))
    log(f"seconds a forward: teacher in {a['dtype']} at {chunk} rows {sec['teacher_dtype_chunk']:.4g}; base engine at "
        f"{chunk} rows {sec['base_chunk']:.4g}; teacher in {sample_dtype} at {pair} rows "
        f"{sec['teacher_sample_dtype_pair']:.4g}; one engine build {sec['build_base']:.3g}")
    log(f"projected minutes: born reads with controls {minutes['reads']:.1f}, r4 {minutes['r4']:.1f}, r3 "
        f"{minutes['r3']:.1f}; total {minutes['total']:.1f} (row-forwards {rf['reads']['total']:,} / "
        f"{rf['r4']['total']:,} / {rf['r3']['total']:,})")
    log("done")
    return result


# ------------------------------------------------------------------ the report
def _gap(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return (max(vals) - min(vals)) if len(vals) == len(values) and len(vals) > 1 else None


def report(result: dict) -> str:
    """result.md: one table a read; for every read made on several seeds, the seeds side by side with their gap."""
    from ..capture import BANDS
    from ..capture.reads import table
    seeds = sorted({s for r in ("controls", "r1", "r2", "r3") for s in result.get(r, {})}, key=int)
    cols = [f"seed {s}" for s in seeds]
    lines = [f"# band needs: {result.get('teacher')}", "",
             f"device {result.get('card')}; torch {result.get('torch')}; package {result.get('version')}; "
             f"row-forwards {result.get('row_forwards', {}).get('total', 0):,}", ""]

    def side(rows_in, get):
        rows = []
        for r in rows_in:
            vals = [get(s, r) for s in seeds]
            row = dict(r)
            row.update({c: v for c, v in zip(cols, vals)})
            row["seed gap"] = _gap(vals)
            rows.append(row)
        return rows

    ctl = result.get("controls", {})
    if ctl:
        lines += ["## controls (the full copy must read 0; the random base against the inherited one; the repeated "
                  "reference must read 0)", ""]
        rows = []
        for what in ("full", "inherited_base", "random_base", "ratio"):
            if all(what in ctl[s] for s in seeds if s in ctl):
                rows += side([{"read": what, "band": b} for b in BANDS],
                             lambda s, r: (ctl.get(s, {}).get(r["read"]) or {}).get(r["band"]))
        if all("repeat" in ctl.get(s, {}) for s in seeds):
            rows += side([{"read": "repeat", "band": "all"}], lambda s, r: ctl[s]["repeat"])
        lines += [table(rows, ["read", "band"] + cols + ["seed gap"])]
    r1 = result.get("r1", {})
    if r1:
        from ..capture.measure import single_changes
        lines += ["## r1: the born read of every single change", ""]
        rows = side([{"band": b, "change": c} for b in BANDS for c, _ in single_changes()],
                    lambda s, r: r1.get(s, {}).get("table", {}).get(r["band"], {}).get(r["change"]))
        lines += [table(rows, ["band", "change"] + cols + ["seed gap"])]
    r2 = result.get("r2", {})
    if r2:
        lines += ["## r2: the born read of the named shapes", ""]
        any_seed = r2[next(iter(r2))]
        rows = side([{"band": b, "shape": n, "params": any_seed[b][n]["params"], "share": any_seed[b][n]["share"]}
                     for b in BANDS for n in NAMED],
                    lambda s, r: r2.get(s, {}).get(r["band"], {}).get(r["shape"], {}).get("mse"))
        lines += [table(rows, ["band", "shape", "params", "share"] + cols + ["seed gap"])]
    r3 = result.get("r3", {})
    if r3:
        lines += ["## r3: the prompt's weight per band (mean squared change of the final latent; shares of the "
                  "empty-everywhere change)", ""]
        rows = side([{"band": b, "read": q} for b in BANDS for q in ("empty", "shuffled", "empty_share",
                                                                      "shuffled_share")],
                    lambda s, r: r3.get(s, {}).get("table", {}).get(r["band"], {}).get(r["read"]))
        rows += side([{"band": "all", "read": q} for q in ("empty_everywhere", "repeat")],
                     lambda s, r: r3.get(s, {}).get(r["read"]))
        lines += [table(rows, ["band", "read"] + cols + ["seed gap"])]
    r4 = result.get("r4", [])
    if r4:
        lines += ["## r4: size and measured seconds a forward", ""]
        lines += [table(r4, ["name", "shape", "params", "share", "seconds", "teacher_seconds", "cost", "batch",
                             "size"])]
    return "\n".join(lines).rstrip() + "\n"


def summarize(root: str) -> str:
    """The report of every job folder under ``root`` holding a result.json of this experiment, one after another."""
    import json
    parts = []
    for dirpath, _, files in sorted(os.walk(root)):
        if "result.json" in files:
            with open(os.path.join(dirpath, "result.json"), encoding="utf-8") as fh:
                r = json.load(fh)
            if r.get("experiment") == EXPERIMENT:
                parts.append(f"# job {os.path.relpath(dirpath, root)}\n\n" + report(r))
    return "\n".join(parts)


if __name__ == "__main__" and not in_notebook():
    main(sys.argv[1:])
