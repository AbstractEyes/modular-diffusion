"""Capture: band engines made from Stable Diffusion 1.5 (the teacher) and read against it.

    python -m geolip.modular_diffusion.run capture variant=band_engine shape=base band=clean data=<shards> out=<folder>
    from geolip.modular_diffusion.experiments import capture; capture.main(variant="born", tiny=1, out=...)

| variant      | what it does                                                                                        |
|--------------|-----------------------------------------------------------------------------------------------------|
| band_engine  | trains one band engine alone (``capture.train``); reads it as born and, with its averaged copy, on  |
|              | its own band and on the two others (the off-band control); the teacher's copy read beside them      |
| born         | reads only: on each band, the teacher's copy (reads 0) and the shape's engine as born               |
| assembly     | reads of an assembly (``parts=``) on each band, and the change of its sampled final latents against |
|              | the teacher's                                                                                       |

The teacher is a local diffusers folder: ``teacher=<folder>``, or found in the local hub cache by ``repo`` and
``revision`` (nothing is ever downloaded). ``tiny=1`` uses a tiny random teacher and random batches instead (16 x 16
latents; for smoke runs and the tests).

Arguments (key=value): variant, teacher, repo, revision, tiny, shape (a name full / base / small / tiny or
<layers>/<mid|nomid>/<stages>), band (clean / middle / noisy), rho (the whole-range share of the band's training law),
steps, batch, lr, lam_out, lam_data, drop (the empty-prompt share), text (0: an engine that ignores the prompt), seed,
data (a folder of latent shards, read by ``capture.data.LatentBatches``), out, device, dtype (fp32 / bf16 / fp16: the
teacher's numbers; an engine's are float32), autocast (auto / none / bf16 / fp16), save_every, log_every,
average_decay, probe_n, probe_per_band, probe_seed, sample_steps, guidance, parts (for assembly: band=teacher or
band=<a band engine's job folder>, comma separated, for example clean=runs/a,middle=teacher,noisy=teacher), threads,
det (empty, warn or on).

Output in ``out``: log.txt (its last line 'done'), result.json, result.md (the reads as a table); band_engine also
writes the job's checkpoint.pt and the engine files (``capture.train``), and resumes from a checkpoint found there.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Callable, Dict

import torch

from .. import __version__
from ..config import in_notebook, parse
from ..determinism import mode as det_mode

EXPERIMENT = "capture"
VARIANTS = ("band_engine", "born", "assembly")
PROBE_PROMPTS = ("a photograph of a red house by a lake", "a bowl of fruit on a wooden table",
                 "a portrait of an old man with a gray beard", "a city street at dusk with lit windows",
                 "a mountain road in the snow", "a cat sleeping on a blue sofa",
                 "a sailing boat on a calm sea", "a plate of pasta with basil")
DEFAULTS = dict(variant="band_engine", teacher="", repo="runwayml/stable-diffusion-v1-5", revision="451f4fe1", tiny=0,
                shape="base", band="clean", rho=0.0, steps=1000, batch=8, lr=5e-5, lam_out=1.0, lam_data=0.0,
                drop=0.1, text=1, seed=0, data="", out="", device="cpu", dtype="fp32", autocast="auto",
                save_every=500, log_every=50, average_decay=0.9999, probe_n=4, probe_per_band=4, probe_seed=1234,
                sample_steps=50, guidance=7.5, parts="", threads=4, det="")
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def main(argv=(), **overrides) -> dict:
    a = parse(DEFAULTS, argv, **overrides)
    if not a["out"]:
        raise ValueError("out= is required")
    if a["variant"] not in VARIANTS:
        raise ValueError(f"unknown variant {a['variant']!r}; known: {list(VARIANTS)}")
    with det_mode(a["det"]) as det:
        return _run(a, det)


def random_source(seed: int, latent_shape, vocab: int = 1000) -> Callable:
    """A batch source of random latents and ids (a pure function of the step), for smoke runs."""
    def source(step: int, batch: int):
        g = torch.Generator().manual_seed(7_919 * (int(seed) + 1) + int(step))
        x0 = torch.randn((int(batch),) + tuple(latent_shape), generator=g)
        ids = torch.randint(2, vocab, (int(batch), 77), generator=g)
        ids[:, 0] = 1
        return x0, ids
    return source


def _teacher(a: dict):
    from ..capture.teacher import find_snapshot, load_teacher, tiny_teacher
    if a["tiny"]:
        return tiny_teacher(int(a["seed"])).to(a["device"], DTYPES[a["dtype"]]), "tiny random teacher"
    path = a["teacher"] or find_snapshot(a["repo"], a["revision"] or None)
    return load_teacher(path, device=a["device"], dtype=DTYPES[a["dtype"]]), path


def _latent_shape(teacher):
    cfg = teacher.unet.config
    return (int(cfg.in_channels), int(cfg.sample_size), int(cfg.sample_size))


def _source(a: dict, teacher):
    if a["tiny"] or not a["data"]:
        if not a["tiny"] and a["variant"] == "band_engine":
            raise ValueError("data= (a folder of latent shards) is required to train on the real teacher")
        return random_source(int(a["seed"]), _latent_shape(teacher))
    from ..capture.data import LatentBatches
    return LatentBatches(a["data"], seed=int(a["seed"]))


def _probe(a: dict, teacher):
    from ..capture.reads import make_probe
    n = int(a["probe_n"])
    prompts = [PROBE_PROMPTS[i % len(PROBE_PROMPTS)] for i in range(n)]
    states = teacher.text_states(teacher.tokenize(prompts))
    return make_probe(states, seed=int(a["probe_seed"]), per_band=int(a["probe_per_band"]),
                      latent_shape=_latent_shape(teacher), empty=teacher.empty_states(1),
                      alphas_cumprod=teacher.alphas_cumprod)


def _parts(a: dict, teacher) -> Dict[str, object]:
    from ..capture import BANDS
    from ..capture.engine import load_engine
    spec = dict(item.split("=", 1) for item in a["parts"].split(",") if item.strip())
    spec = {k.strip(): v.strip() for k, v in spec.items()}
    if set(spec) != set(BANDS):
        raise ValueError(f"parts= names each band {list(BANDS)} once: {a['parts']!r}")
    loaded: Dict[str, object] = {}
    out = {}
    for b in BANDS:
        v = spec[b]
        if v == "teacher":
            out[b] = teacher
        else:
            if v not in loaded:
                loaded[v] = load_engine(teacher, v, average=True)
            out[b] = loaded[v]
    return out


def _run(a: dict, det) -> dict:
    from ..capture import BANDS
    from ..capture import reads
    from ..capture.assemble import Assembly, sample
    from ..capture.train import train_band_engine
    out = a["out"]
    os.makedirs(out, exist_ok=True)
    log_path = os.path.join(out, "log.txt")

    def log(line: str) -> None:
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")

    torch.set_num_threads(int(a["threads"]))
    t0 = time.time()
    teacher, where = _teacher(a)
    log(f"CAPTURE {a['variant']}: teacher {where}; torch {torch.__version__}; package {__version__}")
    probe = _probe(a, teacher)
    refs = {b: reads.teacher_answers(teacher, probe, b) for b in BANDS}
    rows = []
    result = {"experiment": EXPERIMENT, "variant": a["variant"], "teacher": where,
              "args": {k: a[k] for k in sorted(a) if not (k == "det" and not a[k])}, "torch": torch.__version__,
              "version": __version__, "probe_steps": {b: [int(t) for t in probe.steps[b]] for b in BANDS}}
    if a["variant"] == "born":
        for b in BANDS:
            copy_read = reads.band_read(teacher, teacher, b, probe, refs[b])
            born = reads.born_read(teacher, a["shape"], b, probe, text=bool(a["text"]), reference=refs[b])
            rows.append({"band": b, "part": "teacher copy", "mse": copy_read["mse"]})
            rows.append({"band": b, "part": f"{born['shape']} as born", "mse": born["mse"], "share": born["share"]})
            result.setdefault("reads", {})[b] = {"teacher_copy": copy_read, "born": born}
    elif a["variant"] == "band_engine":
        band = a["band"]
        born = reads.born_read(teacher, a["shape"], band, probe, text=bool(a["text"]), reference=refs[band])
        rows.append({"band": band, "part": f"{born['shape']} as born", "mse": born["mse"], "share": born["share"]})
        tr = train_band_engine(teacher, a["shape"], band, _source(a, teacher), steps=int(a["steps"]),
                               batch=int(a["batch"]), seed=int(a["seed"]), out=out, rho=float(a["rho"]),
                               lr=float(a["lr"]), lam_out=float(a["lam_out"]), lam_data=float(a["lam_data"]),
                               drop=float(a["drop"]), text=bool(a["text"]), average_decay=float(a["average_decay"]),
                               save_every=int(a["save_every"]), log_every=int(a["log_every"]),
                               autocast=a["autocast"], log=log)
        avg = tr["average"].eval()
        result["train"] = {"step": tr["step"], "finished": tr["finished"], "first": tr["log"][:1],
                           "last": tr["log"][-1:]}
        result["reads"] = {"born": born}
        for b in BANDS:
            r = reads.band_read(avg, teacher, b, probe, refs[b])
            result["reads"]["trained_" + b] = r
            rows.append({"band": b, "part": "trained (averaged)" + ("" if b == band else ", off its band"),
                         "mse": r["mse"]})
        result["reads"]["teacher_copy"] = reads.band_read(teacher, teacher, band, probe, refs[band])
        rows.append({"band": band, "part": "teacher copy", "mse": result["reads"]["teacher_copy"]["mse"]})
    else:
        model = Assembly(_parts(a, teacher))
        result["parts"] = a["parts"]
        for b in BANDS:
            r = reads.band_read(model, teacher, b, probe, refs[b])
            result.setdefault("reads", {})[b] = r
            rows.append({"band": b, "part": "assembly", "mse": r["mse"]})
        if int(a["sample_steps"]) > 0:
            shape = tuple(probe.x0.shape[1:])
            zt = sample(teacher, probe.states, probe.empty, probe.seed, int(a["sample_steps"]), float(a["guidance"]),
                        latent_shape=shape, alphas_cumprod=probe.alphas_cumprod)
            za = sample(model, probe.states, probe.empty, probe.seed, int(a["sample_steps"]), float(a["guidance"]),
                        latent_shape=shape, alphas_cumprod=probe.alphas_cumprod)
            result["final_latent_change"] = float((za.double() - zt.double()).pow(2).mean())
            rows.append({"band": "all", "part": "final latents against the teacher's",
                         "mse": result["final_latent_change"]})
    result["rows"] = rows
    result["seconds"] = time.time() - t0
    det.record(result)
    reads.write_json(result, os.path.join(out, "result.json"))
    text = f"# capture: {a['variant']}\n\n" + reads.table(rows, ["band", "part", "mse", "share"])
    with open(os.path.join(out, "result.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    log(text.rstrip())
    log("done")
    return result


def summarize(root: str) -> str:
    """A markdown table of every read row of the job folders under ``root`` holding a result.json of this experiment."""
    import json
    from ..capture.reads import table
    rows = []
    for dirpath, _, files in sorted(os.walk(root)):
        if "result.json" in files:
            with open(os.path.join(dirpath, "result.json"), encoding="utf-8") as fh:
                r = json.load(fh)
            if r.get("experiment") == EXPERIMENT:
                job = os.path.relpath(dirpath, root)
                for row in r.get("rows", []):
                    rows.append(dict(row, job=job, variant=r["variant"]))
    return table(rows, ["job", "variant", "band", "part", "mse", "share"])


if __name__ == "__main__" and not in_notebook():
    main(sys.argv[1:])
