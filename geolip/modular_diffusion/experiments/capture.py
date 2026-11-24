"""Capture: band engines made from Stable Diffusion 1.5 (the teacher) and read against it.

    python -m geolip.modular_diffusion.run capture variant=band_engine shape=base band=clean data=<shards> out=<folder>
    from geolip.modular_diffusion.experiments import capture; capture.main(variant="born", tiny=1, out=...)

| variant      | what it does                                                                                        |
|--------------|-----------------------------------------------------------------------------------------------------|
| band_engine  | trains one band engine alone (``capture.train``), from the teacher's numbers or from random ones    |
|              | (``init``); reads it as born, during training (``read_every``) and, with its averaged copy, at the  |
|              | end on its own band and on the two others (the off-band control); the teacher's copy read beside    |
| born         | reads only: on each band, the teacher's copy and the engine of the shape full with the prompt (both |
|              | read exactly 0), and the shape's engine as born                                                     |
| assembly     | reads of an assembly (``parts=``) on each band, and the change of its sampled final latents against |
|              | the teacher's                                                                                       |
| exchange     | one band's part of an assembly (``parts=``) replaced by ``other`` (a band engine's job folder or    |
|              | teacher): every band's read before and after, the largest change where that band is silent, and the |
|              | change of the sampled final latents (``capture.reads.exchange``)                                     |

The teacher is a local diffusers folder: ``teacher=<folder>``, or found in the local hub cache by ``repo`` and
``revision`` (nothing is ever downloaded). ``tiny=1`` uses a tiny random teacher and random batches instead (16 x 16
latents; for smoke runs and the tests).

THE PROBE: ``probe_n`` fixed prompts with standard normal latents; or, with ``probe_data``, the rows ``probe_first`` ..
``probe_first + probe_rows - 1`` of that folder of latent shards in FILE order (``capture.data.rows_in_order``): their
clean latents as x0 and their token ids through the teacher's text encoder. Either way the noise comes from
``probe_seed``, each band is read on ``probe_per_band`` even steps, and the reads put ``chunk`` rows through a forward.
Sampling reads (assembly, exchange) sample the probe's rows ``group`` at a time (``capture.reads.sample_probe``: group j
starts from the noise of probe_seed + 100003 j, the same for every model compared).

READS DURING TRAINING (band_engine, ``read_every`` > 0): every ``read_every`` updates and after the last one, the live
engine and its averaged copy are read on the band's own probe steps against the teacher's answers (computed once);
``result["curve"]`` = [{"step", "live", "averaged"}, ...], its first row the engine as born (step 0).

Arguments (key=value): variant, teacher, repo, revision, tiny, shape (a name full / base / small / tiny or
<layers>/<mid|nomid>/<stages>), band (clean / middle / noisy; for exchange the band whose part is replaced), rho (the
whole-range share of the band's training law), steps, batch, lr, lam_out, lam_data, drop (the empty-prompt share), text
(0: an engine that ignores the prompt), seed, init (inherit / random), read_every, keep_checkpoint (0: checkpoint.pt is
removed once the job has finished), data (a folder of latent shards, read by ``capture.data.LatentBatches``), out,
device, dtype (fp32 / bf16 / fp16: the teacher's numbers; an engine's are float32), autocast (auto / none / bf16 /
fp16), save_every, log_every, average_decay, probe_n, probe_per_band, probe_seed, probe_data, probe_first, probe_rows,
chunk, sample_steps, guidance, group, parts (for assembly and exchange: band=teacher or band=<a band engine's job
folder>, comma separated, for example clean=runs/a,middle=teacher,noisy=teacher), other (for exchange), threads, det
(empty, warn or on).

Output in ``out``: log.txt (its last line 'done'), result.json, result.md (the reads as a table); band_engine also
writes the job's checkpoint.pt and the engine files (``capture.train``), and resumes from a checkpoint found there (a
finished job whose checkpoint was removed is read back, not trained again).
"""
from __future__ import annotations

import os
import sys
import time
from typing import Callable, Dict, Optional

import torch

from .. import __version__
from ..config import in_notebook, parse
from ..determinism import mode as det_mode

EXPERIMENT = "capture"
VARIANTS = ("band_engine", "born", "assembly", "exchange")
PROBE_PROMPTS = ("a photograph of a red house by a lake", "a bowl of fruit on a wooden table",
                 "a portrait of an old man with a gray beard", "a city street at dusk with lit windows",
                 "a mountain road in the snow", "a cat sleeping on a blue sofa",
                 "a sailing boat on a calm sea", "a plate of pasta with basil")
DEFAULTS = dict(variant="band_engine", teacher="", repo="runwayml/stable-diffusion-v1-5", revision="451f4fe1", tiny=0,
                shape="base", band="clean", rho=0.0, steps=1000, batch=8, lr=5e-5, lam_out=1.0, lam_data=0.0,
                drop=0.1, text=1, seed=0, data="", out="", device="cpu", dtype="fp32", autocast="auto",
                save_every=500, log_every=50, average_decay=0.9999, probe_n=4, probe_per_band=4, probe_seed=1234,
                probe_data="", probe_first=0, probe_rows=0, chunk=16, init="inherit", read_every=0, keep_checkpoint=1,
                sample_steps=50, guidance=7.5, group=16, parts="", other="", threads=4, det="")
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def main(argv=(), **overrides) -> dict:
    a = parse(DEFAULTS, argv, **overrides)
    if not a["out"]:
        raise ValueError("out= is required")
    if a["variant"] not in VARIANTS:
        raise ValueError(f"unknown variant {a['variant']!r}; known: {list(VARIANTS)}")
    if a["init"] not in ("inherit", "random"):
        raise ValueError(f"init is inherit or random, not {a['init']!r}")
    if int(a["read_every"]) < 0 or int(a["group"]) < 0 or int(a["chunk"]) < 0:
        raise ValueError("read_every, group and chunk are at least 0")
    if a["variant"] in ("assembly", "exchange") and not a["parts"]:
        raise ValueError(f"the {a['variant']} variant needs parts=")
    if a["variant"] == "exchange" and not a["other"]:
        raise ValueError("the exchange variant needs other= (a band engine's job folder or teacher)")
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
    common = dict(seed=int(a["probe_seed"]), per_band=int(a["probe_per_band"]), empty=teacher.empty_states(1),
                  alphas_cumprod=teacher.alphas_cumprod, batch=int(a["chunk"]))
    if a["probe_data"]:
        from ..capture.data import rows_in_order
        if int(a["probe_rows"]) < 1:
            raise ValueError("probe_data= needs probe_rows= (the held-out rows the probe is made of)")
        x0, ids = rows_in_order(a["probe_data"], int(a["probe_first"]), int(a["probe_rows"]))
        want = _latent_shape(teacher)
        if tuple(x0.shape[1:]) != want:
            raise ValueError(f"the probe rows' latents are {tuple(x0.shape[1:])}; the teacher's are {want}")
        states = teacher.text_states(ids)
        return make_probe(states, x0=x0, **common)
    n = int(a["probe_n"])
    prompts = [PROBE_PROMPTS[i % len(PROBE_PROMPTS)] for i in range(n)]
    states = teacher.text_states(teacher.tokenize(prompts))
    return make_probe(states, latent_shape=_latent_shape(teacher), **common)


def _probe_record(a: dict, probe) -> dict:
    rec = {"rows": int(probe.x0.shape[0]), "seed": int(probe.seed), "chunk": int(probe.batch)}
    if a["probe_data"]:
        rec.update({"data": a["probe_data"], "first": int(a["probe_first"]),
                    "last": int(a["probe_first"]) + int(a["probe_rows"]) - 1, "order": "file"})
    else:
        rec.update({"prompts": int(a["probe_n"]), "x0": "standard normal from the seed"})
    return rec


def _part_spec(text: str) -> Dict[str, str]:
    from ..capture import BANDS
    spec = dict(item.split("=", 1) for item in text.split(",") if item.strip())
    spec = {k.strip(): v.strip() for k, v in spec.items()}
    if set(spec) != set(BANDS):
        raise ValueError(f"parts= names each band {list(BANDS)} once: {text!r}")
    return spec


def _load_part(teacher, where: str, loaded: Dict[str, object]):
    """The teacher, or the averaged engine of a job folder (each folder loaded once: the same folder, the same part)."""
    from ..capture.engine import load_engine
    if where == "teacher":
        return teacher
    if where not in loaded:
        loaded[where] = load_engine(teacher, where, average=True).eval()
    return loaded[where]


def _parts(a: dict, teacher, loaded: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    from ..capture import BANDS
    spec = _part_spec(a["parts"])
    loaded = {} if loaded is None else loaded
    return {b: _load_part(teacher, spec[b], loaded) for b in BANDS}


def _run(a: dict, det) -> dict:
    from ..capture import BANDS
    from ..capture import reads
    from ..capture.assemble import Assembly
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
              "version": __version__, "probe_steps": {b: [int(t) for t in probe.steps[b]] for b in BANDS},
              "probe": _probe_record(a, probe)}
    curve_text = ""
    if a["variant"] == "born":
        from ..capture.engine import build_engine
        full = build_engine(teacher, "full", text=True).eval()     # the shape full with the prompt: must read exactly 0
        for b in BANDS:
            copy_read = reads.band_read(teacher, teacher, b, probe, refs[b])
            full_read = reads.band_read(full, teacher, b, probe, refs[b])
            born = reads.born_read(teacher, a["shape"], b, probe, text=bool(a["text"]), reference=refs[b])
            rows.append({"band": b, "part": "teacher copy", "mse": copy_read["mse"]})
            rows.append({"band": b, "part": "full copy with the prompt", "mse": full_read["mse"]})
            rows.append({"band": b, "part": f"{born['shape']} as born", "mse": born["mse"], "share": born["share"]})
            result.setdefault("reads", {})[b] = {"teacher_copy": copy_read, "full_copy": full_read, "born": born}
        del full
    elif a["variant"] == "band_engine":
        from ..capture.schedule import band_index
        band = BANDS[band_index(a["band"])]
        init = a["init"]
        if init == "random":
            from ..capture.engine import fresh_engine, param_count
            start = fresh_engine(teacher, a["shape"], seed=int(a["seed"]), text=bool(a["text"])).eval()
            born = reads.band_read(start, teacher, band, probe, refs[band])
            p = param_count(start)
            born.update({"shape": start.shape.name, "text": bool(a["text"]), "params": p,
                         "share": p / param_count(teacher), "init": "random"})
            del start
            label = f"{born['shape']} as born (random numbers)"
        else:
            born = reads.born_read(teacher, a["shape"], band, probe, text=bool(a["text"]), reference=refs[band])
            born["init"] = "inherit"
            label = f"{born['shape']} as born"
        rows.append({"band": band, "part": label, "mse": born["mse"], "share": born["share"]})
        every = int(a["read_every"])

        def on_every(n, engine, averaged):
            live = reads.band_read(engine, teacher, band, probe, refs[band])["mse"]
            mean = reads.band_read(averaged, teacher, band, probe, refs[band])["mse"]
            log(f"read at step {n:,}: live {live:.6g}, averaged {mean:.6g} (born {born['mse']:.6g})")
            return {"live": live, "averaged": mean}

        tr = train_band_engine(teacher, a["shape"], band, _source(a, teacher), steps=int(a["steps"]),
                               batch=int(a["batch"]), seed=int(a["seed"]), out=out, rho=float(a["rho"]),
                               lr=float(a["lr"]), lam_out=float(a["lam_out"]), lam_data=float(a["lam_data"]),
                               drop=float(a["drop"]), text=bool(a["text"]), average_decay=float(a["average_decay"]),
                               save_every=int(a["save_every"]), log_every=int(a["log_every"]),
                               autocast=a["autocast"], log=log, init=init, every=every,
                               on_every=on_every if every else None, keep_checkpoint=bool(int(a["keep_checkpoint"])))
        avg = tr["average"].eval()
        result["train"] = {"step": tr["step"], "finished": tr["finished"], "first": tr["log"][:1],
                           "last": tr["log"][-1:], "init": init, "seconds_a_step": tr.get("seconds_a_step"),
                           "trained": tr.get("trained"), "restored": bool(tr.get("restored", False)),
                           "checkpoint_kept": os.path.isfile(os.path.join(out, "checkpoint.pt"))}
        if every:
            curve = [{"step": 0, "live": born["mse"], "averaged": born["mse"]}]
            curve += [{"step": int(r["step"]), "live": r["value"]["live"], "averaged": r["value"]["averaged"]}
                      for r in tr["reads"]]
            result["curve"] = curve
            curve_text = (f"\n## the read on the {band} band during training\n\n"
                          + reads.table(curve, ["step", "live", "averaged"]))
        result["reads"] = {"born": born}
        for b in BANDS:
            r = reads.band_read(avg, teacher, b, probe, refs[b])
            result["reads"]["trained_" + b] = r
            rows.append({"band": b, "part": "trained (averaged)" + ("" if b == band else ", off its band"),
                         "mse": r["mse"]})
        result["reads"]["teacher_copy"] = reads.band_read(teacher, teacher, band, probe, refs[band])
        rows.append({"band": band, "part": "teacher copy", "mse": result["reads"]["teacher_copy"]["mse"]})
    elif a["variant"] == "assembly":
        model = Assembly(_parts(a, teacher))
        result["parts"] = a["parts"]
        for b in BANDS:
            r = reads.band_read(model, teacher, b, probe, refs[b])
            result.setdefault("reads", {})[b] = r
            rows.append({"band": b, "part": "assembly", "mse": r["mse"]})
        if int(a["sample_steps"]) > 0:
            zt = reads.sample_probe(teacher, probe, int(a["sample_steps"]), float(a["guidance"]), int(a["group"]))
            za = reads.sample_probe(model, probe, int(a["sample_steps"]), float(a["guidance"]), int(a["group"]))
            result["final_latent_change"] = float((za.double() - zt.double()).pow(2).mean())
            result["sample_group"] = int(a["group"])
            rows.append({"band": "all", "part": "final latents against the teacher's",
                         "mse": result["final_latent_change"]})
    else:
        loaded: Dict[str, object] = {}
        parts = _parts(a, teacher, loaded)
        other = _load_part(teacher, a["other"], loaded)
        from ..capture.schedule import band_index
        band = BANDS[band_index(a["band"])]
        r = reads.exchange(Assembly(parts), band, other, probe, teacher=teacher, sample_steps=int(a["sample_steps"]),
                           guidance=float(a["guidance"]), group=int(a["group"]))
        result.update({"parts": a["parts"], "band": band, "other": a["other"], "exchange": r})
        for b in BANDS:
            x = r["bands"][b]
            rows.append({"band": b, "part": f"{band} part exchanged", "before": x["before"], "after": x["after"],
                         "change": x["change"], "largest change where silent": x["largest_change_where_silent"]})
        if "final_latent_change" in r:
            result["final_latent_change"] = r["final_latent_change"]
            rows.append({"band": "all", "part": "final latents, after against before",
                         "mse": r["final_latent_change"]})
    result["rows"] = rows
    result["seconds"] = time.time() - t0
    det.record(result)
    reads.write_json(result, os.path.join(out, "result.json"))
    columns = (["band", "part", "before", "after", "change", "largest change where silent", "mse"]
               if a["variant"] == "exchange" else ["band", "part", "mse", "share"])
    text = f"# capture: {a['variant']}\n\n" + reads.table(rows, columns) + curve_text
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
    return table(rows, ["job", "variant", "band", "part", "mse", "share", "change"])


if __name__ == "__main__" and not in_notebook():
    main(sys.argv[1:])
