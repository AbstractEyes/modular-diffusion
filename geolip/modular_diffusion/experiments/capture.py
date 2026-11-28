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
| pictures     | the pictures of an assembly (``source=parts``: sampled from ``noise_seed`` ``group`` rows at a time,  |
|              | then decoded) or the rows' own pictures (``source=rows``: their clean latents decoded) for the rows  |
|              | of ``pic_data``; the judge's features of every picture, the text-to-picture score (``clip``), a      |
|              | contact sheet; no probe is built                                                                     |
| picture_compare | two ``pictures`` job folders (``a``, ``b``) compared: the Frechet distance of their features, the |
|              | kernel distance (``capture.pictures.kid``), and the text-to-picture means and their row-by-row       |
|              | difference; no teacher                                                                               |
| chain_read   | an assembly (``parts=``) against the teacher on the states of the sampler's chain: on its own chain |
|              | (the assembly drives, the teacher is asked) and on the teacher's (the teacher drives, the assembly  |
|              | is asked), per band the mean error on each chain and their ratio (``capture.reads.chain_read``)      |
| steering     | one part (``part=``) against the teacher on one band's probe steps (``band``), each probe row       |
|              | answered with its prompt and with the unguided side's states: the own error R, the steering error S |
|              | of the prompt difference, the guided error G at ``guidance``, the prompt difference's size P, the   |
|              | error on the unguided side U, per band and per step (``capture.reads.steering_read``); the part's   |
|              | weights are recorded by their hashes                                                                 |

The teacher is a local diffusers folder: ``teacher=<folder>``, or found in the local hub cache by ``repo`` and
``revision`` (nothing is ever downloaded). ``tiny=1`` uses a tiny random teacher and random batches instead (16 x 16
latents; for smoke runs and the tests).

THE PROBE: ``probe_n`` fixed prompts with standard normal latents; or, with ``probe_data``, the rows ``probe_first`` ..
``probe_first + probe_rows - 1`` of that folder of latent shards in FILE order (``capture.data.rows_in_order``): their
clean latents as x0 and their token ids through the teacher's text encoder. Either way the noise comes from
``probe_seed``, each band is read on ``probe_per_band`` even steps, and the reads put ``chunk`` rows through a forward.
Sampling reads (assembly, exchange, chain_read) sample the probe's rows ``group`` at a time (``capture.reads.sample_probe``: group j
starts from the noise of probe_seed + 100003 j, the same for every model compared).

READS DURING TRAINING (band_engine, ``read_every`` > 0): every ``read_every`` updates and after the last one, the live
engine and its averaged copy are read on the band's own probe steps against the teacher's answers (computed once);
``result["curve"]`` = [{"step", "live", "averaged"}, ...], its first row the engine as born (step 0).

Arguments (key=value): variant, teacher, repo, revision, tiny, shape (a name full / base / small / tiny or
<layers>/<mid|nomid>/<stages>), band (clean / middle / noisy; for exchange the band whose part is replaced), rho (the
whole-range share of the band's training law), steps, batch, lr, lam_out, lam_data, drop (the empty-prompt share), text
(0: an engine that ignores the prompt), seed, init (inherit / random), read_every, keep_checkpoint (0: checkpoint.pt is
removed once the job has finished), micro (band_engine: above 0 and under batch, a step's rows go through the
forwards and the backward in groups of that many rows, gradients added, one optimizer step: the same update in less
memory; not a setting a checkpoint is checked against; 0: the whole batch at once), data (a folder of latent shards, read by ``capture.data.LatentBatches``), out,
device, dtype (fp32 / bf16 / fp16: the teacher's numbers; an engine's are float32), autocast (auto / none / bf16 /
fp16), save_every, log_every, average_decay, probe_n, probe_per_band, probe_seed, probe_data, probe_first, probe_rows,
chunk, sample_steps, guidance, group, parts (for assembly, exchange, chain_read and pictures: one part a band, comma separated, for
example clean=runs/a,middle=teacher,noisy=born:tiny; a part is ``teacher``, a band engine's job folder (its averaged
engine), ``born:<shape>`` (the shape's engine with every number inherited from the teacher, untrained, reading the
prompt) or ``born:<shape>:notext`` (the same without the prompt); ``<shape>`` is a name or the text form such as
2222/nomid/3), other (for exchange: a part in the same forms), part (for steering: ``teacher``, a band engine's job
folder, ``born:<shape>[:notext]`` or ``random:<shape>[:notext]``, the shape's engine with random numbers from the job's
seed by ``engine.fresh_engine``), threads, det (empty, warn or on).

THE STEERING READ (``variant=steering``): the probe is the one every variant builds from the probe arguments, so with
a training's probe arguments it is that training's own-error read (``read_every``): the same rows, noise and steps, and
the read's ``mse`` equals that read of the same part to the bit. The teacher answers each probe step twice (the rows'
prompts and the unguided side) on the read's band only, and the part the same. result.json holds ``steering`` (the
read), ``weights`` (what the part was: its kind teacher / job / born / random, and for a job folder its job name,
shape, whether it reads the prompt and the SHA-256 of engine.safetensors and engine_avg.safetensors; for an engine the
SHA-256 of its numbers, ``state_sha256``) and the seconds of the teacher's answers, of making the part and of its
answers.

The source (every variant but picture_compare): unet (a folder of the source's own U-Net, a diffusers config.json and
diffusion_pytorch_model.safetensors; the text encoder, tokenizer and image decoder stay the teacher folder's), kind
(eps: the noise on whole steps; flow: the velocity on a continuous time, sampled by Euler steps on its shifted grid),
shift (a flow source's), unguided (empty or zeros; by default the kind's own), kappa (the source's latent-scale ratio:
held-out clean latents given to it are multiplied by kappa, its pictures' latents divided by it before decoding),
unet_seed (tiny=1: the seed of the tiny U-Net alone). A flow source does not train a band engine (band_engine refuses
it). Every read is then made under the source's own process and unguided side.

For pictures: pic_data (a folder of latent shards; with tiny=1 it may be empty, and random rows are used), pic_first,
pic_n (the rows pic_first .. pic_first + pic_n - 1 in FILE order: their token ids are the prompts, their clean latents
the pictures the prompts describe), noise_seed (the sampler's start noise, group j from noise_seed + 100003 j), source
(parts or rows), part_dtype (empty: the engines as stored, float32; fp16 or bf16: every loaded or born engine cast to
it; the teacher keeps ``dtype``), judge (inception or fake: the feature judge of ``score.make_judge``), clip (empty: no
text-to-picture score; fake; or a local CLIP folder or cached hub id), sheet_n (the pictures on the contact sheet; 0:
none), decode_batch. The image decoder is always kept in float32. For picture_compare: a, b (two pictures job
folders), paired (1: the two hold the same rows, or one's rows are a prefix of the other's
and the common rows are compared, row by row; their group, noise_seed, sample_steps and guidance must then agree
unless one job is source=rows; 0: two sets, compared whole), first_n (above 0: only the first first_n rows of both
jobs are compared, inside the common prefix when paired; each job must hold that many), blocks (the kernel distance's
blocks). The result also says whether the compared feature arrays are equal element by element (features_equal) and
their largest absolute difference.

Output in ``out``: log.txt (its last line 'done'), result.json, result.md (the reads as a table); band_engine also
writes the job's checkpoint.pt and the engine files (``capture.train``), and resumes from a checkpoint found there (a
finished job whose checkpoint was removed is read back, not trained again). pictures also writes pictures.npz
(``features`` float32 [N, D], ``rows`` int64 [N], ``clip`` float32 [N] when asked) and sheet.jpg.
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
VARIANTS = ("band_engine", "born", "assembly", "exchange", "pictures", "picture_compare", "chain_read", "steering")
PICTURE_VARIANTS = ("pictures", "picture_compare")
SOURCES = ("parts", "rows")
PROBE_PROMPTS = ("a photograph of a red house by a lake", "a bowl of fruit on a wooden table",
                 "a portrait of an old man with a gray beard", "a city street at dusk with lit windows",
                 "a mountain road in the snow", "a cat sleeping on a blue sofa",
                 "a sailing boat on a calm sea", "a plate of pasta with basil")
DEFAULTS = dict(variant="band_engine", teacher="", repo="runwayml/stable-diffusion-v1-5", revision="451f4fe1", tiny=0,
                shape="base", band="clean", rho=0.0, steps=1000, batch=8, lr=5e-5, lam_out=1.0, lam_data=0.0,
                drop=0.1, text=1, seed=0, data="", out="", device="cpu", dtype="fp32", autocast="auto",
                save_every=500, log_every=50, average_decay=0.9999, probe_n=4, probe_per_band=4, probe_seed=1234,
                probe_data="", probe_first=0, probe_rows=0, chunk=16, init="inherit", read_every=0, keep_checkpoint=1,
                sample_steps=50, guidance=7.5, group=16, parts="", other="", threads=4, det="",
                pic_data="", pic_first=0, pic_n=0, noise_seed=2000, source="parts", part_dtype="", judge="inception",
                clip="", sheet_n=16, decode_batch=8, a="", b="", paired=1, blocks=10, first_n=0, micro=0, part="")
# The source's arguments (shared by the capture experiments): its own U-Net folder (empty: the teacher folder's), its
# kind (eps or flow), the shift of a flow source, its unguided side (empty: the kind's own, empty for eps, zeros for
# flow), its latent-scale ratio kappa, and for tiny=1 the seed of its U-Net (-1: the job's seed).
SOURCE_DEFAULTS = dict(unet="", kind="eps", shift="", unguided="", kappa=1.0, unet_seed=-1)
DEFAULTS.update(SOURCE_DEFAULTS)
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
PART_DTYPES = {"": None, "fp16": torch.float16, "bf16": torch.bfloat16}
BORN = "born:"
RANDOM = "random:"


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
    if int(a["micro"]) < 0:
        raise ValueError("micro is 0 (the whole batch in one forward) or a positive number of rows")
    if a["variant"] in ("assembly", "exchange", "chain_read") and not a["parts"]:
        raise ValueError(f"the {a['variant']} variant needs parts=")
    if a["variant"] == "chain_read" and int(a["sample_steps"]) < 1:
        raise ValueError("the chain_read variant samples: sample_steps is at least 1")
    if a["variant"] == "exchange" and not a["other"]:
        raise ValueError("the exchange variant needs other= (a band engine's job folder or teacher)")
    if a["part_dtype"] not in PART_DTYPES:
        raise ValueError(f"part_dtype is empty, fp16 or bf16, not {a['part_dtype']!r}")
    source_options(a)
    if a["variant"] == "band_engine" and a["kind"] != "eps":
        raise ValueError(f"a band engine is trained from a noise-predicting teacher only (kind=eps); a {a['kind']} "
                         "source's capture loss is not defined in this library")
    if a["variant"] == "pictures":
        if a["source"] not in SOURCES:
            raise ValueError(f"source is parts or rows, not {a['source']!r}")
        if int(a["pic_n"]) < 1:
            raise ValueError("the pictures variant needs pic_n >= 1 (the number of rows)")
        if a["source"] == "parts" and not a["parts"]:
            raise ValueError("the pictures variant with source=parts needs parts=")
        if not a["pic_data"] and not a["tiny"]:
            raise ValueError("the pictures variant needs pic_data= (a folder of latent shards) on the real teacher")
        if int(a["sheet_n"]) < 0 or int(a["decode_batch"]) < 1:
            raise ValueError("sheet_n is at least 0 and decode_batch at least 1")
    if a["variant"] == "picture_compare" and not (a["a"] and a["b"]):
        raise ValueError("the picture_compare variant needs a= and b= (two pictures job folders)")
    if a["variant"] == "steering":
        _check_steering_part(a["part"])
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


def source_options(a: dict) -> dict:
    """The source's keywords of ``load_teacher`` / ``tiny_teacher`` from the job's arguments (``SOURCE_DEFAULTS``)."""
    from ..capture.process import KINDS
    from ..capture.teacher import UNGUIDED
    if a["kind"] not in KINDS:
        raise ValueError(f"kind is one of {list(KINDS)}, not {a['kind']!r}")
    if a["unguided"] and a["unguided"] not in UNGUIDED:
        raise ValueError(f"unguided is empty or one of {list(UNGUIDED)}, not {a['unguided']!r}")
    if a["kind"] == "flow" and not str(a["shift"]).strip():
        raise ValueError("a flow source needs shift= (the shift it was trained at)")
    if a["kind"] == "eps" and str(a["shift"]).strip():
        raise ValueError("shift= belongs to a flow source")
    return dict(kind=a["kind"], shift=float(a["shift"]) if str(a["shift"]).strip() else None,
                unguided=a["unguided"] or None, kappa=float(a["kappa"]))


def load_source(a: dict, vae: bool = False, seed: int = 0, tiny_seed: Optional[int] = None,
                vae_dtype: Optional[torch.dtype] = None):
    """(the source, where it came from) from the job's arguments: the tiny random teacher (``tiny_seed``, the job's
    seed by default; its U-Net from ``unet_seed`` when that is 0 or more), or the teacher folder with ``unet`` as its
    own U-Net folder when given; with the source's kind, shift, unguided side and kappa."""
    from ..capture.teacher import find_snapshot, load_teacher, tiny_teacher, with_unet
    opts = source_options(a)
    if a["tiny"]:
        ts = int(seed) if tiny_seed is None else int(tiny_seed)
        us = int(a["unet_seed"]) if int(a["unet_seed"]) >= 0 else None
        source = tiny_teacher(ts, vae=vae, unet_seed=us, **opts).to(a["device"], DTYPES[a["dtype"]])
        where = "tiny random teacher" + ("" if us is None and opts["kind"] == "eps" else
                                         f" (U-Net seed {ts if us is None else us}, {opts['kind']})")
        if a["unet"]:                                     # a tiny U-Net written to a folder, beside the tiny encoder
            source = with_unet(source, a["unet"])
            where = f"tiny random teacher with the U-Net of {a['unet']} ({opts['kind']})"
    else:
        where = a["teacher"] or find_snapshot(a["repo"], a["revision"] or None)
        extra = {"vae_dtype": vae_dtype} if vae_dtype is not None else {}
        source = load_teacher(where, device=a["device"], dtype=DTYPES[a["dtype"]], vae=vae, unet=a["unet"] or None,
                              **extra, **opts)
        if a["unet"]:
            where = f"{where} with the U-Net of {a['unet']}"
    return source, where


def _teacher(a: dict, vae: bool = False):
    """The teacher and where it came from; with ``vae`` it carries its image decoder, kept in float32 whatever
    ``dtype`` is (cast after the teacher's own move, so nothing casts it back)."""
    teacher, where = load_source(a, vae=vae, seed=int(a["seed"]), vae_dtype=torch.float32 if vae else None)
    if vae:
        teacher.vae.float()
    return teacher, where


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
    common = dict(seed=int(a["probe_seed"]), per_band=int(a["probe_per_band"]), empty=teacher.unguided_states(1),
                  alphas_cumprod=teacher.alphas_cumprod, batch=int(a["chunk"]), process=teacher.process)
    if a["probe_data"]:
        from ..capture.data import rows_in_order
        if int(a["probe_rows"]) < 1:
            raise ValueError("probe_data= needs probe_rows= (the held-out rows the probe is made of)")
        x0, ids = rows_in_order(a["probe_data"], int(a["probe_first"]), int(a["probe_rows"]))
        want = _latent_shape(teacher)
        if tuple(x0.shape[1:]) != want:
            raise ValueError(f"the probe rows' latents are {tuple(x0.shape[1:])}; the teacher's are {want}")
        states = teacher.text_states(ids)
        return make_probe(states, x0=teacher.source_latents(x0), **common)
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


def born_spec(where: str):
    """(shape, text) of a part written ``born:<shape>`` (reads the prompt) or ``born:<shape>:notext``. ValueError
    naming the three forms of a part when it does not parse."""
    from ..capture.shapes import Shape
    forms = ("a part is teacher, a band engine's job folder, born:<shape> or born:<shape>:notext (a shape name or "
             f"<layers>/<mid|nomid>/<stages>), not {where!r}")
    body = where[len(BORN):]
    text = True
    if body.endswith(":notext"):
        body, text = body[:-len(":notext")], False
    if not body or ":" in body:
        raise ValueError(forms)
    try:
        shape = Shape.parse(body)
    except ValueError as e:
        raise ValueError(f"{forms} ({e})") from None
    return shape, text


def _load_part(teacher, where: str, loaded: Dict[str, object]):
    """The teacher; the averaged engine of a job folder; or a born engine (``born:<shape>`` or
    ``born:<shape>:notext``: every number inherited from the teacher, untrained). Each is made once: the same text,
    the same part."""
    from ..capture.engine import build_engine, load_engine
    if where == "teacher":
        return teacher
    if where not in loaded:
        if where.startswith(BORN):
            shape, text = born_spec(where)
            loaded[where] = build_engine(teacher, shape, text=text).eval()
        else:
            loaded[where] = load_engine(teacher, where, average=True).eval()
    return loaded[where]


def _parts(a: dict, teacher, loaded: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    from ..capture import BANDS
    spec = _part_spec(a["parts"])
    loaded = {} if loaded is None else loaded
    return {b: _load_part(teacher, spec[b], loaded) for b in BANDS}


def random_spec(where: str):
    """(shape, text) of a part written ``random:<shape>`` (reads the prompt) or ``random:<shape>:notext``. ValueError
    naming the two forms when it does not parse."""
    from ..capture.shapes import Shape
    forms = ("a random part is random:<shape> or random:<shape>:notext (a shape name or <layers>/<mid|nomid>/<stages>), "
             f"not {where!r}")
    body = where[len(RANDOM):]
    text = True
    if body.endswith(":notext"):
        body, text = body[:-len(":notext")], False
    if not body or ":" in body:
        raise ValueError(forms)
    try:
        shape = Shape.parse(body)
    except ValueError as e:
        raise ValueError(f"{forms} ({e})") from None
    return shape, text


def _check_steering_part(where: str) -> None:
    """ValueError, before anything is loaded, when the steering variant's part is not one of its forms."""
    forms = "teacher, a band engine's job folder (holding engine.json), born:<shape> or random:<shape>"
    if not where:
        raise ValueError(f"the steering variant needs part= ({forms})")
    if where.startswith(RANDOM):
        random_spec(where)
    elif where.startswith(BORN):
        born_spec(where)
    elif where != "teacher" and not os.path.isfile(os.path.join(where, "engine.json")):
        raise ValueError(f"part={where!r} is not {forms}")


def file_sha256(path: str) -> Optional[str]:
    """The SHA-256 of a file's bytes (None when there is no such file)."""
    import hashlib
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def state_sha256(module) -> str:
    """The SHA-256 of a module's numbers: each entry of its state dict in name order, the name and then the tensor's
    float32 bytes on the CPU (two engines with the same numbers give the same hash)."""
    import hashlib
    h = hashlib.sha256()
    for name, v in sorted(module.state_dict().items()):
        h.update(name.encode("utf-8") + b"\0")
        h.update(v.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


def _steering_part(a: dict, teacher):
    """(the part of the steering read, its record): the teacher; an engine with random numbers (``random:<shape>``,
    ``engine.fresh_engine`` from the job's seed); a born engine (``born:<shape>``); or the averaged engine of a band
    engine's job folder, recorded with the SHA-256 of its two weight files."""
    from ..capture.engine import build_engine, fresh_engine, load_engine, param_count
    where = a["part"]
    if where == "teacher":
        return teacher, {"kind": "teacher", "part": where}
    if where.startswith(RANDOM):
        shape, text = random_spec(where)
        part = fresh_engine(teacher, shape, seed=int(a["seed"]), text=text).eval()
        rec = {"kind": "random", "part": where, "shape": part.shape.name, "text": text, "seed": int(a["seed"])}
    elif where.startswith(BORN):
        shape, text = born_spec(where)
        part = build_engine(teacher, shape, text=text).eval()
        rec = {"kind": "born", "part": where, "shape": part.shape.name, "text": text}
    else:
        part = load_engine(teacher, where, average=True).eval()
        rec = {"kind": "job", "part": where, "job": os.path.basename(where.rstrip("/\\")), "shape": part.shape.name,
               "text": bool(part.text), "engine_sha256": file_sha256(os.path.join(where, "engine.safetensors")),
               "engine_avg_sha256": file_sha256(os.path.join(where, "engine_avg.safetensors"))}
    rec.update({"params": param_count(part), "state_sha256": state_sha256(part)})
    return part, rec


STEERING_COLUMNS = ["R", "S", "G", "P", "U", "mse"]


def _run_steering(a: dict, det, log, t0: float) -> dict:
    """The steering variant: the probe, the teacher's answers on the band (with the prompts and the unguided side), the
    part, its read (``capture.reads.steering_read``)."""
    from ..capture import BANDS, reads
    from ..capture.schedule import band_index
    teacher, where = _teacher(a)
    log(f"CAPTURE steering: teacher {where}; torch {torch.__version__}; package {__version__}")
    probe = _probe(a, teacher)
    band = BANDS[band_index(a["band"])]
    t1 = time.time()
    ref = reads.steering_answers(teacher, probe, band)
    t2 = time.time()
    part, record = _steering_part(a, teacher)
    t3 = time.time()
    r = reads.steering_read(part, teacher, band, probe, guidance=float(a["guidance"]), reference=ref)
    t4 = time.time()
    result = _head(a, where)
    result.update({"probe_steps": {b: [int(t) for t in probe.steps[b]] for b in BANDS}, "probe": _probe_record(a, probe),
                   "process": teacher.record(), "band": band, "part": a["part"], "weights": record, "steering": r,
                   "seconds_teacher": t2 - t1, "seconds_part_load": t3 - t2, "seconds_part": t4 - t3,
                   "seconds": time.time() - t0})
    rows = [dict({"band": band, "part": a["part"]}, **{k: r[k] for k in STEERING_COLUMNS})]
    result["rows"] = rows                                # the read's row in the folder's summary (``summarize``)
    per = [dict({"step": int(t)}, **{k: v[k] for k in STEERING_COLUMNS}) for t, v in r["per_step"].items()]
    more = (f"\n## by probe step (G at guidance {r['guidance']:g}; {r['rows']} rows a step)\n\n"
            + reads.table(per, ["step"] + STEERING_COLUMNS))
    return _write_outputs(a, result, rows, ["band", "part"] + STEERING_COLUMNS, det, log, more=more)


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
    if a["variant"] == "picture_compare":                # no teacher, no probe
        return _run_compare(a, det, log, t0)
    if a["variant"] == "pictures":                       # the teacher with its image decoder, no probe
        return _run_pictures(a, det, log, t0)
    if a["variant"] == "steering":                       # the teacher's answers on the read's band only
        return _run_steering(a, det, log, t0)
    teacher, where = _teacher(a)
    log(f"CAPTURE {a['variant']}: teacher {where}; torch {torch.__version__}; package {__version__}")
    probe = _probe(a, teacher)
    refs = None if a["variant"] == "chain_read" else {b: reads.teacher_answers(teacher, probe, b) for b in BANDS}
    rows = []
    result = {"experiment": EXPERIMENT, "variant": a["variant"], "teacher": where,
              "args": {k: a[k] for k in sorted(a) if not (k == "det" and not a[k])}, "torch": torch.__version__,
              "version": __version__, "probe_steps": {b: [int(t) for t in probe.steps[b]] for b in BANDS},
              "probe": _probe_record(a, probe), "process": teacher.record()}
    curve_text = ""
    if a["variant"] == "born":
        from ..capture.engine import build_engine, param_count
        full = build_engine(teacher, "full", text=True).eval()     # the shape full with the prompt: must read exactly 0
        born_eng = build_engine(teacher, a["shape"], text=bool(a["text"])).eval()
        born_params = param_count(born_eng)
        born_share = born_params / param_count(teacher)
        for b in BANDS:
            copy_read = reads.band_read(teacher, teacher, b, probe, refs[b])
            full_read = reads.band_read(full, teacher, b, probe, refs[b])
            born = reads.band_read(born_eng, teacher, b, probe, refs[b])
            born.update({"shape": born_eng.shape.name, "text": bool(a["text"]),
                         "params": born_params, "share": born_share})
            rows.append({"band": b, "part": "teacher copy", "mse": copy_read["mse"]})
            rows.append({"band": b, "part": "full copy with the prompt", "mse": full_read["mse"]})
            rows.append({"band": b, "part": f"{born['shape']} as born", "mse": born["mse"], "share": born["share"]})
            result.setdefault("reads", {})[b] = {"teacher_copy": copy_read, "full_copy": full_read, "born": born}
        del full, born_eng
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
                               on_every=on_every if every else None, keep_checkpoint=bool(int(a["keep_checkpoint"])),
                               micro=int(a["micro"]))
        avg = tr["average"].eval()
        result["train"] = {"step": tr["step"], "finished": tr["finished"], "first": tr["log"][:1],
                           "last": tr["log"][-1:], "init": init, "micro": int(tr.get("micro", 0)),
                           "seconds_a_step": tr.get("seconds_a_step"),
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
    elif a["variant"] == "chain_read":
        model = Assembly(_parts(a, teacher))
        r = reads.chain_read(model, teacher, probe, int(a["sample_steps"]), float(a["guidance"]), int(a["group"]))
        result.update({"parts": a["parts"], "chain": r, "final_latent_change": r["final_latent_change"],
                       "sample_group": int(a["group"])})
        for b in BANDS:
            x = r["bands"][b]
            rows.append({"band": b, "own chain": x["own_mean"], "teacher's chain": x["teacher_mean"],
                         "ratio": x["ratio"], "log ratio SE": x["log_ratio_se"], "rows in the ratio": x["rows_in_ratio"],
                         "rows left out": x["rows_left_out"]})
        rows.append({"band": "all", "part": "final latents, own chain against the teacher's",
                     "mse": r["final_latent_change"]})
    else:
        loaded: Dict[str, object] = {}
        parts = _parts(a, teacher, loaded)
        other = _load_part(teacher, a["other"], loaded)
        from ..capture.schedule import band_index
        band = BANDS[band_index(a["band"])]
        r = reads.exchange(Assembly(parts), band, other, probe, teacher=teacher, sample_steps=int(a["sample_steps"]),
                           guidance=float(a["guidance"]), group=int(a["group"]), references=refs)
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
    if a["variant"] == "exchange":
        columns = ["band", "part", "before", "after", "change", "largest change where silent", "mse"]
    elif a["variant"] == "chain_read":
        columns = ["band", "part", "own chain", "teacher's chain", "ratio", "log ratio SE", "rows in the ratio",
                   "rows left out", "mse"]
    else:
        columns = ["band", "part", "mse", "share"]
    text = f"# capture: {a['variant']}\n\n" + reads.table(rows, columns) + curve_text
    with open(os.path.join(out, "result.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    log(text.rstrip())
    log("done")
    return result


PICTURES = "pictures.npz"


def cast_engines(loaded: Dict[str, object], part_dtype: str) -> None:
    """Every loaded or born engine of ``loaded`` cast in place to ``part_dtype`` (empty: left as stored). The teacher
    is never in ``loaded``, so it keeps its own precision."""
    dtype = PART_DTYPES[part_dtype]
    if dtype is None:
        return
    for engine in loaded.values():
        engine.to(dtype=dtype)


def _picture_rows(a: dict, teacher):
    """(clean latents float32 [N, C, h, w], ids long [N, 77], file rows int64 [N]) of the pictures variant."""
    import numpy as np
    from ..capture.data import rows_in_order
    first, n = int(a["pic_first"]), int(a["pic_n"])
    want = _latent_shape(teacher)
    if a["pic_data"]:
        x0, ids = rows_in_order(a["pic_data"], first, n)
        if tuple(x0.shape[1:]) != want:
            raise ValueError(f"the rows' latents are {tuple(x0.shape[1:])}; the teacher's are {want}")
        return x0, ids, np.arange(first, first + n, dtype=np.int64)
    # random rows (tiny=1 without pic_data): row i is batch i of one row of the random source, so a row's prompt and
    # latents depend only on its number and a job of fewer rows holds a prefix of a job of more
    source = random_source(int(a["seed"]), want)
    drawn = [source(i, 1) for i in range(first, first + n)]
    x0 = torch.cat([d[0] for d in drawn])
    ids = torch.cat([d[1] for d in drawn])
    return x0, ids, np.arange(first, first + n, dtype=np.int64)


def _write_outputs(a: dict, result: dict, rows, columns, det, log, more: str = "") -> dict:
    from ..capture import reads
    out = a["out"]
    det.record(result)
    reads.write_json(result, os.path.join(out, "result.json"))
    text = f"# capture: {a['variant']}\n\n" + reads.table(rows, columns) + more
    with open(os.path.join(out, "result.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    log(text.rstrip())
    log("done")
    return result


def _head(a: dict, teacher: str) -> dict:
    return {"experiment": EXPERIMENT, "variant": a["variant"], "teacher": teacher,
            "args": {k: a[k] for k in sorted(a) if not (k == "det" and not a[k])}, "torch": torch.__version__,
            "version": __version__}


def _run_pictures(a: dict, det, log, t0: float) -> dict:
    import numpy as np
    from .. import score
    from ..capture import pictures
    from ..capture.assemble import Assembly
    out = a["out"]
    teacher, where = _teacher(a, vae=True)
    log(f"CAPTURE pictures: teacher {where}; torch {torch.__version__}; package {__version__}")
    x0, ids, file_rows = _picture_rows(a, teacher)
    n = int(x0.shape[0])
    result = _head(a, where)
    result.update({"n": n, "source": a["source"], "parts": a["parts"] if a["source"] == "parts" else "",
                   "part_dtype": a["part_dtype"], "process": teacher.record()})
    t1 = time.time()
    if a["source"] == "parts":
        loaded: Dict[str, object] = {}
        parts = _parts(a, teacher, loaded)
        cast_engines(loaded, a["part_dtype"])
        states = teacher.text_states(ids)
        z = pictures.sample_latents(Assembly(parts), states, teacher.unguided_states(1), int(a["noise_seed"]),
                                    group=int(a["group"]), steps=int(a["sample_steps"]),
                                    guidance=float(a["guidance"]), latent_shape=_latent_shape(teacher),
                                    alphas_cumprod=teacher.alphas_cumprod, log=log, process=teacher.process)
        z = teacher.picture_latents(z)          # the source's samples on the standard scale (unchanged at kappa 1)
        del parts, loaded
    else:
        z = x0                                  # the rows' own latents are on the standard scale
    t2 = time.time()
    pics = pictures.decode(teacher.vae, z, batch=int(a["decode_batch"]))
    t3 = time.time()
    log(f"decoded {n:,} pictures {tuple(pics.shape[1:])} in {t3 - t2:.1f} s")
    del teacher, z, x0
    if str(a["device"]).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    judge = score.make_judge(a["judge"], a["device"])
    feats = np.asarray(judge.features(pics), dtype=np.float32)
    saved = {"features": feats, "rows": file_rows}
    result.update({"judge": judge.name, "feature_dim": int(feats.shape[1])})
    clip = pictures.make_clip_judge(a["clip"], a["device"])
    if clip is not None:
        s = clip.scores(pics, ids)
        saved["clip"] = s.astype(np.float32)
        result.update({"clip_judge": clip.name, "clip_mean": float(s.mean()),
                       "clip_std": float(s.std(ddof=1)) if s.shape[0] > 1 else None})
    t4 = time.time()
    np.savez(os.path.join(out, PICTURES), **saved)
    if int(a["sheet_n"]) > 0:
        pictures.contact_sheet(pics[: int(a["sheet_n"])]).save(os.path.join(out, "sheet.jpg"), quality=90)
    result.update({"seconds_sampling": t2 - t1, "seconds_decoding": t3 - t2, "seconds_judging": t4 - t3,
                   "seconds": time.time() - t0})
    keys = ("n", "source", "parts", "part_dtype", "judge", "feature_dim", "clip_judge", "clip_mean", "clip_std",
            "seconds_sampling", "seconds_decoding", "seconds_judging", "seconds")
    rows = [{"quantity": k, "value": result[k]} for k in keys if k in result]
    return _write_outputs(a, result, rows, ["quantity", "value"], det, log)


SAMPLING_ARGS = ("group", "noise_seed", "sample_steps", "guidance")


def _read_pictures(folder: str) -> dict:
    import json
    import numpy as np
    with np.load(os.path.join(folder, PICTURES), allow_pickle=False) as d:
        out = {k: d[k] for k in d.files}
    with open(os.path.join(folder, "result.json"), encoding="utf-8") as fh:
        out["result"] = json.load(fh)
    return out


def _paired_rows(pa: dict, pb: dict) -> int:
    """The number of rows two pictures jobs are compared on, row by row: their equal ``rows``, or the shorter one's
    when its rows are a prefix of the other's. Sound because ``sample_latents`` starts group j from a seed that
    depends only on j (and the noise seed): row i gets the same start noise whatever the job's row count, so long as
    the group size, the noise seed, the steps and the guidance are the same, which is checked here (a ``source=rows``
    job samples nothing and has none of them). ValueError on the first difference."""
    import numpy as np
    ra, rb = pa["rows"], pb["rows"]
    n = min(ra.shape[0], rb.shape[0])
    diff = np.nonzero(ra[:n] != rb[:n])[0]
    if diff.size:
        k = int(diff[0])
        raise ValueError(f"paired: the rows differ first at position {k} (a row {int(ra[k])}, b row {int(rb[k])})")
    sa, sb = pa["result"], pb["result"]
    if sa.get("source", "parts") != "rows" and sb.get("source", "parts") != "rows":
        for k in SAMPLING_ARGS:
            va, vb = sa["args"].get(k), sb["args"].get(k)
            if va != vb:
                raise ValueError(f"paired: the two jobs were sampled with other {k} ({va!r} and {vb!r}); their rows "
                                 f"do not start from the same noise (paired=0 compares them as two sets)")
    return int(n)


def _run_compare(a: dict, det, log, t0: float) -> dict:
    import numpy as np
    from .. import score
    from ..capture import pictures
    log(f"CAPTURE picture_compare: {a['a']} against {a['b']}; package {__version__}")
    pa, pb = _read_pictures(a["a"]), _read_pictures(a["b"])
    paired = bool(int(a["paired"]))
    n_a, n_b = int(pa["features"].shape[0]), int(pb["features"].shape[0])
    first_n = int(a["first_n"])
    if first_n < 0:
        raise ValueError("first_n is 0 (every row) or a positive number of rows")
    if first_n and min(n_a, n_b) < first_n:
        raise ValueError(f"first_n={first_n}: a holds {n_a} rows and b {n_b}; both must hold at least {first_n}")
    prefix = False
    cut = None
    if paired:
        cut = _paired_rows(pa, pb)                         # the common prefix first
        prefix = n_a != n_b
    if first_n:
        cut = first_n                                      # then the first first_n rows inside it
    if cut is not None:
        for p in (pa, pb):
            for k in ("features", "clip"):
                if k in p:
                    p[k] = p[k][:cut]
    fa, fb = pa["features"].astype(np.float64), pb["features"].astype(np.float64)
    ma, ca, _ = score.feature_stats(fa)
    mb, cb, _ = score.feature_stats(fb)
    same_shape = fa.shape == fb.shape
    result = _head(a, "")
    del result["teacher"]
    result.update({"a": a["a"], "b": a["b"], "paired": paired, "n_a": n_a, "n_b": n_b, "prefix": prefix,
                   "first_n": first_n, "n": int(fa.shape[0]) if (paired or first_n) else None,
                   "rows_compared_a": int(fa.shape[0]), "rows_compared_b": int(fb.shape[0]),
                   "features_equal": bool(same_shape and np.array_equal(pa["features"], pb["features"])),
                   "max_abs_feature_difference": float(np.abs(fa - fb).max()) if same_shape and fa.size else None,
                   "frechet": score.frechet_distance(ma, ca, mb, cb), "kid": pictures.kid(fa, fb, int(a["blocks"]))})
    if "clip" in pa and "clip" in pb:
        sa, sb = pa["clip"].astype(np.float64), pb["clip"].astype(np.float64)
        result.update({"clip_a": float(sa.mean()), "clip_b": float(sb.mean())})
        if paired:
            d = sa - sb
            result["clip_diff_mean"] = float(d.mean())
            result["clip_diff_se"] = float(d.std(ddof=1) / np.sqrt(d.shape[0])) if d.shape[0] > 1 else None
    result["seconds"] = time.time() - t0
    rows = [{"quantity": k, "value": result[k]} for k in ("n", "n_a", "n_b", "prefix", "first_n", "features_equal",
                                                           "max_abs_feature_difference", "frechet") if k in result]
    rows += [{"quantity": "kid " + k, "value": result["kid"][k]} for k in ("full", "block_mean", "block_std",
                                                                           "blocks", "block_rows")]
    rows += [{"quantity": k, "value": result[k]} for k in ("clip_a", "clip_b", "clip_diff_mean", "clip_diff_se")
             if k in result]
    return _write_outputs(a, result, rows, ["quantity", "value"], det, log)


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
