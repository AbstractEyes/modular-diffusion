"""The toy collective: small engines that each learned one thing alone, served by place and thing addresses carried
by the image tokens (``scenes``, ``frame``, ``collective``, ``scene_judge``, ``scene_reads``).

    python -m geolip.modular_diffusion.run toy_collective variant=parts kind=addressed_thing classes=0-8 bed=F seed=0 out=<f>
    from geolip.modular_diffusion.experiments import toy_collective; toy_collective.main(variant="judge", bed="F", out=...)

| variant  | what it does                                                                                           |
|----------|--------------------------------------------------------------------------------------------------------|
| parts    | trains the engines of one kind (``kind``, ``classes`` = 0-8, one class, or empty for a kind without a   |
|          | class) one after another, each into its own folder under ``out`` named by ``engine_tag`` (a finished   |
|          | one is skipped)                                                                                        |
| judge    | trains the judge of a bed into ``out``; its ceiling                                                     |
| timing   | a short real training of one engine shape (``steps``): its seconds a step over steps ``from_step`` on  |
| compose  | a ROW (``row``) sampled on the request lists and judged; every read by n                               |
| controls | the controls (b) to (g) of the row's bed and seed (``row=R1r``: (c), (d), (f) on the rotary collective) |
| exact    | the exchange and snap-in reads on the rows of ``rows`` (the first ``exact_lists`` lists at n = 4)        |
| leak     | the leak read of the whole-picture parts                                                               |
| wrong    | R1 on bed F with corrupted request lists judged against the true lists                                 |
| draw     | the written layouts of a finished writer cell on the toy bed's captions (``answers``) turned into      |
|          | requests, drawn by R1 with the engines found under ``parts`` and judged; per caption W, C, P, H         |
| parity   | R1 on the earlier run's lists at n = 4 beside that run's stored row (``stored``); the judge's stored    |
|          | ceiling and the real row's scene right at nine things (``real_row``); stored wrong-address curves        |

FROM A WRITTEN LAYOUT TO A PICTURE (``draw``): each answer is read by the table-tolerant read and the strict cell
grammar and handed on as requests by ``layout.bed_requests`` (a thing that is none of the nine and an instance with no
address are dropped; the first written keeps a cell; a missing or foreign colour is drawn from ``random.Random(9000 +
the caption's index)``); the requests are drawn by R1 (``build_row``) in 50 Euler steps from the start noise of the
caption (row k of ``collective.start_noise(100, 7100 + the family's index)``, k the number of its id) and the judge
reads the nine cells. Per caption (``layout.caption_reads``, ``layout.picture_reads``): W written right (the family's
address right of the written layout), H handed right (of the requests, a drawn colour counting as no stated colour), C
drawn as written (all nine cells as the requests say, things, empties and colours) and P picture right (the family's
address right of the picture's layout); an answer that does not parse gives no picture and 0 in every read. Output:
result.json (every caption's record), result.md, sheet.jpg (the first ``sheet_n`` combined captions with the caption and
the written layout under each picture).

THE ROWS (``row``): R1 (nine addressed thing parts and the addressed base, the gate), R1r (the same in the rotary read),
R2 (nine whole-picture thing parts and the empty-scene engine, the gate), R3 (the same parts, the sum with the empty
background), R4 (the whole-picture parts, the sum with the unconditional background), R5a / R5ar (the whole model
trained on mix scenes at the engine's shape, embedding / rotary read; ``model_steps``), R5b (the same at
``model_dim``, ``model_heads``, ``model_depth``), R6 (the whole model trained on one-thing scenes), REAL (real scenes of
the lists, rendered by the bed). The parts are found under ``parts`` (searched for folders named by ``engine_tag`` with
``part_steps``, the row's read form and ``part_cells``); the judge is the folder ``judge``.

EXCHANGING NAMED PARTS (``swap``, the rows R1 and R1r in the variants compose and draw): ``swap=8:<folder>,0:newborn``
serves each named class by another finished engine, named by its folder under ``parts`` (its engine.json must say an
addressed thing part of that class, the job's bed and the row's read form; its seed, steps and shape are free), or by an
untrained part of the job's seed and shape (``newborn``); every other part is the one the job's arguments name. The job
then draws the same lists (or captions) from the same start noise twice, with the exchanged collective and with the
unexchanged one, and stores ``exact``: the largest absolute difference of the two final pictures outside the cells whose
request is an exchanged class, clipped to 0..1 as the variant exact compares them (``largest_outside``) and before the
clamp (``largest_outside_unclamped``), the number of cells compared, and the share of the exchanged classes' cells where
the two pictures differ at all; ``holds`` = both largest differences are exactly 0.0; ``real`` = some exchanged cell
exists and some differs (a draw job with nothing drawn reads None for both). The exchanged collective's reads stay where they are without ``swap``; the unexchanged one's
are stored beside them (compose: ``unswapped``, per n; draw: ``judged_unswapped``, ``C_unswapped`` and ``P_unswapped``
in every caption's record and ``tables_unswapped``). Both reads of a compose job also keep, per list, the thing read of
each cell (``thing_right_cells``: 1 right, 0 wrong, -1 no request) and the nine reads (``cell_reads``, as ``judged``). ``engines`` names every engine that served: its
folder, the class it serves (or the base), its steps, shape, trained numbers and seconds, whether it descended, the
SHA-256 of the weights file read (``weights_sha256``; None for a newborn part), and whether it serves the exchanged
collective only (``exchanged``).

SEVERAL JUDGES (compose and draw): ``judge`` may name several judge folders separated by commas (their base names must
differ). The first fills every key a single judge fills; each further judge reads the same pictures (nothing is sampled
again) and its reads are stored beside, under its folder's base name: compose ``by_judge`` (per n, the reads the row
stores), draw ``judged_by`` in every caption's record and ``tables_by``; with ``swap`` also the unexchanged
collective's (``unswapped_by_judge``; ``judged_unswapped_by``, ``tables_unswapped_by``). ``judges`` lists the base
names in order. With one folder the result is the same as before.

THE LISTS: ``scenes.request_lists(n, count)`` for n in ``ns`` (seed 5000 + n), start noise ``collective.start_noise``
seeded 7000 + n. ``lists=center`` / ``lists=others`` (the address never trained): 500 one-thing lists (seed 5100 /
5101: classes [500], colors [500], and for ``others`` the cell [500] uniform on the eight cells other than the center,
all randint), n = 1, start noise seeded 7001. A made scene is read in pixels clipped to 0..1 (the judge reads that
clipped picture in model space).

Output in ``out``: result.json, result.md, log.txt (its last line 'done'); compose also writes sheet.jpg (the first 8
scenes of n = 4 and of n = 9, enlarged 4 times).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence

import torch

from .. import __version__, determinism, frame, layout
from .. import collective as C
from .. import scene_judge as SJ
from .. import scene_reads as SR
from .. import scenes as S
from ..config import in_notebook, parse

EXPERIMENT = "toy_collective"
VARIANTS = ("parts", "judge", "timing", "compose", "controls", "exact", "leak", "wrong", "draw", "parity")
ROWS = ("R1", "R1r", "R2", "R3", "R4", "R5a", "R5ar", "R5b", "R6", "REAL")
DEFAULTS = dict(variant="parts", kind="addressed_thing", classes="0-8", bed="F", seed=0, steps=3000, dim=128, heads=2,
                depth=3, read="embedding", batch=128, lr=1e-3, warmup=100, every=250, heldout=2048, save_every=500,
                cells="", from_step=20,
                row="R1", rows="R1,R2,R3", parts="", judge="", ns="1,2,3,4,6,9", count=500, chunk=128,
                sample_steps=50, part_steps=3000, part_cells="", model_steps=30000, model_dim=128, model_heads=2,
                model_depth=3, lists="", twin_seed=1, sheet=1, judge_steps=2000, judge_batch=256,
                e="0.05,0.10,0.20", wrong_ns="2,4,6", leak_ns="2,4,9", leak_ts="0.15,0.45,0.85", leak_scenes=256,
                exact_lists=64, answers="", role="", stored="", real_row="", wrong_rows="", sheet_n=16, swap="",
                device="cpu", det="on", threads=4, out="")
SWAP_ROWS = ("R1", "R1r")
SWAP_VARIANTS = ("compose", "draw")
NEWBORN = "newborn"
LIST_SEED, NOISE_SEED = 5000, 7000
CENTER_SEED, OTHERS_SEED = 5100, 5101
CENTER = 4
LEAK_SEED, WRONG_SEED = 8000, 9000
CAPTION_NOISE_SEED = 7100          # the start noise of a caption: seeded 7100 + its family's index, row = its number
PARITY_N = 4
CONTROL_G_TS = (0.1, 0.5, 0.9)
LIMITS = {"a_rate": 0.99, "a_real_n9": 0.90, "b": 0.05, "c": 0.2, "e": 1e-12, "f": 0.2, "g": 1e-5}


# ------------------------------------------------------------------ names and finding engines
def parse_cells(text: str) -> Optional[List[int]]:
    """'0,1,2,3,5,6,7,8' or '01235678' -> the sorted cells; empty -> None (every cell)."""
    text = str(text or "").strip()
    if not text:
        return None
    items = text.split(",") if "," in text else list(text)
    return sorted({int(x) for x in items if x.strip()})


def parse_classes(text: str) -> List[Optional[int]]:
    text = str(text or "").strip()
    if not text:
        return [None]
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",")]


def engine_tag(kind: str, cls: Optional[int], bed: str, seed: int, steps: int, dim: int, heads: int, depth: int,
               read: str = "embedding", cells: Optional[Sequence[int]] = None) -> str:
    """The folder name of an engine: its kind, class, bed, seed, steps, shape, read form and (when restricted) cells."""
    name = C.engine_name(kind, cls)
    tag = f"{name}_{bed}_s{int(seed)}_n{int(steps)}_w{int(dim)}h{int(heads)}d{int(depth)}_{read}"
    if cells is not None:
        tag += "_cells" + "".join(str(int(c)) for c in sorted(cells))
    return tag


def index_engines(root: str) -> Dict[str, str]:
    """{engine folder name: path} of every finished engine (engine.json and train.json) under ``root``."""
    out: Dict[str, str] = {}
    if not root:
        return out
    for dirpath, _, files in os.walk(root):
        if "engine.json" in files and C.TRAIN_JSON in files:
            out.setdefault(os.path.basename(dirpath), dirpath)
    return out


class Parts:
    def __init__(self, a: dict):
        self.a = a
        self.index: Dict[str, str] = {}
        for root in str(a["parts"] or "").split(","):      # one folder, or several separated by commas
            for k, v in index_engines(root.strip()).items():
                self.index.setdefault(k, v)
        self.used: List[dict] = []

    def get(self, kind: str, cls: Optional[int], seed: Optional[int] = None, steps: Optional[int] = None,
            dim: Optional[int] = None, heads: Optional[int] = None, depth: Optional[int] = None,
            read: str = "embedding", cells: Optional[Sequence[int]] = None, newborn: bool = False):
        a = self.a
        seed = int(a["seed"]) if seed is None else int(seed)
        steps = int(a["part_steps"]) if steps is None else int(steps)
        dim = int(a["dim"]) if dim is None else int(dim)
        heads = int(a["heads"]) if heads is None else int(heads)
        depth = int(a["depth"]) if depth is None else int(depth)
        if newborn:
            return C.born(kind, cls, seed, dim, heads, depth, read).to(a["device"]).eval()
        tag = engine_tag(kind, cls, a["bed"], seed, steps, dim, heads, depth, read, cells)
        if tag not in self.index:
            raise FileNotFoundError(f"no finished engine {tag} under {a['parts']}")
        path = self.index[tag]
        with open(os.path.join(path, C.TRAIN_JSON), encoding="utf-8") as fh:
            tj = json.load(fh)
        self.used.append({"tag": tag, "descended": bool(tj.get("descended")), "newborn_read": tj.get("newborn_read"),
                          "last_read": tj.get("last_read"), "trained": tj.get("trained"),
                          "train_seconds": tj.get("train_seconds")})
        return C.load_engine(path, average=True, device=a["device"])

    def named(self, cls: int, folder: str, read: str):
        """The finished engine whose folder is ``folder``, to serve the class ``cls`` of an addressed collective of the
        read form ``read`` (a ValueError naming the pair when it is not found or does not match)."""
        a = self.a
        pair = f"{int(cls)}:{folder}"
        if folder not in self.index:
            raise ValueError(f"swap {pair}: no finished engine named {folder} under {a['parts']}")
        path = self.index[folder]
        with open(os.path.join(path, "engine.json"), encoding="utf-8") as fh:
            info = json.load(fh)
        wrong = []
        if info.get("kind") != "addressed_thing":
            wrong.append(f"its kind is {info.get('kind')!r}, not 'addressed_thing'")
        if info.get("cls") is None or int(info["cls"]) != int(cls):
            wrong.append(f"its class is {info.get('cls')}, not {int(cls)}")
        if info.get("bed") != a["bed"]:
            wrong.append(f"its bed is {info.get('bed')!r}, not the job's {a['bed']!r}")
        if (info.get("settings") or {}).get("read", "embedding") != read:
            wrong.append(f"its read form is {(info.get('settings') or {}).get('read')!r}, not the row's {read!r}")
        if wrong:
            raise ValueError(f"swap {pair}: " + "; ".join(wrong))
        with open(os.path.join(path, C.TRAIN_JSON), encoding="utf-8") as fh:
            tj = json.load(fh)
        self.used.append({"tag": folder, "descended": bool(tj.get("descended")), "newborn_read": tj.get("newborn_read"),
                          "last_read": tj.get("last_read"), "trained": tj.get("trained"),
                          "train_seconds": tj.get("train_seconds")})
        return C.load_engine(path, average=True, device=a["device"])

    def describe(self, entry: dict, serves, exchanged: bool) -> dict:
        """A used engine's entry with what its own files say: the class it serves (or "base"), its seed, steps, shape
        and trained numbers, and whether it serves the exchanged collective only."""
        with open(os.path.join(self.index[entry["tag"]], "engine.json"), encoding="utf-8") as fh:
            info = json.load(fh)
        s = info.get("settings") or {}
        return dict(entry, serves=serves, exchanged=bool(exchanged), newborn=False, seed=info.get("seed"),
                    steps=info.get("steps"), shape=[s.get("dim"), s.get("heads"), s.get("depth")],
                    read=s.get("read", "embedding"), trained=info.get("trained", entry.get("trained")),
                    weights_sha256=weights_sha256(self.index[entry["tag"]]))


def weights_sha256(folder: str) -> str:
    """The SHA-256 of the weights file ``collective.load_engine`` reads for the served numbers: the averaged weights
    (engine_avg.safetensors) when the folder holds them, else engine.safetensors."""
    import hashlib
    path = os.path.join(folder, "engine_avg.safetensors")
    if not os.path.isfile(path):
        path = os.path.join(folder, "engine.safetensors")
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def judge_name(folder: str) -> str:
    """A judge's name in a result: the base name of its folder."""
    return os.path.basename(os.path.normpath(folder))


def judge_folders(text: str) -> List[str]:
    """'<folder>' or '<folder>,<folder>,...' -> the judge folders in order (two with one base name: a ValueError)."""
    folders = [x.strip() for x in str(text or "").split(",") if x.strip()]
    names = [judge_name(f) for f in folders]
    twice = sorted({n for n in names if names.count(n) > 1})
    if twice:
        raise ValueError(f"judge: two judge folders with the same base name {twice[0]!r}")
    return folders


def parse_swap(text: str) -> Dict[int, str]:
    """'8:<engine folder>,0:newborn' -> {class: engine folder or "newborn"}; '' -> {}."""
    out: Dict[int, str] = {}
    for item in [x.strip() for x in str(text or "").split(",") if x.strip()]:
        if ":" not in item:
            raise ValueError(f"swap: {item!r} is not a class:engine pair")
        k, eng = (x.strip() for x in item.split(":", 1))
        if not k.isdigit() or not 0 <= int(k) <= 8:
            raise ValueError(f"swap: the class of {item!r} is not one of 0 to 8")
        if not eng:
            raise ValueError(f"swap: {item!r} names no engine")
        if int(k) in out:
            raise ValueError(f"swap: the class {int(k)} is named twice")
        out[int(k)] = eng
    return out


def swap_rows(row: str, parts: Parts, swap: Dict[int, str]):
    """-> (the exchanged collective's answer, the unexchanged one's, a description, the engines that served). The
    unexchanged collective is the row as the job's arguments name it; the exchanged one shares its engines except for
    the classes of ``swap``."""
    if row not in SWAP_ROWS:
        raise ValueError(f"swap applies to the addressed collective (the rows {', '.join(SWAP_ROWS)}), not the row {row}")
    a = parts.a
    cells = parse_cells(a["part_cells"])
    read = "rotary" if row == "R1r" else "embedding"
    n0 = len(parts.used)
    things = {k: parts.get("addressed_thing", k, read=read, cells=cells) for k in range(9)}
    base = parts.get("addressed_base", None, read=read)
    plain = parts.used[n0:]
    engines = [parts.describe(u, k if k < 9 else "base", False) for k, u in enumerate(plain)]
    exchanged = dict(things)
    for k, folder in sorted(swap.items()):
        if folder == NEWBORN:
            exchanged[k] = parts.get("addressed_thing", k, read=read, newborn=True)
            engines.append({"tag": NEWBORN, "serves": k, "exchanged": True, "newborn": True, "seed": int(a["seed"]),
                            "steps": 0, "shape": [int(a["dim"]), int(a["heads"]), int(a["depth"])], "read": read,
                            "trained": C.param_count(exchanged[k]), "descended": None, "newborn_read": None,
                            "last_read": None, "train_seconds": 0.0, "weights_sha256": None})
        else:
            exchanged[k] = parts.named(k, folder, read)
            engines.append(parts.describe(parts.used[-1], k, True))
    col_x = C.Collective(exchanged, base, addressed=True)
    col_u = C.Collective(things, base, addressed=True)
    desc = f"{row}: addressed parts, the gate ({read} read), classes {sorted(swap)} exchanged"
    return ((lambda z, t, c, k: col_x.answer(z, t, c, k, "gate")), (lambda z, t, c, k: col_u.answer(z, t, c, k, "gate")),
            desc, engines)


def exchange_exact(xa: torch.Tensor, xb: torch.Tensor, cells: torch.Tensor, classes: Sequence[int],
                   raw_a: Optional[torch.Tensor] = None, raw_b: Optional[torch.Tensor] = None) -> dict:
    """Two batches of pictures of the same lists: the largest |xa - xb| over the cells whose request is not one of
    ``classes``, the number of those cells, and how many cells of ``classes`` differ at all; with ``raw_a``, ``raw_b``
    (the same pictures before the clamp to 0..1) also the largest difference outside before the clamp."""
    inside = torch.zeros_like(cells, dtype=torch.bool)
    for k in classes:
        inside |= cells == int(k)

    def largest(a_, b_):
        d_ = S.cell_crops((a_ - b_).abs())                             # [B, 9, C, 16, 16]
        out_v = d_[~inside.to(d_.device)]
        return d_, (float(out_v.max()) if out_v.numel() else 0.0)

    d, lo = largest(xa, xb)
    per_cell = d.flatten(2).amax(dim=-1)                               # [B, 9]
    out = {"largest_outside": lo, "cells_compared": int((~inside).sum()), "exchanged_cells": int(inside.sum()),
           "exchanged_cells_differing": int((per_cell[inside.to(d.device)] > 0).sum())}
    if raw_a is not None and raw_b is not None:
        out["largest_outside_unclamped"] = largest(raw_a, raw_b)[1]
    return out


def exact_summary(parts_: Sequence[dict]) -> dict:
    """The exactness reads of several batches pooled: the largest difference outside (after and before the clamp), the
    cells compared, the share of the exchanged classes' cells that differ (None when the lists hold none); ``holds``
    (both largest differences exactly 0.0) and ``real`` (some exchanged cell exists and some differs)."""
    largest = max((p["largest_outside"] for p in parts_), default=0.0)
    raw = [p.get("largest_outside_unclamped") for p in parts_]
    largest_raw = None if any(v is None for v in raw) else max(raw, default=0.0)
    ex = sum(p["exchanged_cells"] for p in parts_)
    dif = sum(p["exchanged_cells_differing"] for p in parts_)
    return {"compared": "the final pictures, cell by cell, clipped to 0..1 and before the clamp",
            "largest_outside": largest, "largest_outside_unclamped": largest_raw,
            "cells_compared": sum(p["cells_compared"] for p in parts_), "exchanged_cells": ex,
            "exchanged_cells_differing": dif, "inside_share": dif / ex if ex else None,
            "holds": bool(largest == 0.0 and largest_raw == 0.0), "real": bool(ex > 0 and dif > 0)}


def build_row(row: str, parts: Parts, newborn: bool = False, twin: Optional[Dict[int, int]] = None,
              without: Sequence[int] = ()):
    """-> (answer(z, t, cells, colors), description). ``twin`` {class: seed}: that class's part from another seed;
    ``without``: classes left out of the collective."""
    a = parts.a
    twin = twin or {}
    cells = parse_cells(a["part_cells"])
    read = "rotary" if row in ("R1r", "R5ar") else "embedding"
    if row in ("R1", "R1r"):
        things = {k: parts.get("addressed_thing", k, seed=twin.get(k), read=read, cells=cells, newborn=newborn)
                  for k in range(9) if k not in without}
        col = C.Collective(things, parts.get("addressed_base", None, read=read, newborn=newborn), addressed=True)
        return (lambda z, t, c, k: col.answer(z, t, c, k, "gate")), f"{row}: addressed parts, the gate ({read} read)"
    if row in ("R2", "R3", "R4"):
        things = {k: parts.get("whole_thing", k, seed=twin.get(k), newborn=newborn) for k in range(9)
                  if k not in without}
        if row == "R4":
            col = C.Collective(things, None, addressed=False,
                               unconditional=parts.get("whole_unconditional", None, newborn=newborn))
            return (lambda z, t, c, k: col.answer(z, t, c, k, "sum_unconditional")), "R4: the sum, unconditional background"
        col = C.Collective(things, parts.get("whole_empty", None, newborn=newborn), addressed=False)
        join = "gate" if row == "R2" else "sum_empty"
        return (lambda z, t, c, k: col.answer(z, t, c, k, join)), f"{row}: whole-picture parts, {join}"
    if row in ("R5a", "R5ar", "R5b", "R6"):
        kind = "model_one" if row == "R6" else "model_mix"
        shape = dict(dim=a["model_dim"], heads=a["model_heads"], depth=a["model_depth"]) if row == "R5b" else {}
        eng = parts.get(kind, None, steps=int(a["model_steps"]), read=read, newborn=newborn, **shape)
        wm = C.WholeModel(eng)
        return wm.answer, f"{row}: one whole model ({kind}, {read} read)"
    raise ValueError(f"the rows are {ROWS}, not {row!r}")


# ------------------------------------------------------------------ lists, sampling, judging
def lists_for(a: dict) -> Dict[int, dict]:
    """{n: {"cells", "colors", "order", "noise_seed"}} of the row's lists."""
    count = int(a["count"])
    if a["lists"] in ("center", "others"):
        gen = torch.Generator().manual_seed(CENTER_SEED if a["lists"] == "center" else OTHERS_SEED)
        cls = torch.randint(0, 9, (count,), generator=gen)
        col = torch.randint(0, 9, (count,), generator=gen)
        if a["lists"] == "center":
            where = torch.full((count,), CENTER, dtype=torch.long)
        else:
            others = torch.tensor([c for c in range(9) if c != CENTER])
            where = others[torch.randint(0, 8, (count,), generator=gen)]
        cells = torch.full((count, 9), -1, dtype=torch.long)
        colors = torch.full((count, 9), -1, dtype=torch.long)
        cells[torch.arange(count), where] = cls
        colors[torch.arange(count), where] = col
        return {1: {"cells": cells, "colors": colors, "order": where[:, None], "noise_seed": NOISE_SEED + 1}}
    if a["lists"]:
        raise ValueError("lists is empty, center or others")
    out = {}
    for n in [int(x) for x in str(a["ns"]).split(",") if x.strip()]:
        rl = S.request_lists(n, count, seed=LIST_SEED + n)
        out[n] = {"cells": rl.cells, "colors": rl.colors, "order": rl.order, "noise_seed": NOISE_SEED + n}
    return out


def made_pixels(z: torch.Tensor) -> torch.Tensor:
    return S.to_pixels(z).clamp(0.0, 1.0)


def real_pixels(cells, colors, bed: str, seed: int) -> torch.Tensor:
    pix, _ = S.render(cells, colors, bed, torch.Generator().manual_seed(int(seed)))
    return pix


def judge_reads(judge, pixels: torch.Tensor, cells, colors, cell_rights: bool = False) -> dict:
    """The judge's reads of made scenes; ``cell_rights``: also, per list, each cell's thing read (1 right, 0 wrong,
    -1 no request) and its nine reads ([thing, colour], or [None, None] for a cell read as empty)."""
    pt, pc = SJ.predict(judge, S.to_model(pixels))
    s = SR.score_cells(pt, pc, cells, colors)
    req = s["requested"].double()
    per_cell = (s["thing_right"].double().sum(1) / req.sum(1).clamp_min(1)).tolist()
    summ = SR.summarize(pt, pc, cells, colors)
    reqb = s["requested"]
    wrong = reqb & ~s["thing_right"]
    summ["missing_share"] = SR.rate((pt == SR.NOTHING)[wrong])          # wrong requested cells read as NOTHING
    summ["by_class"] = {str(k): SR.rate(s["thing_right"][reqb & (cells == k)]) for k in range(9)}
    summ["per_list"] = {"scene_right": [int(x) for x in s["scene_right"].tolist()],
                        "scene_right_colors": [int(x) for x in s["scene_right_colors"].tolist()],
                        "cell_rate": per_cell, "n_right": [int(x) for x in s["n_right"].tolist()]}
    if cell_rights:
        right = torch.where(reqb, s["thing_right"].long(), torch.full_like(s["thing_right"].long(), -1))
        summ["per_list"]["thing_right_cells"] = right.tolist()
        summ["per_list"]["cell_reads"] = [[[None, None] if int(t) == SR.NOTHING else [int(t), int(c)]
                                           for t, c in zip(tr, cr)] for tr, cr in zip(pt.tolist(), pc.tolist())]
    return summ


def picture_reads(pixels: torch.Tensor, cells, colors, bed: str, real: torch.Tensor) -> dict:
    out = {}
    if bed == "C":
        spread = float(SR.tint_spread(pixels).mean())
        r = float(SR.tint_spread(real).mean())
        sh = float(SR.tint_spread(SR.shuffle_cells(real)).mean()) if real.shape[0] >= 9 else None
        out["tint"] = {"spread": spread, "real": r, "shuffled": sh,
                       "coherence_index": SR.coherence_index(spread, r, sh) if sh is not None and sh != r else None}
        sm, sr_ = SR.seam(pixels, cells), SR.seam(real, cells)
        out["seam"] = {k: sm[k] for k in SR.BORDER_TYPES}
        out["seam_real"] = {k: sr_[k] for k in SR.BORDER_TYPES}
    else:
        g = SR.ghost_ink(pixels, cells)
        out["ghost_ink"] = {"share": g["share"], "n": g["n"]}
    return out


def write_sheet(path: str, rows: Sequence[torch.Tensor]) -> None:
    from PIL import Image
    rows = [r[:8] for r in rows if r is not None and r.shape[0]]
    if not rows:
        return
    w = 8 * S.CANVAS
    sheet = torch.ones(3, len(rows) * S.CANVAS, w)
    for i, r in enumerate(rows):
        for j in range(r.shape[0]):
            sheet[:, i * S.CANVAS:(i + 1) * S.CANVAS, j * S.CANVAS:(j + 1) * S.CANVAS] = r[j]
    arr = (sheet.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(arr).resize((arr.shape[1] * 4, arr.shape[0] * 4), Image.NEAREST).save(path, quality=92)


# ------------------------------------------------------------------ variants
def _run_parts(a: dict, log) -> dict:
    bed = S.ScenesBed(a["bed"], a["device"])
    cells = parse_cells(a["cells"])
    out = []
    for k in parse_classes(a["classes"] if a["kind"] in C.THING_KINDS else ""):
        tag = engine_tag(a["kind"], k, a["bed"], a["seed"], a["steps"], a["dim"], a["heads"], a["depth"], a["read"],
                         cells)
        r = C.train_engine(a["kind"], k, bed, int(a["seed"]), os.path.join(a["out"], tag), steps=int(a["steps"]),
                           batch=int(a["batch"]), lr=float(a["lr"]), warmup=int(a["warmup"]), every=int(a["every"]),
                           heldout=int(a["heldout"]), dim=int(a["dim"]), heads=int(a["heads"]), depth=int(a["depth"]),
                           save_every=int(a["save_every"]), read=a["read"], allowed_cells=cells, log=log)
        out.append({"tag": tag, "cls": k, "newborn_read": r.get("newborn_read"), "last_read": r.get("last_read"),
                    "descended": r.get("descended"), "trained": r.get("trained"),
                    "train_seconds": r.get("train_seconds"), "seconds_a_step": r.get("seconds_a_step"),
                    "curve": r.get("curve")})
    return {"engines": out}


def _run_judge(a: dict, log) -> dict:
    bed = S.ScenesBed(a["bed"], a["device"])
    r = SJ.train_judge(bed, int(a["seed"]), a["out"], steps=int(a["judge_steps"]), batch=int(a["judge_batch"]), log=log)
    c = r["ceiling"]
    r["holds_rates"] = bool(c["thing_rate"]["mean"] >= LIMITS["a_rate"] and c["color_rate"]["mean"] >= LIMITS["a_rate"])
    return {"judge": r}


def _run_timing(a: dict, log) -> dict:
    bed = S.ScenesBed(a["bed"], a["device"])
    folder = os.path.join(a["out"], "timing_engine")
    k = 0 if a["kind"] in C.THING_KINDS else None
    C.train_engine(a["kind"], k, bed, int(a["seed"]), folder, steps=int(a["steps"]), batch=int(a["batch"]),
                   every=0, heldout=int(a["heldout"]), dim=int(a["dim"]), heads=int(a["heads"]),
                   depth=int(a["depth"]), save_every=0, read=a["read"], log=log)
    with open(os.path.join(folder, "checkpoint.json"), encoding="utf-8") as fh:
        secs = json.load(fh)["step_seconds"]
    tail = sorted(secs[int(a["from_step"]):]) or sorted(secs)
    med = tail[len(tail) // 2]
    return {"timing": {"steps": len(secs), "from_step": int(a["from_step"]), "seconds_a_step": med,
                       "mean_seconds_a_step": sum(secs[int(a["from_step"]):]) / max(1, len(secs) - int(a["from_step"])),
                       "shape": [int(a["dim"]), int(a["heads"]), int(a["depth"])], "read": a["read"]}}


def _sample_lists(answer, lists: dict, a: dict) -> torch.Tensor:
    return C.sample(answer, lists["cells"], lists["colors"], lists["noise_seed"], int(a["sample_steps"]), a["device"],
                    int(a["chunk"]))


def _run_compose(a: dict, log) -> dict:
    row = a["row"]
    parts = Parts(a)
    folders = judge_folders(a["judge"])
    if not folders:
        raise ValueError("compose needs judge=<the judge's folder>")
    judge = SJ.load_judge(folders[0], a["device"])
    more = [(judge_name(f), SJ.load_judge(f, a["device"])) for f in folders[1:]]
    by_judge = {name: {} for name, _ in more}
    unswapped_by = {name: {} for name, _ in more}
    swap = parse_swap(a["swap"])
    if swap:
        answer, answer_u, desc, engines = swap_rows(row, parts, swap)
    else:
        answer, desc = (None, "REAL: real scenes of the lists") if row == "REAL" else build_row(row, parts)
    by_n, sheets, unswapped, exact = {}, {}, {}, {}
    for n, L in lists_for(a).items():
        t0 = time.time()
        real = real_pixels(L["cells"], L["colors"], a["bed"], L["noise_seed"])
        z = None if row == "REAL" else _sample_lists(answer, L, a)
        pix = real if row == "REAL" else made_pixels(z)
        rd = judge_reads(judge, pix, L["cells"], L["colors"], cell_rights=bool(swap))
        rd.update(picture_reads(pix, L["cells"], L["colors"], a["bed"], real))
        rd["seconds"] = time.time() - t0
        by_n[str(n)] = rd
        if n in (4, 9) or a["lists"]:
            sheets[n] = pix[:8]
        log(f"{row} n={n}: cell rate {rd['cell_rate']['mean']:.3f}, scene right {rd['scene_right']['mean']:.3f}, "
            f"ghosts {rd['ghosts']['mean'] if rd['ghosts']['mean'] is not None else '-'}; {rd['seconds']:.0f} s")
        for name, jm in more:                    # a further judge reads the same pictures
            by_judge[name][str(n)] = dict(judge_reads(jm, pix, L["cells"], L["colors"], cell_rights=bool(swap)),
                                          **picture_reads(pix, L["cells"], L["colors"], a["bed"], real))
        if swap:
            t1 = time.time()
            z_u = _sample_lists(answer_u, L, a)
            pix_u = made_pixels(z_u)
            ru = judge_reads(judge, pix_u, L["cells"], L["colors"], cell_rights=True)
            ru.update(picture_reads(pix_u, L["cells"], L["colors"], a["bed"], real))
            ru["seconds"] = time.time() - t1
            unswapped[str(n)] = ru
            for name, jm in more:
                unswapped_by[name][str(n)] = dict(judge_reads(jm, pix_u, L["cells"], L["colors"], cell_rights=True),
                                                  **picture_reads(pix_u, L["cells"], L["colors"], a["bed"], real))
            exact[str(n)] = exchange_exact(pix, pix_u, L["cells"], sorted(swap), S.to_pixels(z), S.to_pixels(z_u))
            log(f"{row} n={n} unexchanged: cell rate {ru['cell_rate']['mean']:.3f}, scene right "
                f"{ru['scene_right']['mean']:.3f}; largest difference outside the exchanged cells "
                f"{exact[str(n)]['largest_outside']}; {ru['seconds']:.0f} s")
    if int(a["sheet"]):
        write_sheet(os.path.join(a["out"], "sheet.jpg"), [sheets[k] for k in sorted(sheets)])
    bad = [u["tag"] for u in parts.used if not u["descended"]]
    out = {"row": row, "description": desc, "by_n": by_n, "engines": parts.used, "not_descended": bad,
           "inconclusive": bool(bad), "lists": a["lists"] or "request_lists", "count": int(a["count"])}
    if swap:
        out.update(engines=engines, swap={str(k): v for k, v in sorted(swap.items())}, unswapped=unswapped,
                   exact=dict(exact_summary(list(exact.values())), by_n=exact))
        log(f"exchange exact: {out['exact']['holds']} (largest difference outside {out['exact']['largest_outside']}, "
            f"{out['exact']['cells_compared']} cells compared; exchanged cells that differ {out['exact']['inside_share']})")
    if more:
        out.update(judges=[judge_name(f) for f in folders], by_judge=by_judge)
        if swap:
            out["unswapped_by_judge"] = unswapped_by
    return out


def _gate_identities(lists: Dict[int, dict]) -> dict:
    parts = [[k] for k in range(9)]
    worst_k, one_owner, off_zero = 0.0, True, True
    for L in lists.values():
        wi = frame.token_weights(L["cells"], parts, "integer")
        wk = frame.token_weights(L["cells"], parts, "kernel")
        worst_k = max(worst_k, float((wi - wk).abs().max()))
        one_owner = one_owner and bool(((wi == 1.0).sum(-1) == 1).all()) and bool(((wi == 0) | (wi == 1)).all())
        own = frame.token_owner(L["cells"], parts)
        off_zero = off_zero and bool((wi.gather(-1, own[..., None]) == 1.0).all()) and \
            float(wi.sum(-1).sub(1.0).abs().max()) == 0.0
    return {"kernel_minus_integer": worst_k, "one_owner_weight_one": one_owner, "off_address_zero": off_zero,
            "holds": bool(one_owner and off_zero and worst_k <= LIMITS["e"])}


def _run_controls(a: dict, log) -> dict:
    parts = Parts(a)
    judge = SJ.load_judge(a["judge"], a["device"])
    lists = lists_for(a)
    rotary = a["row"] == "R1r"
    row = "R1r" if rotary else "R1"
    out: Dict[str, dict] = {}
    if not rotary:
        b = {}
        for n, L in lists.items():
            if n < 2:
                continue
            real = real_pixels(L["cells"], L["colors"], a["bed"], L["noise_seed"])
            nxt_c, nxt_k = torch.roll(L["cells"], -1, 0), torch.roll(L["colors"], -1, 0)
            b[str(n)] = judge_reads(judge, real, nxt_c, nxt_k)["scene_right"]
        out["b"] = {"by_n": b, "holds": all(v["mean"] <= LIMITS["b"] for v in b.values()) if b else None}
        out["e"] = _gate_identities(lists)
        things ={k: parts.get("whole_thing", k) for k in range(9)}
        L1 = S.request_lists(1, int(a["exact_lists"]), seed=LIST_SEED + 1)
        z0 = C.start_noise(L1.cells.shape[0], NOISE_SEED + 1).to(a["device"])
        worst = 0.0
        col = C.Collective(things, parts.get("whole_empty", None), addressed=False)
        with torch.no_grad():
            for tv in CONTROL_G_TS:
                t = torch.full((z0.shape[0],), tv, device=a["device"])
                c, k = L1.cells.to(a["device"]), L1.colors.to(a["device"])
                v3 = col.answer(z0, t, c, k, "sum_empty")
                own = torch.zeros_like(v3)
                for kk, (rows, cols, va) in col.pair_answers(z0, t, c, k).items():
                    own[rows] = va
                worst = max(worst, float((v3 - own).abs().max()))
        out["g"] = {"largest_difference": worst, "holds": worst <= LIMITS["g"]}
    answer, _ = build_row(row, parts)
    c_true, c_der = {}, {}
    for n, L in lists.items():
        der = SR.derange_classes(L["cells"])
        z = _sample_lists(answer, {"cells": der, "colors": L["colors"], "noise_seed": L["noise_seed"]}, a)
        pix = made_pixels(z)
        c_true[str(n)] = judge_reads(judge, pix, L["cells"], L["colors"])["cell_rate"]
        c_der[str(n)] = judge_reads(judge, pix, der, L["colors"])["cell_rate"]
    out["c"] = {"against_true": c_true, "against_deranged": c_der,
                "holds_true": all(v["mean"] <= LIMITS["c"] for v in c_true.values()),
                "note": "the second clause (against the deranged lists within the bar of the row) is judged by the "
                        "table tool beside the row's own result"}
    L4 = S.request_lists(4, int(a["count"]), seed=LIST_SEED + 4)
    first = {"cells": L4.cells[:int(a["exact_lists"])], "colors": L4.colors[:int(a["exact_lists"])],
             "noise_seed": NOISE_SEED + 4}
    z1, z2 = _sample_lists(answer, first, a), _sample_lists(answer, first, a)
    out["d"] = {"lists": int(first["cells"].shape[0]), "identical": bool(torch.equal(z1, z2)),
                "holds": bool(torch.equal(z1, z2))}
    nb, _ = build_row(row, parts, newborn=True)
    f = {}
    for n, L in lists.items():
        f[str(n)] = judge_reads(judge, made_pixels(_sample_lists(nb, L, a)), L["cells"], L["colors"])["cell_rate"]
    out["f"] = {"by_n": f, "holds": all(v["mean"] <= LIMITS["f"] for v in f.values())}
    for k, v in out.items():
        log(f"control ({k}) {row}: holds {v.get('holds', v.get('holds_true'))}")
    return {"row": row, "controls": out, "engines": parts.used,
            "inconclusive": any(not u["descended"] for u in parts.used)}


def _run_exact(a: dict, log) -> dict:
    parts = Parts(a)
    L4 = S.request_lists(4, int(a["count"]), seed=LIST_SEED + 4)
    m = int(a["exact_lists"])
    first = {"cells": L4.cells[:m], "colors": L4.colors[:m], "order": L4.order[:m]}
    nostar = (L4.cells != 8).all(dim=1)
    idx = torch.nonzero(nostar, as_tuple=True)[0][:m]
    snap = {"cells": L4.cells[idx], "colors": L4.colors[idx]}
    noise = C.start_noise(int(a["count"]), NOISE_SEED + 4)
    out = {}
    for row in [r.strip() for r in str(a["rows"]).split(",") if r.strip()]:
        base_answer, _ = build_row(row, parts)
        ex = []
        first_cls = first["cells"].gather(1, first["order"][:, :1])[:, 0]      # the class of each list's first request
        for k in range(9):
            sel = torch.nonzero(first_cls == k, as_tuple=True)[0]
            if not sel.numel():
                continue
            cs, ks = first["cells"][sel], first["colors"][sel]
            twin_answer, _ = build_row(row, parts, twin={k: int(a["twin_seed"])})
            z_a = C.sample(base_answer, cs, ks, 0, int(a["sample_steps"]), a["device"], int(a["chunk"]), noise=noise[sel])
            z_b = C.sample(twin_answer, cs, ks, 0, int(a["sample_steps"]), a["device"], int(a["chunk"]), noise=noise[sel])
            o, i = C.outside_inside(made_pixels(z_a), made_pixels(z_b), cs, k)
            ex.append({"class": k, "lists": int(sel.numel()), "largest_outside": o, "mean_inside": i})
        without, _ = build_row(row, parts, without=(8,))
        z_a = C.sample(without, snap["cells"], snap["colors"], 0, int(a["sample_steps"]), a["device"], int(a["chunk"]),
                       noise=noise[idx])
        z_b = C.sample(base_answer, snap["cells"], snap["colors"], 0, int(a["sample_steps"]), a["device"],
                       int(a["chunk"]), noise=noise[idx])
        snap_change = float((made_pixels(z_a) - made_pixels(z_b)).abs().max())
        lo = max((e["largest_outside"] for e in ex), default=None)
        out[row] = {"exchange": ex, "largest_outside": lo, "exchange_exact": lo == 0.0,
                    "snap_in_lists": int(idx.numel()), "snap_in_largest": snap_change, "snap_in_exact": snap_change == 0.0}
        log(f"exact {row}: exchange largest outside {lo}; snap-in largest {snap_change}")
    return {"exact": out, "engines": parts.used, "inconclusive": any(not u["descended"] for u in parts.used)}


def _run_leak(a: dict, log) -> dict:
    parts = Parts(a)
    bed = S.ScenesBed(a["bed"], a["device"])
    things = {k: parts.get("whole_thing", k) for k in range(9)}
    empty = parts.get("whole_empty", None)
    out = []
    ts = [float(x) for x in str(a["leak_ts"]).split(",") if x.strip()]
    for n in [int(x) for x in str(a["leak_ns"]).split(",") if x.strip()]:
        for ti, tv in enumerate(ts):
            r = C.leak_read(things, empty, bed, n, tv, int(a["leak_scenes"]), LEAK_SEED + 100 * n + ti)
            out.append(r)
            log(f"leak n={n} t={tv}: other cells {r['other_cells']}, free cells {r['free_cells']}")
    return {"leak": out, "engines": parts.used, "inconclusive": any(not u["descended"] for u in parts.used)}


def _run_wrong(a: dict, log) -> dict:
    parts = Parts(a)
    judge = SJ.load_judge(a["judge"], a["device"])
    answer, _ = build_row("R1", parts)
    es = [0.0] + [float(x) for x in str(a["e"]).split(",") if x.strip()]
    out = []
    for n in [int(x) for x in str(a["wrong_ns"]).split(",") if x.strip()]:
        rl = S.request_lists(n, int(a["count"]), seed=LIST_SEED + n)
        for e in es:
            gen = torch.Generator().manual_seed(WRONG_SEED + 100 * n + int(round(1000 * e)))
            cor = SR.corrupt_requests(rl.cells, rl.colors, e, gen)
            z = _sample_lists(answer, {"cells": cor.cells, "colors": cor.colors, "noise_seed": NOISE_SEED + n}, a)
            rd = judge_reads(judge, made_pixels(z), rl.cells, rl.colors)
            counts = torch.bincount(cor.action[rl.cells >= 0], minlength=len(SR.CORRUPTIONS)).tolist()
            out.append({"n": n, "e": e, "scene_right": rd["scene_right"], "cell_rate": rd["cell_rate"],
                        "per_list": rd["per_list"]["scene_right"],
                        "corruptions": dict(zip(SR.CORRUPTIONS, counts))})
            log(f"wrong n={n} e={e}: scene right {rd['scene_right']['mean']:.3f}")
    return {"wrong": out, "engines": parts.used, "inconclusive": any(not u["descended"] for u in parts.used)}


# ------------------------------------------------------------------ written layouts drawn and judged
def pictures_sha256(pixels: torch.Tensor) -> str:
    """The sha256 of a batch of pictures (float32, contiguous, on the CPU)."""
    import hashlib
    return hashlib.sha256(pixels.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()).hexdigest()


def _rate_of(values) -> dict:
    return layout.rate([int(v) for v in values])


def _read_answers(folder: str):
    with open(os.path.join(folder, "result.json"), encoding="utf-8") as fh:
        res = json.load(fh)
    with open(os.path.join(folder, "outputs.jsonl"), encoding="utf-8") as fh:
        outputs = [json.loads(x) for x in fh if x.strip()]
    return res, outputs


def caption_noise(records: Sequence[dict]) -> torch.Tensor:
    """The start noise of each caption: row k (the number of its id) of ``start_noise(100, 7100 + its family's
    index)``, the same for every writer and every seed."""
    cache: Dict[str, torch.Tensor] = {}
    rows = []
    for r in records:
        fam = r["family"]
        if fam not in cache:
            cache[fam] = C.start_noise(layout.PER_FAMILY, CAPTION_NOISE_SEED + layout.FAMILIES.index(fam))
        rows.append(cache[fam][int(r["id"].rsplit("_", 1)[1])])
    return torch.stack(rows) if rows else torch.zeros(0, S.CHANNELS, S.CANVAS, S.CANVAS)


def drawn_tables(records: Sequence[dict]) -> dict:
    """The reported numbers of a set of caption records: per family and over the six, P, W, C, H (rates over the
    captions) and the means of W x C and H x C; P by the number of asked instances; the 2 x 2 table of W and C with P
    inside each cell; the handoff loss (W = 1 and H = 0) with its kinds; the per-instance error of the written
    layouts (errors over asked instances, over the captions)."""
    out: dict = {"families": {}}
    groups = [(f, [r for r in records if r["family"] == f]) for f in layout.FAMILIES]
    for name, rs in groups + [("all", list(records))]:
        if not rs:
            continue
        loss = [r for r in rs if r["W"] == 1 and r["H"] == 0]
        kinds: Dict[str, int] = {}
        for r in loss:
            for d in {x["why"] for x in r["dropped"]}:
                kinds[d] = kinds.get(d, 0) + 1
            for k in r.get("kinds_handed") or ():
                kinds["handed: " + k] = kinds.get("handed: " + k, 0) + 1
        errors = sum(int(r["instance_errors"]) for r in rs)
        asked = sum(int(r["instances_asked"]) for r in rs)
        block = {"n": len(rs), **{k: _rate_of(r[k] for r in rs) for k in ("P", "W", "C", "H")},
                 "WC": sum(r["W"] * r["C"] for r in rs) / len(rs), "HC": sum(r["H"] * r["C"] for r in rs) / len(rs),
                 "WH": sum(r["W"] * r["H"] for r in rs) / len(rs),
                 "lucky": sum(1 for r in rs if r["W"] == 0 and r["P"] == 1) / len(rs),
                 "handoff_loss": {"rate": len(loss) / len(rs), "captions": len(loss), "kinds": kinds},
                 "per_instance_error": {"errors": errors, "asked": asked, "rate": errors / asked if asked else None}}
        if name == "all":
            out["all"] = block
        else:
            out["families"][name] = block
    by_n: Dict[str, dict] = {}
    for k in sorted({int(r["instances_asked"]) for r in records}):
        by_n[str(k)] = _rate_of(r["P"] for r in records if int(r["instances_asked"]) == k)
    out["P_by_asked"] = by_n
    out["W_by_C"] = {f"W{w}_C{c}": {"captions": len(sel), "P": _rate_of(r["P"] for r in sel)}
                     for w in (0, 1) for c in (0, 1)
                     for sel in [[r for r in records if r["W"] == w and r["C"] == c]]}
    return out


def write_caption_sheet(path: str, items: Sequence[tuple]) -> None:
    """A sheet of pictures (enlarged 4 times), each with its caption and its written layout under it: ``items`` =
    [(pixels [3, 48, 48] or None, caption, written layout)], four a row."""
    from PIL import Image, ImageDraw, ImageFont
    import textwrap
    if not items:
        return
    side, text_h, width = S.CANVAS * 4, 150, 260
    cols = 4
    rows = -(-len(items) // cols)
    sheet = Image.new("RGB", (cols * width, rows * (side + text_h)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for i, (pix, caption, written) in enumerate(items):
        x, y = (i % cols) * width, (i // cols) * (side + text_h)
        if pix is not None:
            arr = (pix.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
            sheet.paste(Image.fromarray(arr).resize((side, side), Image.NEAREST), (x + (width - side) // 2, y))
        else:
            draw.text((x + 8, y + side // 2), "no picture: the answer does not parse", fill=(0, 0, 0), font=font)
        said = textwrap.wrap(caption, 42)[:5]                 # the caption's lines in black, the written layout's in blue
        lines = [(ln, (0, 0, 0)) for ln in said] + [(ln[:42], (40, 40, 160)) for ln in str(written).splitlines()[:6]]
        for k, (ln, colour) in enumerate(lines):
            draw.text((x + 4, y + side + 2 + 12 * k), ln, fill=colour, font=font)
    sheet.save(path, quality=92)


def _run_draw(a: dict, log) -> dict:
    if not a["answers"]:
        raise ValueError("draw needs answers=<a finished writer cell's folder on the toy bed's captions>")
    if not a["judge"]:
        raise ValueError("draw needs judge=<the judge's folder>")
    if a["bed"] != "F":
        raise ValueError("written layouts are drawn on bed F")
    res, outputs = _read_answers(a["answers"])
    if res.get("captions") != "bed":
        raise ValueError(f"{a['answers']} is not a writer cell on the toy bed's captions")
    prompts = layout.bed_prompts(int(res["args"].get("seed", 0)))
    index = {p["id"]: k for k, p in enumerate(prompts)}
    records = []
    for o in outputs:
        if o["id"] not in index:
            continue
        k = index[o["id"]]
        r = layout.caption_reads(prompts[k], k, o["answer"])
        r["text"] = prompts[k]["text"]
        r["answer"] = o["answer"]
        r["kinds_handed"] = layout.score(prompts[k], layout.requests_reading(r["requests"]), "cell")["kinds"] \
            if r["parsed"] else ["broken format"]
        records.append(r)
    drawn = [r for r in records if r["parsed"]]
    log(f"DRAW: {len(records)} captions of {a['answers']}; {len(drawn)} with a picture; engines of seed {a['seed']}")
    swap = parse_swap(a["swap"])
    if swap and a["row"] != "R1":
        raise ValueError(f"the variant draw draws the row R1, so swap goes with row=R1, not row={a['row']}")
    folders = judge_folders(a["judge"])
    more: Dict[str, tuple] = {}                  # a further judge's name: (its reads of the pictures, of the unexchanged)
    t0 = time.time()
    pix = pix_u = None
    exact = []
    if drawn:
        cells = torch.tensor([[-1 if c[0] is None else c[0] for c in layout.request_cells(r["requests"])] for r in drawn])
        colors = torch.tensor([[-1 if c[1] is None else c[1] for c in layout.request_cells(r["requests"])] for r in drawn])
        parts = Parts(a)
        if swap:
            answer, answer_u, _, described = swap_rows("R1", parts, swap)
        else:
            answer, _ = build_row("R1", parts)
        noise = caption_noise(drawn)
        z = C.sample(answer, cells, colors, 0, int(a["sample_steps"]), a["device"], int(a["chunk"]), noise=noise)
        pix = made_pixels(z)
        judge = SJ.load_judge(folders[0], a["device"])
        pt, pc = SJ.predict(judge, S.to_model(pix))
        for f in folders[1:]:                    # a further judge reads the same pictures
            jm = SJ.load_judge(f, a["device"])
            more[judge_name(f)] = SJ.predict(jm, S.to_model(pix)), None
        if swap:
            z_u = C.sample(answer_u, cells, colors, 0, int(a["sample_steps"]), a["device"], int(a["chunk"]), noise=noise)
            pix_u = made_pixels(z_u)
            pt_u, pc_u = SJ.predict(judge, S.to_model(pix_u))
            exact.append(exchange_exact(pix, pix_u, cells, sorted(swap), S.to_pixels(z), S.to_pixels(z_u)))
            for f in folders[1:]:
                jm = SJ.load_judge(f, a["device"])
                more[judge_name(f)] = more[judge_name(f)][0], SJ.predict(jm, S.to_model(pix_u))
        engines = described if swap else parts.used
        bad = [u["tag"] for u in parts.used if not u["descended"]]
    elif swap:                                   # nothing drawn: the pairs are still checked; nothing to compare
        parts = Parts(a)
        _, _, _, engines = swap_rows("R1", parts, swap)
        bad = [u["tag"] for u in parts.used if not u["descended"]]
    else:
        engines, bad = [], []
    sample_s = time.time() - t0

    def judged_of(p_t, p_c, j):
        return [[None, None] if int(p_t[j, c]) == SR.NOTHING else [int(p_t[j, c]), int(p_c[j, c])] for c in range(9)]

    j = 0
    by_id = {}
    cp_by: Dict[str, list] = {name: [] for name in more}       # a further judge's C and P of every caption
    cp_unswapped_by: Dict[str, list] = {name: [] for name in more}
    for r in records:
        prompt = prompts[index[r["id"]]]
        if r["parsed"]:
            judged = judged_of(pt, pc, j)
            judged_u = judged_of(pt_u, pc_u, j) if swap else None
            by_id[r["id"]] = j
        else:
            judged = judged_u = None
        r["judged"] = judged
        r.update(layout.picture_reads(prompt, r, judged))
        if swap:
            r["judged_unswapped"] = judged_u
            ru = layout.picture_reads(prompt, r, judged_u)
            r["C_unswapped"], r["P_unswapped"] = ru["C"], ru["P"]
        if more:
            r["judged_by"] = {name: judged_of(*m_[0], j) if r["parsed"] else None for name, m_ in more.items()}
            for name in more:
                cp_by[name].append(layout.picture_reads(prompt, r, r["judged_by"][name]))
            if swap:
                r["judged_unswapped_by"] = {name: judged_of(*m_[1], j) if r["parsed"] else None
                                            for name, m_ in more.items()}
                for name in more:
                    cp_unswapped_by[name].append(layout.picture_reads(prompt, r, r["judged_unswapped_by"][name]))
        if r["parsed"]:
            j += 1
    tables = drawn_tables(records)
    first = [r for r in records if r["family"] == "combined"][:int(a["sheet_n"])]
    if int(a["sheet"]) and first:
        write_caption_sheet(os.path.join(a["out"], "sheet.jpg"),
                            [(pix[by_id[r["id"]]] if r["id"] in by_id else None, r["text"], r["answer"]) for r in first])
    out = {"role": a["role"], "answers": os.path.basename(os.path.normpath(a["answers"])),
           "writer": res.get("writer"), "example_set": res.get("example_set"), "captions": len(records),
           "drawn": len(drawn), "records": records, "tables": tables, "engines": engines, "not_descended": bad,
           "inconclusive": bool(bad), "sample_seconds": sample_s,
           "pictures_sha256": pictures_sha256(pix) if pix is not None else None}
    if swap:
        ex_read = exact_summary(exact) if exact else dict(exact_summary([]), holds=None, real=None,
                                                          largest_outside=None, largest_outside_unclamped=None)
        out.update(swap={str(k): v for k, v in sorted(swap.items())}, exact=ex_read,
                   tables_unswapped=drawn_tables([dict(r, C=r["C_unswapped"], P=r["P_unswapped"]) for r in records]),
                   pictures_sha256_unswapped=pictures_sha256(pix_u) if pix_u is not None else None)
        log(f"exchange exact: {out['exact']['holds']} (largest difference outside {out['exact']['largest_outside']}, "
            f"{out['exact']['cells_compared']} cells compared; exchanged cells that differ {out['exact']['inside_share']})")
    if more:
        out.update(judges=[judge_name(f) for f in folders],
                   tables_by={name: drawn_tables([dict(r, **cp) for r, cp in zip(records, cp_by[name])]) for name in more})
        if swap:
            out["tables_unswapped_by"] = {name: drawn_tables([dict(r, **cp) for r, cp in zip(records, cp_unswapped_by[name])])
                                          for name in more}
    return out


def _stored(folder: str) -> Optional[dict]:
    p = os.path.join(folder, "result.json")
    if not folder or not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _run_parity(a: dict, log) -> dict:
    """Part of the earlier run read again: the collective of the fetched engines on that run's own lists at n = 4 (its
    start noise), its cell rate beside the stored one; the judge's stored ceiling and the real row's scene right at
    nine things; the stored wrong-address curves."""
    stored = _stored(a["stored"])
    if stored is None:
        raise ValueError("parity needs stored=<the earlier run's row folder (its result.json)>")
    parts = Parts(a)
    judge = SJ.load_judge(a["judge"], a["device"])
    answer, _ = build_row("R1", parts)
    L = lists_for(dict(a, ns=str(PARITY_N), lists=""))[PARITY_N]
    pix = made_pixels(_sample_lists(answer, L, a))
    rd = judge_reads(judge, pix, L["cells"], L["colors"])
    old = (stored.get("by_n") or {}).get(str(PARITY_N)) or {}
    oc = old.get("cell_rate") or {}
    diff = None if oc.get("mean") is None else rd["cell_rate"]["mean"] - oc["mean"]
    sha = pictures_sha256(pix)
    old_sha = old.get("pictures_sha256")
    parity = {"n": PARITY_N, "lists": int(L["cells"].shape[0]), "cell_rate": rd["cell_rate"], "stored_cell_rate": oc,
              "difference": diff, "bar": None if oc.get("se") is None else 2 * oc["se"],
              "holds": None if diff is None or oc.get("se") is None else abs(diff) <= 2 * oc["se"],
              "pictures_sha256": sha, "stored_pictures_sha256": old_sha,
              "pictures_equal": None if old_sha is None else sha == old_sha,
              "stored_args": {k: stored.get("args", {}).get(k) for k in ("seed", "bed", "count", "sample_steps", "device")},
              "stored_version": stored.get("version")}
    jr = _stored(a["judge"])
    ceil = ((jr or {}).get("judge") or {}).get("ceiling")
    real = _stored(a["real_row"])
    r9 = ((real or {}).get("by_n") or {}).get("9", {}).get("scene_right")
    k = {"ceiling": ceil, "real_scene_right_n9": r9,
         "holds_rates": None if not ceil else bool(ceil["thing_rate"]["mean"] >= LIMITS["a_rate"]
                                                    and ceil["color_rate"]["mean"] >= LIMITS["a_rate"]),
         "holds_real": None if not r9 else bool(r9["mean"] >= LIMITS["a_real_n9"])}
    k["holds"] = None if k["holds_rates"] is None or k["holds_real"] is None else k["holds_rates"] and k["holds_real"]
    curve = []
    for folder in [x.strip() for x in str(a["wrong_rows"]).split(",") if x.strip()]:
        w = _stored(folder)
        for x in (w or {}).get("wrong", []):
            curve.append({"seed": (w.get("args") or {}).get("seed"), "n": x["n"], "e": x["e"],
                          "scene_right": x["scene_right"]})
    log(f"parity: cell rate {rd['cell_rate']['mean']:.4f} against the stored {oc.get('mean')}; holds {parity['holds']}; "
        f"pictures equal {parity['pictures_equal']}; the judge's stored control holds {k['holds']}")
    return {"parity": parity, "judge_control": k, "wrong_curve": curve, "engines": parts.used,
            "inconclusive": any(not u["descended"] for u in parts.used)}


RUNNERS = {"parts": _run_parts, "judge": _run_judge, "timing": _run_timing, "compose": _run_compose,
           "controls": _run_controls, "exact": _run_exact, "leak": _run_leak, "wrong": _run_wrong,
           "draw": _run_draw, "parity": _run_parity}


# ------------------------------------------------------------------ the result
def _f(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _text(result: dict) -> str:
    v = result["variant"]
    lines = [f"# toy_collective: {v}" + (f", row {result.get('row')}" if result.get("row") else ""), ""]
    if v == "compose":
        lines += ["| n | cell rate | se | scene right | scene right with colors | color rate | ghosts |", "|---|---|---|---|---|---|---|"]
        for n, r in result["by_n"].items():
            lines.append(f"| {n} | {_f(r['cell_rate']['mean'])} | {_f(r['cell_rate']['se'])} | {_f(r['scene_right']['mean'])} | "
                         f"{_f(r['scene_right_colors']['mean'])} | {_f(r['color_rate']['mean'])} | {_f(r['ghosts']['mean'])} |")
        lines.append(f"\ninconclusive: {_f(result['inconclusive'])}")
    elif v == "parts":
        lines += ["| engine | newborn read | last read | descended | seconds |", "|---|---|---|---|---|"]
        lines += [f"| {e['tag']} | {_f(e['newborn_read'])} | {_f(e['last_read'])} | {_f(e['descended'])} | "
                  f"{_f(e['train_seconds'])} |" for e in result["engines"]]
    elif v == "controls":
        lines += [f"- ({k}): holds {_f(c.get('holds', c.get('holds_true')))}" for k, c in result["controls"].items()]
    elif v == "draw":
        t = result["tables"]
        lines += [f"writer cell {result['answers']} (role {result['role'] or '-'}); {result['captions']} captions, "
                  f"{result['drawn']} drawn; engines of seed {result['args']['seed']}", "",
                  "P picture right, W written right, C drawn as written, H handed right (the requests after the "
                  "conversion); rates over the captions, se = sqrt(p (1 - p) / n)", "",
                  "| family | n | P | se | W | se | C | se | H | se | mean W x C | mean H x C | W = 1, H = 0 | "
                  "per-instance error |", "|" + "---|" * 14]
        for name, b in list(t["families"].items()) + [("all six", t.get("all"))]:
            if not b:
                continue
            lines.append(f"| {name} | {b['n']} | " + " | ".join(f"{_f(b[k]['mean'])} | {_f(b[k]['se'])}"
                                                               for k in ("P", "W", "C", "H"))
                         + f" | {_f(b['WC'])} | {_f(b['HC'])} | {_f(b['handoff_loss']['rate'])} | "
                           f"{_f(b['per_instance_error']['rate'])} |")
        kinds = (t.get("all") or {}).get("handoff_loss", {}).get("kinds", {})
        lines += ["", "the handoff loss by kind (captions): " + (", ".join(f"{k} {v}" for k, v in kinds.items()) or "none"),
                  "", "| asked instances | P | se | n |", "|---|---|---|---|"]
        lines += [f"| {k} | {_f(x['mean'])} | {_f(x['se'])} | {x['n']} |" for k, x in t["P_by_asked"].items()]
        lines += ["", "| W, C | captions | P |", "|---|---|---|"]
        lines += [f"| {k} | {x['captions']} | {_f(x['P']['mean'])} |" for k, x in t["W_by_C"].items()]
        lines.append(f"\ninconclusive: {_f(result['inconclusive'])}; sampling {result['sample_seconds']:.1f} s")
    elif v == "parity":
        p, k = result["parity"], result["judge_control"]
        lines += [f"- the collective on the earlier run's lists at n = {p['n']}: cell rate {_f(p['cell_rate']['mean'])} "
                  f"against the stored {_f(p['stored_cell_rate'].get('mean'))} (difference {_f(p['difference'])}, bar "
                  f"{_f(p['bar'])}): holds {_f(p['holds'])}",
                  f"- the pictures equal the stored ones bit for bit: "
                  + ("not stored" if p["pictures_equal"] is None else _f(p["pictures_equal"])),
                  f"- the judge's stored control: rates holds {_f(k['holds_rates'])}, real scenes at nine things "
                  f"{_f((k['real_scene_right_n9'] or {}).get('mean'))} holds {_f(k['holds_real'])}",
                  f"- the stored wrong-address curve: {len(result['wrong_curve'])} points"]
    else:
        body = {k: result[k] for k in ("judge", "timing", "exact", "leak") if k in result}
        if "wrong" in result:
            body["wrong"] = [{k: x[k] for k in ("n", "e", "scene_right", "corruptions")} for x in result["wrong"]]
        lines.append("```\n" + json.dumps(body, indent=1, default=str)[:20000] + "\n```")
    if result.get("swap"):
        x = result["exact"]
        lines.append(f"\nexchanged: {', '.join(f'{k} -> {v}' for k, v in result['swap'].items())}; outside the exchanged "
                     f"cells the largest difference is {x['largest_outside']} over {x['cells_compared']} cells (exact: "
                     f"{_f(x['holds'])}); exchanged cells that differ: {_f(x['inside_share'])}")
    lines.append(f"\nseconds: {result['seconds']:.1f}; package {result['version']}")
    return "\n".join(lines) + "\n"


def main(argv=(), **overrides) -> dict:
    a = parse(DEFAULTS, argv, **overrides)
    if not a["out"]:
        raise ValueError("out= is required")
    if a["variant"] not in VARIANTS:
        raise ValueError(f"unknown variant {a['variant']!r}; known: {list(VARIANTS)}")
    if a["row"] not in ROWS:
        raise ValueError(f"the rows are {ROWS}, not {a['row']!r}")
    if len(judge_folders(a["judge"])) > 1 and a["variant"] not in SWAP_VARIANTS:
        raise ValueError(f"several judges apply to the variants {', '.join(SWAP_VARIANTS)}, not {a['variant']}")
    if parse_swap(a["swap"]):
        if a["variant"] not in SWAP_VARIANTS:
            raise ValueError(f"swap applies to the variants {', '.join(SWAP_VARIANTS)}, not {a['variant']}")
        if a["row"] not in SWAP_ROWS:
            raise ValueError(f"swap applies to the addressed collective (the rows {', '.join(SWAP_ROWS)}), "
                             f"not the row {a['row']}")
    os.makedirs(a["out"], exist_ok=True)
    log_path = os.path.join(a["out"], "log.txt")

    def log(line: str) -> None:
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")

    torch.set_num_threads(int(a["threads"]))
    t0 = time.time()
    log(f"TOY COLLECTIVE {a['variant']}: bed {a['bed']}, seed {a['seed']}, device {a['device']}, package {__version__}")
    with determinism.mode(a["det"]) as det:
        result = RUNNERS[a["variant"]](a, log)
        det.record(result)
    result.update({"experiment": EXPERIMENT, "variant": a["variant"], "args": {k: a[k] for k in sorted(a)},
                   "version": __version__, "torch": torch.__version__, "seconds": time.time() - t0})
    result.setdefault("row", a["row"] if a["variant"] in ("compose", "controls") else None)
    with open(os.path.join(a["out"], "result.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=1, default=str)
    text = _text(result)
    with open(os.path.join(a["out"], "result.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    log(text.rstrip())
    log("done")
    return result


def summarize(root: str) -> str:
    """A markdown table of the compose results under ``root``: the row, bed, seed and every n's cell rate and scene
    right."""
    rows = ["| job | row | bed | seed | n | cell rate | scene right | inconclusive |", "|---|---|---|---|---|---|---|---|"]
    for dirpath, _, files in sorted(os.walk(root)):
        if "result.json" not in files:
            continue
        with open(os.path.join(dirpath, "result.json"), encoding="utf-8") as fh:
            r = json.load(fh)
        if r.get("experiment") != EXPERIMENT or r.get("variant") != "compose":
            continue
        for n, x in r["by_n"].items():
            rows.append(f"| {os.path.relpath(dirpath, root)} | {r['row']} | {r['args']['bed']} | {r['args']['seed']} | {n} | "
                        f"{_f(x['cell_rate']['mean'])} | {_f(x['scene_right']['mean'])} | {_f(r['inconclusive'])} |")
    return "\n".join(rows) + "\n"


def summarize_drawn(root: str) -> str:
    """A markdown table of the draw results under ``root``: the writer cell, its role, the seed, and P, W, C, H over the
    six families and on combined."""
    rows = ["| job | writer cell | role | seed | P | W | C | H | P combined | W combined | inconclusive |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for dirpath, _, files in sorted(os.walk(root)):
        if "result.json" not in files:
            continue
        with open(os.path.join(dirpath, "result.json"), encoding="utf-8") as fh:
            r = json.load(fh)
        if r.get("experiment") != EXPERIMENT or r.get("variant") != "draw":
            continue
        t = r["tables"]
        al, cb = t.get("all") or {}, t["families"].get("combined") or {}
        get = lambda b, k: (b.get(k) or {}).get("mean")
        rows.append(f"| {os.path.relpath(dirpath, root)} | {r['answers']} | {r['role'] or '-'} | {r['args']['seed']} | "
                    + " | ".join(_f(get(al, k)) for k in ("P", "W", "C", "H"))
                    + f" | {_f(get(cb, 'P'))} | {_f(get(cb, 'W'))} | {_f(r['inconclusive'])} |")
    return "\n".join(rows) + "\n"


if __name__ == "__main__" and not in_notebook():
    main(sys.argv[1:])
