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

THE ROWS (``row``): R1 (nine addressed thing parts and the addressed base, the gate), R1r (the same in the rotary read),
R2 (nine whole-picture thing parts and the empty-scene engine, the gate), R3 (the same parts, the sum with the empty
background), R4 (the whole-picture parts, the sum with the unconditional background), R5a / R5ar (the whole model
trained on mix scenes at the engine's shape, embedding / rotary read; ``model_steps``), R5b (the same at
``model_dim``, ``model_heads``, ``model_depth``), R6 (the whole model trained on one-thing scenes), REAL (real scenes of
the lists, rendered by the bed). The parts are found under ``parts`` (searched for folders named by ``engine_tag`` with
``part_steps``, the row's read form and ``part_cells``); the judge is the folder ``judge``.

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

from .. import __version__, determinism, frame
from .. import collective as C
from .. import scene_judge as SJ
from .. import scene_reads as SR
from .. import scenes as S
from ..config import in_notebook, parse

EXPERIMENT = "toy_collective"
VARIANTS = ("parts", "judge", "timing", "compose", "controls", "exact", "leak", "wrong")
ROWS = ("R1", "R1r", "R2", "R3", "R4", "R5a", "R5ar", "R5b", "R6", "REAL")
DEFAULTS = dict(variant="parts", kind="addressed_thing", classes="0-8", bed="F", seed=0, steps=3000, dim=128, heads=2,
                depth=3, read="embedding", batch=128, lr=1e-3, warmup=100, every=250, heldout=2048, save_every=500,
                cells="", from_step=20,
                row="R1", rows="R1,R2,R3", parts="", judge="", ns="1,2,3,4,6,9", count=500, chunk=128,
                sample_steps=50, part_steps=3000, part_cells="", model_steps=30000, model_dim=128, model_heads=2,
                model_depth=3, lists="", twin_seed=1, sheet=1, judge_steps=2000, judge_batch=256,
                e="0.05,0.10,0.20", wrong_ns="2,4,6", leak_ns="2,4,9", leak_ts="0.15,0.45,0.85", leak_scenes=256,
                exact_lists=64, device="cpu", det="on", threads=4, out="")
LIST_SEED, NOISE_SEED = 5000, 7000
CENTER_SEED, OTHERS_SEED = 5100, 5101
CENTER = 4
LEAK_SEED, WRONG_SEED = 8000, 9000
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
        self.index = index_engines(a["parts"])
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


def judge_reads(judge, pixels: torch.Tensor, cells, colors) -> dict:
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
    judge = SJ.load_judge(a["judge"], a["device"]) if a["judge"] else None
    if judge is None:
        raise ValueError("compose needs judge=<the judge's folder>")
    answer, desc = (None, "REAL: real scenes of the lists") if row == "REAL" else build_row(row, parts)
    by_n, sheets = {}, {}
    for n, L in lists_for(a).items():
        t0 = time.time()
        real = real_pixels(L["cells"], L["colors"], a["bed"], L["noise_seed"])
        pix = real if row == "REAL" else made_pixels(_sample_lists(answer, L, a))
        rd = judge_reads(judge, pix, L["cells"], L["colors"])
        rd.update(picture_reads(pix, L["cells"], L["colors"], a["bed"], real))
        rd["seconds"] = time.time() - t0
        by_n[str(n)] = rd
        if n in (4, 9) or a["lists"]:
            sheets[n] = pix[:8]
        log(f"{row} n={n}: cell rate {rd['cell_rate']['mean']:.3f}, scene right {rd['scene_right']['mean']:.3f}, "
            f"ghosts {rd['ghosts']['mean'] if rd['ghosts']['mean'] is not None else '-'}; {rd['seconds']:.0f} s")
    if int(a["sheet"]):
        write_sheet(os.path.join(a["out"], "sheet.jpg"), [sheets[k] for k in sorted(sheets)])
    bad = [u["tag"] for u in parts.used if not u["descended"]]
    return {"row": row, "description": desc, "by_n": by_n, "engines": parts.used, "not_descended": bad,
            "inconclusive": bool(bad), "lists": a["lists"] or "request_lists", "count": int(a["count"])}


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


RUNNERS = {"parts": _run_parts, "judge": _run_judge, "timing": _run_timing, "compose": _run_compose,
           "controls": _run_controls, "exact": _run_exact, "leak": _run_leak, "wrong": _run_wrong}


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
    else:
        body = {k: result[k] for k in ("judge", "timing", "exact", "leak") if k in result}
        if "wrong" in result:
            body["wrong"] = [{k: x[k] for k in ("n", "e", "scene_right", "corruptions")} for x in result["wrong"]]
        lines.append("```\n" + json.dumps(body, indent=1, default=str)[:20000] + "\n```")
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


if __name__ == "__main__" and not in_notebook():
    main(sys.argv[1:])
