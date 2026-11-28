"""The addressed collective: member engines that receive the whole written layout and find what is addressed to them by
address pairs inside attention, served by the amplitude gate over the same addresses (``members``, ``member_train``).

    python -m geolip.modular_diffusion.run addressed_collective variant=parts kind=T2 classes=0-8,base bed=F seed=0 out=<f>
    from geolip.modular_diffusion.experiments import addressed_collective; addressed_collective.main(variant="compose", ...)

| variant  | what it does                                                                                           |
|----------|--------------------------------------------------------------------------------------------------------|
| parts    | ``member_train.train_members`` for the engines of ``kind`` and ``classes`` ("0-8,base" or any subset), |
|          | ``bed``, ``seed``, ``steps``, ``exclude`` (a class, or empty), ``held`` ("" or "plus4": class g never  |
|          | asks class (g + 4) mod 9), ``p_point``, ``snapshots``, ``noise`` ("stream" or "device": where a step's |
|          | noise and t are drawn, ``member_train.stack_step_rows``), into ``out``                                 |
| timing   | the seconds a step of a stack of G engines for each G of ``stacks``, ``steps`` each with the first     |
|          | ``from_step`` dropped, on the job's device, with ``noise``; with ``phases=1`` also the median seconds  |
|          | of each phase of a step (a table in result.md); nothing is kept but the result                         |
| compose  | the collective of ``kind`` found under ``parts`` sampled in ``sample_steps`` Euler steps and judged by |
|          | every folder of ``judge``: (a) the stored lists (``scenes.request_lists(n, count)`` seeded 5000 + n,    |
|          | start noise seeded 7000 + n, the lists and noise of ``toy_collective``) per n of ``ns``: things right, |
|          | whole scenes right, colors right; (b) for the told "all" kinds, lists that point, with distinct colors |
|          | (``members.pointer_lists(n, count, 6000 + n, held, distinct=True)`` and the same with                  |
|          | ``only_held=True``; start noise seeded 7500 + n and 7600 + n; n of ``pointer_ns``): on the requests    |
|          | that state their color, thing right and color right; on the pointing request, its thing right and its |
|          | color right (the judged color of its cell is the asked request's); beside it THE TWO ADDRESS-BLIND     |
|          | REFERENCES of the very lists used; a sheet                                                             |
| address  | the address reads at the noise levels 0 and ``ts`` (FIND, SERVE, the two together with their two       |
|          | controls, a missing and an unheld address, the tolerance control)                                       |
| exchange | ``members.exchange_read`` between the collective and the same collective with the engine of            |
|          | ``swap_class`` taken from ``twin_seed`` (one step, ``k`` scenes at each of ``ts``), and over a chain:  |
|          | the first ``exact_lists`` stored lists at n = 4 sampled twice from the same start noise (seed 7004)    |
|          | with the two collectives: the largest absolute difference of the final pictures outside the exchanged  |
|          | class's cells, before and after the clamp to 0..1                                                      |

THE ENGINES are found under ``parts`` (one folder or several, separated by commas) by ``member_train.member_tag`` of
the kind, the class, the bed, the seed, ``part_steps``, the shape, ``exclude`` and ``held``, and read with their
averaged weights; the base is the class "base".

THE TWO ADDRESS-BLIND REFERENCES (compose, per n and per job, each the mean over the lists in which a request points;
written beside each other, with the number of lists and the histogram of k):
    reference_all_stated  a guesser that sees every other request's stated color and no class at all names the most
                          frequent color among them; it is right with probability (1 / the number of colors tied for
                          the most) when the asked color is among them, else 0. With distinct colors it is 1 / (n - 1).
                          The null of a kind that can see no class (pairs not turned).
    reference_eligible    a guesser that reads the other requests' classes but not which one is asked names one of the
                          asking cell's ELIGIBLE DONORS (``members.eligible_donors``: a request whose class stands once,
                          differs from the asking class and is outside the held pairs, or inside them for the held job);
                          given the asking cell the asked one is uniform over them, so with distinct colors it is right
                          with probability 1 / k, k the number of eligible donors. The best any engine can do WITHOUT
                          its own pointer: the stricter bar (never below the other).

THE ADDRESS READS (``address``). The noise levels are t = 0 (pure noise, no trace of the picture) and ``ts``; each is
reported apart. At a level t a picture x in model space and its noise give z = (1 - t) noise + t x.
    FIND         for every thing engine, ``k`` rows of its own training law with p_point = 1 (``members.member_batch``,
                 a generator seeded 30000 + 100 x the class), each row's own pointing request and two candidates:
                 the asked class and the first other class (in cell order) that stands once, differs from the
                 engine's class, states its color and differs in COLOR from the asked one (a row whose candidates all
                 share the asked color is skipped and counted, as is a row without a second candidate);
                 ``members.find_read`` at every block (``vary`` "content" for the identity in the content, else
                 "query"): per block and head the mean softmax weight on the asked candidate under each ask and the
                 mean change (the weight on a candidate when it is asked minus when the other is), the mean change of
                 each pair group's score; and the share of rows in which the engine's one-step picture of the cell,
                 x_hat = z + (1 - t) v, is nearer in mean color over the cell (pixels) to the asked candidate's color
                 than to the other's, under each ask. ``k`` is the number of rows DRAWN for each engine; the rows
                 used are fewer by the skips (per engine and in total: drawn, used, each skip reason). THE RECEIPTS,
                 since a signed mean of zero does not show that nothing moved and two separate rates do not show how
                 often one row switched the right way: per block the largest |weight under ask a - weight under ask
                 b| over rows, heads and both candidates, the same of the pair-group scores, and the rows whose
                 weights and raw answers are exactly equal under the two asks (of ``rows``); per level the largest
                 absolute change of the raw answers (block 0's pass) and the rows in which the picture is nearer the
                 asked color under BOTH asks, under ask a only, under ask b only, under neither (of ``rows``)
    the scenes   of the other reads: ``k`` lists of ``members.pointer_lists(4, k, 31000)`` (no pointer for the told
                 "own" kinds), rendered from a generator seeded 31001, the noise of level t from one seeded 31002 +
                 round(1000 t); on each scene the edited cell is the first request that neither points nor is asked,
                 and the new class the first member class that does not stand in the scene and is not asked (a scene
                 without one is skipped and counted)
    SERVE        ``members.serve_read`` with the parts' raw answers: the old member's weight on the cell before, the
                 new member's after, whether the denominator is the same and whether the reverse edit restores every
                 number
    TOGETHER     ``members.address_swap`` (the gate's weights on the cell before and after; the looks of the old and
                 the new class's engines per block and head; every other member's relative change on its own tokens),
                 with ``turn=False`` (whether the gate's weights are identical, and the looks) and ``gate="neutral"``
                 (the relative change of the old and the new member's raw answers on the cell)
    MISSING, UNHELD  ``members.address_missing`` and ``members.address_unheld``: the largest member weight and the
                 smallest base weight on the cell
    TOLERANCE    for every member, the relative change of its contribution on the tokens it owns (weight above 1/2)
                 when the noise of ONE EMPTY cell (the first) is redrawn and nothing else (a generator seeded 31500 +
                 round(1000 t)): the median and the 95th percentile over the scenes, per member and pooled

Output in ``out``: result.json, result.md, log.txt (its last line 'done'); compose also sheet.jpg (8 scenes at n = 4
and at n = 9 of the stored lists, 8 scenes of the pointing lists at n = 4).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import torch

from .. import __version__, determinism
from .. import collective as C
from .. import member_train as MT
from .. import members as MB
from .. import scene_judge as SJ
from .. import scene_reads as SR
from .. import scenes as S
from ..config import in_notebook, parse
from ..spectral import patchify
from .toy_collective import (exchange_exact, index_engines, judge_folders, judge_name, judge_reads, made_pixels,
                             picture_reads, real_pixels, weights_sha256, write_sheet)

EXPERIMENT = "addressed_collective"
VARIANTS = ("parts", "timing", "compose", "address", "exchange")
DEFAULTS = dict(variant="parts", kind="T2", classes="0-8,base", bed="F", seed=0, steps=3000, dim=128, heads=2,
                depth=3, batch=128, lr=1e-3, warmup=100, every=250, heldout=2048, save_every=500, p_point=0.5,
                exclude="", held="", snapshots="", stacks="1,5,10", from_step=2, parts="", judge="", part_steps=3000,
                ns="1,2,3,4,6,9", pointer_ns="3,4,6,9", count=500, chunk=128, sample_steps=50, sheet=1,
                ts="0.1,0.3", k=64, twin_seed=1, swap_class=-1, exact_lists=64, device="cpu", det="on", threads=4,
                noise="stream", phases=1, out="")
LIST_SEED, NOISE_SEED = 5000, 7000
POINTER_SEED, POINTER_NOISE, POINTER_NOISE_HELD = 6000, 7500, 7600
FIND_SEED, SCENE_SEED, TOLERANCE_SEED = 30000, 31000, 31500
CHAIN_N = 4


# ------------------------------------------------------------------ arguments
def parse_classes(text: str) -> List[Optional[int]]:
    """'0-8,base' -> [0, ..., 8, None]; items are classes, ranges a-b, or 'base'."""
    out: List[Optional[int]] = []
    for item in [x.strip() for x in str(text or "").split(",") if x.strip()]:
        if item == "base":
            out.append(None)
        elif "-" in item:
            lo, hi = item.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(item))
    if not out or len(set(out)) != len(out):
        raise ValueError(f"classes: {text!r} names no class or one twice")
    return out


def held_pairs(text: str) -> tuple:
    """'' -> (); 'plus4' -> every (g, (g + 4) mod 9)."""
    text = str(text or "").strip()
    if not text:
        return ()
    if text == "plus4":
        return tuple((g, (g + 4) % 9) for g in range(9))
    raise ValueError(f"held is '' or 'plus4', not {text!r}")


def _exclude(a: dict) -> Optional[int]:
    return None if str(a["exclude"]).strip() == "" else int(a["exclude"])


def _floats(text: str) -> List[float]:
    return [float(x) for x in str(text).split(",") if x.strip()]


def _ints(text: str) -> List[int]:
    return [int(x) for x in str(text).split(",") if x.strip()]


def _variant(a: dict) -> MB.Variant:
    if a["kind"] not in MB.VARIANTS:
        raise ValueError(f"kind is one of {list(MB.VARIANTS)}, not {a['kind']!r}")
    return MB.VARIANTS[a["kind"]]


# ------------------------------------------------------------------ the collective from its folders
def load_collective(a: dict, seed: Optional[int] = None, twin: Optional[Dict[Optional[int], int]] = None):
    """-> (the collective, the engines used). ``twin`` {class: seed}: that class's engine of another seed."""
    variant = _variant(a)
    index: Dict[str, str] = {}
    for root in str(a["parts"] or "").split(","):
        for k, v in index_engines(root.strip()).items():
            index.setdefault(k, v)
    seed = int(a["seed"]) if seed is None else int(seed)
    twin = twin or {}
    engines, used = {}, []
    for cls in parse_classes(a["classes"]):
        s = int(twin.get(cls, seed))
        tag = MT.member_tag(variant, cls, a["bed"], s, int(a["part_steps"]), int(a["dim"]), int(a["heads"]),
                            int(a["depth"]), _exclude(a), held_pairs(a["held"]), str(a["noise"]))
        if tag not in index:
            raise FileNotFoundError(f"no finished engine {tag} under {a['parts']}")
        path = index[tag]
        with open(os.path.join(path, "train.json"), encoding="utf-8") as fh:
            tj = json.load(fh)
        engines[cls] = MB.load_member(path, average=True, device=a["device"])
        used.append({"tag": tag, "cls": cls, "seed": s, "descended": bool(tj.get("descended")),
                     "newborn_read": tj.get("newborn_read"), "last_read": tj.get("last_read"),
                     "trained": tj.get("trained"), "weights_sha256": weights_sha256(path)})
    things = {c: e for c, e in engines.items() if c is not None}
    return MB.MemberCollective(things, engines.get(None), variant), used


# ------------------------------------------------------------------ compose
def eligible_reference(pl: MB.PointerLists) -> dict:
    """The eligible-donor reference of the lists that point (``reference_eligible`` of the module's text): {"mean",
    "lists", "k_histogram"} (k as text -> the number of lists)."""
    ks = [int(k) for k, p in zip(pl.eligible.tolist(), pl.pointed.tolist()) if p]
    if any(k < 1 for k in ks):
        raise RuntimeError("a list that points has an asking cell without an eligible donor")
    hist: Dict[str, int] = {}
    for k in sorted(ks):
        hist[str(k)] = hist.get(str(k), 0) + 1
    return {"mean": sum(1.0 / k for k in ks) / len(ks) if ks else None, "lists": len(ks), "k_histogram": hist}


def blind_reference(cells: torch.Tensor, colors: torch.Tensor, points: torch.Tensor) -> dict:
    """The all-stated address-blind reference of the lists that point (``reference_all_stated`` of the module's
    text): {"mean", "lists"}."""
    vals = []
    for i in range(cells.shape[0]):
        ps = [c for c in range(9) if int(points[i, c]) >= 0]
        if not ps:
            continue
        a = ps[0]
        others = [int(colors[i, c]) for c in range(9) if int(cells[i, c]) >= 0 and c != a]
        if not others:
            vals.append(0.0)
            continue
        counts = {k: others.count(k) for k in set(others)}
        top = max(counts.values())
        tied = [k for k, v in counts.items() if v == top]
        vals.append(1.0 / len(tied) if int(colors[i, a]) in tied else 0.0)
    return {"mean": sum(vals) / len(vals) if vals else None, "lists": len(vals)}


def pointing_reads(judge, pixels: torch.Tensor, cells, colors, points) -> dict:
    """The judge's reads of pictures of lists that point: on the stating requests, thing right and color right; on the
    pointing request, thing right, color right (the asked color) and both."""
    pt, pc = SJ.predict(judge, S.to_model(pixels))
    s = SR.score_cells(pt, pc, cells.to(pt.device), colors.to(pt.device))
    points = points.to(pt.device)
    stated = s["requested"] & (points < 0)
    ptg = s["requested"] & (points >= 0)
    return {"stated_thing_right": SR.rate(s["thing_right"][stated]),
            "stated_color_right": SR.rate(s["color_right"][stated]),
            "pointing_thing_right": SR.rate(s["thing_right"][ptg]),
            "pointing_color_right": SR.rate(s["color_right"][ptg]),
            "pointing_both_right": SR.rate((s["thing_right"] & s["color_right"])[ptg]),
            "lists": int(cells.shape[0]), "pointed": int(ptg.any(dim=1).sum())}


def _judges(a: dict):
    folders = judge_folders(a["judge"])
    if not folders:
        raise ValueError("compose needs judge=<the judge's folder>")
    return [(judge_name(f), SJ.load_judge(f, a["device"])) for f in folders]


def _run_compose(a: dict, log) -> dict:
    variant = _variant(a)
    col, used = load_collective(a)
    judges = _judges(a)
    held = held_pairs(a["held"])
    count, steps, chunk = int(a["count"]), int(a["sample_steps"]), int(a["chunk"])
    stored, pointing, sheets = {}, {}, []
    for n in _ints(a["ns"]):
        t0 = time.time()
        rl = S.request_lists(n, count, seed=LIST_SEED + n)
        z = MB.sample_lists(col, rl.cells, rl.colors, MB.no_points(rl.cells), NOISE_SEED + n, steps, a["device"], chunk)
        pix = made_pixels(z)
        real = real_pixels(rl.cells, rl.colors, a["bed"], NOISE_SEED + n)
        reads = {}
        for name, jm in judges:
            rd = judge_reads(jm, pix, rl.cells, rl.colors)
            rd.update(picture_reads(pix, rl.cells, rl.colors, a["bed"], real))
            reads[name] = rd
        first = reads[judges[0][0]]
        stored[str(n)] = dict(first, by_judge=reads, seconds=time.time() - t0)
        if n in (4, 9):
            sheets.append(pix[:8])
        log(f"stored lists n={n}: things right {first['cell_rate']['mean']:.3f}, scenes right "
            f"{first['scene_right']['mean']:.3f}, colors right {first['color_rate']['mean']}")
    if variant.told == "all":
        for n in _ints(a["pointer_ns"]):
            for only, noise_seed, label in ((False, POINTER_NOISE, "unheld_pairs"), (True, POINTER_NOISE_HELD,
                                                                                      "held_pairs")):
                if only and not held:
                    continue
                t0 = time.time()
                pl = MB.pointer_lists(n, count, POINTER_SEED + n, held, only_held=only, distinct=True)
                z = MB.sample_lists(col, pl.cells, pl.colors, pl.points, noise_seed + n, steps, a["device"], chunk)
                pix = made_pixels(z)
                reads = {name: pointing_reads(jm, pix, pl.cells, pl.colors, pl.points) for name, jm in judges}
                first = reads[judges[0][0]]
                blind = blind_reference(pl.cells, pl.colors, pl.points)
                elig = eligible_reference(pl)
                pointing.setdefault(label, {})[str(n)] = dict(first, by_judge=reads, reference_all_stated=blind,
                                                              reference_eligible=elig, distinct_colors=True,
                                                              seconds=time.time() - t0)
                if n == 4 and not only:
                    sheets.append(pix[:8])
                log(f"pointing lists ({label}) n={n}: pointing color right {first['pointing_color_right']['mean']} "
                    f"against a guess among every other stated color {blind['mean']} and a guess among the eligible "
                    f"requests {elig['mean']} ({elig['lists']} lists, k {elig['k_histogram']}); stated color right "
                    f"{first['stated_color_right']['mean']}")
    if int(a["sheet"]):
        write_sheet(os.path.join(a["out"], "sheet.jpg"), sheets)
    bad = [u["tag"] for u in used if not u["descended"]]
    return {"kind": variant.name, "stored": stored, "pointing": pointing, "engines": used, "judges":
            [n for n, _ in judges], "not_descended": bad, "inconclusive": bool(bad), "count": count}


# ------------------------------------------------------------------ the address reads
def _levels(a: dict) -> List[float]:
    out = [0.0]
    for t in _floats(a["ts"]):
        if t not in out:
            out.append(t)
    return out


def _noisy_at(x: torch.Tensor, t: float, seed: int) -> torch.Tensor:
    noise = torch.randn(tuple(x.shape), generator=torch.Generator().manual_seed(int(seed)))
    return (1 - t) * noise + t * x


def _mean_color(tokens: torch.Tensor) -> torch.Tensor:
    """[16, 48] model-space tokens of one cell -> the cell's mean color in pixels [3]."""
    return S.to_pixels(tokens.view(tokens.shape[0], S.CHANNELS, -1).mean(dim=(0, 2)))


def _second_candidate(cells_row, colors_row, points_row, cls: int, ask_a: int):
    """The first class (in cell order) that stands once, differs from cls and ask_a, states its color and differs in
    color from ask_a's request -> (class, why None)."""
    row = cells_row.tolist()
    color_a = int(colors_row[row.index(ask_a)])
    same_color = False
    for c in range(9):
        k = row[c]
        if k < 0 or k in (cls, ask_a) or row.count(k) != 1 or int(points_row[c]) >= 0:
            continue
        if int(colors_row[c]) == color_a:
            same_color = True
            continue
        return k, None
    return None, ("same_color" if same_color else "no_candidate")


def _find(a: dict, col: MB.MemberCollective, log) -> dict:
    variant = col.variant
    vary = "content" if variant.identity == "content" else "query"
    bed = S.ScenesBed(a["bed"], "cpu")
    dev = a["device"]
    depth = len(next(iter(col.things.values())).blocks)
    out = {"vary": vary, "k_rows_drawn_per_engine": int(a["k"]), "by_t": {}}
    for t in _levels(a):
        acc = {"rows": 0, "rows_drawn": 0, "skipped_same_color": 0, "skipped_no_candidate": 0,
               "skipped_no_pointer": 0, "nearer_asked_a": 0, "nearer_asked_b": 0, "both_nearer": 0, "only_a": 0,
               "only_b": 0, "neither": 0, "max_abs_answer_delta": 0.0, "per_engine": {}, "blocks": {}}
        sums = {}
        for cls, eng in col.things.items():
            gen = torch.Generator().manual_seed(FIND_SEED + 100 * cls)
            mb = MB.member_batch(variant, cls, bed, int(a["k"]), gen, 1.0, _exclude(a), held_pairs(a["held"]))
            z_all = _noisy_at(mb["x"], t, FIND_SEED + 100 * cls + 1 + int(round(1000 * t)))
            mine = {"drawn": int(mb["x"].shape[0]), "used": 0, "skipped_same_color": 0, "skipped_no_candidate": 0,
                    "skipped_no_pointer": 0}
            acc["per_engine"][str(cls)] = mine
            acc["rows_drawn"] += mine["drawn"]
            for i in range(mb["x"].shape[0]):
                own = [c for c in range(9) if int(mb["cells"][i, c]) == cls and int(mb["points"][i, c]) >= 0]
                if not own:
                    acc["skipped_no_pointer"] += 1
                    mine["skipped_no_pointer"] += 1
                    continue
                cell = own[0]
                ask_a = int(mb["points"][i, cell])
                ask_b, why = _second_candidate(mb["cells"][i], mb["colors"][i], mb["points"][i], cls, ask_a)
                if ask_b is None:
                    acc["skipped_" + why] += 1
                    mine["skipped_" + why] += 1
                    continue
                mine["used"] += 1
                visible = mb["visible"][i:i + 1] if mb["visible"] is not None else None
                z = z_all[i:i + 1].to(dev)
                tt = torch.full((1,), float(t), device=dev)
                row = mb["cells"][i].tolist()
                ca = S.COLOR_VALUES[int(mb["colors"][i, row.index(ask_a)])]
                cb = S.COLOR_VALUES[int(mb["colors"][i, row.index(ask_b)])]
                ca, cb = torch.tensor(ca), torch.tensor(cb)
                acc["rows"] += 1
                for blk in range(depth):
                    r = MB.find_read(eng, variant, z, tt, mb["cells"][i:i + 1], mb["colors"][i:i + 1],
                                     mb["points"][i:i + 1], 0, cell, ask_a, ask_b, blk, visible=visible, vary=vary)
                    s = sums.setdefault(blk, {"w_asked_a": 0.0, "w_asked_b": 0.0, "change": 0.0, "groups": 0.0,
                                              "max_abs_weight_delta": 0.0, "max_abs_score_delta": 0.0,
                                              "rows_exactly_equal": 0})
                    wa, wb = r["weights_a"].double(), r["weights_b"].double()           # [H, 2]
                    s["w_asked_a"] = s["w_asked_a"] + wa[:, 0]
                    s["w_asked_b"] = s["w_asked_b"] + wb[:, 1]
                    s["change"] = s["change"] + 0.5 * ((wa[:, 0] - wb[:, 0]) + (wb[:, 1] - wa[:, 1]))
                    ga = r["scores_a"].double()[:, 0] - r["scores_b"].double()[:, 0]     # [H, 8] on candidate a
                    gb = r["scores_b"].double()[:, 1] - r["scores_a"].double()[:, 1]
                    s["groups"] = s["groups"] + 0.5 * (ga + gb)
                    s["max_abs_weight_delta"] = max(s["max_abs_weight_delta"], float((wa - wb).abs().max()))
                    s["max_abs_score_delta"] = max(s["max_abs_score_delta"],
                                                   float((r["scores_a"].double() - r["scores_b"].double()).abs().max()))
                    s["rows_exactly_equal"] += int(torch.equal(r["weights_a"], r["weights_b"])
                                                   and torch.equal(r["answer_a"], r["answer_b"]))
                    if blk == 0:
                        acc["max_abs_answer_delta"] = max(acc["max_abs_answer_delta"], float(
                            (r["answer_a"].double() - r["answer_b"].double()).abs().max()))
                        zt = patchify(z, S.PATCH)[0, MB._tokens_of(cell, z.device)].cpu()
                        nearer = {}
                        for tag, ans, want, other in (("a", r["answer_a"], ca, cb), ("b", r["answer_b"], cb, ca)):
                            m = _mean_color(zt + (1 - t) * ans.cpu())
                            nearer[tag] = float((m - want).norm()) < float((m - other).norm())
                            if nearer[tag]:
                                acc["nearer_asked_" + tag] += 1
                        acc[{(True, True): "both_nearer", (True, False): "only_a", (False, True): "only_b",
                             (False, False): "neither"}[(nearer["a"], nearer["b"])]] += 1
        rows = max(1, acc["rows"])
        for blk, s in sums.items():
            acc["blocks"][str(blk)] = {
                "weight_on_asked_under_ask_a": (s["w_asked_a"] / rows).tolist(),
                "weight_on_asked_under_ask_b": (s["w_asked_b"] / rows).tolist(),
                "mean_change_of_the_asked_weight": (s["change"] / rows).tolist(),
                "mean_change_by_group": {g: (s["groups"][:, j] / rows).tolist() for j, g in enumerate(MB.GROUPS)},
                "max_abs_weight_delta": s["max_abs_weight_delta"], "max_abs_score_delta": s["max_abs_score_delta"],
                "rows_exactly_equal": s["rows_exactly_equal"], "rows": acc["rows"]}
        acc["share_nearer_asked_a"] = acc["nearer_asked_a"] / acc["rows"] if acc["rows"] else None
        acc["share_nearer_asked_b"] = acc["nearer_asked_b"] / acc["rows"] if acc["rows"] else None
        out["by_t"][str(t)] = acc
        log(f"FIND t={t}: {acc['rows']} rows used of {acc['rows_drawn']} drawn (skipped: {acc['skipped_same_color']} "
            f"same color, {acc['skipped_no_candidate']} no second candidate, {acc['skipped_no_pointer']} no pointer); "
            f"nearer the asked color: {acc['share_nearer_asked_a']} / {acc['share_nearer_asked_b']}; both asks "
            f"right in {acc['both_nearer']} of {acc['rows']} rows (only a {acc['only_a']}, only b {acc['only_b']}, "
            f"neither {acc['neither']}); largest answer change {acc['max_abs_answer_delta']}")
    return out


def _scene_set(a: dict, variant: MB.Variant) -> dict:
    k = int(a["k"])
    pl = MB.pointer_lists(4, k, SCENE_SEED)
    points = pl.points if variant.told == "all" else MB.no_points(pl.cells)
    pix, _ = S.render(pl.cells, pl.colors, a["bed"], torch.Generator().manual_seed(SCENE_SEED + 1))
    return {"cells": pl.cells, "colors": pl.colors, "points": points, "x": S.to_model(pix)}


def _edit_of(col: MB.MemberCollective, cells_row, points_row):
    """(cell, old class, new class) of a scene, or None (see the module's text)."""
    row = cells_row.tolist()
    asked = {int(p) for p in points_row.tolist() if int(p) >= 0}
    cell = next((c for c in range(9) if row[c] >= 0 and int(points_row[c]) < 0 and row[c] not in asked
                 and row[c] in col.things), None)
    if cell is None:
        return None
    new = next((k for k in col.classes if k not in row and k not in asked), None)
    return None if new is None else (cell, row[cell], new)


def _median_p95(values: Sequence[float]) -> dict:
    v = sorted(float(x) for x in values)
    if not v:
        return {"median": None, "p95": None, "n": 0}
    return {"median": v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2]),
            "p95": v[min(len(v) - 1, int(math.ceil(0.95 * len(v))) - 1)], "n": len(v)}


def _rel(a_: torch.Tensor, b_: torch.Tensor) -> float:
    return float((b_ - a_).pow(2).mean().sqrt() / a_.pow(2).mean().sqrt().clamp_min(1e-30))


def _run_address(a: dict, log) -> dict:
    col, used = load_collective(a)
    variant = col.variant
    dev = a["device"]
    out = {"kind": variant.name, "engines": used}
    out["find"] = _find(a, col, log) if variant.told == "all" else {"note": "told 'own': no pointers, no FIND"}
    sc = _scene_set(a, variant)
    edits = [_edit_of(col, sc["cells"][i], sc["points"][i]) for i in range(sc["cells"].shape[0])]
    chosen = [i for i, e in enumerate(edits) if e is not None]
    by_t = {}
    for t in _floats(a["ts"]):
        z = _noisy_at(sc["x"], t, SCENE_SEED + 2 + int(round(1000 * t))).to(dev)
        tt = torch.full((z.shape[0],), float(t), device=dev)
        cells, colors, points = sc["cells"].to(dev), sc["colors"].to(dev), sc["points"].to(dev)
        with torch.no_grad():
            answers, base_answer = col.part_answers(z, tt, cells, colors, points)
        serve, together, unturned, neutral, missing, unheld = [], [], [], [], [], []
        looks = {"old_before": [], "old_after": [], "new_before": [], "new_after": [], "unturned_old_before": [],
                 "unturned_old_after": []}
        others = []
        for i in chosen:
            cell, old, new = edits[i]
            sr = MB.serve_read(col, answers, base_answer, cells, i, cell, new)
            io, inew = col.classes.index(old), col.classes.index(new)
            serve.append({"old_before": float(sr["before"]["w"][:, io].mean()),
                          "new_after": float(sr["after"]["w"][:, inew].mean()),
                          "same_denominator": bool(torch.equal(sr["before"]["denominator"], sr["after"]["denominator"])),
                          "reverse_restores": all(torch.equal(sr["before"][kk], sr["reverse"][kk]) for kk in sr["before"])})
            sw = MB.address_swap(col, z, tt, cells, colors, points, i, cell, new)
            uw = MB.address_swap(col, z, tt, cells, colors, points, i, cell, new, turn=False)
            nw = MB.address_swap(col, z, tt, cells, colors, points, i, cell, new, gate="neutral")
            off = MB._FRAME.cell.to(dev) != cell
            together.append({"old_before": float(sw["before"]["gate"]["w"][:, io].mean()),
                             "new_after": float(sw["after"]["gate"]["w"][:, inew].mean()),
                             "old_after": float(sw["after"]["gate"]["w"][:, io].max()),
                             "other_cells_unchanged": bool(torch.equal(sw["before"]["w_all"][off],
                                                                       sw["after"]["w_all"][off]))})
            others += list(sw["other_relative_change"].values())
            for key, st, tag in (("old_before", sw, "before"), ("old_after", sw, "after")):
                looks[key].append(st[tag]["looks_old"])
            for key, st, tag in (("new_before", sw, "before"), ("new_after", sw, "after")):
                looks[key].append(st[tag]["looks_new"])
            looks["unturned_old_before"].append(uw["before"]["looks_old"])
            looks["unturned_old_after"].append(uw["after"]["looks_old"])
            unturned.append(all(torch.equal(uw[tag]["w_all"], sw[tag]["w_all"]) for tag in ("before", "after")))
            neutral.append({"old": _rel(nw["before"]["answers"][io], nw["after"]["answers"][io]),
                            "new": _rel(nw["before"]["answers"][inew], nw["after"]["answers"][inew])})
            m = MB.address_missing(col, cells, i, cell)
            u = MB.address_unheld(col, cells, i, cell)
            missing.append({"largest_member": float(m["w"].max()), "smallest_base": float(m["base"].min())})
            unheld.append({"largest_member": float(u["w"].max()), "smallest_base": float(u["base"].min()),
                           "address": list(u["address"])})
        tol = _tolerance(col, sc, t, dev)
        mean_looks = {k: (torch.stack([x for x in v if x is not None]).mean(0).tolist()
                          if any(x is not None for x in v) else None) for k, v in looks.items()}
        by_t[str(t)] = {
            "scenes": len(chosen), "skipped": len(edits) - len(chosen),
            "serve": {"old_before": _mean([s["old_before"] for s in serve]),
                      "new_after": _mean([s["new_after"] for s in serve]),
                      "same_denominator": all(s["same_denominator"] for s in serve),
                      "reverse_restores": all(s["reverse_restores"] for s in serve)},
            "together": {"old_before": _mean([x["old_before"] for x in together]),
                         "new_after": _mean([x["new_after"] for x in together]),
                         "old_after_largest": max((x["old_after"] for x in together), default=None),
                         "other_cells_unchanged": all(x["other_cells_unchanged"] for x in together),
                         "looks": mean_looks, "other_relative_change": _median_p95(others)},
            "unturned_gate_identical": all(unturned),
            "neutral": {"old": _median_p95([x["old"] for x in neutral]), "new": _median_p95([x["new"] for x in neutral])},
            "missing": {"largest_member": max((x["largest_member"] for x in missing), default=None),
                        "smallest_base": min((x["smallest_base"] for x in missing), default=None)},
            "unheld": {"largest_member": max((x["largest_member"] for x in unheld), default=None),
                       "smallest_base": min((x["smallest_base"] for x in unheld), default=None)},
            "tolerance": tol}
        log(f"address t={t}: {len(chosen)} scenes; serve {by_t[str(t)]['serve']}; tolerance pooled "
            f"{tol['pooled']}")
    out["by_t"] = by_t
    out["inconclusive"] = any(not u["descended"] for u in used)
    return out


def _mean(values) -> Optional[float]:
    v = [float(x) for x in values]
    return sum(v) / len(v) if v else None


@torch.no_grad()
def _tolerance(col: MB.MemberCollective, sc: dict, t: float, dev) -> dict:
    """The relative change of every member's contribution on its own tokens when one empty cell's noise is redrawn."""
    x = sc["x"]
    gen = torch.Generator().manual_seed(TOLERANCE_SEED + int(round(1000 * t)))
    noise = torch.randn(tuple(x.shape), generator=torch.Generator().manual_seed(SCENE_SEED + 2 + int(round(1000 * t))))
    fresh = torch.randn(tuple(x.shape), generator=gen)
    noise2 = noise.clone()
    per: Dict[str, List[float]] = {str(k): [] for k in col.classes}
    cells, colors, points = sc["cells"].to(dev), sc["colors"].to(dev), sc["points"].to(dev)
    scenes = 0
    for i in range(x.shape[0]):
        empty = [c for c in range(9) if int(sc["cells"][i, c]) < 0]
        if not empty:
            continue
        c = empty[0]
        y0, x0 = (c // 3) * S.CELL, (c % 3) * S.CELL
        noise2[i, :, y0:y0 + S.CELL, x0:x0 + S.CELL] = fresh[i, :, y0:y0 + S.CELL, x0:x0 + S.CELL]
        scenes += 1
    z1, z2 = ((1 - t) * noise + t * x).to(dev), ((1 - t) * noise2 + t * x).to(dev)
    tt = torch.full((x.shape[0],), float(t), device=dev)
    ca = col.contributions(z1, tt, cells, colors, points)
    cb = col.contributions(z2, tt, cells, colors, points)
    w, _ = col.weights(cells)
    for j, k in enumerate(col.classes):
        for i in range(x.shape[0]):
            owned = w[i, :, j] > 0.5
            if bool(owned.any()) and int((sc["cells"][i] < 0).sum()):
                per[str(k)].append(_rel(ca[j, i][owned], cb[j, i][owned]))
    pooled = [v for vs in per.values() for v in vs]
    return {"scenes": scenes, "by_member": {k: _median_p95(v) for k, v in per.items()}, "pooled": _median_p95(pooled)}


# ------------------------------------------------------------------ the exchange
def _run_exchange(a: dict, log) -> dict:
    col, used = load_collective(a)
    k = int(a["swap_class"]) if int(a["swap_class"]) >= 0 else col.classes[0]
    if k not in col.things:
        raise ValueError(f"swap_class {k} has no engine in the collective")
    twin, used_t = load_collective(a, twin={k: int(a["twin_seed"])})
    col_b = MB.MemberCollective({c: (twin.things[c] if c == k else col.things[c]) for c in col.classes}, col.base,
                                col.variant)
    sc = _scene_set(a, col.variant)
    dev = a["device"]
    one = {}
    for t in _floats(a["ts"]):
        z = _noisy_at(sc["x"], t, SCENE_SEED + 2 + int(round(1000 * t))).to(dev)
        r = MB.exchange_read(col, col_b, z, torch.full((z.shape[0],), float(t), device=dev), sc["cells"].to(dev),
                             sc["colors"].to(dev), sc["points"].to(dev))
        one[str(t)] = r
        log(f"exchange one step t={t}: exchanged {r['exchanged']}; others' largest difference "
            f"{max(r['others'].values()) if r['others'] else None}")
    rl = S.request_lists(CHAIN_N, int(a["count"]), seed=LIST_SEED + CHAIN_N)
    m = int(a["exact_lists"])
    cells, colors = rl.cells[:m], rl.colors[:m]
    noise = C.start_noise(int(a["count"]), NOISE_SEED + CHAIN_N)[:m]
    pts = MB.no_points(cells)
    za = C.sample(col.packed_answer, cells, MB.pack(colors, pts), 0, int(a["sample_steps"]), dev, int(a["chunk"]),
                  noise=noise)
    zb = C.sample(col_b.packed_answer, cells, MB.pack(colors, pts), 0, int(a["sample_steps"]), dev, int(a["chunk"]),
                  noise=noise)
    chain = exchange_exact(made_pixels(za), made_pixels(zb), cells, [k], S.to_pixels(za), S.to_pixels(zb))
    log(f"exchange over the chain: largest outside {chain['largest_outside']} (before the clamp "
        f"{chain['largest_outside_unclamped']}); exchanged cells that differ {chain['exchanged_cells_differing']}")
    engines = used + [u for u in used_t if u["cls"] == k]
    return {"kind": col.variant.name, "swap_class": k, "twin_seed": int(a["twin_seed"]), "one_step": one,
            "chain": chain, "engines": engines, "inconclusive": any(not u["descended"] for u in engines)}


# ------------------------------------------------------------------ parts and timing
def _run_parts(a: dict, log) -> dict:
    variant = _variant(a)
    bed = S.ScenesBed(a["bed"], a["device"])
    snaps = _ints(a["snapshots"]) if str(a["snapshots"]).strip() else None
    r = MT.train_members(variant, parse_classes(a["classes"]), bed, int(a["seed"]), a["out"], steps=int(a["steps"]),
                         batch=int(a["batch"]), lr=float(a["lr"]), warmup=int(a["warmup"]), every=int(a["every"]),
                         heldout=int(a["heldout"]), save_every=int(a["save_every"]), p_point=float(a["p_point"]),
                         exclude=_exclude(a), held_pairs=held_pairs(a["held"]), dim=int(a["dim"]),
                         heads=int(a["heads"]), depth=int(a["depth"]), snapshots=snaps, noise=str(a["noise"]), log=log)
    return {"kind": variant.name, "finished": r.get("finished"), "engines": r.get("engines"),
            "seconds_a_step_stack": r.get("seconds_a_step_stack"), "seconds_a_step_engine": r.get("seconds_a_step_engine"),
            "trained": r.get("trained")}


def _run_timing(a: dict, log) -> dict:
    variant = _variant(a)
    bed = S.ScenesBed(a["bed"], a["device"])
    pool = list(range(9)) + [None]
    rows = []
    for g in _ints(a["stacks"]):
        if not 1 <= g <= len(pool):
            raise ValueError(f"a stack holds 1 .. {len(pool)} engines, not {g}")
        r = MT.time_stack(variant, pool[:g], bed, int(a["seed"]), batch=int(a["batch"]), steps=int(a["steps"]),
                          from_step=int(a["from_step"]), p_point=float(a["p_point"]), exclude=_exclude(a),
                          held_pairs=held_pairs(a["held"]), dim=int(a["dim"]), heads=int(a["heads"]),
                          depth=int(a["depth"]), noise=str(a["noise"]), phases=bool(int(a["phases"])))
        rows.append(r)
        log(f"timing G={g}: {r['seconds_a_step_stack']} s a step of the stack, {r['seconds_a_step_engine']} an engine")
        if r.get("phases"):
            log(f"timing G={g} phases (median seconds): " + ", ".join(f"{k} {v:.4f}" for k, v in r["phases"].items()))
    return {"kind": variant.name, "timing": rows}


RUNNERS = {"parts": _run_parts, "timing": _run_timing, "compose": _run_compose, "address": _run_address,
           "exchange": _run_exchange}


# ------------------------------------------------------------------ the result
def _f(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _m(r: dict) -> str:
    return _f((r or {}).get("mean"))


def _text(result: dict) -> str:
    v = result["variant"]
    lines = [f"# addressed_collective: {v}, kind {result.get('kind')}", ""]
    if v == "parts":
        lines += ["| engine | newborn read | last read | descended |", "|---|---|---|---|"]
        lines += [f"| {e['tag']} | {_f(e['newborn_read'])} | {_f(e['last_read'])} | {_f(e['descended'])} |"
                  for e in result.get("engines") or []]
        lines.append(f"\nseconds a step: {_f(result.get('seconds_a_step_stack'))} the stack, "
                     f"{_f(result.get('seconds_a_step_engine'))} an engine")
    elif v == "timing":
        lines += ["| engines | batch | seconds a step (stack) | seconds a step (engine) |", "|---|---|---|---|"]
        lines += [f"| {r['engines']} | {r['batch']} | {_f(r['seconds_a_step_stack'])} | {_f(r['seconds_a_step_engine'])} |"
                  for r in result["timing"]]
        timed = [r for r in result["timing"] if r.get("phases")]
        if timed:
            names = list(timed[0]["phases"])
            lines += ["", f"PHASES (median seconds a step, the card waited for after each; noise "
                      f"{timed[0].get('noise')})", "", "| engines | " + " | ".join(names) + " |",
                      "|---|" + "---|" * len(names)]
            lines += [f"| {r['engines']} | " + " | ".join(_f(r["phases"][k]) for k in names) + " |" for r in timed]
    elif v == "compose":
        lines += ["THE STORED LISTS", "", "| n | things right | se | whole scenes right | colors right |",
                  "|---|---|---|---|---|"]
        lines += [f"| {n} | {_m(r['cell_rate'])} | {_f(r['cell_rate']['se'])} | {_m(r['scene_right'])} | "
                  f"{_m(r['color_rate'])} |" for n, r in result["stored"].items()]
        for label, by_n in result["pointing"].items():
            lines += ["", f"LISTS THAT POINT ({label.replace('_', ' ')}; distinct colors)", "",
                      "| n | stated: thing right | stated: color right | pointing: thing right | pointing: color right "
                      "| a guess among every other stated color | a guess among the eligible requests only | lists "
                      "| k (eligible requests): lists |", "|---|---|---|---|---|---|---|---|---|"]
            lines += [f"| {n} | {_m(r['stated_thing_right'])} | {_m(r['stated_color_right'])} | "
                      f"{_m(r['pointing_thing_right'])} | {_m(r['pointing_color_right'])} | "
                      f"{_f(r['reference_all_stated']['mean'])} | {_f(r['reference_eligible']['mean'])} | "
                      f"{r['reference_eligible']['lists']} | "
                      f"{', '.join(f'{k}: {c}' for k, c in r['reference_eligible']['k_histogram'].items())} |"
                      for n, r in by_n.items()]
        lines.append(f"\ninconclusive: {_f(result['inconclusive'])}")
    elif v == "address":
        find = result["find"]
        if "by_t" in find:
            lines += [f"FIND (vary {find['vary']}; k = {find.get('k_rows_drawn_per_engine')} rows drawn for each "
                      "engine)", "", "| t | rows | skipped: same color | block | head | weight on the "
                      "asked (ask a) | (ask b) | mean change | nearer the asked color (a / b) | largest weight change "
                      "| largest score change | rows exactly equal |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for t, x in find["by_t"].items():
                for blk, b in x["blocks"].items():
                    for h in range(len(b["mean_change_of_the_asked_weight"])):
                        lines.append(f"| {t} | {x['rows']} | {x['skipped_same_color']} | {blk} | {h} | "
                                     f"{_f(b['weight_on_asked_under_ask_a'][h])} | {_f(b['weight_on_asked_under_ask_b'][h])}"
                                     f" | {_f(b['mean_change_of_the_asked_weight'][h])} | "
                                     f"{_f(x['share_nearer_asked_a'])} / {_f(x['share_nearer_asked_b'])} | "
                                     f"{_f(b['max_abs_weight_delta'])} | {_f(b['max_abs_score_delta'])} | "
                                     f"{b['rows_exactly_equal']} of {b['rows']} |")
            lines += ["", "| t | rows used | nearer the asked color under both asks | ask a only | ask b only | neither "
                      "| largest change of the raw answers |", "|---|---|---|---|---|---|---|"]
            lines += [f"| {t} | {x['rows']} | {x['both_nearer']} | {x['only_a']} | {x['only_b']} | {x['neither']} | "
                      f"{_f(x['max_abs_answer_delta'])} |" for t, x in find["by_t"].items()]
            lines += ["", "| t | engine | rows drawn | rows used | skipped: no pointer | same color | no second "
                      "candidate |", "|---|---|---|---|---|---|---|"]
            for t, x in find["by_t"].items():
                for cls, e in x["per_engine"].items():
                    lines.append(f"| {t} | {cls} | {e['drawn']} | {e['used']} | {e['skipped_no_pointer']} | "
                                 f"{e['skipped_same_color']} | {e['skipped_no_candidate']} |")
                lines.append(f"| {t} | all | {x['rows_drawn']} | {x['rows']} | {x['skipped_no_pointer']} | "
                             f"{x['skipped_same_color']} | {x['skipped_no_candidate']} |")
        lines += ["", "| t | scenes | serve: old before | serve: new after | same denominator | together: new after | "
                  "other cells unchanged | unturned gate identical | others' change (median) | missing: base | "
                  "unheld: base | tolerance (median / p95) |", "|" + "---|" * 12]
        for t, x in result["by_t"].items():
            lines.append(f"| {t} | {x['scenes']} | {_f(x['serve']['old_before'])} | {_f(x['serve']['new_after'])} | "
                         f"{_f(x['serve']['same_denominator'])} | {_f(x['together']['new_after'])} | "
                         f"{_f(x['together']['other_cells_unchanged'])} | {_f(x['unturned_gate_identical'])} | "
                         f"{_f(x['together']['other_relative_change']['median'])} | "
                         f"{_f(x['missing']['smallest_base'])} | {_f(x['unheld']['smallest_base'])} | "
                         f"{_f(x['tolerance']['pooled']['median'])} / {_f(x['tolerance']['pooled']['p95'])} |")
    elif v == "exchange":
        lines += ["| t | exchanged | largest difference of the others |", "|---|---|---|"]
        lines += [f"| {t} | {r['exchanged']} | {_f(max(r['others'].values()) if r['others'] else None)} |"
                  for t, r in result["one_step"].items()]
        c = result["chain"]
        lines.append(f"\nover the chain (n = {CHAIN_N}, {c['cells_compared']} cells compared): largest difference "
                     f"outside the exchanged class's cells {c['largest_outside']} (before the clamp "
                     f"{c['largest_outside_unclamped']}); exchanged cells that differ {c['exchanged_cells_differing']}")
    lines.append(f"\nseconds: {result['seconds']:.1f}; package {result['version']}")
    return "\n".join(lines) + "\n"


def main(argv=(), **overrides) -> dict:
    a = parse(DEFAULTS, argv, **overrides)
    if not a["out"]:
        raise ValueError("out= is required")
    if a["variant"] not in VARIANTS:
        raise ValueError(f"unknown variant {a['variant']!r}; known: {list(VARIANTS)}")
    _variant(a)
    os.makedirs(a["out"], exist_ok=True)
    log_path = os.path.join(a["out"], "log.txt")

    def log(line: str) -> None:
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")

    torch.set_num_threads(int(a["threads"]))
    t0 = time.time()
    log(f"ADDRESSED COLLECTIVE {a['variant']}: kind {a['kind']}, bed {a['bed']}, seed {a['seed']}, device "
        f"{a['device']}, package {__version__}")
    with determinism.mode(a["det"]) as det:
        result = RUNNERS[a["variant"]](a, log)
        det.record(result)
    result.update({"experiment": EXPERIMENT, "variant": a["variant"], "noise": str(a["noise"]),
                   "args": {k: a[k] for k in sorted(a)},
                   "version": __version__, "torch": torch.__version__, "seconds": time.time() - t0})
    with open(os.path.join(a["out"], "result.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=1, default=str)
    text = _text(result)
    with open(os.path.join(a["out"], "result.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    log(text.rstrip())
    log("done")
    return result


if __name__ == "__main__" and not in_notebook():
    main(sys.argv[1:])
