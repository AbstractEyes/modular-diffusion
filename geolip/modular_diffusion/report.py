"""Tables from result files. No model is loaded; every builder returns markdown (and writes it when ``out`` is given).

Result files are the JSON files a training run writes, named ``md_<tag>.json`` (and ``mr_<tag>.json`` for results of
the original research script, which have the same keys). Build files (``*.build.json``) and unit cards are skipped.

    grade(title, folders)                                   loss, levels, bands and the importance of each part
    grade_map(title, base, map_folder, content=None)        expansions trained on a frozen base, one per cell
    check_expansion(base, one, map_folder)                  exactness of one expansion trained alone
    check_merge(body, solos, joint, merged, end_to_end)     members trained alone and merged by name
    seed_spread(folders, title)                             the spread of one setting over seeds
"""
from __future__ import annotations

import glob
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence

LEVELS = ("0.05", "0.15", "0.25", "0.35", "0.45", "0.55", "0.65", "0.75", "0.85", "0.95")
BANDS = ("fine <4", "mid 4-8", "coarse 8+")
BAND_NAMES = {0: "fine (under 4 tokens)", 1: "mid (4 to 8)", 2: "coarse (8 and longer)"}
PREFIXES = ("md_", "mr_")
_SKIP_SUFFIXES = (".build.json", ".card.json")


def result_files(folder: str) -> List[str]:
    """The result files of a folder, sorted by path."""
    found = []
    for prefix in PREFIXES:
        found += glob.glob(os.path.join(folder, prefix + "*.json"))
    return sorted(p for p in found if not p.endswith(_SKIP_SUFFIXES))


def _read(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load(folder: str) -> Dict[int, dict]:
    """{seed: result} of a folder (a later file of the same seed replaces an earlier one)."""
    runs = {}
    for p in result_files(folder):
        d = _read(p)
        runs[d["seed"]] = d
    return runs


def _write(text: str, out: Optional[str]) -> str:
    if out:
        folder = os.path.dirname(out)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    return text


def variant_of(tag: str) -> str:
    """A run's tag without its seed: the runs of one setting share it."""
    return re.sub(r"_s\d+_steps", "_steps", tag)


# ------------------------------------------------------------------ grade
def grade(title: str, folders: Sequence[str], out: Optional[str] = None) -> str:
    """Runs grouped by their tag without the seed: the loss, by noise level, by band, and what each part is worth."""
    runs: Dict[str, Dict[int, dict]] = {}
    for folder in folders:
        for p in result_files(folder):
            d = _read(p)
            runs.setdefault(variant_of(d["tag"]), {})[d["seed"]] = d
    variants = list(runs)
    L = [f"# {title}", ""]
    L.append("## The loss (held-out pictures or fields, fixed noise; lower is better)")
    L.append("| variant | trained numbers | seeds | mean over ten noise levels, per seed | mean | seed spread | the anchor alone (linear physics) | all members off against the sum of single removals | minutes a run |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for a in variants:
        seeds = sorted(runs[a])
        vals = [runs[a][s]["loss_over_levels"] for s in seeds]
        mean = sum(vals) / len(vals)
        spread = max(vals) - min(vals)
        d0 = runs[a][seeds[0]]
        ind = d0.get("independence_ratio_over_levels")
        L.append(f"| {a} | {d0['trained_numbers']:,} | {seeds} | " + " / ".join(f"{v:.4f}" for v in vals) + f" | {mean:.4f} | {spread:.4f} | "
                 f"{d0['linear_physics_loss_over_levels']:.4f} | {'-' if ind is None else f'{ind:.4f}'} | {d0['train_seconds'] / 60:.1f} |")
    L.append("")
    L.append("## By noise level (t = 0 pure noise, 1 the picture): the mean over seeds, and under it what is left of the anchor's error")
    L.append("| variant | " + " | ".join(LEVELS) + " |")
    L.append("|---|" + "---|" * len(LEVELS))
    lin = None
    for a in variants:
        seeds = sorted(runs[a])
        row = [sum(runs[a][s]["levels"][lv]["error"]["all"] for s in seeds) / len(seeds) for lv in LEVELS]
        lin = [runs[a][seeds[0]]["levels"][lv]["linear_physics"]["all"] for lv in LEVELS]
        L.append(f"| {a} | " + " | ".join(f"{x:.4f}" for x in row) + " |")
        L.append("| ... as a share of the anchor's error | " + " | ".join(f"{x / y:.3f}" for x, y in zip(row, lin)) + " |")
    if lin:
        L.append("| the anchor alone (linear physics) | " + " | ".join(f"{x:.4f}" for x in lin) + " |")
    L.append("")
    L.append("## By frequency band at three noise levels (error per frequency; mean over seeds; the anchor alone in brackets)")
    L.append("| variant | t 0.25: fine / mid / coarse | t 0.45: fine / mid / coarse | t 0.65: fine / mid / coarse |")
    L.append("|---|---|---|---|")
    for a in variants:
        seeds = sorted(runs[a])
        cells = []
        for lv in ("0.25", "0.45", "0.65"):
            e = [sum(runs[a][s]["levels"][lv]["error"][b] for s in seeds) / len(seeds) for b in BANDS]
            p = [runs[a][seeds[0]]["levels"][lv]["linear_physics"][b] for b in BANDS]
            cells.append(" / ".join(f"{x:.3f}" for x in e) + " (" + " / ".join(f"{x:.3f}" for x in p) + ")")
        L.append(f"| {a} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## THE IMPORTANCE LEDGER: what the loss gains when a part is switched off, by noise level (the first seed of each variant)")
    for a in variants:
        seeds = sorted(runs[a])
        d = runs[a][seeds[0]]
        parts = []
        for lv in LEVELS:
            for k in d["levels"][lv]["off_cost"]:
                if k not in parts:
                    parts.append(k)
        if not parts:
            continue
        L.append(f"### {a} (seed {seeds[0]})")
        L.append("| part | " + " | ".join(LEVELS) + " | share of its output by band (fine / mid / coarse) at its most important level |")
        L.append("|---|" + "---|" * len(LEVELS) + "---|")
        for k in parts:
            row = [d["levels"][lv]["off_cost"].get(k, {}).get("all") for lv in LEVELS]
            best = max(LEVELS, key=lambda lv: abs(d["levels"][lv]["off_cost"].get(k, {}).get("all") or 0.0))
            sh = d["levels"][best]["output_share"].get(k)
            shs = " / ".join(f"{sh[b]:.2f}" for b in BANDS) if sh else "-"
            L.append(f"| {k} | " + " | ".join("-" if x is None else f"{x:+.4f}" for x in row) + f" | {shs} |")
        if "all_members_off" in d["levels"]["0.45"]:
            L.append("| all members off | " + " | ".join(f"{d['levels'][lv].get('all_members_off', 0):+.4f}" for lv in LEVELS) + " | - |")
            L.append("| sum of the single member removals | " + " | ".join(f"{d['levels'][lv].get('sum_of_single_member_off', 0):+.4f}" for lv in LEVELS) + " | - |")
        att = d["levels"]["0.45"].get("attention")
        if att:
            L.append("Where the members' first attention looks at t 0.45 (mean distance in tokens; weight on itself; in the 3 x 3 block): "
                     + "; ".join(f"{k} {v['mean_distance']:.2f} / {v['on_itself']:.3f} / {v['in_3x3_block']:.3f}" for k, v in att.items()) + ".")
        L.append("")
    return _write("\n".join(L) + "\n", out)


# ------------------------------------------------------------------ grade_map
def _inside(lo: float, hi: float) -> List[str]:
    return [lv for lv in LEVELS if lo <= float(lv) < hi]


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Rank correlation (ties ranked by position)."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k, i in enumerate(order):
            r[i] = float(k)
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den > 0 else float("nan")


def _cells_of(d: dict, prefix: str) -> dict:
    out = {}
    for name in d["members"]:
        m = re.match(rf"^{prefix}(\d*)_b(\d)_([0-9.]+)-([0-9.]+)$", name)
        if m:
            out[name] = (int(m.group(2)), float(m.group(3)), float(m.group(4)), m.group(1))
    return out


def _cell_mean(d: dict, name: str, lo: float, hi: float, key: str = "all") -> float:
    lv = _inside(lo, hi)
    return sum(d["levels"][x]["off_cost"].get(name, {}).get(key, 0.0) for x in lv) / len(lv)


def grade_map(title: str, base_dir: str, map_dir: str, content_dir: Optional[str] = None, out: Optional[str] = None) -> str:
    """Expansions trained together on a frozen base, one per cell (band x window of the noise level), and optionally
    class-addressed expansions in the same cells. A cell's gain is the loss that returns when its expansion is switched
    off, averaged over the read's noise levels inside the cell's window; the base member's importance in a cell is what
    the loss gains when the base member of that band is switched off, read on the base model at the same levels."""
    base, mp = load(base_dir), load(map_dir)
    seeds = sorted(set(base) & set(mp))
    L = [f"# {title}", ""]
    L.append("## The whole model: the frozen base against the base with every expansion on")
    L.append("| seed | base, mean over ten noise levels | with the expansions | gain | trained numbers added | frozen numbers unchanged | minutes |")
    L.append("|---|---|---|---|---|---|---|")
    for s in seeds:
        b, m = base[s], mp[s]
        L.append(f"| {s} | {b['loss_over_levels']:.4f} | {m['loss_over_levels']:.4f} | {b['loss_over_levels'] - m['loss_over_levels']:+.4f} | "
                 f"{m['trained_numbers']:,} | {m['frozen_unchanged']} | {m['train_seconds'] / 60:.1f} |")
    L.append("")
    L.append("## By noise level (t = 0 pure noise, 1 the picture): the base's error, and the gain of the expansions (mean over seeds)")
    L.append("| | " + " | ".join(LEVELS) + " |")
    L.append("|---|" + "---|" * len(LEVELS))
    be = [sum(base[s]["levels"][lv]["error"]["all"] for s in seeds) / len(seeds) for lv in LEVELS]
    me = [sum(mp[s]["levels"][lv]["error"]["all"] for s in seeds) / len(seeds) for lv in LEVELS]
    li = [base[seeds[0]]["levels"][lv]["linear_physics"]["all"] for lv in LEVELS]
    L.append("| the anchor alone (linear physics) | " + " | ".join(f"{x:.4f}" for x in li) + " |")
    L.append("| the base | " + " | ".join(f"{x:.4f}" for x in be) + " |")
    L.append("| what training added over the anchor (anchor minus base) | " + " | ".join(f"{x - y:+.4f}" for x, y in zip(li, be)) + " |")
    L.append("| the gain of the expansions | " + " | ".join(f"{x - y:+.4f}" for x, y in zip(be, me)) + " |")
    L.append("| ... as a share of the base's error | " + " | ".join(f"{(x - y) / x:.3f}" for x, y in zip(be, me)) + " |")
    L.append("")
    cells = _cells_of(mp[seeds[0]], "x")
    windows = sorted({(lo, hi) for _, lo, hi, _ in cells.values()})
    bands = sorted({b for b, _, _, _ in cells.values()})
    L.append("## THE MAP: the gain of one expansion in each cell (per seed), and under it the base member's importance in that cell")
    L.append("| band | " + " | ".join(f"t {lo:g} to {hi:g}" for lo, hi in windows) + " |")
    L.append("|---|" + "---|" * len(windows))
    gain, imp = {s: {} for s in seeds}, {s: {} for s in seeds}
    for b in bands:
        row_g, row_i = [], []
        for lo, hi in windows:
            name = next((n for n, c in cells.items() if c[0] == b and c[1] == lo and c[2] == hi), None)
            if name is None:
                row_g.append("-")
                row_i.append("-")
                continue
            base_name = next((n for n in base[seeds[0]]["members"] if n.startswith(f"b{b}_")), None)
            for s in seeds:
                gain[s][(b, lo, hi)] = _cell_mean(mp[s], name, lo, hi)
                imp[s][(b, lo, hi)] = _cell_mean(base[s], base_name, lo, hi) if base_name else float("nan")
            row_g.append(" / ".join(f"{gain[s][(b, lo, hi)]:+.4f}" for s in seeds))
            row_i.append(" / ".join(f"{imp[s][(b, lo, hi)]:+.4f}" for s in seeds))
        L.append(f"| {BAND_NAMES.get(b, b)}: expansion's gain | " + " | ".join(row_g) + " |")
        L.append("| ... the base member's importance there | " + " | ".join(row_i) + " |")
    L.append("")
    for s in seeds:
        keys = sorted(gain[s])
        if len(keys) >= 4:
            rho = spearman([imp[s][k] for k in keys], [gain[s][k] for k in keys])
            best = max(keys, key=lambda k: gain[s][k])
            L.append(f"Seed {s}: rank correlation between a cell's importance and its expansion's gain {rho:+.2f} over {len(keys)} cells; "
                     f"the largest gain: {BAND_NAMES.get(best[0], best[0])}, t {best[1]:g} to {best[2]:g} ({gain[s][best]:+.4f}).")
    L.append("")
    L.append("## Gain by window, the three bands together (per seed), and its share of the whole gain over ten levels")
    L.append("| window | levels read | gain per level, bands summed | share of the whole gain |")
    L.append("|---|---|---|---|")
    for lo, hi in windows:
        lv = _inside(lo, hi)
        cols, shares = [], []
        for s in seeds:
            g = sum(gain[s].get((b, lo, hi), 0.0) for b in bands)
            tot = sum(sum(gain[s].get((b, l2, h2), 0.0) for b in bands) * len(_inside(l2, h2)) for l2, h2 in windows)
            cols.append(f"{g:+.4f}")
            shares.append(f"{g * len(lv) / tot:.2f}" if tot else "-")
        L.append(f"| t {lo:g} to {hi:g} | {len(lv)} | " + " / ".join(cols) + " | " + " / ".join(shares) + " |")
    L.append("")
    if content_dir:
        ct = load(content_dir)
        cs = sorted(set(seeds) & set(ct))
        ccells = _cells_of(ct[cs[0]], "c")
        L.append("## CONTENT: two class-addressed expansions in a cell (animals / vehicles, each on only for its pictures) against the one unaddressed expansion")
        L.append("| band | window | unaddressed gain (per seed) | the content pair's gain (per seed) | pair minus unaddressed |")
        L.append("|---|---|---|---|---|")
        for b in bands:
            for lo, hi in windows:
                names = [n for n, c in ccells.items() if c[0] == b and c[1] == lo and c[2] == hi]
                if not names:
                    continue
                pair = [sum(_cell_mean(ct[s], n, lo, hi) for n in names) for s in cs]
                un = [gain[s].get((b, lo, hi), float("nan")) for s in cs]
                L.append(f"| {BAND_NAMES.get(b, b)} | t {lo:g} to {hi:g} | " + " / ".join(f"{x:+.4f}" for x in un) + " | "
                         + " / ".join(f"{x:+.4f}" for x in pair) + " | " + " / ".join(f"{p - u:+.4f}" for p, u in zip(pair, un)) + " |")
        L.append("")
        L.append("| seed | base | with the unaddressed expansions | with the content expansions |")
        L.append("|---|---|---|---|")
        for s in cs:
            L.append(f"| {s} | {base[s]['loss_over_levels']:.4f} | {mp[s]['loss_over_levels']:.4f} | {ct[s]['loss_over_levels']:.4f} |")
        L.append("")
    d0 = base[seeds[0]]
    if "error_without_caption" in d0["levels"]["0.45"]:
        L.append("## What the caption is worth to the base, by noise level (the error without the caption minus the error with it; mean over seeds)")
        L.append("| | " + " | ".join(LEVELS) + " |")
        L.append("|---|" + "---|" * len(LEVELS))
        cap = [sum(base[s]["levels"][lv]["error_without_caption"]["all"] - base[s]["levels"][lv]["error"]["all"] for s in seeds) / len(seeds) for lv in LEVELS]
        L.append("| the caption's worth | " + " | ".join(f"{x:+.4f}" for x in cap) + " |")
        spread = []
        for lv in LEVELS:
            v = [sum(base[s]["levels"][lv]["error_by_class"][c] for s in seeds) / len(seeds) for c in range(10)]
            mean = sum(v) / 10
            spread.append((sum((x - mean) ** 2 for x in v) / 10) ** 0.5 / mean)
        L.append("| spread of the error across the ten classes (standard deviation over the mean) | " + " | ".join(f"{x:.3f}" for x in spread) + " |")
        L.append("")
    return _write("\n".join(L) + "\n", out)


# ------------------------------------------------------------------ check_expansion
def check_expansion(base_dir: str, one_dir: str, map_dir: str, out: Optional[str] = None) -> str:
    """One expansion trained alone on a frozen base, against the base and against the map of expansions.

    EXACT: outside the expansion's window every noise level reads the base's numbers, and inside it the other bands do,
    with a largest change of 0.0; no frozen number moved. TOGETHER = ALONE: the cell's gain in the map (its expansion
    switched off) against the gain of the expansion trained alone."""
    base, one, mp = load(base_dir), load(one_dir), load(map_dir)
    L = ["# One expansion trained alone on the frozen base: the exactness marks", ""]
    L.append("| seed | the expansion | frozen numbers unchanged | largest change at noise levels outside its window | largest change in the other bands (all levels) | "
             "its gain alone (mean over its window's levels) | the same cell's gain inside the nine-cell map | difference |")
    L.append("|---|---|---|---|---|---|---|---|")
    ok = True
    for s in sorted(set(base) & set(one)):
        b, o = base[s], one[s]
        new = [n for n in o["members"] if n not in b["members"]]
        if len(new) != 1:
            raise ValueError(f"expected exactly one new member in seed {s}: {new}")
        m = re.match(r"^x\w*_b(\d)_([0-9.]+)-([0-9.]+)$", new[0])
        band, lo, hi = int(m.group(1)), float(m.group(2)), float(m.group(3))
        inside = [lv for lv in LEVELS if lo <= float(lv) < hi]
        outside = [lv for lv in LEVELS if lv not in inside]
        d_out = max(abs(o["levels"][lv]["error"][k] - b["levels"][lv]["error"][k]) for lv in outside for k in BANDS + ("all",))
        other = [k for i, k in enumerate(BANDS) if i != band]
        d_other = max(abs(o["levels"][lv]["error"][k] - b["levels"][lv]["error"][k]) for lv in LEVELS for k in other)
        gain_alone = sum(b["levels"][lv]["error"]["all"] - o["levels"][lv]["error"]["all"] for lv in inside) / len(inside)
        gain_map = float("nan")
        if s in mp and new[0] in mp[s]["members"]:
            gain_map = sum(mp[s]["levels"][lv]["off_cost"][new[0]]["all"] for lv in inside) / len(inside)
        ok = ok and o["frozen_unchanged"] and d_out == 0.0 and d_other == 0.0
        L.append(f"| {s} | {new[0]} | {o['frozen_unchanged']} | {d_out:.1e} | {d_other:.1e} | {gain_alone:+.6f} | {gain_map:+.6f} | {gain_alone - gain_map:+.1e} |")
    L.append("")
    L.append("EXACT: " + ("MET" if ok else "MISSED") + " (the mark asks for a largest change of exactly 0.0 and no frozen number moved).")
    return _write("\n".join(L) + "\n", out)


# ------------------------------------------------------------------ check_merge
def check_merge(body_dir: str, solo_dirs: Sequence[str], joint_dir: str, merged_dir: str,
                end_to_end_dir: Optional[str] = None, out: Optional[str] = None) -> str:
    """Members trained ALONE on a frozen body and merged by name.

    ADDITIVE: the merged model's gain over the body equals the sum of the solo gains. MERGED = JOINT: against the same
    members trained together on the same frozen body. THE PRICE OF BUILDING IN STAGES: merged against the same arrangement
    trained end to end. ``solo_dirs`` are the fine, mid and coarse solo runs, in that order."""
    e2e = load(end_to_end_dir) if end_to_end_dir else {}
    body = load(body_dir)
    solos = [load(d) for d in solo_dirs]
    joint = load(joint_dir)
    merged = load(merged_dir)
    L = ["# Members trained alone on a frozen body, then merged by name", "",
         "| seed | the body alone | + fine alone | + mid alone | + coarse alone | sum of the three solo gains | the merged model | its gain | merged gain minus the sum | "
         "the three trained together on the frozen body | merged minus together | the same arrangement trained end to end | merged minus end to end |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in sorted(merged):
        b = body[s]["loss_over_levels"]
        so = [x[s]["loss_over_levels"] for x in solos]
        gsum = sum(b - x for x in so)
        m = merged[s]["loss_over_levels"]
        j = joint[s]["loss_over_levels"] if s in joint else None
        e = e2e[s]["loss_over_levels"] if s in e2e else None
        L.append(f"| {s} | {b:.6f} | " + " | ".join(f"{x:.6f}" for x in so) + f" | {gsum:+.6f} | {m:.6f} | {b - m:+.6f} | {(b - m) - gsum:+.1e} | "
                 + ("-" if j is None else f"{j:.6f}") + " | " + ("-" if j is None else f"{m - j:+.1e}") + " | "
                 + ("-" if e is None else f"{e:.4f}") + " | " + ("-" if e is None else f"{m - e:+.4f}") + " |")
    L.append("")
    s0 = sorted(merged)[0]
    md = merged[s0]
    L.append(f"The merged model (seed {s0}): members loaded by name {md['loaded']['members_loaded']}; none new {md['loaded']['members_new']}; "
             f"all members off against the sum of the single removals {md['independence_ratio_over_levels']:.10f}; "
             f"frozen numbers unchanged in every solo run: {all(x[s]['frozen_unchanged'] for x in solos for s in x)}.")
    L.append("")
    L.append("## By noise level (seed " + str(s0) + "): the body alone, the merged model, the end-to-end model")
    L.append("| | " + " | ".join(LEVELS) + " |")
    L.append("|---|" + "---|" * len(LEVELS))
    L.append("| the body alone | " + " | ".join(f"{body[s0]['levels'][lv]['error']['all']:.4f}" for lv in LEVELS) + " |")
    L.append("| merged | " + " | ".join(f"{md['levels'][lv]['error']['all']:.4f}" for lv in LEVELS) + " |")
    if s0 in e2e:
        L.append("| end to end | " + " | ".join(f"{e2e[s0]['levels'][lv]['error']['all']:.4f}" for lv in LEVELS) + " |")
    return _write("\n".join(L) + "\n", out)


# ------------------------------------------------------------------ seed_spread
SEED_SPREAD_TITLE = "Seed spread ({n} seeds)"


def seed_spread(folders: Sequence[str], title: str = SEED_SPREAD_TITLE, out: Optional[str] = None) -> str:
    """The loss over ten noise levels per seed, the mean, the sample standard deviation and the range, the same per
    noise level, and a size class for differences: LARGE at 4 standard deviations or more, MODERATE from 2 to 4, SMALL
    under 2. ``title`` may hold ``{n}``, the number of seeds."""
    runs = {}
    for folder in folders:
        runs.update(load(folder))
    seeds = sorted(runs)
    vals = [runs[s]["loss_over_levels"] for s in seeds]
    n = len(vals)
    mean = sum(vals) / n
    sd = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5
    L = [f"# {title.format(n=n)}", "",
         "| seed | " + " | ".join(str(s) for s in seeds) + " | mean | standard deviation | range |",
         "|---|" + "---|" * (n + 3),
         "| loss over ten noise levels | " + " | ".join(f"{v:.4f}" for v in vals) + f" | {mean:.4f} | {sd:.4f} | {max(vals) - min(vals):.4f} |", ""]
    L.append("| noise level | " + " | ".join(LEVELS) + " |")
    L.append("|---|" + "---|" * len(LEVELS))
    sds = []
    for lv in LEVELS:
        x = [runs[s]["levels"][lv]["error"]["all"] for s in seeds]
        m = sum(x) / n
        sds.append((sum((v - m) ** 2 for v in x) / (n - 1)) ** 0.5)
    L.append("| standard deviation across seeds | " + " | ".join(f"{v:.4f}" for v in sds) + " |")
    L.append("")
    L.append(f"Size classes used in the reports: LARGE = {4 * sd:.4f} or more (4 standard deviations); MODERATE = {2 * sd:.4f} to {4 * sd:.4f}; "
             f"SMALL = under {2 * sd:.4f}. Most comparisons are paired (the same seed gives both arms the same starting numbers "
             "for every part they share and the same order of pictures and noise), which this unpaired spread does not measure; "
             "a small difference with the same sign on both paired seeds is called suggestive, not settled.")
    return _write("\n".join(L) + "\n", out)
