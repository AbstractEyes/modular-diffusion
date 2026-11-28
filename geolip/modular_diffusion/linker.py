"""The linker: a model assembled from a build folder, units snapped in and out, and what goes stale.

Rules.

1. Content hash: see ``build``. A unit file whose hash differs from its build entry or from its card is refused.
2. Parents. A unit's card records, as {name: content hash}, every other unit in its computation path while it was fit,
   frozen or co-trained:
     - upstream units whose window of the noise level overlaps its own (the anchor is upstream only when the stem reads
       it; stage units of overlapping windows count as each other's upstream, since they share one mix);
     - for a trunk unit (stem, stage body, growth block, body), also the downstream units its loss passed through;
     - for an output unit (anchor answer, head, band member, content expert), also the output units whose band, window
       and classes overlap its own.
   ``parent_hash`` is the SHA-256 of the canonical parents map.
3. Stale. Two interacting units (either one is a parent of the other by rule 2) are vouched when at least one card records
   the other's current hash; a unit still at its birth state (its hash equals its birth hash) vouches trivially, since it
   was never fit against anything. A unit is stale when a recorded parent is absent from the build, or it sits in an
   unvouched interacting pair.
4. ``link`` loads each unit file by name, recomputes its hash against the build entry and the card, then checks the
   geometry, rule 3 and the marks (born_gap == 0.0 for a zero-born unit when measured; frozen_unchanged when the recipe
   froze anything). Any failure raises StaleUnit with the unit and the reason.
5. ``snap_in`` copies a unit file and its card into the build folder, appends the unit and links. For a newborn (still at
   its birth state) it measures born_gap on a probe batch and requires 0.0.
6. ``snap_out`` removes the unit from the build and deletes its files from the folder (absence, never a multiplication by
   zero). When no remaining card lists the unit, the folder is then exactly what it was before the unit's snap_in;
   otherwise it raises with the stale list unless ``force=True``.
7. ``stale_list(build, name)`` is the transitive closure of "lists ``name`` as a parent", in build order.

Conversion: ``export_build`` writes a model as a build folder (one weight file and one card per unit); ``from_legacy``
does the same for a flat weight file with its result JSON beside it.
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import torch
from torch import Tensor

from .build import (BIRTH_STATE_OF, BUILD_FILE, CARD_SUFFIX, MARK_KEYS, WEIGHTS_SUFFIX, Build, UnitCard, UnitEntry,
                    canonical_json, content_hash, read_build, read_card, read_weights, save_unit, state_hash, unit_files,
                    write_build, write_card)
from .model import Model
from .units import born, make

TRUNK = ("Stem", "StageBody", "GrowthBlock", "Body", "DepthBlock")
STAGE = ("StageBody", "GrowthBlock", "Body", "DepthBlock")       # trunk units after the stem: each other's parents
OUTPUT_UNITS = ("Head", "BandMember", "ContentExpert")
_POS = {"Anchor": 0, "Stem": 1, "StageBody": 2, "GrowthBlock": 2, "Body": 2, "DepthBlock": 2, "Head": 3, "BandMember": 3,
        "ContentExpert": 3}


class StaleUnit(Exception):
    """A unit that cannot be linked (or removed) as it stands: ``unit``, ``reason`` and, where it applies, the units
    that list it (``stale``)."""

    def __init__(self, unit: str, reason: str, stale: Sequence[str] = ()):
        self.unit, self.reason, self.stale = unit, reason, list(stale)
        super().__init__(f"{unit}: {reason}" + (f"; stale: {self.stale}" if self.stale else ""))


# ------------------------------------------------------------------ rule 2: who interacts with whom
def _window(e: UnitEntry) -> Tuple[float, float]:
    lo, hi = e.address.get("window", ["0", "1"])
    soft = float(e.address.get("soft", "0") or 0)
    return float(lo) - soft, float(hi) + soft


def _overlap(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _band_overlap(a: UnitEntry, b: UnitEntry) -> bool:
    x, y = a.address.get("band", "all"), b.address.get("band", "all")
    return x == "all" or y == "all" or x == y


def _class_overlap(a: UnitEntry, b: UnitEntry) -> bool:
    if a.spec.get("gate", "class") != b.spec.get("gate", "class"):
        return True                                       # a symbol and a class gate can be on for the same row
    x, y = a.address.get("classes"), b.address.get("classes")
    return x is None or y is None or bool(set(int(c) for c in x) & set(int(c) for c in y))


def _is_output(e: UnitEntry) -> bool:
    return e.type in OUTPUT_UNITS or (e.type == "Anchor" and bool(e.spec.get("output", True)))


def recorded_parents(build: Build, name: str) -> List[str]:
    """Rule 2: the names a unit's card records as its parents when it is fit in this build (build order)."""
    me = build.entry(name)
    if me.type == "Anchor":
        return []                                                # closed form: fit against nothing
    stem = next((e for e in build.units if e.type == "Stem"), None)
    stem_reads_anchor = stem is not None and int(stem.spec.get("anchor_in", 0)) > 0
    out = []
    for e in build.units:
        if e.name == name:
            continue
        win = _overlap(_window(me), _window(e))
        p_me, p_e = _POS[me.type], _POS[e.type]
        hit = False
        if e.type == "Anchor":
            hit = stem_reads_anchor or (_is_output(me) and _is_output(e))
        elif p_e < p_me and win:                                  # upstream
            hit = True
        elif me.type in STAGE and e.type in STAGE and win:        # stage units in one mix
            hit = True
        elif me.type in TRUNK and p_e > p_me and win:            # downstream of a trunk unit
            hit = True
        elif _is_output(me) and _is_output(e) and win and _band_overlap(me, e) and _class_overlap(me, e):
            hit = True
        if hit:
            out.append(e.name)
    return out


def interacting_pairs(build: Build) -> List[Tuple[str, str]]:
    """Every unordered pair that interacts by rule 2 (in build order)."""
    rel = {e.name: set(recorded_parents(build, e.name)) for e in build.units}
    names = build.names()
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if b in rel[a] or a in rel[b]:
                pairs.append((a, b))
    return pairs


# ------------------------------------------------------------------ rule 3 and the marks
def _at_birth(card: UnitCard, h: str) -> bool:
    return bool(card.birth.get("hash")) and card.birth.get("hash") == h


def grafted_onto(card: UnitCard) -> Dict[str, str]:
    """The units a grafted part was put onto, as {name: content hash} (its card's recipe "graft" "onto"; empty for a
    unit that is no graft). Its recorded parents stay the units it was trained with."""
    g = (card.recipe or {}).get("graft") or {}
    return dict(g.get("onto") or {})


def check(build: Build, cards: Dict[str, UnitCard], hashes: Dict[str, str], grafts: bool = False) -> Dict[str, List[str]]:
    """Rule 3 and the marks -> {unit: [reasons]} (empty when the build links). With ``grafts`` an interacting pair is
    also vouched when one card is a graft's and records the other's current hash as a unit it was put onto
    (``grafted_onto``); off by default."""
    bad: Dict[str, List[str]] = {}
    names = set(build.names())
    for e in build.units:
        missing = [p for p in cards[e.name].parents if p not in names]
        if missing:
            bad.setdefault(e.name, []).append(f"recorded parents absent from the build: {missing}")
    for a, b in interacting_pairs(build):
        ca, cb = cards[a], cards[b]
        ok = (ca.parents.get(b) == hashes[b] or cb.parents.get(a) == hashes[a]
              or _at_birth(ca, hashes[a]) or _at_birth(cb, hashes[b]))
        if not ok and grafts:
            ok = grafted_onto(ca).get(b) == hashes[b] or grafted_onto(cb).get(a) == hashes[a]
        if not ok:
            for x, y in ((a, b), (b, a)):
                bad.setdefault(x, []).append(f"interacts with {y!r} and neither card records the other's current hash")
    for e in build.units:
        c = cards[e.name]
        gap = c.marks.get("born_gap")
        if c.zero_born and gap is not None and float(gap) != 0.0:
            bad.setdefault(e.name, []).append(f"born_gap is {gap}, not 0.0")
        if c.recipe.get("frozen") and c.marks.get("frozen_unchanged") is not True:
            bad.setdefault(e.name, []).append("the recipe froze units and frozen_unchanged is not True")
    return bad


def _open(build: Union[str, Build]) -> Build:
    if isinstance(build, Build):
        if not build.folder:
            raise ValueError("this build has no folder; read it with read_build(path)")
        return build
    return read_build(build)


def _cards(build: Build) -> Dict[str, UnitCard]:
    return {e.name: read_card(unit_files(build.folder, e.name)[1]) for e in build.units}


def _weights_path(build: Build, e: UnitEntry) -> str:
    return os.path.join(build.folder, e.file) if e.file else unit_files(build.folder, e.name)[0]


# ------------------------------------------------------------------ rule 4: link
def link(build_path: Union[str, Build], device="cpu", strict: bool = True, grafts: bool = False):
    """The model of a build folder. Every file is checked against its build entry and card (always); rule 3 and the
    marks raise StaleUnit when ``strict``, and are reported in ``model.link_report`` either way. ``grafts``: a grafted
    part (trained on another parent, put onto this one; ``check``) is vouched by its graft record; off by default."""
    b = _open(build_path)
    b.validate()
    units, cards, hashes = [], {}, {}
    for e in b.units:
        cpath = unit_files(b.folder, e.name)[1]
        if not os.path.exists(cpath):
            raise StaleUnit(e.name, "its card is missing")
        card = read_card(cpath)
        if (card.name, card.type) != (e.name, e.type) or canonical_json([card.address, card.spec]) != canonical_json([e.address, e.spec]):
            raise StaleUnit(e.name, "its card does not describe the build entry (name, type, address or spec)")
        if card.geometry and canonical_json(card.geometry) != canonical_json(b.geometry):
            raise StaleUnit(e.name, f"made for another geometry: {card.geometry} against {b.geometry}")
        wpath = _weights_path(b, e)
        if not os.path.exists(wpath):
            raise StaleUnit(e.name, "its weight file is missing")
        state = read_weights(wpath)
        h = state_hash(e.type, e.address, e.spec, state)
        if e.hash is not None and h != e.hash:
            raise StaleUnit(e.name, "its weight file's content hash differs from the build entry")
        if h != card.hash:
            raise StaleUnit(e.name, "its weight file's content hash differs from its card")
        unit = make(e.type, e.name, e.address, e.spec, b.geometry)
        try:
            unit.load_owned(state)
        except RuntimeError as err:
            raise StaleUnit(e.name, f"its tensors do not fit the geometry: {err}") from None
        units.append(unit)
        cards[e.name], hashes[e.name] = card, h
    bad = check(b, cards, hashes, grafts=True) if grafts else check(b, cards, hashes)
    if bad and strict:
        first = next(n for n in b.names() if n in bad)
        raise StaleUnit(first, "; ".join(bad[first]), [n for n in b.names() if n in bad])
    model = Model(b, units).to(device)
    model.link_report = {"hashes": hashes, "problems": bad,
                         "unmeasured_born_gap": [n for n, c in cards.items() if c.zero_born and c.marks.get("born_gap") is None]}
    return model


# ------------------------------------------------------------------ rule 7, 5, 6
def stale_list(build: Union[str, Build], name: str) -> List[str]:
    """The units that list ``name`` as a parent, directly or through others (build order; ``name`` excluded)."""
    b = _open(build)
    cards = _cards(b)
    hit: Set[str] = {name}
    grew = True
    while grew:
        grew = False
        for e in b.units:
            if e.name not in hit and any(p in hit for p in cards[e.name].parents):
                hit.add(e.name)
                grew = True
    return [n for n in b.names() if n in hit and n != name]


def probe_batch(geometry: Dict[str, Any], device="cpu", seed: int = 20251121):
    """A fixed batch for silence checks: noise levels on both sides of every common window edge and its crossfade, every
    class and the null class."""
    t = torch.tensor([0.0, 0.03, 0.1, 0.19, 0.2, 0.22, 0.25, 0.28, 0.31, 0.4, 0.5, 0.59, 0.62, 0.65, 0.68, 0.71,
                      0.8, 0.9, 0.97, 1.0])
    n = t.shape[0]
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn((n,) + tuple(int(x) for x in geometry["field"]), generator=gen)
    cls = torch.arange(n) % int(geometry["classes"])
    return z.to(device), t.to(device), cls.to(device)


def _velocity(model, probe) -> Tensor:
    """The velocity on the probe with the parts added one at a time, in the model's own order. A part that is exactly
    zero then changes no digit on any device; a stacked sum may pair the other parts differently when one more part is
    present, which moves the last digit on some devices although the added part is silent."""
    from .model import symbol_gated
    with torch.no_grad():
        # a model with symbol-gated units reads the probe's classes as the rows' symbols too, so those units are on
        kw = {"symbols": probe[2]} if symbol_gated(model) else {}
        parts, _, _, _ = model.components(*probe, **kw)
        v = torch.zeros_like(probe[0])
        for p in parts.values():
            v = v + p
        return v


def _device_of(model) -> torch.device:
    return next((x.device for x in list(model.parameters()) + list(model.buffers())), torch.device("cpu"))


def born_gap(with_unit, without_unit) -> float:
    """The largest change of the velocity on the probe batch when a unit is added (the probe is drawn on the CPU and
    moved to the models' device, so its values do not depend on the device)."""
    probe = probe_batch(with_unit.build.geometry, device=_device_of(with_unit))
    return float((_velocity(with_unit, probe) - _velocity(without_unit, probe)).abs().max())


def _without(build: Build, name: str) -> Build:
    return Build(name=build.name, geometry=build.geometry, units=[e for e in build.units if e.name != name],
                 info=build.info, format=build.format, folder=build.folder)


def snap_in(build: Union[str, Build], card_path: str) -> Build:
    """Copy a unit (its card and the weight file beside it) into the build folder, append it and link. A newborn must
    leave the probe batch's velocity exactly unchanged. On any failure the folder is put back as it was."""
    b = _open(build)
    card = read_card(card_path)
    name = card.name
    if name in b.names():
        raise ValueError(f"the build already holds a unit named {name!r}")
    src_w = os.path.join(os.path.dirname(os.path.abspath(card_path)), name + WEIGHTS_SUFFIX)
    dst_w, dst_c = unit_files(b.folder, name)
    build_file = os.path.join(b.folder, BUILD_FILE)
    with open(build_file, "rb") as fh:
        before = fh.read()
    copied = []
    try:
        for src, dst in ((src_w, dst_w), (card_path, dst_c)):
            if os.path.abspath(src) != os.path.abspath(dst):
                if os.path.exists(dst):
                    raise FileExistsError(f"{dst} exists and is not part of the build")
                shutil.copyfile(src, dst)
                copied.append(dst)
        entry = UnitEntry(name, card.type, dict(card.address), dict(card.spec), stream=card.birth.get("stream"),
                          index=int(card.birth.get("index", 0) or 0), file=name + WEIGHTS_SUFFIX, hash=card.hash)
        new = Build(name=b.name, geometry=b.geometry, units=list(b.units) + [entry], info=b.info, format=b.format,
                    folder=b.folder)
        write_build(new, build_file)
        if card.birth.get("hash") and card.birth.get("hash") == card.hash:          # a newborn
            m_new = link(new, strict=False)
            m_old = Model(b, list(m_new.units)[:-1])
            gap = born_gap(m_new, m_old)
            if gap != 0.0:
                raise StaleUnit(name, f"a newborn must change nothing; born_gap {gap}")
            if card.marks.get("born_gap") is None and dst_c in copied:
                card.marks["born_gap"] = gap
                write_card(card, dst_c)
            bad = m_new.link_report["problems"]
            if bad:
                first = next(n for n in new.names() if n in bad)
                raise StaleUnit(first, "; ".join(bad[first]), [n for n in new.names() if n in bad])
        else:
            link(new, strict=True)
        return new
    except BaseException:
        with open(build_file, "wb") as fh:
            fh.write(before)
        for p in copied:
            if os.path.exists(p):
                os.remove(p)
        raise


def snap_out(build: Union[str, Build], name: str, force: bool = False) -> Build:
    """Remove a unit from the build and delete its files from the folder. Raises StaleUnit with the stale list when
    other cards list it, unless ``force``."""
    b = _open(build)
    e = b.entry(name)
    stale = stale_list(b, name)
    if stale and not force:
        raise StaleUnit(name, "other units list it as a parent", stale)
    new = _without(b, name)
    write_build(new, os.path.join(b.folder, BUILD_FILE))
    for p in (_weights_path(b, e), unit_files(b.folder, name)[1]):
        if os.path.exists(p):
            os.remove(p)
    return new


# ------------------------------------------------------------------ conversion
def _same_as_base(base: Optional[Build], name: str, h: str) -> Optional[str]:
    if base is None or name not in base.names():
        return None
    _, cpath = unit_files(base.folder, name)
    if os.path.exists(cpath) and read_card(cpath).hash == h:
        return cpath
    return None


def export_build(model, folder: str, seed: Optional[int] = None, base: Union[None, str, Build] = None,
                 trained: Optional[Iterable[str]] = None, recipe: Optional[Dict[str, Any]] = None,
                 data: Optional[Dict[str, Any]] = None, marks: Optional[Dict[str, Any]] = None,
                 license: Optional[str] = None, name: Optional[str] = None) -> Build:
    """Write a model as a build folder: build.json plus one weight file and one card per unit.

    seed: the run seed of the units' birth (it fills each card's birth hash and, for zero-born units, the born_gap mark).
    base: a build folder the model was loaded from; a unit whose content hash equals the base card's keeps that card
    (the later fit vouches; earlier cards are never rewritten). trained: the units fit in this run (default: every unit
    but the anchor). A unit that was not trained and has no base card is assumed to have been fit together with the
    other untrained units. recipe, data, marks: copied into the new cards (marks such as loss_over_levels and
    frozen_unchanged describe this run)."""
    b = model.build
    base_b = _open(base) if base is not None else None
    os.makedirs(folder, exist_ok=True)
    names = b.names()
    trained = set(names) - {u.name for u in model.units if u.unit_type == "Anchor"} if trained is None else set(trained)
    hashes = {u.name: content_hash(u) for u in model.units}
    entries = []
    for u, e in zip(model.units, b.units):
        h = hashes[u.name]
        dst_w, dst_c = unit_files(folder, u.name)
        keep = _same_as_base(base_b, u.name, h)
        if keep:
            src_w = _weights_path(base_b, base_b.entry(u.name))
            if os.path.abspath(src_w) != os.path.abspath(dst_w):
                shutil.copyfile(src_w, dst_w)
                shutil.copyfile(keep, dst_c)
        else:
            rel = recorded_parents(b, u.name)
            if u.name in trained:
                parents = rel
            else:
                parents = [p for p in rel if p not in trained]
            held = [p for p in rel if p not in trained and b.entry(p).type != "Anchor"]     # the anchor has no weights
            frozen = held if u.name in trained else []
            card = UnitCard(name=u.name, type=u.unit_type, parents={p: hashes[p] for p in parents},
                            geometry=dict(b.geometry), license=license)
            card.birth = {"seed": seed, "stream": e.stream or e.name, "index": e.index,
                          "state": BIRTH_STATE_OF.get(u.unit_type, "random"), "hash": None}
            newborn = None
            if seed is not None and u.unit_type != "Anchor":
                newborn = born(u.unit_type, seed, u.name, u.address, u.spec, b.geometry, stream=e.stream, index=e.index)
                card.birth["hash"] = content_hash(newborn)
            elif u.unit_type == "Anchor":
                card.birth["hash"] = h
            card.recipe = dict(recipe or {})
            if u.unit_type == "Anchor":
                card.recipe = {"optimizer": None, "fit": "closed form from the measured spectrum"}
            card.recipe["trained_with"] = sorted(p for p in rel if p in trained) if u.name in trained else []
            card.recipe["frozen"] = frozen
            card.data = dict(data or {})
            card.marks = {k: None for k in MARK_KEYS}
            if u.name in trained:
                card.marks.update({k: v for k, v in (marks or {}).items() if k in MARK_KEYS})
                if not frozen:
                    card.marks["frozen_unchanged"] = None
            if card.zero_born and newborn is not None:
                dev = _device_of(model)
                newborn = newborn.to(dev)
                units = [newborn if x.name == u.name else x for x in model.units]
                card.marks["born_gap"] = born_gap(Model(b, units).to(dev),
                                                  Model(_without(b, u.name), [x for x in model.units if x.name != u.name]).to(dev))
            save_unit(u, card, folder, _hash=h)
        entries.append(UnitEntry(e.name, e.type, dict(e.address), dict(e.spec), stream=e.stream, index=e.index,
                                 file=u.name + WEIGHTS_SUFFIX, hash=h))
    out = Build(name=name or b.name, geometry=dict(b.geometry), units=entries, info=dict(b.info), folder=os.path.abspath(folder))
    write_build(out, os.path.join(folder, BUILD_FILE))
    return out


def from_legacy(pt_path: str, folder: str, spectrum: Tensor, json_path: Optional[str] = None,
                base: Union[None, str, Build] = None, data: Optional[Dict[str, Any]] = None,
                license: Optional[str] = None) -> Build:
    """Convert a flat weight file (with its result JSON, by default ``<file>.json``) into a build folder. The spectrum
    (which a flat file does not hold) comes from the data. The units trained in that run follow from its ``train`` and
    ``init`` arguments; its read gives the marks."""
    from . import compat
    model, res = compat.units_from_legacy(pt_path, json_path, spectrum)
    a = res["args"]
    trained = {n for n, p in _trainable_units(model)}
    marks = {"loss_over_levels": res.get("loss_over_levels"), "frozen_unchanged": res.get("frozen_unchanged")}
    return export_build(model, folder, seed=int(res["seed"]), base=base, trained=trained,
                        recipe=compat.recipe_from_result(res), data=dict(compat.data_from_result(res), **(data or {})),
                        marks=marks, license=license, name=compat.describe(compat.full_args(a)))


def _trainable_units(model):
    for u in model.units:
        if any(p.requires_grad for p in u.parameters()):
            yield u.name, u
