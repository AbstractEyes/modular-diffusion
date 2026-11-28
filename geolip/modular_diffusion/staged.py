"""A model of stage clusters: one whole model per stretch of the noise level, blended by the stage weights.

    v_hat(z_t, t, c) = sum_s w_s(t) v_s(z_t, t, c)            w_s = window_weight(t, cuts[s], cuts[s + 1], soft)
                     = a(z_t, t) + sum_s w_s(t) [v_s - a]      when every cluster adds the same anchor a

The weights sum to 1 at every t (the crossfade ramps of neighboring stages are complementary), so when every cluster
adds the same anchor its answer is computed once and each cluster contributes its other parts weighted by its stage
weight: a model of newborn clusters (zero-born heads) is exactly the anchor, also inside a crossfade. Cluster s runs only
on the rows whose weight w_s(t) is above zero.

A cluster is a ``Model`` with its own build; clusters may differ in width and depth but share the field, the token grid
and the class count. Units shared between clusters are the same Python objects (``share``), so they are trained once
and counted once.

A staged folder holds ``staged.json`` and one build folder per cluster:

    {"format": "modular-diffusion/staged/1", "cuts": ["0.0", "0.25", "0.65", "1.0"], "soft": "0.06",
     "clusters": ["stage_0", "stage_1", "stage_2"], "shared": {"<unit name>": "<content hash>"}, "info": {}}

``link_staged`` links every cluster folder and makes units of equal (type, name, content hash) one object again.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .build import content_hash, read_build
from .compat import num
from .linker import StaleUnit, link, stale_list
from .model import Model, draw_symbols, symbol_gated
from . import frame
from .windows import window_weight  # noqa: F401  (kept importable from here, as in earlier versions)

STAGED_FORMAT = "modular-diffusion/staged/1"
STAGED_FILE = "staged.json"


class Staged(nn.Module):
    def __init__(self, clusters: Sequence[Model], cuts: Sequence[float], soft: float = 0.0):
        super().__init__()
        clusters = list(clusters)
        cuts = [float(c) for c in cuts]
        if not clusters:
            raise ValueError("a staged model needs at least one cluster")
        if len(cuts) != len(clusters) + 1:
            raise ValueError(f"{len(clusters)} clusters need {len(clusters) + 1} cuts, not {len(cuts)}")
        if cuts[0] != 0.0 or cuts[-1] != 1.0 or any(b <= a for a, b in zip(cuts, cuts[1:])):
            raise ValueError(f"cuts must rise from 0 to 1: {cuts}")
        first = clusters[0]
        for c in clusters[1:]:
            for key in ("field_shape", "grid", "patch", "token_dim"):
                if getattr(c, key) != getattr(first, key):
                    raise ValueError(f"the clusters differ in {key}: {getattr(first, key)} against {getattr(c, key)}")
            if int(c.build.geometry["classes"]) != int(first.build.geometry["classes"]):
                raise ValueError("the clusters differ in their class count")
        adds = [c.anchor is not None and c.anchor.output for c in clusters]
        if any(adds) and not all(adds):
            raise ValueError("either every cluster adds an output anchor, or none does")
        if all(adds):
            for c in clusters[1:]:
                if not torch.equal(c.anchor.spectrum.cpu(), first.anchor.spectrum.cpu()):
                    raise ValueError("the clusters' anchors hold different spectra")
                if c.anchor.symbols != first.anchor.symbols or (first.anchor.symbols and not (
                        torch.equal(c.anchor.means.cpu(), first.anchor.means.cpu())
                        and torch.equal(c.anchor.spectra.cpu(), first.anchor.spectra.cpu()))):
                    raise ValueError("the clusters' anchors hold different class statistics")
        self.adds_anchor = bool(all(adds))
        seen: Dict[str, Any] = {}
        for c in clusters:
            for m in c.members:
                # a member of one name in several clusters: one object, or each cluster's own copy of one part (the
                # same type, address and spec, other weights; its outputs are blended by the stage weights by name)
                if m.name in seen and seen[m.name] is not m and _entry_key(seen[m.name]) != _entry_key(m):
                    raise ValueError(f"two clusters hold different members named {m.name!r}")
                seen.setdefault(m.name, m)
        self.clusters = nn.ModuleList(clusters)
        self.cuts: List[float] = cuts
        self.soft = float(soft)
        out_m, seen_m = [], set()
        for c in clusters:
            for m in c.members:
                if id(m) not in seen_m:
                    seen_m.add(id(m))
                    out_m.append(m)
        self._members = out_m
        self._stage_owners = [frame.owner(cuts[s], cuts[s + 1], self.soft, name=f"stage_{s}")
                              for s in range(len(clusters))]

    # ------------------------------------------------------------------ shapes and roles
    @property
    def field_shape(self):
        return self.clusters[0].field_shape

    @property
    def grid(self) -> int:
        return self.clusters[0].grid

    @property
    def patch(self) -> int:
        return self.clusters[0].patch

    @property
    def token_dim(self) -> int:
        return self.clusters[0].token_dim

    @property
    def anchor(self):
        return self.clusters[0].anchor

    @property
    def members(self) -> list:
        return list(self._members)

    @property
    def stage_units(self) -> list:
        return []

    @property
    def edges(self) -> list:
        return list(self.cuts[1:-1])

    def windows(self) -> List[tuple]:
        """(lo, hi) of every stage."""
        return [(self.cuts[s], self.cuts[s + 1]) for s in range(len(self.clusters))]

    # ------------------------------------------------------------------ the blend
    def weights(self, t: Tensor) -> Tensor:
        """[S, B]: the stage weights at t (the stage factors of the stages' addresses, ``frame.stage_weights``)."""
        return frame.stage_weights(self._stage_owners, t)

    def anchor_velocity(self, zt: Tensor, t: Tensor, cls_idx: Optional[Tensor] = None) -> Tensor:
        return self.clusters[0].anchor_velocity(zt, t, cls_idx)

    def components(self, zt: Tensor, t: Tensor, cls_idx: Tensor, only: Optional[Iterable[str]] = None,
                   scale: Optional[Dict[str, Tensor]] = None, skip: Iterable[str] = (),
                   symbols: Optional[Tensor] = None):
        """-> (parts {label: field [B, C, H, W]}, None, None, None).

        The anchor once (when the clusters add it and it is not skipped), then every other label summed over the
        clusters with the stage weight, each cluster run on its rows only and scattered to the full batch (zeros on the
        rows it did not run). only, scale, skip and the rows' symbols are passed to every cluster."""
        skip = set(skip)
        out: Dict[str, Tensor] = {}
        va_all = None
        if self.adds_anchor and self.anchor.name not in skip:
            va_all = self.anchor_velocity(zt, t, cls_idx)
            out[self.anchor.part] = va_all
        w = self.weights(t)
        b = zt.shape[0]
        for s, c in enumerate(self.clusters):
            ws_row = w[s]
            idx = torch.nonzero(ws_row > 0, as_tuple=False).squeeze(1)
            n_on = idx.numel()
            if n_on == 0:
                continue
            inner = skip | ({c.anchor.name} if self.adds_anchor else set())
            if n_on == b:
                kw = {} if symbols is None else {"symbols": symbols}
                if va_all is not None and c.stem.anchor_in:
                    kw["_va"] = va_all
                parts, _, _, _ = c.components(zt, t, cls_idx, only=only, scale=scale, skip=inner, **kw)
                all_one = bool((ws_row == 1.0).all())
                if all_one:
                    for label, v in parts.items():
                        out[label] = v if label not in out else out[label] + v
                else:
                    ws = ws_row.view(-1, 1, 1, 1)
                    for label, v in parts.items():
                        term = ws.to(v.dtype) * v
                        out[label] = term if label not in out else out[label] + term
            else:
                sc = None if scale is None else {k: v[idx] for k, v in scale.items()}
                kw = {} if symbols is None else {"symbols": symbols[idx]}
                if va_all is not None and c.stem.anchor_in:
                    kw["_va"] = va_all[idx]
                parts, _, _, _ = c.components(zt[idx], t[idx], cls_idx[idx], only=only, scale=sc, skip=inner, **kw)
                ws = ws_row[idx].view(-1, 1, 1, 1)
                for label, v in parts.items():
                    if label not in out:
                        out[label] = zt.new_zeros(zt.shape, dtype=v.dtype)
                    out[label] = out[label].index_add(0, idx, ws.to(v.dtype) * v)
        return out, None, None, None

    def velocity(self, zt: Tensor, t: Tensor, cls_idx: Tensor, symbols: Optional[Tensor] = None) -> Tensor:
        parts, _, _, _ = self.components(zt, t, cls_idx, symbols=symbols)
        if not parts:
            return torch.zeros_like(zt)
        return torch.stack(list(parts.values())).sum(dim=0)

    def loss(self, z1: Tensor, cls_idx: Tensor, gen: torch.Generator, cond_drop: float, mdrop: float = 0.0,
             t_sampling: str = "uniform", t_range: Sequence[float] = (0.0, 1.0), t_mix: float = 0.0) -> Tensor:
        """The flow-matching loss of one batch, with the draws of ``Model.loss`` in the same order."""
        b = z1.shape[0]
        zt, t, cls_idx, target = self.draw(z1, cls_idx, gen, cond_drop, t_sampling, t_range, t_mix)
        scale = None
        members = self.members
        if self.training and mdrop > 0 and len(members):
            scale = {m.name: (torch.rand(b, generator=gen, device=z1.device) >= mdrop).to(z1.dtype) for m in members}
        symbols = draw_symbols(cls_idx, gen) if symbol_gated(self) else None
        parts, _, _, _ = self.components(zt, t, cls_idx, scale=scale, symbols=symbols)
        v = torch.stack(list(parts.values())).sum(dim=0) if parts else torch.zeros_like(z1)
        return ((v - target) ** 2).mean()

    draw = staticmethod(Model.draw)

    def trained_numbers(self) -> int:
        """The trainable numbers, each shared parameter once."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def rotated(self, k: int = 1) -> "Staged":
        """The same clusters served to the wrong stages: cluster (s + k) mod S at stage s."""
        n = len(self.clusters)
        return Staged([self.clusters[(s + k) % n] for s in range(n)], self.cuts, self.soft)


# ------------------------------------------------------------------ sharing, files
def _entry_key(unit) -> str:
    """A unit's type, address and spec as one text (two copies of one part agree on it)."""
    return json.dumps([unit.unit_type, unit.address, unit.spec], sort_keys=True)


def _device_of(model) -> torch.device:
    return next((x.device for x in list(model.parameters()) + list(model.buffers())), torch.device("cpu"))


def share(clusters: Sequence[Model], names: Iterable[str]) -> List[Model]:
    """Clusters 1.. rebuilt so that the named units are cluster 0's objects. Each named unit must have the same content
    hash in every cluster at this moment (ValueError otherwise). Returns the new list (cluster 0 unchanged)."""
    clusters = list(clusters)
    names = list(names)
    first = clusters[0]
    want = {n: content_hash(first.unit(n)) for n in names}
    out = [first]
    for c in clusters[1:]:
        for n in names:
            if content_hash(c.unit(n)) != want[n]:
                raise ValueError(f"unit {n!r} differs between cluster 0 and another cluster; it cannot be shared")
        units = [first.unit(u.name) if u.name in want else u for u in c.units]
        out.append(Model(c.build, units).to(_device_of(c)))
    return out


def write_staged(folder: str, clusters: Sequence[str], cuts: Sequence[float], soft: float,
                 info: Optional[Dict[str, Any]] = None) -> dict:
    """Write ``staged.json`` into ``folder``. ``clusters``: the cluster build folders (stored relative to ``folder``).
    "shared" lists the units held by every cluster with one (type, content hash). Returns the dict written."""
    folder = os.path.abspath(folder)
    rels, builds = [], []
    for c in clusters:
        path = c if os.path.isabs(c) else os.path.join(folder, c)
        rels.append(os.path.relpath(os.path.abspath(path), folder).replace("\\", "/"))
        builds.append(read_build(path))
    if len(cuts) != len(rels) + 1:
        raise ValueError(f"{len(rels)} clusters need {len(rels) + 1} cuts")
    shared = {}
    if len(builds) > 1:
        for e in builds[0].units:
            same = [b.entry(e.name) for b in builds[1:] if e.name in b.names()]
            if len(same) == len(builds) - 1 and all((x.type, x.hash) == (e.type, e.hash) for x in same) and e.hash:
                shared[e.name] = e.hash
    d = {"format": STAGED_FORMAT, "cuts": [num(c) for c in cuts], "soft": num(soft), "clusters": rels, "shared": shared,
         "info": dict(info or {})}
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, STAGED_FILE), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(d, fh, indent=1)
        fh.write("\n")
    return d


def read_staged(folder: str) -> dict:
    with open(os.path.join(folder, STAGED_FILE), encoding="utf-8") as fh:
        d = json.load(fh)
    if d.get("format") != STAGED_FORMAT:
        raise ValueError(f"not a staged file of format {STAGED_FORMAT!r}: {d.get('format')!r}")
    return d


def link_staged(folder: str, device="cpu", strict: bool = True, grafts: bool = False) -> Staged:
    """The staged model of a folder: every cluster through ``linker.link``; units with equal (type, name, content hash)
    across clusters become one object. With ``strict`` a unit listed as shared must have its recorded hash in every
    cluster (StaleUnit otherwise). ``grafts``: passed to ``linker.link`` (grafted parts vouched by their graft record;
    off by default)."""
    d = read_staged(folder)
    if grafts:
        linked = [link(os.path.join(folder, rel), device=device, strict=strict, grafts=True) for rel in d["clusters"]]
    else:
        linked = [link(os.path.join(folder, rel), device=device, strict=strict) for rel in d["clusters"]]
    if strict:
        for name, h in d.get("shared", {}).items():
            for rel, m in zip(d["clusters"], linked):
                if m.link_report["hashes"].get(name) != h:
                    raise StaleUnit(name, f"listed as shared, and its hash in {rel} differs from the recorded one")
    pool: Dict[tuple, Any] = {}
    clusters = []
    for m in linked:
        hashes = m.link_report["hashes"]
        units = []
        for u in m.units:
            key = (u.unit_type, u.name, hashes[u.name])
            units.append(pool.setdefault(key, u))
        new = Model(m.build, units).to(device)
        new.link_report = m.link_report
        clusters.append(new)
    return Staged(clusters, [float(c) for c in d["cuts"]], float(d["soft"]))


def staged_stale_list(folder: str, name: str) -> Dict[str, List[str]]:
    """{cluster folder (as in staged.json): linker.stale_list of ``name`` there} for every cluster holding ``name``
    (a shared unit goes stale in every cluster)."""
    d = read_staged(folder)
    out = {}
    for rel in d["clusters"]:
        b = read_build(os.path.join(folder, rel))
        out[rel] = stale_list(b, name) if name in b.names() else []
    return out
