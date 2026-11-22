"""The build, the unit card, content hashes and unit files.

A build folder holds one build file and, per unit, one weight file and one card:

    build.json                  {"format": "modular-diffusion/build/1", "name": ...,
                                 "geometry": {"field": [C, H, W], "grid", "patch", "token_dim", "dim", "heads",
                                              "time_freq", "classes", "bands"},
                                 "units": [{"name", "type", "address", "spec", "stream", "index", "file", "hash"}, ...],
                                 "info": {...}}          descriptive fields only (never read to build the model)
    <unit>.safetensors          the unit's owned tensors, keys relative to the unit
    <unit>.card.json            the unit card (UnitCard)

Windows are stored as decimal strings and read with float(), so a stored edge never drifts by re-rendering.

Content hash of a unit: SHA-256 over the canonical JSON (sorted keys, compact separators) of {type, address, spec}, then
for each owned tensor in sorted key order its key, dtype, shape and raw little-endian CPU bytes. Owned means the unit's
parameters plus the anchor's spectrum; tables that follow from the spec (rotary tables, band and class masks, time
frequencies) are never stored and never hashed.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from torch import Tensor

BUILD_FORMAT = "modular-diffusion/build/1"
CARD_FORMAT = "modular-diffusion/unit/1"
BUILD_FILE = "build.json"
WEIGHTS_SUFFIX = ".safetensors"
CARD_SUFFIX = ".card.json"
_BAD_NAME_CHARS = set('/\\:*?"<>|')


@dataclass
class UnitEntry:
    name: str
    type: str
    address: Dict[str, Any] = field(default_factory=dict)
    spec: Dict[str, Any] = field(default_factory=dict)
    stream: Optional[str] = None          # the random stream of its birth (default: its name)
    index: int = 0                        # its position in a shared stream
    file: Optional[str] = None            # the weight file, relative to the build folder
    hash: Optional[str] = None            # the unit's content hash

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "type": self.type, "address": self.address, "spec": self.spec,
                "stream": self.stream, "index": self.index, "file": self.file, "hash": self.hash}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UnitEntry":
        return cls(name=d["name"], type=d["type"], address=dict(d.get("address", {})), spec=dict(d.get("spec", {})),
                   stream=d.get("stream"), index=int(d.get("index", 0)), file=d.get("file"), hash=d.get("hash"))


@dataclass
class Build:
    name: str
    geometry: Dict[str, Any]
    units: List[UnitEntry] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)
    format: str = BUILD_FORMAT
    folder: Optional[str] = field(default=None, compare=False, repr=False)   # where it was read from (not stored)

    def names(self) -> List[str]:
        return [u.name for u in self.units]

    def entry(self, name: str) -> UnitEntry:
        for u in self.units:
            if u.name == name:
                return u
        raise KeyError(name)

    def validate(self) -> None:
        names = self.names()
        if len(set(names)) != len(names):
            raise ValueError(f"two units share a name: {names}")

    def to_dict(self) -> Dict[str, Any]:
        return {"format": self.format, "name": self.name, "geometry": self.geometry,
                "units": [u.to_dict() for u in self.units], "info": self.info}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Build":
        if d.get("format") != BUILD_FORMAT:
            raise ValueError(f"not a build file of format {BUILD_FORMAT!r}: {d.get('format')!r}")
        return cls(name=d["name"], geometry=dict(d["geometry"]), units=[UnitEntry.from_dict(u) for u in d["units"]],
                   info=dict(d.get("info", {})), format=d["format"])


def write_build(build: Build, path: str) -> None:
    """Write the build file. The text is deterministic: the same build always gives the same bytes."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(build.to_dict(), fh, indent=1, sort_keys=False)
        fh.write("\n")


def read_build(path: str) -> Build:
    """Read a build file (or the build file of a folder). The build remembers its folder."""
    if os.path.isdir(path):
        path = os.path.join(path, BUILD_FILE)
    with open(path, encoding="utf-8") as fh:
        b = Build.from_dict(json.load(fh))
    b.folder = os.path.dirname(os.path.abspath(path))
    return b


# ------------------------------------------------------------------ hashes
def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tensor_bytes(t: Tensor) -> bytes:
    flat = t.detach().to("cpu").contiguous().reshape(-1)
    raw = flat.view(torch.uint8).numpy().tobytes() if flat.numel() else b""
    if sys.byteorder != "little" and flat.element_size() > 1:
        import numpy as np
        raw = np.frombuffer(raw, dtype=np.uint8).reshape(-1, flat.element_size())[:, ::-1].tobytes()
    return raw


def state_hash(unit_type: str, address: Mapping[str, Any], spec: Mapping[str, Any], state: Mapping[str, Tensor]) -> str:
    """The content hash of a unit given its description and owned state (see the module docstring)."""
    h = hashlib.sha256()
    h.update(canonical_json({"type": unit_type, "address": address, "spec": spec}).encode("utf-8"))
    for k in sorted(state):
        t = state[k]
        h.update(b"\0" + k.encode("utf-8") + b"\0" + str(t.dtype).encode("ascii") + b"\0"
                 + canonical_json(list(t.shape)).encode("ascii") + b"\0")
        h.update(_tensor_bytes(t))
    return h.hexdigest()


def content_hash(unit) -> str:
    """The content hash of a unit module (its type, address, spec and owned tensors)."""
    return state_hash(unit.unit_type, unit.address, unit.spec, unit.owned_state())


def parent_hash(parents: Mapping[str, str]) -> str:
    """SHA-256 of the canonical JSON of a {parent name: parent content hash} map."""
    return sha256_text(canonical_json(dict(parents)))


# ------------------------------------------------------------------ the unit card
BIRTH_STATES = ("random", "zero_head", "identity", "closed_form")
BIRTH_STATE_OF = {"Anchor": "closed_form", "Stem": "random", "StageBody": "random", "GrowthBlock": "identity",
                  "Head": "zero_head", "BandMember": "zero_head", "ContentExpert": "zero_head"}
MARK_KEYS = ("born_gap", "frozen_unchanged", "off_window_max", "off_band_share", "loss_over_levels")


@dataclass
class UnitCard:
    """What a unit is, where it came from and what it was fit with.

    parents: {name: content hash} of every other unit in its computation path while it was fit (frozen or co-trained);
    birth: {seed, stream, index, state (random / zero_head / identity / closed_form), hash (the newborn's hash)};
    parameters: {count, tensors {key: shape}}; recipe: the optimizer and the schedule; data: the training data;
    marks: {born_gap, frozen_unchanged, off_window_max, off_band_share, loss_over_levels} (None = not measured).
    geometry is the build geometry the unit was made for."""
    name: str
    type: str
    address: Dict[str, Any] = field(default_factory=dict)
    spec: Dict[str, Any] = field(default_factory=dict)
    hash: Optional[str] = None
    parents: Dict[str, str] = field(default_factory=dict)
    parent_hash: Optional[str] = None
    birth: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    recipe: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    license: Optional[str] = None
    marks: Dict[str, Any] = field(default_factory=lambda: {k: None for k in MARK_KEYS})
    geometry: Dict[str, Any] = field(default_factory=dict)
    format: str = CARD_FORMAT

    def to_dict(self) -> Dict[str, Any]:
        return {"format": self.format, "name": self.name, "type": self.type, "address": self.address, "spec": self.spec,
                "hash": self.hash, "parents": dict(self.parents), "parent_hash": self.parent_hash, "birth": self.birth,
                "parameters": self.parameters, "recipe": self.recipe, "data": self.data, "license": self.license,
                "marks": self.marks, "geometry": self.geometry}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UnitCard":
        if d.get("format") != CARD_FORMAT:
            raise ValueError(f"not a unit card of format {CARD_FORMAT!r}: {d.get('format')!r}")
        return cls(name=d["name"], type=d["type"], address=dict(d.get("address", {})), spec=dict(d.get("spec", {})),
                   hash=d.get("hash"), parents=dict(d.get("parents", {})), parent_hash=d.get("parent_hash"),
                   birth=dict(d.get("birth", {})), parameters=dict(d.get("parameters", {})),
                   recipe=dict(d.get("recipe", {})), data=dict(d.get("data", {})), license=d.get("license"),
                   marks=dict(d.get("marks", {})), geometry=dict(d.get("geometry", {})), format=d["format"])

    @property
    def zero_born(self) -> bool:
        return self.birth.get("state") in ("zero_head", "identity")


def write_card(card: UnitCard, path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(card.to_dict(), fh, indent=1, sort_keys=False)
        fh.write("\n")


def read_card(path: str) -> UnitCard:
    with open(path, encoding="utf-8") as fh:
        return UnitCard.from_dict(json.load(fh))


def check_name(name: str) -> str:
    """A unit name is used as a file name: no path separators or reserved characters."""
    if not name or name in (".", "..") or any(c in _BAD_NAME_CHARS for c in name) or name != name.strip():
        raise ValueError(f"a unit name must be usable as a file name: {name!r}")
    return name


def unit_files(folder: str, name: str) -> Tuple[str, str]:
    """(weight file, card file) of a unit in a folder."""
    check_name(name)
    return os.path.join(folder, name + WEIGHTS_SUFFIX), os.path.join(folder, name + CARD_SUFFIX)


def save_unit(unit, card: UnitCard, folder: str) -> UnitCard:
    """Write a unit's weight file and card into ``folder``. The card's name, type, address, spec, hash, parameters and
    parent_hash are filled in from the unit; returns the card written."""
    from safetensors.torch import save_file
    os.makedirs(folder, exist_ok=True)
    wpath, cpath = unit_files(folder, unit.name)
    state = {k: v.detach().to("cpu").contiguous().clone() for k, v in unit.owned_state().items()}
    card.name, card.type = unit.name, unit.unit_type
    card.address, card.spec = dict(unit.address), dict(unit.spec)
    card.hash = state_hash(unit.unit_type, unit.address, unit.spec, state)
    card.parameters = {"count": int(sum(v.numel() for v in state.values())),
                       "tensors": {k: list(state[k].shape) for k in sorted(state)}}
    card.parent_hash = parent_hash(card.parents)
    if not card.geometry:
        card.geometry = dict(unit.geometry)
    save_file(state, wpath, metadata={"format": CARD_FORMAT, "name": unit.name, "type": unit.unit_type,
                                      "hash": card.hash})
    write_card(card, cpath)
    return card


def read_weights(path: str) -> Dict[str, Tensor]:
    """A weight file read into memory (no memory map is left open, so the file can be replaced or removed at once)."""
    from safetensors.torch import load
    with open(path, "rb") as fh:
        return load(fh.read())


def load_state(folder: str, name: str) -> Dict[str, Tensor]:
    wpath, _ = unit_files(folder, name)
    return read_weights(wpath)


def load_unit(folder: str, name: str, device="cpu", geometry: Optional[Dict[str, Any]] = None):
    """-> (unit, card). The unit is built from its card (and ``geometry``, by default the card's) and loaded strictly."""
    from .units import make
    _, cpath = unit_files(folder, name)
    card = read_card(cpath)
    g = geometry if geometry is not None else card.geometry
    if not g:
        raise ValueError(f"the card of {name!r} records no geometry; pass geometry=")
    unit = make(card.type, card.name, card.address, card.spec, g)
    unit.load_owned(load_state(folder, name))
    return unit.to(device), card
