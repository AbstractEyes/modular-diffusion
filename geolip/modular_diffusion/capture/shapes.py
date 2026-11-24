"""Engine shapes: which blocks of the teacher an engine keeps. A shape is declared, never learned.

The teacher's U-Net has four stages; a down stage holds two layers (a residual block, with an attention block in the
three outer stages) and its mirrored up stage holds three. A shape keeps, per down stage i, ``layers[i]`` of its two
layers (1 or 2), and the mirrored up stage keeps ``layers[i] + 1`` of its three, so blocks are always removed in
mirrored pairs. ``mid`` keeps or drops the middle block; ``stages`` 3 drops the innermost down stage with its mirrored
up stage (``layers[3]`` is then not used).

    full   (2, 2, 2, 2), mid, 4 stages      the teacher's own shape
    base   (1, 1, 1, 1), mid, 4 stages
    small  (1, 1, 1, 1), no mid, 4 stages
    tiny   (1, 1, 1, 1), no mid, 3 stages

A shape is written in text as a name or as ``<layers>/<mid|nomid>/<stages>``, for example ``1211/mid/4``.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import List, Tuple, Union


@dataclass(frozen=True)
class Shape:
    layers: Tuple[int, int, int, int] = (2, 2, 2, 2)
    mid: bool = True
    stages: int = 4

    def __post_init__(self):
        layers = tuple(int(n) for n in self.layers)
        if len(layers) != 4 or any(n not in (1, 2) for n in layers):
            raise ValueError(f"layers are four counts, each 1 or 2: {self.layers}")
        if int(self.stages) not in (3, 4):
            raise ValueError(f"stages is 4 or 3, not {self.stages}")
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "mid", bool(self.mid))
        object.__setattr__(self, "stages", int(self.stages))

    @property
    def kept(self) -> Tuple[int, ...]:
        """The layer counts of the kept down stages (the first ``stages`` entries)."""
        return self.layers[: self.stages]

    @property
    def text(self) -> str:
        return "".join(str(n) for n in self.layers) + ("/mid/" if self.mid else "/nomid/") + str(self.stages)

    @property
    def name(self) -> str:
        """The named shape this is (full, base, small, tiny), or its text form."""
        for k, v in SHAPES.items():
            if v.layers == self.layers and v.mid == self.mid and v.stages == self.stages:
                return k
        return self.text

    @staticmethod
    def parse(spec: str) -> "Shape":
        """A name of ``SHAPES`` or the text form ``<layers>/<mid|nomid>/<stages>``."""
        spec = str(spec).strip()
        if spec in SHAPES:
            return SHAPES[spec]
        parts = spec.split("/")
        if len(parts) != 3 or len(parts[0]) != 4 or parts[1] not in ("mid", "nomid"):
            raise ValueError(f"a shape is a name {sorted(SHAPES)} or <layers>/<mid|nomid>/<stages>, not {spec!r}")
        return Shape(tuple(int(c) for c in parts[0]), parts[1] == "mid", int(parts[2]))


SHAPES = {
    "full": Shape((2, 2, 2, 2), True, 4),
    "base": Shape((1, 1, 1, 1), True, 4),
    "small": Shape((1, 1, 1, 1), False, 4),
    "tiny": Shape((1, 1, 1, 1), False, 3),
}


def named_shape(name: str) -> Shape:
    if name not in SHAPES:
        raise ValueError(f"unknown shape {name!r}; known: {sorted(SHAPES)}")
    return SHAPES[name]


def shape_from(spec: Union[str, Shape]) -> Shape:
    return spec if isinstance(spec, Shape) else Shape.parse(spec)


def all_shapes() -> List[Shape]:
    """Every declared shape: the 16 layer tuples, with and without the middle block, with 4 and 3 stages (64)."""
    return [Shape(layers, mid, stages) for layers in itertools.product((1, 2), repeat=4)
            for mid in (True, False) for stages in (4, 3)]
