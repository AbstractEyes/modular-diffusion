"""The entry point of the experiments in ``geolip.modular_diffusion.experiments``.

    python -m geolip.modular_diffusion.run time_split variant=split seed=0 data=<npz> out=<folder> device=cuda
    from geolip.modular_diffusion import run; run.main("time_split", variant="split", seed=0, data=..., out=..., device="cpu")

``main(experiment, argv, **overrides)`` imports ``geolip.modular_diffusion.experiments.<experiment>`` and calls its
``main(argv, **overrides)`` (``key=value`` items and/or keywords, read with ``config.parse``).
"""
from __future__ import annotations

import importlib
import sys
from typing import Any, Iterable

from .config import in_notebook

EXPERIMENTS_PACKAGE = "geolip.modular_diffusion.experiments"


def main(experiment: str, argv: Iterable[str] = (), **overrides: Any):
    if not experiment.isidentifier():
        raise ValueError(f"not an experiment name: {experiment!r}")
    try:
        mod = importlib.import_module(f"{EXPERIMENTS_PACKAGE}.{experiment}")
    except ModuleNotFoundError as e:
        if e.name == f"{EXPERIMENTS_PACKAGE}.{experiment}":
            raise ValueError(f"no experiment named {experiment!r} in {EXPERIMENTS_PACKAGE}") from None
        raise
    return mod.main(list(argv), **overrides)


if __name__ == "__main__" and not in_notebook():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m geolip.modular_diffusion.run <experiment> key=value ...")
    main(sys.argv[1], sys.argv[2:])
