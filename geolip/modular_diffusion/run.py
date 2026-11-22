"""The entry point of the experiments in ``geolip.modular_diffusion.experiments``.

    python -m geolip.modular_diffusion.run time_split variant=split seed=0 data=<npz> out=<folder> device=cuda
    from geolip.modular_diffusion import run; run.main("time_split", variant="split", seed=0, data=..., out=..., device="cpu")

``main(experiment, argv, **overrides)`` imports ``geolip.modular_diffusion.experiments.<experiment>`` and calls its
``main(argv, **overrides)`` (``key=value`` items and/or keywords, read with ``config.parse``).

The experiments (``EXPERIMENTS``): time_split (what a stage costs when it shares nothing), chain_reads (reads on saved
stage models), content (where and how the class enters; class experts; a class snapped in), closed_forms (heads solved
by least squares; units trained on stored states), stages (K-stage models that continue one run, tails, growth) and
load (a CPU load beside another job).

``det=on`` (any experiment): the matrix library's fixed workspace (``determinism``) is put into the process environment
here, before the experiment module is imported and before anything touches a card.
"""
from __future__ import annotations

import importlib
import sys
from typing import Any, Iterable

from .config import in_notebook

EXPERIMENTS_PACKAGE = "geolip.modular_diffusion.experiments"
EXPERIMENTS = ("time_split", "chain_reads", "content", "closed_forms", "stages", "load")


def _det_value(argv: Iterable[str], overrides: dict) -> str:
    value = ""
    for item in argv:
        if str(item).startswith("det="):
            value = str(item)[4:]
    return str(overrides.get("det", value) or "")


def main(experiment: str, argv: Iterable[str] = (), **overrides: Any):
    if not experiment.isidentifier():
        raise ValueError(f"not an experiment name: {experiment!r}")
    argv = list(argv)
    if _det_value(argv, overrides) == "on":
        from .determinism import set_workspace
        set_workspace()
    try:
        mod = importlib.import_module(f"{EXPERIMENTS_PACKAGE}.{experiment}")
    except ModuleNotFoundError as e:
        if e.name == f"{EXPERIMENTS_PACKAGE}.{experiment}":
            raise ValueError(f"no experiment named {experiment!r} in {EXPERIMENTS_PACKAGE}; known: "
                             f"{list(EXPERIMENTS)}") from None
        raise
    return mod.main(argv, **overrides)


if __name__ == "__main__" and not in_notebook():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m geolip.modular_diffusion.run <experiment> key=value ...")
    main(sys.argv[1], sys.argv[2:])
