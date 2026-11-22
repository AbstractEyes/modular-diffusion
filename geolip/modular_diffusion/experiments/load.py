"""A CPU load: keeps a number of CPU threads busy with float32 matrix products for a number of seconds, so that another
job can be measured beside a known load. It never touches a card.

    python -m geolip.modular_diffusion.run load threads=48 seconds=600 out=<folder>
    from geolip.modular_diffusion.experiments import load; load.main(threads=2, seconds=2, out=...)

Arguments (key=value): threads (the intra-op threads of the products), seconds, out, size (the matrices' side, 512),
det (accepted for a uniform command line; see ``determinism``).

Output in ``out``: log.txt (its first line written as soon as the load runs, its last line 'done') and result.json
(threads, seconds, products, and the seconds actually spent).
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

from .. import __version__
from ..config import in_notebook, parse
from ..determinism import mode as det_mode

EXPERIMENT = "load"
DEFAULTS = dict(threads=4, seconds=60.0, out="", size=512, det="")


def main(argv=(), **overrides) -> dict:
    a = parse(DEFAULTS, argv, **overrides)
    if not a["out"]:
        raise ValueError("out= is required")
    with det_mode(a["det"]) as det:
        return _run(a, det)


def _run(a: dict, det) -> dict:
    out = a["out"]
    os.makedirs(out, exist_ok=True)
    log_path = os.path.join(out, "log.txt")

    def log(line: str) -> None:
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")

    threads, seconds, size = int(a["threads"]), float(a["seconds"]), int(a["size"])
    torch.set_num_threads(threads)
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(size, size, generator=gen)
    y = torch.randn(size, size, generator=gen) / size ** 0.5
    t0 = time.time()
    x = x @ y
    products = 1
    log(f"LOAD running: {threads} threads, float32 products of side {size}, for {seconds:g} s; torch {torch.__version__}")
    while time.time() - t0 < seconds:
        x = torch.tanh(x @ y)                   # kept bounded, so the numbers never overflow however long it runs
        products += 1
    spent = time.time() - t0
    result = {"experiment": EXPERIMENT, "threads": threads, "seconds": seconds, "products": products,
              "spent_seconds": spent, "size": size,
              "args": {k: a[k] for k in sorted(a) if not (k == "det" and not a[k])}, "torch": torch.__version__,
              "version": __version__}
    det.record(result)
    with open(os.path.join(out, "result.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=1)
        fh.write("\n")
    log(f"RESULT {products} products in {spent:.1f} s")
    log("done")
    return result


def summarize(root: str) -> str:
    """A markdown table, one row per job folder under ``root`` holding a result.json of this experiment."""
    rows = []
    for dirpath, _, files in os.walk(root):
        if "result.json" in files:
            with open(os.path.join(dirpath, "result.json"), encoding="utf-8") as fh:
                r = json.load(fh)
            if r.get("experiment") == EXPERIMENT:
                rows.append(r)
    lines = ["| threads | seconds | products | spent |", "|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["threads"], r["seconds"])):
        lines.append(f"| {r['threads']} | {r['seconds']:g} | {r['products']:,} | {r['spent_seconds']:.1f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__" and not in_notebook():
    main(sys.argv[1:])
