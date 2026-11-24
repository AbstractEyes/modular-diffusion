"""Reads: how far a part's answers are from the teacher's, on fixed probes. No read trains anything.

A PROBE is fixed prompt states, fixed clean latents and fixed noise (from a seed), and per band a few fixed steps taken
evenly over the band's step set. At a step t the probe's input is z_t = sqrt(a_t) x0 + sqrt(1 - a_t) eps.

    band_read(part, teacher, band, probe)     mean squared difference, per element, between the part's answer and the
                                              teacher's on the band's probe steps; the teacher's own copy reads 0
    born_read(teacher, shape, band, probe)    band_read of the shape's engine as it is born (inherited, untrained)
    exchange(assembly, band, other, probe)    what changes when one band's part is replaced: every band's read, and the
                                              final latents of the sampler
    off_band(part, teacher, served, trained, probe)
                                              a part read on a band it was not made for (the control that fails)

Every read returns a plain dict of numbers (and names), ready for ``write_json`` and ``table``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union

import torch
from torch import Tensor

from . import BANDS
from .assemble import Assembly, sample
from .engine import as_part, build_engine, param_count
from .schedule import add_noise, band_index, band_steps, band_weights, sd15_alphas_cumprod
from .shapes import Shape, shape_from


@dataclass
class Probe:
    states: Tensor                     # [N, 77, D] prompt states
    x0: Tensor                         # [N, 4, h, w] clean latents
    eps: Tensor                        # [N, 4, h, w] the noise
    steps: Dict[str, Tensor]           # band -> its probe steps
    alphas_cumprod: Tensor
    empty: Optional[Tensor] = None     # [N or 1, 77, D] the empty prompt's states (for sampling reads)
    seed: int = 0
    batch: int = 0                     # rows a forward in the reads (0: all the probe's rows at once)

    def z(self, t: int) -> Tensor:
        """z_t of every probe row at the step t."""
        tt = torch.full((self.x0.shape[0],), int(t), dtype=torch.long)
        return add_noise(self.x0, self.eps, tt, self.alphas_cumprod)


def even_steps(band: Union[str, int], n: int) -> Tensor:
    """n steps taken evenly over the band's step set (its first and last step included when n > 1)."""
    own = band_steps(band)
    if n <= 1:
        return own[len(own) // 2: len(own) // 2 + 1]
    idx = torch.linspace(0, len(own) - 1, int(n), dtype=torch.float64).round().long()
    return own[idx]


def make_probe(states: Tensor, seed: int = 0, x0: Optional[Tensor] = None, per_band: int = 4,
               latent_shape: Sequence[int] = (4, 64, 64), empty: Optional[Tensor] = None,
               alphas_cumprod: Optional[Tensor] = None, batch: int = 0) -> Probe:
    """A probe from prompt states [N, 77, D]: the clean latents ``x0`` (or standard normal ones of ``latent_shape``
    from the seed), the noise from the seed, ``per_band`` even steps of each band. ``batch``: how many rows the reads
    put through a part at once (0: all of them); the teacher's answers and a part's are cut the same way."""
    n = states.shape[0]
    g = torch.Generator().manual_seed(int(seed))
    if x0 is None:
        x0 = torch.randn((n,) + tuple(int(x) for x in latent_shape), generator=g)
    # The probe's latents and noise are always float32, whatever the precision of the prompt states: every read
    # compares answers in full precision, and each part casts its inputs to its own precision.
    x0 = x0.detach().to(device=states.device, dtype=torch.float32)
    eps = torch.randn(tuple(x0.shape), generator=g).to(device=states.device, dtype=torch.float32)
    ac = (alphas_cumprod if alphas_cumprod is not None else sd15_alphas_cumprod()).double()
    steps = {b: even_steps(b, per_band) for b in BANDS}
    return Probe(states, x0, eps, steps, ac, empty, int(seed), int(batch))


def _answer(part, z: Tensor, t: int, states: Tensor, batch: int = 0) -> Tensor:
    tt = torch.full((z.shape[0],), int(t), dtype=torch.long, device=z.device)
    part = as_part(part)
    n = z.shape[0]
    with torch.no_grad():
        if not batch or n <= batch:
            return part(z, tt, states)
        return torch.cat([part(z[i:i + batch], tt[i:i + batch], states[i:i + batch]) for i in range(0, n, int(batch))])


def teacher_answers(teacher, probe: Probe, band: str) -> Dict[int, Tensor]:
    """{t: the teacher's answer} on a band's probe steps (pass as ``reference=`` to read many parts against them)."""
    return {int(t): _answer(teacher, probe.z(int(t)), int(t), probe.states, probe.batch) for t in probe.steps[band]}


def band_read(part, teacher, band: str, probe: Probe, reference: Optional[Dict[int, Tensor]] = None) -> dict:
    """The mean squared difference, per element, between the part's and the teacher's answers on the band's probe
    steps (float64), with the same per step."""
    band = BANDS[band_index(band)]
    ref = reference if reference is not None else teacher_answers(teacher, probe, band)
    per, total, count = {}, 0.0, 0
    for t in probe.steps[band]:
        t = int(t)
        o = _answer(part, probe.z(t), t, probe.states, probe.batch).double()
        d = (o - ref[t].double()).pow(2)
        per[str(t)] = float(d.mean())
        total += float(d.sum())
        count += d.numel()
    return {"band": band, "mse": total / count, "per_step": per, "steps": [int(t) for t in probe.steps[band]],
            "elements": count}


def born_read(teacher, shape: Union[str, Shape], band: str, probe: Probe, text: bool = True,
              reference: Optional[Dict[int, Tensor]] = None) -> dict:
    """``band_read`` of the shape's engine as it is born (inherited, untrained), with its size."""
    shape = shape_from(shape)
    eng = build_engine(teacher, shape, text=text)
    r = band_read(eng, teacher, band, probe, reference)
    p = param_count(eng)
    r.update({"shape": shape.name, "text": bool(text), "params": p, "share": p / param_count(teacher)})
    return r


def exchange(assembly: Assembly, band: str, other_part, probe: Probe, teacher=None, sample_steps: int = 50,
             guidance: float = 7.5, seed: Optional[int] = None) -> dict:
    """Replace one band's part of an assembly by ``other_part`` and read what moves.

    Per band: its read against the teacher (``teacher``; the assembly itself when none is given, so the reads are the
    change of the answers) before and after, and the largest change of the answers on the probe steps where the
    replaced band's weight is exactly zero (outside its crossfade it must be exactly 0.0). Then the mean squared change
    of the final latents of the sampler (the probe's states and empty states, ``sample_steps`` steps, the probe's seed)
    when the probe holds empty states."""
    band = BANDS[band_index(band)]
    parts = {b: assembly.part(b) for b in BANDS}
    parts[band] = other_part
    after = Assembly(parts)
    ref_model = teacher if teacher is not None else assembly
    out = {"band": band, "bands": {}}
    for b in BANDS:
        ref = teacher_answers(ref_model, probe, b)
        r0 = band_read(assembly, ref_model, b, probe, ref)
        r1 = band_read(after, ref_model, b, probe, ref)
        silent = [int(t) for t in probe.steps[b] if float(band_weights(int(t))[0, band_index(band)]) == 0.0]
        moved = 0.0
        for t in silent:
            z = probe.z(t)
            d = (_answer(after, z, t, probe.states, probe.batch).double()
                 - _answer(assembly, z, t, probe.states, probe.batch).double())
            moved = max(moved, float(d.abs().max()))
        out["bands"][b] = {"before": r0["mse"], "after": r1["mse"], "change": r1["mse"] - r0["mse"],
                           "silent_steps": silent, "largest_change_where_silent": moved}
    if probe.empty is not None and int(sample_steps) > 0:
        s = probe.seed if seed is None else int(seed)
        shape = tuple(probe.x0.shape[1:])
        z0 = sample(assembly, probe.states, probe.empty, s, sample_steps, guidance, latent_shape=shape,
                    alphas_cumprod=probe.alphas_cumprod)
        z1 = sample(after, probe.states, probe.empty, s, sample_steps, guidance, latent_shape=shape,
                    alphas_cumprod=probe.alphas_cumprod)
        out["final_latent_change"] = float((z1.double() - z0.double()).pow(2).mean())
    return out


def off_band(part, teacher, band_served: str, band_trained: str, probe: Probe) -> dict:
    """A part made for ``band_trained`` read on ``band_served``, beside its read on its own band."""
    served = band_read(part, teacher, band_served, probe)
    own = band_read(part, teacher, band_trained, probe)
    return {"served": served["band"], "trained": own["band"], "off_band_mse": served["mse"], "on_band_mse": own["mse"],
            "ratio": served["mse"] / own["mse"] if own["mse"] > 0 else float("inf")}


# ------------------------------------------------------------------ output
def write_json(obj, path: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1)
        fh.write("\n")


def _cell(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}" if v != 0.0 else "0"
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:,}"
    return str(v)


def table(rows: Sequence[dict], columns: Sequence[str]) -> str:
    """A markdown table of the given columns of plain dicts (a missing value is shown as -)."""
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for r in rows:
        lines.append("| " + " | ".join("-" if r.get(c) is None else _cell(r.get(c)) for c in columns) + " |")
    return "\n".join(lines) + "\n"
