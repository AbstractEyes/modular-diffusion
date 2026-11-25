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
    sample_probe(model, probe, steps, guidance, group)
                                              the sampler's final latents of every probe row, ``group`` rows at a time
    chain_read(model, teacher, probe, sample_steps, guidance, group)
                                              how far the model's guided answers are from the teacher's on the states
                                              of the sampler's chain, on the model's own chain and on the teacher's

THE CHAIN READ: two chains from the same start noise (the rule of ``sample_probe``). On the model's OWN chain the
sampler is driven by the model and the teacher is asked at every step on the same state, with the same prompt states,
empty states and guidance; on the TEACHER'S chain the sampler is driven by the teacher and the model is asked. At a
step t and a row i, e_i(t) = the mean over the latent's entries of (the model's guided answer - the teacher's)^2, in
float64 from the answers as the sampler uses them. Per band b and row, E_i(b) = sum_t w_b(t) e_i(t) / sum_t w_b(t)
over the sampler's steps (w_b: the band's weight at the step, ``schedule.band_weights``). Per band: the mean of E_i(b)
over the rows on each chain, and the ratio own over teacher's, exp(mean_i log(E_own,i(b) / E_teacher,i(b))), with the
standard error of the mean log ratio over the rows (std with n - 1 over sqrt(n)); a row where either error is exactly
0 is left out of the ratio and counted, and a band with no row left has no ratio (None). The teacher read against
itself reads exactly 0 at every step on both chains; the own chain's final latents are ``sample_probe``'s of the model
and the teacher's chain's are the teacher's own, to the bit.

GROUPED SAMPLING: the probe's rows are cut into consecutive groups of ``group`` rows (the last may be smaller; 0 = one
group of every row). Group j starts from ``assemble.initial_noise(group_seed(probe.seed, j), rows of the group)``,
group_seed(s, j) = s + 100003 j, so group 0 starts from the probe's own seed (one group gives exactly the sampling of
all rows in one batch) and the noise depends only on the probe's seed, the group size and the group's index: two models
sampled with the same probe and group size start every row from the same noise. The same rows with ANOTHER group size
get other noise (not promised to match). With guidance a forward holds 2 x group rows.

Every read returns a plain dict of numbers (and names), ready for ``write_json`` and ``table``.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from . import BANDS
from .assemble import Assembly, ddim_timesteps, guided, initial_noise, sample
from .engine import UNetPart, as_part, build_engine, param_count
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


def group_seed(seed: int, j: int) -> int:
    """The start-noise seed of the sampling group j of a probe of seed ``seed`` (group 0: the seed itself)."""
    return int(seed) + 100_003 * int(j)


def sample_groups(rows: int, group: int):
    """[(first row, rows)] of the sampling groups of ``rows`` rows (``group`` 0: one group)."""
    rows, group = int(rows), int(group or 0)
    if group < 0:
        raise ValueError("group is 0 (one group) or a positive number of rows")
    if group == 0 or group >= rows:
        return [(0, rows)]
    return [(i, min(group, rows - i)) for i in range(0, rows, group)]


@torch.no_grad()
def sample_probe(model, probe: Probe, sample_steps: int = 50, guidance: float = 7.5, group: int = 0,
                 seed: Optional[int] = None) -> Tensor:
    """The sampler's final latents [N, C, h, w] of every probe row (its states, its empty states, ``sample_steps``
    steps), ``group`` rows at a time; group j starts from the noise of ``group_seed(seed, j)`` (``seed``: the probe's
    by default). See the module's text for the rule."""
    if probe.empty is None:
        raise ValueError("sampling a probe needs its empty prompt's states")
    s = probe.seed if seed is None else int(seed)
    shape = tuple(probe.x0.shape[1:])
    out = []
    for j, (first, n) in enumerate(sample_groups(probe.states.shape[0], group)):
        z = initial_noise(group_seed(s, j), n, shape)
        out.append(sample(model, probe.states[first:first + n], probe.empty, s, sample_steps, guidance,
                          latent_shape=shape, alphas_cumprod=probe.alphas_cumprod, z=z))
    return torch.cat(out)


def _reuse(parts: Dict[str, object], other_part):
    """``other_part`` itself, or the assembly's own wrapped part when ``other_part`` is the same model (the same object,
    or a teacher or U-Net whose U-Net is the same object): a part put back for itself then shares its slot as before."""
    for p in parts.values():
        if p is other_part:
            return p
        if isinstance(p, UNetPart):
            unet = other_part if hasattr(other_part, "down_blocks") else getattr(other_part, "unet", None)
            if unet is not None and p.unet is unet:
                return p
    return other_part


def exchange(assembly: Assembly, band: str, other_part, probe: Probe, teacher=None, sample_steps: int = 50,
             guidance: float = 7.5, seed: Optional[int] = None, group: int = 0) -> dict:
    """Replace one band's part of an assembly by ``other_part`` and read what moves.

    Per band: its read against the teacher (``teacher``; the assembly itself when none is given, so the reads are the
    change of the answers) before and after, and the largest change of the answers on the probe steps where the
    replaced band's weight is exactly zero (outside its crossfade it must be exactly 0.0). Then the mean squared change
    over all rows of the final latents of the sampler (``sample_probe``: the probe's states and empty states,
    ``sample_steps`` steps, ``group`` rows at a time, the same start noise for both assemblies) when the probe holds
    empty states. A part put back for itself (the same model) changes nothing: every number is exactly 0."""
    band = BANDS[band_index(band)]
    parts = {b: assembly.part(b) for b in BANDS}
    parts[band] = _reuse(parts, other_part)
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
        z0 = sample_probe(assembly, probe, sample_steps, guidance, group, seed)
        z1 = sample_probe(after, probe, sample_steps, guidance, group, seed)
        out["final_latent_change"] = float((z1.double() - z0.double()).pow(2).mean())
        out["sample_group"] = int(group or 0)
    return out


def latent_hash(z: Tensor) -> str:
    """The sha256 of latents' float32 bytes (on the CPU, rows in order, each row's entries in memory order)."""
    return hashlib.sha256(z.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes()).hexdigest()


@torch.no_grad()
def chain_errors(driver, asked, probe: Probe, sample_steps: int = 50, guidance: float = 7.5, group: int = 0,
                 seed: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    """(e [steps, N] float64, the final latents [N, C, h, w]): the sampler driven by ``driver`` from the start noise of
    ``sample_probe`` (the same groups and seeds), and at every step ``asked`` given the same state, prompt states,
    empty states and guidance; e[k, i] = the mean over row i's latent entries of the squared difference of the two
    guided answers at the k-th step (``assemble.ddim_timesteps`` order, noisiest first). The final latents are the
    driver's ``sample_probe``."""
    if probe.empty is None:
        raise ValueError("the chain read samples, so the probe needs its empty prompt's states")
    s = probe.seed if seed is None else int(seed)
    shape = tuple(probe.x0.shape[1:])
    ts = ddim_timesteps(sample_steps)
    index = {t: k for k, t in enumerate(ts)}
    n = probe.states.shape[0]
    e = torch.zeros(len(ts), n, dtype=torch.float64)
    other = as_part(asked)
    finals = []
    for j, (first, rows) in enumerate(sample_groups(n, group)):
        states = probe.states[first:first + rows]
        empty = probe.empty.to(device=states.device, dtype=states.dtype)      # as the sampler casts it

        def watch(t, z, eps, first=first, rows=rows, states=states, empty=empty):
            tt = torch.full((rows,), int(t), dtype=torch.long, device=z.device)
            other_eps = guided(other, z, tt, states, empty, guidance)
            d = (eps.double() - other_eps.double()).pow(2).reshape(rows, -1).mean(dim=1)
            e[index[int(t)], first:first + rows] = d.cpu()

        z = initial_noise(group_seed(s, j), rows, shape)
        finals.append(sample(driver, states, probe.empty, s, sample_steps, guidance, latent_shape=shape,
                             alphas_cumprod=probe.alphas_cumprod, z=z, on_step=watch))
    return e, torch.cat(finals)


def band_errors(e: Tensor, steps: Sequence[int]) -> Dict[str, Tensor]:
    """{band: [N] float64}: E_i(b) = sum_t w_b(t) e_i(t) / sum_t w_b(t) over the given steps (the rows of ``e``)."""
    w = band_weights(list(steps))                           # [steps, 3] float64
    out = {}
    for b in BANDS:
        wb = w[:, band_index(b)]
        out[b] = (wb[:, None] * e).sum(dim=0) / wb.sum()
    return out


def log_ratio(own: Sequence[float], teacher: Sequence[float]) -> dict:
    """The ratio of two per-row errors as exp(mean log(own / teacher)) with the standard error of the mean log ratio
    (std with n - 1 over sqrt(n)); rows where either is exactly 0 are left out and counted."""
    logs: List[float] = [math.log(a / b) for a, b in zip(own, teacher) if a != 0.0 and b != 0.0]
    left_out = len(list(own)) - len(logs)
    if not logs:
        return {"ratio": None, "log_ratio_mean": None, "log_ratio_se": None, "rows_in_ratio": 0, "rows_left_out": left_out}
    m = sum(logs) / len(logs)
    se = (math.sqrt(sum((x - m) ** 2 for x in logs) / (len(logs) - 1)) / math.sqrt(len(logs))
          if len(logs) > 1 else None)
    return {"ratio": math.exp(m), "log_ratio_mean": m, "log_ratio_se": se, "rows_in_ratio": len(logs),
            "rows_left_out": left_out}


def chain_read(model, teacher, probe: Probe, sample_steps: int = 50, guidance: float = 7.5, group: int = 0,
               seed: Optional[int] = None) -> dict:
    """The chain read of ``model`` (an assembly or any part) against ``teacher``: see the module's text. Returns the
    per-band means on both chains, the ratio and its standard error, the per-step means over rows on both chains, the
    per-row band errors, the mean squared difference of the two chains' final latents and both chains' content
    hashes (``latent_hash``)."""
    t0 = time.time()
    steps = ddim_timesteps(sample_steps)
    e_own, z_own = chain_errors(model, teacher, probe, sample_steps, guidance, group, seed)
    e_teacher, z_teacher = chain_errors(teacher, model, probe, sample_steps, guidance, group, seed)
    own, tea = band_errors(e_own, steps), band_errors(e_teacher, steps)
    bands = {}
    for b in BANDS:
        o, t = [float(x) for x in own[b]], [float(x) for x in tea[b]]
        r = {"own_mean": sum(o) / len(o), "teacher_mean": sum(t) / len(t)}
        r.update(log_ratio(o, t))
        bands[b] = r
    return {"steps": steps, "rows": int(e_own.shape[1]), "sample_steps": int(sample_steps), "guidance": float(guidance),
            "group": int(group or 0), "seed": probe.seed if seed is None else int(seed), "bands": bands,
            "own_per_step": [float(x) for x in e_own.mean(dim=1)],
            "teacher_per_step": [float(x) for x in e_teacher.mean(dim=1)],
            "own_per_row": {b: [float(x) for x in own[b]] for b in BANDS},
            "teacher_per_row": {b: [float(x) for x in tea[b]] for b in BANDS},
            "final_latent_change": float((z_own.double() - z_teacher.double()).pow(2).mean()),
            "own_final_hash": latent_hash(z_own), "teacher_final_hash": latent_hash(z_teacher),
            "seconds": time.time() - t0}


def off_band(part, teacher, band_served: str, band_trained: str, probe: Probe) -> dict:
    """A part made for ``band_trained`` read on ``band_served``, beside its read on its own band."""
    served = band_read(part, teacher, band_served, probe)
    own = band_read(part, teacher, band_trained, probe)
    return {"served": served["band"], "trained": own["band"], "off_band_mse": served["mse"], "on_band_mse": own["mse"],
            "ratio": served["mse"] / own["mse"] if own["mse"] > 0 else None}      # None: the on-band read is exactly 0


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
