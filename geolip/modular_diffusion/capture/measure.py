"""Measuring the teacher per band, with no training.

    removal_scores(teacher, probe)       for each band, the born read of every SINGLE change of the teacher's shape:
                                         down stage i keeps one layer instead of two (i = 0 .. 3; its mirrored up stage
                                         keeps two of three, so each is a paired removal), the middle block removed, the
                                         innermost stage removed. Six rows a band: which removals each band tolerates.
    text_need(teacher, prompts, seed)    how much each band needs the prompt: the final latent sampled with the true
                                         prompt on every step, against the same sample with the EMPTY prompt, or with
                                         another row's prompt (SHUFFLED), served inside one band only; beside the same
                                         change for the empty prompt on every step.
    cost(teacher, shape)                 measured seconds of one forward of a shape's engine against the teacher's.

A born read is the mean squared difference, per element, between an untrained engine's answer (every number copied
from the teacher) and the teacher's answer on a band's probe states (``reads.band_read``). Every result is a plain dict
of numbers with a list of ``rows`` for ``reads.table``.

"Inside one band only" means through the band weights of the assembly: on a step the answer is the weighted sum of the
teacher's answer with the true prompt (the other bands' weights) and with the changed prompt (the band's weight), so the
change is served exactly where the band's weight is above zero and fades with it across the crossfades.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .schedule import BANDS
from .shapes import Shape, shape_from

FULL = Shape((2, 2, 2, 2), True, 4)


def single_changes(shape: Shape = FULL) -> List[Tuple[str, Shape]]:
    """The six single changes of the teacher's shape, by name: ``layers0`` .. ``layers3`` (down stage i keeps one
    layer), ``mid_off``, ``stages3``."""
    out = []
    for i in range(4):
        layers = list(shape.layers)
        layers[i] = 1
        out.append((f"layers{i}", Shape(tuple(layers), shape.mid, shape.stages)))
    out.append(("mid_off", Shape(shape.layers, False, shape.stages)))
    out.append(("stages3", Shape(shape.layers, shape.mid, 3)))
    return out


@torch.no_grad()
def removal_scores(teacher, probe, bands: Sequence[str] = BANDS, text: bool = True,
                   log=print) -> dict:
    """For each band, the born read of each single change of the teacher's shape (``single_changes``). The teacher's
    answers are computed once a band and every engine once, so the cost is six engine builds and, per band, the probe's
    forwards of the teacher and of the six engines. Returns {"rows": [{band, change, shape, mse, params, share}, ...],
    "table": {band: {change: mse}}, "teacher_params"}."""
    from .engine import build_engine, param_count
    from .reads import band_read, teacher_answers
    unet = teacher.unet
    total = param_count(unet)
    refs = {b: teacher_answers(teacher, probe, b) for b in bands}
    rows: List[dict] = []
    table: Dict[str, Dict[str, float]] = {b: {} for b in bands}
    for change, shape in single_changes():
        engine = build_engine(teacher, shape, text=text)
        engine.eval()
        n = param_count(engine)
        for b in bands:
            r = band_read(engine, teacher, b, probe, reference=refs[b])
            rows.append({"band": b, "change": change, "shape": shape.text, "mse": float(r["mse"]), "params": int(n),
                         "share": n / total})
            table[b][change] = float(r["mse"])
        if log:
            log(f"removal_scores: {change} ({shape.text}) " + "  ".join(f"{b} {table[b][change]:.4g}" for b in bands))
        del engine
    return {"rows": rows, "table": table, "teacher_params": int(total)}


class PromptSwap(nn.Module):
    """The teacher answering with a changed prompt: every row of the states that is one of the true prompts' rows is
    replaced by the same row of ``replacement``; every other row (the empty prompt of the guidance branch) passes as
    it is. The rows are found by exact equality, so the true prompts' states must differ from the empty prompt's."""
    is_part = True

    def __init__(self, teacher, true_states: Tensor, replacement: Tensor):
        super().__init__()
        if true_states.shape != replacement.shape:
            raise ValueError("the replacement states have the true states' shape")
        self.teacher = teacher
        self.true_states = true_states
        self.replacement = replacement

    def swap(self, states: Tensor) -> Tensor:
        flat = states.reshape(states.shape[0], -1)
        ref = self.true_states.to(states.device, states.dtype).reshape(self.true_states.shape[0], -1)
        same = (flat[:, None, :] == ref[None, :, :]).all(dim=2)          # [rows, prompts]
        hit = same.any(dim=1)
        if not bool(hit.any()):
            return states
        which = same.float().argmax(dim=1)
        rep = self.replacement.to(states.device, states.dtype)[which]
        return torch.where(hit.view(-1, 1, 1), rep, states)

    def forward(self, z: Tensor, t: Tensor, states: Tensor) -> Tensor:
        return self.teacher.predict(z, t, self.swap(states))


def _mse(a: Tensor, b: Tensor) -> float:
    return float((a.double() - b.double()).pow(2).mean())


@torch.no_grad()
def text_need(teacher, prompts: Sequence[str], seed: int = 0, steps: int = 50, guidance: float = 7.5,
              bands: Sequence[str] = BANDS, latent_shape: Optional[Tuple[int, int, int]] = None, log=print,
              repeat: bool = True) -> dict:
    """How much each band needs the prompt (see the module's text). ``prompts``: at least two, none of them empty,
    as strings or as their token ids [N, 77] (the shuffled prompt of row i is row i + 1's, cyclically). One shared
    start noise from ``seed``; the deterministic sampler of ``assemble.sample``. Returns {"rows": [{band, empty,
    shuffled}, ...], "empty_everywhere", "repeat", "table", "steps", "guidance", "seed", "prompts"}: each number the
    mean squared change of the final latent, per element, against the true-prompt sample; ``repeat`` is that change
    for the true-prompt sample made a second time (the control: exactly 0 where the sampler repeats itself; None
    when ``repeat`` is off, which saves one sampling)."""
    from .assemble import Assembly, sample
    ids = prompts if isinstance(prompts, Tensor) else teacher.tokenize([str(p) for p in prompts])
    n = int(ids.shape[0])
    if n < 2:
        raise ValueError("text_need needs at least two prompts (one to shuffle into another)")
    states = teacher.text_states(ids)
    empty = teacher.empty_states(n)
    if bool((states.reshape(n, -1) == empty[:1].reshape(1, -1)).all(dim=1).any()):
        raise ValueError("a prompt's states equal the empty prompt's; text_need needs prompts that are not empty")
    shuffled = states.roll(1, dims=0)
    cfg = teacher.unet.config
    shape = tuple(latent_shape) if latent_shape is not None else (int(cfg.in_channels), int(cfg.sample_size),
                                                                   int(cfg.sample_size))
    kw = dict(steps=int(steps), guidance=float(guidance), latent_shape=shape, alphas_cumprod=teacher.alphas_cumprod)
    ref = sample(teacher, states, empty, int(seed), **kw)
    again = _mse(sample(teacher, states, empty, int(seed), **kw), ref) if repeat else None
    everywhere = _mse(sample(teacher, empty, empty, int(seed), **kw), ref)
    rows: List[dict] = []
    table: Dict[str, Dict[str, float]] = {}
    for b in bands:
        row = {"band": b}
        for name, rep in (("empty", empty), ("shuffled", shuffled)):
            parts = {o: teacher for o in BANDS}
            parts[b] = PromptSwap(teacher, states, rep)
            out = sample(Assembly(parts), states, empty, int(seed), **kw)
            row[name] = _mse(out, ref)
        rows.append(row)
        table[b] = {"empty": row["empty"], "shuffled": row["shuffled"]}
        if log:
            log(f"text_need: {b} empty {row['empty']:.4g} shuffled {row['shuffled']:.4g} "
                f"(empty everywhere {everywhere:.4g})")
    return {"rows": rows, "table": table, "empty_everywhere": everywhere, "repeat": again, "steps": int(steps),
            "guidance": float(guidance), "seed": int(seed), "prompts": n}


def cost(teacher, shape, batch: int = 1, size: Optional[int] = None, repeats: int = 3, step: int = 500) -> dict:
    """Measured seconds of one forward of the shape's engine and of the teacher, at ``batch`` rows of ``size`` x
    ``size`` latents (the teacher's sample size by default), at one step: {"shape", "params", "teacher_params",
    "share", "seconds", "teacher_seconds", "cost", "batch", "size"} (``engine.shape_report``)."""
    from .engine import shape_report
    return shape_report(teacher, shape_from(shape), batch=int(batch), size=size, repeats=int(repeats), step=int(step))
