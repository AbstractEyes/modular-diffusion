"""The process of a source: how it noises clean latents, what its answer means, and how its sampler steps.

Two kinds share one interface. A time ``t`` is always the number fed to the U-Net, and the stretch coordinate is
s = t / 1000 on both kinds (a stretch's edges sit on each source's own coordinate):

| piece                       | ``eps`` (the noise-predicting teacher)            | ``flow`` (a velocity-predicting source)     |
|-----------------------------|---------------------------------------------------|---------------------------------------------|
| times                       | whole steps t = 0 .. 999                          | real t = 1000 s, s in [0, 1]                |
| the noised input            | z = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps        | z = (1 - s) x0 + s eps                      |
| the answer                  | the noise eps                                     | the velocity v = eps - x0                   |
| the clean latents           | x0 = (z - sqrt(1 - abar_t) eps_hat) / sqrt(abar_t) | x0 = z - s v_hat                            |
| the sampling grid           | DDIM: i (1000 // N) + offset, i = N - 1 .. 0      | s_i = k u_i / (1 + (k - 1) u_i), u_i = 1 - i / N, i = 0 .. N - 1 (k: the shift) |
| one step from t to t_next   | x0 as above, z <- sqrt(abar_next) x0 + sqrt(1 - abar_next) eps_hat (the last step to abar_0) | Euler: z <- z + (s_next - s) v_hat (the last step to s = 0) |
| signal-to-noise             | abar_t / (1 - abar_t)                             | ((1 - s) s0 / s)^2, s0 the clean latents' standard deviation |

The ``eps`` kind is the teacher's process as built: its noised input is ``schedule.add_noise``, its grid is
``assemble.ddim_timesteps``, and its step is the DDIM step the sampler has always taken (the same expressions in the same
order), so a sampler routed through it gives the same latents to the bit.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Union

import torch
from torch import Tensor

from .schedule import STEPS, add_noise, sd15_alphas_cumprod

KINDS = ("eps", "flow")

Times = Union[int, float, Sequence[float], Tensor]


def _row_times(t: Times, n: int, device) -> Tensor:
    """[n] float64 times on ``device`` (one time for every row, or one per row)."""
    x = torch.as_tensor(t, dtype=torch.float64) if not isinstance(t, Tensor) else t.to(torch.float64)
    x = x.reshape(-1).to(device)
    if x.numel() == 1 and n > 1:
        x = x.expand(n)
    return x


def shifted(u: float, shift: float) -> float:
    """s = k u / (1 + (k - 1) u): a uniform draw ``u`` warped by the shift k."""
    k = float(shift)
    return k * float(u) / (1.0 + (k - 1.0) * float(u))


class Process:
    """The common interface; see the module's text. ``kind`` is "eps" or "flow"."""
    kind = ""

    def coordinate(self, t: Times) -> Tensor:
        """[B] float64: the stretch coordinate s = t / 1000."""
        return _row_times(t, 1, "cpu") / STEPS

    def fed(self, rows: int, t: float, device) -> Tensor:
        """The time tensor given to the U-Net for ``rows`` rows at the time ``t``."""
        raise NotImplementedError

    def noised(self, x0: Tensor, eps: Tensor, t: Times) -> Tensor:
        raise NotImplementedError

    def clean(self, z: Tensor, answer: Tensor, t: Times) -> Tensor:
        raise NotImplementedError

    def grid(self, steps: int, offset: int = 1) -> List[float]:
        raise NotImplementedError

    def after(self, grid: Sequence[float], i: int) -> float:
        """The time the sampler steps to from ``grid[i]``."""
        raise NotImplementedError

    def step(self, z: Tensor, answer: Tensor, t: float, t_next: float) -> Tensor:
        raise NotImplementedError

    def snr(self, t: Times, s0: Optional[float] = None) -> Tensor:
        raise NotImplementedError

    def record(self) -> dict:
        return {"kind": self.kind}


class EpsProcess(Process):
    """The noise-predicting teacher's process on its schedule ``alphas_cumprod`` (float64, 1,000 steps)."""
    kind = "eps"

    def __init__(self, alphas_cumprod: Optional[Tensor] = None):
        self.alphas_cumprod = (alphas_cumprod if alphas_cumprod is not None else sd15_alphas_cumprod()).double()

    def fed(self, rows: int, t: float, device) -> Tensor:
        return torch.full((int(rows),), int(t), dtype=torch.long, device=device)

    def noised(self, x0: Tensor, eps: Tensor, t: Times) -> Tensor:
        tt = t if isinstance(t, Tensor) else torch.full((x0.shape[0],), int(t), dtype=torch.long)
        if tt.numel() == 1 and x0.shape[0] > 1:
            tt = tt.reshape(1).expand(x0.shape[0])
        return add_noise(x0, eps, tt.reshape(-1), self.alphas_cumprod)

    @staticmethod
    def _x0(z: Tensor, answer: Tensor, a: float) -> Tensor:
        """The clean latents from a noise answer at alpha_bar ``a`` (the DDIM step's own expression)."""
        return (z - math.sqrt(1.0 - a) * answer) / math.sqrt(a)

    def clean(self, z: Tensor, answer: Tensor, t: Times) -> Tensor:
        """(z - sqrt(1 - abar_t) eps_hat) / sqrt(abar_t), one step for all rows or one per row."""
        ac = self.alphas_cumprod.cpu()
        tt = _row_times(t, z.shape[0], "cpu").long()
        if tt.numel() == 1 or bool((tt == tt[0]).all()):
            return self._x0(z, answer, float(ac[int(tt[0])]))
        return torch.stack([self._x0(z[i:i + 1], answer[i:i + 1], float(ac[int(tt[i])]))[0] for i in range(z.shape[0])])

    def grid(self, steps: int, offset: int = 1) -> List[float]:
        from .assemble import ddim_timesteps
        return ddim_timesteps(steps, offset)

    def after(self, grid: Sequence[float], i: int) -> float:
        return int(grid[i]) - STEPS // len(grid)

    def step(self, z: Tensor, answer: Tensor, t: float, t_next: float) -> Tensor:
        """One DDIM step (eta 0) from the step t to t_next; a t_next below 0 steps to alpha_bar of step 0."""
        ac = self.alphas_cumprod.cpu()
        a = float(ac[int(t)])
        a_prev = float(ac[int(t_next)]) if int(t_next) >= 0 else float(ac[0])
        x0 = self._x0(z, answer, a)
        return math.sqrt(a_prev) * x0 + math.sqrt(1.0 - a_prev) * answer

    def snr(self, t: Times, s0: Optional[float] = None) -> Tensor:
        a = self.alphas_cumprod.cpu()[_row_times(t, 1, "cpu").long()]
        return a / (1.0 - a)


class FlowProcess(Process):
    """A velocity-predicting source (answer v = eps - x0) on a continuous time, sampled by Euler steps on the grid
    warped by ``shift``."""
    kind = "flow"

    def __init__(self, shift: float):
        shift = float(shift)
        if not shift > 0.0:
            raise ValueError(f"the shift of a flow source is a positive number, not {shift}")
        self.shift = shift

    def fed(self, rows: int, t: float, device) -> Tensor:
        return torch.full((int(rows),), float(t), dtype=torch.float32, device=device)

    def _s(self, t: Times, n: int, x: Tensor) -> Tensor:
        s = _row_times(t, n, x.device) / STEPS
        if bool(((s < 0) | (s > 1)).any()):
            raise ValueError("a flow source's times lie in 0 .. 1000")
        return s.to(x.dtype).reshape((-1,) + (1,) * (x.dim() - 1))

    def noised(self, x0: Tensor, eps: Tensor, t: Times) -> Tensor:
        """(1 - s) x0 + s eps, one time per row (or one for all), in x0's dtype."""
        s = self._s(t, x0.shape[0], x0)
        return (1.0 - s) * x0 + s * eps

    def clean(self, z: Tensor, answer: Tensor, t: Times) -> Tensor:
        """z - s v."""
        return z - self._s(t, z.shape[0], z) * answer

    def grid(self, steps: int, offset: int = 1) -> List[float]:
        """The times asked, noisiest first: 1000 s_i, i = 0 .. N - 1 (``offset`` is not used by this kind)."""
        n = int(steps)
        if n < 1:
            raise ValueError("steps is at least 1")
        return [STEPS * shifted(1.0 - i / n, self.shift) for i in range(n)]

    def after(self, grid: Sequence[float], i: int) -> float:
        n = len(grid)
        return STEPS * shifted(1.0 - (i + 1) / n, self.shift) if i + 1 < n else 0.0

    def step(self, z: Tensor, answer: Tensor, t: float, t_next: float) -> Tensor:
        """Euler: z + (s_next - s) v."""
        return z + (float(t_next) / STEPS - float(t) / STEPS) * answer

    def snr(self, t: Times, s0: Optional[float] = None) -> Tensor:
        if s0 is None:
            raise ValueError("the signal-to-noise of a flow source needs the clean latents' standard deviation s0")
        s = _row_times(t, 1, "cpu") / STEPS
        return ((1.0 - s) * float(s0) / s) ** 2

    def record(self) -> dict:
        return {"kind": self.kind, "shift": self.shift}


def make_process(kind: str = "eps", shift: Optional[float] = None,
                 alphas_cumprod: Optional[Tensor] = None) -> Process:
    """The process of a kind: ``eps`` on the teacher's schedule (or ``alphas_cumprod``), ``flow`` with its ``shift``
    (required: the shift a source was trained at is part of the source)."""
    if kind == "eps":
        return EpsProcess(alphas_cumprod)
    if kind == "flow":
        if shift is None:
            raise ValueError("a flow source needs its shift")
        return FlowProcess(shift)
    raise ValueError(f"the kind of a source is one of {list(KINDS)}, not {kind!r}")
