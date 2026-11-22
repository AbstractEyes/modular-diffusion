"""Data beds: Gaussian fields of a known power-law spectrum, and CIFAR-style pictures.

A bed provides ``field_shape``, ``n_classes``, ``spectrum`` ([C, H, W] float64 on the CPU, unit noise = 1, the anchor's
input), ``train_batch(n, gen)`` and ``heldout(n)``.

Pictures are scaled to [0, 1], normalized per channel by the training set's mean and standard deviation (computed on the
CPU), and a training batch is flipped left-right at random (probability 1/2 per picture). Random draws of a training
batch from ``gen``, in order: the picture indices, then the flips.
"""
from __future__ import annotations

import copy
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from .spectral import class_statistics, field_spectrum, measure_spectrum, tilt_gain, token_filter

NULL_CLASS = 10
SPECTRUM_PICTURES = 20000            # the spectrum (and the tilt gain) is measured on the first 20,000 training pictures
HELDOUT_FIELDS_SEED = 987654321


class FieldsBed:
    """Fresh Gaussian fields of the power law field_spectrum(grid, slope, dc_share) in every channel; no classes."""
    kind = "fields"

    def __init__(self, grid: int, channels: int, slope: float, dc_share: float, device="cpu"):
        self.device = device
        self.tilt = None
        s = field_spectrum(grid, slope, dc_share)
        self.spectrum = s[None].repeat(channels, 1, 1)
        self.sqrt_s = s.sqrt().to(device=device, dtype=torch.float32)
        self.field_shape = (channels, grid, grid)
        self.n_classes = 0

    def train_batch(self, n: int, gen: torch.Generator) -> Tuple[Tensor, Tensor]:
        c, g, _ = self.field_shape
        w = torch.randn(n, c, g, g, generator=gen, device=self.device)
        x = torch.fft.ifft2(torch.fft.fft2(w) * self.sqrt_s).real.contiguous()
        return x, torch.full((n,), NULL_CLASS, device=self.device, dtype=torch.long)

    def heldout(self, n: int) -> Tuple[Tensor, Tensor]:
        gen = torch.Generator(device=self.device).manual_seed(HELDOUT_FIELDS_SEED)
        return self.train_batch(n, gen)


class PicturesBed:
    """Pictures from an npz holding uint8 x_train [N, H, W, C], y_train, x_test, y_test.

    tilt: refilter every token channel on the token grid so that its spectrum is the power law (slope, dc_share), with
    gains measured on the training pictures only.

    without_classes / only_classes: the training stream draws only from the kept pictures (the classes not listed in
    ``without_classes``, and, when ``only_classes`` is given, only those). The mean, the std, the pooled spectrum, the
    held-out set and ``train_pictures()`` stay those of the whole data set; ``class_stats()`` uses the kept pictures.
    With nothing excluded the stream's code path and draws are exactly the unrestricted ones."""
    kind = "pictures"

    def __init__(self, npz: str, device="cpu", grid: int = 16, tilt: bool = False, slope: float = 1.225,
                 dc_share: float = 0.060, without_classes=(), only_classes=None):
        self.device = device
        self.tilt = None
        d = np.load(npz)
        xtr = torch.from_numpy(d["x_train"]).permute(0, 3, 1, 2).float() / 255.0
        xte = torch.from_numpy(d["x_test"]).permute(0, 3, 1, 2).float() / 255.0
        self.mean = xtr.mean(dim=(0, 2, 3), keepdim=True)
        self.std = xtr.std(dim=(0, 2, 3), keepdim=True)
        xtr, xte = (xtr - self.mean) / self.std, (xte - self.mean) / self.std
        if tilt:
            self.tilt = tilt_gain(xtr[:SPECTRUM_PICTURES], grid, slope, dc_share)
            xtr = torch.cat([token_filter(xtr[i:i + 5000], self.tilt, grid) for i in range(0, xtr.shape[0], 5000)])
            xte = token_filter(xte, self.tilt, grid)
        self.grid = grid
        self.xtr = xtr.to(device)
        self.ytr = torch.from_numpy(d["y_train"]).to(device)
        self.xte = xte.to(device)
        self.yte = torch.from_numpy(d["y_test"]).to(device)
        self.spectrum = measure_spectrum(self.xtr[:SPECTRUM_PICTURES]).cpu()       # float64 on the run device, then CPU
        self.field_shape = tuple(self.xtr.shape[1:])
        self.n_classes = 10
        self._restrict(without_classes, only_classes)

    def _restrict(self, without_classes=(), only_classes=None) -> None:
        self.without_classes = tuple(int(c) for c in without_classes)
        self.only_classes = None if only_classes is None else tuple(int(c) for c in only_classes)
        self._class_stats = None
        self.kept = None                         # the kept training rows (None: every row)
        if self.without_classes or self.only_classes is not None:
            keep = torch.ones_like(self.ytr, dtype=torch.bool)
            if self.only_classes is not None:
                keep &= torch.isin(self.ytr, torch.tensor(self.only_classes, device=self.ytr.device, dtype=self.ytr.dtype))
            if self.without_classes:
                keep &= ~torch.isin(self.ytr, torch.tensor(self.without_classes, device=self.ytr.device,
                                                           dtype=self.ytr.dtype))
            self.kept = torch.nonzero(keep, as_tuple=False).squeeze(1)
            if self.kept.numel() == 0:
                raise ValueError("the class restriction keeps no training picture")

    def restricted(self, without_classes=(), only_classes=None) -> "PicturesBed":
        """The same bed (its tensors shared, not copied) with another training-stream restriction."""
        other = copy.copy(self)
        other._restrict(without_classes, only_classes)
        return other

    def train_batch(self, n: int, gen: torch.Generator) -> Tuple[Tensor, Tensor]:
        if self.kept is None:
            idx = torch.randint(0, self.xtr.shape[0], (n,), generator=gen, device=self.device)
        else:
            j = torch.randint(0, self.kept.shape[0], (n,), generator=gen, device=self.device)
            idx = self.kept[j]
        x = self.xtr[idx]
        flip = torch.rand(n, generator=gen, device=self.device) < 0.5
        x = torch.where(flip.view(n, 1, 1, 1), x.flip(-1), x)
        return x, self.ytr[idx]

    def heldout(self, n: int) -> Tuple[Tensor, Tensor]:
        return self.xte[:n], self.yte[:n]

    def train_pictures(self) -> Tensor:
        """Every training picture in model space, unflipped, on the bed's device (the picture score's reference)."""
        return self.xtr

    def train_labels(self) -> Tensor:
        """The classes of ``train_pictures()``, in the same order."""
        return self.ytr

    def test_pictures(self) -> Tuple[Tensor, Tensor]:
        """Every test picture in model space with its class (x, y), on the bed's device."""
        return self.xte, self.yte

    def class_stats(self) -> Tuple[Tensor, Tensor]:
        """(means [10, C, H, W], spectra [10, C, H, W]) float32 on the CPU: ``spectral.class_statistics`` of the kept
        training pictures with their mirror images; a class with no kept picture gets a zero mean and the pooled
        spectrum. Computed once per bed."""
        if self._class_stats is None:
            rows = self.kept
            x = self.xtr if rows is None else self.xtr[rows]
            y = self.ytr if rows is None else self.ytr[rows]
            m, s = class_statistics(x, y, self.n_classes, flips=True, pooled=self.spectrum)
            self._class_stats = (m.cpu(), s.cpu())
        return self._class_stats


def make_bed(a: dict, device="cpu"):
    """The bed of an argument set (keys bed, data, grid, channels, slope, dc_share, tilt)."""
    if a["bed"] == "fields":
        return FieldsBed(int(a["grid"]), int(a["channels"]), float(a["slope"]), float(a["dc_share"]), device)
    if a["bed"] == "pictures":
        if not a["data"]:
            raise ValueError("bed=pictures needs data=<path to the npz>")
        return PicturesBed(a["data"], device, int(a["grid"]), bool(int(a["tilt"])), float(a["slope"]), float(a["dc_share"]))
    raise ValueError(f"unknown bed {a['bed']!r}")
