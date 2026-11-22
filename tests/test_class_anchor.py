"""Content as an element (pass 5): the numbers of earlier builds unchanged (T0), the per-class anchor (T1), the stem's
class window (T2) and class-restricted training streams (T3)."""
import os
import zlib

import numpy as np
import pytest
import torch

from geolip.modular_diffusion.beds import PicturesBed
from geolip.modular_diffusion.build import content_hash
from geolip.modular_diffusion.forms import cluster_build
from geolip.modular_diffusion.linker import export_build, link
from geolip.modular_diffusion.model import Model
from geolip.modular_diffusion.spectral import anchor_velocity, class_statistics, measure_spectrum
from geolip.modular_diffusion.units import NULL_CLASS, Anchor

# ------------------------------------------------------------------ T0: recorded from the code before this pass
BEFORE = {
    "hashes": {"anchor": "36af0a5ab814fc97cd196f9bd00e259558177b2dd9b388bcf9c10a9bbe342595",
               "stem": "a0f7ad8ba82fdfc53afaf4533c86fb656b16eed0983e22c3b5dd90ef98a03876",
               "head": "2a357ed5d757c0e658f72a158608f62ed6955b19635a10f1fe3379e1f2095f0b"},
    "velocity_sum": -20.795355619862676,
    "velocity_first8": [1.0983805656433105, 1.124253749847412, 0.24446693062782288, 0.4232964515686035,
                        -0.828010082244873, -0.6751306056976318, 0.3083052635192871, 2.063628673553467],
    "h_sum": 2147.117268435657,
    "h_first8": [0.9509532451629639, -0.1592814326286316, -0.08762997388839722, -1.4281154870986938,
                 0.15712426602840424, -0.09534017741680145, 0.0917326807975769, 0.03877323120832443],
    "batch_index_sum": 264248,
    "batch_indices": [36044, 48239, 14933, 43760, 48963, 48379, 5427, 18503],
    "batch_x_sum": 1319.3762462120503,
    "batch_labels": [8, 5, 4, 2, 0, 4, 3, 6],
}
PICTURE_FIELD = (3, 32, 32)


def test_T0_hashes_and_output_unchanged():
    model = Model.born(cluster_build(PICTURE_FIELD, stem_depth=2), 0, torch.ones(PICTURE_FIELD)).eval()
    assert {u.name: content_hash(u) for u in model.units} == BEFORE["hashes"]
    assert sorted(model.unit("anchor").state_dict()) == ["spectrum"]
    gen = torch.Generator().manual_seed(0)
    z = torch.randn((4,) + PICTURE_FIELD, generator=gen)
    t = torch.tensor([0.1, 0.4, 0.7, 0.95])
    cls = torch.tensor([0, 3, 7, 10])
    with torch.no_grad():
        v = model.velocity(z, t, cls)
        _, h, _, _ = model.components(z, t, cls)
    assert float(v.double().sum()) == BEFORE["velocity_sum"]
    assert v.reshape(-1)[:8].tolist() == BEFORE["velocity_first8"]
    assert float(h.double().sum()) == BEFORE["h_sum"]
    assert h.reshape(-1)[:8].tolist() == BEFORE["h_first8"]


def test_T0_first_batch_unchanged(cifar_npz):
    bed = PicturesBed(cifar_npz, "cpu")
    x, y = bed.train_batch(8, torch.Generator().manual_seed(0))
    idx = torch.randint(0, bed.xtr.shape[0], (8,), generator=torch.Generator().manual_seed(0))
    assert idx.tolist() == BEFORE["batch_indices"] and int(idx.sum()) == BEFORE["batch_index_sum"]
    assert float(x.double().sum()) == BEFORE["batch_x_sum"]
    assert y.tolist() == BEFORE["batch_labels"]
    assert torch.equal(y, bed.ytr[idx])


# ------------------------------------------------------------------ T1: the per-class anchor
SMALL = (1, 8, 8)


def _power_law(slope, seed):
    f = torch.fft.fftfreq(8, dtype=torch.float64)
    r = torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    s = torch.where(r > 0, r.clamp_min(1e-9) ** (-slope), torch.tensor(4.0, dtype=torch.float64))
    return (s / s.mean())[None]                                        # [1, 8, 8], mean 1, symmetric


def _gaussian(n, mean, spectrum, gen):
    w = torch.randn((n,) + SMALL, generator=gen, dtype=torch.float64)
    return mean[None] + torch.fft.ifft2(torch.fft.fft2(w) * spectrum.sqrt()).real


def _anchor(k, stats=None, pooled=None):
    geo = {"field": list(SMALL)}
    return Anchor("anchor", {"site": "anchor"}, {"output": True, "symbols": k}, geo, spectrum=pooled, class_stats=stats)


def test_T1_no_class_row_is_the_pooled_anchor_bit_for_bit():
    gen = torch.Generator().manual_seed(3)
    pooled = torch.rand(SMALL, generator=gen) + 0.2
    m = torch.randn((4,) + SMALL, generator=gen)
    s = torch.rand((4,) + SMALL, generator=gen) + 0.1
    a = _anchor(4, (m, s), pooled)
    z = torch.randn((6,) + SMALL, generator=gen)
    t = torch.tensor([0.0, 0.1, 0.35, 0.6, 0.9, 1.0])
    ref = anchor_velocity(z, t, pooled.float())
    for cls in (None, torch.full((6,), 4), torch.full((6,), NULL_CLASS)):
        assert torch.equal(a(z, t, cls), ref)
    mixed = a(z, t, torch.tensor([0, 4, 1, 10, 3, 4]))
    assert torch.equal(mixed[[1, 3, 5]], ref[[1, 3, 5]])
    assert not torch.equal(mixed[0], ref[0])
    plain = Anchor("anchor", {"site": "anchor"}, {"output": True}, {"field": list(SMALL)}, spectrum=pooled)
    assert torch.equal(plain(z, t, torch.zeros(6, dtype=torch.long)), ref)          # no symbols: the class is ignored


def test_T1_two_gaussian_classes_reach_the_best_affine_predictor():
    gen = torch.Generator().manual_seed(11)
    ii, jj = torch.meshgrid(torch.arange(8.0), torch.arange(8.0), indexing="ij")
    mu = [torch.zeros(SMALL, dtype=torch.float64) + 0.8 * torch.cos(ii / 3.0)[None].double(),
          torch.zeros(SMALL, dtype=torch.float64) - 0.6 + 0.0 * jj[None].double()]
    spec = [_power_law(1.0, 0) * 1.5, _power_law(2.0, 1) * 0.6]
    n = 200_000
    fit = [_gaussian(n, mu[c], spec[c], gen) for c in range(2)]
    x = torch.cat(fit)
    y = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
    means, spectra = class_statistics(x, y, 2, flips=False)
    pooled = measure_spectrum(x)
    a = _anchor(2, (means, spectra), pooled)
    for t_val in (0.3, 0.7):
        for c in range(2):
            x1 = _gaussian(n, mu[c], spec[c], gen)
            z0 = torch.randn(x1.shape, generator=gen, dtype=torch.float64)
            zt = (1 - t_val) * z0 + t_val * x1
            v = x1 - z0
            tt = torch.full((n,), t_val)
            err_class = float(((a(zt.float(), tt, torch.full((n,), c)).double() - v) ** 2).mean())
            err_pooled = float(((a(zt.float(), tt, None).double() - v) ** 2).mean())
            # the best affine predictor: a float64 least-squares fit of v on [z_t, 1] over these rows (in sample)
            zf = torch.cat([zt.reshape(n, -1), torch.ones(n, 1, dtype=torch.float64)], dim=1)
            vf = v.reshape(n, -1)
            w = torch.linalg.lstsq(zf, vf).solution
            err_best = float(((zf @ w - vf) ** 2).mean())
            assert abs(err_class - err_best) / err_best < 1e-3, (t_val, c, err_class, err_best)
            assert err_class < err_pooled, (t_val, c, err_class, err_pooled)


def test_T1_class_statistics_estimator_and_flips():
    gen = torch.Generator().manual_seed(5)
    x = torch.randn((40, 2, 8, 8), generator=gen)
    y = torch.tensor([0] * 15 + [2] * 25)
    pooled = torch.full((2, 8, 8), 7.0, dtype=torch.float64)
    m, s = class_statistics(x, y, 3, flips=True, pooled=pooled, chunk=7)
    for c in (0, 2):
        rows = torch.cat([x[y == c], x[y == c].flip(-1)]).double()
        mu = rows.mean(0)
        assert torch.allclose(m[c].double(), mu, atol=1e-6)
        assert torch.allclose(s[c].double(), measure_spectrum(rows - mu), rtol=1e-5, atol=1e-6)
    assert m.dtype == torch.float32 and s.dtype == torch.float32
    assert torch.equal(m[1], torch.zeros(2, 8, 8)) and torch.equal(s[1], pooled.float())
    m0, _ = class_statistics(x, y, 3, flips=False, pooled=pooled)
    assert torch.allclose(m0[0].double(), x[y == 0].double().mean(0), atol=1e-6)
    with pytest.raises(ValueError):
        class_statistics(x, y, 3, flips=False)                     # class 1 is empty and no pooled spectrum


def test_T1_symbols_round_trip_through_export_and_link(synthetic_npz, tmp_path):
    bed = PicturesBed(synthetic_npz, "cpu")
    build = cluster_build(bed.field_shape, stem_depth=1, dim=32, anchor_symbols=10)
    with pytest.raises(ValueError):
        Model.born(build, 0, bed.spectrum)                         # the class statistics are required
    model = Model.born(build, 0, bed.spectrum, class_stats=bed.class_stats())
    assert sorted(model.anchor.state_dict()) == ["means", "spectra", "spectrum"]
    assert torch.equal(model.anchor.spectra[10], model.anchor.spectrum)
    assert torch.equal(model.anchor.means[10], torch.zeros(bed.field_shape))
    out = export_build(model, str(tmp_path / "m"), seed=0)
    again = link(str(tmp_path / "m"))
    assert {u.name: content_hash(u) for u in again.units} == {u.name: content_hash(u) for u in model.units}
    z, cls = bed.heldout(6)
    t = torch.linspace(0.05, 0.95, 6)
    with torch.no_grad():
        assert torch.equal(again.velocity(z, t, cls), model.velocity(z, t, cls))
        # the stem's anchor input reads the class anchor's answer: two classes give two different trunk states
        _, h0, _, _ = model.components(z, t, torch.zeros(6, dtype=torch.long))
        v_anchor = model.anchor_velocity(z, t, torch.zeros(6, dtype=torch.long))
        assert not torch.equal(v_anchor, model.anchor_velocity(z, t))
    assert out.entry("anchor").spec == {"output": True, "symbols": 10}


# ------------------------------------------------------------------ T2: the stem's class window
def _live(model, salt=""):
    with torch.no_grad():
        for u in model.units:
            for k, p in u.named_parameters():
                g = torch.Generator().manual_seed(zlib.crc32(f"{salt}{u.name}/{k}".encode()))
                p.add_(torch.randn(p.shape, generator=g) * 0.05)


def test_T2_class_window():
    field = (3, 16, 16)
    spec = torch.rand(field, generator=torch.Generator().manual_seed(2), dtype=torch.float64) + 0.1
    gen = torch.Generator().manual_seed(4)
    z = torch.randn((8,) + field, generator=gen)
    t = torch.tensor([0.0, 0.1, 0.2, 0.2499, 0.25, 0.3, 0.7, 1.0])
    c1 = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    c2 = torch.tensor([9, 8, 7, 6, 5, 4, 3, 2])
    never = Model.born(cluster_build(field, stem_depth=2, dim=32, grid=8, class_window=["0", "0"]), 0, spec)
    _live(never)
    assert never.stem.spec["class_window"] == ["0", "0"]
    with torch.no_grad():
        assert torch.equal(never.velocity(z, t, c1), never.velocity(z, t, c2))
        assert torch.equal(never.velocity(z, t, c1), never.velocity(z, t, torch.full_like(c1, NULL_CLASS)))
    start = Model.born(cluster_build(field, stem_depth=2, dim=32, grid=8, class_window=["0", "0.25"]), 0, spec)
    _live(start)
    with torch.no_grad():
        v1, v2 = start.velocity(z, t, c1), start.velocity(z, t, c2)
    late = t >= 0.25
    assert torch.equal(v1[late], v2[late])
    for i in torch.nonzero(~late).squeeze(1).tolist():
        assert not torch.equal(v1[i], v2[i]), i
    # without the key: the build of earlier versions (the same spec, the same hashes)
    plain = cluster_build(field, stem_depth=2, dim=32, grid=8)
    assert "class_window" not in plain.entry("stem").spec and "symbols" not in plain.entry("anchor").spec


# ------------------------------------------------------------------ T3: class-restricted streams
def test_T3_restricted_streams(synthetic_npz):
    whole = PicturesBed(synthetic_npz, "cpu")
    bed = PicturesBed(synthetic_npz, "cpu", without_classes=(9, 3))
    only = whole.restricted(only_classes=(9,))
    for step in range(30):
        _, y = bed.train_batch(16, torch.Generator().manual_seed(step))
        assert not bool(((y == 9) | (y == 3)).any())
        _, y9 = only.train_batch(16, torch.Generator().manual_seed(step))
        assert bool((y9 == 9).all())
    # the whole set's numbers are kept
    assert torch.equal(bed.mean, whole.mean) and torch.equal(bed.std, whole.std)
    assert torch.equal(bed.spectrum, whole.spectrum)
    assert torch.equal(bed.train_pictures(), whole.train_pictures())
    assert torch.equal(bed.heldout(50)[0], whole.heldout(50)[0])
    xte, yte = bed.test_pictures()
    assert xte.shape[0] == 300 and yte.shape[0] == 300
    # with nothing excluded the draws are exactly the unrestricted ones
    free = whole.restricted()
    a = whole.train_batch(8, torch.Generator().manual_seed(1))
    b = free.train_batch(8, torch.Generator().manual_seed(1))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    # class statistics of a restricted bed: the excluded classes get the pooled row
    m, s = bed.class_stats()
    pooled = whole.spectrum.float()
    for c in (3, 9):
        assert torch.equal(m[c], torch.zeros(bed.field_shape)) and torch.equal(s[c], pooled)
    mw, sw = whole.class_stats()
    for c in (0, 1, 2, 4):
        assert torch.equal(m[c], mw[c]) and torch.equal(s[c], sw[c])
    assert not torch.equal(sw[9], pooled)
    assert bed.class_stats() is bed.class_stats()                  # computed once
