"""Sampling on a time grid: the uniform grid against the plain sampler, stage grids, the division of steps among stages,
grid checks, and the picture score on a grid."""
import pytest
import torch

from geolip.modular_diffusion.beds import PicturesBed
from geolip.modular_diffusion.forms import cluster_build
from geolip.modular_diffusion.model import Model
from geolip.modular_diffusion.sample import (check_grid, euler, euler_grid, sample_sheet, share_counts, stage_grid,
                                             uniform_grid)
from geolip.modular_diffusion.score import FakeJudge, picture_score, reference_stats, sample_pictures

CUTS = (0.0, 0.25, 0.65, 1.0)


@pytest.fixture(scope="module")
def setup(synthetic_npz):
    bed = PicturesBed(synthetic_npz, "cpu", 16)
    model = Model.born(cluster_build(bed.field_shape, stem_depth=1, dim=32), 0, bed.spectrum)
    with torch.no_grad():                                  # the zero-born head speaks, so the steps matter
        gen = torch.Generator().manual_seed(7)
        for p in model.head.parameters():
            p.add_(torch.randn(p.shape, generator=gen) * 0.05)
    return bed, model.eval()


def test_uniform_grid_equals_the_plain_sampler(setup):
    bed, model = setup
    gen = torch.Generator().manual_seed(3)
    z = torch.randn((12,) + tuple(bed.field_shape), generator=gen)
    cls = torch.arange(12) % 10
    for n in (1, 7, 50):
        a = euler(model, z, cls, steps=n)
        b = euler_grid(model, z, cls, uniform_grid(n))
        assert float((a - b).abs().max()) <= 1e-5, n
        assert not torch.equal(a, z)
    t = torch.tensor(uniform_grid(4))                     # a tensor grid is taken as well
    assert torch.equal(euler_grid(model, z, cls, t), euler_grid(model, z, cls, uniform_grid(4)))


def test_stage_grid_points():
    g = stage_grid(CUTS, [2, 4, 1])
    assert len(g) == 8 and g[-1] == 1.0 and g[0] == 0.0
    w = 0.65 - 0.25
    assert g == [0.0, 0.125, 0.25, 0.25 + w / 4, 0.25 + 2 * w / 4, 0.25 + 3 * w / 4, 0.65, 1.0]
    for counts in ([13, 20, 17], [1, 1, 1], [5, 3, 9]):
        g = stage_grid(CUTS, counts)
        assert len(g) == sum(counts) + 1 and g[-1] == 1.0
        for s, c in enumerate(counts):
            assert sum(1 for x in g[:-1] if CUTS[s] <= x < CUTS[s + 1]) == c
    with pytest.raises(ValueError):
        stage_grid(CUTS, [1, 0, 1])
    with pytest.raises(ValueError):
        stage_grid(CUTS, [1, 2])
    with pytest.raises(ValueError):
        stage_grid((0.0, 0.5, 0.4, 1.0), [1, 1, 1])


def test_share_counts():
    assert share_counts((0.25, 0.40, 0.35), 50) == [13, 20, 17]     # a tie of remainders goes to the earlier stage
    assert share_counts((0.25, 0.40, 0.35), 100) == [25, 40, 35]
    assert share_counts((0.50, 0.30, 0.20), 50) == [25, 15, 10]
    assert share_counts((1, 1, 1), 10) == [4, 3, 3]
    assert share_counts((0.98, 0.01, 0.01), 3) == [1, 1, 1]           # every stage keeps at least one step
    assert share_counts((1.0, 0.0, 0.0), 5) == [3, 1, 1]
    for shares in ((0.25, 0.40, 0.35), (0.2, 0.6, 0.2), (0.9, 0.05, 0.05)):
        for n in range(3, 40):
            c = share_counts(shares, n)
            assert sum(c) == n and min(c) >= 1, (shares, n, c)
    with pytest.raises(ValueError):
        share_counts((0.25, 0.40, 0.35), 2)
    with pytest.raises(ValueError):
        share_counts((0.5, -0.1, 0.6), 10)


def test_bad_grids_raise(setup):
    bed, model = setup
    z = torch.zeros((2,) + tuple(bed.field_shape))
    cls = torch.zeros(2, dtype=torch.long)
    for bad in ([0.0, 0.5, 0.5, 1.0], [0.0, 0.6, 0.4, 1.0], [0.1, 0.5, 1.0], [0.0, 0.5, 0.9], [0.0], [],
                torch.zeros(2, 2)):
        with pytest.raises(ValueError):
            euler_grid(model, z, cls, bad)
    assert check_grid((0, 0.5, 1)) == [0.0, 0.5, 1.0]


def test_picture_score_on_a_grid(setup, tmp_path):
    bed, model = setup
    judge = FakeJudge(dim=16)
    ref = reference_stats(bed, judge, None, batch=256)
    plain = picture_score(model, bed, judge, ref, n=40, steps=6, batch=20)
    assert "grid" not in plain and plain["steps"] == 6
    on_uniform = picture_score(model, bed, judge, ref, n=40, steps=99, batch=20, grid=uniform_grid(6))
    assert on_uniform["steps"] == 6 and on_uniform["grid"] == uniform_grid(6)
    assert abs(on_uniform["score"] - plain["score"]) <= 1e-4 * abs(plain["score"])
    g = stage_grid(CUTS, [1, 3, 2])
    staged = picture_score(model, bed, judge, ref, n=40, batch=20, grid=g)
    assert staged["steps"] == 6 and staged["grid"] == g
    assert set(staged) == {"score", "n", "steps", "seed", "judge", "seconds", "grid"}
    pics = sample_pictures(model, bed, 20, batch=20, grid=g)
    assert torch.equal(pics, sample_pictures(model, bed, 20, steps=3, batch=20, grid=g))
    sample_sheet(model, bed, str(tmp_path / "grid.png"), "cpu", per_class=1, grid=g)
    assert (tmp_path / "grid.png").exists()
