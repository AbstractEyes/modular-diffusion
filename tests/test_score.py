"""The picture score (test 10): the Frechet distance in closed cases, deterministic class-balanced sampling, the test
judge end to end, the reference cache. The Inception judge is never built here."""
import numpy as np
import pytest
import torch

from geolip.modular_diffusion.beds import PicturesBed
from geolip.modular_diffusion.forms import cluster_build
from geolip.modular_diffusion.model import Model
from geolip.modular_diffusion.score import (FakeJudge, feature_stats, frechet_distance, picture_score, reference_stats,
                                            sample_pictures, score_classes, to_pictures)


def _cov(d=6, seed=0):
    a = np.random.default_rng(seed).standard_normal((d, d))
    return a @ a.T + 0.1 * np.eye(d)


def test_10_frechet_closed_cases():
    c = _cov()
    mu = np.arange(6.0)
    assert abs(frechet_distance(mu, c, mu, c)) < 1e-9
    d = np.array([1.0, -2.0, 0.5, 0, 0, 3.0])
    assert abs(frechet_distance(mu, c, mu + d, c) - float(d @ d)) < 1e-9
    for k in (0.25, 2.0, 9.0):
        want = np.trace(c) * (1 + k - 2 * np.sqrt(k))
        assert abs(frechet_distance(mu, c, mu, k * c) - want) < 1e-8 * max(1.0, want)
    f = np.random.default_rng(1).standard_normal((50, 4))
    m, cv, n = feature_stats(f)
    assert n == 50 and np.allclose(cv, np.cov(f, rowvar=False)) and cv.dtype == np.float64


@pytest.fixture(scope="module")
def setup(synthetic_npz):
    bed = PicturesBed(synthetic_npz, "cpu", 16)
    model = Model.born(cluster_build(bed.field_shape, stem_depth=1, dim=32), 0, bed.spectrum)
    return bed, model


def test_10_sampling_is_deterministic_and_balanced(setup):
    bed, model = setup
    a = sample_pictures(model, bed, 23, steps=2, batch=10, seed=5)
    b = sample_pictures(model, bed, 23, steps=2, batch=10, seed=5)
    c = sample_pictures(model, bed, 23, steps=2, batch=10, seed=6)
    assert a.dtype == torch.uint8 and a.shape == (23, 3, 32, 32)
    assert torch.equal(a, b) and not torch.equal(a, c)
    counts = torch.bincount(score_classes(40), minlength=10)
    assert counts.tolist() == [4] * 10
    pics = to_pictures(bed.train_pictures()[:4], bed)
    assert pics.dtype == torch.uint8 and int(pics.max()) <= 255


def test_10_fake_judge_end_to_end_and_reference_cache(setup, tmp_path):
    bed, model = setup
    judge = FakeJudge(dim=16)
    path = str(tmp_path / "ref.npz")
    ref = reference_stats(bed, judge, path, batch=256)
    assert ref[2] == 600 and ref[0].shape == (16,)
    again = reference_stats(bed, judge, path)
    assert np.array_equal(ref[0], again[0]) and np.array_equal(ref[1], again[1])
    with pytest.raises(ValueError):
        reference_stats(bed, FakeJudge(dim=16, seed=1), path)
    s1 = picture_score(model, bed, judge, ref, n=64, steps=2, batch=32)
    s2 = picture_score(model, bed, judge, ref, n=64, steps=2, batch=32)
    assert set(s1) == {"score", "n", "steps", "seed", "judge", "seconds", "mean_term", "spread_term"}
    assert s1["score"] == s2["score"] and s1["score"] > 0 and s1["n"] == 64 and s1["judge"] == judge.name
    # the distance's two parts: the feature means' squared distance (computed here again) and the rest
    from geolip.modular_diffusion.score import sample_pictures
    feats = judge.features(sample_pictures(model, bed, 64, steps=2, batch=32))
    gap = feats.mean(axis=0) - ref[0]
    assert abs(s1["mean_term"] - float(gap @ gap)) <= 1e-12 * max(1.0, s1["mean_term"]) and s1["mean_term"] >= 0
    assert s1["mean_term"] + s1["spread_term"] == pytest.approx(s1["score"], rel=1e-12)
