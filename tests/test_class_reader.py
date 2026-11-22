"""The class reader (T5): a ridge map from the judge's features to the classes, in closed form from the reference
statistics; equal to a direct ridge fit, right on separated classes, at chance on shuffled ones; in the picture score;
an old reference cache upgraded and written whole."""
import os

import numpy as np
import pytest
import torch

from geolip.modular_diffusion import score
from geolip.modular_diffusion.beds import PicturesBed
from geolip.modular_diffusion.forms import cluster_build
from geolip.modular_diffusion.model import Model


def _ref_of(feats, labels, k):
    """The reference statistics of features, the way the reference pass keeps them (computed here directly)."""
    mu = feats.mean(axis=0)
    cov = np.cov(feats, rowvar=False, ddof=1)
    class_n = np.array([(labels == c).sum() for c in range(k)], dtype=np.int64)
    class_mu = np.stack([feats[labels == c].mean(axis=0) for c in range(k)])
    return {"mu": mu, "cov": cov, "n": feats.shape[0], "class_mu": class_mu, "class_n": class_n}


def test_T5_equals_a_direct_ridge_fit():
    rng = np.random.default_rng(0)
    n, d, k = 3000, 12, 4
    labels = rng.integers(0, k, n)
    feats = rng.standard_normal((n, d)) @ rng.standard_normal((d, d)) + 0.7 * rng.standard_normal((k, d))[labels]
    ridge = 1e-2
    reader = score.class_reader(_ref_of(feats, labels, k), ridge=ridge)
    # direct: centered features against centered one-hot classes, penalty (n - 1) ridge mean(diag(cov))
    fc = feats - feats.mean(axis=0)
    y = np.eye(k)[labels]
    yc = y - y.mean(axis=0)
    cov = fc.T @ fc / (n - 1)
    lam = (n - 1) * ridge * np.mean(np.diag(cov))
    w = np.linalg.solve(fc.T @ fc + lam * np.eye(d), fc.T @ yc)
    assert np.abs(reader["w"] - w).max() <= 1e-8 * max(1.0, np.abs(w).max())
    assert np.allclose(reader["prior"], y.mean(axis=0), rtol=0, atol=1e-15)
    pred_direct = np.argmax(fc @ w + y.mean(axis=0), axis=1)
    assert (score.read_classes(feats, reader) == pred_direct).all()


def test_T5_separated_classes_and_shuffled_labels():
    rng = np.random.default_rng(1)
    n, d, k = 4000, 16, 10
    labels = rng.integers(0, k, n)
    centers = 10.0 * rng.standard_normal((k, d))
    feats = centers[labels] + rng.standard_normal((n, d))
    reader = score.class_reader(_ref_of(feats, labels, k))
    test_labels = rng.integers(0, k, 2000)
    test = centers[test_labels] + rng.standard_normal((2000, d))
    r = score.class_read(test, test_labels, reader)
    assert r["accuracy"] == 1.0 and r["n"] == 2000 and len(r["by_class"]) == k
    shuffled = rng.permutation(labels)
    blind = score.class_reader(_ref_of(feats, shuffled, k))
    r2 = score.class_read(test, test_labels, blind)
    assert abs(r2["accuracy"] - 1.0 / k) < 0.05, r2["accuracy"]


def test_T5_picture_score_with_a_reader_and_the_reader_check(synthetic_npz, tmp_path):
    bed = PicturesBed(synthetic_npz, "cpu")
    judge = score.FakeJudge()
    path = str(tmp_path / "ref.npz")
    ref = score.reference_classes(bed, judge, path, batch=256)
    assert ref["fresh"] and ref["class_mu"].shape == (10, judge.dim) and int(ref["class_n"].sum()) == 600
    again = score.reference_classes(bed, judge, path, batch=256)
    assert not again["fresh"] and np.array_equal(again["class_mu"], ref["class_mu"])
    mu, cov, n = score.reference_stats(bed, judge, path)                  # the same file serves the plain statistics
    assert np.array_equal(mu, ref["mu"]) and np.array_equal(cov, ref["cov"]) and n == 600
    reader = score.class_reader(ref)
    model = Model.born(cluster_build(bed.field_shape, stem_depth=1, dim=32), 0, bed.spectrum)
    res = score.picture_score(model, bed, judge, (mu, cov, n), n=20, steps=2, batch=10, reader=reader)
    cr = res["class_read"]
    assert cr["n"] == 20 and 0.0 <= cr["accuracy"] <= 1.0 and len(cr["by_class"]) == 10
    plain = score.picture_score(model, bed, judge, (mu, cov, n), n=20, steps=2, batch=10)
    assert "class_read" not in plain and plain["score"] == res["score"]
    chk = score.reader_check(bed, judge, reader, batch=100)
    assert chk["n"] == 300 and 0.0 <= chk["test_accuracy"] <= 1.0
    # the check computed directly: features of the test pictures, the read class by the formula
    x, y = bed.test_pictures()
    feats = judge.features(score.to_pictures(x, bed))
    pred = np.argmax((feats - reader["mu"]) @ reader["w"] + reader["prior"], axis=1)
    assert chk["test_accuracy"] == float((pred == y.numpy()).mean())


def test_T5_an_old_reference_cache_is_upgraded_and_written_whole(synthetic_npz, tmp_path):
    bed = PicturesBed(synthetic_npz, "cpu")
    judge = score.FakeJudge()
    folder = tmp_path / "ref"
    folder.mkdir()
    path = str(folder / "reference.npz")
    feats = np.concatenate([judge.features(score.to_pictures(bed.train_pictures()[i:i + 256], bed))
                            for i in range(0, 600, 256)])
    mu, cov, n = score.feature_stats(feats)
    np.savez(path, mu=mu, cov=cov, n=np.int64(n), judge=np.str_(judge.name))         # the format of earlier versions
    with np.load(path) as d:
        assert "class_mu" not in d.files
    assert np.array_equal(score.reference_stats(bed, judge, path)[0], mu)             # an old cache still reads
    ref = score.reference_classes(bed, judge, path, batch=256)
    assert ref["fresh"]
    assert sorted(os.listdir(folder)) == ["reference.npz"]                           # no temporary file left beside it
    with np.load(path) as d:
        assert {"mu", "cov", "n", "judge", "class_mu", "class_n"} <= set(d.files)
        assert np.allclose(d["mu"], mu) and np.allclose(d["cov"], cov)
        labels = bed.train_labels().numpy()
        for c in range(10):
            assert np.allclose(d["class_mu"][c], feats[labels == c].mean(axis=0))
            assert int(d["class_n"][c]) == int((labels == c).sum())
    with pytest.raises(ValueError):
        score.reference_classes(bed, score.FakeJudge(dim=8), path)                    # another judge is refused
