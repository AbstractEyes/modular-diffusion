"""Capture, the reads: the teacher's copy reads 0 and a random engine reads worse than an inherited one (test 3); the
exchange test: replacing one band's part leaves the other bands' answers exactly unchanged on the steps where that band
is silent (test 6); the off-band control; the reads are plain dicts that write as JSON and a table."""
import json

import pytest
import torch

pytest.importorskip("diffusers")

from geolip.modular_diffusion.capture import BANDS, reads
from geolip.modular_diffusion.capture.assemble import Assembly
from geolip.modular_diffusion.capture.engine import build_engine, fresh_engine
from geolip.modular_diffusion.capture.schedule import band_steps, band_weight
from geolip.modular_diffusion.capture.teacher import tiny_teacher


@pytest.fixture(scope="module")
def teacher():
    return tiny_teacher(0)


@pytest.fixture(scope="module")
def probe(teacher):
    states = teacher.text_states(teacher.tokenize(["a red house", "two boats", "a cat"]))
    return reads.make_probe(states, seed=11, per_band=4, latent_shape=(4, 16, 16), empty=teacher.empty_states(1))


def test_the_probe_is_fixed_and_even(teacher, probe):
    for b in BANDS:
        own = band_steps(b)
        st = probe.steps[b]
        assert len(st) == 4 and int(st[0]) == int(own[0]) and int(st[-1]) == int(own[-1])
        assert all(float(band_weight(int(t), b)) > 0 for t in st)
    again = reads.make_probe(probe.states, seed=11, per_band=4, latent_shape=(4, 16, 16))
    assert torch.equal(again.x0, probe.x0) and torch.equal(again.eps, probe.eps)
    assert torch.equal(probe.z(500), again.z(500))


def test_3_the_teachers_copy_reads_zero_and_inheritance_beats_random(teacher, probe):
    for b in BANDS:
        r = reads.band_read(teacher, teacher, b, probe)
        assert r["mse"] == 0.0 and all(v == 0.0 for v in r["per_step"].values())
        assert r["elements"] == 4 * 3 * 4 * 16 * 16
        assert reads.band_read(build_engine(teacher, "full"), teacher, b, probe)["mse"] == 0.0
        for shape in ("base", "small", "tiny"):
            born = reads.born_read(teacher, shape, b, probe)
            rand = reads.band_read(fresh_engine(teacher, shape, seed=1), teacher, b, probe)
            assert 0.0 < born["mse"] < rand["mse"], (b, shape, born["mse"], rand["mse"])
            assert born["shape"] == shape and 0 < born["share"] < 1


def test_6_exchange_moves_nothing_where_the_band_is_silent(teacher, probe):
    parts = {"clean": build_engine(teacher, "base"), "middle": build_engine(teacher, "small"),
             "noisy": build_engine(teacher, "tiny")}
    model = Assembly(parts)
    for band in BANDS:
        other = fresh_engine(teacher, "small", seed=4)
        r = reads.exchange(model, band, other, probe, teacher=teacher, sample_steps=10)
        assert r["band"] == band
        for b in BANDS:
            row = r["bands"][b]
            if b == band:
                assert row["change"] != 0.0
                assert row["silent_steps"] == []
            else:
                assert row["silent_steps"], (band, b)                 # the probe holds steps outside the crossfade
                assert row["largest_change_where_silent"] == 0.0, (band, b)
        assert r["final_latent_change"] > 0.0
        # per step: the other bands' reads are identical, to the bit, where the replaced band is silent
        swapped = dict(parts, **{band: other})
        after = Assembly(swapped)
        for b in BANDS:
            if b == band:
                continue
            r0 = reads.band_read(model, teacher, b, probe)["per_step"]
            r1 = reads.band_read(after, teacher, b, probe)["per_step"]
            for t in r["bands"][b]["silent_steps"]:
                assert r0[str(t)] == r1[str(t)]
    json.dumps(r)


def test_off_band_is_the_control_that_fails(teacher, probe):
    # a part that is the teacher on one band only: an assembly of the teacher there and a random engine elsewhere
    rand = fresh_engine(teacher, "base", seed=7)
    part = Assembly({"clean": teacher, "middle": rand, "noisy": rand})
    r = reads.off_band(part, teacher, "noisy", "clean", probe)
    assert r["served"] == "noisy" and r["trained"] == "clean"
    assert r["off_band_mse"] > r["on_band_mse"] > 0.0
    json.dumps(r)


def test_reads_write_as_json_and_a_table(teacher, probe, tmp_path):
    rows = [dict(reads.born_read(teacher, "base", b, probe), part="base") for b in BANDS]
    path = tmp_path / "reads.json"
    reads.write_json(rows, str(path))
    assert json.loads(path.read_text(encoding="utf-8"))[0]["band"] == "clean"
    text = reads.table(rows, ["band", "part", "mse", "share", "missing"])
    assert text.count("\n") == 5 and "| clean | base |" in text and text.rstrip().endswith("| - |")
