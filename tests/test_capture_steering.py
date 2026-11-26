"""Capture, the steering read: the teacher read as a part reads R = S = G = 0 exactly; a part of random numbers reads a
large steering error; the guided error splits into the error on the unguided side and the steering error times the
guidance (the sampler's own guided answer); the read repeats to the bit with the same seed; its own error is the
training's own-error read (band_read) to the bit, on the training's own probe; an engine that ignores the prompt reads
S = 1 exactly; the variant's part forms, its record of the part's weights and its files."""
import json
import os

import pytest
import torch

pytest.importorskip("diffusers")        # the capture tests need the optional extra

from geolip.modular_diffusion.capture import BANDS, reads  # noqa: E402
from geolip.modular_diffusion.capture.assemble import guided  # noqa: E402
from geolip.modular_diffusion.capture.data import synthetic_shards  # noqa: E402
from geolip.modular_diffusion.capture.engine import build_engine, fresh_engine  # noqa: E402
from geolip.modular_diffusion.capture.teacher import tiny_teacher  # noqa: E402
from geolip.modular_diffusion.experiments import capture  # noqa: E402

PROMPTS = ["a red house", "two boats", "a cat", "a bowl of fruit"]
G = 7.5


@pytest.fixture(scope="module")
def teacher():
    return tiny_teacher(0)


@pytest.fixture(scope="module")
def probe(teacher):
    states = teacher.text_states(teacher.tokenize(PROMPTS))
    return reads.make_probe(states, seed=11, per_band=3, latent_shape=(4, 16, 16), empty=teacher.unguided_states(1),
                            batch=3)


def test_the_teacher_read_as_a_part_reads_zero(teacher, probe):
    for b in BANDS:
        for part in (teacher, build_engine(teacher, "full")):         # the teacher, and its full copy (its own numbers)
            r = reads.steering_read(part, teacher, b, probe, guidance=G)
            assert r["R"] == r["S"] == r["G"] == r["U"] == r["mse"] == 0.0, (b, r)
            assert all(v["R"] == v["S"] == v["G"] == v["U"] == 0.0 for v in r["per_step"].values())
            assert r["P"] > 0.0 and r["sums"]["ref_d"] > 0.0            # the prompt moves the teacher's answer
            assert r["steps"] == [int(t) for t in probe.steps[b]] and r["rows"] == len(PROMPTS)
            assert r["elements"] == len(r["steps"]) * len(PROMPTS) * 4 * 16 * 16
            json.dumps(r)


def test_a_part_of_random_numbers_reads_a_large_steering_error(teacher, probe):
    for b in BANDS:
        rand = reads.steering_read(fresh_engine(teacher, "2222/nomid/3", seed=1), teacher, b, probe, guidance=G)
        born = reads.steering_read(build_engine(teacher, "2222/nomid/3"), teacher, b, probe, guidance=G)
        assert rand["S"] > 0.5 and rand["G"] > 0.5, (b, rand["S"], rand["G"])
        assert 0.0 < born["S"] < rand["S"] and 0.0 < born["R"] < rand["R"], (b, born["S"], rand["S"])


def test_the_guided_error_splits_into_the_unguided_error_and_the_steering_error(teacher, probe):
    part = build_engine(teacher, "tiny")
    empty = reads.unguided_rows(probe)
    for b in BANDS:
        ref = reads.steering_answers(teacher, probe, b)
        own = reads.steering_answers(part, probe, b)
        for t in probe.steps[b]:
            t = int(t)
            (tc, tu), (ec, eu) = [(x.double(), y.double()) for x, y in (ref[t], own[t])]
            lhs = (eu + G * (ec - eu)) - (tu + G * (tc - tu))              # guided_E - guided_T
            rhs = (eu - tu) + G * ((ec - eu) - (tc - tu))                  # err_empty + g (d_E - d_T)
            assert torch.allclose(lhs, rhs, rtol=0.0, atol=1e-12 * float(lhs.abs().max())), (b, t)
            # the read's guided answer is the sampler's (u + g (c - u), one forward of both halves) to float32 rounding:
            # the two forwards hold other batches, so the answers differ by about 2e-6 of their size, which the guided
            # answer multiplies by up to 1 + 2g (measured on this model: 1.1e-5 of the largest guided entry)
            z = probe.z(t)
            tt = torch.full((z.shape[0],), t, dtype=torch.long)
            with torch.no_grad():
                want = guided(part, z, tt, probe.states, empty[:1], G).double()
            assert torch.allclose(eu + G * (ec - eu), want, rtol=0.0, atol=1e-4 * float(want.abs().max())), (b, t)
        r = reads.steering_read(part, teacher, b, probe, guidance=G, reference=ref)
        s = r["sums"]
        split = s["err_u"] + 2.0 * G * s["cross"] + G * G * s["err_d"]
        assert abs(s["err_g"] - split) <= 1e-9 * s["err_g"], (b, s["err_g"], split)
        one = reads.steering_read(part, teacher, b, probe, guidance=1.0, reference=ref)
        assert abs(one["G"] - one["R"]) <= 1e-9 * one["R"]               # guidance 1: the prompted answer itself
        assert one["S"] == r["S"] and one["R"] == r["R"]                  # S and R do not depend on the guidance


def test_the_own_error_is_the_trainings_band_read_to_the_bit(teacher, probe):
    for b in BANDS:
        for part in (build_engine(teacher, "2222/nomid/3"), fresh_engine(teacher, "tiny", seed=2)):
            r = reads.steering_read(part, teacher, b, probe, guidance=G)
            br = reads.band_read(part, teacher, b, probe)
            assert r["mse"] == br["mse"] and r["elements"] == br["elements"]
            for t, v in r["per_step"].items():
                assert v["mse"] == pytest.approx(br["per_step"][t], rel=1e-12)


def test_an_engine_that_ignores_the_prompt_reads_a_steering_error_of_one(teacher, probe):
    for b in BANDS:
        r = reads.steering_read(build_engine(teacher, "tiny", text=False), teacher, b, probe, guidance=G)
        assert r["S"] == 1.0 and r["sums"]["err_d"] == r["sums"]["ref_d"]
        assert all(v["S"] == 1.0 for v in r["per_step"].values())


def test_unguided_rows_need_the_probes_empty_states(teacher, probe):
    rows = reads.unguided_rows(probe)
    assert rows.shape == probe.states.shape and torch.equal(rows[0], rows[-1])
    assert torch.equal(rows[:1], teacher.unguided_states(1))
    bare = reads.make_probe(probe.states, seed=11, per_band=2, latent_shape=(4, 16, 16))
    with pytest.raises(ValueError, match="unguided"):
        reads.steering_read(teacher, teacher, "clean", bare)


# ------------------------------------------------------------------ the variant
def _rows(tmp_path, n=10):
    folder = str(tmp_path / "rows")
    synthetic_shards(folder, rows=n, shard_size=4, size=16, seed=9)
    return folder


def _common(folder):
    # the probe arguments of a training's own reads: rows in file order, a probe seed, steps a band, rows a forward
    return dict(tiny=1, threads=2, probe_data=folder, probe_first=2, probe_rows=5, probe_seed=1000, probe_per_band=3,
                chunk=2)


def test_the_steering_variant_on_a_trainings_probe_reads_its_own_error_read(tmp_path):
    folder = _rows(tmp_path)
    common = _common(folder)
    job = tmp_path / "job"
    trained = capture.main(variant="band_engine", shape="tiny", band="middle", steps=3, batch=2, seed=0, read_every=2,
                           keep_checkpoint=0, log_every=0, out=str(job), **common)
    res = {}
    for b in BANDS:
        r = res[b] = capture.main(variant="steering", part=str(job), band=b, seed=0, out=str(tmp_path / f"steer_{b}"),
                                  **common)
        assert r["steering"]["mse"] == trained["reads"][f"trained_{b}"]["mse"], b      # the same read, to the bit
        assert r["probe"] == trained["probe"] and r["probe_steps"] == trained["probe_steps"]
        assert r["steering"]["steps"] == trained["probe_steps"][b] and r["band"] == b
        assert r["weights"] == res["clean"]["weights"]                                  # the same part in every band
    r = res["clean"]
    assert r["steering"]["guidance"] == 7.5                                             # the default guidance
    w = r["weights"]
    assert w["kind"] == "job" and w["job"] == "job" and w["shape"] == "tiny" and w["text"] is True
    assert w["engine_sha256"] == capture.file_sha256(str(job / "engine.safetensors"))
    assert w["engine_avg_sha256"] == capture.file_sha256(str(job / "engine_avg.safetensors"))
    assert w["engine_sha256"] != w["engine_avg_sha256"] and len(w["state_sha256"]) == 64
    out = tmp_path / "steer_clean"
    stored = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert stored["steering"] == r["steering"] and stored["variant"] == "steering" and stored["part"] == str(job)
    assert all(stored[k] >= 0.0 for k in ("seconds_teacher", "seconds_part_load", "seconds_part", "seconds"))
    md = (out / "result.md").read_text(encoding="utf-8")
    assert "| band | part | R | S | G | P | U | mse |" in md and "## by probe step" in md
    assert (out / "log.txt").read_text(encoding="utf-8").rstrip().endswith("done")
    assert stored["rows"][0]["S"] == r["steering"]["S"]


def test_the_steering_variant_part_forms_and_the_same_seed(tmp_path):
    folder = _rows(tmp_path)
    common = dict(_common(folder), band="noisy", guidance=5.0)
    t = capture.main(variant="steering", part="teacher", out=str(tmp_path / "t"), **common)
    assert t["steering"]["R"] == t["steering"]["S"] == t["steering"]["G"] == 0.0
    assert t["weights"] == {"kind": "teacher", "part": "teacher"} and t["steering"]["guidance"] == 5.0
    born = capture.main(variant="steering", part="born:2222/nomid/3", out=str(tmp_path / "b"), **common)
    assert born["weights"]["kind"] == "born" and born["weights"]["shape"] == "2222/nomid/3" and born["weights"]["text"]
    assert born["steering"]["P"] == t["steering"]["P"]                      # P is the teacher's: the same in every read
    teacher = tiny_teacher(0)
    assert born["weights"]["state_sha256"] == capture.state_sha256(build_engine(teacher, "2222/nomid/3"))
    one = capture.main(variant="steering", part="random:2222/nomid/3", seed=0, out=str(tmp_path / "r1"), **common)
    two = capture.main(variant="steering", part="random:2222/nomid/3", seed=0, out=str(tmp_path / "r2"), **common)
    assert one["steering"] == two["steering"] and one["weights"] == two["weights"]       # the same seed: to the bit
    assert one["weights"]["kind"] == "random" and one["weights"]["seed"] == 0
    assert one["weights"]["state_sha256"] == capture.state_sha256(fresh_engine(teacher, "2222/nomid/3", seed=0))
    other = capture.main(variant="steering", part="random:2222/nomid/3", seed=1, out=str(tmp_path / "r3"), **common)
    assert other["weights"]["state_sha256"] != one["weights"]["state_sha256"]
    assert other["steering"]["S"] != one["steering"]["S"] and min(one["steering"]["S"], other["steering"]["S"]) > 0.5
    notext = capture.main(variant="steering", part="random:tiny:notext", seed=0, out=str(tmp_path / "r4"), **common)
    assert notext["weights"]["text"] is False and notext["steering"]["S"] == 1.0
    for bad in ("random:", "random:huge", "random:tiny:other", "random::notext"):
        with pytest.raises(ValueError, match="random:<shape>:notext"):
            capture.main(variant="steering", part=bad, out=str(tmp_path / "x"), **common)
    with pytest.raises(ValueError, match="part="):
        capture.main(variant="steering", out=str(tmp_path / "y"), **common)
    with pytest.raises(ValueError, match="engine.json"):                    # a folder that is not a band engine's job
        capture.main(variant="steering", part=str(tmp_path / "rows"), out=str(tmp_path / "y"), **common)
    assert not os.path.exists(tmp_path / "y")                               # refused before anything was loaded
