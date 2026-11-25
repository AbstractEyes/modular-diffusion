"""Capture, the chain read: an assembly against the teacher on the states of the sampler's own chain and on the
teacher's chain. The teacher read against itself is exactly 0 at every step on both chains; the own chain ends where
``sample_probe`` of the same assembly ends and the teacher's chain where the teacher's own does, to the bit; a part that
differs from the teacher reads above 0; the numbers equal a slow reference written here; a same-seed repeat is
identical; the experiment's variant writes the read and agrees with the assembly variant's final latents."""
import json
import math

import pytest
import torch

pytest.importorskip("diffusers")        # the capture tests need the optional extra

from geolip.modular_diffusion.capture import BANDS, reads  # noqa: E402
from geolip.modular_diffusion.capture.assemble import Assembly, ddim_timesteps, guided, initial_noise  # noqa: E402
from geolip.modular_diffusion.capture.engine import as_part, build_engine  # noqa: E402
from geolip.modular_diffusion.capture.schedule import band_weights, sd15_alphas_cumprod  # noqa: E402
from geolip.modular_diffusion.capture.teacher import tiny_teacher  # noqa: E402
from geolip.modular_diffusion.experiments import capture  # noqa: E402

PROMPTS = ["a red house", "two boats", "a cat"]
STEPS, GUIDANCE, GROUP = 10, 3.0, 2          # 3 rows in groups of 2: an uneven last group


@pytest.fixture(scope="module")
def teacher():
    return tiny_teacher(0)


@pytest.fixture(scope="module")
def probe(teacher):
    states = teacher.text_states(teacher.tokenize(PROMPTS))
    return reads.make_probe(states, seed=17, per_band=2, latent_shape=(4, 16, 16), empty=teacher.empty_states(1))


@pytest.fixture(scope="module")
def mixed(teacher):
    """The teacher in the clean and noisy bands, the born tiny engine in the middle band."""
    return Assembly({"clean": teacher, "middle": build_engine(teacher, "tiny").eval(), "noisy": teacher})


def test_the_teacher_against_itself_reads_zero_at_every_step_on_both_chains(teacher, probe):
    for model in (teacher, Assembly({b: teacher for b in BANDS})):
        r = reads.chain_read(model, teacher, probe, STEPS, GUIDANCE, GROUP)
        assert r["own_per_step"] == [0.0] * STEPS and r["teacher_per_step"] == [0.0] * STEPS
        for b in BANDS:
            x = r["bands"][b]
            assert x["own_mean"] == 0.0 and x["teacher_mean"] == 0.0
            assert x["ratio"] is None and x["rows_in_ratio"] == 0 and x["rows_left_out"] == len(PROMPTS)
        assert r["final_latent_change"] == 0.0 and r["own_final_hash"] == r["teacher_final_hash"]


def test_the_chains_end_where_the_sampler_ends(teacher, probe, mixed):
    e_own, z_own = reads.chain_errors(mixed, teacher, probe, STEPS, GUIDANCE, GROUP)
    e_tea, z_tea = reads.chain_errors(teacher, mixed, probe, STEPS, GUIDANCE, GROUP)
    want_own = reads.sample_probe(mixed, probe, STEPS, GUIDANCE, GROUP)
    want_tea = reads.sample_probe(teacher, probe, STEPS, GUIDANCE, GROUP)
    assert torch.equal(z_own, want_own) and torch.equal(z_tea, want_tea)
    r = reads.chain_read(mixed, teacher, probe, STEPS, GUIDANCE, GROUP)
    assert r["own_final_hash"] == reads.latent_hash(want_own) and r["teacher_final_hash"] == reads.latent_hash(want_tea)
    assert r["final_latent_change"] == float((want_own.double() - want_tea.double()).pow(2).mean())
    assert e_own.shape == e_tea.shape == (STEPS, len(PROMPTS)) and e_own.dtype == torch.float64


def test_a_part_that_differs_reads_above_zero(teacher, probe, mixed):
    r = reads.chain_read(mixed, teacher, probe, STEPS, GUIDANCE, GROUP)
    w = band_weights(ddim_timesteps(STEPS))
    for k in range(STEPS):
        live = float(w[k, BANDS.index("middle")]) > 0
        assert (r["own_per_step"][k] > 0) == live and (r["teacher_per_step"][k] > 0) == live, k
    m = r["bands"]["middle"]
    assert m["own_mean"] > 0 and m["teacher_mean"] > 0 and m["ratio"] is not None and m["rows_in_ratio"] == 3
    assert m["log_ratio_se"] is not None and m["log_ratio_se"] >= 0
    assert r["final_latent_change"] > 0 and r["own_final_hash"] != r["teacher_final_hash"]


def _reference(driver, asked, probe):
    """A plain loop: the sampler's steps written out here, group by group; e per step and per row."""
    ac = sd15_alphas_cumprod().double()
    ts = ddim_timesteps(STEPS)
    ratio = 1000 // STEPS
    n = probe.states.shape[0]
    e = [[None] * n for _ in ts]
    finals = []
    d_part, a_part = as_part(driver), as_part(asked)
    for j, (first, rows) in enumerate(reads.sample_groups(n, GROUP)):
        states = probe.states[first:first + rows]
        z = initial_noise(reads.group_seed(probe.seed, j), rows, (4, 16, 16)).float()
        for k, t in enumerate(ts):
            tt = torch.full((rows,), t, dtype=torch.long)
            eps = guided(d_part, z, tt, states, probe.empty, GUIDANCE)
            other = guided(a_part, z, tt, states, probe.empty, GUIDANCE)
            for i in range(rows):
                diff = eps[i].double() - other[i].double()
                e[k][first + i] = float((diff * diff).sum()) / diff.numel()
            a = float(ac[t])
            a_prev = float(ac[t - ratio]) if t - ratio >= 0 else float(ac[0])
            x0 = (z - math.sqrt(1.0 - a) * eps) / math.sqrt(a)
            z = math.sqrt(a_prev) * x0 + math.sqrt(1.0 - a_prev) * eps
        finals.append(z)
    return e, torch.cat(finals)


def _close(a, b):
    return a == b or abs(a - b) <= 1e-12 * max(abs(a), abs(b))


def test_the_read_equals_a_slow_reference(teacher, probe, mixed):
    r = reads.chain_read(mixed, teacher, probe, STEPS, GUIDANCE, GROUP)
    e_own, z_own = _reference(mixed, teacher, probe)
    e_tea, z_tea = _reference(teacher, mixed, probe)
    assert torch.equal(z_own, reads.sample_probe(mixed, probe, STEPS, GUIDANCE, GROUP))
    ts = ddim_timesteps(STEPS)
    w = band_weights(ts)
    n = len(PROMPTS)
    for k in range(STEPS):
        assert _close(r["own_per_step"][k], sum(e_own[k]) / n) and _close(r["teacher_per_step"][k], sum(e_tea[k]) / n)
    for bi, b in enumerate(BANDS):
        ws = [float(w[k, bi]) for k in range(STEPS)]
        own = [sum(ws[k] * e_own[k][i] for k in range(STEPS)) / sum(ws) for i in range(n)]
        tea = [sum(ws[k] * e_tea[k][i] for k in range(STEPS)) / sum(ws) for i in range(n)]
        x = r["bands"][b]
        assert all(_close(p, q) for p, q in zip(r["own_per_row"][b], own))
        assert all(_close(p, q) for p, q in zip(r["teacher_per_row"][b], tea))
        assert _close(x["own_mean"], sum(own) / n) and _close(x["teacher_mean"], sum(tea) / n)
        logs = [math.log(p / q) for p, q in zip(own, tea) if p != 0.0 and q != 0.0]
        assert x["rows_in_ratio"] == len(logs) and x["rows_left_out"] == n - len(logs)
        if not logs:
            assert x["ratio"] is None and x["log_ratio_se"] is None
            continue
        m = sum(logs) / len(logs)
        se = math.sqrt(sum((v - m) ** 2 for v in logs) / (len(logs) - 1)) / math.sqrt(len(logs))
        assert _close(x["log_ratio_mean"], m) and _close(x["ratio"], math.exp(m)) and _close(x["log_ratio_se"], se)
    assert r["bands"]["middle"]["ratio"] is not None and r["bands"]["clean"]["rows_in_ratio"] in (0, n)


def test_a_same_seed_repeat_is_identical(teacher, probe, mixed):
    a = reads.chain_read(mixed, teacher, probe, STEPS, GUIDANCE, GROUP)
    b = reads.chain_read(mixed, teacher, probe, STEPS, GUIDANCE, GROUP)
    a.pop("seconds")
    b.pop("seconds")
    assert a == b


def test_the_log_ratio_leaves_out_zero_rows():
    r = reads.log_ratio([2.0, 0.0, 8.0], [1.0, 1.0, 2.0])
    assert r["rows_in_ratio"] == 2 and r["rows_left_out"] == 1
    assert _close(r["ratio"], math.exp((math.log(2.0) + math.log(4.0)) / 2))
    one = reads.log_ratio([2.0], [1.0])
    assert one["ratio"] == 2.0 and one["log_ratio_se"] is None
    assert reads.log_ratio([0.0, 0.0], [0.0, 1.0])["ratio"] is None


def test_the_chain_read_variant(tmp_path):
    common = dict(tiny=1, probe_n=3, probe_per_band=2, sample_steps=5, group=2, threads=2)
    same = capture.main(variant="chain_read", parts="clean=teacher,middle=teacher,noisy=teacher",
                        out=str(tmp_path / "same"), **common)
    assert same["chain"]["own_per_step"] == [0.0] * 5 and same["final_latent_change"] == 0.0
    job = tmp_path / "job"
    capture.main(variant="band_engine", shape="tiny", band="middle", steps=2, batch=2, out=str(job), log_every=0,
                 **common)
    parts = f"clean=teacher,middle={job},noisy=born:tiny"
    r = capture.main(variant="chain_read", parts=parts, out=str(tmp_path / "chain"), **common)
    asm = capture.main(variant="assembly", parts=parts, out=str(tmp_path / "asm"), **common)
    assert r["final_latent_change"] == asm["final_latent_change"] > 0.0     # the distance read's sampled latents
    assert r["chain"]["bands"]["middle"]["ratio"] is not None and r["chain"]["rows"] == 3
    with open(tmp_path / "chain" / "result.json", encoding="utf-8") as fh:
        assert json.load(fh)["chain"]["bands"] == r["chain"]["bands"]
    with open(tmp_path / "chain" / "result.md", encoding="utf-8") as fh:
        assert "teacher's chain" in fh.read()
    with pytest.raises(ValueError):
        capture.main(variant="chain_read", out=str(tmp_path / "x"), **common)
    with pytest.raises(ValueError):
        capture.main(variant="chain_read", parts=parts, out=str(tmp_path / "y"), **dict(common, sample_steps=0))
