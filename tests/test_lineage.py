"""The lineage reads on tiny sources: the named groups cover the U-Net; the group exchange with its controls (a source
against its own copy, nothing exchanged, everything exchanged, a random donor) and the restore after each exchange;
the weight distance against a direct computation; a child against its parent by stretch and slice; the latent scale
and its rule; the sampler check; the new files read clean."""
import math
import os
import re

import pytest
import torch

pytest.importorskip("diffusers")

from geolip.modular_diffusion.capture import BANDS  # noqa: E402
from geolip.modular_diffusion.capture.teacher import _perturb, tiny_unet  # noqa: E402
from geolip.modular_diffusion.experiments import lineage  # noqa: E402

FLOW = dict(tiny=1, kind="flow", shift="2.5", unet_seed=5, probe_n=3, probe_per_band=2, chunk=2)


@pytest.fixture(scope="module")
def near_donor(tmp_path_factory):
    """A U-Net folder: the reference's U-Net (seed 5) with one percent of noise on every tensor (a 'real' donor)."""
    unet = tiny_unet(5)
    _perturb(unet, 5)
    g = torch.Generator().manual_seed(99)
    with torch.no_grad():
        for p in unet.parameters():
            p.add_(0.01 * p.abs().mean() * torch.randn(p.shape, generator=g))
    folder = tmp_path_factory.mktemp("donor") / "unet"
    unet.save_pretrained(str(folder), safe_serialization=True)
    return str(folder)


def test_the_groups_cover_every_tensor_once():
    keys = list(tiny_unet(0).state_dict())
    groups = lineage.group_keys(keys)
    assert list(groups) == list(lineage.GROUPS)
    assert sorted(k for v in groups.values() for k in v) == sorted(keys)
    assert lineage.group_of("conv_out.weight") == "conv_norm_out+conv_out"
    assert lineage.group_of("up_blocks.3.attentions.2.proj_in.weight") == "up_blocks.3"
    parts = lineage.group_keys(keys, ["down_blocks.1", "mid_block"])
    assert list(parts) == ["down_blocks.1.resnets", "down_blocks.1.attentions", "down_blocks.1.rest",
                           "mid_block.resnets", "mid_block.attentions"]
    assert sorted(k for v in parts.values() for k in v) == sorted(k for k in keys if k.startswith(("down_blocks.1.",
                                                                                                    "mid_block.")))
    with pytest.raises(ValueError):
        lineage.group_of("add_embedding.linear_1.weight")
    with pytest.raises(ValueError):
        lineage.group_keys(keys, ["conv_in"])


def test_a_source_against_its_own_copy_reads_zero(tmp_path):
    r = lineage.main(variant="group_exchange", donor="tiny", donor_seed=5, out=str(tmp_path / "own"), **FLOW)
    assert all(v == 0.0 for row in r["table"].values() for v in row.values())       # control (a)
    assert all(r["controls"].values())


def test_the_exchange_controls_and_the_restore(tmp_path, near_donor):
    r = lineage.main(variant="group_exchange", donor=near_donor, out=str(tmp_path / "near"), **FLOW)
    c = r["controls"]
    assert c["none_reads_zero"] and c["restored_each"] and c["restored_reads_zero"]       # (b) and the restore
    assert c["all_equals_donor"] and c["all_read_equals_donor_read"]                       # (c)
    assert [row["unit"] for row in r["rows"]] == ["none"] + list(lineage.GROUPS) + ["all"]
    assert all(r["table"][g][b] > 0 for g in lineage.GROUPS for b in BANDS)
    rnd = lineage.main(variant="group_exchange", donor="random", donor_seed=3, groups="mid_block,up_blocks.3",
                       out=str(tmp_path / "random"), **FLOW)
    assert [row["unit"] for row in rnd["rows"]] == ["none", "mid_block", "up_blocks.3", "all"]
    for g in ("mid_block", "up_blocks.3"):                                                 # (d)
        assert all(rnd["table"][g][b] >= 10 * r["table"][g][b] for b in BANDS)
    assert all(rnd["controls"].values())


def test_exchanging_a_group_changes_only_that_group():
    ref, don = tiny_unet(5), tiny_unet(7)
    _perturb(ref, 5)
    _perturb(don, 7)
    sr, sd = ref.state_dict(), don.state_dict()
    groups = lineage.group_keys(list(sr))
    original = {k: v.clone() for k, v in sr.items()}
    for g, keys in groups.items():                          # the two sources differ in every group
        assert any(not torch.equal(sr[k], sd[k]) for k in keys), g
    for g in ("conv_in", "mid_block", "up_blocks.2", "conv_norm_out+conv_out"):
        inside = set(groups[g])
        with lineage.exchanged(sr, sd, groups[g]) as restore:
            now = ref.state_dict()                          # the module itself, read again
            assert all(torch.equal(now[k], sd[k]) for k in inside)
            assert all(torch.equal(now[k], original[k]) for k in now if k not in inside)
        assert restore["ok"] and all(torch.equal(ref.state_dict()[k], original[k]) for k in original)


def test_split_reads_the_parts_of_a_group(tmp_path, near_donor):
    r = lineage.main(variant="group_exchange", donor=near_donor, split="up_blocks.1", out=str(tmp_path / "s"), **FLOW)
    assert [row["unit"] for row in r["rows"]] == ["none", "up_blocks.1.resnets", "up_blocks.1.attentions",
                                                  "up_blocks.1.rest", "all"]
    assert all(r["controls"].values())


def test_the_distance_against_a_direct_computation(tmp_path):
    r = lineage.main(variant="distance", donor="tiny", donor_seed=7, out=str(tmp_path / "d"), **FLOW)
    a, b = tiny_unet(5), tiny_unet(7)
    _perturb(a, 5)
    _perturb(b, 7)
    sa, sb = a.state_dict(), b.state_dict()
    keys = [k for k in sa if lineage.group_of(k) == "mid_block"]
    ra = torch.cat([sa[k].double().reshape(-1) for k in keys])
    rb = torch.cat([sb[k].double().reshape(-1) for k in keys])
    assert r["table"]["mid_block"]["relative"] == pytest.approx(float((rb - ra).norm() / ra.norm()), rel=1e-12)
    assert r["table"]["mid_block"]["cosine"] == pytest.approx(float(ra @ rb / (ra.norm() * rb.norm())), rel=1e-12)
    lines = open(tmp_path / "d" / "distance.csv", encoding="utf-8").read().strip().split("\n")
    assert len(lines) == 1 + len(sa)
    same = lineage.main(variant="distance", donor="tiny", donor_seed=5, out=str(tmp_path / "same"), **FLOW)
    assert all(v["relative"] == 0.0 and v["cosine"] == pytest.approx(1.0, abs=1e-12) for v in same["table"].values())


def test_a_child_against_its_parent(tmp_path, near_donor):
    steps = lineage.slice_steps(10, 4)
    assert len(steps) == 10 and steps[0] == [0, 33, 66, 99] and steps[9][-1] == 999
    assert all(100 * k <= t < 100 * (k + 1) for k, st in enumerate(steps) for t in st)
    r = lineage.main(variant="child_parent", parent=near_donor, slices=10, per_slice=2, out=str(tmp_path / "c"), **FLOW)
    assert len(r["slices"]) == 10 and all(x["read"] > 0 for x in r["slices"]) and set(r["stretches"]) == set(BANDS)
    same = lineage.main(variant="child_parent", parent="tiny", donor_seed=5, slices=10, per_slice=2,
                        out=str(tmp_path / "same"), **FLOW)
    assert all(x["read"] == 0.0 for x in same["slices"]) and all(v == 0.0 for v in same["stretches"].values())


def test_the_scale_and_its_rule(tmp_path):
    assert lineage.scale_rule(0.95)["kappa_to_use"] == 1.0 and lineage.scale_rule(1.1)["rule"] == "standard"
    assert lineage.scale_rule(0.18) == {"kappa": 0.18, "rule": "scaled", "kappa_to_use": 0.18}
    r = lineage.main(variant="scale", pic_n=4, group=2, sample_steps=3, out=str(tmp_path / "k"), **FLOW)
    assert r["kappa"] == pytest.approx(r["samples"]["unguided"]["all"] / r["rows_latents"]["all"], rel=1e-12)
    assert len(r["samples"]["guided"]["by_channel"]) == 4 and r["samples"]["unguided"]["guidance"] == 1.0
    assert r["rule"] in ("standard", "scaled")


def test_the_sampler_check(tmp_path):
    r = lineage.main(variant="sampler_check", pic_n=3, sample_steps=5, out=str(tmp_path / "sc"), **FLOW)
    assert r["first_step_equal_to_rounding"] and r["repeat_equal"]
    with pytest.raises(ValueError):
        lineage.main(variant="sampler_check", tiny=1, pic_n=2, out=str(tmp_path / "eps"))


def test_stretch_snr():
    eps = lineage.stretch_snr("eps")
    flow = lineage.stretch_snr("flow", s0=1.0)
    assert flow["clean"]["snr_at_lo"] == math.inf and eps["clean"]["lo"] == 0.0
    s = flow["noisy"]["hi"]
    assert flow["noisy"]["snr_at_hi"] == pytest.approx(((1 - s) / s) ** 2)
    assert eps["middle"]["snr_at_hi"] < eps["middle"]["snr_at_lo"]


def test_the_new_files_read_clean(request):
    # the hygiene test's words and a few more; each word written with a class so this line does not match itself
    root = str(request.config.rootpath)                      # the repository (the folder of pyproject.toml)
    assert os.path.isfile(os.path.join(root, "pyproject.toml")), root
    pattern = (r"(?i)\bE\d\d\b|\bm[d]0|\bm[r]_|\bm[i]nd\b|\bl[u]ne\b|\bd[o]cket|\bP[h]il\b|\ba[r]m\b|\bc[a]non\b"
               r"|\bn[i]ght\b")
    for rel in ("geolip/modular_diffusion/capture/process.py", "geolip/modular_diffusion/experiments/lineage.py",
                "tests/test_process.py", "tests/test_lineage.py"):
        text = open(os.path.join(root, *rel.split("/")), encoding="utf-8").read()
        m = re.search(pattern, text)
        assert m is None, (rel, text[max(0, m.start() - 40):m.end() + 40])
