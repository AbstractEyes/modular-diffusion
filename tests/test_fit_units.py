"""A fit by unit names (test 8): fit_units on the plain cluster build equals the flat entry's fit of the plain build in
every tensor, bit for bit; frozen units stay bit-equal; an unknown unit name raises."""
import os

import pytest
import torch

from geolip.modular_diffusion import compat, train
from geolip.modular_diffusion.beds import make_bed
from geolip.modular_diffusion.build import content_hash
from geolip.modular_diffusion.forms import cluster_build
from geolip.modular_diffusion.model import Model
from geolip.modular_diffusion.train import fit_units

COMMON = dict(steps=4, batch=8, heldout=32, log_every=1, dim=64, sheet=0, device="cpu", seeds="0")


@pytest.mark.parametrize("bed_kind", ["fields", "pictures"])
def test_8_fit_units_equals_the_flat_fit(bed_kind, synthetic_npz, tmp_path):
    a = dict(train.DEFAULTS, bed=bed_kind, data=synthetic_npz if bed_kind == "pictures" else "", **COMMON)
    old = train.main(out=str(tmp_path / "old"), arch="plain", plain_depth=4, anchor=1, anchor_in=1,
                     **{k: v for k, v in a.items() if k in ("bed", "data", *COMMON)})[0]
    flat = torch.load(os.path.join(tmp_path / "old", old["tag"] + ".pt"))
    bed = make_bed(a, "cpu")
    legacy_build = compat.build_from_args(dict(a, arch="plain", plain_depth=4, anchor=1, anchor_in=1), bed.field_shape)
    build = cluster_build(bed.field_shape, stem_depth=4, dim=64)
    assert [u.to_dict() for u in build.units] == [u.to_dict() for u in legacy_build.units]
    assert build.geometry == legacy_build.geometry and build.info == legacy_build.info
    torch.manual_seed(0)
    model = Model.born(build, 0, bed.spectrum)
    res = fit_units(model, bed, 0, steps=4, batch=8, heldout=32, log_every=1, log=lambda s: None,
                    expect_anchor_at_birth=True)
    mine = compat.to_legacy_state(model)
    assert list(mine) == list(flat)
    for k in flat:
        assert torch.equal(flat[k], mine[k]), k
    assert res["curve"] == old["curve"]
    assert res["loss_over_levels"] == old["loss_over_levels"] and res["heldout_loss"] == old["heldout_loss"]
    assert res["frozen_unchanged"] is True and res["peak_memory_mib"] is None
    assert res["recipe"]["t_range"] == [0.0, 1.0] and res["recipe"]["cond_drop"] == (0.1 if bed.n_classes else 0.0)


def test_8_frozen_units_stay_bit_equal_and_unknown_names_raise():
    bed = make_bed(dict(train.DEFAULTS, bed="fields"), "cpu")
    model = Model.born(cluster_build(bed.field_shape, stem_depth=1, body_depth=1, dim=32), 0, bed.spectrum)
    stem_before, body_before = content_hash(model.stem), content_hash(model.body)
    res = fit_units(model, bed, 0, steps=3, trainable=["head"], batch=8, heldout=32, warmup=1, log=lambda s: None)
    assert res["frozen_unchanged"] is True and res["trainable"] == ["head"]
    assert content_hash(model.stem) == stem_before and content_hash(model.body) == body_before
    assert float(model.head.proj.weight.detach().abs().max()) > 0
    assert res["trained_numbers"] == sum(p.numel() for p in model.head.parameters())
    with pytest.raises(KeyError):
        fit_units(model, bed, 0, steps=1, trainable=["nope"], log=lambda s: None)
    with pytest.raises(ValueError):
        fit_units(model, bed, 0, steps=1, trainable="some", log=lambda s: None)
