"""Beds and the read against the original script on a small synthetic npz and on Gaussian fields (P4)."""
import math

import pytest
import torch

from geolip.modular_diffusion import compat, train
from geolip.modular_diffusion.beds import make_bed
from geolip.modular_diffusion.model import Model
from geolip.modular_diffusion.read import read


def _leaves_equal(a, b, path="$"):
    """Every leaf equal: floats bit for bit (NaN equal to NaN), the same keys and lengths."""
    assert type(a) is type(b) or (isinstance(a, (int, float)) and isinstance(b, (int, float))), (path, type(a), type(b))
    if isinstance(a, dict):
        assert list(a) == list(b), (path, list(a), list(b))
        for k in a:
            _leaves_equal(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _leaves_equal(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and math.isnan(a):
        assert isinstance(b, float) and math.isnan(b), path
    else:
        assert a == b, (path, a, b)


@pytest.mark.parametrize("tilt", [0, 1])
def test_p4_pictures_bed_matches_original(legacy, synthetic_npz, tilt):
    a = dict(train.DEFAULTS, bed="pictures", data=synthetic_npz, tilt=tilt)
    ref = legacy.Bed(dict(legacy.ARGS, bed="pictures", data=synthetic_npz, tilt=tilt), "cpu")
    mine = make_bed(a, "cpu")
    for k in ("xtr", "ytr", "xte", "yte", "mean", "std", "spectrum"):
        assert torch.equal(getattr(ref, k), getattr(mine, k)), k
    assert (ref.tilt is None) == (mine.tilt is None)
    if tilt:
        assert torch.equal(ref.tilt, mine.tilt)
    assert ref.field_shape == mine.field_shape and ref.n_classes == mine.n_classes
    for step in range(3):
        g1, g2 = torch.Generator().manual_seed(step), torch.Generator().manual_seed(step)
        x1, y1 = ref.train_batch(32, g1)
        x2, y2 = mine.train_batch(32, g2)
        assert torch.equal(x1, x2) and torch.equal(y1, y2)
        assert torch.equal(g1.get_state(), g2.get_state())
    for n in (5, 300, 1000):
        assert all(torch.equal(p, q) for p, q in zip(ref.heldout(n), mine.heldout(n)))


def test_p4_fields_bed_matches_original(legacy):
    a = dict(train.DEFAULTS, bed="fields")
    ref = legacy.Bed(dict(legacy.ARGS, bed="fields"), "cpu")
    mine = make_bed(a, "cpu")
    assert torch.equal(ref.spectrum, mine.spectrum) and torch.equal(ref.sqrt_s, mine.sqrt_s)
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    assert all(torch.equal(p, q) for p, q in zip(ref.train_batch(16, g1), mine.train_batch(16, g2)))
    assert all(torch.equal(p, q) for p, q in zip(ref.heldout(40), mine.heldout(40)))


# heldout 260 spans two chunks of the read (256 + 4); the others use one short chunk to keep the CPU time down
READ_CASES = {
    "pictures_base": ("pictures", dict(anchor=1, anchor_in=1, heldout=260)),
    "pictures_stages_content": ("pictures", dict(anchor=1, stages="3", soft=0.05, content="0.25-0.65:2", stage_body="3",
                                                 heldout=40)),
    "pictures_plain": ("pictures", dict(arch="plain", plain_depth=1, anchor=0, heldout=40)),
    "fields_base": ("fields", dict(anchor=1, heldout=40)),
}


@pytest.mark.parametrize("case", sorted(READ_CASES))
def test_p4_read_matches_original(legacy, synthetic_npz, case):
    bed_kind, kw = READ_CASES[case]
    a = dict(train.DEFAULTS, bed=bed_kind, data=synthetic_npz, **kw)
    la = dict(legacy.ARGS, **{k: v for k, v in a.items() if k in legacy.ARGS})
    ref_bed = legacy.Bed(la, "cpu")
    bed = make_bed(a, "cpu")
    ref = legacy.Collective(la, ref_bed.field_shape, ref_bed.spectrum, 0)
    for p in ref.parameters():
        if float(p.detach().abs().max()) == 0.0:
            torch.manual_seed(int(p.numel()))
            with torch.no_grad():
                p.copy_(torch.randn_like(p) * 0.02)
    mine = Model.born(compat.build_from_args(a, bed.field_shape), 0, bed.spectrum)
    compat.load_legacy_state(mine, ref.state_dict(), [m.name for m in ref.members])
    r_ref = legacy.read(ref, ref_bed, la, "cpu")
    r_mine = read(mine, bed, a["heldout"], "cpu")
    _leaves_equal(r_ref, r_mine)
