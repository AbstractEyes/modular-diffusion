"""The feature cache (test 13): a cached batch equals the live forward bit for bit, and K steps trained from the cache
equal K steps of the trainer."""
import os

import pytest
import torch

from geolip.modular_diffusion import compat, train
from geolip.modular_diffusion.beds import make_bed
from geolip.modular_diffusion.cache import FeatureCache
from geolip.modular_diffusion.linker import StaleUnit
from geolip.modular_diffusion.model import Model

K = 5

CASES = {
    # bed, the body's arguments, the arguments that add the units trained from the cache
    "fields_band_member": ("fields", dict(base=0), dict(extra="1@0.25-0.65")),
    "pictures_content_with_stage_units": ("pictures", dict(base=0, stage_body="3"), dict(content="0.25-0.65:2", extra="a@0.65-1")),
}


def _common(bed, npz):
    c = dict(bed=bed, keep=1, codes="rung", anchor=1, anchor_in=1, dim=64, batch=8, heldout=32, log_every=1,
             device="cpu", seeds="0", sheet=0, warmup=3)
    if bed == "pictures":
        c["data"] = npz
    return c


@pytest.mark.parametrize("case", sorted(CASES))
def test_13_cache_equals_live(synthetic_npz, tmp_path, case):
    bed_kind, body_kw, more_kw = CASES[case]
    common = _common(bed_kind, synthetic_npz)
    body = train.main(out=str(tmp_path / "body"), steps=6, **body_kw, **common)[0]
    body_pt = os.path.join(tmp_path / "body", body["tag"] + ".pt")
    live_kw = dict(common, **body_kw, **more_kw, init=body_pt, train="new", equalize=0, steps=K)
    live = train.main(out=str(tmp_path / "live"), **live_kw)[0]

    a = dict(train.DEFAULTS, **live_kw)
    bed = make_bed(a, "cpu")
    model = Model.born(compat.build_from_args(a, bed.field_shape), 0, bed.spectrum)
    loaded = compat.load_legacy_files(model, [body_pt])
    new = loaded["members_new"]
    assert new and not loaded["stage_new"]
    folder = str(tmp_path / "cache")
    FeatureCache.write(model, bed, 0, K, a["batch"], folder, train=new)
    cache = FeatureCache.open(folder, model)

    # a cached batch equals the live forward
    cond_drop = a["cond_drop"] if bed.n_classes else 0.0
    with torch.no_grad():
        for step in range(K):
            gen = train.step_generator(0, step, "cpu")
            x, cls = bed.train_batch(a["batch"], gen)
            zt, t, c, target = model.draw(x, cls, gen, cond_drop, "uniform")
            parts, h, ctx, e = model.components(zt, t, c)
            rec = cache.batch(step)
            for k, v in (("t", t), ("cls", c), ("target", target), ("h", h), ("ctx", ctx), ("e", e)):
                assert torch.equal(rec[k], v), (step, k)
            again = cache.parts(model, rec)
            assert list(again) == list(parts)
            for k in parts:
                assert torch.equal(again[k], parts[k]), (step, k)
            assert torch.equal(torch.stack(list(again.values())).sum(0), torch.stack(list(parts.values())).sum(0))

    # K steps from the cache equal K live steps of the trainer
    curve = cache.fit(model, lr=a["lr"], warmup=a["warmup"], log_every=a["log_every"])
    assert curve == live["curve"]
    w_live = torch.load(os.path.join(tmp_path / "live", live["tag"] + ".pt"))
    w_cache = compat.to_legacy_state(model)
    assert list(w_live) == list(w_cache)
    for k in w_live:
        assert torch.equal(w_live[k], w_cache[k]), k

    # a frozen unit that changed makes the cache stale
    with torch.no_grad():
        model.stem.embed.bias[0] += 1.0
    with pytest.raises(StaleUnit) as err:
        FeatureCache.open(folder, model)
    assert err.value.unit == "stem"
