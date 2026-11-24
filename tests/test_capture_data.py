"""Capture data: the shard stream's order is fixed by its seed and starts anywhere; the encoder writes shards of the
stated layout and a stopped job continues to the same shards; random batches for tests."""
import os

import numpy as np
import pytest
import torch

pytest.importorskip("diffusers")        # the capture tests need the optional extra

from geolip.modular_diffusion.capture import data


def _row_keys(ids):
    return [tuple(r.tolist()) for r in ids]


def test_stream_is_deterministic_and_covers_each_row_once_a_pass(tmp_path):
    index = data.synthetic_shards(str(tmp_path), rows=20, shard_size=7, size=4, seed=3)
    assert index["counts"] == [7, 7, 6]
    all_ids = torch.cat([data.read_shard(os.path.join(str(tmp_path), f))[1] for f in index["shards"]])
    a = data.LatentBatches(str(tmp_path), batch=4, seed=11)
    b = data.LatentBatches(str(tmp_path), batch=4, seed=11)
    assert len(a) == 20
    sa = a.stream()
    sb = iter(b)
    got = []
    for _ in range(10):                                           # two passes
        (xa, ia), (xb, ib) = next(sa), next(sb)
        assert xa.dtype == torch.float32 and ia.dtype == torch.long
        assert xa.shape == (4, 4, 4, 4) and ia.shape == (4, 77)
        assert torch.equal(xa, xb) and torch.equal(ia, ib)
        got.append(ia)
    first, second = torch.cat(got[:5]), torch.cat(got[5:])
    assert sorted(_row_keys(first)) == sorted(_row_keys(all_ids))
    assert sorted(_row_keys(second)) == sorted(_row_keys(all_ids))
    assert _row_keys(first) != _row_keys(second)                  # a new order each pass
    other = next(iter(data.LatentBatches(str(tmp_path), batch=4, seed=12)))[1]
    assert not torch.equal(other, got[0])


def test_stream_started_late_gives_the_same_batches(tmp_path):
    data.synthetic_shards(str(tmp_path), rows=17, shard_size=5, size=4, seed=0)
    ref = data.LatentBatches(str(tmp_path), batch=3, seed=2).stream()
    ref_batches = [next(ref) for _ in range(14)]                  # runs across shards and passes
    for k in (0, 1, 5, 6, 11):
        late = data.LatentBatches(str(tmp_path), batch=3, seed=2).stream(start=k)
        for j in range(k, 14):
            x, i = next(late)
            assert torch.equal(x, ref_batches[j][0]) and torch.equal(i, ref_batches[j][1]), (k, j)


def test_shard_layout_and_values(tmp_path):
    lat = torch.randn(3, 4, 8, 8)
    ids = torch.randint(0, 49408, (3, 77))
    path = os.path.join(str(tmp_path), data.shard_name(0))
    data.write_shard(path, lat, ids)
    l2, i2 = data.read_shard(path)
    assert l2.dtype == torch.float16 and i2.dtype == torch.int32
    assert torch.equal(l2, lat.half()) and torch.equal(i2.long(), ids)
    with pytest.raises(ValueError):
        data.write_shard(path, lat, ids[:, :70])


def test_synthetic_batches():
    src = data.synthetic_batches(size=8, seed=5)
    assert src.latent_shape == (4, 8, 8)
    a = list(src.take(3, 2))
    b = [data.synthetic_batches(size=8, seed=5)(k, 2) for k in (2, 1, 0)][::-1]     # any order, same batches
    assert len(a) == 3
    for (x, i), (y, j) in zip(a, b):
        assert x.shape == (2, 4, 8, 8) and i.shape == (2, 77) and i.dtype == torch.long
        assert torch.equal(x, y) and torch.equal(i, j)
    assert not torch.equal(a[0][0], a[1][0])


def test_latent_batches_as_a_source(tmp_path):
    data.synthetic_shards(str(tmp_path), rows=13, shard_size=4, size=4, seed=1)
    it = data.LatentBatches(str(tmp_path), batch=3, seed=7)
    src = data.LatentBatches(str(tmp_path), seed=7)
    assert src.latent_shape == (4, 4, 4)
    ref = it.stream()
    for step in range(9):
        x, i = next(ref)
        y, j = src(step, 3)
        assert torch.equal(x, y) and torch.equal(i, j), step
    # the order does not depend on the batch size: batch 1 of size 6 is batches 2 and 3 of size 3
    y6, j6 = src(1, 6)
    x2, i2 = src(2, 3)
    x3, i3 = src(3, 3)
    assert torch.equal(y6, torch.cat([x2, x3])) and torch.equal(j6, torch.cat([i2, i3]))
    with pytest.raises(ValueError):
        iter(src)


def _tiny_vae():
    from diffusers import AutoencoderKL
    torch.manual_seed(0)
    return AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
                         up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"), block_out_channels=(8, 8),
                         layers_per_block=1, latent_channels=4, norm_num_groups=4, sample_size=32).eval()


def _pictures(n, seed=0):
    from PIL import Image
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        w, h = (40, 32) if k % 2 else (32, 48)                     # landscape and portrait: the center crop
        out.append(Image.fromarray(rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)))
    return out


def _tokenize(captions):
    return torch.tensor([[len(c) % 97] + [k % 50 for k in range(76)] for c in captions], dtype=torch.long)


def test_center_crop():
    pic = _pictures(2)[1]                                         # 40 x 32
    x = data.center_crop(pic, 16)
    assert x.shape == (3, 16, 16) and float(x.min()) >= -1.0 and float(x.max()) <= 1.0


def test_encode_shards_layout_and_resume(tmp_path):
    vae = _tiny_vae()
    pics = _pictures(11)
    caps = [f"caption {k}" * (k + 1) for k in range(11)]
    full = os.path.join(str(tmp_path), "full")
    index = data.encode_shards(pics, caps, full, vae, _tokenize, shard_size=4, batch=3, size=32, log=None)
    assert index["counts"] == [4, 4, 3]
    lat, ids = data.read_shard(os.path.join(full, index["shards"][0]))
    assert lat.shape == (4, 4, 16, 16) and lat.dtype == torch.float16
    assert torch.equal(ids.long(), _tokenize(caps[:4]))
    # the latent is the encoder's (seeded) sample times the scale
    gen = torch.Generator().manual_seed(0)
    x = torch.stack([data.center_crop(p, 32) for p in pics[:3]])
    with torch.no_grad():
        want = vae.encode(x).latent_dist.sample(generator=gen) * data.LATENT_SCALE
    assert torch.equal(lat[:3], want.half())

    # a job stopped inside its second shard, run again, ends with the same shards
    part = os.path.join(str(tmp_path), "part")

    def stops_after(n):
        for k, p in enumerate(pics):
            if k == n:
                raise RuntimeError("stopped")
            yield p

    with pytest.raises(RuntimeError):
        data.encode_shards(stops_after(6), caps, part, vae, _tokenize, shard_size=4, batch=3, size=32, log=None)
    assert data.read_index(part)["counts"] == [4]
    again = data.encode_shards(pics, caps, part, vae, _tokenize, shard_size=4, batch=3, size=32, log=None)
    assert again["counts"] == [4, 4, 3]
    for name in index["shards"]:
        with open(os.path.join(full, name), "rb") as fa, open(os.path.join(part, name), "rb") as fb:
            assert fa.read() == fb.read(), name
    with pytest.raises(ValueError):                               # other settings into the same folder
        data.encode_shards(pics, caps, part, vae, _tokenize, shard_size=5, batch=3, size=32, log=None)
    # the stream reads what the encoder wrote
    x0, tok = next(iter(data.LatentBatches(full, batch=5, seed=0)))
    assert x0.shape == (5, 4, 16, 16) and tok.shape == (5, 77)
