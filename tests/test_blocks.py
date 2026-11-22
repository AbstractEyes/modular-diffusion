"""Blocks and rotary codes: parity with the block library (P1) and the rotary closed form (test 4)."""
import math

import pytest
import torch

from geolip.modular_diffusion import blocks as B
from geolip.modular_diffusion import rotary as R


def _params_equal(a, b):
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    assert list(pa) == list(pb)
    for k in pa:
        assert torch.equal(pa[k], pb[k]), k


def _grads_equal(a, b):
    for (ka, pa), (kb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert ka == kb
        assert (pa.grad is None) == (pb.grad is None), ka
        if pa.grad is not None:
            assert torch.equal(pa.grad, pb.grad), ka


def _live(module, seed=5):
    """Give zero-born tensors small random values so that every path carries signal."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            if float(p.abs().max()) == 0.0:
                p.copy_(torch.randn(p.shape, generator=g) * 0.02)


@pytest.mark.parametrize("seed", [0, 3])
def test_p1_block_against_library(gf, seed):
    torch.manual_seed(seed)
    ref = gf["blocks"].TransformerBlock(128, 2)
    torch.manual_seed(seed)
    mine = B.Block(128, 2)
    _params_equal(ref, mine)                       # same draws, same order (zeroed layers included)
    _live(ref)
    mine.load_state_dict(ref.state_dict())
    gen = torch.Generator().manual_seed(11)
    x = torch.randn(4, 256, 128, generator=gen)
    ctx = torch.randn(4, 1, 128, generator=gen)
    e = torch.randn(4, 128, generator=gen)
    code = R.AxialBandCode(64, 16, 2.0, 64.0)
    gf_rot = gf["blocks"].table_rotary(code.cos, code.sin)
    x1, x2 = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    y1 = ref(x1, ctx, None, e, gf_rot)
    y2 = mine(x2, ctx, None, e, code.rotary())
    assert torch.equal(y1, y2)
    w = torch.randn(y1.shape, generator=gen)
    (y1 * w).sum().backward()
    (y2 * w).sum().backward()
    assert torch.equal(x1.grad, x2.grad)
    _grads_equal(ref, mine)


def test_p1_draw_count_matches(gf):
    """After constructing one block the global generator is in the same state in both (zero_linear's discarded draws)."""
    torch.manual_seed(1)
    gf["blocks"].TransformerBlock(64, 2)
    a = torch.randn(5)
    torch.manual_seed(1)
    B.Block(64, 2)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_p1_zero_head_and_time_embedding(gf):
    torch.manual_seed(2)
    ref = gf["towers"].ZeroHead(128, 12)
    torch.manual_seed(2)
    mine = B.ZeroHead(128, 12)
    _params_equal(ref, mine)
    _live(ref)
    mine.load_state_dict(ref.state_dict())
    h = torch.randn(3, 256, 128)
    assert torch.equal(ref(h), mine(h))

    torch.manual_seed(4)
    ref_t = gf["model"].TimeEmbedding(64, 128, 1000.0)
    torch.manual_seed(4)
    mine_t = B.TimeEmbedding(64, 128, 1000.0)
    _params_equal(ref_t, mine_t)
    assert torch.equal(ref_t.freqs, mine_t.freqs)
    t = torch.rand(9)
    y1, y2 = ref_t(t), mine_t(t)
    assert torch.equal(y1, y2)
    y1.square().sum().backward()
    y2.square().sum().backward()
    _grads_equal(ref_t, mine_t)


def test_p1_rotary_helpers(gf, legacy):
    x = torch.randn(2, 2, 256, 64)
    assert torch.equal(gf["blocks"].rotate_half(x), R.rotate_half(x))
    for lo, hi in ((2.0, 64.0), (2.0, 4.0), (4.0, 8.0), (8.0, 64.0)):
        ref = legacy.AxialBandCode(64, 16, lo, hi)
        mine = R.AxialBandCode(64, 16, lo, hi)
        assert torch.equal(ref.cos, mine.cos) and torch.equal(ref.sin, mine.sin)
        q, k = torch.randn(2, 2, 256, 64), torch.randn(2, 2, 256, 64)
        a = ref.rotary()(q, k)
        b = mine.rotary()(q, k)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        assert len(mine.state_dict()) == 0                  # tables are derived, never saved
    assert R.BANDS3 == legacy.BANDS3


@pytest.mark.parametrize("lo,hi", [(2.0, 4.0), (4.0, 8.0), (8.0, 64.0), (2.0, 64.0)])
def test_4_rung_neighbor_score(lo, hi):
    """Token 0 against its right neighbor: the score of all-ones q and k is 0.5 (1 + mean cos(rates))."""
    head_dim, grid = 64, 16
    code = R.AxialBandCode(head_dim, grid, lo, hi)
    ones = torch.ones(1, 1, grid * grid, head_dim)
    q, k = code.rotary()(ones, ones)
    got = float((q[0, 0, 0] * k[0, 0, 1]).sum() / head_dim)
    want = 0.5 * (1.0 + float(torch.cos(torch.tensor(code.rates, dtype=torch.float64)).mean()))
    assert abs(got - want) < 1e-5
    assert all(2 * math.pi / hi < r <= 2 * math.pi / lo + 1e-12 for r in code.rates)
