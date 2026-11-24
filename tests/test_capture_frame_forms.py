"""The two proposed forms of the address frame: the overlap form over owners' arguments and the gate argument of a set
of requests (a sum over requests of a product over digits). Library functions only; no model uses them."""
import itertools

import pytest
import torch

from geolip.modular_diffusion import frame


def _discounted(u):
    u = u.double()
    return torch.sinh(u) / (1.0 + (torch.cosh(u) - 1.0).sum(dim=-1, keepdim=True))


def test_overlap_an_owner_at_rest_weighs_exactly_zero():
    gen = torch.Generator().manual_seed(0)
    u = 6.0 * torch.rand(200, 7, generator=gen, dtype=torch.float64)
    rest = torch.rand(200, 7, generator=gen) < 0.4
    u[rest] = 0.0
    w, base = frame.overlap_weights(u)
    assert w.shape == (200, 7) and base.shape == (200,)
    assert bool((w[rest] == 0.0).all())
    assert bool((w[~rest] > 0).all())
    # every owner at rest: the base weight holds everything
    w0, b0 = frame.overlap_weights(torch.zeros(3, 4))
    assert bool((w0 == 0.0).all()) and bool((b0 == 1.0).all())


def test_overlap_equals_the_discounted_form_with_at_most_one_owner_above_zero():
    gen = torch.Generator().manual_seed(1)
    n = 5
    u = torch.zeros(300, n, dtype=torch.float64)
    who = torch.randint(0, n, (300,), generator=gen)
    val = 8.0 * torch.rand(300, generator=gen, dtype=torch.float64)
    u[torch.arange(300), who] = val
    u[:20] = 0.0                                                  # rows with no owner above zero
    w, base = frame.overlap_weights(u)
    ref = _discounted(u)
    assert float((w - ref).abs().max()) < 1e-12
    # one owner at u: the weight is tanh(u), the base exp(-u) / cosh(u) = 1 - tanh(u)
    one = val[20:]
    assert float((w[torch.arange(20, 300), who[20:]] - torch.tanh(one)).abs().max()) < 1e-12
    assert float((base[20:] - torch.exp(-one) / torch.cosh(one)).abs().max()) < 1e-12


def test_overlap_sums_to_one_and_never_above_one():
    gen = torch.Generator().manual_seed(2)
    for n in (1, 2, 3, 8, 32):
        u = 10.0 * torch.rand(500, n, generator=gen, dtype=torch.float64)
        u[torch.rand(500, n, generator=gen) < 0.3] = 0.0
        w, base = frame.overlap_weights(u)
        assert float((w.sum(-1) + base - 1.0).abs().max()) < 1e-12, n
        assert float(w.sum(-1).max()) <= 1.0 + 1e-15, n              # to the rounding of the sum
        assert bool((base >= 0).all()) and bool((w >= 0).all())
    # several owners above zero share the mass, where the discounted form sums above one
    u = torch.tensor([[3.0, 3.0, 3.0]], dtype=torch.float64)
    w, _ = frame.overlap_weights(u)
    assert float(w.sum()) <= 1.0 and float(_discounted(u).sum()) > 1.0
    # a batch layout of any rank: the last axis is the owners
    w3, b3 = frame.overlap_weights(torch.rand(2, 3, 4, generator=gen))
    assert w3.shape == (2, 3, 4) and b3.shape == (2, 3)
    with pytest.raises(ValueError):
        frame.overlap_weights(torch.tensor([[-0.1, 1.0]]))


def _superposed(requests, owners, U):
    """The form that is NOT the library's (kept only here): a product over digits of per-digit sums over requests."""
    req = torch.as_tensor(requests, dtype=torch.long)
    own = torch.as_tensor(owners, dtype=torch.long)
    per_digit = (req[:, None, :] == own[None, :, :]).double().sum(dim=0)      # [n, L]
    return float(U) * per_digit.prod(dim=1)


def test_requests_fire_on_exactly_their_addresses():
    for L, base in ((2, 4), (3, 3), (4, 3)):
        owners = list(itertools.product(range(base), repeat=L))
        for k in (1, 2, 3):
            # k requests that differ on every digit
            requests = [tuple((j + i) % base for i in range(L)) for j in range(k)]
            u = frame.request_arguments(requests, owners, U=5.0)
            fired = {owners[i] for i in torch.nonzero(u > 0).squeeze(1).tolist()}
            assert fired == set(requests), (L, k)
            assert bool((u[u > 0] == 5.0).all())
            sup = _superposed(requests, owners, 5.0)
            assert int((sup > 0).sum()) == k ** L, (L, k)
            w, b = frame.request_weights(requests, owners, U=5.0)
            assert int((w > 0).sum()) == k
            assert abs(float(w.sum() + b) - 1.0) < 1e-12
    # a request given twice adds its argument twice on its one address
    owners = list(itertools.product(range(3), repeat=2))
    u = frame.request_arguments([(1, 2), (1, 2)], owners, U=2.0)
    assert float(u[owners.index((1, 2))]) == 4.0 and int((u > 0).sum()) == 1
    with pytest.raises(ValueError):
        frame.request_arguments([(1, 2, 0)], owners)
