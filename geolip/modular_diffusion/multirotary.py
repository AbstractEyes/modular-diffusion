"""The multi-rotary frame: one declared rotary structure for a head, its address levels, the lifted kernel, the look,
the amplitude gate over addresses, the key binding by summed phases and the read of a block's attention weights.

A head of D numbers holds D / 2 rotate-half pairs (pair i joins the numbers i and i + D / 2, as ``rotary.rotate_half``
pairs them). A pair turned by the angle a acts on (x_i, x_{i + D/2}) as the complex number x e^{i a}, so for a query
turned by a_q and a key turned by a_k

    q' . k' = sum over pairs i of Re[ q_i conj(k_i) e^{i (a_q,i - a_k,i)} ]

and a pair whose two angles are equal scores as if it were not turned.

THE STRUCTURE (``Structure``) divides the pairs by job: ``row`` and ``column`` (the relative band: the token's row and
column on geometric wavelengths lam_j = lo (hi / lo)^(j / n), j = 0 .. n - 1, each axis its own n pairs, the angle
position x 2 pi / lam_j, computed as ``rotary.anchored_angles`` computes it), ``content`` (never turned), ``held``
(declared for a later use, never turned now) and the address ``levels``. Every pair has exactly one job.

A LEVEL (``Level``) is a digit with ``modulus`` anchors (a null anchor included when ``null`` is set). Pair i of the
level carries the integer multiple ks[i] and turns, for the digit d, by

    angle_i(d) = 2 pi ks[i] (d - offset) / modulus,          d = 0 .. modulus - 1

so on that pair a query at digit a and a key at digit b score cos(2 pi ks[i] (a - b) / modulus): only the difference of
the digits enters (the offset cancels). A SIMPLEX level has the distinct multiples 1 .. P and modulus 2 P + 1. When
each multiple is carried by one pair,

    sum over the level's pairs of cos(angle(a) - angle(b)) = P when a = b,   -1/2 otherwise          (``pair_cosine_sum``)

(a multiple carried by c pairs counts c times). The LIFTED KERNEL over the distinct multiples,

    k(a, b) = (1 + 2 sum_{k = 1 .. P} cos(2 pi k (a - b) / modulus)) / modulus                      (``level_kernel``)

is 1 when a = b and 0 otherwise, the null anchor included (the anchors are orthonormal once a plain dimension of
length 1 / sqrt(modulus) is added). In float64 the cosine sum reaches those values to rounding (within 1e-15 on the
moduli used); for whole-number digits ``level_kernel`` rounds that off, so the kernel is exactly 1.0 or 0.0, the
arguments of the gate are exactly 0 off the address, and a member's weight there is exactly 0 (sinh(0) = 0).

THE LOOK AND THE GATE (float64; nothing is compared, no softmax). Token m with place digits and request slot r with its
place digits and its member address:

    look(m, r)  = prod over the place levels of k_l(place_l(m), place_l(r)),     0 where the slot holds no request
    match(r, i) = prod over the member levels of k_l(member_l(r), g_l(i)),        g(i) = member i's address digits
    u_i(m)      = U sum_r look(m, r) match(r, i)

    rest form:       w_i = sinh(u_i) / (1 + sum_j (cosh(u_j) - 1)),              base = 1 - sum_i w_i
                     (cosh(u) - 1 is computed as 2 sinh(u / 2)^2)
    candidate form:  w_i = sinh(u_i) / (sum_j sinh(u_j) + exp(-sum_j u_j)),     base = exp(-sum_j u_j) / (that denominator)

With one owner at u = U both forms give tanh(U). With several owners above zero the rest form's weights can sum past
one (two at u = 1: 1.1267) and its base goes negative; the candidate form's weights and base always sum to one.

    serve:  v = base v_0 + sum_i w_i v_i                                         (the weights cast to the answers' dtype)

THE KEY BINDING (``phase_table``, ``name_phases``, ``name_kernel``). A table P[j, k, b] of phases uniform on [0, 2 pi)
for pair j, distance k from the end of a name and byte b, drawn once in float64 from a CPU generator seeded
crc32(tag) (torch.rand of shape [pairs, reach, 256], times 2 pi). A name x_1 .. x_n (n <= reach bytes) gets

    phi_j(x) = sum_{k = 0}^{n - 1} P[j, k, x_{n - k}]   (mod 2 pi)       k = 0 is the last byte; summed from k = 0 up

and two names score name_kernel(a, b) = sum_j cos(phi_j(a) - phi_j(b)): the number of pairs for the same name, and for
names that differ in any byte a sum of unrelated cosines. The sums run in the table's dtype.

THE LOOK READ (``attention_weights``, ``record_looks``): the softmax weights of ``blocks.SelfAttention`` recomputed from
its own q, k projections and the rotary, with the same rule for the key mask (a key is read when the mask allows it or
it is the query itself).
"""
from __future__ import annotations

import contextlib
import math
import zlib
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .blocks import Rotary, SelfAttention
from .rotary import WHOLE, rotate_half

TWO_PI = 2 * math.pi
VOCAB = 256


# ------------------------------------------------------------------ levels
@dataclass(frozen=True)
class Level:
    """An address digit on its own pairs: see the module's text."""
    name: str
    pairs: Tuple[int, ...]
    ks: Tuple[int, ...]
    modulus: int
    offset: int = 0
    null: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pairs", tuple(int(p) for p in self.pairs))
        object.__setattr__(self, "ks", tuple(int(k) for k in self.ks))
        if len(self.pairs) == 0 or len(self.pairs) != len(self.ks):
            raise ValueError(f"level {self.name!r}: one multiple a pair, at least one pair")
        if len(set(self.pairs)) != len(self.pairs):
            raise ValueError(f"level {self.name!r} names a pair twice")
        if int(self.modulus) < 2 or any(k < 1 for k in self.ks):
            raise ValueError(f"level {self.name!r}: modulus at least 2, multiples at least 1")
        if self.null is not None and not 0 <= int(self.null) < int(self.modulus):
            raise ValueError(f"level {self.name!r}: the null is a digit 0 .. modulus - 1")

    def rates(self, device=None) -> Tensor:
        """[P] float64: 2 pi ks[i] / modulus."""
        return torch.tensor([TWO_PI * k / self.modulus for k in self.ks], dtype=torch.float64, device=device)

    def check_digits(self, digits: Tensor) -> None:
        if digits.numel() and (int(digits.min()) < 0 or int(digits.max()) >= self.modulus):
            raise ValueError(f"level {self.name!r}: digits are 0 .. {self.modulus - 1}")

    def angles(self, digits: Tensor) -> Tensor:
        """[..., P] float64: the angle of each of the level's pairs for the digits [...]."""
        self.check_digits(digits)
        return self.rates(digits.device) * (digits.double() - self.offset)[..., None]

    def distinct_ks(self) -> Tuple[int, ...]:
        return tuple(sorted(set(self.ks)))

    def is_simplex(self) -> bool:
        p = len(self.distinct_ks())
        return self.distinct_ks() == tuple(range(1, p + 1)) and self.modulus == 2 * p + 1


def _as_pairs(x) -> Tuple[int, ...]:
    return tuple(int(p) for p in x)


def _band_rates(n: int, lo: float, hi: float) -> Tensor:
    lam = float(lo) * (float(hi) / float(lo)) ** (torch.arange(n, dtype=torch.float64) / n)
    return 2 * math.pi / lam


# ------------------------------------------------------------------ the structure
@dataclass(frozen=True)
class Structure:
    """The pairs of a head by job and its address levels: see the module's text. The pair sets must be disjoint and
    cover range(head_dim // 2)."""
    head_dim: int
    row: Tuple[int, ...]
    column: Tuple[int, ...]
    content: Tuple[int, ...]
    held: Tuple[int, ...] = ()
    levels: Tuple[Level, ...] = ()
    band: Tuple[float, float] = WHOLE
    names: Tuple[str, ...] = field(init=False, default=())

    def __post_init__(self) -> None:
        for name in ("row", "column", "content", "held"):
            object.__setattr__(self, name, _as_pairs(getattr(self, name)))
        object.__setattr__(self, "levels", tuple(self.levels))
        object.__setattr__(self, "band", (float(self.band[0]), float(self.band[1])))
        if int(self.head_dim) % 2 or int(self.head_dim) < 2:
            raise ValueError("a head holds an even number of numbers")
        names = tuple(lv.name for lv in self.levels)
        if len(set(names)) != len(names):
            raise ValueError("two levels share a name")
        object.__setattr__(self, "names", names)
        used: List[int] = list(self.row) + list(self.column) + list(self.content) + list(self.held)
        for lv in self.levels:
            used += list(lv.pairs)
        if len(set(used)) != len(used):
            raise ValueError("a pair has two jobs")
        if sorted(used) != list(range(self.head_dim // 2)):
            raise ValueError(f"the pairs by job must cover exactly 0 .. {self.head_dim // 2 - 1}")

    @property
    def n_pairs(self) -> int:
        return self.head_dim // 2

    def level(self, name: str) -> Level:
        for lv in self.levels:
            if lv.name == name:
                return lv
        raise KeyError(f"no level {name!r}; the levels are {self.names}")

    def angles(self, rows: Tensor, cols: Tensor, digits: Mapping[str, Tensor]) -> Tensor:
        """[..., head_dim // 2] float32: every pair's angle for tokens at (rows, cols) [...] (floats) with the levels'
        digits [...] (every level named). The relative pairs as ``rotary.anchored_angles`` computes them (float64,
        ``.float()`` last); 0 on the content and held pairs."""
        missing = [n for n in self.names if n not in digits]
        if missing:
            raise KeyError(f"no digits for the levels {missing}")
        dev = rows.device
        r, c = rows.double(), cols.double()
        out = torch.zeros(r.shape + (self.n_pairs,), dtype=torch.float64, device=dev)
        lo, hi = self.band
        if self.row:
            out[..., list(self.row)] = r[..., None] * _band_rates(len(self.row), lo, hi).to(dev)
        if self.column:
            out[..., list(self.column)] = c[..., None] * _band_rates(len(self.column), lo, hi).to(dev)
        for lv in self.levels:
            out[..., list(lv.pairs)] = lv.angles(digits[lv.name].to(dev))
        return out.float()

    def rotary(self, rows: Tensor, cols: Tensor, key_digits: Mapping[str, Tensor],
               query_digits: Optional[Mapping[str, Tensor]] = None,
               head_angles: Optional[Mapping[int, Tuple[Tensor, Tensor]]] = None) -> Rotary:
        """The rotary (q, k) -> (q, k) on [B, heads, N, head_dim]: the keys turned by the angles of ``key_digits``, the
        queries by those of ``query_digits`` (the keys' when None); rows, cols and digits [B, N]. ``head_angles``
        {head: (query angles, key angles)}, each [B, N, head_dim // 2], replace the structure's angles on that head
        (``angle_rotary`` builds the callable from the angles)."""
        ak = self.angles(rows, cols, key_digits)
        aq = ak if query_digits is None else self.angles(rows, cols, query_digits)
        return angle_rotary(aq, ak, head_angles)


def _tables(ang: Tensor) -> Tuple[Tensor, Tensor]:
    """[..., N, P] angles -> (cos, sin) [..., 1, N, 2 P] in the rotate-half layout (as ``rotary.anchored_rotary``): the
    head axis is put before the token axis, whatever stands in front."""
    a = torch.cat([ang, ang], dim=-1).unsqueeze(-3)
    return a.cos(), a.sin()


def angle_rotary(query_angles: Tensor, key_angles: Tensor,
                 head_angles: Optional[Mapping[int, Tuple[Tensor, Tensor]]] = None) -> Rotary:
    """The rotary (q, k) -> (q, k) on [B, heads, N, 2 P] from angles [B, N, P] (tables in float32, as
    ``rotary.anchored_rotary`` builds them): the queries turned by ``query_angles``, the keys by ``key_angles`` (pass
    the same tensor for both when they are equal). ``head_angles`` {head: (query angles, key angles)}, each [B, N, P],
    replace both on that head only; every other head keeps the two given."""
    ck, sk = _tables(key_angles.float())
    cq, sq = (ck, sk) if query_angles is key_angles else _tables(query_angles.float())
    heads = {int(h): (_tables(hq.float()), _tables(hk.float())) for h, (hq, hk) in (head_angles or {}).items()}
    hd = 2 * int(key_angles.shape[-1])

    def _rot(q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        if q.shape[-1] != hd:
            raise ValueError(f"the angles are for a head of {hd} numbers, not {q.shape[-1]}")
        qc, qs, kc, ks = cq, sq, ck, sk
        if heads:
            n_heads = q.shape[-3]
            if max(heads) >= n_heads:
                raise ValueError(f"head {max(heads)} named; the forward has {n_heads} heads")
            index = torch.arange(n_heads, device=q.device).view(n_heads, 1, 1)
            for h, ((hqc, hqs), (hkc, hks)) in heads.items():
                on = index == h                                        # [heads, 1, 1], broadcast over [B, heads, N, 2 P]
                qc, qs = torch.where(on, hqc, qc), torch.where(on, hqs, qs)
                kc, ks = torch.where(on, hkc, kc), torch.where(on, hks, ks)
        return (q * qc.to(q.dtype) + rotate_half(q) * qs.to(q.dtype),
                k * kc.to(k.dtype) + rotate_half(k) * ks.to(k.dtype))

    return _rot


# The anchored rotary of ``rotary.anchored_rotary`` as a structure: four pairs a place digit at 2 pi d / 3.
PLACE_ONLY = Structure(
    head_dim=64, row=range(0, 10), column=range(10, 20), content=range(20, 24),
    levels=(Level("place_column", (24, 25, 26, 27), (1, 1, 1, 1), 3),
            Level("place_row", (28, 29, 30, 31), (1, 1, 1, 1), 3)),
    band=WHOLE)

# One pair a place digit; two member levels of two pairs at 2 pi k / 5 (k = 1, 2), the digit 4 the null anchor (angle 0);
# pairs 30-31 held.
PLACE_MEMBER = Structure(
    head_dim=64, row=range(0, 10), column=range(10, 20), content=range(20, 24), held=range(30, 32),
    levels=(Level("place_column", (24,), (1,), 3),
            Level("place_row", (25,), (1,), 3),
            Level("member_0", (26, 27), (1, 2), 5, offset=4, null=4),
            Level("member_1", (28, 29), (1, 2), 5, offset=4, null=4)),
    band=WHOLE)


# ------------------------------------------------------------------ kernels
def pair_cosine_sum(level: Level, a: Tensor, b: Tensor) -> Tensor:
    """float64 [...]: sum over the level's pairs of cos(angle(a) - angle(b)) for digits a, b (broadcast)."""
    a, b = torch.as_tensor(a), torch.as_tensor(b)
    return torch.cos(level.angles(a) - level.angles(b)).sum(dim=-1)


def level_kernel(level: Level, a: Tensor, b: Tensor) -> Tensor:
    """float64 [...]: the lifted kernel (1 + 2 sum over the distinct multiples k of cos(2 pi k (a - b) / modulus)) /
    modulus; defined when the distinct multiples are 1 .. (modulus - 1) / 2. For whole-number digits (integer tensors)
    the value is exactly 1.0 or 0.0: the cosine sum lands within 1e-15 of it and is rounded to the nearest whole number,
    so a product of kernels is exactly 0 off the address."""
    a, b = torch.as_tensor(a), torch.as_tensor(b)
    p = (level.modulus - 1) // 2
    if level.modulus % 2 == 0 or level.distinct_ks() != tuple(range(1, p + 1)):
        raise ValueError(f"level {level.name!r}: the lifted kernel needs the multiples 1 .. (modulus - 1) / 2")
    level.check_digits(a)
    level.check_digits(b)
    d = (a - b).double()
    s = torch.zeros_like(d)
    for k in range(1, p + 1):
        s = s + torch.cos((TWO_PI * k / level.modulus) * d)
    out = (1.0 + 2.0 * s) / level.modulus
    if not (a.is_floating_point() or b.is_floating_point()):
        out = torch.round(out)            # whole-number digits: the value is 1 or 0; the cosines' rounding is taken off
    return out


def look(levels: Sequence[Level], token_digits: Mapping[str, Tensor], request_digits: Mapping[str, Tensor],
         request_mask: Tensor) -> Tensor:
    """[B, N, R] float64: the product over the (place) levels of level_kernel(token digit [B, N], request digit [B, R]),
    0 where the request slot is masked out (request_mask [B, R] bool)."""
    out = None
    for lv in levels:
        k = level_kernel(lv, token_digits[lv.name][:, :, None], request_digits[lv.name][:, None, :])
        out = k if out is None else out * k
    if out is None:
        raise ValueError("the look needs at least one level")
    return out * request_mask[:, None, :].to(out.dtype)


def member_match(levels: Sequence[Level], request_digits: Mapping[str, Tensor], addresses: Tensor) -> Tensor:
    """[B, R, M] float64: the product over the (member) levels of level_kernel(request digit [B, R], member address
    digit); ``addresses`` [M, len(levels)] long, column l the digit of levels[l]."""
    addresses = torch.as_tensor(addresses, dtype=torch.long)
    if addresses.dim() != 2 or addresses.shape[1] != len(levels):
        raise ValueError("addresses are [members, number of member levels]")
    out = None
    for l, lv in enumerate(levels):
        k = level_kernel(lv, request_digits[lv.name][:, :, None], addresses[None, None, :, l].to(
            request_digits[lv.name].device))
        out = k if out is None else out * k
    if out is None:
        raise ValueError("the match needs at least one level")
    return out


def arguments(look_: Tensor, match: Tensor, U: float) -> Tensor:
    """[B, N, M] float64: u_i(m) = U sum_r look(m, r) match(r, i)."""
    return float(U) * torch.einsum("bnr,brm->bnm", look_.double(), match.double())


# ------------------------------------------------------------------ the gate
def rest_weights(u: Tensor) -> Tuple[Tensor, Tensor]:
    """The rest form over the last axis (the members) -> (w [..., M], base [...]), float64:
    w_i = sinh(u_i) / (1 + sum_j 2 sinh(u_j / 2)^2), base = 1 - sum_i w_i. The sums run one member at a time in order."""
    u = u.double()
    den = torch.ones(u.shape[:-1], dtype=torch.float64, device=u.device)
    for j in range(u.shape[-1]):
        den = den + 2.0 * torch.sinh(u[..., j] / 2.0) ** 2
    w = torch.sinh(u) / den[..., None]
    total = torch.zeros_like(den)
    for j in range(u.shape[-1]):
        total = total + w[..., j]
    return w, 1.0 - total


def candidate_weights(u: Tensor) -> Tuple[Tensor, Tensor]:
    """The candidate form over the last axis -> (w [..., M], base [...]), float64: w_i = sinh(u_i) / (sum_j sinh(u_j)
    + exp(-sum_j u_j)), base = exp(-sum_j u_j) / (the same denominator). The expressions of ``frame.overlap_weights``,
    without its refusal of negative arguments (kernel arguments can sit a rounding below zero)."""
    u = u.double()
    s = torch.sinh(u)
    rest = torch.exp(-u.sum(dim=-1, keepdim=True))
    den = s.sum(dim=-1, keepdim=True) + rest
    return s / den, (rest / den).squeeze(-1)


def serve(base_answer: Tensor, answers: Tensor, w: Tensor, base: Tensor) -> Tensor:
    """[B, N, D]: v = base v_0 + sum_i w_i v_i; base_answer [B, N, D], answers [M, B, N, D], w [B, N, M], base [B, N]
    (the weights cast to the answers' dtype at the product; the sum in member order)."""
    if answers.shape[0] != w.shape[-1]:
        raise ValueError("one answer a member")
    v = base.to(base_answer.dtype)[..., None] * base_answer
    for i in range(answers.shape[0]):
        v = v + w[..., i].to(answers.dtype)[..., None] * answers[i]
    return v


# ------------------------------------------------------------------ the key binding
def phase_table(tag: str, pairs: int, reach: int) -> Tensor:
    """[pairs, reach, 256] float64: phases uniform on [0, 2 pi) from a CPU generator seeded crc32(tag)."""
    g = torch.Generator().manual_seed(zlib.crc32(str(tag).encode()))
    return torch.rand(int(pairs), int(reach), VOCAB, generator=g, dtype=torch.float64).mul_(TWO_PI)


def name_phases(table: Tensor, name: bytes) -> Tensor:
    """[pairs] float64: phi_j = sum_{k = 0}^{n - 1} table[j, k, name[n - 1 - k]] (mod 2 pi), summed from the last byte
    (k = 0) in the table's dtype; a name of 1 .. reach bytes."""
    name = bytes(name)
    if not 1 <= len(name) <= table.shape[1]:
        raise ValueError(f"a name holds 1 .. {table.shape[1]} bytes, not {len(name)}")
    phi = None
    for k in range(len(name)):
        p = table[:, k, name[len(name) - 1 - k]]
        phi = p if phi is None else phi + p
    return torch.remainder(phi, TWO_PI).double()


def name_kernel(phases_a: Tensor, phases_b: Tensor) -> Tensor:
    """sum_j cos(phase_j(a) - phase_j(b)), float64."""
    return torch.cos(phases_a.double() - phases_b.double()).sum(dim=-1)


# ------------------------------------------------------------------ the look read
@torch.no_grad()
def attention_weights(attn: SelfAttention, x: Tensor, rotary: Rotary, key_mask: Optional[Tensor] = None) -> Tensor:
    """[B, heads, N, N] float32: the softmax weights of ``attn`` on x [B, N, dim] with the rotary: q, k from
    ``attn.qkv``, logits q k^T / sqrt(head_dim), a key read when key_mask [B, N] allows it or it is the query itself."""
    b, n, d = x.shape
    qkv = attn.qkv(x).view(b, n, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
    q, k = rotary(qkv[0], qkv[1])
    logits = (q @ k.transpose(-1, -2)) / math.sqrt(attn.head_dim)
    if key_mask is not None:
        eye = torch.eye(n, dtype=torch.bool, device=x.device)
        allowed = key_mask.to(device=x.device, dtype=torch.bool)[:, None, None, :] | eye[None, None]
        logits = logits.masked_fill(~allowed, float("-inf"))
    return torch.softmax(logits.float(), dim=-1)


@contextlib.contextmanager
def record_looks(blocks: Sequence[nn.Module]) -> Iterator[List[Dict[str, object]]]:
    """Forward pre-hooks on each block's ``attn``: every call appends {"block": index, "weights": [B, heads, N, N]}
    computed by ``attention_weights`` from the very inputs the attention received. The hooks are removed on exit."""
    looks: List[Dict[str, object]] = []
    handles = []
    for i, blk in enumerate(blocks):
        def hook(module, args, i=i):
            x, rot = args[0], args[1]
            km = args[2] if len(args) > 2 else None
            looks.append({"block": i, "weights": attention_weights(module, x.detach(), rot, km)})
        handles.append(blk.attn.register_forward_pre_hook(hook))
    try:
        yield looks
    finally:
        for h in handles:
            h.remove()
