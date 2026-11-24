"""Transformer block, zero-initialized layers and the time embedding.

Block (pre-norm, three residual branches, no learned gates):
    x = x + SelfAttn( mod_1(LN(x)) )          rotary applied to q, k
    x = x + CrossAttn( mod_2(LN(x)), C )      C = the conditioning sequence
    x = x + MLP( mod_3(LN(x)) )
    mod_i(y) = y * (1 + ds_i(e)) + dh_i(e)

The modulation's last layer (e -> 6 d) starts at zero, so every norm starts out passing its input through.

The attribute names (attn.qkv, attn.out, cross.q, cross.kv, cross.out, mlp.0, mlp.2, modulation.1, proj) and the order in
which layers are constructed are part of the weight-file format and of the random-draw order at initialization: a layer
that is zeroed after construction still draws its default initialization first.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

Rotary = Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]


def identity_rotary(q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
    return q, k


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: Tensor, rotary: Rotary, key_mask: Optional[Tensor] = None) -> Tensor:
        """key_mask [B, N] bool (True = this token may be read), or None (every token reads every token, the path of
        earlier versions). With a mask a query reads key y only when key_mask[y] is True or y is the query itself."""
        b, n, d = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = rotary(q, k)
        if key_mask is None:
            y = F.scaled_dot_product_attention(q, k, v)
        else:
            eye = torch.eye(n, dtype=torch.bool, device=x.device)
            allowed = key_mask.to(device=x.device, dtype=torch.bool)[:, None, None, :] | eye[None, None]
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        return self.out(y.transpose(1, 2).reshape(b, n, d))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: Tensor, ctx: Tensor, ctx_mask: Optional[Tensor]) -> Tensor:
        b, n, d = x.shape
        l = ctx.shape[1]
        q = self.q(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(ctx).view(b, l, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        attn_mask = None
        if ctx_mask is not None:
            attn_mask = ctx_mask[:, None, None, :]                      # [B, 1, 1, L], True = attend
        y = F.scaled_dot_product_attention(q, kv[0], kv[1], attn_mask=attn_mask)
        return self.out(y.transpose(1, 2).reshape(b, n, d))


def zero_linear(in_dim: int, out_dim: int) -> nn.Linear:
    """A Linear layer whose weight and bias are zero. It draws its default initialization first (and discards it)."""
    lin = nn.Linear(in_dim, out_dim)
    nn.init.zeros_(lin.weight)
    nn.init.zeros_(lin.bias)
    return lin


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross = CrossAttention(dim, num_heads)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim))
        self.modulation = nn.Sequential(nn.SiLU(), zero_linear(dim, 6 * dim))

    def forward(self, x: Tensor, ctx: Tensor, ctx_mask: Optional[Tensor], e: Tensor, rotary: Rotary,
                key_mask: Optional[Tensor] = None) -> Tensor:
        """key_mask: see ``SelfAttention.forward`` (None: the path of earlier versions)."""
        ds1, dh1, ds2, dh2, ds3, dh3 = self.modulation(e).unsqueeze(1).chunk(6, dim=-1)
        if key_mask is None:
            x = x + self.attn(self.norm1(x) * (1 + ds1) + dh1, rotary)
        else:
            x = x + self.attn(self.norm1(x) * (1 + ds1) + dh1, rotary, key_mask)
        x = x + self.cross(self.norm2(x) * (1 + ds2) + dh2, ctx, ctx_mask)
        x = x + self.mlp(self.norm3(x) * (1 + ds3) + dh3)
        return x

    def zero_outputs(self) -> None:
        """Zero the three residual-branch output layers (weights and biases): the block becomes the identity map while
        its gradients still flow."""
        with torch.no_grad():
            for lin in (self.attn.out, self.cross.out, self.mlp[2]):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)


class ZeroHead(nn.Module):
    """LayerNorm (no affine) then a Linear(dim, out_dim) whose weight and bias start at zero."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.proj = zero_linear(dim, out_dim)

    def forward(self, h: Tensor) -> Tensor:
        return self.proj(self.norm(h))


class TimeEmbedding(nn.Module):
    """Sinusoidal features of t * scale, then a two-layer MLP."""

    def __init__(self, freq_dim: int, dim: int, scale: float = 1000.0):
        super().__init__()
        half = freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)
        self.scale = float(scale)
        self.mlp = nn.Sequential(nn.Linear(2 * half, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: Tensor) -> Tensor:
        arg = (t.float() * self.scale)[:, None] * self.freqs[None, :]
        return self.mlp(torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1))
