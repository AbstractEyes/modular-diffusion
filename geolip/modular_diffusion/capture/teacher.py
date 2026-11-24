"""The frozen teacher: Stable Diffusion 1.5's U-Net, its text encoder and tokenizer (and optionally its VAE), loaded from
a local diffusers folder. Nothing here downloads: every load passes ``local_files_only``.

    find_snapshot("runwayml/stable-diffusion-v1-5", revision="451f4fe1")    the folder in the local hub cache
    teacher = load_teacher(folder, device="cpu", dtype=torch.float32)
    ids = teacher.tokenize(["a red house"]); states = teacher.text_states(ids)
    eps = teacher.predict(z_t, t, states)

``tiny_teacher(seed)`` makes a teacher of the same kind with random numbers and no files (a tiny U-Net with the
teacher's block types and layer counts, a tiny text encoder, a character tokenizer), for the tests.
"""
from __future__ import annotations

import json
import os
from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from .schedule import BETA_END, BETA_START, STEPS, sd15_alphas_cumprod

TOKENS = 77

SD15_DOWN = ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")
SD15_UP = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")

# The tiny test U-Net: the teacher's block types and layer counts, narrow channels, 16 x 16 latents.
TINY = dict(sample_size=16, in_channels=4, out_channels=4, down_block_types=SD15_DOWN, up_block_types=SD15_UP,
            block_out_channels=(32, 64, 64, 64), layers_per_block=2, cross_attention_dim=32, attention_head_dim=4,
            norm_num_groups=8)
TINY_VOCAB = 1000


def _freeze(m: nn.Module) -> nn.Module:
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


class CharTokenizer:
    """A tokenizer for the tests: one id per character (1 begins, 0 ends and pads, the rest 2 .. 999), 77 ids a row."""
    bos, eos = 1, 0

    def __call__(self, prompts: Sequence[str]) -> Tensor:
        rows = []
        for p in prompts:
            body = [2 + (ord(c) % (TINY_VOCAB - 2)) for c in str(p)][:TOKENS - 2]
            row = [self.bos] + body + [self.eos]
            rows.append(row + [self.eos] * (TOKENS - len(row)))
        return torch.tensor(rows, dtype=torch.long)


class TinyTextEncoder(nn.Module):
    """A stand-in text encoder for the tests: ids [B, 77] -> ([B, 77, dim],) (a tuple, like the library encoders)."""

    def __init__(self, dim: int = 32, vocab: int = TINY_VOCAB, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed) + 17)
        self.embed = nn.Parameter(torch.randn(vocab, dim, generator=g) * 0.5)
        self.position = nn.Parameter(torch.randn(TOKENS, dim, generator=g) * 0.1)
        self.mix = nn.Parameter(torch.randn(dim, dim, generator=g) / dim ** 0.5)

    def forward(self, ids: Tensor):
        x = self.embed[ids.long()] + self.position[: ids.shape[1]]
        return (torch.tanh(x @ self.mix),)


class Teacher:
    """The frozen teacher. ``unet`` is a diffusers ``UNet2DConditionModel``; ``alphas_cumprod`` the schedule of
    ``schedule`` (float64); ``tokenize`` gives [B, 77] ids, ``text_states`` [B, 77, D] states."""

    def __init__(self, unet: nn.Module, text_encoder: nn.Module, tokenizer, vae: Optional[nn.Module] = None,
                 device="cpu", dtype: torch.dtype = torch.float32):
        self.unet = _freeze(unet)
        self.text_encoder = _freeze(text_encoder)
        self.tokenizer = tokenizer
        self.vae = _freeze(vae) if vae is not None else None
        self.device = torch.device(device)
        self.dtype = dtype
        self.alphas_cumprod = sd15_alphas_cumprod()
        self.context_dim = int(unet.config.cross_attention_dim)

    def tokenize(self, prompts: Sequence[str]) -> Tensor:
        prompts = [str(p) for p in prompts]
        if isinstance(self.tokenizer, CharTokenizer):
            return self.tokenizer(prompts).to(self.device)
        out = self.tokenizer(prompts, padding="max_length", max_length=TOKENS, truncation=True, return_tensors="pt")
        return out.input_ids.to(self.device)

    @torch.no_grad()
    def text_states(self, ids: Tensor) -> Tensor:
        out = self.text_encoder(ids.to(self.device).long())
        return out[0].to(self.dtype)

    def empty_ids(self, batch: int) -> Tensor:
        return self.tokenize([""] * int(batch))

    def empty_states(self, batch: int) -> Tensor:
        one = self.text_states(self.empty_ids(1))
        return one.expand(int(batch), -1, -1).clone()

    def to(self, device=None, dtype: Optional[torch.dtype] = None) -> "Teacher":
        """This teacher moved to a device and/or cast to a precision (its modules are moved in place)."""
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        for m in (self.unet, self.text_encoder, self.vae):
            if m is not None:
                m.to(device=self.device, dtype=self.dtype)
        return self

    @torch.no_grad()
    def predict(self, z: Tensor, t: Tensor, states: Tensor) -> Tensor:
        """The teacher's answer. The inputs are cast to the U-Net's own precision and the answer is returned in the
        latents' precision, so a teacher kept in half precision answers full-precision latents."""
        own = next(self.unet.parameters()).dtype
        return self.unet(z.to(own), t, encoder_hidden_states=states.to(own)).sample.to(z.dtype)

    def __call__(self, z: Tensor, t: Tensor, states: Tensor) -> Tensor:
        return self.predict(z, t, states)


def tiny_unet(seed: int = 0):
    """The tiny random U-Net of the tests (``TINY``), its numbers drawn from the seed."""
    from diffusers import UNet2DConditionModel
    torch.manual_seed(int(seed))
    return UNet2DConditionModel(**TINY)


def _perturb(unet: nn.Module, seed: int) -> None:
    """Small random numbers on every tensor a fresh U-Net starts at zero (its output layer, its attention outputs),
    so that every part of the tiny teacher takes part in its answer."""
    g = torch.Generator().manual_seed(int(seed) + 101)
    with torch.no_grad():
        for p in unet.parameters():
            if p.dim() > 0 and bool((p == 0).all()):
                p.copy_(torch.randn(p.shape, generator=g) * 0.02)


def tiny_teacher(seed: int = 0) -> Teacher:
    """A teacher with random numbers and no files: ``tiny_unet`` (every tensor nonzero), ``TinyTextEncoder``, a
    character tokenizer. CPU, float32."""
    unet = tiny_unet(seed)
    _perturb(unet, seed)
    return Teacher(unet, TinyTextEncoder(TINY["cross_attention_dim"], seed=seed), CharTokenizer())


def _hub_cache(cache_dir: Optional[str]) -> str:
    if cache_dir:
        return cache_dir
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return os.path.join(home, "hub")


def find_snapshot(repo_id: str = "runwayml/stable-diffusion-v1-5", revision: Optional[str] = None,
                  cache_dir: Optional[str] = None) -> str:
    """The snapshot folder of a model in the local hub cache layout (``<cache>/models--org--name/snapshots/<rev>``),
    found by reading folders only. ``revision`` may be a commit, a prefix of one, or a branch name kept under refs/;
    without it the branch main is used, or the only snapshot. FileNotFoundError when it is not there."""
    root = os.path.join(_hub_cache(cache_dir), "models--" + repo_id.replace("/", "--"))
    snaps = os.path.join(root, "snapshots")
    if not os.path.isdir(snaps):
        raise FileNotFoundError(f"{repo_id} is not in the local hub cache under {root}")
    names = sorted(os.listdir(snaps))
    rev = revision or "main"
    ref = os.path.join(root, "refs", rev)
    if os.path.isfile(ref):
        with open(ref, encoding="utf-8") as fh:
            rev = fh.read().strip()
    found = [n for n in names if n.startswith(rev)]
    if not found and revision is None and len(names) == 1:
        found = names
    if len(found) != 1:
        raise FileNotFoundError(f"revision {revision!r} of {repo_id}: snapshots {names}")
    return os.path.join(snaps, found[0])


def check_schedule(path: str) -> None:
    """A folder's scheduler settings must be the teacher's (scaled linear, 0.00085 .. 0.012, 1,000 steps, eps)."""
    cfg_path = os.path.join(path, "scheduler", "scheduler_config.json")
    if not os.path.isfile(cfg_path):
        return
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    want = dict(beta_start=BETA_START, beta_end=BETA_END, beta_schedule="scaled_linear", num_train_timesteps=STEPS)
    for k, v in want.items():
        if k in cfg and cfg[k] != v:
            raise ValueError(f"the folder's scheduler has {k} = {cfg[k]!r}, the teacher's schedule needs {v!r}")
    if cfg.get("prediction_type", "epsilon") != "epsilon":
        raise ValueError("the teacher predicts the noise (prediction_type epsilon)")


def load_teacher(path: str, device="cpu", dtype: torch.dtype = torch.float32, vae: bool = False) -> Teacher:
    """The teacher from a local diffusers folder (unet/, text_encoder/, tokenizer/, scheduler/, and vae/ when asked),
    frozen, on ``device`` in ``dtype``. Never downloads."""
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    check_schedule(path)
    unet = UNet2DConditionModel.from_pretrained(path, subfolder="unet", local_files_only=True)
    text = CLIPTextModel.from_pretrained(path, subfolder="text_encoder", local_files_only=True)
    tok = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer", local_files_only=True)
    v = AutoencoderKL.from_pretrained(path, subfolder="vae", local_files_only=True) if vae else None
    unet = _owned(unet.to(device=device, dtype=dtype))
    text = _owned(text.to(device=device, dtype=dtype))
    if v is not None:
        v = _owned(v.to(device=device, dtype=dtype))
    return Teacher(unet, text, tok, v, device, dtype)


def _owned(module: nn.Module) -> nn.Module:
    """Every tensor of a loaded module copied into ordinary memory of its own. Tensors read from a memory-mapped file
    can sit at addresses the CPU kernels treat differently from freshly allocated ones, so the teacher and a copy of
    its numbers (an inherited engine) could disagree in the last digits; after this they take the same path."""
    with torch.no_grad():
        for p in module.parameters():
            p.data = p.data.clone()
        for name, b in list(module.named_buffers()):
            owner_name, _, leaf = name.rpartition(".")
            owner = module.get_submodule(owner_name) if owner_name else module
            owner._buffers[leaf] = b.clone()
    return module
