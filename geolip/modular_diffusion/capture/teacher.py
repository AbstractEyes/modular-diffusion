"""The frozen teacher: Stable Diffusion 1.5's U-Net, its text encoder and tokenizer (and optionally its VAE), loaded from
a local diffusers folder. Nothing here downloads: every load passes ``local_files_only``.

    find_snapshot("runwayml/stable-diffusion-v1-5", revision="451f4fe1")    the folder in the local hub cache
    teacher = load_teacher(folder, device="cpu", dtype=torch.float32)
    ids = teacher.tokenize(["a red house"]); states = teacher.text_states(ids)
    eps = teacher.predict(z_t, t, states)

``tiny_teacher(seed)`` makes a teacher of the same kind with random numbers and no files (a tiny U-Net with the
teacher's block types and layer counts, a tiny text encoder, a character tokenizer), for the tests; with ``vae=True`` it
carries ``tiny_vae(seed)``, a tiny random image decoder (16 x 16 latents to 32 x 32 pictures).

A SOURCE of another kind with the same U-Net layout (a flow-matching fine-tune that predicts a velocity on a continuous
time) is loaded by the same call with its own U-Net folder:

    source = load_teacher(sd15_folder, unet=unet_folder, kind="flow", shift=2.5)     # unguided side: zero states

Its process (``capture.process``), its unguided side and its latent-scale ratio ``kappa`` travel with it; the reads,
the sampler and the pictures take them from the source.
"""
from __future__ import annotations

import json
import os
from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from .process import EpsProcess, Process, make_process
from .schedule import BETA_END, BETA_START, STEPS, sd15_alphas_cumprod

TOKENS = 77
UNGUIDED = ("empty", "zeros")
UNGUIDED_DEFAULT = {"eps": "empty", "flow": "zeros"}       # a flow source was trained with zero states as its unguided

SD15_DOWN = ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")
SD15_UP = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")

# The tiny test U-Net: the teacher's block types and layer counts, narrow channels, 16 x 16 latents.
TINY = dict(sample_size=16, in_channels=4, out_channels=4, down_block_types=SD15_DOWN, up_block_types=SD15_UP,
            block_out_channels=(32, 64, 64, 64), layers_per_block=2, cross_attention_dim=32, attention_head_dim=4,
            norm_num_groups=8)
TINY_VOCAB = 1000
# The tiny test image decoder: 4 latent channels, two stages (one downsampling: 16 x 16 latents, 32 x 32 pictures).
TINY_VAE = dict(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
                up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"), block_out_channels=(4, 4), layers_per_block=1,
                latent_channels=4, norm_num_groups=2, sample_size=32)


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
    """The frozen teacher (a source). ``unet`` is a diffusers ``UNet2DConditionModel``; ``alphas_cumprod`` the schedule
    of ``schedule`` (float64); ``tokenize`` gives [B, 77] ids, ``text_states`` [B, 77, D] states.

    A source also carries:
        process     how it noises, answers and steps (``process.Process``; the noise-predicting ``eps`` by default)
        unguided    the unguided side of guidance: ``empty`` (the empty prompt's states) or ``zeros`` (zero states of the
                    same shape); ``unguided_states(batch)`` is what every guided path is given
        kappa       the ratio of the source's clean-latent scale to the standard one (1.0 by default): clean latents given
                    to the source are multiplied by it (``source_latents``), its samples divided by it before decoding
                    (``picture_latents``)"""

    def __init__(self, unet: nn.Module, text_encoder: nn.Module, tokenizer, vae: Optional[nn.Module] = None,
                 device="cpu", dtype: torch.dtype = torch.float32, process: Optional[Process] = None,
                 unguided: Optional[str] = None, kappa: float = 1.0):
        self.unet = _freeze(unet)
        self.text_encoder = _freeze(text_encoder)
        self.tokenizer = tokenizer
        self.vae = _freeze(vae) if vae is not None else None
        self.device = torch.device(device)
        self.dtype = dtype
        self.alphas_cumprod = sd15_alphas_cumprod()
        self.context_dim = int(unet.config.cross_attention_dim)
        self.process = process if process is not None else EpsProcess(self.alphas_cumprod)
        unguided = unguided or UNGUIDED_DEFAULT[self.process.kind]
        if unguided not in UNGUIDED:
            raise ValueError(f"the unguided side is one of {list(UNGUIDED)}, not {unguided!r}")
        self.unguided = unguided
        self.kappa = float(kappa)
        if not self.kappa > 0.0:
            raise ValueError(f"kappa is a positive number, not {kappa}")

    @property
    def kind(self) -> str:
        return self.process.kind

    def unguided_states(self, batch: int) -> Tensor:
        """The unguided side's states for ``batch`` rows: the empty prompt's, or zeros of the same shape."""
        if self.unguided == "empty":
            return self.empty_states(batch)
        return torch.zeros(int(batch), TOKENS, self.context_dim, device=self.device, dtype=self.dtype)

    def source_latents(self, x0: Tensor) -> Tensor:
        """Clean latents of the standard scale as this source reads them: multiplied by kappa (unchanged at 1.0)."""
        return x0 if self.kappa == 1.0 else x0 * self.kappa

    def picture_latents(self, z: Tensor) -> Tensor:
        """This source's samples on the standard scale, for decoding: divided by kappa (unchanged at 1.0)."""
        return z if self.kappa == 1.0 else z / self.kappa

    def record(self) -> dict:
        """The source's process, unguided side and kappa, for a result file."""
        rec = dict(self.process.record())
        rec.update({"unguided": self.unguided, "kappa": self.kappa})
        return rec

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
        cached = getattr(self, "_empty_one_cache", None)
        if cached is None or cached[0] != self.device or cached[1] != self.dtype:
            one = self.text_states(self.empty_ids(1))
            self._empty_one_cache = (self.device, self.dtype, one)
        else:
            one = cached[2]
        return one.expand(int(batch), -1, -1).clone()

    def to(self, device=None, dtype: Optional[torch.dtype] = None) -> "Teacher":
        """This teacher moved to a device and/or cast to a precision (its modules are moved in place)."""
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        self._empty_one_cache = None
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


def tiny_vae(seed: int = 0):
    """The tiny random image decoder of the tests (``TINY_VAE``): a diffusers ``AutoencoderKL`` with 4 latent channels
    and one downsampling, so a 16 x 16 latent decodes to a 32 x 32 picture. Its numbers are drawn from the seed; frozen,
    eval."""
    from diffusers import AutoencoderKL
    g = torch.Generator().manual_seed(int(seed) + 211)
    vae = AutoencoderKL(**TINY_VAE)
    with torch.no_grad():
        for p in vae.parameters():
            if p.dim() > 1:
                fan_in = p[0].numel()
                p.copy_(torch.randn(p.shape, generator=g) / fan_in ** 0.5)
            else:
                p.copy_(torch.randn(p.shape, generator=g) * 0.02 + (1.0 if bool((p == 1).all()) else 0.0))
    return _freeze(vae)


def tiny_teacher(seed: int = 0, vae: bool = False, kind: str = "eps", shift: Optional[float] = None,
                 unguided: Optional[str] = None, kappa: float = 1.0, unet_seed: Optional[int] = None) -> Teacher:
    """A teacher with random numbers and no files: ``tiny_unet`` (every tensor nonzero), ``TinyTextEncoder``, a
    character tokenizer; with ``vae`` also ``tiny_vae(seed)`` as its image decoder. CPU, float32. ``kind``, ``shift``,
    ``unguided`` and ``kappa`` as ``load_teacher``'s; ``unet_seed`` (the seed by default) draws the U-Net alone, so two
    sources can share the text encoder and the image decoder and differ in their U-Nets only."""
    useed = int(seed) if unet_seed is None else int(unet_seed)
    unet = tiny_unet(useed)
    _perturb(unet, useed)
    process = make_process(kind, shift)
    return Teacher(unet, TinyTextEncoder(TINY["cross_attention_dim"], seed=seed), CharTokenizer(),
                   tiny_vae(seed) if vae else None, process=process, unguided=unguided, kappa=kappa)


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


def load_unet_folder(folder: str):
    """A U-Net from a folder of its own (a diffusers ``config.json`` and ``diffusion_pytorch_model.safetensors``),
    read from safetensors only, local files only, float32 on the CPU."""
    from diffusers import UNet2DConditionModel
    if not (os.path.isfile(os.path.join(folder, "config.json"))
            and os.path.isfile(os.path.join(folder, "diffusion_pytorch_model.safetensors"))):
        raise FileNotFoundError(f"a U-Net folder holds config.json and diffusion_pytorch_model.safetensors: {folder}")
    return UNet2DConditionModel.from_pretrained(folder, local_files_only=True, use_safetensors=True)


def with_unet(source: Teacher, folder: str) -> Teacher:
    """The source with its U-Net replaced by the one of ``folder`` (``load_unet_folder``), on the source's device and in
    its precision; everything else kept (for a tiny teacher's text encoder beside a U-Net written to a folder)."""
    unet = _owned(load_unet_folder(folder).to(device=source.device, dtype=source.dtype))
    if int(unet.config.cross_attention_dim) != source.context_dim:
        raise ValueError(f"the U-Net of {folder} reads states of width {unet.config.cross_attention_dim}, the source's "
                         f"text encoder gives {source.context_dim}")
    source.unet = _freeze(unet)
    return source


def load_teacher(path: str, device="cpu", dtype: torch.dtype = torch.float32, vae: bool = False,
                 unet: Optional[str] = None, kind: str = "eps", shift: Optional[float] = None,
                 unguided: Optional[str] = None, kappa: float = 1.0,
                 vae_dtype: Optional[torch.dtype] = None) -> Teacher:
    """The teacher from a local diffusers folder (unet/, text_encoder/, tokenizer/, scheduler/, and vae/ when asked),
    frozen, on ``device`` in ``dtype``. Never downloads.

    Another source of the same U-Net layout: ``unet`` names a folder of its own U-Net (a diffusers ``config.json`` and
    ``diffusion_pytorch_model.safetensors``, read from safetensors only), while the text encoder, the tokenizer and the
    image decoder still come from ``path``. ``kind`` is ``eps`` (the noise; the folder's scheduler is checked) or
    ``flow`` (the velocity on a continuous time; ``shift`` required); ``unguided`` is ``empty`` or ``zeros`` (by
    default ``empty`` for eps, ``zeros`` for flow); ``kappa`` the source's latent-scale ratio."""
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    process = make_process(kind, shift)
    if process.kind == "eps":
        check_schedule(path)
    if unet:
        unet = load_unet_folder(unet)
    else:
        unet = UNet2DConditionModel.from_pretrained(path, subfolder="unet", local_files_only=True)
    text = CLIPTextModel.from_pretrained(path, subfolder="text_encoder", local_files_only=True)
    tok = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer", local_files_only=True)
    v = AutoencoderKL.from_pretrained(path, subfolder="vae", local_files_only=True) if vae else None
    unet = _owned(unet.to(device=device, dtype=dtype))
    text = _owned(text.to(device=device, dtype=dtype))
    if v is not None:
        v = _owned(v.to(device=device, dtype=dtype if vae_dtype is None else vae_dtype))
    return Teacher(unet, text, tok, v, device, dtype, process=process, unguided=unguided, kappa=kappa)


def _owned(module: nn.Module) -> nn.Module:
    """Every CPU tensor of a loaded module copied into ordinary memory of its own. Tensors read from a memory-mapped
    file can sit at addresses the CPU kernels treat differently from freshly allocated ones, so the teacher and a copy
    of its numbers (an inherited engine) could disagree in the last digits; after this they take the same path."""
    with torch.no_grad():
        for p in module.parameters():
            if p.device.type == "cpu":
                p.data = p.data.clone()
        for name, b in list(module.named_buffers()):
            if b is not None and b.device.type == "cpu":
                owner_name, _, leaf = name.rpartition(".")
                owner = module.get_submodule(owner_name) if owner_name else module
                owner._buffers[leaf] = b.clone()
    return module
