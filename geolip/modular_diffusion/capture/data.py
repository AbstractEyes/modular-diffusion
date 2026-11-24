"""Training data for capture: latent shards on disk, the encoder that writes them, and random batches for tests.

A SHARD is one safetensors file holding

    latents   float16 [N, 4, h, w]   the image encoder's latents, already multiplied by the latent scale (0.18215)
    ids       int32   [N, 77]        the tokenizer's ids of each row's caption (padded to 77, truncated)

and a folder of shards carries ``index.json``: the shard file names in order, each shard's row count, the shard size,
the picture size and the latent scale. Rows are numbered by position: shard 0's rows first, in the order they were
encoded.

    LatentBatches(folder, batch, seed)     an endless, deterministic stream of (x0 float32 [B, 4, h, w], ids long [B, 77]);
                                           also a batch source: source(step, batch) gives batch number ``step``
    encode_shards(images, captions, out_folder, vae, tokenize, ...)
                                           writes shards from pictures and captions; a stopped job continues where it
                                           stopped
    synthetic_batches(size, ...)           a source of random latents and ids, for tests
    parquet_pictures(path, n)              the first n pictures and captions of one parquet file, in file order

The stream's order: pass e (e = 0, 1, ...) visits the shards in a random order drawn from ``seed`` and ``e``, and the
rows of each shard in a random order drawn from ``seed``, ``e`` and the shard's position; a batch takes the next B rows
of that sequence and may run across shards and passes. The order depends only on the seed, the shards' row counts and
the batch size, so a stream started at batch k gives exactly the batches k, k + 1, ... of a stream started at 0.

Nothing here chooses or downloads a dataset; the pictures and captions are the caller's.
"""
from __future__ import annotations

import json
import math
import os
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

INDEX = "index.json"
TOKENS = 77
LATENT_SCALE = 0.18215
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def shard_name(i: int) -> str:
    return f"shard_{i:05d}.safetensors"


def read_index(folder: str) -> dict:
    with open(os.path.join(folder, INDEX), encoding="utf-8") as fh:
        return json.load(fh)


def _write_index(folder: str, index: dict) -> None:
    tmp = os.path.join(folder, INDEX + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1)
    os.replace(tmp, os.path.join(folder, INDEX))


def read_shard(path: str) -> Tuple[Tensor, Tensor]:
    """(latents float16 [N, 4, h, w], ids int32 [N, 77]) of one shard file."""
    from safetensors.torch import load_file
    d = load_file(path)
    return d["latents"], d["ids"]


def write_shard(path: str, latents: Tensor, ids: Tensor) -> None:
    """Writes one shard (through a temporary name, so a stopped write never leaves a whole-looking file)."""
    from safetensors.torch import save_file
    if latents.dim() != 4 or ids.dim() != 2 or ids.shape[1] != TOKENS or latents.shape[0] != ids.shape[0]:
        raise ValueError(f"a shard holds latents [N, C, h, w] and ids [N, {TOKENS}]: {tuple(latents.shape)}, "
                         f"{tuple(ids.shape)}")
    tmp = path + ".tmp"
    save_file({"latents": latents.detach().to("cpu", torch.float16).contiguous(),
               "ids": ids.detach().to("cpu", torch.int32).contiguous()}, tmp)
    os.replace(tmp, path)


# ------------------------------------------------------------------ the stream of batches
class LatentBatches:
    """An endless, deterministic stream of training batches from a folder of shards (see the module's text for the
    order). Two ways to read it, giving the same batches:

        source(step, batch)    batch number ``step`` of size ``batch``: rows step * batch .. (step + 1) * batch - 1 of
                               the row sequence; a pure function of (step, batch), so a resumed job sees the same data
        iter(source)           batches of the size given at construction, from batch 0 (``stream(start)`` from batch
                               ``start``)

    Both give (x0 float32 [B, 4, h, w], ids long [B, 77]) on ``device``. One shard is held in memory at a time, and
    the shards before a starting row are never read. ``latent_shape`` is (C, h, w)."""

    def __init__(self, folder: str, batch: Optional[int] = None, seed: int = 0,
                 device: Union[str, torch.device] = "cpu"):
        self.folder = str(folder)
        self.index = read_index(self.folder)
        self.files: List[str] = list(self.index["shards"])
        self.counts: List[int] = [int(c) for c in self.index["counts"]]
        if not self.files or sum(self.counts) == 0:
            raise ValueError(f"no rows in {self.folder}")
        self.batch = None if batch is None else int(batch)
        if self.batch is not None and self.batch < 1:
            raise ValueError("batch is at least 1")
        self.seed = int(seed)
        self.device = torch.device(device)
        self.rows = sum(self.counts)
        self._held: Optional[Tuple[int, Tensor, Tensor]] = None
        from safetensors import safe_open
        with safe_open(os.path.join(self.folder, self.files[0]), framework="pt") as fh:
            self.latent_shape = tuple(int(x) for x in fh.get_slice("latents").get_shape()[1:])

    def __call__(self, step: int, batch: int) -> Tuple[Tensor, Tensor]:
        """Batch number ``step`` of size ``batch``."""
        if int(batch) < 1 or int(step) < 0:
            raise ValueError("step >= 0 and batch >= 1")
        return next(self._batches(int(step) * int(batch), int(batch)))

    def __len__(self) -> int:
        """Rows in one pass."""
        return self.rows

    def _shard_order(self, e: int) -> List[int]:
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + e)
        return torch.randperm(len(self.files), generator=g).tolist()

    def _row_order(self, e: int, shard: int) -> Tensor:
        g = torch.Generator().manual_seed((self.seed * 1_000_003 + e) * 100_003 + shard)
        return torch.randperm(self.counts[shard], generator=g)

    def _shard(self, i: int) -> Tuple[Tensor, Tensor]:
        if self._held is None or self._held[0] != i:
            lat, ids = read_shard(os.path.join(self.folder, self.files[i]))
            self._held = (i, lat, ids)
        return self._held[1], self._held[2]

    def _rows(self, start_row: int) -> Iterator[Tuple[Tensor, Tensor]]:
        """Pieces (latents, ids) of the row sequence from row ``start_row`` on, in order."""
        e, r = divmod(int(start_row), self.rows)
        while True:
            for s in self._shard_order(e):
                n = self.counts[s]
                if r >= n:
                    r -= n
                    continue
                if n == 0:
                    continue
                lat, ids = self._shard(s)
                order = self._row_order(e, s)[r:]
                r = 0
                yield lat[order], ids[order]
            e += 1

    def _batches(self, start_row: int, batch: int) -> Iterator[Tuple[Tensor, Tensor]]:
        lat_parts: List[Tensor] = []
        id_parts: List[Tensor] = []
        have = 0
        for lat, ids in self._rows(start_row):
            while lat.shape[0]:
                take = min(batch - have, lat.shape[0])
                lat_parts.append(lat[:take])
                id_parts.append(ids[:take])
                lat, ids = lat[take:], ids[take:]
                have += take
                if have == batch:
                    x0 = torch.cat(lat_parts).to(self.device, torch.float32)
                    tok = torch.cat(id_parts).to(self.device, torch.long)
                    lat_parts, id_parts, have = [], [], 0
                    yield x0, tok

    def stream(self, start: int = 0) -> Iterator[Tuple[Tensor, Tensor]]:
        """Batches of the construction's size from batch number ``start`` on."""
        if self.batch is None:
            raise ValueError("iterating needs the batch size given at construction")
        return self._batches(int(start) * self.batch, self.batch)

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        return self.stream(0)


# ------------------------------------------------------------------ the encoder
def _image_files(folder: str) -> List[str]:
    return sorted(os.path.join(folder, n) for n in os.listdir(folder) if n.lower().endswith(IMAGE_SUFFIXES))


def center_crop(picture, size: int = 512) -> Tensor:
    """[3, size, size] float32 in [-1, 1]: the picture (a PIL image or a path) resized so its shorter side is ``size``
    (bilinear) and cropped at the center, the preparation of the published text-to-image fine-tuning recipe."""
    from PIL import Image
    img = Image.open(picture) if isinstance(picture, (str, os.PathLike)) else picture
    img = img.convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    nw, nh = max(size, round(w * scale)), max(size, round(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    left, top = (nw - size) // 2, (nh - size) // 2
    img = img.crop((left, top, left + size, top + size))
    a = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).float()
    return a / 127.5 - 1.0


def encode_shards(images: Union[str, Iterable], captions: Sequence[str], out_folder: str, vae,
                  tokenize: Callable[[List[str]], Tensor], shard_size: int = 1000, batch: int = 8, size: int = 512,
                  scale: float = LATENT_SCALE, latent: str = "sample", seed: int = 0,
                  device: Union[str, torch.device] = "cpu", dtype: torch.dtype = torch.float32,
                  log: Optional[Callable[[str], None]] = print) -> dict:
    """Encodes pictures and their captions into shards of ``shard_size`` rows in ``out_folder``; returns the index.

    ``images``: a folder (its picture files in sorted name order) or an iterable of PIL images or paths, aligned with
    ``captions``. ``vae``: an image encoder with the diffusers interface (``vae.encode(x).latent_dist``) on ``device``;
    the latent is ``latent_dist.sample`` (the published recipe; a generator per shard seeded from ``seed`` and the
    shard's number, so a shard re-encoded gives the same rows) or, with ``latent="mean"``, its mean; times ``scale``.
    ``tokenize``: a callable from a list of captions to ids [B, 77].

    Resumable: the index lists only finished shards, so a stopped job re-run with the same arguments skips the rows of
    the finished shards and encodes the rest. The last shard may be short."""
    if latent not in ("sample", "mean"):
        raise ValueError(f"latent is sample or mean, not {latent!r}")
    captions = list(captions)
    pictures = _image_files(images) if isinstance(images, (str, os.PathLike)) else images
    if hasattr(pictures, "__len__") and len(pictures) != len(captions):
        raise ValueError(f"{len(pictures)} pictures and {len(captions)} captions")
    os.makedirs(out_folder, exist_ok=True)
    settings = {"shard_size": int(shard_size), "size": int(size), "scale": float(scale), "latent": latent,
                "seed": int(seed), "rows": len(captions)}
    path = os.path.join(out_folder, INDEX)
    if os.path.exists(path):
        index = read_index(out_folder)
        old = {k: index.get(k) for k in settings}
        if old != settings:
            raise ValueError(f"{out_folder} holds shards made with other settings: {old} != {settings}")
    else:
        index = dict(settings, shards=[], counts=[])
        _write_index(out_folder, index)
    done_shards = len(index["shards"])
    skip = done_shards * int(shard_size)
    n_shards = math.ceil(len(captions) / int(shard_size)) if captions else 0
    if log:
        log(f"encode_shards: {len(captions)} rows, {n_shards} shards of {shard_size}; {done_shards} done")
    lat_rows: List[Tensor] = []
    pic_buf: List[Tensor] = []
    shard = done_shards
    gen = None

    def flush_pictures():
        nonlocal gen
        if not pic_buf:
            return
        if gen is None:
            gen = torch.Generator(device=torch.device(device)).manual_seed(int(seed) * 1_000_003 + shard)
        x = torch.stack(pic_buf).to(device, dtype)
        with torch.no_grad():
            dist = vae.encode(x).latent_dist
            z = dist.sample(generator=gen) if latent == "sample" else dist.mean
        lat_rows.append((z.float() * float(scale)).cpu())
        pic_buf.clear()

    def flush_shard(n_rows: int):
        nonlocal shard, gen
        first = shard * int(shard_size)
        lat = torch.cat(lat_rows)
        ids = tokenize(captions[first:first + n_rows])
        ids = torch.as_tensor(ids)
        write_shard(os.path.join(out_folder, shard_name(shard)), lat, ids)
        index["shards"].append(shard_name(shard))
        index["counts"].append(int(n_rows))
        _write_index(out_folder, index)
        if log:
            log(f"encode_shards: shard {shard + 1}/{n_shards} written ({n_rows} rows)")
        lat_rows.clear()
        shard += 1
        gen = None

    in_shard = 0
    for i, pic in enumerate(pictures):
        if i >= len(captions):
            raise ValueError("more pictures than captions")
        if i < skip:
            continue
        pic_buf.append(center_crop(pic, size))
        in_shard += 1
        if len(pic_buf) == int(batch) or in_shard == int(shard_size):
            flush_pictures()
        if in_shard == int(shard_size):
            flush_shard(in_shard)
            in_shard = 0
    if in_shard:
        flush_pictures()
        flush_shard(in_shard)
    if sum(index["counts"]) != len(captions):
        raise ValueError(f"{sum(index['counts'])} rows encoded for {len(captions)} captions (fewer pictures?)")
    return index


# ------------------------------------------------------------------ pictures from a parquet file
def parquet_pictures(path: str, n: int, image: str = "image", caption: str = "answer",
                     keep: Sequence[str] = ("id", "license", "file_name")) -> Tuple[list, List[str], List[dict]]:
    """The first ``n`` rows of ONE parquet file, in file order: (pictures: RGB PIL images, captions: str, meta: one dict
    a row of the ``keep`` columns the file has). The file is read in batches of rows, never whole. The image cell is a
    struct with ``bytes`` (an encoded picture, decoded here from memory); the caption cell is a string or a list of
    strings (the first is taken). ValueError when the file holds fewer than ``n`` rows, lacks the image or caption
    column, or a row's picture or caption is empty."""
    import io
    import pyarrow.parquet as pq
    from PIL import Image
    n = int(n)
    if n < 1:
        raise ValueError("n is at least 1")
    pf = pq.ParquetFile(path)
    total = int(pf.metadata.num_rows)
    if total < n:
        raise ValueError(f"{path} holds {total} rows, fewer than the {n} asked for")
    names = list(pf.schema_arrow.names)
    for col in (image, caption):
        if col not in names:
            raise ValueError(f"{path} has no column {col!r}; its columns: {names}")
    kept = [k for k in keep if k in names and k not in (image, caption)]
    pictures: list = []
    captions: List[str] = []
    meta: List[dict] = []
    for batch in pf.iter_batches(batch_size=min(64, n), columns=[image, caption] + kept):
        cols = batch.to_pydict()
        for i in range(batch.num_rows):
            row = len(pictures)
            if row == n:
                break
            cell = cols[image][i]
            raw = cell.get("bytes") if isinstance(cell, dict) else cell
            if not raw:
                raise ValueError(f"row {row} of {path} has no picture bytes")
            with Image.open(io.BytesIO(raw)) as img:
                pictures.append(img.convert("RGB"))
            text = cols[caption][i]
            if isinstance(text, (list, tuple)):
                text = text[0] if text else None
            text = "" if text is None else str(text)
            if not text.strip():
                raise ValueError(f"row {row} of {path} has an empty caption")
            captions.append(text)
            meta.append({k: cols[k][i] for k in kept})
        if len(pictures) == n:
            break
    if len(pictures) != n:
        raise ValueError(f"{len(pictures)} rows read from {path}, {n} asked for")
    return pictures, captions, meta


# ------------------------------------------------------------------ random batches for tests
class SyntheticBatches:
    """Random batches for tests, as a batch source: ``source(step, batch)`` gives random latents (x0 float32
    [B, channels, size, size], unit normal) and random ids (long [B, 77] in [0, vocab)) drawn from a generator seeded
    by (seed, step), so any batch can be asked for in any order. ``take(n, batch)`` yields batches 0 .. n - 1."""

    def __init__(self, size: int = 16, channels: int = 4, vocab: int = 1000, seed: int = 0,
                 device: Union[str, torch.device] = "cpu"):
        self.size, self.channels, self.vocab, self.seed = int(size), int(channels), int(vocab), int(seed)
        self.device = torch.device(device)
        self.latent_shape = (self.channels, self.size, self.size)

    def __call__(self, step: int, batch: int) -> Tuple[Tensor, Tensor]:
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + int(step))
        x0 = torch.randn((int(batch),) + self.latent_shape, generator=g)
        ids = torch.randint(0, self.vocab, (int(batch), TOKENS), generator=g)
        return x0.to(self.device), ids.to(self.device)

    def take(self, n: int, batch: int) -> Iterator[Tuple[Tensor, Tensor]]:
        for step in range(int(n)):
            yield self(step, batch)


def synthetic_batches(size: int = 16, channels: int = 4, vocab: int = 1000, seed: int = 0,
                      device: Union[str, torch.device] = "cpu") -> SyntheticBatches:
    """A source of random batches for tests (``SyntheticBatches``)."""
    return SyntheticBatches(size, channels, vocab, seed, device)


def synthetic_shards(folder: str, rows: int, shard_size: int, size: int = 16, channels: int = 4, vocab: int = 1000,
                     seed: int = 0) -> dict:
    """A folder of shards of random rows (for tests of ``LatentBatches``); returns its index."""
    os.makedirs(folder, exist_ok=True)
    g = torch.Generator().manual_seed(int(seed))
    index = {"shard_size": int(shard_size), "size": int(size) * 8, "scale": LATENT_SCALE, "latent": "synthetic",
             "seed": int(seed), "rows": int(rows), "shards": [], "counts": []}
    for s, first in enumerate(range(0, int(rows), int(shard_size))):
        n = min(int(shard_size), int(rows) - first)
        lat = torch.randn(n, int(channels), int(size), int(size), generator=g)
        ids = torch.randint(0, int(vocab), (n, TOKENS), generator=g)
        write_shard(os.path.join(folder, shard_name(s)), lat, ids)
        index["shards"].append(shard_name(s))
        index["counts"].append(n)
    _write_index(folder, index)
    return index
