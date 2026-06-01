"""Efficient random-access dataset for pre-encoded FLUX2 latents."""

from __future__ import annotations

import json
import os
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import torch
from safetensors import safe_open
from torch.utils.data import Dataset

_INDEX_FILENAME = "shard_index.json"
_SAFE_HANDLES: Dict[str, "safe_open"] = {}


def _get_handle(path: str):
    handle = _SAFE_HANDLES.get(path)
    if handle is None:
        handle = safe_open(path, framework="pt", device="cpu")
        _SAFE_HANDLES[path] = handle
    return handle


def _read_shard_size(path: str) -> int:
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_slice("labels").get_shape()[0]


def build_shard_index(data_dir: str, force: bool = False) -> List[Tuple[str, int]]:
    cache_path = os.path.join(data_dir, _INDEX_FILENAME)
    shard_paths = sorted(glob(os.path.join(data_dir, "latents_*.safetensors")))
    if not shard_paths:
        raise FileNotFoundError(f"No latent shards found under {data_dir}")

    if not force and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)
            cached_paths = [os.path.join(data_dir, name) for name, _ in cached]
            if cached_paths == shard_paths:
                return [(p, int(n)) for p, (_, n) in zip(shard_paths, cached)]
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    index: List[Tuple[str, int]] = []
    for path in shard_paths:
        index.append((path, _read_shard_size(path)))

    try:
        with open(cache_path, "w") as f:
            json.dump([(os.path.basename(p), n) for p, n in index], f)
    except OSError:
        pass
    return index


class Flux2LatentDataset(Dataset):
    def __init__(self, data_dir: str, use_flip: bool = True) -> None:
        self.data_dir = data_dir
        self.use_flip = use_flip
        self._shards: List[Tuple[str, int]] = build_shard_index(data_dir)
        self._cum_sizes = np.zeros(len(self._shards) + 1, dtype=np.int64)
        for i, (_, n) in enumerate(self._shards):
            self._cum_sizes[i + 1] = self._cum_sizes[i] + n
        self._length = int(self._cum_sizes[-1])

    def __len__(self) -> int:
        return self._length

    def _locate(self, idx: int) -> Tuple[str, int]:
        if idx < 0 or idx >= self._length:
            raise IndexError(idx)
        shard_idx = int(np.searchsorted(self._cum_sizes, idx, side="right") - 1)
        local_idx = idx - int(self._cum_sizes[shard_idx])
        return self._shards[shard_idx][0], local_idx

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        shard_path, local_idx = self._locate(idx)
        handle = _get_handle(shard_path)
        key = "latents_flip" if (self.use_flip and torch.rand(()).item() > 0.5) else "latents"
        latent = handle.get_slice(key)[local_idx : local_idx + 1].squeeze(0).to(torch.float32)
        label = handle.get_slice("labels")[local_idx : local_idx + 1].squeeze(0).to(torch.long)
        return latent, label
