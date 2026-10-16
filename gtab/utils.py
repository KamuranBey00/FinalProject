"""Genel yardımcılar."""

from pathlib import Path

import numpy as np
import torch


def set_seed(seed=1):
    """Tekrarlanabilirlik: numpy + torch (CPU ve CUDA)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def guard_new_checkpoint(path: str, overwrite: bool = False) -> None:
    """
    Stop before training if the output checkpoint already exists.

    Training scripts save at epoch 0, so a forgotten --ckpt/--out silently replaced an older
    model (e.g. train_onset default 'tabcrnn_onset.pt', train_edition_seq default = the live
    'edition_seq.pt'). The project rule is "never overwrite, add a new file".
    """
    if Path(path).exists() and not overwrite:
        raise SystemExit(f"HATA: cikti checkpoint'i zaten var: {path}\n"
                         f"  Yeni bir ad verin (--ckpt / --out) ya da bilerek ezmek icin --overwrite ekleyin.")
