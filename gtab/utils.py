"""Genel yardımcılar."""

import numpy as np
import torch


def set_seed(seed=1):
    """Tekrarlanabilirlik: numpy + torch (CPU ve CUDA)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"
