"""
Kayıplar / türetilmiş olasılıklar.

Perde etiketli veri (GAPS) için perde olasılığı tel olasılıklarından türetilir:
    P(perde p) = 1 - Π_tel (1 - P_tel(fret = p - akort_tel))     (noisy-OR)
Böylece TEK bir tab modeli hem tab hem perde etiketli veriden öğrenir.
"""

import numpy as np
import torch

from gtab.core.instrument import STANDARD_6


def pitch_index(instrument=STANDARD_6):
    """(S, P) sınıf indeksi; o tel o perdeyi çalamıyorsa -1."""
    lo, hi = instrument.pitch_range()
    P = hi - lo + 1
    idx = np.full((instrument.num_strings, P), -1, np.int64)
    for s, op in enumerate(instrument.tuning):
        for p in range(P):
            f = lo + p - op
            if 0 <= f <= instrument.num_frets:
                idx[s, p] = f + 1
    return torch.from_numpy(idx)


def pitch_probs(logits, idx):
    """logits (B,L,S,C) -> (B,L,P) perde olasılığı (tel üstünde noisy-OR)."""
    probs = torch.softmax(logits, dim=-1)                       # (B,L,S,C)
    valid = (idx >= 0)                                          # (S,P)
    g = torch.gather(probs, 3, idx.clamp(min=0).unsqueeze(0).unsqueeze(0)
                     .expand(probs.shape[0], probs.shape[1], -1, -1))   # (B,L,S,P)
    g = g * valid.to(g.dtype)
    log_none = torch.log1p(-g.clamp(max=1 - 1e-6)).sum(2)       # (B,L,P)
    return 1.0 - torch.exp(log_none)


def pitch_onset_probs(logits, onset_logits, idx):
    """
    Katman 3.9: perde etiketli veri (GAPS) için perde-ONSET olasılığı.
        P(onset p) = 1 - Π_tel (1 - σ(onset_tel) · P_tel(fret = p - akort_tel))
    logits (B,L,S,C), onset_logits (B,L,S) -> (B,L,P)
    """
    probs = torch.softmax(logits, dim=-1)
    valid = (idx >= 0)
    g = torch.gather(probs, 3, idx.clamp(min=0).unsqueeze(0).unsqueeze(0)
                     .expand(probs.shape[0], probs.shape[1], -1, -1))   # (B,L,S,P)
    g = g * valid.to(g.dtype) * torch.sigmoid(onset_logits).unsqueeze(-1)
    log_none = torch.log1p(-g.clamp(max=1 - 1e-6)).sum(2)
    return 1.0 - torch.exp(log_none)
