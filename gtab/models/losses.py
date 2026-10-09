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


def tab_pitch_target(y, idx):
    """
    Katman 3.10 Adım 2 — tab etiketi (B,L,S) (0 = sessiz, -100 = dolgu) -> (B,L,P) perde hedefi.
    GuitarSet'e de GAPS'teki gibi doğrudan perde (noisy-OR) kaybı uygulamak için.
    """
    hit = (y.unsqueeze(-1) == idx.view(1, 1, *idx.shape)) & (idx >= 0).view(1, 1, *idx.shape)
    return hit.any(2).float()


def _rising(on):
    """(B,L,K) onset hedefi (keskin ya da genişletilmiş) -> onset'in İLK karesi (bool)."""
    hard = on >= 0.999
    prev = torch.zeros_like(hard); prev[:, 1:] = hard[:, :-1]
    return hard & ~prev


def _offset_from(cur_on, same_next, onset, soft):
    """Ortak: nota bu karede biter = aktif & (sonraki kare aynı nota değil ya da yeni onset)."""
    nxt_on = torch.zeros_like(cur_on); nxt_on[:, :-1] = _rising(onset)[:, 1:]
    end = cur_on & (~same_next | nxt_on)
    tgt = end.float()
    if soft is not None:                       # onset ile simetrik: bitişten önceki kare 'soft'
        prv = torch.zeros_like(tgt); prv[:, :-1] = tgt[:, 1:] * float(soft)
        tgt = torch.maximum(tgt, prv * cur_on.float())
    mask = torch.ones(tgt.shape[:2], device=tgt.device)
    mask[:, -1] = 0                            # chunk'ın son karesi: devamı bilinmiyor
    return tgt, mask


def string_offset_target(y, onset, soft=None):
    """
    Katman 3.10 Adım 3c — tab etiketi (B,L,S) + tel onset hedefi (B,L,S) -> (B,L,S) OFFSET hedefi
    (notanın son karesi = 1) ve (B,L) kare maskesi.
    """
    cur = y > 0
    same = torch.zeros_like(cur); same[:, :-1] = y[:, 1:] == y[:, :-1]
    return _offset_from(cur, same, onset, soft)


def pitch_offset_target(frame, onset, soft=None):
    """Katman 3.10 Adım 3c — perde roll'u (B,L,P) + onset hedefi -> (B,L,P) offset hedefi, (B,L) maske."""
    cur = frame > 0.5
    same = torch.zeros_like(cur); same[:, :-1] = cur[:, 1:]
    return _offset_from(cur, same, onset, soft)


def tab_pitch_onset_target(y, onset, idx):
    """
    Katman 3.12 — tab etiketi (B,L,S) + tel onset hedefi (B,L,S; keskin/soft) -> (B,L,P) perde-onset hedefi:
    her perde için o perdeyi çalan tellerin onset hedeflerinin en büyüğü (perde-onset kafası, GuitarSet).
    """
    hit = (y.unsqueeze(-1) == idx.view(1, 1, *idx.shape)) & (idx >= 0).view(1, 1, *idx.shape)
    return (hit.float() * onset.unsqueeze(-1)).amax(2)


def tab_pitch_weight(y, w, idx):
    """(B,L,S) tel ağırlığı -> (B,L,P) perde ağırlığı (o perdeyi çalan tellerin en büyüğü; çalınmayan perde 1)."""
    hit = (y.unsqueeze(-1) == idx.view(1, 1, *idx.shape)) & (idx >= 0).view(1, 1, *idx.shape)
    return torch.clamp((hit.float() * w.unsqueeze(-1)).amax(2), min=1.0)
