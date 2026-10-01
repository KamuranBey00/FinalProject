"""
Değerlendirme metrikleri (tüm script'lerin paylaştığı tek kaynak).

- prf             : tp/fp/fn -> (P, R, F1)
- onehot_pitch    : (N,S) tel sınıfı -> (N,128) perde maskesi (tel bilgisi atılır)
- tab_scores      : kare-seviye (tel, fret) token tp/fp/fn  (Katman 3+ ana metrik)
- evaluate_tab_seq: CRNN sekans loader'ı üstünde perde F1 + tab F1
- eval_pitch_frames: perde etiketli (GAPS) loader'da kare perde F1
"""

import numpy as np
import torch

from gtab.core.instrument import STANDARD_6

PITCH_AXIS = 128
PAD = -100


def prf(tp, fp, fn):
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    return p, r, 2 * p * r / (p + r + 1e-9)


def onehot_pitch(cls, tuning):
    """(N,S) sınıf -> (N,128) bool perde maskesi."""
    N, S = cls.shape
    oh = np.zeros((N, PITCH_AXIS), dtype=bool)
    for s in range(S):
        m = cls[:, s] > 0
        if m.any():
            pitches = tuning[s] + (cls[:, s][m] - 1)
            oh[np.where(m)[0], pitches] = True
    return oh


def tab_scores(pred_frames, gt_frames):
    """Kare-seviye (tel,fret) token -> (tp, fp, fn)."""
    n = min(len(pred_frames), len(gt_frames))
    p, g = pred_frames[:n], gt_frames[:n]
    ap, ag = p > 0, g > 0
    tp = int(((p == g) & ap).sum())
    return tp, int(ap.sum()) - tp, int(ag.sum()) - tp


def evaluate_tab_seq(model, loader, device, instrument=STANDARD_6):
    """(x, tab, L) sekans loader'ı -> ((pitch P,R,F1), (tab P,R,F1)); dolgu kareleri hariç."""
    model.eval()
    tuning = np.array(instrument.tuning)
    ptp = pfp = pfn = ttp = tfp = tfn = 0
    with torch.no_grad():
        for x, y, _ in loader:
            pred = model(x.to(device)).argmax(-1).cpu().numpy()   # (B,L,S)
            gt = y.numpy()
            valid = gt[:, :, 0] != PAD
            p, g = pred[valid], gt[valid]                         # (N,S)
            ap, ag = p > 0, g > 0
            tp_t = int(((p == g) & ap).sum())
            ttp += tp_t; tfp += int(ap.sum()) - tp_t; tfn += int(ag.sum()) - tp_t
            ohp, ohg = onehot_pitch(p, tuning), onehot_pitch(g, tuning)
            tp_p = int((ohp & ohg).sum())
            ptp += tp_p; pfp += int(ohp.sum()) - tp_p; pfn += int(ohg.sum()) - tp_p
    return prf(ptp, pfp, pfn), prf(ttp, tfp, tfn)


def eval_pitch_frames(model, loader, idx, device, thr=0.5):
    """(x, frame, L) perde loader'ı -> kare perde (P, R, F1). idx: losses.pitch_index()."""
    from gtab.models.losses import pitch_probs
    model.eval(); tp = fp = fn = 0
    with torch.no_grad():
        for x, f, L in loader:
            pp = pitch_probs(model(x.to(device)), idx).cpu() > thr
            mask = torch.arange(f.shape[1])[None, :] < L[:, None]
            g = f > 0.5
            pp, g = pp[mask], g[mask]
            tp += int((pp & g).sum()); fp += int((pp & ~g).sum()); fn += int((~pp & g).sum())
    return prf(tp, fp, fn)
