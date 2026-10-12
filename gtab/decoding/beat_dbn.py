"""
Katman 5 Adım 1b — Bar-pointer DBN (Whiteley 2006, Krebs 2015; madmom'un DBNDownBeatTracker'ı ile aynı yapı, numpy).

Durum = (ölçüdeki vuruş b, tempo = vuruş aralığı tau kare, vuruş içi konum p = 0 .. tau - 1). Konum her karede 1 ilerler;
vuruş sonunda bir sonraki vuruşa geçilir ve tempo yalnız orada değişebilir: P(tau' | tau) ~ exp(-trans_lambda |tau'/tau - 1|)
(rubato: tempo vuruştan vuruşa esner). Ölçü (vuruş/ölçü) her olası değer için ayrı çözülür, en olası yol seçilir.
Gözlem (ağın kare başına olasılıkları p_none / p_beat / p_downbeat): vuruşun İLK KARESİ p_beat (b = 0 ise p_downbeat),
diğer kareler p_none / (obs_lambda - 1) -> vuruş başına log(obs_lambda - 1) ödülü (yarım / çift tempo dengesi).
madmom'daki tempoyla genişleyen vuruş bölgesi (tau / obs_lambda kare) burada kullanılmaz: 43 kare/sn'de yumuşak bir
tepenin tamamını yavaş tempo yolunun geniş bölgesi yutuyor ve yol 2-3 kat yavaş tempoya kilitleniyordu (sentetik test).
"""

import numpy as np

from gtab.config import FRAME_RATE


def _space(M, taus):
    """Durum dizini: (b, k, p) blokları ardışık -> (b, k, p dizileri, giriş (M, K), çıkış (M, K) indeksleri)."""
    b = np.concatenate([np.full(t, i) for i in range(M) for t in taus])
    k = np.concatenate([np.full(t, j) for _ in range(M) for j, t in enumerate(taus)])
    p = np.concatenate([np.arange(t) for _ in range(M) for t in taus])
    start = np.flatnonzero(p == 0).reshape(M, len(taus))
    return b, k, p, start, start + np.asarray(taus)[None] - 1


def _viterbi(lo, M, taus, log_a, obs_lambda):
    """lo (T, 3) log gözlem -> (en iyi log olasılık, durum yolu (T,), b, p)."""
    b, k, p, start, end = _space(M, taus)
    cls = np.where(p == 0, np.where(b == 0, 2, 1), 0)
    inner = np.flatnonzero(p > 0)
    prev_b = (np.arange(M) - 1) % M
    T, K = len(lo), len(taus)
    ptr = np.zeros((T, M, K), np.int16)
    delta = lo[0, cls].copy()
    for t in range(1, T):
        new = np.empty_like(delta)
        new[inner] = delta[inner - 1]
        cand = delta[end][prev_b][:, :, None] + log_a[None]            # (M, K_önceki, K)
        ptr[t] = cand.argmax(1)
        new[start] = cand.max(1)
        delta = new + lo[t, cls]
    s = int(delta.argmax())
    best = float(delta[s])
    path = np.empty(T, int)
    for t in range(T - 1, -1, -1):
        path[t] = s
        if t and p[s] == 0:
            s = end[prev_b[b[s]], ptr[t, b[s], k[s]]]
        elif t:
            s -= 1
    return best, path, b, p


def peak_beats(act, thr=0.3, fps=FRAME_RATE):
    """Basit taban (DBN'siz): vuruş = (p_beat + p_downbeat) tepeleri, ölçü başı = p_downbeat tepeleri (>= thr)."""
    from scipy.signal import find_peaks
    act = np.asarray(act, float)
    b, _ = find_peaks(act[:, 1] + act[:, 2], height=thr, distance=5)
    d, _ = find_peaks(act[:, 2], height=thr, distance=10)
    return b / fps, d / fps


def dbn_decode(act, meters=(1, 2, 3, 4), min_bpm=25.0, max_bpm=260.0, n_tempi=50, trans_lambda=30.0,
               obs_lambda=16, fps=FRAME_RATE):
    """
    act (T, 3): kare başına (p_none, p_beat, p_downbeat) -> (vuruşlar sn, ölçü başları sn, ölçüdeki vuruş sayısı).
    """
    act = np.clip(np.asarray(act, float), 1e-7, 1)
    if len(act) < 2:
        return np.zeros(0), np.zeros(0), 0
    lo = np.log(np.column_stack([act[:, 0] / (obs_lambda - 1), act[:, 1], act[:, 2]]))
    taus = np.unique(np.round(np.geomspace(60 * fps / max_bpm, 60 * fps / min_bpm, n_tempi)).astype(int))
    r = taus[None, :] / taus[:, None]
    log_a = -trans_lambda * np.abs(r - 1)
    log_a -= np.log(np.exp(log_a).sum(1, keepdims=True))
    best = None
    for M in meters:
        sc, path, b, p = _viterbi(lo, M, taus, log_a, obs_lambda)
        if best is None or sc > best[0]:
            best = (sc, path, b, p, M)
    _, path, b, p, M = best
    on = np.flatnonzero(p[path] == 0)
    return on / fps, on[b[path][on] == 0] / fps, M
