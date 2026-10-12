"""
Katman 5 — Ritim: vuruş takibi, ölçü başı + ölçü tahmini, kuantalama (eğitimsiz taban; Adım 0).

- envelope_cqt   : CQT (dB) -> spektral akı başlangıç zarfı (sesin kendisi)
- envelope_notes : nota başlangıçları -> darbe dizisi, Gauss ile yumuşatılmış (sembolik kanıt)
  (model zarfı = perde-onset kafasının perdeler üstü toplamı; çağıran hesaplar)
- track_beats    : zarf -> vuruş zamanları. 'dp' = librosa dinamik programlama (sabit tempo varsayımı,
                   'tightness' sapmaya izin), 'plp' = baskın yerel nabız (tempo değişebilir)
- downbeats      : vuruşlar + notalar -> (ölçü başları, ölçüdeki vuruş sayısı); vurgu = başlayan nota sayısı +
                   pes nota ağırlığı; ölçü (2/3/4) ve faz vurgu farkını en çok artıran
- quantize       : başlangıçlar + vuruşlar -> vuruş biriminde konum (k + m/d). Her vuruşta bölüm d, hata +
                   karmaşıklık cezası + çakışma cezasıyla seçilir (ikili ve üçlü bölümler aynı aramada).
"""

import numpy as np

from gtab.config import FRAME_RATE, HOP_LENGTH, SAMPLE_RATE

GRIDS = (1, 2, 3, 4, 6, 8, 12)                       # vuruş içi bölüm sayıları (12 = bileşik vuruşta 32'lik / üçlemeli 16'lık)
COMPLEXITY = {1: 0, 2: 1, 3: 2, 4: 2, 6: 3, 8: 3, 12: 4}


# ----------------------------------------------------------------- başlangıç zarfları
def envelope_cqt(cqt_db):
    """(T, n_bins) dB -> (T,) spektral akı (CQT zaten ortalanmış kareler: center=False, kayma yok)."""
    import librosa
    return librosa.onset.onset_strength(S=np.asarray(cqt_db, np.float32).T, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                        center=False)


def envelope_notes(onsets, n_frames, sigma=1.0):
    """Nota başlangıçları (sn) -> (T,) darbe dizisi (aynı karedeki notalar toplanır), Gauss yumuşatma."""
    from scipy.ndimage import gaussian_filter1d
    env = np.zeros(n_frames, np.float32)
    k = np.clip(np.round(np.asarray(onsets, float) * FRAME_RATE).astype(int), 0, n_frames - 1)
    np.add.at(env, k, 1.0)
    return gaussian_filter1d(env, sigma) if sigma > 0 else env


# ----------------------------------------------------------------- vuruş takibi
def _trim(frames, env):
    """librosa'nın uç kırpması (dp'de yerleşik): baştaki/sondaki zayıf vuruşlar atılır; plp için aynı kural."""
    if len(frames) < 3:
        return frames
    s = np.convolve(env[frames], np.hanning(5), "same")
    thr = 0.5 * np.sqrt(np.mean(s ** 2))
    on = np.flatnonzero(s > thr)
    return frames[on[0]:on[-1] + 1] if len(on) else frames[:0]


def track_beats(env, method="dp", tightness=100.0):
    """(T,) başlangıç zarfı -> vuruş zamanları (sn)."""
    import librosa
    env = np.asarray(env, np.float64)
    if not env.any():
        return np.zeros(0)
    if method == "dp":
        _, b = librosa.beat.beat_track(onset_envelope=env, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                       tightness=tightness, units="frames")
        return np.asarray(b, float) / FRAME_RATE
    if method == "plp":
        pulse = librosa.beat.plp(onset_envelope=env, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
        return _trim(np.flatnonzero(librosa.util.localmax(pulse)), env).astype(float) / FRAME_RATE
    raise ValueError(method)


# ----------------------------------------------------------------- ölçü başı
def downbeats(beats, notes, meters=(2, 3, 4), bass_weight=1.0, tol=0.07):
    """
    beats (sn), notes (N, 3) (başlangıç, bitiş, perde) -> (ölçü başı zamanları, ölçüdeki vuruş sayısı).
    Vuruş vurgusu = ±tol içinde başlayan nota sayısı + bass_weight x pes nota sayısı (kaydın alt %20 perdesi).
    """
    beats = np.asarray(beats, float)
    if len(beats) < 2:
        return beats[:0], 0
    acc = np.zeros(len(beats))
    notes = np.asarray(notes, float).reshape(-1, 3)
    if len(notes):
        k = np.clip(np.searchsorted(beats, notes[:, 0]), 1, len(beats) - 1)
        k = np.where(np.abs(notes[:, 0] - beats[k - 1]) < np.abs(notes[:, 0] - beats[k]), k - 1, k)
        near = np.abs(notes[:, 0] - beats[k]) <= tol
        bass = notes[:, 2] <= np.percentile(notes[:, 2], 20)
        np.add.at(acc, k[near], 1.0 + bass_weight * bass[near])
    best = max(((acc[ph::m].mean() - acc.mean(), m, ph) for m in meters for ph in range(m) if len(beats) > m),
               default=(0, meters[0], 0))
    return beats[best[2]::best[1]], best[1]


# ----------------------------------------------------------------- kuantalama
def beat_position(t, beats):
    """Zaman (sn) -> vuruş biriminde sürekli konum (beats indeksi; uçlarda komşu aralıkla uzatılır)."""
    k = np.clip(np.searchsorted(beats, t, side="right") - 1, 0, len(beats) - 2)
    return k + (t - beats[k]) / (beats[k + 1] - beats[k])


def quantize(onsets, beats, lam=0.05, mu=0.1, chord_tol=0.03, grids=GRIDS):
    """
    Başlangıçlar (sn) + vuruşlar (sn) -> (N,) kuantalanmış konum (vuruş birimi, beats indeksine göre).
    Akor: ilk notadan <= chord_tol içindeki başlangıçlar tek konum alır (grubun ilk başlangıcı).
    Her vuruş k için bölüm d: sum |u - m/d| + lam * karmaşıklık(d) + mu * (aynı noktaya düşen grup sayısı) en küçük.
    """
    onsets = np.asarray(onsets, float)
    out = np.full(len(onsets), np.nan)
    beats = np.asarray(beats, float)
    if len(beats) < 2 or not len(onsets):
        return out
    o = np.argsort(onsets, kind="stable")
    t = onsets[o]
    gid = np.zeros(len(t), int)
    start = t[0]
    for i in range(1, len(t)):
        new = t[i] - start > chord_tol
        gid[i] = gid[i - 1] + new
        if new:
            start = t[i]
    first = np.r_[0, np.flatnonzero(np.diff(gid)) + 1]
    pos = beat_position(t[first], beats)
    kb = np.floor(pos).astype(int)
    q = np.empty(len(pos))
    for k in np.unique(kb):
        idx = np.flatnonzero(kb == k)
        u = pos[idx] - k
        best = None
        for d in grids:
            m = np.round(u * d)
            c = np.abs(u - m / d).sum() + lam * COMPLEXITY.get(d, 4) + mu * (len(m) - len(np.unique(m)))
            if best is None or c < best[0]:
                best = (c, m / d)
        q[idx] = k + best[1]
    out[o] = q[gid]
    return out
