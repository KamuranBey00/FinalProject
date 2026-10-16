"""
Katman 5 Adım 3 — Tap modu: kullanıcının dinlerken her vuruşa dokunduğu zamanlar -> vuruşlar + ölçü başları.

İnsan dokunuşu gürültülü (± 20–40 ms), sabit bir gecikme taşır, ara sıra kaçar ya da fazla düşer. Ölçü başları
"her M vuruşta bir" sayılırsa tek bir sayım hatası sonraki tüm ölçüleri kaydırır (ölçüldü: fermatada körlemesine boşluk
doldurma kusursuz dokunuşta bile 5190 vuruşa 23 vuruş ekledi). Adımlar:
  1. gecikme: dokunuşlar δ kaydırılır, δ in [-max_lag, max_lag] vuruş aktivasyonunun (BeatNet p_beat + p_downbeat)
     dokunuşlardaki toplamını en büyüten; max_lag = min(300 ms, 0.4 x medyan dokunuş aralığı) (Bluetooth kulaklık
     150–250 ms gecikir; aralığın yarısına yaklaşan arama komşu vuruşa kayar)
  (aktivasyon tepesine yapıştırma denendi: her koşulda zararlı, gaps_val; README17)
  3. fazla dokunuş: yerel vuruş aralığının yarısından yakın olan atılır
  4. kaçan dokunuş: yerel aralığın 1.5 katından uzun boşluğa, YALNIZ ara noktada aktivasyon >= fill_thr ise eşit aralıklı
     vuruş eklenir (fermata = tutulan akor, yeni nota yok -> eklenmez)
  5. ölçü fazı (meter_phase): durum = vuruşun ölçüdeki yeri; geçiş +1 (normal) | +2 (kaçan vuruş: araya eklenir) |
     +0 (fazla vuruş: silinir), gözlem = vuruştaki ölçü başı payı p_db / (p_beat + p_db); ilk dokunuş ölçü başı
     (kullanıcıya: "ilk ölçü başından başlayarak her vuruşa dokun"). -> düzeltilmiş vuruşlar + ölçü başları
"""

import numpy as np

from gtab.config import FRAME_RATE


def load_taps(path):
    """tap_recorder.html çıktısı -> dokunuş zamanları (sn); satır, boşluk ya da virgülle ayrılmış."""
    with open(path) as f:
        return np.array(f.read().replace(",", " ").split(), float)


def _local_ibi(t):
    """Her aralık için yerel vuruş aralığı: komşu 5 aralığın medyanı."""
    d = np.diff(t)
    return np.array([np.median(d[max(0, i - 2):i + 3]) for i in range(len(d))])


def _peak(x, t, w=2, fps=FRAME_RATE):
    """x (T,) dizisinin t (sn) çevresindeki (± w kare) en büyük değeri."""
    f = np.clip(np.round(np.asarray(t) * fps).astype(int), 0, len(x) - 1)
    return np.array([x[max(0, i - w):i + w + 1].max() for i in np.atleast_1d(f)])


def tap_lag(taps, act, max_lag=0.15, fps=FRAME_RATE):
    """Dokunuşlara eklenecek sabit düzeltme δ (sn; adım 1): vuruş aktivasyonunun dokunuşlardaki toplamını en büyüten."""
    b = np.asarray(act)[:, 1] + np.asarray(act)[:, 2]
    lags = np.arange(-int(max_lag * fps), int(max_lag * fps) + 1)
    f = np.round(np.asarray(taps, float) * fps).astype(int)
    return lags[int(np.argmax([b[np.clip(f + k, 0, len(b) - 1)].sum() for k in lags]))] / fps


def taps_to_beats(taps, act, max_lag=0.3, fill_thr=0.05, fps=FRAME_RATE):
    """taps (sn), act (T, 3) BeatNet -> vuruşlar (sn), sıralı (adım 1, 3, 4; fill_thr gaps_val'de seçildi)."""
    t = np.sort(np.asarray(taps, float))
    if len(t) < 3:
        return t
    b = np.asarray(act)[:, 1] + np.asarray(act)[:, 2]
    t = t + tap_lag(t, act, min(max_lag, 0.4 * np.median(np.diff(t))), fps)
    t = t[np.r_[True, np.diff(t) >= 0.5 * _local_ibi(t)]]
    if len(t) < 3:
        return t
    out = [t[0]]
    for a, c, g in zip(t[:-1], t[1:], _local_ibi(t)):
        n = int(round((c - a) / g))
        if n >= 2 and (c - a) > 1.5 * g:
            mid = a + (c - a) * np.arange(1, n) / n
            out.extend(mid[_peak(b, mid) >= fill_thr])
        out.append(c)
    return np.array(out)


def meter_phase(beats, act, M, eps=0.003, fps=FRAME_RATE):
    """
    Vuruşlar + aktivasyon + ölçü (vuruş / ölçü) -> (düzeltilmiş vuruşlar, ölçü başları). Viterbi (adım 5).
    """
    beats = np.asarray(beats, float)
    if M < 2 or len(beats) < 2:
        return beats, beats[::max(M, 1)]
    act = np.asarray(act)
    pb, pd = _peak(act[:, 1] + act[:, 2], beats), _peak(act[:, 2], beats)
    r = np.clip(pd / np.maximum(pb, 1e-6), 1e-3, 1 - 1e-3)
    emit = np.where(np.arange(M)[None] == 0, np.log(r)[:, None], np.log(1 - r)[:, None])     # (N, M)
    moves = ((1, np.log(1 - 2 * eps)), (2, np.log(eps)), (0, np.log(eps)))
    N = len(beats)
    delta = np.full(M, -np.inf); delta[0] = emit[0, 0]
    ptr = np.zeros((N, M), int); mv = np.zeros((N, M), int)
    for i in range(1, N):
        cand = np.full((3, M), -np.inf)
        for j, (k, lp) in enumerate(moves):
            cand[j] = np.roll(delta, k) + lp                       # durum s'ye (s - k) mod M'den gelinir
        mv[i] = cand.argmax(0)
        ptr[i] = (np.arange(M) - np.array([k for k, _ in moves])[mv[i]]) % M
        delta = cand.max(0) + emit[i]
    s = int(delta.argmax())
    path, kind = np.empty(N, int), np.empty(N, int)
    for i in range(N - 1, -1, -1):
        path[i], kind[i] = s, moves[mv[i, s]][0] if i else 1
        s = ptr[i, s] if i else s
    out, pos = [], []
    for i in range(N):
        if i and kind[i] == 0:                                     # fazla vuruş: sil
            continue
        if i and kind[i] == 2:                                     # kaçan vuruş: araya ekle
            out.append((beats[i - 1] + beats[i]) / 2); pos.append((path[i] - 1) % M)
        out.append(beats[i]); pos.append(path[i])
    out, pos = np.array(out), np.array(pos)
    return out, out[pos == 0]
