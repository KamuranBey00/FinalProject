"""
Katman 5 — Ritim metrikleri.

- beat_scores     : mir_eval vuruş F (±70 ms), CMLt (doğru metrik düzey), AMLt (yarım/çift tempo ve ters faz kabul);
                    ilk 5 sn kırpılır (mir_eval standardı). Ölçü başı F aynı fonksiyonla ölçü başı dizisinde.
- tempo_of / tempo_ok : medyan vuruş aralığından BPM; Acc1 (±%4), Acc2 (x 1/3, 1/2, 2, 3 kabul)
- position_accuracy: kuantalanmış konum = partisyon konumu (aynı vuruş numaralaması; yalnız oracle vuruşta anlamlı)
- ioi_accuracy    : ardışık iki başlangıç grubu (partisyonda aynı konum = bir grup) arasındaki süre partisyondaki
                    süreye eşit mi (vuruş biriminde). Tahmini vuruşta kayıt başına tek ölçek r (metrik düzey:
                    1/4 .. 4) seçilir; faz ve vuruş numarasından bağımsız -> "nota değerleri doğru mu?"
- chord_together  : partisyonda aynı konumdaki (>= 2 nota) grupların tek konuma kuantalanma oranı
"""

import numpy as np

SCALES = (1 / 4, 1 / 3, 1 / 2, 2 / 3, 1, 3 / 2, 2, 3, 4)
EPS = 1e-3


def beat_scores(ref, est):
    """-> (F, CMLt, AMLt)."""
    import mir_eval
    r, e = mir_eval.beat.trim_beats(np.asarray(ref, float)), mir_eval.beat.trim_beats(np.asarray(est, float))
    if len(r) < 2 or len(e) < 2:
        return 0.0, 0.0, 0.0
    _, cmlt, _, amlt = mir_eval.beat.continuity(r, e)
    return float(mir_eval.beat.f_measure(r, e)), float(cmlt), float(amlt)


def in_span(est, ref):
    """Tahmini olayları etiketli aralıkla (± yarım vuruş) sınırla (GAPS etiketi eşleşen notalarla sınırlı)."""
    if len(ref) < 2 or not len(est):
        return est
    h = np.median(np.diff(ref)) / 2
    return est[(est >= ref[0] - h) & (est <= ref[-1] + h)]


def onset_f(ref, est):
    """Ölçü başı F (aynı ±70 ms penceresi)."""
    import mir_eval
    r, e = mir_eval.beat.trim_beats(np.asarray(ref, float)), mir_eval.beat.trim_beats(np.asarray(est, float))
    return float(mir_eval.beat.f_measure(r, e)) if len(r) and len(e) else 0.0


def tempo_of(beats):
    return 60.0 / np.median(np.diff(beats)) if len(beats) >= 3 else np.nan


def tempo_ok(ref, est, tol=0.04):
    """-> (Acc1, Acc2)."""
    if not (np.isfinite(ref) and np.isfinite(est)):
        return False, False
    a1 = abs(est / ref - 1) <= tol
    return a1, any(abs(est * f / ref - 1) <= tol for f in (1 / 3, 1 / 2, 1, 2, 3))


def _groups(s, onsets, valid):
    """Partisyon konumu gruplarının sıralı konumları + her grubun temsilcisi (icrada en erken başlayan nota)."""
    sr = np.round(s, 6)
    us = np.unique(sr[valid])
    rep = []
    for u in us:
        idx = np.flatnonzero(valid & (sr == u))
        rep.append(idx[np.argmin(onsets[idx])])
    return us, np.array(rep, int)


def position_accuracy(q, s):
    """-> (doğru, toplam) not başına."""
    v = ~np.isnan(q) & ~np.isnan(s)
    return int((np.abs(q[v] - s[v]) < EPS).sum()), int(v.sum())


def ioi_detail(q, s, onsets, scales=SCALES):
    """-> (doğru maskesi, partisyon IOI'leri ds, tahmini IOI'ler r * dq, seçilen ölçek r)."""
    q, s, onsets = (np.asarray(a, float) for a in (q, s, onsets))
    v = ~np.isnan(q) & ~np.isnan(s)
    us, rep = _groups(s, onsets, v)
    if len(us) < 2:
        return np.zeros(0, bool), np.zeros(0), np.zeros(0), 1.0
    ds, dq = np.diff(us), np.diff(q[rep])
    best = max(scales, key=lambda r: ((np.abs(r * dq - ds) < EPS).sum(), -abs(np.log(r))))
    return np.abs(best * dq - ds) < EPS, ds, best * dq, best


def ioi_accuracy(q, s, onsets, scales=SCALES):
    """-> (doğru, toplam, seçilen ölçek r). scales=(1,) -> ölçek sabit (oracle vuruş)."""
    ok, ds, _, r = ioi_detail(q, s, onsets, scales)
    return int(ok.sum()), len(ds), r


def chord_together(q, s):
    """-> (tek konuma düşen grup, >= 2 notalı grup)."""
    v = ~np.isnan(q) & ~np.isnan(s)
    sr = np.round(s, 6)
    ok = n = 0
    for u in np.unique(sr[v]):
        idx = np.flatnonzero(v & (sr == u))
        if len(idx) >= 2:
            n += 1; ok += np.ptp(q[idx]) < EPS
    return int(ok), n


def grid_coverage(s, grids):
    """Partisyon konumlarının ızgara sözlüğüyle gösterilebilen oranı (kuantalayıcının erişebileceği tavan)."""
    f = np.asarray(s, float)[~np.isnan(s)] % 1
    ok = np.zeros(len(f), bool)
    for d in grids:
        ok |= np.abs(f * d - np.round(f * d)) < EPS * d
    return float(ok.mean()) if len(f) else np.nan
