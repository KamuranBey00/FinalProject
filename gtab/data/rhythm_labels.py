"""
Katman 5 — Ritim etiketleri (vuruş, ölçü başı, nota başına partisyon konumu).

GAPS (klasik gitar, rubato'lu gerçek icra):
  align() her MIDI notasını partisyondaki konumuna (açılmış tick) eşler. Eşleşen notalar = (partisyon tick'i, ses
  zamanı) çapaları -> monoton zaman haritası. Vuruş / ölçü başı tick'leri ölçü imlerinden (MusicXML <time>) çıkar,
  haritayla saniyeye çevrilir. Böylece etiket icracının gerçek (rubato'lu) vuruş zamanlarını verir.
  Vuruş birimi: x/4 -> çeyrek, x/2 -> ikilik, x/8 ve beats % 3 == 0 -> noktalı çeyrek (bileşik; 3/8 = ölçüde 1 vuruş),
  diğer x/8 -> sekizlik. İlk ölçü bir sonrakinin ölçü iminden kısaysa anakruz: vuruşları ölçü SONUNA hizalanır,
  başı ölçü başı sayılmaz.
GuitarSet: JAMS 'beat_position' (metronom; position == 1 -> ölçü başı) + 'tempo'.
"""

import json
import os

import numpy as np

from gtab.data.gaps_score import TPQ, align, load_sync, time_signatures


def beat_ticks_of(sig):
    """(beats, beat_type) -> (ölçüdeki vuruş sayısı, vuruş uzunluğu tick)."""
    n, d = sig
    if d >= 8 and n % 3 == 0:
        return n // 3, 3 * TPQ * 4 // d
    return n, TPQ * 4 // d


def beat_grid(lens, sigs):
    """Açılmış ölçü uzunlukları + ölçü imleri -> (vuruş tick'leri, ölçü başı tick'leri)."""
    cum = np.concatenate([[0.0], np.cumsum(lens)])
    nominal = lambda s: s[0] * TPQ * 4 / s[1]
    # anakruz: ilk ölçü kendi iminden kısa ya da ayrı bir kısa imle (1/4, 1/8 ...) yazılmış ön vuruş
    pickup = len(lens) > 1 and (lens[0] < nominal(sigs[0]) - 1e-6 or lens[0] <= nominal(sigs[1]) / 2 + 1e-6)
    beats, downs = [], []
    for b, L in enumerate(lens):
        n, bl = beat_ticks_of(sigs[1] if b == 0 and pickup else sigs[b])
        if b == 0 and pickup:                                           # vuruşlar ölçü sonuna hizalı
            k = np.arange(int(L // bl), 0, -1)
            beats += list(cum[1] - k * bl)
            continue
        downs.append(cum[b])
        beats += list(cum[b] + bl * np.arange(int(np.ceil(L / bl - 1e-6))))
    return np.array(beats), np.array(downs)


def time_map(score_x, onsets):
    """Eşleşen notalar -> monoton çapalar (tick, sn): aynı tick'te en erken başlangıç (arpej), sırayı bozan atılır."""
    ok = ~np.isnan(score_x)
    xs, ts = score_x[ok], onsets[ok]
    ux = np.unique(xs)
    ut = np.array([ts[xs == u].min() for u in ux])
    keep, last = [], -np.inf
    for i, t in enumerate(ut):
        if t > last:
            keep.append(i); last = t
    return ux[keep], ut[keep]


def gaps_rhythm(gaps_dir, tid, tuning):
    """
    Bir GAPS kaydı -> dict(consistent, matched (oran), beats, downbeats (sn), note_beat (MIDI notası başına
    vuruş konumu, beats dizisinin indeksine göre; eşleşmeyen / ızgara dışı = nan), notes, n_anchor, sync_err).
    Vuruşlar yalnız çapaların kapsadığı aralıkta tutulur (dışı enterpolasyon olmaz).
    """
    r = align(gaps_dir, tid, tuning)
    sigs = time_signatures(os.path.join(gaps_dir, "musicxml", f"{tid}.xml"))
    sigs = [sigs[i] if i < len(sigs) else sigs[-1] for i in r["order"]]
    bt, dt = beat_grid(r["lens"], sigs)
    notes = np.array(r["notes"], float).reshape(-1, 3)
    ax, at = time_map(r["score_x"], notes[:, 0])
    out = dict(consistent=r["consistent"], matched=float(r["matched"].mean()), notes=notes, n_anchor=len(ax))
    if len(ax) < 2:
        return dict(out, beats=np.zeros(0), downbeats=np.zeros(0), note_beat=np.full(len(notes), np.nan),
                    sync_err=np.nan)
    inside = lambda x: (x >= ax[0]) & (x <= ax[-1])
    bt = bt[inside(bt)]
    dt = dt[inside(dt)]
    beats, downs = np.interp(bt, ax, at), np.interp(dt, ax, at)
    # nota konumu vuruş birimiyle: k + (x - B_k) / (B_k+1 - B_k); son vuruştan sonrası son aralıkla
    nb = np.full(len(notes), np.nan)
    if len(bt) >= 2:
        sx = r["score_x"]
        k = np.clip(np.searchsorted(bt, sx, side="right") - 1, 0, len(bt) - 2)
        val = ~np.isnan(sx)
        nb[val] = k[val] + (sx[val] - bt[k[val]]) / (bt[k[val] + 1] - bt[k[val]])
        nb[(sx < bt[0]) | (sx > bt[-1] + (bt[-1] - bt[-2]))] = np.nan
    # tutarlılık: haritadan gelen ölçü başları vs syncpoint'ler (ölçü düzeyi el hizalaması)
    sync = {int(e[0]): e[1] for e in load_sync(gaps_dir, tid) if len(e) < 3 or not e[2]}
    cum = np.concatenate([[0.0], np.cumsum(r["lens"])])
    err = [abs(np.interp(cum[b], ax, at) - t) for b, t in sync.items() if b < len(cum) and inside(cum[b])]
    return dict(out, beats=beats, downbeats=downs, note_beat=nb, sync_err=float(np.median(err)) if err else np.nan)


def guitarset_rhythm(jams_path):
    """GuitarSet JAMS -> dict(beats, downbeats (sn), tempo (BPM))."""
    with open(jams_path) as f:
        ann = json.load(f)["annotations"]
    bp = next(a["data"] for a in ann if a["namespace"] == "beat_position")
    tempo = next(a["data"][0]["value"] for a in ann if a["namespace"] == "tempo")
    beats = np.array([d["time"] for d in bp])
    downs = np.array([d["time"] for d in bp if d["value"]["position"] == 1])
    return dict(beats=beats, downbeats=downs, tempo=float(tempo))
