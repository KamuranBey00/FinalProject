"""
Katman 3.13 — "Klasik edisyon önerisi" için nota dizisi yardımcıları (sembolik; ses yok).

Model: gtab/models/edition_net.py (iki yönlü dizi modeli, checkpoints/edition_seq.pt).
Reddedilen log-lineer aday (Adım 2a) arşivde: archive/gtab/decoding/edition_loglinear.py.

- order_notes   : notalar -> sıra + akor bayrağı (akor = başlangıçları ≤ 30 ms, akor içinde perde artan)
- label_sequence: build_gaps_tab etiket dosyası -> sıralı notalar + tel / perde / parmak
- finger_baseline: taban parmak kuralı "parmak = perde − pozisyon + 1" (ölçüm karşılaştırması)
"""

import numpy as np

CHORD_TOL = 0.03                     # sn; aynı akor sayılan başlangıç farkı (ölçüldü: tel sırası %97 tutarlı)


def order_notes(notes, tol=CHORD_TOL):
    """[(başlangıç, bitiş, perde)] -> (sıra indeksleri, akor bayrağı): akor (ilk notadan <= tol) içinde perde artan."""
    notes = np.asarray(notes, float).reshape(-1, 3)
    o = np.argsort(notes[:, 0], kind="stable")
    grp = np.zeros(len(o), int); start = notes[o[0], 0] if len(o) else 0.0
    for k in range(1, len(o)):
        grp[k] = grp[k - 1] + (notes[o[k], 0] - start > tol)
        if grp[k] != grp[k - 1]:
            start = notes[o[k], 0]
    o = o[np.lexsort((notes[o, 2], grp))]
    grp = np.sort(grp)
    chord = np.r_[False, grp[1:] == grp[:-1]]
    return o, chord


def finger_baseline(notes, positions, window=3):
    """
    Taban: parmak = perde − pozisyon + 1; pozisyon = komşu notalarda (±window, aynı el kutusu: perde−3..perde)
    en düşük basılı perde. Boş tel -> 0.
    """
    out = []
    for i, sf in enumerate(positions):
        if sf is None or sf[0] is None:
            out.append(None); continue
        f = sf[1]
        if f == 0:
            out.append(0); continue
        near = [q[1] for q in positions[max(0, i - window):i + window + 1]
                if q is not None and q[0] is not None and 1 <= q[1] and f - 3 <= q[1] <= f]
        out.append(int(np.clip(f - min(near) + 1, 1, 4)))
    return out


def label_sequence(path):
    """build_gaps_tab etiket dosyası -> sıralı (notes (N,3), akor (N,), tel, perde, parmak) (etiketsiz = -1)."""
    with np.load(path) as L:
        notes, s, f, g = L["notes"].astype(float), L["string"], L["fret"], L["finger_note"]
    o, chord = order_notes(notes)
    return notes[o], chord, s[o].astype(int), f[o].astype(int), g[o].astype(int)
