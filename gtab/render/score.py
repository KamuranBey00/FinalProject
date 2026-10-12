"""
Katman 5 Adım 2 — Ölçülü partisyon modeli: kuantalanmış nota konumları + ölçü başı -> ölçüler, olaylar, nota değerleri.

Konumlar vuruş biriminde (1q çıktısı; 1/24 vuruş ızgarası = TICKS). Tek ritim sesi (gitar TAB'larında yaygın): aynı konumdaki
notalar bir akor, süresi bir sonraki başlangıca kadar; son akor ölçü sonuna kadar. Ölçü çizgisini aşan süre bağla bölünür,
her parça yazılabilir nota değerlerine (noktalı, üçleme) ayrılır. İlk notadan önceki boşluk sus.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from gtab.data.gaps_score import TPQ
from gtab.data.rhythm_labels import beat_ticks_of
from gtab.models.rhythm_net import TICKS

TYPE_Q = {"whole": 4, "half": 2, "quarter": 1, "eighth": .5, "16th": .25, "32nd": .125, "64th": .0625, "128th": .03125}


@dataclass
class Piece:
    """Yazılan bir nota / akor / sus parçası (bir ölçünün içinde)."""
    tick: int                                  # ölçü içi başlangıç (TICKS / vuruş)
    dur: int                                   # süre (tick)
    type: str
    dots: int
    triplet: bool
    notes: List[int] = field(default_factory=list)    # nota indeksleri; boş = sus
    tie_start: bool = False
    tie_stop: bool = False


def meter_of_signature(sig):
    """(pay, payda) -> (ölçüdeki vuruş sayısı, vuruş uzunluğu çeyrek cinsinden); x/8 ve pay 3'ün katı = bileşik."""
    n, ticks = beat_ticks_of(sig)
    return n, ticks / TPQ


def note_values(beat_q):
    """Vuruş uzunluğu (çeyrek) -> [(tick, tip, nokta, üçleme)] büyükten küçüğe (tam sayı tick olanlar)."""
    out = set()
    for t, q in TYPE_Q.items():
        for dots, f in ((0, 1), (1, 1.5), (2, 1.75)):
            for trip, g in ((False, 1), (True, 2 / 3)):
                x = q * f * g / beat_q * TICKS
                if (dots == 0 or not trip) and abs(x - round(x)) < 1e-9 and x >= 1:
                    out.add((int(round(x)), t, dots, trip))
    return sorted(out, key=lambda v: (-v[0], v[2], v[3]))


def split_duration(dur, values):
    """Süre (tick) -> yazılabilir değerler (açgözlü, büyükten küçüğe)."""
    parts = []
    for x, t, d, tr in values:
        while dur >= x:
            parts.append((x, t, d, tr)); dur -= x
    return parts


def build_measures(q, downbeat, beats_per_bar, beat_q):
    """
    q (N,) nota konumu (vuruş birimi; nan = yazılmaz), downbeat = bir ölçü başının vuruş indeksi,
    -> (ölçüler [ [Piece, ...], ... ], ilk ölçünün vuruş indeksi).
    """
    q = np.asarray(q, float)
    ok = np.flatnonzero(~np.isnan(q))
    t = np.round(q[ok] * TICKS).astype(int)
    starts = np.unique(t)
    if not len(starts):
        return [], 0
    L = beats_per_bar * TICKS
    t0 = int(round(downbeat * TICKS))
    m0 = (starts[0] - t0) // L
    base = t0 + m0 * L                                       # ilk ölçünün başı (tick)
    ends = np.r_[starts[1:], base + ((starts[-1] - base) // L + 1) * L]
    n_meas = (ends[-1] - base) // L
    meas = [[] for _ in range(n_meas)]
    values = note_values(beat_q)
    spans = ([(base, starts[0], [])] if starts[0] > base else []) + \
            [(a, b, ok[t == a].tolist()) for a, b in zip(starts, ends)]
    for a, b, idx in spans:
        segs = []
        x = a
        while x < b:                                         # ölçü çizgilerinde böl
            y = min(b, base + ((x - base) // L + 1) * L)
            for v in split_duration(y - x, values):
                segs.append((x, v)); x += v[0]
        for k, (x, (d, typ, dots, trip)) in enumerate(segs):
            m = (x - base) // L
            tie = bool(idx)
            meas[m].append(Piece(int(x - base - m * L), int(d), typ, dots, trip, idx,
                                 tie and k < len(segs) - 1, tie and k > 0))
    return meas, base / TICKS
