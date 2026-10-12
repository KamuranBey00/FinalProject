"""
Katman 3.13 Adım 2 — "Klasik edisyon önerisi": konum (tel/perde) + sol el parmağı, SEMBOLİK.

Sunumun 4. (çalınabilirlik) ve 7. (parmaklandırma) modülleri tek katmanda. Girdi nota dizisi (perde, başlangıç,
bitiş; akorlar başlangıç toleransıyla gruplanır), çıktı her nota için (tel, perde, parmak). Ses eğitimi yok:
ağırlıklar GAPS partisyonlarındaki editör kararlarından öğrenilir (koşullu log-lineer = öğrenilen, açıklanabilir
maliyet; sunum slayt 10). Ses kullanılmaz: "duyulan konum" (ses) ve "edisyon önerisi" ayrı çıktı modlarıdır
(karışım GuitarSet'te duyulan teli 0.878 -> 0.573 düşürdü; kullanıcı kararı, README14).

  P(tel, perde, parmak | önceki) = P_konum(tel, perde | önceki konum, akor içi mi)
                                   x P_parmak(parmak | tel, perde, önceki konum + parmak, akor içi mi)

- ChordPositionModel: Katman 3.6b TransitionModel'in özellikleri + akor içi Δtel / Δperde blokları
  (akor içinde ardışık notalar arasında sıralı Δ'lar yerine akor şekli öğrenilir).
- FingerModel: parmak, el pozisyonu (işaret parmağının perdesi = perde − parmak + 1) değişimi (akor içi / sıralı),
  aynı parmakla kayma, barre (aynı parmak + aynı perde), uzun boşlukta serbest taşıma.
- decode: (tel, perde, parmak) durumları üstünde Viterbi.
"""

import numpy as np

from gtab.core.instrument import STANDARD_6
from gtab.decoding.transitions import TransitionModel, DF_MAX, LONG_GAP_SEC

CHORD_TOL = 0.03                     # sn; aynı akor sayılan başlangıç farkı (ölçüldü: tel sırası %97 tutarlı)
DH_MAX = 5                           # el pozisyonu değişimi |Δh| bu değere kadar ayrı öğrenilir


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


class ChordPositionModel(TransitionModel):
    """TransitionModel + akor içi Δtel / Δperde blokları."""

    def __init__(self, instrument=STANDARD_6, w=None):
        super().__init__(instrument)
        S, o = instrument.num_strings, self.dim
        self._cds = slice(o, o + 2 * S - 1); o = self._cds.stop
        self._cdf = slice(o, o + 2 * DF_MAX + 2); o = self._cdf.stop
        self.dim = o
        self.w = np.zeros(self.dim) if w is None else np.asarray(w, dtype=np.float64)

    def features(self, prev, cands, gap_sec=0.0, chord=False):
        if not chord or prev is None:
            return super().features(prev, cands, gap_sec)
        X = super().features(None, cands)                      # yalnız mutlak özellikler
        S = self.instrument.num_strings
        c = np.asarray(cands, dtype=np.int64).reshape(-1, 2); cs, cf = c[:, 0], c[:, 1]
        r = np.arange(len(c)); ps, pf = prev
        X[r, self._cds.start + (cs - ps) + (S - 1)] = 1.0
        fretted = (cf > 0) & (pf > 0); df = cf - pf
        idx = np.where(np.abs(df) <= DF_MAX, df + DF_MAX, 2 * DF_MAX + 1)
        X[r[fretted], self._cdf.start + idx[fretted]] = 1.0
        return X

    def logp(self, prev, cands, gap_sec=0.0, chord=False):
        z = self.features(prev, cands, gap_sec, chord) @ self.w
        z = z - z.max()
        return z - np.log(np.exp(z).sum())

    def _build(self, sequences):
        """sequences: [(başlangıç, bitiş, perde, tel, perde_no, parmak, akor)]; tel < 0 = etiketsiz (atlanır)."""
        Xs, ys = [], []
        for seq in sequences:
            prev = prev_off = None
            for (on, off, pitch, s, f, _, ch) in seq:
                if s < 0:
                    prev = prev_off = None
                    continue
                cands = self.instrument.pitch_to_positions(int(pitch))
                if len(cands) > 1 and (s, f) in cands:
                    gap = 0.0 if prev_off is None else max(0.0, on - prev_off)
                    Xs.append(self.features(prev, cands, gap, bool(ch) and prev is not None))
                    ys.append(cands.index((s, f)))
                prev, prev_off = (s, f), off
        return _pack(Xs, ys)


class FingerModel(TransitionModel):
    """P(parmak | şimdiki konum, önceki konum + parmak). Boş tel -> parmak 0 (tek aday)."""

    def __init__(self, instrument=STANDARD_6, w=None):
        self.instrument = instrument
        nd = 2 * DH_MAX + 2                                     # Δh one-hot + "uzak"
        o = 5
        self._dh = slice(o, o + nd); o += nd                    # sıralı
        self._cdh = slice(o, o + nd); o += nd                   # akor içi
        self._same_shift, self._same_barre, self._csame_shift, self._csame_barre = o, o + 1, o + 2, o + 3; o += 4
        self._noprev, self._far = o, o + 1; o += 2
        self.dim = o
        self.w = np.zeros(self.dim) if w is None else np.asarray(w, dtype=np.float64)

    @staticmethod
    def cands(fret):
        return [0] if fret == 0 else [1, 2, 3, 4]

    def features(self, prev, cur, gap_sec=0.0, chord=False):
        """prev: (tel, perde, parmak) | None; cur: (tel, perde). -> (K, dim) (K = parmak adayları)"""
        s, f = cur
        g = np.array(self.cands(f)); K = len(g); r = np.arange(K)
        X = np.zeros((K, self.dim)); X[r, g] = 1.0
        if prev is None or prev[2] < 1 or f == 0:
            X[:, self._noprev] = 1.0
            return X
        ps, pf, pg = prev
        dh = (f - g + 1) - (pf - pg + 1)
        idx = np.where(np.abs(dh) <= DH_MAX, dh + DH_MAX, 2 * DH_MAX + 1)
        X[r, (self._cdh if chord else self._dh).start + idx] = 1.0
        same = g == pg
        X[:, self._csame_shift if chord else self._same_shift] = same & (f != pf)
        X[:, self._csame_barre if chord else self._same_barre] = same & (f == pf) & (s != ps)
        if not chord and gap_sec > LONG_GAP_SEC:
            X[:, self._far] = np.abs(dh) / 12.0
        return X

    def logp(self, prev, cur, gap_sec=0.0, chord=False):
        z = self.features(prev, cur, gap_sec, chord) @ self.w
        z = z - z.max()
        return z - np.log(np.exp(z).sum())

    def _build(self, sequences):
        """Parmak etiketli, basılı (perde > 0) notalar; önceki nota parmaklı değilse 'önceki yok' özelliği."""
        Xs, ys = [], []
        for seq in sequences:
            prev = prev_off = None
            for (on, off, pitch, s, f, g, ch) in seq:
                if s < 0:
                    prev = prev_off = None
                    continue
                if f > 0 and g >= 1:
                    gap = 0.0 if prev_off is None else max(0.0, on - prev_off)
                    Xs.append(self.features(prev, (s, f), gap, bool(ch) and prev is not None)); ys.append(g - 1)
                prev, prev_off = (s, f, g), off
        return _pack(Xs, ys)


def _pack(Xs, ys):
    if not Xs:
        raise ValueError("Egitim ornegi yok.")
    K = max(len(x) for x in Xs); dim = Xs[0].shape[1]
    X = np.zeros((len(Xs), K, dim), np.float32); mask = np.zeros((len(Xs), K), bool)
    for i, x in enumerate(Xs):
        X[i, :len(x)] = x; mask[i, :len(x)] = True
    return X, mask, np.asarray(ys)


def save_edition(path, pos, fing):
    np.savez(path, w_pos=pos.w, w_fing=fing.w, tuning=np.array(pos.instrument.tuning))


def load_edition(path, instrument=STANDARD_6):
    d = np.load(path)
    if tuple(d["tuning"]) != tuple(instrument.tuning):
        raise ValueError("Edisyon modeli farkli bir akort icin egitilmis.")
    return ChordPositionModel(instrument, d["w_pos"]), FingerModel(instrument, d["w_fing"])


def decode(notes, chord, pos, fing, w_pos=1.0, w_fing=1.0, fixed=None):
    """
    Sıralı notalar [(başlangıç, bitiş, perde)] + akor bayrağı -> [(tel, perde, parmak)] (aday yoksa (None,)*3).
    fixed: not başına (tel, perde) | None -> konum sabit (yalnız parmak çözülür; parmak ölçümü için).
    """
    instr = pos.instrument
    out = [(None, None, None)] * len(notes)
    states, score, back, prev_i = None, None, [], None
    segs = []                                                 # aday yoksa zincir kırılır

    def flush():
        if states is None:
            return
        k = int(np.argmax(score)); path = [k]
        for bp in reversed(back):
            k = int(bp[k]); path.append(k)
        path.reverse()
        for i, k in zip(segs, path):
            out[i] = st_all[i][k]

    st_all, cache = {}, {}
    for i, (on, off, p) in enumerate(notes):
        pc = instr.pitch_to_positions(int(p))
        if fixed is not None and fixed[i] is not None:
            pc = [fixed[i]] if fixed[i] in pc else pc
        if not pc:
            flush(); states = score = None; back, segs, prev_i = [], [], None
            continue
        cur = [(s, f, g) for (s, f) in pc for g in FingerModel.cands(f)]
        if states is None:
            lp_pos = pos.logp(None, pc)
            score = np.array([w_pos * lp_pos[pc.index((s, f))] for s, f, _ in cur])
            score += np.array([w_fing * fing.logp(None, (s, f))[FingerModel.cands(f).index(g)] for s, f, g in cur])
        else:
            gap = max(0.0, on - notes[prev_i][1]); ch = bool(chord[i])
            far = gap > LONG_GAP_SEC                         # özellikler boşluğa yalnız bu eşikle bağlı
            gap = 1.0 if far else 0.0
            tot = np.zeros((len(states), len(cur)))
            pkey = tuple(pc)
            for j, (ps, pf, pg) in enumerate(states):
                k = ((ps, pf), pkey, ch, far)
                if k not in cache:
                    lpp = pos.logp((ps, pf), pc, gap, ch)
                    cache[k] = np.array([lpp[pc.index((s, f))] for s, f, _ in cur])
                kf = ((ps, pf, pg), pkey, ch, far)
                if kf not in cache:
                    cache[kf] = np.concatenate([fing.logp((ps, pf, pg), (s, f), gap, ch) for (s, f) in pc])
                tot[j] = w_pos * cache[k] + w_fing * cache[kf]
            tot += score[:, None]
            bp = tot.argmax(0); back.append(bp)
            score = tot[bp, np.arange(len(cur))]
        states = cur; st_all[i] = cur; segs.append(i); prev_i = i
    flush()
    return out


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


def as_tuples(seq):
    """label_sequence -> ChordPositionModel / FingerModel eğitim dizisi."""
    notes, chord, s, f, g = seq
    return [(a, b, p, si, fi, gi, c) for (a, b, p), si, fi, gi, c in zip(notes, s, f, g, chord)]
