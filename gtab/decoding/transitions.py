"""
Katman 3.6b — ÖĞRENİLEN geçiş modeli (tel ataması için müzikal önsel).

NEDEN: Katman 3.6'da elle yazılmış el-hareketi maliyeti (fret mesafesi + tel
atlama + açık tel indirimi) gerçek val setinde katkı vermedi; tarama
w_transition=0'ı seçti. Muhtemel sebep: maliyetin BİÇİMİ gitaristlerin gerçek
alışkanlığına uymuyor (ör. solo çalarken pozisyonda kalıp tel değiştirmek,
belirli tel/perde bölgelerini tercih etmek).

ÇÖZÜM: Kuralı elle yazmak yerine GERÇEK TABLARDAN ÖĞREN.
    P(şimdiki pozisyon | önceki pozisyon, şimdiki perde)
şimdiki perdenin aday pozisyonları üstünde bir softmax (koşullu log-lineer
model). Özellikler yorumlanabilir kalır (Δfret, Δtel, açık tel, pozisyon
bölgesi, kutu içinde kalma, uzun boşluk) -- ağırlıklar ise veriden gelir.

- Sadece TRAIN oyuncularının etiketleriyle eğitilir (val oyuncusu 05 hariç)
  -> val ölçümünde sızıntı yok.
- Ses gerektirmez: sadece (tel, fret) dizileri. Bu yüzden ileride sembolik tab
  verisiyle (DadaGP, SynthTab etiketleri) ölçeklenebilir -- model yeniden
  eğitilmeden önsel büyütülebilir.
- Retrain GEREKTİRMEZ: akustik model çıktıları emisyon olarak kullanılır.
"""

import numpy as np
from typing import List, Optional, Sequence, Tuple

from gtab.core.instrument import Instrument, STANDARD_6
from gtab.config import FRAME_RATE
from gtab.decoding.decode import tab_to_transcription

DF_MAX = 7                                  # |Δfret| bu değere kadar ayrı ayrı öğrenilir
FRET_BUCKETS = np.array([0, 1, 4, 7, 10, 13, 16])   # pozisyon bölgeleri (alt sınırlar)
LONG_GAP_SEC = 0.5                          # bundan uzun boşlukta el rahat taşınır


# ---------------------------------------------------------------- etiket -> nota dizisi
def notes_from_tab(tab: np.ndarray, instrument: Instrument = STANDARD_6,
                   frame_rate: float = FRAME_RATE):
    """
    (T, S) gerçek tab etiketi -> zaman sıralı [(onset_sn, offset_sn, pitch, string, fret)].
    min_frames=1: etiketteki her nota korunur (kısa notalar da gerçek veridir).
    """
    tr = tab_to_transcription(tab, instrument, frame_rate, min_frames=1, smooth=1)
    return [(n.onset, n.offset, n.pitch, n.string, n.fret) for n in tr.notes]


# ---------------------------------------------------------------- model
class TransitionModel:
    """Koşullu log-lineer geçiş modeli: aday pozisyonlar üstünde softmax."""

    def __init__(self, instrument: Instrument = STANDARD_6, w: Optional[np.ndarray] = None):
        self.instrument = instrument
        S = instrument.num_strings
        # özellik düzeni (dilimler)
        self._df = slice(0, 2 * DF_MAX + 2)                  # Δfret one-hot + "uzak"
        o = self._df.stop
        self._open_any = o; self._cand_open = o + 1; o += 2
        self._ds = slice(o, o + 2 * S - 1); o = self._ds.stop  # Δtel one-hot
        self._box = o; self._far_gap = o + 1; o += 2
        self._fb = slice(o, o + len(FRET_BUCKETS)); o = self._fb.stop   # pozisyon bölgesi
        self._st = slice(o, o + S); o = self._st.stop                   # mutlak tel
        self.dim = o
        self.w = np.zeros(self.dim) if w is None else np.asarray(w, dtype=np.float64)

    # ------------------------------------------------------------ özellikler
    def features(self, prev: Optional[Tuple[int, int]], cands: Sequence[Tuple[int, int]],
                 gap_sec: float = 0.0) -> np.ndarray:
        """prev=None -> ilk nota (sadece mutlak pozisyon özellikleri). -> (K, dim)"""
        S = self.instrument.num_strings
        K = len(cands)
        X = np.zeros((K, self.dim))
        c = np.asarray(cands, dtype=np.int64).reshape(K, 2)
        cs, cf = c[:, 0], c[:, 1]
        r = np.arange(K)

        X[r, self._fb.start + np.searchsorted(FRET_BUCKETS, cf, side="right") - 1] = 1.0
        X[r, self._st.start + cs] = 1.0
        X[:, self._cand_open] = (cf == 0)
        if prev is None:
            return X

        ps, pf = prev
        X[r, self._ds.start + (cs - ps) + (S - 1)] = 1.0
        fretted = (cf > 0) & (pf > 0)
        X[:, self._open_any] = ~fretted
        df = cf - pf
        idx = np.where(np.abs(df) <= DF_MAX, df + DF_MAX, 2 * DF_MAX + 1)
        X[r[fretted], self._df.start + idx[fretted]] = 1.0
        # aynı el pozisyonu ("kutu"): işaret parmağı pf civarı, serçe pf+3/+4
        X[:, self._box] = fretted & (df >= -1) & (df <= 4)
        if gap_sec > LONG_GAP_SEC:
            X[:, self._far_gap] = np.where(fretted, np.abs(df) / 12.0, 0.0)
        return X

    def logp(self, prev, cands, gap_sec=0.0) -> np.ndarray:
        """Adaylar üstünde log-softmax. -> (K,)"""
        z = self.features(prev, cands, gap_sec) @ self.w
        z = z - z.max()
        return z - np.log(np.exp(z).sum())

    # ------------------------------------------------------------ eğitim
    def _build(self, sequences):
        """Her nota bir örnek: (aday özellikleri, doğru aday). Tek adaylılar bilgi taşımaz."""
        Xs, ys = [], []
        for seq in sequences:
            prev = None; prev_off = None
            for (on, off, pitch, s, f) in seq:
                cands = self.instrument.pitch_to_positions(pitch)
                if len(cands) > 1 and (s, f) in cands:
                    gap = 0.0 if prev_off is None else max(0.0, on - prev_off)
                    X = np.zeros((6 if len(cands) <= 6 else len(cands), self.dim))
                    X[:len(cands)] = self.features(prev, cands, gap)
                    Xs.append((X, len(cands))); ys.append(cands.index((s, f)))
                prev, prev_off = (s, f), off
        if not Xs:
            raise ValueError("Egitim icin cok-adayli nota bulunamadi.")
        Kmax = max(x.shape[0] for x, _ in Xs)
        X = np.zeros((len(Xs), Kmax, self.dim))
        mask = np.zeros((len(Xs), Kmax), dtype=bool)
        for i, (x, k) in enumerate(Xs):
            X[i, :x.shape[0]] = x; mask[i, :k] = True
        return X, mask, np.asarray(ys)

    @staticmethod
    def _nll_grad(w, X, mask, y, l2):
        z = X @ w
        z = np.where(mask, z, -1e9)
        z = z - z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        n = len(y)
        nll = -np.log(p[np.arange(n), y] + 1e-12).mean() + 0.5 * l2 * (w @ w)
        p[np.arange(n), y] -= 1.0
        grad = np.einsum("nk,nkd->d", p, X) / n + l2 * w
        return nll, grad

    def fit(self, sequences, l2=1e-3, iters=600, lr=0.05, verbose=True):
        """Adam ile tam-batch; problem konveks -> tek global optimum."""
        X, mask, y = self._build(sequences)
        w = np.zeros(self.dim); m = np.zeros_like(w); v = np.zeros_like(w)
        b1, b2 = 0.9, 0.999
        for t in range(1, iters + 1):
            nll, g = self._nll_grad(w, X, mask, y, l2)
            m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
            w -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + 1e-8)
            if verbose and (t == 1 or t % 200 == 0):
                print(f"  iter {t:4d}  nll {nll:.4f}")
        self.w = w
        return self

    def nll(self, sequences, l2=0.0):
        X, mask, y = self._build(sequences)
        return self._nll_grad(self.w, X, mask, y, l2)[0]

    # ------------------------------------------------------------ kayıt
    def save(self, path):
        np.savez(path, w=self.w, tuning=np.array(self.instrument.tuning),
                 num_frets=self.instrument.num_frets)

    @classmethod
    def load(cls, path, instrument: Instrument = STANDARD_6):
        d = np.load(path)
        if tuple(d["tuning"]) != tuple(instrument.tuning):
            raise ValueError("Gecis modeli farkli bir akort icin egitilmis.")
        return cls(instrument, d["w"])


# ---------------------------------------------------------------- kafes (lattice) + Viterbi
def build_lattice(notes, probs, tm: TransitionModel, frame_rate: float = FRAME_RATE,
                  time_unit="frames", eps=1e-6):
    """
    Ağırlıklardan BAĞIMSIZ kısmı bir kez hesaplar (tarama hızlı olsun diye).
    notes: [(start, end, pitch)]; time_unit='frames' (start/end kare indeksi)
           ya da 'sec' (saniye).
    probs: (T,S,ncls) ya da None (sadece önsel).
    -> (cands_list, E_list, T_list)
       E: adaylar üstünde NORMALİZE emisyon log-olasılığı = log P(tel | perde, ses)
          (önselle aynı ölçekte; w_transition=0 -> birebir greedy).
       T: (K_prev, K_cur) öğrenilen log P(cur | prev).
    """
    to_frame = (lambda x: x) if time_unit == "frames" else (lambda x: int(round(x * frame_rate)))
    to_sec = (lambda x: x / frame_rate) if time_unit == "frames" else (lambda x: x)
    instr = tm.instrument
    cands_list, E_list, T_list = [], [], []
    prev_cands, prev_end = None, None
    for (a, b, pitch) in notes:
        cands = instr.pitch_to_positions(pitch)
        cands_list.append(cands)
        if not cands:
            E_list.append(np.zeros(0)); T_list.append(None)
            prev_cands, prev_end = None, b
            continue
        if probs is None:
            E = np.zeros(len(cands))
        else:
            fa, fb = to_frame(a), max(to_frame(a) + 1, to_frame(b))
            m = np.array([probs[fa:fb, s, f + 1].mean() if fb > fa else 0.0
                          for (s, f) in cands])
            E = np.log(m + eps) - np.log(m.sum() + eps * len(cands))
        E_list.append(E)
        if prev_cands:
            gap = max(0.0, to_sec(a) - to_sec(prev_end))
            T_list.append(np.stack([tm.logp(pc, cands, gap) for pc in prev_cands]))
        else:
            T_list.append(tm.logp(None, cands)[None, :])     # ilk nota / zincir kırığı
        prev_cands, prev_end = cands, b
    return cands_list, E_list, T_list


def decode_lattice(lattice, w_emission=1.0, w_transition=1.0):
    """Kafes üstünde Viterbi. -> her nota için (string, fret) ya da (None, None)."""
    cands_list, E_list, T_list = lattice
    n = len(cands_list)
    out = [(None, None)] * n
    i = 0
    while i < n:                       # zincir kırıklarında (aday yok) bağımsız parçalar
        if not cands_list[i]:
            i += 1; continue
        j = i
        while j < n and cands_list[j]:
            j += 1
        # parça [i, j)
        score = w_emission * E_list[i] + w_transition * T_list[i][0]
        back = []
        for k in range(i + 1, j):
            tot = score[:, None] + w_transition * T_list[k]          # (Kp, Kc)
            bp = tot.argmax(0)
            score = tot[bp, np.arange(tot.shape[1])] + w_emission * E_list[k]
            back.append(bp)
        idx = int(score.argmax())
        path = [idx]
        for bp in reversed(back):
            idx = int(bp[idx]); path.append(idx)
        path.reverse()
        for k, p in zip(range(i, j), path):
            out[k] = cands_list[k][p]
        i = j
    return out
