"""
PyTorch veri okuyucular. Önbellekteki .npz'lerden CQT pencereleri üretir.
- GuitarSetFrames (Katman 2): pencere -> perde vektörü (multi-label)
- GuitarSetTab    (Katman 3): pencere -> (6,) tel başına fret sınıfı (int)
- GuitarSetSeq    (Katman 3.5+): sekans -> (L, 6) tel sınıfı; split listesi + çoğaltma
- PitchSeq        (Katman 3.8): sekans -> (L, P) perde roll'u (GAPS)

Normalizasyon: dB [-80,0] -> [0,1], deterministik.
"""

import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset


def _normalize(cqt: np.ndarray) -> np.ndarray:
    return np.clip((cqt + 80.0) / 80.0, 0.0, 1.0).astype(np.float32)


def _window(cqt, t, context):
    half = context // 2
    T, n_bins = cqt.shape
    lo, hi = t - half, t + half + 1
    w = np.zeros((context, n_bins), dtype=np.float32)
    a, b = max(0, lo), min(T, hi)
    w[a - lo:b - lo] = cqt[a:b]
    return w


class GuitarSetFrames(Dataset):
    """Katman 2: pencere -> perde multi-hot vektörü."""
    def __init__(self, cache_dir, split, context=9):
        assert context % 2 == 1
        self.context = context
        self.tracks, self.index = [], []
        paths = sorted(glob.glob(os.path.join(cache_dir, split, "*.npz")))
        if not paths:
            raise FileNotFoundError(f"{cache_dir}/{split} bos. build_dataset.py calisti mi?")
        for path in paths:
            d = np.load(path)
            cqt = _normalize(d["cqt"]); frame = d["frame"].astype(np.float32)
            ti = len(self.tracks); self.tracks.append((cqt, frame))
            self.index.extend((ti, t) for t in range(cqt.shape[0]))

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        ti, t = self.index[i]
        cqt, frame = self.tracks[ti]
        x = torch.from_numpy(_window(cqt, t, self.context)).unsqueeze(0)
        return x, torch.from_numpy(frame[t])

    def pos_weight(self):
        pos = np.zeros(self.tracks[0][1].shape[1], np.float64); total = 0
        for _, frame in self.tracks:
            pos += frame.sum(0); total += frame.shape[0]
        neg = total - pos
        return torch.tensor(np.where(pos > 0, neg / np.maximum(pos, 1), 1.0), dtype=torch.float32)


class GuitarSetTab(Dataset):
    """Katman 3: pencere -> (num_strings,) fret sınıfı (int). 'tab' etiketi gerekir."""
    def __init__(self, cache_dir, split, context=9):
        assert context % 2 == 1
        self.context = context
        self.tracks, self.index = [], []
        paths = sorted(glob.glob(os.path.join(cache_dir, split, "*.npz")))
        if not paths:
            raise FileNotFoundError(f"{cache_dir}/{split} bos.")
        for path in paths:
            d = np.load(path)
            if "tab" not in d:
                raise KeyError(f"{path} icinde 'tab' yok. build_tab_labels.py calisti mi?")
            cqt = _normalize(d["cqt"]); tab = d["tab"].astype(np.int64)
            ti = len(self.tracks); self.tracks.append((cqt, tab))
            self.index.extend((ti, t) for t in range(cqt.shape[0]))

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        ti, t = self.index[i]
        cqt, tab = self.tracks[ti]
        x = torch.from_numpy(_window(cqt, t, self.context)).unsqueeze(0)
        return x, torch.from_numpy(tab[t])          # (num_strings,) long

    def class_weights(self, n_classes, scheme="sqrt", cap=5.0):
        """
        Sessiz sınıf (0) baskın; sınıf ağırlığıyla dengeleriz.
        scheme='inv'  -> ters frekans  (eski/agresif: recall'u sisirir, precision duser)
        scheme='sqrt' -> ters-KAREKOK frekans (yumusak, DENGELI: onerilen)
        scheme='none' -> tek agirlik
        cap: uc degerleri [1/cap, cap] araligina kirpar. Kucuk cap = daha dengeli.
        """
        counts = np.zeros(n_classes, np.float64)
        for _, tab in self.tracks:
            counts += np.bincount(tab.reshape(-1), minlength=n_classes)
        counts = np.maximum(counts, 1.0)
        # inv = counts.sum()/(n_classes*counts): sabit ölçekli ters frekans (çalışan formül).
        # DİKKAT: mean'e BÖLME. Kullanılmayan fret sınıfları (counts=1) devasa değer
        # alır; mean'e bölersen gerçek ağırlıkları sıfıra çeker -> sessizliğe çökme.
        # Kullanılmayan sınıflar hiçbir örnekte "doğru sınıf" olmadığından ağırlıkları
        # zaten hiç kullanılmaz; dokunmamıza gerek yok.
        inv = counts.sum() / (n_classes * counts)
        if scheme == "inv":
            w = inv
        elif scheme == "sqrt":        # daha YUMUŞAK oran (precision lehine), çökme yok
            w = np.sqrt(inv)
        else:                         # "none"
            w = np.ones(n_classes)
        return torch.tensor(np.clip(w, 1.0 / cap, cap), dtype=torch.float32)


class GuitarSetSeq(Dataset):
    """
    Katman 3.5 (CRNN): pencere değil, SEKANS verir.
    Her track sabit uzunlukta (chunk) parçalara bölünür; son parça sıfırla
    doldurulur ve etiketi -100 ile maskelenir (kayıpta yok sayılır).
    Çıktı: (1, L, n_bins) cqt, (L, num_strings) tab (-100 = pad), gerçek uzunluk L.
    BiLSTM tüm chunk'ı görüp zamansal tutarlılık kurar -> orta-nota tel sıçraması azalır.
    """
    PAD = -100

    def __init__(self, cache_dir, split, chunk=200, augment=False, num_frets=24,
                 bins_per_semitone=2, onsets=False, onset_soft=None, short_frames=0, short_weight=1.0):
        """
        split: tek split ("train") ya da liste (["train", "train_comp"]) -- Katman 3.7.
        augment: True ise her chunk'a rastgele CQT-uzayı çoğaltma (sadece EĞİTİMDE).
        onsets: True ise tel başına onset hedefi de verilir (Katman 3.9) ->
                __getitem__ (x, tab, onset (L,S), L) döndürür.
        """
        self.chunk = chunk
        self.with_onsets = onsets
        self.onsets = []
        self.weights = []                  # Katman 3.10: kısa nota ağırlığı (onsets=True iken)
        self.augment = augment
        self.num_frets = num_frets
        self.bps = bins_per_semitone
        self.items = []
        splits = [split] if isinstance(split, str) else list(split)
        paths = []
        for sp in splits:
            sp_paths = sorted(glob.glob(os.path.join(cache_dir, sp, "*.npz")))
            if not sp_paths:
                raise FileNotFoundError(f"{cache_dir}/{sp} bos.")
            paths += sp_paths
        for path in paths:
            d = np.load(path)
            if "tab" not in d:
                raise KeyError(f"{path} icinde 'tab' yok. build_tab_labels.py calisti mi?")
            cqt = _normalize(d["cqt"]); tab = d["tab"].astype(np.int64)
            if onsets:
                from gtab.data.tab_labels import string_onsets, short_note_weights
                roll = d["onset"] if "onset" in d else None
                on_all = string_onsets(tab, roll, soft=onset_soft)
                w_all = (short_note_weights(tab, string_onsets(tab, roll, dilate=1) > 0,
                                            short_frames, short_weight)
                         if short_frames > 0 else np.ones(tab.shape, np.float32))
            T, n_bins = cqt.shape
            for start in range(0, T, chunk):
                c = cqt[start:start + chunk]; tb = tab[start:start + chunk]
                L = len(c)
                if L < chunk:
                    pc = np.zeros((chunk, n_bins), np.float32); pc[:L] = c
                    pt = np.full((chunk, tab.shape[1]), self.PAD, np.int64); pt[:L] = tb
                    c, tb = pc, pt
                self.items.append((c, tb, L))
                if onsets:
                    on = np.zeros((chunk, tab.shape[1]), np.float16)
                    on[:L] = on_all[start:start + chunk]
                    self.onsets.append(on)
                    wt = np.ones((chunk, tab.shape[1]), np.float16)
                    wt[:L] = w_all[start:start + chunk]
                    self.weights.append(wt)

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        c, tb, L = self.items[i]
        if self.augment:
            c, tb = self._augment(c, tb)
        if self.with_onsets:
            return (torch.from_numpy(c).unsqueeze(0), torch.from_numpy(tb),
                    torch.from_numpy(self.onsets[i].astype(np.float32)),
                    torch.from_numpy(self.weights[i].astype(np.float32)), L)
        return torch.from_numpy(c).unsqueeze(0), torch.from_numpy(tb), L

    # ------------------------------------------------------------ Katman 3.7
    def _augment(self, c, tb, max_shift=2, p_shift=0.5, gain_db=6.0, noise_std=0.02):
        """
        Etiketi KORUYAN CQT-uzayı çoğaltma (ham ses gerekmez):
        1) Perde kaydırma: k yarım ses = k*bins_per_semitone bin kaydır; etiket
           AYNI TELDE fret+k. Tel kimliği (tını) korunur, model her teli daha çok
           perdede duyar. Herhangi bir fret [0, num_frets] dışına taşarsa k=0.
        2) Kazanç: dB uzayında ±gain_db kayma (normalize: /80) -> kayıt seviyesi.
        3) Gürültü: küçük Gauss gürültüsü -> mikrofon/oda dayanıklılığı.
        Dolgu (PAD) kareleri ve sessiz (0) sınıflar değişmez.
        """
        # torch RNG'den tohum: DataLoader worker'larında hem farklı hem tekrarlanabilir
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        c = c.copy(); tb = tb.copy()
        valid = tb[:, 0] != self.PAD               # dolgu kareleri sıfır kalsın
        active = tb > 0
        if rng.random() < p_shift and active.any():
            k = int(rng.integers(-max_shift, max_shift + 1))
            fret = tb[active] - 1 + k
            if k != 0 and fret.min() >= 0 and fret.max() <= self.num_frets:
                b = k * self.bps
                shifted = np.zeros_like(c)
                if b > 0:
                    shifted[:, b:] = c[:, :-b]
                else:
                    shifted[:, :b] = c[:, -b:]
                c = shifted
                tb[active] = fret + 1
        if gain_db > 0:
            c[valid] += rng.uniform(-gain_db, gain_db) / 80.0
        if noise_std > 0:
            c[valid] += rng.normal(0.0, noise_std, size=c[valid].shape).astype(np.float32)
        return np.clip(c, 0.0, 1.0).astype(np.float32), tb

    def class_weights(self, n_classes, scheme="inv", cap=20.0):
        counts = np.zeros(n_classes, np.float64)
        for _, tab, _ in self.items:
            valid = tab[tab >= 0]
            counts += np.bincount(valid.reshape(-1), minlength=n_classes)
        counts = np.maximum(counts, 1.0)
        inv = counts.sum() / (n_classes * counts)
        if scheme == "inv":
            w = inv
        elif scheme == "sqrt":
            w = np.sqrt(inv)
        else:
            w = np.ones(n_classes)
        return torch.tensor(np.clip(w, 1.0 / cap, cap), dtype=torch.float32)


class PitchSeq(Dataset):
    """
    Katman 3.8: PERDE etiketli sekanslar (GAPS; tel etiketi yok).
    GuitarSetSeq ile aynı chunk mantığı; hedef (L, P) perde roll'u.
    RAM için CQT float16 saklanır, __getitem__'da float32'ye çevrilir.
    Çıktı: (1, L, n_bins) cqt, (L, P) frame, gerçek uzunluk L.
    """

    def __init__(self, files, chunk=200, onsets=False, dilate=2, onset_soft=None):
        """onsets: True ise perde onset roll'u da verilir -> (x, frame, onset (L,P), L)."""
        self.items = []
        self.with_onsets = onsets
        self.onsets = []
        for path in files:
            with np.load(path) as d:
                cqt = _normalize(d["cqt"]).astype(np.float16)
                fr = d["frame"].astype(np.uint8)
                on_all = d["onset"].astype(np.uint8) if onsets else None
            if onsets and onset_soft is not None:     # Katman 3.10: keskin hedef (tepe 1, sonraki kare soft)
                o1 = on_all.astype(np.float32); nx = np.zeros_like(o1); nx[1:] = o1[:-1] * float(onset_soft)
                on_all = np.maximum(o1, nx) * fr
            elif onsets and dilate > 1:               # tel onset'iyle aynı tolerans
                base = on_all.copy()
                for k in range(1, dilate):
                    on_all[k:] |= base[:-k]
                on_all &= fr
            T, nb = cqt.shape
            for s in range(0, T, chunk):
                c, f = cqt[s:s + chunk], fr[s:s + chunk]
                L = len(c)
                if L < chunk:
                    c = np.vstack([c, np.zeros((chunk - L, nb), np.float16)])
                    f = np.vstack([f, np.zeros((chunk - L, f.shape[1]), np.uint8)])
                self.items.append((c, f, L))
                if onsets:
                    o = np.zeros((chunk, fr.shape[1]), np.float16); o[:L] = on_all[s:s + chunk]
                    self.onsets.append(o)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        c, f, L = self.items[i]
        x, fr = torch.from_numpy(c.astype(np.float32)).unsqueeze(0), torch.from_numpy(f.astype(np.float32))
        if self.with_onsets:
            return x, fr, torch.from_numpy(self.onsets[i].astype(np.float32)), L
        return x, fr, L
