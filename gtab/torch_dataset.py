"""
PyTorch veri okuyucular. Önbellekteki .npz'lerden CQT pencereleri üretir.
- GuitarSetFrames (Katman 2): pencere -> perde vektörü (multi-label)
- GuitarSetTab    (Katman 3): pencere -> (6,) tel başına fret sınıfı (int)

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

    def __init__(self, cache_dir, split, chunk=200):
        self.chunk = chunk
        self.items = []
        paths = sorted(glob.glob(os.path.join(cache_dir, split, "*.npz")))
        if not paths:
            raise FileNotFoundError(f"{cache_dir}/{split} bos.")
        for path in paths:
            d = np.load(path)
            if "tab" not in d:
                raise KeyError(f"{path} icinde 'tab' yok. build_tab_labels.py calisti mi?")
            cqt = _normalize(d["cqt"]); tab = d["tab"].astype(np.int64)
            T, n_bins = cqt.shape
            for start in range(0, T, chunk):
                c = cqt[start:start + chunk]; tb = tab[start:start + chunk]
                L = len(c)
                if L < chunk:
                    pc = np.zeros((chunk, n_bins), np.float32); pc[:L] = c
                    pt = np.full((chunk, tab.shape[1]), self.PAD, np.int64); pt[:L] = tb
                    c, tb = pc, pt
                self.items.append((c, tb, L))

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        c, tb, L = self.items[i]
        return torch.from_numpy(c).unsqueeze(0), torch.from_numpy(tb), L

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
