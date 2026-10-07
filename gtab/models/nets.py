"""
Modeller. Ortak bir CNN gövdesi; iki farklı kafa:
- PitchCNN (Katman 2): CQT penceresi -> perde logitleri (multi-label)
- TabCNN   (Katman 3): CQT penceresi -> 6 tel × sınıf logitleri (tel başına softmax)

TabCNN, Katman 2'nin aynı pencereli girdisini kullanır; sadece çıktı kafası değişir.
Tel başına ayrı softmax = "her telde en fazla bir nota" fiziksel kısıtı.

Sekans modelleri (Katman 3.5+): TabCRNN, TabCRNNOnset (3.9 Adım 1),
isteğe bağlı HarmonicStack girişi (3.9 Adım 2).
"""

import math

import torch
import torch.nn as nn

# Katman 3.9 / Adım 2 varsayılan harmonikleri (Basic Pitch ile aynı küme)
DEFAULT_HARMONICS = (0.5, 1, 2, 3, 4, 5)


class HarmonicStack(nn.Module):
    """
    Harmonik istifleme: log-frekanslı CQT'yi h. harmoniğe denk gelen bin sayısı
    kadar kaydırıp kanal olarak üst üste koyar -> (B, H, L, F).
    Bir notanın temel frekansı ile 2., 3., ... harmoniği AYNI frekans konumunda
    hizalanır; konvolüsyon notanın harmonik profilini (tının, dolayısıyla telin
    imzası) doğrudan görür. Kaydırma = round(bins_per_octave · log2 h)
    (24 bin/oktav: 0.5→-24, 1→0, 2→24, 3→38, 4→48, 5→56). Aralık dışı = 0 (sessizlik).
    Parametresi yok; mevcut CQT önbelleğiyle çalışır (yeniden üretim gerekmez).
    """
    def __init__(self, harmonics=DEFAULT_HARMONICS, bins_per_octave=24):
        super().__init__()
        self.harmonics = tuple(harmonics)
        self.shifts = [int(round(bins_per_octave * math.log2(h))) for h in self.harmonics]

    def forward(self, x):                        # x: (B, 1, L, F)
        F = x.shape[-1]
        chans = []
        for k in self.shifts:
            if k == 0:
                chans.append(x)
            elif k > 0:                          # bin f <- bin f+k (üst harmonik)
                chans.append(torch.cat([x[..., k:], x.new_zeros(*x.shape[:-1], min(k, F))], -1)[..., :F])
            else:                                # bin f <- bin f+k (alt harmonik)
                chans.append(torch.cat([x.new_zeros(*x.shape[:-1], min(-k, F)), x[..., :k]], -1)[..., :F])
        return torch.cat(chans, dim=1)


def _backbone():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout(0.25),
    )


class PitchCNN(nn.Module):
    def __init__(self, n_pitches, context=9, n_bins=192):
        super().__init__()
        self.features = _backbone()
        h, w = context // 2, n_bins // 2
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * h * w, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, n_pitches),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class TabCNN(nn.Module):
    """Çıktı: (batch, num_strings, n_classes) logit. Tel başına softmax ile eğitilir."""
    def __init__(self, num_strings=6, n_classes=26, context=9, n_bins=192):
        super().__init__()
        self.num_strings = num_strings
        self.n_classes = n_classes
        self.features = _backbone()
        h, w = context // 2, n_bins // 2
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * h * w, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_strings * n_classes),
        )

    def forward(self, x):
        z = self.head(self.features(x))
        return z.view(-1, self.num_strings, self.n_classes)


class TabCRNN(nn.Module):
    """
    Katman 3.5 — CNN gövdesi + BiLSTM zamansal kafa.
    Girdi: (B, 1, L, n_bins) bir SEKANS. CNN her kareden frekans deseni çıkarır
    (zaman ekseni L korunur; sadece frekans havuzlanır), BiLSTM L boyunca zamansal
    tutarlılık kurar. Çıktı: (B, L, num_strings, n_classes).
    Amaç: orta-nota tel sıçramasını kaynağında azaltmak -> daha yüksek precision,
    daha stabil nota sınırları (Katman 4 teknik atfı için şart).
    """
    def __init__(self, num_strings=6, n_classes=26, n_bins=192,
                 lstm_hidden=128, lstm_layers=2, harmonics=None):
        super().__init__()
        self.num_strings = num_strings
        self.n_classes = n_classes
        self.harmonics = tuple(harmonics) if harmonics else None
        self.hstack = HarmonicStack(self.harmonics) if self.harmonics else None
        in_ch = len(self.harmonics) if self.harmonics else 1
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((1, 4)),               # SADECE frekans havuzla; zaman (L) korunur
            nn.Dropout(0.25),
        )
        feat = 64 * (n_bins // 4)
        self.proj = nn.Linear(feat, 256)
        self.lstm = nn.LSTM(256, lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True,
                            dropout=0.3 if lstm_layers > 1 else 0.0)
        self.head = nn.Linear(lstm_hidden * 2, num_strings * n_classes)

    def encode(self, x):                         # x: (B, 1, L, n_bins)
        if self.hstack is not None:
            x = self.hstack(x)                   # (B, H, L, n_bins)
        z = self.cnn(x)                          # (B, 64, L, n_bins/4)
        B, C, L, F = z.shape
        z = z.permute(0, 2, 1, 3).reshape(B, L, C * F)   # (B, L, feat)
        z = torch.relu(self.proj(z))             # (B, L, 256)
        z, _ = self.lstm(z)                      # (B, L, hidden*2)
        return z

    def forward(self, x):
        z = self.encode(x)
        B, L, _ = z.shape
        return self.head(z).view(B, L, self.num_strings, self.n_classes)


class TabCRNNOnset(TabCRNN):
    """
    Katman 3.9 / Adım 1 — TabCRNN + ONSET kafası (Onsets & Frames ilkesi).
    Ek çıktı: (B, L, num_strings) onset logit'i = "bu karede bu telde yeni nota başladı".
    Gövde ve tab kafası TabCRNN ile birebir aynı -> TabCRNN checkpoint'i
    (ör. tabcrnn_gaps.pt) strict=False ile yüklenip ince ayar yapılabilir.
    forward() yalnızca tab logit'i döndürür -> tüm eski değerlendirme kodu çalışır.
    """
    def __init__(self, num_strings=6, n_classes=26, n_bins=192, lstm_hidden=128, lstm_layers=2,
                 harmonics=None, offset=False):
        super().__init__(num_strings, n_classes, n_bins, lstm_hidden, lstm_layers, harmonics)
        self.onset_head = nn.Linear(lstm_hidden * 2, num_strings)
        # Katman 3.10 Adım 3c: OFFSET kafası ("bu karede bu teldeki nota bitiyor"); offset=False
        # iken modül yok -> eski checkpoint'ler birebir yüklenir.
        self.offset_head = nn.Linear(lstm_hidden * 2, num_strings) if offset else None
        if offset:
            nn.init.constant_(self.offset_head.bias, -2.94)
        # Seyrek hedef (karelerin ~%5'i onset): bias'ı bu önsel orana göre başlat
        # (logit(0.05) ≈ -2.94). Sıfır bias ile kafa her yerde ~0.5 tahminle başlıyor
        # ve tek başına bu önseli öğrenmek epoch'lar sürüyordu (duman testi).
        nn.init.constant_(self.onset_head.bias, -2.94)

    def forward_both(self, x):
        z = self.encode(x)
        B, L, _ = z.shape
        tab = self.head(z).view(B, L, self.num_strings, self.n_classes)
        return tab, self.onset_head(z)           # (B,L,S,C), (B,L,S)

    def forward_all(self, x):
        """-> tab (B,L,S,C), onset logit (B,L,S), offset logit (B,L,S) | None"""
        z = self.encode(x)
        B, L, _ = z.shape
        tab = self.head(z).view(B, L, self.num_strings, self.n_classes)
        off = self.offset_head(z) if self.offset_head is not None else None
        return tab, self.onset_head(z), off


def warm_start_state(state, model):
    """
    Eski bir checkpoint'in ağırlıklarını (yeni) modele taşır, YAPILANLARI KORUYARAK:
      - onset_head / offset_head yoksa: modelin kendi (önsel bias'lı) başlangıcı kalır
      - giriş 1 kanal -> H harmonik kanal: h=1 kanalına eski ağırlık, diğerleri 0
        -> yeni model başlangıçta eski modelle BİREBİR aynı çıktıyı verir;
           eğitim harmonik kanalları kullanmayı öğrenir.
    -> (yüklenen state_dict, eksik anahtarlar)
    """
    own = model.state_dict()
    out = {}
    for k, v in state.items():
        if k not in own:
            continue
        if v.shape != own[k].shape and k == "cnn.0.weight" and getattr(model, "harmonics", None):
            w = torch.zeros_like(own[k])
            w[:, list(model.harmonics).index(1)] = v[:, 0]
            v = w
        out[k] = v
    missing = [k for k in own if k not in out]
    own.update(out)
    model.load_state_dict(own)
    return missing
