"""
Modeller. Ortak bir CNN gövdesi; iki farklı kafa:
- PitchCNN (Katman 2): CQT penceresi -> perde logitleri (multi-label)
- TabCNN   (Katman 3): CQT penceresi -> 6 tel × sınıf logitleri (tel başına softmax)

TabCNN, Katman 2'nin aynı pencereli girdisini kullanır; sadece çıktı kafası değişir.
Tel başına ayrı softmax = "her telde en fazla bir nota" fiziksel kısıtı.
"""

import torch
import torch.nn as nn


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
                 lstm_hidden=128, lstm_layers=2):
        super().__init__()
        self.num_strings = num_strings
        self.n_classes = n_classes
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
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

    def forward(self, x):                        # x: (B, 1, L, n_bins)
        z = self.cnn(x)                          # (B, 64, L, n_bins/4)
        B, C, L, F = z.shape
        z = z.permute(0, 2, 1, 3).reshape(B, L, C * F)   # (B, L, feat)
        z = torch.relu(self.proj(z))             # (B, L, 256)
        z, _ = self.lstm(z)                      # (B, L, hidden*2)
        z = self.head(z)                         # (B, L, S*ncls)
        return z.view(B, L, self.num_strings, self.n_classes)
