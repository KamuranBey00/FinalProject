"""
Katman 5 Adım 1b — Öğrenilen vuruş + ölçü başı aktivasyonu.

Girdi: ses modelinin (deep2) DONMUŞ gövde temsili `encode` (T, 256) — nota başlangıcı / perde / pes bilgisini zaten taşıyor;
ses modeli değişmez. Ağ: Linear -> 2 katmanlı BiLSTM -> kare başına 3 sınıf (yok / vuruş / ölçü başı) softmax.
Çözümleme: gtab/decoding/beat_dbn.py (bar-pointer DBN).
"""

import numpy as np
import torch
import torch.nn as nn

from gtab.config import FRAME_RATE
from gtab.models.inference import _chunked
from gtab.paths import ckpt_path

N_IN = 256


@torch.no_grad()
def encode_features(model, cqt_norm, device):
    """Ses modeli gövdesi: normalize CQT (T, n_bins) -> (T, 256) float16 (parça parça, predict_heads ile aynı pencereler)."""
    return _chunked(model.encode, cqt_norm, device)[0].astype(np.float16)


class BeatNet(nn.Module):
    def __init__(self, n_in=N_IN, hidden=128):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(0.2), nn.Linear(n_in, hidden), nn.ReLU())
        self.rnn = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.head = nn.Linear(2 * hidden, 3)

    def forward(self, x):
        """(B, T, n_in) -> (B, T, 3) logit (yok, vuruş, ölçü başı)."""
        h, _ = self.rnn(self.inp(x))
        return self.head(h)


def frame_targets(beats, downbeats, T, widen=0.5):
    """
    Vuruşlar / ölçü başları (sn) -> (T, 3) yumuşak hedef: olay karesi 1, komşu kareler `widen` (±1 kare; etiket zamanı
    ~23 ms kare çözünürlüğünde belirsiz).
    """
    y = np.zeros((T, 3), np.float32)
    db = set(np.round(np.asarray(downbeats) * FRAME_RATE).astype(int).tolist())
    for f in np.round(np.asarray(beats) * FRAME_RATE).astype(int):
        c = 2 if f in db else 1
        for g, w in ((f, 1.0), (f - 1, widen), (f + 1, widen)):
            if 0 <= g < T:
                y[g, c] = max(y[g, c], w)
    y[:, 0] = np.clip(1 - y[:, 1:].sum(1), 0, 1)
    return y / y.sum(1, keepdims=True)


@torch.no_grad()
def activations(net, feats, device):
    """(T, 256) -> (T, 3) softmax (p_none, p_beat, p_downbeat); tüm kayıt tek seferde."""
    net.eval()
    x = torch.as_tensor(np.asarray(feats, np.float32), device=device).unsqueeze(0)
    return torch.softmax(net(x)[0], -1).cpu().numpy()


def load_beat_net(name, device):
    """Checkpoint adı (checkpoints/ altında) ya da yolu -> (değerlendirme kipinde BeatNet, kayıtlı DBN ayarları)."""
    ck = torch.load(ckpt_path(name), map_location=device)
    net = BeatNet().to(device)
    net.load_state_dict(ck["model"])
    return net.eval(), ck.get("dbn", {})
