"""
Katman 5 Adım 1q — Bağlamlı kuantalayıcı (sembolik; notalar üzerinde iki yönlü LSTM, ses yok).

Kural kuantalayıcı (gtab/decoding/rhythm.quantize) her vuruşu bağımsız çözüyordu; hataları "yanlış vuruş içi bölüm"
(gerçek vuruşta notaların %97'si hedefe <= 2/24 vuruş yakın). Burada her nota için:
  girdi : vuruşa göre sürekli konum u -> en yakın 1/24 tick c (vuruş içi yeri one-hot) + artık, yerel vuruş aralığı,
          komşu notalara uzaklık (vuruş ve saniye), perde, süre, ±CTX vuruştaki notaların vuruş içi tick dağılımı
          (üçlemeli pasajlar azınlıkta; yalnız dizi bağlamıyla ağ ikili ızgaraya kayıyordu)
  çıktı : hedef tick = c + delta, delta in [-WIN, WIN] (sınıflandırma) -> konum (c + delta) / TICKS (vuruş birimi)
Akor ve arpej ayrı kural yok: aynı konum ağ tarafından verilir (önceki notaya uzaklık girdide).
Çözümleme (decode): nota başına olasılıklar + vuruş başına ortak ızgara (kuralın yapısı, seçim ağın olasılıklarıyla).
"""

import numpy as np
import torch
import torch.nn as nn

from gtab.decoding.rhythm import COMPLEXITY, GRIDS, beat_position
from gtab.paths import ckpt_path

TICKS = 24                                           # vuruş içi çözünürlük (GRIDS 2, 3, 4, 6, 8, 12'nin ortak katı)
WIN = 12                                             # hedef = en yakın tick ± WIN (yarım vuruş)
N_CLS = 2 * WIN + 1
CTX = 8.0                                            # yerel konum histogramı penceresi (± vuruş)
N_FEAT = 2 * TICKS + 8


def note_features(notes, beats):
    """(N, 3) notalar (başlangıç sırasıyla) + vuruşlar (sn) -> (özellikler (N, N_FEAT), en yakın tick c (N,))."""
    notes, beats = np.asarray(notes, float), np.asarray(beats, float)
    on, N = notes[:, 0], len(notes)
    u = beat_position(on, beats)
    c = np.round(u * TICKS).astype(int)
    k = np.clip(np.searchsorted(beats, on, side="right") - 1, 0, len(beats) - 2)
    ibi = np.diff(beats)[k]
    lb = lambda x: np.log2(np.clip(x, 1 / 64, 8)) / 6           # vuruş biriminde uzaklık
    ls = lambda x: np.log(np.clip(x, 0.005, 4)) / 5             # saniye
    du, dt = np.diff(u), np.diff(on)
    f = np.zeros((N, N_FEAT), np.float32)
    f[np.arange(N), c % TICKS] = 1.0
    # ±CTX vuruştaki notaların vuruş içi tick dağılımı: pasaj ikili mi üçlü mü (u sıralı)
    cs = np.r_[np.zeros((1, TICKS)), np.cumsum(f[:, :TICKS], 0)]
    h = cs[np.searchsorted(u, u + CTX, "right")] - cs[np.searchsorted(u, u - CTX, "left")]
    f[:, TICKS:2 * TICKS] = h / h.sum(1, keepdims=True)
    f[:, 2 * TICKS:] = np.column_stack([u * TICKS - c, np.log(ibi / 0.7), lb(np.r_[8, du]), lb(np.r_[du, 8]),
                                    ls(np.r_[4, dt]), ls(np.r_[dt, 4]), (notes[:, 2] - 64) / 24,
                                    lb((notes[:, 1] - on) / ibi)])
    return f, c


class RhythmNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.inp = nn.Linear(N_FEAT, hidden)
        self.rnn = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.head = nn.Linear(2 * hidden, N_CLS)

    def forward(self, x):
        """(B, N, N_FEAT) -> (B, N, N_CLS) delta logit'leri."""
        h, _ = self.rnn(torch.relu(self.inp(x)))
        return self.head(h)


def decode(c, lp, lam):
    """
    Yapısal çözümleme: her vuruşta (en yakın tick'e göre) ortak bölüm d in GRIDS; d'nin ızgarasındaki en olası tick
    nota başına seçilir, d = argmax sum log p - lam * karmaşıklık(d). (Ağ tek başına bir vuruşta ikili ve üçlü
    konumları karıştırabiliyor.) -> (N,) tick.
    """
    tick = c[:, None] + np.arange(-WIN, WIN + 1)[None]
    kb, t = np.floor_divide(c, TICKS), np.empty(len(c), int)
    for k in np.unique(kb):
        idx = np.flatnonzero(kb == k)
        best = None
        for d in GRIDS:
            m = np.where(tick[idx] % (TICKS // d) == 0, lp[idx], -np.inf)
            sc = m.max(1).sum() - lam * COMPLEXITY[d]
            if best is None or sc > best[0]:
                best = (sc, tick[idx, m.argmax(1)])
        t[idx] = best[1]
    return t


@torch.no_grad()
def predict(net, notes, beats, device, lam=1.0):
    """Bir kayıt -> (N,) kuantalanmış konum (vuruş birimi, beats indeksine göre), notes sırasıyla."""
    notes = np.asarray(notes, float).reshape(-1, 3)
    out = np.full(len(notes), np.nan)
    if not len(notes) or len(beats) < 2:
        return out
    o = np.lexsort((notes[:, 2], notes[:, 0]))
    x, c = note_features(notes[o], beats)
    net.eval()
    lp = torch.log_softmax(net(torch.as_tensor(x, device=device).unsqueeze(0))[0], -1).cpu().numpy()
    out[o] = decode(c, lp, lam) / TICKS
    return out


def load_rhythm_net(name, device):
    """Checkpoint adı (checkpoints/ altında) ya da yolu -> (değerlendirme kipinde RhythmNet, seçilen lam)."""
    ck = torch.load(ckpt_path(name), map_location=device)
    net = RhythmNet().to(device)
    net.load_state_dict(ck["model"])
    return net.eval(), ck.get("lam", 1.0)
