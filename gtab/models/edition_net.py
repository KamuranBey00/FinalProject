"""
Katman 3.13 Adım 2b — "Klasik edisyon önerisi" dizi modeli (sembolik; notalar üzerinde iki yönlü LSTM).

Adım 2a'daki log-lineer model yalnız önceki notayı görüyordu; konum ve parmak pasajın ileri/geri bağlamıyla
belirlenir. Burada her nota için:
  aşama 1 (konum): nota özellikleri -> BiLSTM -> tel (yalnız o perdenin çalınabildiği teller; perde = pitch − açık tel)
  aşama 2 (parmak): nota özellikleri + konum (tel, perde) -> BiLSTM -> parmak (boş tel = 0, basılı = 1..4)
Akor içinde aynı tel iki notaya verilmez (olasılık sırasıyla çözülür). Ses kullanılmaz: "duyulan konum" (ses) ile
"edisyon önerisi" ayrı çıktı modlarıdır (README14).
"""

import numpy as np
import torch
import torch.nn as nn

from gtab.core.instrument import STANDARD_6

INSTR = STANDARD_6
P_LO, N_P = INSTR.tuning[0], 49                      # perde gömmesi: E2 .. E6
N_FRET = INSTR.num_frets + 1


def note_features(notes, chord):
    """(N,3) sıralı notalar + akor bayrağı -> (perde indeksi (N,), sürekli özellikler (N, 12), geçerli tel (N, 6))."""
    notes = np.asarray(notes, float); N = len(notes)
    p = notes[:, 2].astype(int)
    valid = np.array([[0 <= q - t <= INSTR.num_frets for t in INSTR.tuning] for q in p], bool).reshape(N, 6)
    grp = np.cumsum(~np.asarray(chord, bool)) - 1
    size = np.bincount(grp)[grp]
    first = np.r_[0, np.flatnonzero(np.diff(grp)) + 1]
    pos_in = np.arange(N) - first[grp]
    gap = np.r_[0.0, np.diff(notes[:, 0])].clip(0, 4)
    dp = np.r_[0.0, np.diff(p)].clip(-24, 24) / 12
    cont = np.column_stack([chord, size / 6, pos_in / 6, np.log1p(gap * 10), np.log1p((notes[:, 1] - notes[:, 0]).clip(0, 8) * 10),
                            dp, valid]).astype(np.float32)
    return np.clip(p - P_LO, 0, N_P - 1), cont, valid


def position_features(strings, notes):
    """Konum (tel indeksi | -1) -> (N, 6 + 25 + 1): tel one-hot, perde one-hot, bilinmiyor."""
    N = len(strings); x = np.zeros((N, 6 + N_FRET + 1), np.float32)
    for i, s in enumerate(strings):
        f = int(notes[i, 2]) - INSTR.tuning[s] if s >= 0 else -1
        if not 0 <= f < N_FRET:                      # bilinmiyor / gitar aralığı dışı
            x[i, -1] = 1.0; continue
        x[i, s] = 1.0; x[i, 6 + f] = 1.0
    return x


class EditionNet(nn.Module):
    def __init__(self, hidden=128, emb=16):
        super().__init__()
        self.emb = nn.Embedding(N_P, emb)
        self.inp = nn.Linear(emb + 12, hidden)
        self.rnn = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.string_head = nn.Linear(2 * hidden, 6)
        self.inp2 = nn.Linear(emb + 12 + 6 + N_FRET + 1, hidden)
        self.rnn2 = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.finger_head = nn.Linear(2 * hidden, 5)

    def _base(self, pidx, cont):
        return torch.cat([self.emb(pidx), cont], -1)

    def strings(self, pidx, cont, valid):
        """-> (B, N, 6) tel log-olasılığı (geçersiz teller -inf)."""
        h, _ = self.rnn(torch.relu(self.inp(self._base(pidx, cont))))
        z = self.string_head(h).masked_fill(~valid, -1e9)
        return torch.log_softmax(z, -1)

    def fingers(self, pidx, cont, posf, fret0):
        """posf: position_features; fret0: (B, N) boş tel mi -> (B, N, 5) parmak log-olasılığı (0 = boş tel)."""
        h, _ = self.rnn2(torch.relu(self.inp2(torch.cat([self._base(pidx, cont), posf], -1))))
        z = self.finger_head(h)
        m = torch.ones_like(z, dtype=torch.bool)
        m[..., 0] = fret0; m[..., 1:] = ~fret0.unsqueeze(-1)
        return torch.log_softmax(z.masked_fill(~m, -1e9), -1)


def resolve_chords(lp, chord):
    """(N, 6) tel log-olasılığı -> tel dizisi; akor içinde her tel en çok bir notaya (en olası atama önce)."""
    lp = np.asarray(lp); N = len(lp); out = lp.argmax(1).copy()
    grp = np.cumsum(~np.asarray(chord, bool)) - 1
    for g in np.unique(grp):
        idx = np.flatnonzero(grp == g)
        if len(idx) < 2:
            continue
        pairs = sorted(((lp[i, s], i, s) for i in idx for s in range(6) if lp[i, s] > -1e8), reverse=True)
        done, used = {}, set()
        for _, i, s in pairs:
            if i not in done and s not in used:
                done[i] = s; used.add(s)
        for i in idx:
            out[i] = done.get(i, out[i])
    return out


@torch.no_grad()
def predict(net, notes, chord, device, fixed=None):
    """
    Bir kayıt -> (tel (N,), perde (N,), parmak (N,)). Ses kullanılmaz ("edisyon önerisi" modu).
    fixed: not başına tel | -1 (verilirse konum sabit; parmak ölçümü).
    """
    net.eval()
    pidx, cont, valid = note_features(notes, chord)
    t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device).unsqueeze(0)
    lp = net.strings(t(pidx, torch.long), t(cont, torch.float32), t(valid, torch.bool))[0].cpu().numpy()
    s = np.where(valid.any(1), resolve_chords(lp, chord), -1)    # hiçbir telde çalınamayan nota: -1
    if fixed is not None:
        s = np.where(np.asarray(fixed) >= 0, fixed, s)
    fret = np.where(s >= 0, np.asarray(notes)[:, 2].astype(int) - np.array(INSTR.tuning)[s], -1)
    fl = net.fingers(t(pidx, torch.long), t(cont, torch.float32), t(position_features(s, np.asarray(notes)), torch.float32),
                     t(fret == 0, torch.bool))[0]
    return s, fret, fl.argmax(-1).cpu().numpy()
