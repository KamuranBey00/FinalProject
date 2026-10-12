"""
Katman 5 Adım 1q — Bağlamlı kuantalayıcı (RhythmNet) eğitimi. Sembolik (nota başlangıçları + vuruşlar), ses yok, dakikalar.

Veri: ritim önbelleği (build_rhythm), YALNIZ gaps_fit; seçim gaps_val oracle IOI doğruluğu (gerçek vuruş + gerçek nota).
Gerçek vuruşlar aynı notalardan türetildiği için fazla temiz -> artırma (kayıt başına her epoch, olasılık 1 - p_clean, varsayılan 0.5):
  vuruş   : bağımsız gürültü (sigma ~ U(0, 0.08) vuruş aralığı) + sabit kayma U(±0.1) + metrik düzey (yarım / çift vuruş)
  nota    : başlangıç sapması (10 ms), düşürme U(0, %20), sahte nota (fragman, etiketsiz) U(0, %10)  -> deep2 hataları
Kayıp: delta sınıfı CE (konumu bilinen, hedefi 1/24 ızgarada ve pencerede olan notalar).
Epoch seçimi yapısal çözümlemeyle (lam = 1); sonunda lam gaps_val'de taranır ve checkpoint'e yazılır.
Çıktı: checkpoints/rhythm_q.pt

Çalıştırma:
    python -m scripts.train.train_rhythm_q              # 80 epoch, ~11 dk (GPU); Adam + kosinüs lr
    python -m scripts.train.train_rhythm_q --epochs 2      # duman testi
"""

import argparse
import os

import numpy as np
import torch

from gtab.data.gaps import gaps_files
from gtab.decoding.rhythm import quantize
from gtab.evaluation.rhythm_metrics import ioi_accuracy
from gtab.models.rhythm_net import N_CLS, N_FEAT, TICKS, WIN, RhythmNet, note_features, predict
from gtab.paths import CACHE_DIR, ckpt_path
from gtab.utils import get_device, set_seed

RHY_DIR = os.path.join(CACHE_DIR, "rhythm")
CHUNK = 256
RULE = dict(lam=0.2, mu=0.3, chord_tol=0.03)         # Adım 0'da gaps_val'de seçilen kural (karşılaştırma)
BEAT_JIT, BEAT_SHIFT, P_METRIC = 0.08, 0.1, 0.15
NOTE_JIT, DROP, ADD = 0.01, 0.2, 0.1
LAMS = (0.0, 0.5, 1.0, 2.0, 3.0)                    # yapısal çözümleme karmaşıklık cezası (epoch seçimi lam = 1)


def load(files):
    out = []
    for f in files:
        rf = os.path.join(RHY_DIR, os.path.basename(f))
        if os.path.exists(rf):
            with np.load(rf) as d:
                out.append(dict(notes=d["notes"].astype(float), s=d["note_beat"], beats=d["beats"]))
    return out


def augment(d, rng):
    """-> (notlar, konum, vuruşlar) bozulmuş kopya (konum vuruş numaralamasıyla birlikte dönüştürülür)."""
    notes, s, beats = d["notes"].copy(), d["s"].copy(), d["beats"].copy()
    r = rng.random()
    if r < P_METRIC and len(beats) >= 6:                          # yarım tempo: her ikinci vuruş
        ph = rng.integers(2); beats = beats[ph::2]; s = (s - ph) / 2
    elif r < 2 * P_METRIC:                                        # çift tempo: ara vuruşlar
        beats = np.sort(np.r_[beats, (beats[1:] + beats[:-1]) / 2]); s = 2 * s
    ibi = np.r_[np.diff(beats), np.diff(beats)[-1]]
    beats = np.sort(beats + ibi * (rng.normal(0, rng.uniform(0, BEAT_JIT), len(beats))
                                   + rng.uniform(-BEAT_SHIFT, BEAT_SHIFT)))
    notes[:, :2] += rng.normal(0, NOTE_JIT, len(notes))[:, None]
    keep = rng.random(len(notes)) >= rng.uniform(0, DROP)
    notes, s = notes[keep], s[keep]
    add = rng.random(len(notes)) < rng.uniform(0, ADD)
    frag = notes[add].copy(); frag[:, :2] += rng.uniform(0.05, 0.3, len(frag))[:, None]
    return np.r_[notes, frag], np.r_[s, np.full(len(frag), np.nan)], beats


def targets(s, c):
    """Konum -> delta sınıfı | -100 (konum yok, 1/24 ızgara dışı ya da pencere dışı)."""
    t = s * TICKS
    ok = np.isfinite(t) & (np.abs(t - np.round(np.nan_to_num(t))) < 1e-3)
    y = np.round(np.nan_to_num(t)).astype(int) - c + WIN
    return np.where(ok & (y >= 0) & (y < N_CLS), y, -100)


def epoch_items(data, rng, p_clean):
    """Her kayıt bir kez bozulur -> (özellik, hedef) -> rastgele 256 notalık kesitler."""
    items = []
    for d in data:
        notes, s, beats = (d["notes"], d["s"], d["beats"]) if rng.random() < p_clean else augment(d, rng)
        o = np.lexsort((notes[:, 2], notes[:, 0]))
        x, c = note_features(notes[o], beats)
        y = targets(s[o], c)
        for a in range(0, len(x), CHUNK):
            a = min(a + int(rng.integers(0, CHUNK // 2)), max(0, len(x) - CHUNK)) if a else 0
            items.append((x[a:a + CHUNK], y[a:a + CHUNK]))
    rng.shuffle(items)
    return items


def collate(chunk, device):
    L = max(len(x) for x, _ in chunk)
    X = np.zeros((len(chunk), L, N_FEAT), np.float32); Y = np.full((len(chunk), L), -100, np.int64)
    for b, (x, y) in enumerate(chunk):
        X[b, :len(x)], Y[b, :len(y)] = x, y
    return torch.as_tensor(X, device=device), torch.as_tensor(Y, device=device)


def ioi(data, quant, rng=None):
    """gaps_val IOI doğruluğu (ölçek sabit r = 1). rng verilirse vuruşlar sabit tohumlu gürültüyle bozulur."""
    ok = n = 0
    for d in data:
        beats = d["beats"]
        if rng is not None:
            ibi = np.r_[np.diff(beats), np.diff(beats)[-1]]
            beats = np.sort(beats + ibi * (rng.normal(0, 0.05, len(beats)) + rng.uniform(-0.1, 0.1)))
        a, b, _ = ioi_accuracy(quant(d["notes"], beats), d["s"], d["notes"][:, 0], (1,))
        ok += a; n += b
    return ok / max(n, 1)


def main(out, epochs, lr, bs, p_clean, seed):
    set_seed(seed); device = get_device(); rng = np.random.default_rng(seed)
    fit_f, val_f = gaps_files("gaps_train")
    fit, val = load(fit_f), load(val_f)
    print(f"gaps_fit {len(fit)} kayit ({sum(len(d['notes']) for d in fit)} nota) | gaps_val {len(val)} kayit | {device}")
    rule = lambda n, b: quantize(n[:, 0], b, **RULE)
    print(f"kural (Adim 0) gaps_val IOI: oracle {ioi(val, rule):.3f} | gurultulu vurus {ioi(val, rule, np.random.default_rng(0)):.3f}")
    net = RhythmNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
    best = -1
    for ep in range(1, epochs + 1):
        net.train(); tl = n = 0
        items = epoch_items(fit, rng, p_clean)
        for i in range(0, len(items), bs):
            X, Y = collate(items[i:i + bs], device)
            loss = ce(net(X).reshape(-1, N_CLS), Y.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); n += 1
        sched.step()
        quant = lambda nt, b: predict(net, nt, b, device)
        a, a_n = ioi(val, quant), ioi(val, quant, np.random.default_rng(0))
        mark = ""
        if a > best:
            best = a; mark = "  <- kaydedildi"
            torch.save({"model": net.state_dict()}, ckpt_path(out))
        print(f"ep {ep:2d} | kayip {tl / n:.4f} | gaps_val IOI oracle {a:.3f} gurultulu vurus {a_n:.3f}{mark}", flush=True)
    # yapısal çözümleme cezası: en iyi epoch'ta gaps_val oracle IOI ile seçilir, checkpoint'e yazılır
    ck = torch.load(ckpt_path(out), map_location=device)
    net.load_state_dict(ck["model"])
    print("lam (yapisal cozumleme) taramasi, gaps_val:")
    sc = {}
    for lam in LAMS:
        quant = lambda nt, b: predict(net, nt, b, device, lam)
        sc[lam] = ioi(val, quant)
        print(f"  lam {lam:<4} oracle IOI {sc[lam]:.3f} | gurultulu vurus {ioi(val, quant, np.random.default_rng(0)):.3f}")
    ck["lam"] = max(LAMS, key=sc.get)
    torch.save(ck, ckpt_path(out))
    print(f"En iyi gaps_val oracle IOI {sc[ck['lam']]:.3f} (lam {ck['lam']}) -> {ckpt_path(out)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rhythm_q.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--p-clean", type=float, default=0.5, help="bozulmadan kullanılan kayıt oranı")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    main(a.out, a.epochs, a.lr, a.batch_size, a.p_clean, a.seed)
