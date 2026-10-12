"""
Katman 5 Adım 1b — Öğrenilen vuruş + ölçü başı ağı (BeatNet) eğitimi. Önce: python -m scripts.data.build_beat_feats

Girdi: deep2 gövde temsili önbelleği (donmuş; ses modeli değişmez). Hedef: kare başına (yok, vuruş, ölçü başı) yumuşak
etiket (±1 kare 0.5). GAPS etiketi yalnız eşleşen notaların kapsadığı aralıkta (± yarım vuruş) -> dışı maskelenir.
Tempo artırması (--stretch): her kesit rastgele hızda (x0.8 .. x1.25) yeniden örneklenir, etiketler aynı oranla.
Veri: gaps_fit + GuitarSet train + train_comp (metronom, 4/4); seçim gaps_val (vuruş F + ölçü başı F) / 2, tepe seçimiyle
(hızlı vekil). Sonunda en iyi epoch'ta DBN ayarları (tempo geçiş cezası, vuruş ödülü) gaps_val'de taranır, checkpoint'e
yazılır. GuitarSet val yalnız rapor.
Çıktı: checkpoints/beat_net.pt

Çalıştırma:
    python -m scripts.train.train_beat                           # 30 epoch + DBN taramasi, ~25 dk (GPU)
    python -m scripts.train.train_beat --epochs 2 --limit 4      # duman testi
"""

import argparse
import glob
import os

import numpy as np
import torch

from gtab.config import FRAME_RATE
from gtab.data.gaps import gaps_files
from gtab.decoding.beat_dbn import dbn_decode, peak_beats
from gtab.evaluation.rhythm_metrics import beat_scores, in_span, onset_f
from gtab.models.beat_net import N_IN, BeatNet, activations, frame_targets
from gtab.paths import CACHE_DIR, ckpt_path
from gtab.utils import get_device, set_seed

RHY_DIR = os.path.join(CACHE_DIR, "rhythm")
CROP = 1500                                          # ~35 sn
DBN_GRID = [dict(trans_lambda=t, obs_lambda=o) for t in (3.0, 10.0, 30.0) for o in (16, 32, 64, 128)]


def load(files, feat_dir, gaps):
    out = []
    for f in files:
        tid = os.path.basename(f)[:-4]
        ff, rf = os.path.join(feat_dir, tid + ".npy"), os.path.join(RHY_DIR, tid + ".npz")
        if not (os.path.exists(ff) and os.path.exists(rf)):
            continue
        x = np.load(ff, mmap_mode="r")
        with np.load(rf) as g:
            beats, downs = g["beats"], g["downbeats"]
        T = len(x)
        mask = np.ones(T, np.float32)
        if gaps:                                     # etiketli aralık dışı bilinmiyor
            h = np.median(np.diff(beats)) / 2
            t = np.arange(T) / FRAME_RATE
            mask = ((t >= beats[0] - h) & (t <= beats[-1] + h)).astype(np.float32)
        out.append(dict(id=tid, x=x, y=frame_targets(beats, downs, T), mask=mask, beats=beats, downs=downs))
    return out


def crops(data, rng):
    items = []
    for d in data:
        T = len(d["x"])
        for _ in range(max(1, int(np.ceil(T / CROP)))):
            a = int(rng.integers(0, max(1, T - CROP + 1)))
            if d["mask"][a:a + CROP].any():
                items.append((d, a))
    rng.shuffle(items)
    return items


def stretched(d, a, s):
    """Tempo artırması: kesitte yeni kare j = kaynak kare a + j*s (s > 1 hızlanır). Temsil doğrusal ara değer,
    hedef vuruş zamanlarından yeniden kurulur (aynı ±1 kare genişletme), maske en yakın kare."""
    L = int(min(CROP, (len(d["x"]) - 2 - a) / s))
    src = a + np.arange(L) * s
    i0 = np.floor(src).astype(int); w = (src - i0)[:, None].astype(np.float32)
    seg = np.asarray(d["x"][i0[0]:i0[-1] + 2], np.float32); j = i0 - i0[0]
    t0 = a / FRAME_RATE
    return (seg[j] * (1 - w) + seg[j + 1] * w, frame_targets((d["beats"] - t0) / s, (d["downs"] - t0) / s, L),
            d["mask"][np.round(src).astype(int)])


def collate(chunk, device):
    parts = [(d["x"][a:a + CROP], d["y"][a:a + CROP], d["mask"][a:a + CROP]) if s == 1 else stretched(d, a, s)
             for d, a, s in chunk]
    L = max(len(x) for x, _, _ in parts)
    X = np.zeros((len(chunk), L, N_IN), np.float32); Y = np.zeros((len(chunk), L, 3), np.float32)
    Mk = np.zeros((len(chunk), L), np.float32)
    for i, (x, y, m) in enumerate(parts):
        n = len(x)
        X[i, :n], Y[i, :n], Mk[i, :n] = x, y, m
    T = lambda z: torch.as_tensor(z, device=device)
    return T(X), T(Y), T(Mk)


def scores(data, acts, decode):
    """-> (ortalama vuruş F, ortalama ölçü başı F) etiketli aralıkta."""
    fb, fd = [], []
    for d, a in zip(data, acts):
        b, db = decode(a)[:2]
        fb.append(beat_scores(d["beats"], in_span(b, d["beats"]))[0])
        fd.append(onset_f(d["downs"], in_span(db, d["downs"])))
    return float(np.mean(fb)), float(np.mean(fd))


def main(ckpt, out, epochs, lr, bs, seed, limit, stretch):
    set_seed(seed); device = get_device(); rng = np.random.default_rng(seed)
    feat_dir = os.path.join(CACHE_DIR, "beat_feats", os.path.splitext(os.path.basename(ckpt))[0])
    fit_f, val_f = gaps_files("gaps_train")
    gs = lambda s: sorted(glob.glob(os.path.join(CACHE_DIR, s, "*.npz")))
    train = load(fit_f[:limit], feat_dir, True) + load((gs("train") + gs("train_comp"))[:limit], feat_dir, False)
    val, gs_val = load(val_f[:limit], feat_dir, True), load(gs("val")[:limit], feat_dir, False)
    n_fr = sum(int(d["mask"].sum()) for d in train)
    print(f"egitim {len(train)} kayit ({n_fr / FRAME_RATE / 3600:.2f} saat etiketli) | gaps_val {len(val)} | GS val {len(gs_val)} | {device}")
    net = BeatNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best = -1
    for ep in range(1, epochs + 1):
        net.train(); tl = n = 0
        items = crops(train, rng)
        sf = np.exp(rng.uniform(-np.log1p(stretch), np.log1p(stretch), len(items))) if stretch > 0 else np.ones(len(items))
        items = [(d, a, float(s)) for (d, a), s in zip(items, sf)]
        for i in range(0, len(items), bs):
            X, Y, Mk = collate(items[i:i + bs], device)
            ce = -(Y * torch.log_softmax(net(X), -1)).sum(-1)
            loss = (ce * Mk).sum() / Mk.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); n += 1
        sched.step()
        fb, fd = scores(val, [activations(net, d["x"], device) for d in val], peak_beats)
        gb, gd = scores(gs_val, [activations(net, d["x"], device) for d in gs_val], peak_beats)
        sc = (fb + fd) / 2
        mark = ""
        if sc > best:
            best = sc; mark = "  <- kaydedildi"
            torch.save({"model": net.state_dict(), "feats": ckpt}, ckpt_path(out))
        print(f"ep {ep:2d} | kayip {tl / n:.4f} | tepe secimi: gaps_val vurus F {fb:.3f} olcu basi F {fd:.3f} | "
              f"GS val {gb:.3f} / {gd:.3f}{mark}", flush=True)
    # DBN ayarları: en iyi epoch'ta gaps_val (vuruş F + ölçü başı F) / 2
    ck = torch.load(ckpt_path(out), map_location=device)
    net.load_state_dict(ck["model"])
    acts = [activations(net, d["x"], device) for d in val]
    print("DBN taramasi (gaps_val): vurus F / olcu basi F")
    res = []
    for p in DBN_GRID:
        fb, fd = scores(val, acts, lambda a: dbn_decode(a, **p))
        res.append(((fb + fd) / 2, p))
        print(f"  {p}: {fb:.3f} / {fd:.3f}", flush=True)
    ck["dbn"] = max(res, key=lambda r: r[0])[1]
    torch.save(ck, ckpt_path(out))
    print(f"secilen DBN {ck['dbn']} -> {ckpt_path(out)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="tabcrnn_deep2.pt", help="temsil önbelleğinin ses modeli")
    ap.add_argument("--out", default="beat_net.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="bolme basina en cok kayit (duman testi)")
    ap.add_argument("--stretch", type=float, default=0.25, help="tempo artirmasi: hiz carpani ~ log-U(1/(1+s), 1+s); 0 = kapali")
    a = ap.parse_args()
    main(a.ckpt, a.out, a.epochs, a.lr, a.batch_size, a.seed, a.limit, a.stretch)
