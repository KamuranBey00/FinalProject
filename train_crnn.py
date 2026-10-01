"""
Katman 3.5 — CRNN (CNN + BiLSTM) eğitimi + değerlendirme + çıkarım.

Neden: kare-bazlı TabCNN'in tavanı ~0.58 tab F1 idi; düşük precision yanlış-tel
karışıklığından (orta-nota tel sıçraması) geliyordu. BiLSTM zamansal tutarlılık
kurup bunu kaynağında azaltır ve nota sınırlarını stabilize eder (Katman 4 şartı).

Girdi/etiket/çözümleme altyapısı aynı; sadece model SEKANS işler.
Ortak yardımcılar (metrik, marj çözümleme, seed) train_tab'den alınır.

Çalıştırma:
    python train_crnn.py --epochs 30
    python train_crnn.py --sweep
    python train_crnn.py --demo data/cache/val/05_Rock1-130-A_solo.npz --margin 0.1 --smooth 5
"""

import argparse
import glob
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from gtab.torch_dataset import GuitarSetSeq, _normalize
from gtab.model import TabCRNN
from gtab.tab_labels import n_tab_classes
from gtab.instrument import STANDARD_6
from gtab.decode import tab_to_transcription, render_ascii_tab
from train_tab import set_seed, _onehot_pitch, _prf, decode_with_margin, _time_smooth_probs

CACHE = "data/cache"
CKPT = "tabcrnn.pt"
INSTR = STANDARD_6
CHUNK = 200
PAD = GuitarSetSeq.PAD


def _flatten_valid(pred, gt):
    """(B,L,S) pred/gt -> sadece pad OLMAYAN kareleri (N,S) olarak düzleştir."""
    valid = gt[:, :, 0] != PAD                 # (B,L)
    return pred[valid], gt[valid]              # (N,S), (N,S)


def evaluate(model, loader, device):
    model.eval()
    tuning = np.array(INSTR.tuning)
    ptp = pfp = pfn = ttp = tfp = tfn = 0
    with torch.no_grad():
        for x, y, _ in loader:
            pred = model(x.to(device)).argmax(-1).cpu().numpy()   # (B,L,S)
            gt = y.numpy()
            p, g = _flatten_valid(pred, gt)                       # (N,S)
            ap, ag = p > 0, g > 0
            tp_t = int(((p == g) & ap).sum())
            ttp += tp_t; tfp += int(ap.sum()) - tp_t; tfn += int(ag.sum()) - tp_t
            ohp, ohg = _onehot_pitch(p, tuning), _onehot_pitch(g, tuning)
            tp_p = int((ohp & ohg).sum())
            ptp += tp_p; pfp += int(ohp.sum()) - tp_p; pfn += int(ohg.sum()) - tp_p
    return _prf(ptp, pfp, pfn), _prf(ttp, tfp, tfn)


def train(epochs=30, batch_size=16, lr=1e-3, weight_scheme="inv", weight_cap=20.0, seed=1):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Cihaz:", device, "| chunk", CHUNK, "| weight", weight_scheme, weight_cap)
    ncls = n_tab_classes(INSTR)

    tr_ds = GuitarSetSeq(CACHE, "train", CHUNK)
    va_ds = GuitarSetSeq(CACHE, "val", CHUNK)
    print(f"train {len(tr_ds)} chunk | val {len(va_ds)} chunk | sınıf/tel {ncls}")

    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    va_dl = DataLoader(va_ds, batch_size=batch_size, num_workers=2)

    model = TabCRNN(INSTR.num_strings, ncls).to(device)
    w = tr_ds.class_weights(ncls, scheme=weight_scheme, cap=weight_cap).to(device)
    # ignore_index=PAD: dolgu kareleri kayba girmez
    criterion = torch.nn.CrossEntropyLoss(weight=w, ignore_index=PAD)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best = 0.0
    for ep in range(1, epochs + 1):
        model.train(); run = 0.0; nseen = 0
        for x, y, _ in tr_dl:
            x, y = x.to(device), y.to(device)        # x:(B,1,L,nbins) y:(B,L,S)
            opt.zero_grad()
            logits = model(x)                         # (B,L,S,ncls)
            B, L, S, C = logits.shape
            loss = criterion(logits.reshape(B * L * S, C), y.reshape(B * L * S))
            loss.backward(); opt.step()
            run += loss.item() * B; nseen += B
        sched.step()
        (pP, pR, pF), (tP, tR, tF) = evaluate(model, va_dl, device)
        print(f"ep {ep:2d} | loss {run/nseen:.3f} | lr {sched.get_last_lr()[0]:.1e} | "
              f"pitch F1 {pF:.3f} (P{pP:.2f}/R{pR:.2f}) | tab F1 {tF:.3f} (P{tP:.2f}/R{tR:.2f})")
        if tF > best:
            best = tF
            torch.save({"model": model.state_dict(), "n_classes": ncls}, CKPT)
    print(f"En iyi val tab F1: {best:.3f} -> {CKPT}")


def _predict_probs_full(model, cqt, device):
    """Tüm track'i TEK sekans olarak işle -> (T, S, ncls) softmax."""
    with torch.no_grad():
        x = torch.from_numpy(cqt).unsqueeze(0).unsqueeze(0).to(device)   # (1,1,T,nbins)
        logits = model(x)[0]                                             # (T,S,ncls)
        return torch.softmax(logits, dim=-1).cpu().numpy()


def _load(ckpt, device):
    ck = torch.load(ckpt, map_location=device)
    model = TabCRNN(INSTR.num_strings, ck["n_classes"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    return model


def transcribe_file(npz_path, ckpt=CKPT, margin=0.0, smooth=1, prob_smooth=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load(ckpt, device)
    d = np.load(npz_path); cqt = _normalize(d["cqt"])
    probs = _time_smooth_probs(_predict_probs_full(model, cqt, device), prob_smooth)
    preds = decode_with_margin(probs, margin)
    tr = tab_to_transcription(preds, INSTR, smooth=smooth)
    print(f"{npz_path}: {len(tr.notes)} nota (margin={margin}, smooth={smooth})\n")
    print(render_ascii_tab(tr))


def sweep_margin(ckpt=CKPT):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load(ckpt, device)
    files = sorted(glob.glob(os.path.join(CACHE, "val", "*.npz")))
    cache = [(_predict_probs_full(model, _normalize(np.load(f)["cqt"]), device),
              np.load(f)["tab"].astype(np.int64)) for f in files]
    print(f"{'marj':>6} {'tabP':>7} {'tabR':>7} {'tabF1':>7}")
    best = (0.0, 0.0)
    for margin in np.arange(-0.30, 0.51, 0.05):
        ttp = tfp = tfn = 0
        for probs, gt in cache:
            pred = decode_with_margin(probs, margin)
            ap, ag = pred > 0, gt > 0
            tp = int(((pred == gt) & ap).sum())
            ttp += tp; tfp += int(ap.sum()) - tp; tfn += int(ag.sum()) - tp
        p, r, f1 = _prf(ttp, tfp, tfn)
        mark = ""
        if f1 > best[1]:
            best = (float(margin), float(f1)); mark = "  <-"
        print(f"{margin:6.2f} {p:7.3f} {r:7.3f} {f1:7.3f}{mark}")
    print(f"\nEn iyi sessizlik marjı: {best[0]:.2f}  (tab F1 {best[1]:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-scheme", type=str, default="inv", choices=["inv", "sqrt", "none"])
    ap.add_argument("--weight-cap", type=float, default=20.0)
    ap.add_argument("--demo", type=str, default=None)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--smooth", type=int, default=1)
    ap.add_argument("--prob-smooth", type=int, default=1)
    args = ap.parse_args()
    if args.sweep:
        sweep_margin()
    elif args.demo:
        transcribe_file(args.demo, margin=args.margin, smooth=args.smooth, prob_smooth=args.prob_smooth)
    else:
        train(epochs=args.epochs, lr=args.lr,
              weight_scheme=args.weight_scheme, weight_cap=args.weight_cap)
