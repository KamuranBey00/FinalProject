"""
Katman 3.5 — CRNN (CNN + BiLSTM) eğitimi + değerlendirme + çıkarım.

Neden: kare-bazlı TabCNN'in tavanı ~0.58 tab F1 idi; düşük precision yanlış-tel
karışıklığından (orta-nota tel sıçraması) geliyordu. BiLSTM zamansal tutarlılık
kurup bunu kaynağında azaltır ve nota sınırlarını stabilize eder (Katman 4 şartı).

Girdi/etiket/çözümleme altyapısı aynı; sadece model SEKANS işler.
Ortak yardımcılar (metrik, marj çözümleme, seed, çıkarım) gtab kütüphanesinden gelir.
--ckpt çıplak bir ad ise checkpoints/ altında aranır/kaydedilir.

Çalıştırma:
    python -m scripts.train.train_crnn --epochs 30
    python -m scripts.train.train_crnn --sweep
    python -m scripts.train.train_crnn --demo data/cache/val/05_Rock1-130-A_solo.npz --margin 0.1 --smooth 5

Katman 3.7 (veri genişletme) -- eski checkpoint'in ÜZERİNE YAZMAMAK için --ckpt:
    python -m scripts.train.train_crnn --splits train,train_comp --ckpt tabcrnn_comp.pt
    python -m scripts.train.train_crnn --sweep --ckpt tabcrnn_comp.pt --val-split val_comp
"""

import argparse
import glob
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from gtab.core.instrument import STANDARD_6
from gtab.data.torch_dataset import GuitarSetSeq, _normalize
from gtab.data.tab_labels import n_tab_classes
from gtab.decoding.decode import (tab_to_transcription, render_ascii_tab,
                                  decode_with_margin, time_smooth_probs)
from gtab.evaluation.metrics import prf, evaluate_tab_seq
from gtab.models.nets import TabCRNN
from gtab.models.inference import load_model, predict_probs
from gtab.paths import CACHE_DIR, ckpt_path
from gtab.utils import set_seed, get_device

CKPT = "tabcrnn.pt"
INSTR = STANDARD_6
CHUNK = 200
PAD = GuitarSetSeq.PAD


def evaluate(model, loader, device):
    return evaluate_tab_seq(model, loader, device, INSTR)


def train(epochs=30, batch_size=16, lr=1e-3, weight_scheme="inv", weight_cap=20.0, seed=1,
          splits=("train",), augment=False, ckpt=CKPT):
    set_seed(seed)
    device = get_device()
    ckpt = ckpt_path(ckpt)
    print("Cihaz:", device, "| chunk", CHUNK, "| weight", weight_scheme, weight_cap,
          "| splits", list(splits), "| augment", augment, "| ckpt", ckpt)
    ncls = n_tab_classes(INSTR)

    tr_ds = GuitarSetSeq(CACHE_DIR, list(splits), CHUNK, augment=augment)
    va_ds = GuitarSetSeq(CACHE_DIR, "val", CHUNK)      # model seçimi HER ZAMAN 05 solo (kıyaslanabilir)
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
            torch.save({"model": model.state_dict(), "n_classes": ncls}, ckpt)
    print(f"En iyi val tab F1: {best:.3f} -> {ckpt}")


def _probs(model, ck, cqt, device):
    """Tüm track -> (T, S, ncls) softmax (uzun kayıtlarda parça parça)."""
    return predict_probs(model, ck, "crnn", cqt, device)


def transcribe_file(npz_path, ckpt=CKPT, margin=0.0, smooth=1, prob_smooth=1):
    device = get_device()
    model, ck, _ = load_model("crnn", device, ckpt)
    d = np.load(npz_path); cqt = _normalize(d["cqt"])
    probs = time_smooth_probs(_probs(model, ck, cqt, device), prob_smooth)
    preds = decode_with_margin(probs, margin)
    tr = tab_to_transcription(preds, INSTR, smooth=smooth)
    print(f"{npz_path}: {len(tr.notes)} nota (margin={margin}, smooth={smooth})\n")
    print(render_ascii_tab(tr))


def sweep_margin(ckpt=CKPT, val_split="val"):
    device = get_device()
    model, ck, _ = load_model("crnn", device, ckpt)
    files = sorted(glob.glob(os.path.join(CACHE_DIR, val_split, "*.npz")))
    print(f"ckpt {ckpt} | {val_split}: {len(files)} kayit")
    cache = [(_probs(model, ck, _normalize(np.load(f)["cqt"]), device),
              np.load(f)["tab"].astype(np.int64)) for f in files]
    print(f"{'marj':>6} {'tabP':>7} {'tabR':>7} {'tabF1':>7}")
    best = (0.0, 0.0)
    for margin in np.arange(-0.30, 0.96, 0.05):   # 3.6 olcumunde egri 0.90'da hala yukseliyordu
        ttp = tfp = tfn = 0
        for probs, gt in cache:
            pred = decode_with_margin(probs, margin)
            ap, ag = pred > 0, gt > 0
            tp = int(((pred == gt) & ap).sum())
            ttp += tp; tfp += int(ap.sum()) - tp; tfn += int(ag.sum()) - tp
        p, r, f1 = prf(ttp, tfp, tfn)
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
    ap.add_argument("--ckpt", type=str, default=CKPT, help="kayit/yukleme yolu")
    ap.add_argument("--splits", type=str, default="train",
                    help="egitim splitleri, virgulle: train,train_comp")
    ap.add_argument("--augment", action="store_true", help="CQT-uzayi cogaltma (Katman 3.7)")
    ap.add_argument("--val-split", type=str, default="val", help="--sweep icin: val | val_comp")
    args = ap.parse_args()
    if args.sweep:
        sweep_margin(args.ckpt, args.val_split)
    elif args.demo:
        transcribe_file(args.demo, ckpt=args.ckpt, margin=args.margin, smooth=args.smooth,
                        prob_smooth=args.prob_smooth)
    else:
        train(epochs=args.epochs, lr=args.lr,
              weight_scheme=args.weight_scheme, weight_cap=args.weight_cap,
              splits=[s.strip() for s in args.splits.split(",") if s.strip()],
              augment=args.augment, ckpt=args.ckpt)
