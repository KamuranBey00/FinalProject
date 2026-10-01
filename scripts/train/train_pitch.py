"""
Katman 2 — Eğitim döngüsü.

Önbellekten okur, PitchCNN'i eğitir, val'da kare-seviye F1 ölçer, en iyi modeli
kaydeder. Sonunda bir val kaydını uçtan uca transkribe edip ASCII tab basar --
ilk gerçek çıktımız.

Çalıştırma:
    python -m scripts.train.train_pitch --epochs 15
    python -m scripts.train.train_pitch --demo data/cache/val/05_Rock1-130-A_solo.npz   # sadece bir dosyayi transkribe et
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from gtab.data.torch_dataset import GuitarSetFrames, _normalize
from gtab.data.labels import n_pitches
from gtab.decoding.decode import frames_to_notes, assign_naive, render_ascii_tab
from gtab.models.nets import PitchCNN
from gtab.paths import CACHE_DIR as CACHE, ckpt_path

CONTEXT = 9
CKPT = ckpt_path("pitchcnn.pt")


def frame_f1(logits, targets, threshold=0.0):
    """logits>threshold (=olasilik>0.5) ile hedefleri karsilastirir -> P, R, F1."""
    pred = (logits > threshold)
    tgt = (targets > 0.5)
    tp = (pred & tgt).sum().item()
    fp = (pred & ~tgt).sum().item()
    fn = (~pred & tgt).sum().item()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return prec, rec, f1


def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x) > 0.0
            tgt = y > 0.5
            tp += (pred & tgt).sum().item()
            fp += (pred & ~tgt).sum().item()
            fn += (~pred & tgt).sum().item()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return prec, rec, f1


def train(epochs=15, batch_size=256, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Cihaz:", device)

    train_ds = GuitarSetFrames(CACHE, "train", context=CONTEXT)
    val_ds = GuitarSetFrames(CACHE, "val", context=CONTEXT)
    print(f"train ornek: {len(train_ds)} | val ornek: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=2)

    model = PitchCNN(n_pitches=n_pitches(), context=CONTEXT).to(device)
    pos_weight = train_ds.pos_weight().to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_f1 = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_ds)
        prec, rec, f1 = evaluate(model, val_dl, device)
        print(f"epoch {epoch:2d} | loss {train_loss:.4f} | val P {prec:.3f} R {rec:.3f} F1 {f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model": model.state_dict(), "context": CONTEXT}, CKPT)
    print(f"En iyi val F1: {best_f1:.3f} -> {CKPT}")


def transcribe_file(npz_path, ckpt=CKPT):
    """Bir onbellek dosyasini transkribe edip tab basar."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt, map_location=device)
    model = PitchCNN(n_pitches=n_pitches(), context=ck["context"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()

    d = np.load(npz_path)
    cqt = _normalize(d["cqt"])
    T = cqt.shape[0]
    half = ck["context"] // 2
    probs = np.zeros((T, n_pitches()), dtype=np.float32)
    with torch.no_grad():
        for t in range(T):
            lo, hi = t - half, t + half + 1
            win = np.zeros((ck["context"], cqt.shape[1]), dtype=np.float32)
            a, b = max(0, lo), min(T, hi)
            win[a - lo:b - lo] = cqt[a:b]
            x = torch.from_numpy(win).unsqueeze(0).unsqueeze(0).to(device)
            probs[t] = torch.sigmoid(model(x)).cpu().numpy()[0]

    tr = assign_naive(frames_to_notes(probs))
    print(f"{npz_path}: {len(tr.notes)} nota bulundu\n")
    print(render_ascii_tab(tr))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--demo", type=str, default=None)
    args = ap.parse_args()
    if args.demo:
        transcribe_file(args.demo)
    else:
        train(epochs=args.epochs)
