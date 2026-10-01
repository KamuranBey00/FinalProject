"""
Katman 3 (konsolide) — TabCNN eğitimi + değerlendirme + çıkarım.

Katman 2'den farkı: hedef "hangi telde hangi fret". Kayıp = 6 telin ayrı
cross-entropy toplamı. İki metrik: pitch F1 (Katman 2 ile kıyas) ve tab F1 (katı).

Konsolidasyon eklemeleri:
- Tekrarlanabilir seed
- Cosine LR decay (eğri platoya girmemişti; decay ile son epoch'larda oturur)
- Ayarlanabilir sınıf ağırlığı (--weight-scheme sqrt/inv/none, --weight-cap)
- Vektörize (hızlı) değerlendirme
- Batch'li çıkarım (eski hali kare-kare idi, çok yavaştı)
- Sessizlik-marjı: retrain'siz precision kaldıracı (Katman 2 eşik taramasının tab karşılığı)
- Zamansal düzeltme (mode-filtre) ile hayalet-nota temizliği

Çalıştırma:
    python train_tab.py --epochs 40
    python train_tab.py --sweep                         # en iyi sessizlik marjını bul
    python train_tab.py --demo data/cache/val/05_Rock1-130-A_solo.npz --margin 0.15 --smooth 5
"""

import argparse
import glob
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from gtab.torch_dataset import GuitarSetTab, _normalize, _window
from gtab.model import TabCNN
from gtab.tab_labels import n_tab_classes
from gtab.instrument import STANDARD_6
from gtab.decode import tab_to_transcription, render_ascii_tab

CACHE = "data/cache"
CONTEXT = 9
CKPT = "tabcnn.pt"
INSTR = STANDARD_6
PITCH_AXIS = 128


def set_seed(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------- Vektörize metrik ----------------
def _onehot_pitch(cls, tuning):
    """(N,S) sınıf -> (N,128) bool perde maskesi."""
    N, S = cls.shape
    oh = np.zeros((N, PITCH_AXIS), dtype=bool)
    for s in range(S):
        m = cls[:, s] > 0
        if m.any():
            pitches = tuning[s] + (cls[:, s][m] - 1)
            oh[np.where(m)[0], pitches] = True
    return oh


def _prf(tp, fp, fn):
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    return p, r, 2 * p * r / (p + r + 1e-9)


def evaluate(model, loader, device):
    model.eval()
    tuning = np.array(INSTR.tuning)
    ptp = pfp = pfn = ttp = tfp = tfn = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(-1).cpu().numpy()   # (B,S)
            gt = y.numpy()
            # tab tokenleri (tel+fret): sinif esitligi
            ap, ag = pred > 0, gt > 0
            tp_t = int(((pred == gt) & ap).sum())
            ttp += tp_t; tfp += int(ap.sum()) - tp_t; tfn += int(ag.sum()) - tp_t
            # perde (teli yok say)
            ohp, ohg = _onehot_pitch(pred, tuning), _onehot_pitch(gt, tuning)
            tp_p = int((ohp & ohg).sum())
            ptp += tp_p; pfp += int(ohp.sum()) - tp_p; pfn += int(ohg.sum()) - tp_p
    return _prf(ptp, pfp, pfn), _prf(ttp, tfp, tfn)


# ---------------- Eğitim ----------------
def train(epochs=40, batch_size=256, lr=1e-3, weight_scheme="inv", weight_cap=20.0, seed=1):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Cihaz:", device, "| weight:", weight_scheme, "cap", weight_cap)
    ncls = n_tab_classes(INSTR)

    tr_ds = GuitarSetTab(CACHE, "train", CONTEXT)
    va_ds = GuitarSetTab(CACHE, "val", CONTEXT)
    print(f"train {len(tr_ds)} | val {len(va_ds)} | sınıf/tel {ncls}")

    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    va_dl = DataLoader(va_ds, batch_size=batch_size, num_workers=2)

    model = TabCNN(INSTR.num_strings, ncls, CONTEXT).to(device)
    w = tr_ds.class_weights(ncls, scheme=weight_scheme, cap=weight_cap).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best = 0.0
    for ep in range(1, epochs + 1):
        model.train(); run = 0.0
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = sum(criterion(logits[:, s, :], y[:, s]) for s in range(INSTR.num_strings))
            loss.backward(); opt.step()
            run += loss.item() * x.size(0)
        sched.step()
        (pP, pR, pF), (tP, tR, tF) = evaluate(model, va_dl, device)
        print(f"ep {ep:2d} | loss {run/len(tr_ds):.3f} | lr {sched.get_last_lr()[0]:.1e} | "
              f"pitch F1 {pF:.3f} (P{pP:.2f}/R{pR:.2f}) | tab F1 {tF:.3f} (P{tP:.2f}/R{tR:.2f})")
        if tF > best:
            best = tF
            torch.save({"model": model.state_dict(), "context": CONTEXT, "n_classes": ncls}, CKPT)
    print(f"En iyi val tab F1: {best:.3f} -> {CKPT}")


# ---------------- Çıkarım (batch'li) ----------------
def predict_probs(model, cqt, context, device, batch=1024):
    """cqt (T,n_bins) -> (T, S, ncls) softmax olasılıkları. Batch'li = hızlı."""
    T = cqt.shape[0]
    wins = np.stack([_window(cqt, t, context) for t in range(T)])   # (T,ctx,n_bins)
    out = None
    with torch.no_grad():
        for i in range(0, T, batch):
            xb = torch.from_numpy(wins[i:i+batch]).unsqueeze(1).to(device)  # (b,1,ctx,n_bins)
            p = torch.softmax(model(xb), dim=-1).cpu().numpy()
            if out is None:
                out = np.zeros((T, p.shape[1], p.shape[2]), np.float32)
            out[i:i+batch] = p
    return out


def decode_with_margin(probs, margin=0.0):
    """
    Sessizlik-marjı: aktif (fret) tahmini SADECE en iyi fret olasılığı
    sessizlik olasılığını 'margin' kadar geçerse yapılır; aksi halde sessiz.
    margin=0 -> düz argmax. margin>0 -> daha çok sessiz -> precision artar.
    """
    silent = probs[:, :, 0]
    nonsilent = probs[:, :, 1:]
    best = nonsilent.argmax(-1) + 1
    best_p = nonsilent.max(-1)
    return np.where(best_p > silent + margin, best, 0).astype(np.int64)


def _time_smooth_probs(probs, k):
    if k <= 1:
        return probs
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(probs, size=k, axis=0, mode="nearest")


def transcribe_file(npz_path, ckpt=CKPT, margin=0.0, smooth=1, prob_smooth=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt, map_location=device)
    model = TabCNN(INSTR.num_strings, ck["n_classes"], ck["context"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()

    d = np.load(npz_path); cqt = _normalize(d["cqt"])
    probs = predict_probs(model, cqt, ck["context"], device)
    probs = _time_smooth_probs(probs, prob_smooth)
    preds = decode_with_margin(probs, margin)
    tr = tab_to_transcription(preds, INSTR, smooth=smooth)
    print(f"{npz_path}: {len(tr.notes)} nota (margin={margin}, smooth={smooth})\n")
    print(render_ascii_tab(tr))


# ---------------- Sessizlik-marjı taraması ----------------
def sweep_margin(ckpt=CKPT):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt, map_location=device)
    model = TabCNN(INSTR.num_strings, ck["n_classes"], ck["context"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    tuning = np.array(INSTR.tuning)

    files = sorted(glob.glob(os.path.join(CACHE, "val", "*.npz")))
    cache = []
    for f in files:
        d = np.load(f)
        cache.append((predict_probs(model, _normalize(d["cqt"]), ck["context"], device),
                      d["tab"].astype(np.int64)))

    # Marj NEGATIF de olabilir: <0 -> daha cok AKTIF (recall artar),
    # >0 -> daha cok SESSIZ (precision artar). Tek modelle tum P/R egrisi.
    print(f"{'marj':>6} {'tabP':>7} {'tabR':>7} {'tabF1':>7}")
    best = (0.0, 0.0)
    for margin in np.arange(-0.50, 0.51, 0.05):
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
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-scheme", type=str, default="inv", choices=["sqrt", "inv", "none"])
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
