"""
Katman 2.5 — Karar eşiği taraması (retrain YOK).

Model logit üretir; biz sigmoid ile olasılığa çevirip bir eşikle "var/yok" deriz.
Eşik 0.5 keyfi bir seçim. Precision düşük / recall yüksekse eşiği yükseltmek
yanlış pozitifleri kırpar. Bu script val setinde tüm eşikleri deneyip en iyi
F1'i vereni bulur. Bulunan eşiği çözümlemede (frames_to_notes) kullanacağız.

Çalıştırma:
    python sweep_threshold.py
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from gtab.torch_dataset import GuitarSetFrames
from gtab.model import PitchCNN
from gtab.labels import n_pitches

CACHE = "data/cache"
CKPT = "pitchcnn.pt"


def collect_probs(model, loader, device):
    probs, targets = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            p = torch.sigmoid(model(x.to(device))).cpu().numpy()
            probs.append(p)
            targets.append(y.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CKPT, map_location=device)
    model = PitchCNN(n_pitches=n_pitches(), context=ck["context"]).to(device)
    model.load_state_dict(ck["model"])

    val = GuitarSetFrames(CACHE, "val", context=ck["context"])
    loader = DataLoader(val, batch_size=512, num_workers=2)
    probs, targets = collect_probs(model, loader, device)
    tgt = targets > 0.5

    print(f"{'esik':>5} {'P':>7} {'R':>7} {'F1':>7}")
    best = (0.5, 0.0)
    for thr in np.arange(0.30, 0.96, 0.05):
        pred = probs > thr
        tp = (pred & tgt).sum()
        fp = (pred & ~tgt).sum()
        fn = (~pred & tgt).sum()
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        mark = ""
        if f1 > best[1]:
            best = (float(thr), float(f1)); mark = "  <-"
        print(f"{thr:5.2f} {prec:7.3f} {rec:7.3f} {f1:7.3f}{mark}")

    print(f"\nEn iyi eşik: {best[0]:.2f}  (F1 {best[1]:.3f})")


if __name__ == "__main__":
    main()
