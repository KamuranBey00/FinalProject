"""
Katman 5 Adım 1b — Vuruş ağı girdisi: ses modelinin donmuş gövde temsili (T, 256) float16 önbelleği.

    data/cache/beat_feats/<ckpt adı>/<id>.npy

Yalnız ritim etiketi olan kayıtlar (build_rhythm) ve eğitim / seçim bölmeleri: gaps_train (gaps_fit + gaps_val),
GuitarSet train, train_comp, val. gaps_test önbelleğe alınmaz (eval_rhythm temsili kendisi hesaplar).
Var olan dosyaların üstüne yazılmaz (yarıda kalırsa kaldığı yerden sürer).

Çalıştırma:
    python -m scripts.data.build_beat_feats              # ~1 GB
    python -m scripts.data.build_beat_feats --limit 2    # duman testi
"""

import argparse
import glob
import os

import numpy as np

from gtab.data.torch_dataset import _normalize
from gtab.models.beat_net import encode_features
from gtab.models.inference import load_model
from gtab.paths import CACHE_DIR
from gtab.utils import get_device

SPLITS = ("gaps_train", "train", "train_comp", "val")


def main(ckpt, limit):
    device = get_device()
    model, _, _ = load_model("crnn", device, ckpt)
    out_dir = os.path.join(CACHE_DIR, "beat_feats", os.path.splitext(os.path.basename(ckpt))[0])
    os.makedirs(out_dir, exist_ok=True)
    rhy = os.path.join(CACHE_DIR, "rhythm")
    n = frames = 0
    for split in SPLITS:
        files = [f for f in sorted(glob.glob(os.path.join(CACHE_DIR, split, "*.npz")))
                 if os.path.exists(os.path.join(rhy, os.path.basename(f)))][:limit]
        for k, f in enumerate(files, 1):
            out = os.path.join(out_dir, os.path.basename(f)[:-4] + ".npy")
            if os.path.exists(out):
                continue
            with np.load(f) as d:
                cqt = d["cqt"]
            z = encode_features(model, _normalize(cqt), device)
            np.save(out, z)
            n += 1; frames += len(z)
            print(f"[{split} {k}/{len(files)}] {os.path.basename(out)} {z.shape}", flush=True)
    print(f"{n} yeni kayit, {frames / 43.07 / 3600:.2f} saat -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="tabcrnn_deep2.pt")
    ap.add_argument("--limit", type=int, default=None, help="bolme basina en cok kayit (duman testi)")
    a = ap.parse_args()
    main(a.ckpt, a.limit)
