"""
Katman 3.8a — GAPS -> önbellek (CQT + perde roll'u).

GuitarSet önbelleğiyle AYNI şema (aynı CQT ayarları, aynı 'frame'/'onset'
roll'ları) -> eval ve eğitim kodları iki veri setinde birebir aynı çalışır.
Fark: GAPS'te tel/perde etiketi yok (partisyon TAB'ı 3.8e'de), o yüzden
'tab' anahtarı YAZILMAZ; eğitimde GAPS sadece PERDE denetimi verir.

Bölünme:
  test  -> data/cache/gaps_test    (resmi test; sadece raporlama)
  train -> data/cache/gaps_train   (resmi train)
  Atanmamış (NaN) kayıtlar kullanılmaz.
  --disjoint: test icracılarının train kayıtları atlanır (performer-disjoint;
  sunumdaki değerlendirme ilkesi). Atlananlar konsola ve gaps_meta.csv'ye yazılır.

Çalıştırma:
    python -m scripts.data.get_gaps --splits train test
    python -m scripts.data.build_gaps --disjoint
"""

import argparse
import os

import numpy as np
import pandas as pd

from gtab.data.features import cqt_from_file
from gtab.data.labels import notes_to_rolls
from gtab.data.gaps import GAPS_DIR, CACHE_META, load_metadata, midi_to_transcription
from gtab.paths import CACHE_DIR


def build(disjoint=False, overwrite=False):
    meta = load_metadata()
    meta = meta[meta["split"].isin(["train", "test"])].copy()

    test_perf = set(meta.loc[meta.split == "test", "performer_name"].dropna())
    meta["skipped"] = disjoint & (meta.split == "train") & meta.performer_name.isin(test_perf)
    if disjoint:
        print(f"performer-disjoint: {int(meta.skipped.sum())} train kaydi atlaniyor "
              f"(test icracilari: {len(test_perf)})")

    rows = []
    for k, r in enumerate(meta.itertuples(), 1):
        if r.skipped:
            continue
        wav = os.path.join(GAPS_DIR, r.audio_path)
        mid = os.path.join(GAPS_DIR, r.midi_path)
        if not (os.path.exists(wav) and os.path.exists(mid)):
            print(f"  eksik dosya, atlandi: {r.id}")
            continue
        split = f"gaps_{r.split}"
        out_dir = os.path.join(CACHE_DIR, split)
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"{r.id}.npz")
        rows.append({"id": r.id, "split": split, "performer": r.performer_name})
        if os.path.exists(out) and not overwrite:
            continue

        cqt = cqt_from_file(wav)                         # 22050 Hz mono'ya indirilir
        tr = midi_to_transcription(mid)
        frame, onset = notes_to_rolls(tr, cqt.shape[0])
        # float16: GAPS uzun kayıtlar içeriyor; disk/RAM yarıya iner (dB hassasiyeti yeterli)
        np.savez_compressed(out, cqt=cqt.astype(np.float16), frame=frame, onset=onset)
        poly = (frame.sum(1) >= 2).mean()
        print(f"[{k}/{len(meta)}] {split:10s} {r.id}  cqt={cqt.shape}  nota={len(tr.notes)}  "
              f"polifonik kare=%{100 * poly:.0f}")

    pd.DataFrame(rows).to_csv(CACHE_META, index=False)
    print("Bitti.", {s: sum(r["split"] == s for r in rows) for s in ("gaps_train", "gaps_test")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--disjoint", action="store_true", help="test icracilarini train'den cikar")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    build(a.disjoint, a.overwrite)
