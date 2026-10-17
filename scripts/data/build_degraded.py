"""
Katman 6 Adım 2 — Bozulmuş CQT önbellekleri (eğitim artırması; README18). Her eğitim kaydının ham sesine rastgele 1–2 bozulma
zinciri (gtab/data/degrade.random_chain: yankı, telefon, gürültü, şarkı / konuşma) uygulanır, CQT yeniden hesaplanır. Etiketler
değişmez (aynı kareler); eğitimde `train_onset --degrade-root` temiz CQT'nin yerine olasılıkla bu kopyalardan birini koyar.

  data/cache/degraded/<k>/<split>/<id>.npz   (cqt float16, chain = uygulanan zincir; k = 0 .. copies-1)

Splitler: GuitarSet train + train_comp (mikrofon sesi), gaps_train (doğrulama kayıtları dahil: seçim ölçümü için; onlarda
ses bankasının şarkıcı-ayrık "eval" parçası kullanılır, eğitim kayıtlarında "train"). Önce: get_voices. Var olan dosya atlanır.
Disk: kopya başına ~0.7 GB. Süre: kopya başına ~30–40 dk.

Çalıştırma:
    python -m scripts.data.build_degraded                 # 2 kopya
    python -m scripts.data.build_degraded --limit 2       # duman testi
    python -m scripts.data.build_degraded --no-voices --out degraded_nv   # insan sesi yok (ses mix-vocals ile; README18)
"""

import argparse
import glob
import json
import os
import zlib

import numpy as np

from gtab.config import GUITARSET_DATA_HOME
from gtab.data.degrade import apply_chain, random_chain
from gtab.data.features import compute_cqt, load_audio
from gtab.data.gaps import GAPS_DIR, gaps_files
from gtab.paths import CACHE_DIR

OUT = os.path.join(CACHE_DIR, "degraded")      # --out ile değişir


def sources(limit):
    """-> [(split, id, wav, ses bankası parçası)]."""
    out = []
    for sp in ("train", "train_comp"):
        for f in sorted(glob.glob(os.path.join(CACHE_DIR, sp, "*.npz")))[:limit]:
            tid = os.path.splitext(os.path.basename(f))[0]
            out.append((sp, tid, os.path.join(GUITARSET_DATA_HOME, "audio_mono-mic", tid + "_mic.wav"), "train"))
    fit, val = gaps_files("gaps_train")
    for f, part in [(f, "train") for f in fit[:limit]] + [(f, "eval") for f in val[:limit]]:
        tid = os.path.splitext(os.path.basename(f))[0]
        out.append(("gaps_train", tid, os.path.join(GAPS_DIR, "audio", tid + ".wav"), part))
    return out


def main(copies, limit, voices=True):
    src = sources(limit)
    print(f"{len(src)} kayit x {copies} kopya -> {OUT}")
    for i, (sp, tid, wav, part) in enumerate(src, 1):
        todo = [k for k in range(copies) if not os.path.exists(os.path.join(OUT, str(k), sp, tid + ".npz"))]
        if not todo:
            continue
        with np.load(os.path.join(CACHE_DIR, sp, tid + ".npz")) as d:
            T, nb = d["cqt"].shape
        y = load_audio(wav)
        for k in todo:
            seed = zlib.crc32(tid.encode()) + 1000003 * k
            chain = random_chain(np.random.default_rng(seed), voices=voices)
            c = compute_cqt(apply_chain(y, chain, seed=seed, part=part))[:T]
            if len(c) < T:
                c = np.vstack([c, np.full((T - len(c), nb), c.min(), np.float32)])
            os.makedirs(os.path.join(OUT, str(k), sp), exist_ok=True)
            np.savez_compressed(os.path.join(OUT, str(k), sp, tid + ".npz"), cqt=c.astype(np.float16),
                                chain=json.dumps(chain))
        print(f"[{i}/{len(src)}] {sp:10s} {tid}  {[c[0] for c in chain]} ({part})", flush=True)
    print("Bitti.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="split basina ilk N kayit (duman testi)")
    ap.add_argument("--no-voices", action="store_true", help="zincirde sarki / konusma yok")
    ap.add_argument("--out", default="degraded", help="data/cache/<out>")
    a = ap.parse_args()
    OUT = os.path.join(CACHE_DIR, a.out)
    main(a.copies, a.limit, not a.no_voices)
