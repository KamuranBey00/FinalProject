"""
Katman 4 Faz 0 — Çıkarım yolu regresyon testi ("teknik modülü kapalıyken eski yol değişmedi").

Katman 4 pipeline'a yeni bir kafa/modül ekleyecek. Bu script ana modelin (varsayılan
`tabcrnn_deep2.pt`) çıkarım çıktısından bir PARMAK İZİ üretir (`gtab.evaluation.fingerprint`):
kafa başına ortalama/std + tab argmax sınıf histogramı. Faz 0'da referans kaydedilir; sonraki
fazlarda aynı komut referansla karşılaştırılır — fark varsa yeni modül eski yolu kırmıştır.

Tam metrik koşusu (eval_pitch / eval_hmm) dakikalar-saat sürer; parmak izi saniyeler sürer ve
kare düzeyinde karşılaştırdığı için metrikten daha hassastır.

Varsayılan cihaz CPU: GPU çekirdek seçimi koşudan koşuya ±0.002 oynayabiliyor (README14),
CPU'da çıktı deterministik. Veri, checkpoint ve kalibrasyon dosyalarına YAZMAZ.

Çalıştırma:
    python -m scripts.eval.check_inference_regression                 # referansla karsilastir
    python -m scripts.eval.check_inference_regression --save          # referansi ilk kez yaz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from gtab.data.torch_dataset import _normalize
from gtab.evaluation.fingerprint import HIST_TOL, TOL, compare, track_fingerprint
from gtab.models.inference import load_model, predict_heads
from gtab.models.stack import active
from gtab.paths import CACHE_DIR, ROOT, ckpt_path

REF_PATH = Path(ROOT) / "docs" / "baselines" / "inference_fingerprint.json"


def fingerprint(ckpt: str, split: str, n_files: int, device: str) -> dict:
    """Split'in ilk n kaydında tüm kafaların parmak izi."""
    model, ck, _ = load_model("crnn", device, ckpt)
    files = sorted(Path(CACHE_DIR, split).glob("*.npz"))[:n_files]
    if not files:
        raise FileNotFoundError(f"{CACHE_DIR}/{split} bos")
    tracks = {}
    for f in files:
        with np.load(f) as d:
            cqt = _normalize(d["cqt"])
        tracks[f.stem] = track_fingerprint(*predict_heads(model, cqt, device, with_pitch_onset=True))
    return {"ckpt": Path(ckpt).name, "split": split, "device": device,
            "onset_head": bool(ck.get("onset")), "deep": int(ck.get("deep") or 0),
            "pitch_onset_head": bool(ck.get("pitch_onset")), "kayitlar": tracks}


def save_reference(cur: dict, force: bool) -> int:
    if REF_PATH.exists() and not force:
        print(f"HATA: referans zaten var ({REF_PATH}); uzerine yazmak icin --force")
        return 1
    REF_PATH.parent.mkdir(parents=True, exist_ok=True)
    REF_PATH.write_text(json.dumps(cur, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Referans yazildi: {REF_PATH}")
    return 0


def main(a) -> int:
    cur = fingerprint(ckpt_path(a.ckpt), a.split, a.files, a.device)
    print(f"Parmak izi: {cur['ckpt']} | {cur['split']} ilk {a.files} kayit | cihaz {cur['device']} | "
          f"kafalar onset={cur['onset_head']} pitch_onset={cur['pitch_onset_head']} deep={cur['deep']}")
    for tid, s in cur["kayitlar"].items():
        print(f"  {tid}: kare {s['frames']} | tab guven {s['tab']['mean']:.6f} | "
              f"onset mean {s['onset']['mean']:.6f} | perde-onset mean {s['pitch_onset']['mean']:.6f}")
    if a.save:
        return save_reference(cur, a.force)
    if not REF_PATH.exists():
        print(f"Referans yok ({REF_PATH}); once --save ile olusturun.")
        return 1
    ref = json.loads(REF_PATH.read_text(encoding="utf-8"))
    bad = compare(ref, cur, a.tol, a.hist_tol)
    print("\n" + ("REGRESYON YOK: cikarim yolu referansla ayni." if not bad else
                  "REGRESYON:\n  " + "\n  ".join(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=active()["acoustic"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--files", type=int, default=2)
    ap.add_argument("--device", default="cpu", help="cpu (deterministik) | cuda")
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--hist-tol", type=float, default=HIST_TOL)
    ap.add_argument("--save", action="store_true", help="parmak izini referans olarak yaz")
    ap.add_argument("--force", action="store_true", help="mevcut referansin uzerine yaz")
    sys.exit(main(ap.parse_args()))
