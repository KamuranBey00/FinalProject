"""
Katman 6 Adım 0 — Dayanıklılık ölçümü (eğitim yok; README18). Aktif taban modeli (gtab/models/stack.py) ve kalibrasyonu
(gaps_val seçimi) SABİT; yalnız kayıt koşulu değişir. Her GAPS doğrulama kaydının ham sesine bozulma uygulanır
(gtab/data/degrade.py: yankı, telefon, gürültü), CQT yeniden hesaplanır, etiketler aynı kalır (bozulmalar uzunluğu korur).

Rapor (koşul başına): nota P / R / F1 (50 ms), @100 ms, kare F1, polifoni düzeyine göre kare F1 (1 / 2 / 3 / 4+),
<100 ms tekrar kaçma, temize göre fark. "temiz" satırı eval_pitch gaps_val ile aynı olmalı (tutarlılık kontrolü).

Çalıştırma:
    python -m scripts.eval.eval_robust                       # gaps_val, 6 koşul (~15 dk)
    python -m scripts.eval.eval_robust --limit 2             # duman testi
    python -m scripts.eval.eval_robust --conditions temiz telefon
"""

import argparse
import os

from gtab.core.instrument import STANDARD_6
from gtab.data.degrade import CONDITIONS, degrade
from gtab.data.features import compute_cqt, load_audio
from gtab.data.gaps import GAPS_DIR, gaps_files
from gtab.data.torch_dataset import _normalize
from gtab.decoding.viterbi import energy_rise
from gtab.evaluation.pitch_eval import aggregate, load_selection, track_scores
from gtab.models.inference import load_model, predict_heads
from gtab.models.stack import active
from gtab.paths import ckpt_path
from gtab.utils import get_device

import numpy as np

INSTR = STANDARD_6
POLY = ("1", "2", "3", "4+")


def main(conditions, limit):
    s, device = active(), get_device()
    model, _, _ = load_model("crnn", device, s["acoustic"])
    _, thr, ot, dec = load_selection(ckpt_path(s["cal"]))["gaps_val"]
    files = gaps_files("gaps_train")[1][:limit]
    print(f"taban {s['acoustic']} | kalibrasyon {s['cal']} [gaps_val] | {len(files)} kayit | kosullar {conditions}")
    res = {c: [] for c in conditions}
    for k, f in enumerate(files):
        tid = os.path.splitext(os.path.basename(f))[0]
        y = load_audio(os.path.join(GAPS_DIR, "audio", tid + ".wav"))
        with np.load(f) as d:
            frame, onset = d["frame"], d["onset"]
        for c in conditions:
            kind, params = CONDITIONS[c]
            cqt = compute_cqt(degrade(y, kind, params, seed=k))[:len(frame)]
            probs, ons, offs, pons = predict_heads(model, _normalize(cqt), device, with_pitch_onset=True)
            res[c].append(track_scores(probs, ons, frame, onset, thr, ot, dec, energy_rise(cqt, INSTR), offs, pons))
        print(f"  [{k + 1}/{len(files)}] {tid}", flush=True)

    agg = {c: aggregate(r) for c, r in res.items()}
    base = agg.get("temiz")
    print(f"\n{'kosul':<14}{'nota P':>8}{'R':>7}{'F1':>7}{'fark':>8}{'@100ms':>8}{'kare F1':>9}   "
          f"polifoni kare F1 {' / '.join(POLY):<14}{'<100ms kacma':>13}")
    for c, r in agg.items():
        d = f"{r['note'][2] - base['note'][2]:+.3f}" if base else "   -"
        pol = " / ".join(f"{r['poly'][g][2]:.2f}" if g in r["poly"] else " - " for g in POLY)
        n, miss = r["rep"]["<100"]
        print(f"{c:<14}{r['note'][0]:>8.3f}{r['note'][1]:>7.3f}{r['note'][2]:>7.3f}{d:>8}{r['note100'][2]:>8.3f}"
              f"{r['frame'][2]:>9.3f}   {pol:<31}{miss:>8.1%} (n={n})")
    print("\nKarar kurali: docs/devlog/README18.md (Adim 0).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--limit", type=int, default=None, help="ilk N gaps_val kaydi (duman testi)")
    a = ap.parse_args()
    main(a.conditions, a.limit)
