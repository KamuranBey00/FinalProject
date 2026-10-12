"""
Katman 3.13 — GAPS'te TEL ölçümü (partisyon TAB etiketleriyle; build_gaps_tab çıktısı).

1) Oracle tel doğruluğu: gerçek notalar (MIDI perdesi + sınırları) verilir, model yalnız teli seçer
   (adaylar içinde ortalama tab olasılığı en yüksek konum = eval_hmm 'ses / greedy'). Yalnız etiketli notalar.
   Karşılaştırma: naif (en düşük perde) ve GuitarSet'teki oracle (deep2: 0.878).
   -> Partisyon TAB'ı icracının çaldığı teli gösteriyorsa model ile uyum GuitarSet'e yakın çıkmalı.
2) Uçtan uca kare tab F1 — yalnız 'known' karelerde (karedeki tüm notalar etiketli). Çözümleme ve perde eşiği
   kalibrasyon dosyasındaki gaps_val seçimi (test setine bakılmaz); tel ataması greedy.

Çalıştırma:
    python -m scripts.eval.eval_gaps_tab --ckpt tabcrnn_deep2.pt --load-cal checkpoints/tabcrnn_deep2.cal.json
"""

import argparse
import glob
import os

import numpy as np

from gtab.config import FRAME_RATE
from gtab.core.instrument import STANDARD_6
from gtab.data.gaps import GAPS_TAB_DIR, gaps_files
from gtab.data.torch_dataset import _normalize
from gtab.decoding.decode import notes_to_frames
from gtab.decoding.viterbi import energy_rise, segment
from gtab.evaluation.metrics import prf, tab_scores
from gtab.evaluation.pitch_eval import load_selection
from gtab.models.inference import load_model, predict_heads
from gtab.paths import CACHE_DIR
from gtab.utils import get_device

INSTR = STANDARD_6
TAB_DIR = GAPS_TAB_DIR


def split_files(split):
    if split == "gaps_val":
        files = gaps_files("gaps_train")[1]
    elif split == "gaps_fit":
        files = gaps_files("gaps_train")[0]
    else:
        files = sorted(glob.glob(os.path.join(CACHE_DIR, split, "*.npz")))
    return [(f, os.path.join(TAB_DIR, os.path.basename(f))) for f in files
            if os.path.exists(os.path.join(TAB_DIR, os.path.basename(f)))]


def greedy(probs, a, b, pitch):
    """Notanın karelerinde ortalama tab olasılığı en yüksek (tel, fret)."""
    c = INSTR.pitch_to_positions(pitch)
    return max(c, key=lambda sf: probs[a:max(b, a + 1), sf[0], sf[1] + 1].mean()) if c else (None, None)


def main(ckpt, load_cal, splits):
    device = get_device()
    model, ck, kind = load_model("crnn", device, ckpt)
    _, thr, ot, dec = load_selection(load_cal)["gaps_val"]
    print(f"Model: {ckpt} | cihaz: {device} | cozumleme (gaps_val secimi): esik {thr}, onset@{ot}, {dec}")
    rows = []
    for split in splits:
        pairs = split_files(split)
        acc = dict(n=0, ok=0, nm=0, okm=0, naive=0, nb=0); tp = fp = fn = 0
        for f, lf in pairs:
            with np.load(f) as d, np.load(lf) as L:
                probs, ons, offs, pons = predict_heads(model, _normalize(d["cqt"]), device, with_pitch_onset=True)
                rise = energy_rise(d["cqt"], INSTR)
                notes, string, fret = L["notes"], L["string"], L["fret"]
                tab, known = L["tab"], L["known"]
            n = min(len(probs), len(tab))
            # 1) oracle tel
            for (st, en, p), s, fr in zip(notes, string, fret):
                if s < 0:
                    continue
                a = int(round(st * FRAME_RATE)); b = int(round(en * FRAME_RATE))
                if a >= n:
                    continue
                ps, pf = greedy(probs, a, min(b, n), int(p))
                multi = len(INSTR.pitch_to_positions(int(p))) > 1
                ok = (ps, pf) == (int(s), int(fr))
                acc["n"] += 1; acc["ok"] += ok; acc["nm"] += multi; acc["okm"] += ok and multi
                acc["naive"] += min(INSTR.pitch_to_positions(int(p)), key=lambda sf: sf[1]) == (int(s), int(fr))
                acc["nb"] += (not ok) and ps is not None and abs(ps - int(s)) == 1
            # 2) uçtan uca, yalnız bilinen kareler
            pp = pons if dec.get("onset_source") == "pitch" else None
            segs = segment(probs[:n], ons[:n], thr, ot, INSTR, rise=rise[:n], offsets=offs[:n],
                           pitch_onsets=None if pp is None else pp[:n], **dec)
            pred = notes_to_frames(segs, [greedy(probs, a, b, p) for a, b, p in segs], n)
            k = known[:n]
            a_, b_, c_ = tab_scores(pred[k], tab[:n][k]); tp += a_; fp += b_; fn += c_
        P, R, F = prf(tp, fp, fn)
        rows.append((split, len(pairs), acc, (P, R, F)))

    print("\n" + "=" * 96)
    print(f"{'split':<10}{'kayit':>6}{'nota':>8} | {'oracle tel':>10}{'cok-adayli':>11}{'naif':>7}{'hata komsu tel':>15}"
          f" | {'tab P':>6}{'R':>6}{'F1':>7} (bilinen kare)")
    print("-" * 96)
    for split, nf, a, (P, R, F) in rows:
        err = a["n"] - a["ok"]
        print(f"{split:<10}{nf:>6}{a['n']:>8} | {a['ok'] / max(a['n'], 1):>10.3f}{a['okm'] / max(a['nm'], 1):>11.3f}"
              f"{a['naive'] / max(a['n'], 1):>7.3f}{100 * a['nb'] / max(err, 1):>14.1f}% | {P:>6.3f}{R:>6.3f}{F:>7.3f}")
    print("=" * 96)
    print("Referans (GuitarSet solo val, deep2): oracle tel 0.878, naif 0.206, tab F1 0.777.")
    print("Oracle tel GAPS'te belirgin dusukse: ya model naylonda teli ayiramiyor ya da partisyon TAB'i icracinin")
    print("caldigi teli gostermiyor -> hatalarin komsu tel orani ve kayit bazli dagilim ayirt eder (README14).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="tabcrnn_deep2.pt")
    ap.add_argument("--load-cal", default="checkpoints/tabcrnn_deep2.cal.json")
    ap.add_argument("--splits", nargs="+", default=["gaps_val", "gaps_test"])
    a = ap.parse_args()
    main(a.ckpt, a.load_cal, a.splits)
