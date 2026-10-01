"""
Katman 3.8b — Perde/nota değerlendirmesi (sunum slayt 12'deki plan).

Aynı kod hem GAPS (klasik gitar, tel etiketi yok) hem GuitarSet için çalışır:
her iki önbellekte de 'frame' ve 'onset' roll'ları aynı şemada.

Metrikler:
  - NOTA seviyesi P/R/F1 (mir_eval, onset toleransı 50 ms, offset yok sayılır)
  - KARE seviyesi perde P/R/F1
  - POLİFONİ kırılımı: aynı anda 1 / 2 / 3 / 4+ gerçek nota çalan karelerde kare F1

Model çıktısı (tel × fret olasılıkları) perdeye, Katman 3.6'daki pipeline ile
indirgenir: pitch_matrix (tel üzerinde maksimum) + eşik + segment_notes.

Eşik seçimi: --select-split ile VERİLEN split'te seçilir, --splits'te raporlanır.
Test setinde eşik seçmek iyimser olur; varsayılan seçim split'i GuitarSet val.

Çalıştırma:
    python -m scripts.eval.eval_pitch --model crnn --ckpt tabcrnn_comp.pt --splits gaps_test val val_comp
"""

import argparse
import glob
import os

import mir_eval
import numpy as np

from gtab.config import FRAME_RATE
from gtab.core.instrument import STANDARD_6
from gtab.data.torch_dataset import _normalize
from gtab.decoding.viterbi import pitch_matrix, segment_notes
from gtab.evaluation.metrics import prf
from gtab.models.inference import load_model, predict_probs
from gtab.paths import CACHE_DIR
from gtab.utils import get_device

INSTR = STANDARD_6
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ----------------------------------------------------------------- yardımcılar
def roll_to_notes(frame, onset, lo):
    """Gerçek roll'lardan nota listesi: aktif koşular, onset işaretlerinde bölünür."""
    notes = []
    T, P = frame.shape
    for p in range(P):
        t = 0
        while t < T:
            if frame[t, p] > 0:
                s = t; t += 1
                while t < T and frame[t, p] > 0 and onset[t, p] == 0:
                    t += 1
                notes.append((s, t, lo + p))
            else:
                t += 1
    return notes


def _to_mir(notes):
    if not notes:
        return np.zeros((0, 2)), np.zeros(0)
    iv = np.array([[a / FRAME_RATE, max(b, a + 1) / FRAME_RATE] for a, b, _ in notes])
    hz = mir_eval.util.midi_to_hz(np.array([p for _, _, p in notes], float))
    return iv, hz


def track_scores(probs, frame, onset, thr):
    """-> dict(frame tp/fp/fn, note tp-sayıları, polifoni grupları)"""
    pm, lo = pitch_matrix(probs, INSTR)
    n = min(len(pm), len(frame))
    pm, frame, onset = pm[:n], frame[:n] > 0, onset[:n] > 0
    pred = pm > thr

    tp = int((pred & frame).sum()); fp = int((pred & ~frame).sum()); fn = int((~pred & frame).sum())

    est = segment_notes(pm, lo, thr)
    ref = roll_to_notes(frame, onset, lo)
    ri, rp = _to_mir(ref); ei, ep = _to_mir(est)
    if len(ref) and len(est):
        matched = len(mir_eval.transcription.match_notes(ri, rp, ei, ep, onset_tolerance=0.05,
                                                          offset_ratio=None))
    else:
        matched = 0

    poly = {}
    k = frame.sum(1)
    for name, mask in (("1", k == 1), ("2", k == 2), ("3", k == 3), ("4+", k >= 4)):
        if mask.any():
            pf, ff = pred[mask], frame[mask]
            poly[name] = (int((pf & ff).sum()), int((pf & ~ff).sum()), int((~pf & ff).sum()))
    return {"frame": (tp, fp, fn), "note": (matched, len(est), len(ref)), "poly": poly}


def aggregate(results):
    F = np.zeros(3); N = np.zeros(3); poly = {}
    for r in results:
        F += r["frame"]; N += r["note"]
        for g, v in r["poly"].items():
            poly[g] = poly.get(g, np.zeros(3)) + v
    m, ne, nr = N
    note = (m / max(ne, 1), m / max(nr, 1))
    note = note + (2 * note[0] * note[1] / max(note[0] + note[1], 1e-9),)
    return {"frame": prf(*F), "note": note, "poly": {g: prf(*v) for g, v in poly.items()}}


def load_split(split, model, ck, kind, device, limit=None):
    files = sorted(glob.glob(os.path.join(CACHE_DIR, split, "*.npz")))[:limit]
    if not files:
        raise FileNotFoundError(f"{CACHE_DIR}/{split} bos.")
    data = []
    for f in files:
        with np.load(f) as d:
            probs = predict_probs(model, ck, kind, _normalize(d["cqt"]), device)
            data.append((probs, d["frame"], d["onset"]))
    return data


def evaluate_split(data, thresholds=THRESHOLDS):
    return {thr: aggregate([track_scores(p, fr, on, thr) for p, fr, on in data])
            for thr in thresholds}


# ----------------------------------------------------------------- ana akış
def main(kind, ckpt, splits, select_split, limit=None):
    device = get_device()
    model, ck, kind = load_model(kind, device, ckpt)
    print(f"Model: {kind} ({ckpt or 'varsayilan'}) | cihaz: {device}")

    sel = evaluate_split(load_split(select_split, model, ck, kind, device, limit))
    thr = max(sel, key=lambda t: sel[t]["note"][2])
    print(f"\nEsik secimi '{select_split}' uzerinde (nota F1): esik={thr} "
          f"(nota F1 {sel[thr]['note'][2]:.3f})")

    summary = []
    for split in splits:
        res = evaluate_split(load_split(split, model, ck, kind, device, limit))
        print(f"\n=== {split} ===")
        print(f"  {'esik':>5} | {'nota P':>7} {'R':>6} {'F1':>6} | {'kare P':>7} {'R':>6} {'F1':>6}")
        for t, r in res.items():
            mark = "  <- secilen" if t == thr else ""
            print(f"  {t:5.2f} | {r['note'][0]:7.3f} {r['note'][1]:6.3f} {r['note'][2]:6.3f} | "
                  f"{r['frame'][0]:7.3f} {r['frame'][1]:6.3f} {r['frame'][2]:6.3f}{mark}")
        r = res[thr]
        print("  polifoni (kare F1, secilen esik): " +
              "  ".join(f"{g}: {v[2]:.3f}" for g, v in sorted(r["poly"].items())))
        summary.append((split, r))

    print("\n" + "=" * 66)
    print(f"{'split':<14}{'nota F1':>10}{'kare F1':>10}   polifoni 1 / 2 / 3 / 4+")
    print("-" * 66)
    for split, r in summary:
        pol = " / ".join(f"{r['poly'][g][2]:.2f}" if g in r["poly"] else "  - "
                         for g in ("1", "2", "3", "4+"))
        print(f"{split:<14}{r['note'][2]:>10.3f}{r['frame'][2]:>10.3f}   {pol}")
    print("=" * 66)
    print(f"esik={thr} ('{select_split}' uzerinde secildi). Not: segment_notes ayni perdenin "
          "art arda tekrarini tek nota sayar -> nota F1 icin alt sinir.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="crnn", choices=["cnn", "crnn"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--splits", nargs="+", default=["gaps_test", "val", "val_comp"])
    ap.add_argument("--select-split", default="val", help="esigin secildigi split (test OLMAMALI)")
    ap.add_argument("--limit", type=int, default=None, help="split basina en fazla kayit (hizli deneme)")
    a = ap.parse_args()
    main(a.model, a.ckpt, a.splits, a.select_split, a.limit)
