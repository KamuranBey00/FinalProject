"""
Katman 3.6 — Viterbi tel ataması: GERÇEK VERİDE ölçüm ve ayar.

NEDEN BU DOSYA VAR:
Sentetik test Viterbi'nin faydasını kanıtlayamaz. Sebep: sentetik gürültü
bağımsızdır, orada greedy zaten optimaldir. Gerçek modelin hataları ise
SİSTEMATİKTİR (emin biçimde yanlış tel) -- çalınabilirlik önseli asıl orada
kazandırır. O yüzden faydayı gerçek val setinde ölçüyoruz.

DÜRÜSTLÜK GARANTİSİ:
w_transition = 0  ->  Viterbi birebir GREEDY'ye indirgenir (test edildi).
Yani tarama w_transition=0'ı en iyi bulursa, "Viterbi katkı vermedi" demektir
ve bunu saklamaz. Kendimizi kandırmayan bir kurulum.

Karşılaştırılan 3 yöntem (hepsi AYNI kare-seviye tab F1 metriğiyle):
  1) baseline : argmax + sessizlik marjı            (mevcut yöntem)
  2) greedy   : nota başına en yüksek olasılıklı (tel,fret)
  3) viterbi  : emisyon + çalınabilirlik, DP ile global optimum

Çalıştırma:
    python -m scripts.eval.eval_viterbi                      # CNN modeli (checkpoints/tabcnn.pt)
    python -m scripts.eval.eval_viterbi --model crnn         # CRNN modeli (checkpoints/tabcrnn.pt)
    python -m scripts.eval.eval_viterbi --model crnn --ckpt tabcrnn_comp.pt
    python -m scripts.eval.eval_viterbi --quick              # küçük grid (hızlı)
"""

import argparse
import glob
import os
import numpy as np

from gtab.config import FRAME_RATE
from gtab.core.instrument import STANDARD_6
from gtab.core.note_event import NoteEvent, Transcription
from gtab.data.torch_dataset import _normalize
from gtab.decoding.decode import decode_with_margin, decode_greedy, notes_to_frames
from gtab.decoding.viterbi import pitch_matrix, segment_notes, viterbi_assign
from gtab.evaluation.metrics import prf, tab_scores
from gtab.models.inference import load_model, predict_probs
from gtab.paths import CACHE_DIR
from gtab.utils import get_device

INSTR = STANDARD_6


# ----------------------------------------------------------------- ana akış
def main(kind="cnn", quick=False, ckpt=None):
    device = get_device()
    model, ck, kind = load_model(kind, device, ckpt)
    print(f"Model: {kind} | cihaz: {device}")

    files = sorted(glob.glob(os.path.join(CACHE_DIR, "val", "*.npz")))
    if not files:
        raise FileNotFoundError("data/cache/val bos.")
    data = []
    for f in files:
        d = np.load(f)
        probs = predict_probs(model, ck, kind, _normalize(d["cqt"]), device)
        data.append((probs, d["tab"].astype(np.int64)))
    print(f"{len(data)} val kaydi islendi.\n")

    # ---- 1) baseline: argmax + marj
    print("--- 1) BASELINE (argmax + sessizlik marji) ---")
    best_base = (None, 0.0)
    # ince adim: train_tab --sweep ile ayni cozunurluk (0.586 tavani kacirilmasin);
    # o egri 0.50'de tavan yapmadan bittigi icin 0.90'a kadar uzatildi.
    for margin in ([0.0, 0.25, 0.5] if quick else np.arange(0.0, 0.91, 0.05)):
        tp = fp = fn = 0
        for probs, gt in data:
            a, b, c = tab_scores(decode_with_margin(probs, margin), gt)
            tp += a; fp += b; fn += c
        p, r, f1 = prf(tp, fp, fn)
        print(f"  marj {margin:4.2f} | P {p:.3f} R {r:.3f} F1 {f1:.3f}")
        if f1 > best_base[1]:
            best_base = (margin, f1)
    print(f"  => en iyi baseline F1 {best_base[1]:.3f} (marj {best_base[0]})\n")

    # ---- 2) perde segmentasyonu + greedy  (esik taramasi)
    print("--- 2) GREEDY (nota basi en iyi tel) ---")
    thresholds = [0.3, 0.5, 0.7] if quick else [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    seg_cache = {}
    best_greedy = (None, 0.0)
    for thr in thresholds:
        tp = fp = fn = 0
        for i, (probs, gt) in enumerate(data):
            pm, lo = pitch_matrix(probs, INSTR)
            segs = segment_notes(pm, lo, thr)
            seg_cache[(thr, i)] = segs
            fr = notes_to_frames(segs, decode_greedy(probs, segs), len(gt))
            a, b, c = tab_scores(fr, gt)
            tp += a; fp += b; fn += c
        p, r, f1 = prf(tp, fp, fn)
        print(f"  esik {thr:3.1f} | P {p:.3f} R {r:.3f} F1 {f1:.3f}")
        if f1 > best_greedy[1]:
            best_greedy = (thr, f1)
    print(f"  => en iyi greedy F1 {best_greedy[1]:.3f} (esik {best_greedy[0]})\n")

    # ---- 3) viterbi grid
    print("--- 3) VITERBI (emisyon + calinabilirlik) ---")
    print("  NOT: w_tr=0 greedy'ye denktir; tarama onu secerse Viterbi katki vermemistir.")
    w_trs = [0.0, 0.3, 0.6] if quick else [0.0, 0.1, 0.2, 0.4, 0.6, 1.0, 1.5]
    w_strings = [0.35] if quick else [0.2, 0.35, 0.6]
    opens = [0.3] if quick else [0.0, 0.3, 0.6]
    hfps = [0.02] if quick else [0.0, 0.02, 0.05]

    best = {"f1": 0.0}
    thr_list = [best_greedy[0]] if quick else thresholds
    for thr in thr_list:
        for w_tr in w_trs:
            for w_st in w_strings:
                for od in opens:
                    for hfp in hfps:
                        tp = fp = fn = 0
                        for i, (probs, gt) in enumerate(data):
                            segs = seg_cache.get((thr, i))
                            if segs is None:
                                pm, lo = pitch_matrix(probs, INSTR)
                                segs = segment_notes(pm, lo, thr)
                                seg_cache[(thr, i)] = segs
                            asg = viterbi_assign(
                                segs, probs, INSTR, FRAME_RATE,
                                w_transition=w_tr, w_string=w_st,
                                open_discount=od, high_fret_penalty=hfp)
                            fr = notes_to_frames(segs, asg, len(gt))
                            a, b, c = tab_scores(fr, gt)
                            tp += a; fp += b; fn += c
                        p, r, f1 = prf(tp, fp, fn)
                        if f1 > best["f1"]:
                            best = {"f1": f1, "P": p, "R": r, "thr": thr,
                                    "w_tr": w_tr, "w_st": w_st, "od": od, "hfp": hfp}
    print(f"  => en iyi viterbi F1 {best['f1']:.3f} "
          f"(P {best['P']:.3f} R {best['R']:.3f})")
    print(f"     parametreler: esik={best['thr']} w_transition={best['w_tr']} "
          f"w_string={best['w_st']} open_discount={best['od']} high_fret_penalty={best['hfp']}\n")

    # ---- özet
    print("=" * 58)
    print(f"{'yontem':<28}{'tab F1':>10}")
    print("-" * 58)
    print(f"{'baseline (argmax+marj)':<28}{best_base[1]:>10.3f}")
    print(f"{'greedy (nota basi)':<28}{best_greedy[1]:>10.3f}")
    print(f"{'VITERBI':<28}{best['f1']:>10.3f}")
    print("=" * 58)
    if best["w_tr"] == 0.0:
        print("SONUC: w_transition=0 secildi -> calinabilirlik onseli KATKI VERMEDI.")
    else:
        gain = best["f1"] - max(best_base[1], best_greedy[1])
        print(f"SONUC: Viterbi en iyi alternatife gore {gain:+.3f} F1 kazandirdi.")


def demo(npz_path, kind="cnn", threshold=0.5, ckpt=None, **vit_kwargs):
    """Tek bir .npz kaydini Viterbi tel atamasiyla transkribe edip tab basar."""
    from gtab.decoding.decode import render_ascii_tab
    device = get_device()
    model, ck, kind = load_model(kind, device, ckpt)
    d = np.load(npz_path)
    probs = predict_probs(model, ck, kind, _normalize(d["cqt"]), device)

    pm, lo = pitch_matrix(probs, INSTR)
    segs = segment_notes(pm, lo, threshold)
    asg = viterbi_assign(segs, probs, INSTR, FRAME_RATE, **vit_kwargs)

    events = []
    for (a, b, pitch), (s, f) in zip(segs, asg):
        ev = NoteEvent(a / FRAME_RATE, b / FRAME_RATE, pitch)
        ev.string, ev.fret = s, f
        events.append(ev)
    tr = Transcription(notes=events).sort()
    print(f"{npz_path}: {len(tr.notes)} nota (Viterbi, esik={threshold})\n")
    print(render_ascii_tab(tr, INSTR))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn", choices=["cnn", "crnn"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--demo", default=None, help="tek .npz dosyasini Viterbi ile transkribe et")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--w-transition", type=float, default=0.6)
    ap.add_argument("--w-string", type=float, default=0.35)
    ap.add_argument("--open-discount", type=float, default=0.3)
    ap.add_argument("--high-fret-penalty", type=float, default=0.02)
    ap.add_argument("--ckpt", default=None, help="checkpoints/ altindaki model (varsayilan tabcnn.pt/tabcrnn.pt)")
    args = ap.parse_args()
    if args.demo:
        demo(args.demo, args.model, args.threshold, args.ckpt,
             w_transition=args.w_transition, w_string=args.w_string,
             open_discount=args.open_discount,
             high_fret_penalty=args.high_fret_penalty)
    else:
        main(args.model, args.quick, args.ckpt)
