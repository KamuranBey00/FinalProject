"""
Katman 3.6b — Öğrenilen geçiş modeliyle tel ataması: TEŞHİS + ÖLÇÜM.

ADIMLAR
  0) Geçiş modelini TRAIN etiketlerinden öğren (ses yok, sadece tel/fret dizileri)
     -> transitions.npz. Val oyuncusu (05) hiç görülmez.
  1) TEŞHİS (oracle notalar): val'in GERÇEK nota sınırları/perdeleri verilir,
     sadece "hangi tel?" sorusu ölçülür. Segmentasyon hatası devre dışı kalır.
     - el yapımı maliyet gerçekten gitaristin yolunu mu tercih ediyor?
     - önsel tek başına / ses tek başına / ikisi birlikte tel doğruluğu
  2) UÇTAN UCA: eval_viterbi ile AYNI kare-seviye tab F1
     baseline (argmax+marj) · greedy · öğrenilen-Viterbi

DÜRÜSTLÜK: emisyon adaylar üstünde normalize -> w_transition=0 BİREBİR greedy.
Tarama w_transition=0 seçerse "öğrenilen önsel katkı vermedi" yazılır.

Çalıştırma:
    python -m scripts.eval.eval_hmm --model crnn            # önerilen (en iyi akustik model)
    python -m scripts.eval.eval_hmm --model cnn
    python -m scripts.eval.eval_hmm --model crnn --quick
    python -m scripts.eval.eval_hmm --refit                 # geçiş modelini yeniden öğren
    python -m scripts.eval.eval_hmm --model crnn --ckpt tabcrnn_aug.pt   # Katman 3.7 modeli
    python -m scripts.eval.eval_hmm --model crnn --demo data/cache/val/05_Rock1-130-A_solo.npz \
        --threshold 0.8 --w-transition 1.0
"""

import argparse
import glob
import os
import numpy as np

from gtab.config import FRAME_RATE
from gtab.core.instrument import STANDARD_6
from gtab.core.note_event import NoteEvent, Transcription
from gtab.data.torch_dataset import _normalize
from gtab.decoding.decode import decode_with_margin, notes_to_frames
from gtab.decoding.viterbi import energy_rise, pitch_matrix, segment, viterbi_assign, transition_cost
from gtab.decoding.transitions import (TransitionModel, notes_from_tab, build_lattice,
                                       decode_lattice)
from gtab.evaluation.metrics import prf, tab_scores
from gtab.models.inference import load_model, predict_probs, predict_heads
from gtab.paths import CACHE_DIR, ckpt_path
from gtab.utils import get_device

INSTR = STANDARD_6
TM_PATH = ckpt_path("transitions.npz")


def _label_seqs(split):
    files = sorted(glob.glob(os.path.join(CACHE_DIR, split, "*.npz")))
    if not files:
        raise FileNotFoundError(f"{CACHE_DIR}/{split} bos.")
    seqs = []
    for f in files:
        with np.load(f) as d:
            seqs.append(notes_from_tab(d["tab"].astype(np.int64), INSTR))
    return files, seqs


def get_transition_model(refit=False):
    if os.path.exists(TM_PATH) and not refit:
        print(f"Gecis modeli yuklendi: {TM_PATH}")
        return TransitionModel.load(TM_PATH, INSTR)
    _, train_seqs = _label_seqs("train")
    print(f"Gecis modeli ogreniliyor: {len(train_seqs)} train kaydi, "
          f"{sum(len(s) for s in train_seqs)} nota")
    tm = TransitionModel(INSTR).fit(train_seqs)
    tm.save(TM_PATH)
    print(f"  -> {TM_PATH}")
    return tm


# ----------------------------------------------------------------- 1) teşhis
def diagnose(tm, val_files, val_seqs, probs_list, quick=False):
    print("\n" + "=" * 64)
    print("1) TESHIS - oracle notalar (gercek perde + sinirlar), sadece TEL secimi")
    print("=" * 64)
    print(f"  val NLL (ogrenilen onsel, cok-adayli notalar): {tm.nll(val_seqs):.3f}  "
          f"| uniform: {_uniform_nll(val_seqs):.3f}  (dusuk = iyi)")

    gt_all, multi = [], []
    res = {}

    def acc(name, preds):
        res[name] = preds

    lat_prior, lat_full = [], []
    for seq, probs in zip(val_seqs, probs_list):
        notes = [(on, off, p) for (on, off, p, s, f) in seq]
        gt_all.append([(s, f) for (*_, s, f) in seq])
        multi.append([len(INSTR.pitch_to_positions(p)) > 1 for (_, _, p) in notes])
        lat_prior.append(build_lattice(notes, None, tm, time_unit="sec"))
        lat_full.append(build_lattice(notes, probs, tm, time_unit="sec"))

    # a) naif: en düşük perde
    acc("naif (en dusuk perde)",
        [[min(INSTR.pitch_to_positions(p), key=lambda sf: sf[1]) for (_, _, p, *_ ) in seq]
         for seq in val_seqs])
    # b) el yapımı maliyet, sadece önsel (emisyon ağırlığı 0)
    hand = []
    for seq, probs in zip(val_seqs, probs_list):
        segs = [(int(round(on * FRAME_RATE)), max(int(round(on * FRAME_RATE)) + 1,
                 int(round(off * FRAME_RATE))), p) for (on, off, p, *_ ) in seq]
        hand.append(viterbi_assign(segs, probs, INSTR, FRAME_RATE, w_emission=0.0,
                                   w_transition=1.0))
    acc("el yapimi maliyet (sadece onsel)", hand)
    # c) öğrenilen önsel tek başına
    acc("OGRENILEN onsel (ses yok)", [decode_lattice(l, 0.0, 1.0) for l in lat_prior])
    # d) ses tek başına (greedy)
    acc("ses / greedy (onsel yok)", [decode_lattice(l, 1.0, 0.0) for l in lat_full])
    # e) ikisi birlikte
    best_w, best_a = 0.0, -1.0
    for w in ([0.5, 1.0, 2.0] if quick else [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]):
        preds = [decode_lattice(l, 1.0, w) for l in lat_full]
        a = _acc(preds, gt_all, multi)[1]
        if a > best_a:
            best_w, best_a = w, a
    acc(f"ses + OGRENILEN onsel (w_tr={best_w})",
        [decode_lattice(l, 1.0, best_w) for l in lat_full])

    print(f"\n  {'yontem':<40}{'tum notalar':>12}{'cok-adayli':>12}")
    print("  " + "-" * 62)
    for name, preds in res.items():
        a_all, a_multi = _acc(preds, gt_all, multi)
        print(f"  {name:<40}{a_all:>12.3f}{a_multi:>12.3f}")

    # el yapımı maliyet gerçek yolu açıklıyor mu?
    def mean_cost(paths):
        c = n = 0
        for seq, path in zip(val_seqs, paths):
            for i in range(1, len(seq)):
                if path[i - 1][0] is None or path[i][0] is None:
                    continue
                gap = max(0.0, seq[i][0] - seq[i - 1][1])
                c += transition_cost(path[i - 1], path[i], gap); n += 1
        return c / max(n, 1)
    print(f"\n  el yapimi maliyet / gecis:  GERCEK yol {mean_cost(gt_all):.2f}  |  "
          f"maliyet-minimum yol {mean_cost(hand):.2f}  |  greedy yol "
          f"{mean_cost(res['ses / greedy (onsel yok)']):.2f}")
    print("  (gercek yol minimumdan cok pahaliysa: gitaristler 'en az hareket' ile "
          "calmiyor -> el yapimi onselin bicimi yanlis)")
    return best_w


def _acc(preds, gt_all, multi):
    c = n = cm = nm = 0
    for p, g, m in zip(preds, gt_all, multi):
        for pi, gi, mi in zip(p, g, m):
            ok = tuple(pi) == tuple(gi)
            c += ok; n += 1
            if mi:
                cm += ok; nm += 1
    return c / max(n, 1), cm / max(nm, 1)


def _uniform_nll(seqs):
    tot = n = 0
    for seq in seqs:
        for (_, _, p, s, f) in seq:
            k = len(INSTR.pitch_to_positions(p))
            if k > 1 and (s, f) in INSTR.pitch_to_positions(p):
                tot += np.log(k); n += 1
    return tot / max(n, 1)


# ----------------------------------------------------------------- 2) uçtan uca
def end_to_end(tm, data, quick=False, onset_thr=None, dec=None):
    print("\n" + "=" * 64)
    print("2) UCTAN UCA - kare-seviye tab F1 (eval_viterbi ile ayni metrik)")
    print("=" * 64)

    def score(fn):
        tp = fp = fn_ = 0
        for i, (probs, gt, *_) in enumerate(data):
            a, b, c = tab_scores(fn(i, probs, gt), gt)
            tp += a; fp += b; fn_ += c
        return prf(tp, fp, fn_)

    best_base = (None, -1, None)
    for m in ([0.5, 0.8, 0.9] if quick else np.round(np.arange(0.0, 0.96, 0.05), 2)):
        res = score(lambda i, pr, gt: decode_with_margin(pr, m))
        if res[2] > best_base[1]:
            best_base = (m, res[2], res)

    thresholds = [0.8, 0.9] if quick else [0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    w_trs = [0.0, 0.5, 1.0, 2.0] if quick else [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    best_greedy = (None, -1, None)
    best_hmm = {"f1": -1}
    print(f"  {'esik':>5} {'w_tr':>5} {'P':>7} {'R':>7} {'F1':>7}")
    for thr in thresholds:
        segs_all, lats = [], []
        for probs, gt, ons, rise, offs in data:
            segs = segment(probs, ons, thr, onset_thr, INSTR, rise=rise, offsets=offs, **(dec or {}))
            segs_all.append(segs)
            lats.append(build_lattice(segs, probs, tm, time_unit="frames"))
        for w in w_trs:
            res = score(lambda i, pr, gt: notes_to_frames(
                segs_all[i], decode_lattice(lats[i], 1.0, w), len(gt)))
            mark = ""
            if w == 0.0 and res[2] > best_greedy[1]:
                best_greedy = (thr, res[2], res)
            if res[2] > best_hmm["f1"]:
                best_hmm = {"f1": res[2], "prf": res, "thr": thr, "w": w}; mark = "  <-"
            print(f"  {thr:5.2f} {w:5.2f} {res[0]:7.3f} {res[1]:7.3f} {res[2]:7.3f}{mark}")

    print("\n" + "=" * 64)
    print(f"{'yontem':<34}{'P':>8}{'R':>8}{'tab F1':>10}")
    print("-" * 64)
    for name, res, extra in [
        ("baseline (argmax+marj)", best_base[2], f"marj={best_base[0]}"),
        ("greedy (nota basi)", best_greedy[2], f"esik={best_greedy[0]}"),
        ("OGRENILEN Viterbi", best_hmm["prf"], f"esik={best_hmm['thr']} w_tr={best_hmm['w']}"),
    ]:
        print(f"{name:<34}{res[0]:>8.3f}{res[1]:>8.3f}{res[2]:>10.3f}   {extra}")
    print("=" * 64)
    if best_hmm["w"] == 0.0:
        print("SONUC: w_transition=0 secildi -> ogrenilen onsel KATKI VERMEDI.")
    else:
        gain = best_hmm["f1"] - max(best_base[1], best_greedy[1])
        print(f"SONUC: ogrenilen Viterbi en iyi alternatife gore {gain:+.3f} F1.")
    edge = [("baseline marj", best_base[0], 0.95), ("esik", best_hmm["thr"], thresholds[-1]),
            ("w_tr", best_hmm["w"], w_trs[-1])]
    for name, v, hi in edge:
        if v is not None and v >= hi:
            print(f"UYARI: en iyi {name} taramanin ucunda ({v}); aralik genisletilmeli.")
    print("NOT: parametreler val'de secildi -> mutlak sayilar hafif iyimser.")


# ----------------------------------------------------------------- ana akış
def _predict(model, ck, kind, cqt, device):
    if hasattr(model, "onset_head"):
        return predict_heads(model, cqt, device)
    return predict_probs(model, ck, kind, cqt, device), None, None


def main(kind="crnn", quick=False, refit=False, ckpt=None, onset_thr=None, dec=None):
    tm = get_transition_model(refit)
    device = get_device()
    model, ck, kind = load_model(kind, device, ckpt)
    print(f"Model: {kind} ({ckpt or 'varsayilan'}) | cihaz: {device}")

    val_files, val_seqs = _label_seqs("val")
    data, probs_list = [], []
    for f in val_files:
        with np.load(f) as d:
            probs, ons, offs = _predict(model, ck, kind, _normalize(d["cqt"]), device)
            data.append((probs, d["tab"].astype(np.int64), ons, energy_rise(d["cqt"], INSTR), offs))
        probs_list.append(probs)
    print(f"{len(data)} val kaydi islendi.")

    diagnose(tm, val_files, val_seqs, probs_list, quick)
    if onset_thr is None:
        onset_thr = ck.get("onset_thr") or 0.5
    use_on = onset_thr if (data[0][2] is not None and onset_thr >= 0) else None
    print(f"\nUctan uca cozumleme: {'onset (esik ' + str(use_on) + ')' if use_on is not None else 'kare-esik'}")
    if use_on is not None and dec:
        print(f"Katman 3.10 cozumleme kurallari: {dec}")
    end_to_end(tm, data, quick, use_on, dec)


def demo(npz_path, kind="crnn", threshold=0.8, w_transition=1.0, ckpt=None, onset_thr=None, dec=None):
    from gtab.decoding.decode import render_ascii_tab
    tm = get_transition_model(False)
    device = get_device()
    model, ck, kind = load_model(kind, device, ckpt)
    with np.load(npz_path) as d:
        probs, ons, offs = _predict(model, ck, kind, _normalize(d["cqt"]), device)
        rise = energy_rise(d["cqt"], INSTR)
    if onset_thr is None:
        onset_thr = ck.get("onset_thr") or 0.5
    segs = segment(probs, ons, threshold, onset_thr if onset_thr >= 0 else None, INSTR, rise=rise,
                   offsets=offs, **(dec or {}))
    asg = decode_lattice(build_lattice(segs, probs, tm), 1.0, w_transition)
    events = []
    for (a, b, pitch), (s, f) in zip(segs, asg):
        ev = NoteEvent(a / FRAME_RATE, b / FRAME_RATE, pitch)
        ev.string, ev.fret = s, f
        events.append(ev)
    tr = Transcription(notes=events).sort()
    print(f"{npz_path}: {len(tr.notes)} nota (ogrenilen Viterbi, esik={threshold}, "
          f"w_tr={w_transition})\n")
    print(render_ascii_tab(tr, INSTR))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="crnn", choices=["cnn", "crnn"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--refit", action="store_true", help="gecis modelini yeniden ogren")
    ap.add_argument("--demo", default=None)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--w-transition", type=float, default=1.0)
    ap.add_argument("--ckpt", default=None, help="model agirlik dosyasi (varsayilan: tabcnn.pt/tabcrnn.pt)")
    ap.add_argument("--onset-thr", type=float, default=None, help="onset esigi (bos = checkpoint'teki); -1 = kare-esik cozumlemesi")
    ap.add_argument("--off-ratio", type=float, default=1.0, help="Katman 3.10 histerezis (eval_pitch'in sectigi deger)")
    ap.add_argument("--refractory", type=int, default=0, help="Katman 3.10 refrakter pencere (kare)")
    ap.add_argument("--peak", action="store_true", help="Katman 3.10 Adim 2: notayi onset tepe noktasindan baslat")
    ap.add_argument("--reattack", type=float, default=0.0, help="Katman 3.10 Adim 3b: nota surerken yeniden vurus icin onset tepe esigi (0 = kapali)")
    ap.add_argument("--rise-keep", type=float, default=0.0, help="Adim 3 rev.: ses kesilmeden yeniden vurus icin enerji yukselisi esigi (dB)")
    ap.add_argument("--rise-split", type=float, default=0.0, help="Adim 3 rev.: gomulu tekrari enerji kanitiyla ayirma esigi (dB)")
    ap.add_argument("--fallback", type=int, default=0, help="Katman 3.10 onset'siz yedek nota (kare)")
    ap.add_argument("--offset-thr", type=float, default=0.0, help="Adim 3c: offset kafasi esigi (0 = kapali)")
    args = ap.parse_args()
    if args.demo:
        dec = dict(off_ratio=args.off_ratio, refractory=args.refractory, fallback=args.fallback, peak=args.peak, reattack=args.reattack,
                   rise_keep=args.rise_keep, rise_split=args.rise_split,
                   offset_threshold=args.offset_thr)
        demo(args.demo, args.model, args.threshold, args.w_transition, args.ckpt, args.onset_thr, dec)
    else:
        dec = dict(off_ratio=args.off_ratio, refractory=args.refractory, fallback=args.fallback, peak=args.peak, reattack=args.reattack,
                   rise_keep=args.rise_keep, rise_split=args.rise_split,
                   offset_threshold=args.offset_thr)
        main(args.model, args.quick, args.refit, args.ckpt, args.onset_thr, dec)
