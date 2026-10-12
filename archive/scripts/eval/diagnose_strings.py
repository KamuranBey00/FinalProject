"""
Katman 3.9 / Adım 0 — Tel/perde hata analizi (eğitim YOK, sadece ölçüm).

Amaç: "nasıl iyileştiririz?" sorusunu tahminle değil ölçümle cevaplamak.

A) KAYIP AYRIŞTIRMASI (uçtan uca, greedy pipeline):
   Gerçek her (kare, tel) hücresi şunlardan biri:
     doğru | perde bulundu ama TEL yanlış | perde HİÇ bulunamadı
   Her yanlış pozitif:  perde doğru ama yanlış telde | olmayan perde (hayalet)
   -> Hangi tarafın (tel mi, perde/segmentasyon mu) daha çok kaybettirdiğini söyler.

B) TEL KARIŞIKLIĞI (oracle notalar: gerçek perde + sınırlar verilir, sadece tel seçilir):
   tel karışıklık matrisi; doğruluk: tel / perde bölgesi / aday sayısı / nota süresi /
   eşzamanlı nota sayısına göre.

C) TEL BİLGİSİ NOTANIN NERESİNDE? Teli notanın farklı bölümlerinden seçince doğruluk:
   ilk 1/3/5 kare (vuruş) | vuruş hariç orta | son kısım | tamamı.
   İlk kareler daha iyiyse -> onset tabanlı nota-düzeyi karar (Adım 1) kazandırır.

D) GÜVEN: yanlış seçimler emin mi (sistematik) yoksa kararsız mı?

Çalıştırma:
    python -m scripts.eval.diagnose_strings --ckpt tabcrnn_gaps.pt --splits val val_comp
"""

import argparse
import glob
import os
from collections import Counter, defaultdict

import numpy as np

from gtab.config import FRAME_RATE
from gtab.core.instrument import STANDARD_6
from gtab.data.torch_dataset import _normalize
from gtab.decoding.decode import decode_greedy, notes_to_frames
from gtab.decoding.transitions import notes_from_tab
from gtab.decoding.viterbi import pitch_matrix, segment
from gtab.models.inference import load_model, predict_probs, predict_with_onsets
from gtab.paths import CACHE_DIR
from gtab.utils import get_device

INSTR = STANDARD_6
NAMES = ["E", "A", "D", "G", "B", "e"]
FRET_REGIONS = [("acik (0)", 0, 0), ("1-4", 1, 4), ("5-9", 5, 9), ("10-14", 10, 14), ("15+", 15, 99)]


def load(split, model, ck, kind, device):
    out = []
    for f in sorted(glob.glob(os.path.join(CACHE_DIR, split, "*.npz"))):
        with np.load(f) as d:
            cqt = _normalize(d["cqt"])
            if hasattr(model, "onset_head"):
                probs, ons = predict_with_onsets(model, cqt, device)
            else:
                probs, ons = predict_probs(model, ck, kind, cqt, device), None
            out.append((probs, d["tab"].astype(np.int64), ons))
    if not out:
        raise FileNotFoundError(f"{CACHE_DIR}/{split} bos.")
    return out


# ----------------------------------------------------------------- A) kayıp ayrıştırması
def loss_decomposition(data, thr, onset_thr=None):
    c = Counter()
    lo = INSTR.pitch_range()[0]
    tuning = np.array(INSTR.tuning)
    for probs, gt, ons in data:
        pm, _ = pitch_matrix(probs, INSTR)
        segs = segment(probs, ons, thr, onset_thr, INSTR)
        pred = notes_to_frames(segs, decode_greedy(probs, segs), len(gt))
        n = min(len(pred), len(gt)); pred, g = pred[:n], gt[:n]
        if onset_thr is None:
            active = pm[:n] > thr                                # (T, P) bulunan perdeler
        else:                                                    # onset modunda: notalardan
            active = np.zeros_like(pm[:n], dtype=bool)
            for a, b, p in segs:
                active[a:min(b, n), p - lo] = True
        # gerçek hücreler
        for t, s in zip(*np.nonzero(g > 0)):
            p = tuning[s] + g[t, s] - 1 - lo
            if pred[t, s] == g[t, s]:
                c["dogru"] += 1
            elif 0 <= p < active.shape[1] and active[t, p]:
                c["perde var, tel yanlis"] += 1
            else:
                c["perde bulunamadi"] += 1
        # yanlış pozitifler
        gt_p = set()
        for t, s in zip(*np.nonzero(pred > 0)):
            if pred[t, s] == g[t, s]:
                continue
            p = tuning[s] + pred[t, s] - 1
            same = any(g[t, s2] > 0 and tuning[s2] + g[t, s2] - 1 == p for s2 in range(INSTR.num_strings))
            c["FP: perde dogru, yanlis tel" if same else "FP: olmayan perde"] += 1
    tot_gt = c["dogru"] + c["perde var, tel yanlis"] + c["perde bulunamadi"]
    tot_fp = c["FP: perde dogru, yanlis tel"] + c["FP: olmayan perde"]
    print(f"  gercek (kare,tel) hucreleri: {tot_gt}")
    for k in ("dogru", "perde var, tel yanlis", "perde bulunamadi"):
        print(f"    {k:<28}{c[k] / max(tot_gt, 1):7.1%}")
    print(f"  yanlis pozitifler: {tot_fp}")
    for k in ("FP: perde dogru, yanlis tel", "FP: olmayan perde"):
        print(f"    {k:<28}{c[k] / max(tot_fp, 1):7.1%}")
    return c


# ----------------------------------------------------------------- B/C/D) oracle notalar
def window(a, b, mode):
    L = b - a
    if mode == "tamami":
        return a, b
    if mode.startswith("ilk"):
        k = int(mode.split()[1]); return a, min(b, a + k)
    if mode == "vurus haric orta":
        return (a + 3, b - 2) if L > 6 else (a, b)
    if mode == "son 1/3":
        return max(a, b - max(1, L // 3)), b
    raise ValueError(mode)


MODES = ["tamami", "ilk 1", "ilk 3", "ilk 5", "vurus haric orta", "son 1/3"]


def oracle(data):
    rows = []
    for probs, gt, _ in data:
        T = len(probs)
        notes = notes_from_tab(gt, INSTR)
        # eşzamanlı nota sayısı (gerçek), nota başlangıcında
        poly = (gt > 0).sum(1)
        for (on, off, pitch, s, f) in notes:
            a, b = int(round(on * FRAME_RATE)), int(round(off * FRAME_RATE))
            b = min(max(b, a + 1), T)
            if a >= T:
                continue
            cands = INSTR.pitch_to_positions(pitch)
            if (s, f) not in cands:
                continue
            row = {"s": s, "f": f, "pitch": pitch, "K": len(cands), "dur": b - a,
                   "poly": int(poly[a]) if a < len(poly) else 1}
            for m in MODES:
                wa, wb = window(a, b, m)
                sc = np.array([probs[wa:wb, cs, cf + 1].mean() for (cs, cf) in cands])
                row[m] = cands[int(sc.argmax())]
                if m == "tamami":
                    srt = np.sort(sc / max(sc.sum(), 1e-9))[::-1]
                    row["margin"] = srt[0] - (srt[1] if len(srt) > 1 else 0.0)
            rows.append(row)
    return rows


def report_oracle(rows):
    multi = [r for r in rows if r["K"] > 1]
    acc = lambda rs, m="tamami": np.mean([r[m] == (r["s"], r["f"]) for r in rs]) if rs else float("nan")
    print(f"  nota: {len(rows)} | cok-adayli: {len(multi)} | tel dogrulugu (cok-adayli): {acc(multi):.3f}")

    print("\n  C) Teli notanin hangi bolumunden secersek? (cok-adayli notalar)")
    for m in MODES:
        print(f"    {m:<20}{acc(multi, m):7.3f}")
    long = [r for r in multi if r["dur"] >= 10]
    print(f"    (>=10 karelik notalarda: tamami {acc(long):.3f} | ilk 3 {acc(long, 'ilk 3'):.3f} | "
          f"orta {acc(long, 'vurus haric orta'):.3f}, n={len(long)})")

    print("\n  B) Tel karisiklik matrisi (satir = gercek tel, sutun = secilen tel), cok-adayli")
    M = np.zeros((6, 6), int)
    for r in multi:
        M[r["s"], r["tamami"][0]] += 1
    print("        " + "".join(f"{n:>7}" for n in NAMES) + "   dogruluk")
    for s in range(6):
        tot = M[s].sum()
        print(f"    {NAMES[s]:>3} " + "".join(f"{v:>7}" for v in M[s]) +
              f"   {M[s, s] / max(tot, 1):6.3f} (n={tot})")
    d = Counter(abs(r["tamami"][0] - r["s"]) for r in multi if r["tamami"] != (r["s"], r["f"]))
    tot = sum(d.values())
    print("    hatali secimlerde tel uzakligi: " +
          "  ".join(f"{k} tel: {v / max(tot, 1):.0%}" for k, v in sorted(d.items())))
    hi = sum(1 for r in multi if r["tamami"] != (r["s"], r["f"]) and r["tamami"][1] > r["f"])
    print(f"    hatalarin %{100 * hi / max(tot, 1):.0f}'inde model GERCEKTEN DAHA YUKSEK perde (kola dogru) secti")

    print("\n  Dogruluk kirilimlari (cok-adayli):")
    for name, lo, hi_ in FRET_REGIONS:
        rs = [r for r in multi if lo <= r["f"] <= hi_]
        print(f"    gercek perde {name:<9} {acc(rs):6.3f}  (n={len(rs)})")
    for k in sorted({r["K"] for r in multi}):
        rs = [r for r in multi if r["K"] == k]
        print(f"    aday sayisi {k}         {acc(rs):6.3f}  (n={len(rs)})")
    for name, lo, hi_ in (("kisa <5 kare", 0, 4), ("orta 5-14", 5, 14), ("uzun 15+", 15, 10**6)):
        rs = [r for r in multi if lo <= r["dur"] <= hi_]
        print(f"    nota suresi {name:<10} {acc(rs):6.3f}  (n={len(rs)})")
    for name, lo, hi_ in (("tek", 0, 1), ("2", 2, 2), ("3", 3, 3), ("4+", 4, 99)):
        rs = [r for r in multi if lo <= r["poly"] <= hi_]
        if rs:
            print(f"    eszamanli nota {name:<7} {acc(rs):6.3f}  (n={len(rs)})")

    print("\n  D) Guven: tel secimindeki marj (1. aday - 2. aday, normalize)")
    wrong = [r["margin"] for r in multi if r["tamami"] != (r["s"], r["f"])]
    right = [r["margin"] for r in multi if r["tamami"] == (r["s"], r["f"])]
    if wrong:
        print(f"    dogru secimler: medyan marj {np.median(right):.2f} | yanlis secimler: medyan {np.median(wrong):.2f}")
        print(f"    yanlislarin %{100 * np.mean(np.array(wrong) > 0.5):.0f}'i EMIN (marj > 0.5) -> sistematik hata")


def main(ckpt, splits, thr, onset_thr=None):
    device = get_device()
    model, ck, kind = load_model("crnn", device, ckpt)
    if onset_thr is None:
        onset_thr = ck.get("onset_thr") or 0.5
    if not hasattr(model, "onset_head") or onset_thr < 0:
        onset_thr = None
    print(f"Model: {ckpt} | cihaz: {device} | segmentasyon esigi {thr} | "
          f"cozumleme: {'onset ' + str(onset_thr) if onset_thr is not None else 'kare-esik'}")
    for split in splits:
        data = load(split, model, ck, kind, device)
        print(f"\n{'=' * 70}\n{split}  ({len(data)} kayit)\n{'=' * 70}")
        print("A) Kayip ayristirmasi (uctan uca, greedy)")
        loss_decomposition(data, thr, onset_thr)
        print("\nOracle notalar (gercek perde + sinirlar, sadece tel secimi)")
        report_oracle(oracle(data))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="tabcrnn_gaps.pt")
    ap.add_argument("--splits", nargs="+", default=["val", "val_comp"])
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--onset-thr", type=float, default=None, help="onset esigi (bos = checkpoint'teki); -1 = kare-esik")
    a = ap.parse_args()
    main(a.ckpt, a.splits, a.threshold, a.onset_thr)
