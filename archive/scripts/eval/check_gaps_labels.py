"""
Katman 3.10 / Adım 2a-2 — GAPS etiket kalitesi kontrolü (eğitim YOK).

Soru: GAPS'teki hayaletlerin %70'i "yanlış perde". Bunun ne kadarı model hatası,
ne kadarı etiket gürültüsü? GAPS etiketleri partisyonun sesle hizalanmasından gelir.

Veri: data/raw/gaps_hf/match/<id>.match (score ↔ performans eşleşmesi).
  snote(...)-note(...)  = eşleşen nota
  snote(...)-deletion   = partisyonda var, performansta eşleşmedi
  insertion-note(...)   = performansta (etiket MIDI'sinde) var, partisyonda eşleşmedi
Etiket MIDI'si ≈ eşleşen + insertion notalarıdır (001_mvswc: 498 + 1260 ≈ 1745, kontrol edildi).
Parça başına "eşleşme oranı" = eşleşen / (eşleşen + insertion + deletion): düşükse o parçanın
partisyon-performans hizalaması (dolayısıyla etiketi) daha az güvenilir.

Yöntem: her parça için modelin nota F1'i (sabit çözümleme) ile eşleşme oranının ilişkisi.
Model hatası etiket kalitesinden bağımsızsa korelasyon ~0; etiket gürültüsü belirleyiciyse
düşük oranlı parçalarda F1 belirgin düşük ve hayaletler orada yoğun olur.

Yalnızca TEST DIŞI parçalar kullanılır (gaps_train içi eğitim + gaps_val). Eğitim parçalarında
F1 iyimserdir ama karşılaştırma parçalar arasında olduğu için ilişki yine okunabilir.
Hiçbir parametre burada seçilmez.

Çözümleme varsayılanı: README11'de GAPS doğrulamasında seçilen ayar
(onset@0.2, perde eşiği 0.4, histerezis 0.5, refrakter 6, yedek 10).

Çalıştırma:
    python -m scripts.eval.check_gaps_labels --ckpt tabcrnn_onset_h.pt
"""

import argparse
import os

import numpy as np

from gtab.data.gaps import GAPS_DIR, gaps_files
from gtab.data.torch_dataset import _normalize
from gtab.evaluation.pitch_eval import track_scores
from gtab.models.inference import load_model, predict_with_onsets
from gtab.utils import get_device


def match_stats(path):
    m = ins = dele = 0
    with open(path, encoding="utf8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("snote(") and "-note(" in line:
                m += 1
            elif line.startswith("snote(") and "deletion" in line:
                dele += 1
            elif line.startswith("insertion"):
                ins += 1
    tot = m + ins + dele
    return m / max(tot, 1), ins / max(m + ins, 1), dele / max(m + dele, 1)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main(ckpt, thr, onset_thr, dec):
    device = get_device()
    model, ck, kind = load_model("crnn", device, ckpt)
    tr, va = gaps_files("gaps_train")
    mdir = os.path.join(GAPS_DIR, "match")
    rows = []
    for f in tr + va:
        tid = os.path.basename(f)[:-4]
        mp = os.path.join(mdir, tid + ".match")
        if not os.path.exists(mp):
            continue
        with np.load(f) as d:
            probs, ons = predict_with_onsets(model, _normalize(d["cqt"]), device)
            r = track_scores(probs, ons, d["frame"], d["onset"], thr, onset_thr, dec)
        mt, ne, nr = r["note"]
        p, rc = mt / max(ne, 1), mt / max(nr, 1)
        f1 = 2 * p * rc / max(p + rc, 1e-9)
        rows.append((tid, *match_stats(mp), f1, p, rc, f in va))
    if not rows:
        raise FileNotFoundError(f"{mdir} icinde test disi parcalara ait .match yok.")

    rows.sort(key=lambda r: r[1])
    print(f"Model: {ckpt} | cozumleme: perde {thr}, onset@{onset_thr}, {dec}")
    print(f"{len(rows)} parca (test disi; {sum(r[7] for r in rows)} tanesi gaps_val)\n")
    print(f"{'parca':<16}{'eslesme':>9}{'insert.':>9}{'delet.':>9}{'nota F1':>9}{'P':>7}{'R':>7}")
    for r in rows:
        print(f"{r[0]:<16}{r[1]:>9.2f}{r[2]:>9.2f}{r[3]:>9.2f}{r[4]:>9.3f}{r[5]:>7.2f}{r[6]:>7.2f}"
              + ("  (val)" if r[7] else ""))
    ratio = np.array([r[1] for r in rows]); f1 = np.array([r[4] for r in rows])
    prec = np.array([r[5] for r in rows])
    h = len(rows) // 2
    print("\nOzet:")
    print(f"  eslesme orani medyan {np.median(ratio):.2f} (min {ratio.min():.2f}, maks {ratio.max():.2f})")
    print(f"  nota F1: dusuk eslesmeli yari {f1[:h].mean():.3f} | yuksek eslesmeli yari {f1[h:].mean():.3f}")
    print(f"  precision (hayalet olcusu): dusuk yari {prec[:h].mean():.3f} | yuksek yari {prec[h:].mean():.3f}")
    print(f"  Spearman(eslesme orani, nota F1) = {spearman(ratio, f1):+.2f} | "
          f"Spearman(eslesme orani, precision) = {spearman(ratio, prec):+.2f}")
    print("  Yorum: korelasyon guclu pozitifse (>~0.5) GAPS hatalarinin onemli kismi etiket/hizalama")
    print("  kaynakli; ~0 ise hatalar modelin kendisinde.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="tabcrnn_onset_h.pt")
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--onset-thr", type=float, default=0.2)
    ap.add_argument("--off-ratio", type=float, default=0.5)
    ap.add_argument("--refractory", type=int, default=6)
    ap.add_argument("--fallback", type=int, default=10)
    a = ap.parse_args()
    main(a.ckpt, a.threshold, a.onset_thr,
         dict(off_ratio=a.off_ratio, refractory=a.refractory, fallback=a.fallback))
