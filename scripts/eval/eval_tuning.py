"""
Katman 6 Adım 3 — Akort / kapo desteği ölçümü (GAPS scordatura kayıtları).

Etiket ve akort MusicXML'den (gtab.data.gaps_score: score_notes dördüncü dönüşü = <staff-tuning>; align ile MIDI notasına
tel), gaps_tab önbelleği KULLANILMAZ (o, standart akortla hizalandı: scordatura kayıtlarında perde numarası yanlış / eksik).
Etkin akort = partisyon akortu + align'ın kayıt geneli perde kayması (kayma > 0 -> kapo; < 0 -> tüm teller pes);
etiket perdesi = MIDI − etkin açık tel = partisyon perdesi.

Sayım: split başına kaç kayıt, hangi akort (yalnız MusicXML; hızlı).
Ölçüm (eşleştirilmiş): etkin akortu standart olmayan kayıtlar iki kez çözülür — standart akortla (akort desteği yokken
kullanıcının alacağı TAB) ve doğru akortla (--tuning / --capo). Notalar iki çözümde birebir aynı (akustik çözümleme hep
STANDARD_6); fark yalnız konum. deep2 notaları etiketli MIDI notalarına eşlenir (aynı perde, ± 50 ms):
  - konum doğruluğu (tel + perde) ve tel doğruluğu, bulunan etiketli notalarda; fark = akort desteğinin kazancı
  - kapsam dışı: model perde aralığı (E2..E6, MIDI 40..88) dışındaki etiketli notalar (drop-D'de D2 / D#2) -> bulunamaz
  - standart akortlu kayıtlarda tel doğruluğu = gösterge (README18 Adım 3: scordatura tel doğruluğu bununla ± 0.02)
Modlar: ses (duyulan konum) ve edisyon (EditionNet; standart akortta eğitildi -> başka akort dağılım dışı, ölçülür).
Bu adımda ayar seçilmez; ölçüm gaps_val'de, gaps_test yalnız açıkça (--splits gaps_test, kapanışta). gaps_fit (45
scordatura kaydı, akustik model bu seslerle eğitildi) yalnız gösterge.

Çalıştırma:
    python -m scripts.eval.eval_tuning --count-only
    python -m scripts.eval.eval_tuning                          # gaps_val
    python -m scripts.eval.eval_tuning --splits gaps_fit       # gösterge (daha çok scordatura kaydı)
    python -m scripts.eval.eval_tuning --limit 2               # duman testi
"""

import argparse
import glob
import os
from collections import Counter

import numpy as np

from gtab.core.instrument import STANDARD_6, Instrument
from gtab.data.gaps import GAPS_DIR, gaps_files
from gtab.data.gaps_score import align, match_notes, score_notes
from gtab.evaluation.protocol import is_quarantined
from gtab.paths import CACHE_DIR
from gtab.pipeline import analyze, load_models
from gtab.utils import get_device

ONSET_TOL = 0.05
LO, HI = STANDARD_6.pitch_range()                     # model perde aralığı (40..88)
MODES = ("ses", "edisyon")


def split_files(split):
    if split in ("gaps_fit", "gaps_val"):
        return gaps_files("gaps_train")[split == "gaps_val"]
    return sorted(glob.glob(os.path.join(CACHE_DIR, split, "*.npz")))


def score_tuning(tid):
    """MusicXML <staff-tuning> -> açık teller (pesten tize) | None."""
    r = score_notes(os.path.join(GAPS_DIR, "musicxml", f"{tid}.xml"))
    return None if r is None or not r[3] else tuple(r[3][k] for k in range(1, 7))


def effective(tuning, shift):
    """Partisyon akortu + kayıt geneli kayma -> etkin enstrüman (kayma > 0 = kapo: perdeler kapodan)."""
    t = tuple(p + shift for p in tuning)
    if t == STANDARD_6.tuning:
        return STANDARD_6
    return Instrument(f"etkin_{'_'.join(map(str, tuning))}{shift:+d}", t, STANDARD_6.num_frets - max(shift, 0))


def out_of_model(pitch):
    """Model çıkış aralığı (MIDI 40..88) dışındaki notalar: akort desteğiyle de bulunamaz ("kapsam dışı")."""
    pitch = np.asarray(pitch)
    return (pitch < LO) | (pitch > HI)


def count(splits):
    print("=" * 100)
    print("SAYIM — MusicXML <staff-tuning> (pesten tize, MIDI)")
    print("=" * 100)
    for sp in splits:
        c = Counter(score_tuning(os.path.basename(f)[:-4]) for f in split_files(sp))
        sc = sum(v for k, v in c.items() if k is not None and k != STANDARD_6.tuning)
        print(f"  {sp:<10} {sum(c.values()):>4} kayit | scordatura {sc}")
        for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
            print(f"      {v:>4}  {k}{'  (standart)' if k == STANDARD_6.tuning else ''}")


def score_rec(r, an):
    """-> (etiketli, kapsam dışı, bulunan, tel doğru, konum doğru) sayaçları; r = align çıktısı, an = analyze çıktısı."""
    gt = np.array(r["notes"], float).reshape(-1, 3)
    lab = r["string"] >= 0
    det = an["notes"]
    mi, _ = match_notes([(on, int(p)) for on, _, p in gt], [dict(pitch=int(p)) for p in det[:, 2]], det[:, 0], ONSET_TOL)
    hit = lab & (mi >= 0)
    j = mi[hit]
    s_ok = an["string"][j] == r["string"][hit]
    p_ok = s_ok & (an["fret"][j] == r["fret"][hit])
    return np.array([lab.sum(), (lab & out_of_model(gt[:, 2])).sum(), hit.sum(), s_ok.sum(), p_ok.sum()])


def measure(m, split, limit):
    files = split_files(split)
    rows = {k: np.zeros(5, int) for k in [(g, mode, dec) for g in ("scordatura", "standart") for mode in MODES
                                         for dec in ("standart", "dogru")]}
    n_rec = Counter()
    for f in files:
        tid = os.path.basename(f)[:-4]
        tuning = score_tuning(tid)
        if tuning is None or is_quarantined(tid):
            continue
        if limit is not None and n_rec["scordatura" if tuning != STANDARD_6.tuning else "standart"] >= limit:
            continue
        r = align(GAPS_DIR, tid, tuning, recover=True)
        if not (r["consistent"] or r["order_source"] == "content") or (r["string"] >= 0).mean() < 0.5:
            continue                                     # build_gaps_tab ile aynı kullanım kuralı
        eff = effective(tuning, r["shift"])              # kaymalı standart partisyon (kapo / pes akort) da "scordatura"
        grp = "standart" if eff is STANDARD_6 else "scordatura"
        lab = r["string"] >= 0                           # etiket perdesi = MIDI − etkin açık tel = partisyon perdesi
        r["fret"] = np.where(lab, r["fret"] - r["shift"], -1)
        with np.load(f) as d:
            cqt = d["cqt"]
        n_rec[grp] += 1
        for mode in MODES:
            a_std = analyze(cqt, m, mode)
            rows[(grp, mode, "standart")] += score_rec(r, a_std)
            if grp == "scordatura":
                a_ok = analyze(cqt, m, mode, eff)
                assert np.array_equal(a_ok["notes"], a_std["notes"])     # akustik çözümleme akorttan bağımsız
                rows[(grp, mode, "dogru")] += score_rec(r, a_ok)
        print(f"  [{split}] {tid} {grp:<10} akort {eff.tuning} (kayma {r['shift']:+d})", flush=True)
    print(f"\n  {split}: scordatura / kapo {n_rec['scordatura']} kayit | standart {n_rec['standart']} kayit")
    print(f"  {'grup':<11}{'mod':<9}{'cozum akortu':<14}{'etiketli':>9}{'kapsam disi':>12}{'bulunan':>9}"
          f"{'tel':>8}{'konum':>8}")
    for (g, mode, dec), (n, oo, h, s, p) in rows.items():
        if n:
            print(f"  {g:<11}{mode:<9}{dec:<14}{n:>9}{oo:>12}{h:>9}{s / max(h, 1):>8.3f}{p / max(h, 1):>8.3f}")
    for mode in MODES:
        a, b, c = rows[("scordatura", mode, "standart")], rows[("scordatura", mode, "dogru")], rows[("standart", mode, "standart")]
        if a[2] and b[2]:
            ref = f" | standart kayitlarda tel {c[3] / c[2]:.3f} (fark {b[3] / b[2] - c[3] / c[2]:+.3f})" if c[2] else ""
            print(f"  {mode}: konum dogrulugu dogru akort - standart akort = {b[4] / b[2] - a[4] / a[2]:+.3f}{ref}")


def main(splits, limit, count_only):
    count(("gaps_fit", "gaps_val", "gaps_test"))
    if count_only:
        return
    m = load_models(get_device())
    for sp in splits:
        print("\n" + "=" * 100)
        print(f"OLCUM — {sp} (eslestirilmis: ayni kayitlar standart ve dogru akortla)")
        print("=" * 100)
        measure(m, sp, limit)
    print("\nKarar kurali: dogru akortla konum dogrulugu > standart akortla (fark raporlanir); scordatura tel dogrulugu "
          "standart kayitlarla +- 0.02 gosterge (README18 Adim 3).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["gaps_val"], choices=["gaps_fit", "gaps_val", "gaps_test"],
                    help="gaps_test yalniz kapanista (acikca verilir)")
    ap.add_argument("--limit", type=int, default=None, help="grup basina en cok kayit (duman testi)")
    ap.add_argument("--count-only", action="store_true", help="yalniz akort sayimi")
    a = ap.parse_args()
    main(a.splits, a.limit, a.count_only)
