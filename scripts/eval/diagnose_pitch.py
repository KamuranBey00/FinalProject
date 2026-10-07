"""
Katman 3.10 / Adım 0 — Polifonide PERDE hata analizi (eğitim YOK, sadece ölçüm).

Katman 3.9 sonrası kalan kayıp perde tarafında (val_comp: perde bulunamayan hücre
%23.5, yanlış pozitiflerin %75'i hayalet nota). Bu script o iki kaybı parçalara ayırır.
Çözümleme ve eşikler eval_pitch ile AYNI: alan-içi kalibrasyon (gtab.evaluation.pitch_eval).

Ölçüm DOĞRULAMA setlerinde yapılır (val_comp, gaps_val); test setine bakarak çözüm
tasarlamak sızıntı olur.

A) KAÇAN NOTALAR (eşleşmeyen gerçek notalar, onset toleransı 50 ms) — neden?
   perde hiç aktif değil | perde aktif ama nota başlatılmamış (onset kaçtı) |
   nota var ama onset zamanı kaymış | yerine oktav/harmonik seçilmiş
   + kırılım: süre, register, eşzamanlı nota sayısı, aynı perdenin tekrarı
B) HAYALET NOTALAR (eşleşmeyen tahminler) — gerçek notalarla ilişkisi?
   aynı perde (çift tetik / zamanlama) | oktav | 5'li-12'li harmonik | komşu yarım/tam ses |
   sessizlikte | diğer   + süre dağılımı
C) KARE KAYBI notanın neresinde? baş / orta / son (sönümleme) + eşleşen notaların
   kapsama oranı (nota erken mi bitiyor?)
D) (Katman 3.10 Adım 3a) HAYALET ALT KIRILIMI — çözümleme kuralı seçmek için:
   fragman (aynı perdede ZATEN BULUNMUŞ gerçek notanın içinde ikinci parça) |
   kayık başlangıç (gerçek nota var ama bulunamamış, başlangıç >50 ms kayık) | nota bitti
   sonra uzama | farklı perde; hangi kural üretti (onset / yedek); onset tepe değeri;
   "yeniden vuruş" (aynı perdede önceki tahmin yeni bitmişken başlayan nota) için
   onset eşiği tablosu: eşik x -> kaç hayalet / kaç doğru nota elenir.

Çalıştırma:
    python -m scripts.eval.diagnose_pitch --ckpt tabcrnn_onset_h.pt --splits val_comp gaps_val
"""

import argparse
from collections import Counter

import mir_eval
import numpy as np

from gtab.core.instrument import STANDARD_6
from gtab.decoding.viterbi import pitch_matrix, pitch_onset_matrix, segment
from gtab.evaluation.pitch_eval import (DEC_GRID, calibrate, load_split, roll_to_notes,
                                        selection_split, to_mir)
from gtab.models.inference import load_model
from gtab.utils import get_device

INSTR = STANDARD_6
TOL = 2                      # 50 ms ≈ 2 kare (onset toleransı, mir_eval ile aynı ölçek)


def pct(c, tot):
    return f"{100 * c / max(tot, 1):5.1f}%"


def interval_class(d):
    d = abs(d)
    if d == 0:
        return "ayni perde (cift tetik/zamanlama)"
    if d in (12, 24):
        return "oktav"
    if d in (7, 19, 5, 17):
        return "5'li/12'li harmonik"
    if d in (1, 2):
        return "komsu yarim/tam ses"
    return "diger aralik"


def analyse(data, thr, ot, dec=None):
    lo = INSTR.pitch_range()[0]
    miss = Counter(); miss_by = {k: Counter() for k in ("sure", "register", "polifoni", "tekrar")}
    tot_by = {k: Counter() for k in ("sure", "register", "polifoni", "tekrar")}
    ghost = Counter(); ghost_dur = Counter(); n_ref = n_est = 0
    pos_miss = Counter(); pos_tot = Counter(); cover = []
    gsub = Counter(); gsrc = Counter(); frag_gap = []; late_off = []
    peak_ok, peak_gh = [], []                    # onset tepe değeri: eşleşen / hayalet
    re_ok, re_gh = [], []                        # yalnız "yeniden vuruş" notaları

    for probs, ons, frame, onset in data:
        pm, _ = pitch_matrix(probs, INSTR)
        n = min(len(pm), len(frame))
        pm, fr, on = pm[:n], frame[:n] > 0, onset[:n] > 0
        est = segment(probs[:n], None if ons is None else ons[:n], thr, ot, INSTR, **(dec or {}))
        ref = roll_to_notes(fr, on, lo)
        n_ref += len(ref); n_est += len(est)
        ri, rp = to_mir(ref); ei, ep = to_mir(est)
        pairs = mir_eval.transcription.match_notes(ri, rp, ei, ep, onset_tolerance=0.05,
                                                   offset_ratio=None) if len(ref) and len(est) else []
        m_ref = {i for i, _ in pairs}; m_est = {j for _, j in pairs}
        poly = fr.sum(1)
        # tahmin kareleri (onset modunda notalardan; değilse eşikten)
        pred = np.zeros_like(fr)
        for a, b, p in est:
            pred[a:b, p - lo] = True
        if ons is None or ot is None:
            pred = pm > thr

        # ---- A) kaçan notalar
        prev_end = {}
        for i, (a, b, p) in enumerate(ref):
            L = b - a
            keys = {
                "sure": "kisa <5" if L < 5 else ("orta 5-19" if L < 20 else "uzun 20+"),
                "register": "pes <52" if p < 52 else ("orta 52-63" if p < 64 else "tiz 64+"),
                "polifoni": str(min(int(poly[a]), 4)).replace("4", "4+"),
                "tekrar": "tekrar (ayni perde, <=3 kare once bitti)"
                          if (p in prev_end and a - prev_end[p] <= 3) else "tekrar degil",
            }
            prev_end[p] = b
            for k, v in keys.items():
                tot_by[k][v] += 1
            if i in m_ref:
                continue
            col = pm[a:b, p - lo]
            same = [e for e in est if e[2] == p and e[0] < b and e[1] > a]
            octv = [e for e in est if e[2] != p and abs(e[0] - a) <= TOL
                    and interval_class(e[2] - p) in ("oktav", "5'li/12'li harmonik")]
            if same:
                why = "nota var, onset zamani kaymis"
            elif col.max() <= thr:
                why = "oktav/harmonik secilmis" if octv else "perde hic aktif degil"
            else:
                why = "perde aktif ama nota baslatilmamis (onset kacti)"
            miss[why] += 1
            for k, v in keys.items():
                miss_by[k][v] += 1

        # ---- D) hayalet alt kırılımı + onset tepe değerleri
        om = pitch_onset_matrix(probs[:n], ons[:n], INSTR) if ons is not None else None
        no_fb = None
        if dec and dec.get("fallback", 0) and ot is not None:
            no_fb = {(e[0], e[2]) for e in segment(probs[:n], ons[:n], thr, ot, INSTR,
                                                    **dict(dec, fallback=0))}
        last_end = {}
        for j, (a, b, p) in enumerate(est):
            pk = float(om[a:min(n, a + 3), p - lo].max()) if om is not None and a < n else np.nan
            reat = p in last_end and a - last_end[p] <= 3          # önceki aynı perde tahmin yeni bitti
            last_end[p] = max(last_end.get(p, -99), b)
            (peak_ok if j in m_est else peak_gh).append(pk)
            if reat:
                (re_ok if j in m_est else re_gh).append(pk)
            if j in m_est:
                continue
            if no_fb is not None:
                gsrc["yedek kural (onset'siz)" if (a, p) not in no_fb else "onset ile baslatilmis"] += 1
            inside = [i for i, (ra, rb, rp_) in enumerate(ref) if rp_ == p and ra <= a < rb]
            before = [i for i, (ra, rb, rp_) in enumerate(ref) if rp_ == p and rb <= a <= rb + 3]
            if inside and any(i in m_ref for i in inside):
                gsub["fragman (gercek nota zaten bulunmus)"] += 1
                prev = [e for e in est if e[2] == p and e[1] <= a + 1 and e[0] < a]
                if prev:
                    frag_gap.append(a - max(e[1] for e in prev))
            elif inside:
                gsub["kayik baslangic (gercek nota bulunamamis)"] += 1
                late_off.append(1000.0 * (a - ref[inside[0]][0]) / 43.07)
            elif before:
                gsub["nota bittikten sonra (uzama/yeniden tetik)"] += 1
            else:
                gsub["farkli perde / sessizlik"] += 1

        # ---- B) hayalet notalar
        for j, (a, b, p) in enumerate(est):
            if j in m_est:
                continue
            ghost_dur["kisa <5" if b - a < 5 else ("orta 5-19" if b - a < 20 else "uzun 20+")] += 1
            act = np.nonzero(fr[a:min(b, a + 3)].any(0))[0] + lo        # notanın başındaki gerçek perdeler
            if not len(act):
                ghost["sessizlikte"] += 1
                continue
            d = min((p - q for q in act), key=abs)
            ghost[interval_class(d)] += 1

        # ---- C) kare kaybı notanın neresinde
        for i, (a, b, p) in enumerate(ref):
            L = b - a
            if L < 5:
                continue
            seg = pred[a:b, p - lo]
            for name, sl in (("bas %20", slice(0, max(1, L // 5))), ("orta", slice(L // 5, L - L // 5)),
                             ("son %20", slice(L - max(1, L // 5), L))):
                pos_tot[name] += seg[sl].size; pos_miss[name] += int((~seg[sl]).sum())
            if i in m_ref:
                cover.append(seg.mean())

    return dict(miss=miss, miss_by=miss_by, tot_by=tot_by, ghost=ghost, ghost_dur=ghost_dur,
                n_ref=n_ref, n_est=n_est, pos_miss=pos_miss, pos_tot=pos_tot, cover=cover,
                gsub=gsub, gsrc=gsrc, frag_gap=frag_gap, late_off=late_off,
                peak_ok=np.array(peak_ok), peak_gh=np.array(peak_gh),
                re_ok=np.array(re_ok), re_gh=np.array(re_gh))


def report(r):
    nm, ng = sum(r["miss"].values()), sum(r["ghost"].values())
    print(f"  gercek nota {r['n_ref']} | tahmin {r['n_est']} | KACAN {nm} ({pct(nm, r['n_ref'])}) | "
          f"HAYALET {ng} ({pct(ng, r['n_est'])} of tahmin)")
    print("\n  A) Kacan notalar - neden?")
    for k, v in r["miss"].most_common():
        print(f"    {k:<48}{pct(v, nm)}  (n={v})")
    print("  A') Kacma ORANI (o gruptaki gercek notalarin yuzde kaci kaciyor):")
    for k in ("sure", "register", "polifoni", "tekrar"):
        row = "   ".join(f"{v}: {pct(r['miss_by'][k][v], r['tot_by'][k][v])}"
                         for v in sorted(r["tot_by"][k]))
        print(f"    {k:<9} {row}")
    print("\n  B) Hayalet notalar - gercek notalarla iliskisi:")
    for k, v in r["ghost"].most_common():
        print(f"    {k:<48}{pct(v, ng)}  (n={v})")
    print("    sure: " + "   ".join(f"{k}: {pct(v, ng)}" for k, v in sorted(r["ghost_dur"].items())))
    print("\n  C) Kare kaybi notanin neresinde? (>=5 karelik gercek notalar, kacan kare orani)")
    for k in ("bas %20", "orta", "son %20"):
        print(f"    {k:<10}{pct(r['pos_miss'][k], r['pos_tot'][k])}")
    if r["cover"]:
        c = np.array(r["cover"])
        print(f"    eslesen notalarin kare kapsamasi: medyan {np.median(c):.2f} | "
              f"%{100 * np.mean(c < 0.7):.0f}'i notanin %70'inden azini kapsiyor (erken bitis)")

    ng = sum(r["gsub"].values())
    print()
    print("  D) Hayalet alt kirilimi (Adim 3a):")
    for k, v in r["gsub"].most_common():
        print(f"    {k:<48}{pct(v, ng)}  (n={v})")
    if r["gsrc"]:
        print("    kaynak: " + "   ".join(f"{k}: {pct(v, ng)}" for k, v in r["gsrc"].most_common()))
    if r["frag_gap"]:
        g = np.array(r["frag_gap"])
        print(f"    fragmanlarda onceki parcayla bosluk (kare): medyan {np.median(g):.0f} | "
              f"<=2: %{100 * np.mean(g <= 2):.0f} | <=5: %{100 * np.mean(g <= 5):.0f}")
    if r["late_off"]:
        o = np.array(r["late_off"])
        print(f"    kayik baslangiclarda gecikme (ms): medyan {np.median(o):.0f} | "
              f">100 ms: %{100 * np.mean(o > 100):.0f}")
    po, pg = r["peak_ok"], r["peak_gh"]
    if len(po) and not np.isnan(po).all():
        print(f"    onset tepe degeri: dogru notalar medyan {np.nanmedian(po):.2f} | "
              f"hayaletler medyan {np.nanmedian(pg):.2f}")
        for name, ok, gh in (("tum notalar", po, pg), ("yeniden vurus", r["re_ok"], r["re_gh"])):
            if not len(gh):
                continue
            row = "   ".join(f">={x}: hayalet -%{100 * np.mean(gh < x):.0f} / dogru -%{100 * np.mean(ok < x):.0f}"
                             for x in (0.3, 0.5, 0.7, 0.9))
            print(f"    tepe esigi ({name}, n={len(ok)}+{len(gh)}): {row}")


def main(ckpt, splits, onset_thr=None, dec_search=True, criterion="note"):
    device = get_device()
    model, ck, kind = load_model("crnn", device, ckpt)
    has_on = hasattr(model, "onset_head")
    on_grid = [None] + sorted({ck.get("onset_thr") or 0.5, 0.1, 0.2, 0.3, 0.5}) if has_on and onset_thr is None \
        else [None if (onset_thr is None or onset_thr < 0) else onset_thr]
    print(f"Model: {ckpt} | cihaz: {device}")
    cache = {}
    for split in splits:
        sel = selection_split(split, "auto")
        if sel not in cache:
            cache[sel] = load_split(sel, model, ck, kind, device)
        f1, thr, ot, dec = calibrate(cache[sel], on_grid, DEC_GRID if dec_search else None, criterion)
        data = cache[sel] if sel == split else load_split(split, model, ck, kind, device)
        print(f"\n{'=' * 78}\n{split}  ({len(data)} kayit) | esikler '{sel}' uzerinde: perde {thr}, "
              f"cozumleme {'kare-esik' if ot is None else f'onset@{ot}'}"
              + (f" {dec}" if ot is not None else "") + f"\n{'=' * 78}")
        report(analyse(data, thr, ot, dec))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="tabcrnn_onset_h.pt")
    ap.add_argument("--splits", nargs="+", default=["val_comp", "gaps_val"])
    ap.add_argument("--onset-thr", type=float, default=None, help="bos = dogrulamada sec; -1 = kare-esik")
    ap.add_argument("--no-dec-search", action="store_true", help="Katman 3.9 cozumlemesi (Adim 0 olcumu)")
    ap.add_argument("--criterion", default="note", choices=["note", "mix"],
                    help="esik secim olcutu: note = nota F1 (3.9); mix = (nota F1 + kare F1)/2")
    a = ap.parse_args()
    main(a.ckpt, a.splits, a.onset_thr, not a.no_dec_search, a.criterion)
