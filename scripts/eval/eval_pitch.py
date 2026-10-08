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

Eşik seçimi (Katman 3.9 kalibrasyonu): varsayılan --select-split auto = her test
setinin eşikleri KENDİ ALANININ doğrulamasında seçilir:
    GuitarSet (val, val_comp, ...) -> 'val'      (oyuncu 05 solo)
    GAPS (gaps_test, ...)          -> 'gaps_val' (gaps_train içinden icracı-ayrık
                                       doğrulama; train_domain/train_onset ile AYNI bölme)
Onset kafalı modelde çözümleme yöntemi de (onset'li / eski kare-eşik) doğrulamada,
nota F1 ile seçilir. Test setlerine seçimde HİÇ bakılmaz.
--select-split val = eski davranış (her şey GuitarSet val'de; Katman 3.8–3.9 tabloları).

Çalıştırma:
    python -m scripts.eval.eval_pitch --model crnn --ckpt tabcrnn_onset_h.pt --splits gaps_test val val_comp
"""

import argparse

from gtab.evaluation.pitch_eval import (DEC_GRID, calibrate, evaluate_split, load_split,
                                        selection_split)
from gtab.models.inference import load_model
from gtab.utils import get_device


# ----------------------------------------------------------------- ana akış
def main(kind, ckpt, splits, select_split, limit=None, onset_thr=None, dec_search=True, criterion="note",
         force_rise=None, peak_search=True):
    device = get_device()
    model, ck, kind = load_model(kind, device, ckpt)
    has_on = hasattr(model, "onset_head")
    if not has_on or (onset_thr is not None and onset_thr < 0):
        on_grid = [None]                                  # yalniz eski kare-esik cozumlemesi
    elif onset_thr is not None:
        on_grid = [onset_thr]                             # kullanici sabitledi
    else:                                                 # yontem + esik dogrulamada secilir
        on_grid = [None] + sorted({ck.get("onset_thr") or 0.5, 0.1, 0.2, 0.3, 0.5})
    print(f"Model: {kind} ({ckpt or 'varsayilan'}) | cihaz: {device}")
    print(f"Cozumleme adaylari: " + ", ".join("kare-esik" if o is None else f"onset@{o}" for o in on_grid)
          + f" | esik secimi: {select_split}")

    cal = {}                                              # dogrulama split'i -> secim
    def get_cal(sp):
        if sp not in cal:
            cal[sp] = calibrate(load_split(sp, model, ck, kind, device, limit), on_grid,
                                DEC_GRID if dec_search else None, criterion, peak_search)
            f1, t, o, dec = cal[sp]
            print(f"  [{sp}] secim: perde esigi={t}, cozumleme="
                  f"{'kare-esik' if o is None else f'onset@{o}'}"
                  + (f", histerezis={dec['off_ratio']}, refrakter={dec['refractory']}, yedek={dec['fallback']}"
                     f", tepe={dec.get('peak', False)}, yeniden_vurus={dec.get('reattack', 0.0)}"
                     f", enerji_kabul={dec.get('rise_keep', 0.0)}, enerji_bol={dec.get('rise_split', 0.0)}"
                     f", offset={dec.get('offset_threshold', 0.0)}, birlestirme={dec.get('combine', 'max')}"
                     f", yerel_tepe={dec.get('peak_pick', False)}"
                     + (f" (vadi={dec['prominence']}, ara={dec['min_dist']})" if dec.get('peak_pick') else "")
                     if o is not None else "") + f" (dogrulama skoru {f1:.3f})")
        return cal[sp]

    summary = []
    for split in splits:
        sp = selection_split(split, select_split)
        _, thr, ot, dec = get_cal(sp)
        if force_rise is not None and ot is not None:   # ablasyon: enerji kurallarını sabitle
            dec = dict(dec, rise_keep=force_rise[0], rise_split=force_rise[1])
            print(f"  [{split}] ABLASYON: enerji_kabul={force_rise[0]}, enerji_bol={force_rise[1]} sabitlendi "
                  f"(digerleri '{sp}' secimi)")
        res = evaluate_split(load_split(split, model, ck, kind, device, limit), onset_thr=ot, dec=dec)
        print(f"\n=== {split}  (esikler '{sp}' uzerinde secildi; cozumleme: "
              f"{'kare-esik' if ot is None else f'onset@{ot}'}) ===")
        print(f"  {'esik':>5} | {'nota P':>7} {'R':>6} {'F1':>6} | {'kare P':>7} {'R':>6} {'F1':>6}")
        for t, r in res.items():
            mark = "  <- secilen" if t == thr else ""
            print(f"  {t:5.2f} | {r['note'][0]:7.3f} {r['note'][1]:6.3f} {r['note'][2]:6.3f} | "
                  f"{r['frame'][0]:7.3f} {r['frame'][1]:6.3f} {r['frame'][2]:6.3f}{mark}")
        r = res[thr]
        print("  polifoni (kare F1, secilen esik): " +
              "  ".join(f"{g}: {v[2]:.3f}" for g, v in sorted(r["poly"].items())))
        summary.append((split, sp, thr, ot, r))

    print("\n" + "=" * 92)
    print(f"{'split':<12}{'secim':<10}{'cozumleme':<14}{'esik':>5}{'nota F1':>10}{'@100ms':>8}{'kare F1':>10}   polifoni 1 / 2 / 3 / 4+")
    print("-" * 100)
    for split, sp, thr, ot, r in summary:
        pol = " / ".join(f"{r['poly'][g][2]:.2f}" if g in r["poly"] else "  - "
                         for g in ("1", "2", "3", "4+"))
        dec = "kare-esik" if ot is None else f"onset@{ot}"
        print(f"{split:<12}{sp:<10}{dec:<14}{thr:>5}{r['note'][2]:>10.3f}{r['note100'][2]:>8.3f}{r['frame'][2]:>10.3f}   {pol}")
    print("=" * 100)
    print("@100ms = ayni secimle 100 ms onset toleransinda nota F1 (fark buyukse hata zamanlamada).")
    print("Esikler ve cozumleme yontemi yalnizca dogrulama split'lerinde secildi; test setlerine bakilmadi.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="crnn", choices=["cnn", "crnn"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--splits", nargs="+", default=["gaps_test", "val", "val_comp"])
    ap.add_argument("--select-split", default="auto",
                    help="auto = her test setinin esigi kendi alaninin dogrulamasinda (GuitarSet->val, "
                         "GAPS->gaps_val); 'val' = eski davranis. Test split'i OLMAMALI.")
    ap.add_argument("--limit", type=int, default=None, help="split basina en fazla kayit (hizli deneme)")
    ap.add_argument("--no-dec-search", action="store_true",
                    help="Katman 3.10 cozumleme kurallarini (histerezis/refrakter/yedek) arama; 3.9 davranisi")
    ap.add_argument("--onset-thr", type=float, default=None,
                    help="onset esigi (bos = checkpoint'te dogrulamada secilen); -1 = eski kare-esik cozumlemesi")
    ap.add_argument("--force-rise", type=float, nargs=2, default=None, metavar=("KABUL_DB", "BOL_DB"),
                    help="ablasyon: enerji kurallarini tum splitlerde bu degerlere sabitle (or. 4 6)")
    ap.add_argument("--criterion", default="note", choices=["note", "mix"],
                    help="esik secim olcutu: note = nota F1 (3.9); mix = (nota F1 + kare F1)/2")
    ap.add_argument("--no-peak-search", action="store_true",
                    help="Katman 3.11 aramasini (noisy-OR, yerel tepe) yapma; 3.10 secimi birebir")
    a = ap.parse_args()
    main(a.model, a.ckpt, a.splits, a.select_split, a.limit, a.onset_thr, not a.no_dec_search, a.criterion,
         a.force_rise, not a.no_peak_search)
