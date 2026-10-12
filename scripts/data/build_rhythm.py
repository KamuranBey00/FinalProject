"""
Katman 5 Adım 0 — Ritim etiketleri önbelleği (mevcut önbelleğe dokunulmaz).

    data/cache/rhythm/<id>.npz
        beats, downbeats (sn)                          GAPS + GuitarSet
        tempo (BPM; GuitarSet JAMS, GAPS = 60 / medyan vuruş aralığı)
        notes (N, 3) (başlangıç, bitiş, perde), note_beat (N,) vuruş konumu | nan       yalnız GAPS
    data/cache/rhythm/summary.csv

GAPS kullanılır kayıt: tekrar yapısı tutarlı (build_gaps_tab ile aynı ölçüt), eşleşen nota >= --min-matched,
>= 8 vuruş. GuitarSet: önbellekteki tüm kayıtlar (train, val, train_comp, val_comp).

Çalıştırma:
    python -m scripts.data.build_rhythm
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from gtab.config import GUITARSET_DATA_HOME
from gtab.core.instrument import STANDARD_6
from gtab.data.gaps import GAPS_DIR
from gtab.data.rhythm_labels import gaps_rhythm, guitarset_rhythm
from gtab.paths import CACHE_DIR

OUT_DIR = os.path.join(CACHE_DIR, "rhythm")


def main(min_matched):
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    gaps = sorted(glob.glob(os.path.join(CACHE_DIR, "gaps_train", "*.npz")) +
                  glob.glob(os.path.join(CACHE_DIR, "gaps_test", "*.npz")))
    for k, f in enumerate(gaps, 1):
        tid = os.path.basename(f)[:-4]
        r = gaps_rhythm(GAPS_DIR, tid, STANDARD_6.tuning)
        nb = r["note_beat"]
        use = bool(r["consistent"] and r["matched"] >= min_matched and len(r["beats"]) >= 8)
        tempo = 60 / np.median(np.diff(r["beats"])) if len(r["beats"]) >= 2 else np.nan
        frac = nb[~np.isnan(nb)] % 1
        rows.append(dict(id=tid, split=os.path.basename(os.path.dirname(f)), use=use, consistent=r["consistent"],
                         matched=round(r["matched"], 4), positioned=round(float((~np.isnan(nb)).mean()), 4),
                         beats=len(r["beats"]), downbeats=len(r["downbeats"]), tempo=round(tempo, 1),
                         sync_err_ms=round(1000 * r["sync_err"], 1),
                         on_beat=round(float(np.mean(np.minimum(frac, 1 - frac) < 1e-3)), 4) if len(frac) else np.nan))
        if use:
            np.savez_compressed(os.path.join(OUT_DIR, f"{tid}.npz"), beats=r["beats"], downbeats=r["downbeats"],
                                tempo=tempo, notes=r["notes"].astype(np.float32), note_beat=nb)
        print(f"[{k}/{len(gaps)}] {tid} {'YAZILDI' if use else 'atlandi'}  tutarli={r['consistent']} "
              f"eslesen %{100 * r['matched']:.0f}  vurus {len(r['beats'])}  olcu {len(r['downbeats'])}  "
              f"tempo {tempo:.0f}  syncpoint farki {1000 * r['sync_err']:.0f} ms", flush=True)

    for split in ("train", "val", "train_comp", "val_comp"):
        for f in sorted(glob.glob(os.path.join(CACHE_DIR, split, "*.npz"))):
            tid = os.path.basename(f)[:-4]
            g = guitarset_rhythm(os.path.join(GUITARSET_DATA_HOME, "annotation", f"{tid}.jams"))
            np.savez_compressed(os.path.join(OUT_DIR, f"{tid}.npz"), **g)
            rows.append(dict(id=tid, split=split, use=True, beats=len(g["beats"]), downbeats=len(g["downbeats"]),
                             tempo=g["tempo"]))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    u = df[df.use & df.split.str.startswith("gaps")]
    print("\n" + "=" * 80)
    for sp in ("gaps_train", "gaps_test"):
        a = df[df.split == sp]
        print(f"{sp:10s}: {int(a.use.sum())}/{len(a)} kayit kullanilir | tekrar tutarsiz {int((~a.consistent.astype(bool)).sum())}")
    print(f"GAPS kullanilan: eslesen nota %{100 * u.matched.mean():.1f} | konumlu nota %{100 * u.positioned.mean():.1f} | "
          f"vuruşta baslayan nota %{100 * u.on_beat.mean():.1f} | vurus {int(u.beats.sum())} | olcu basi {int(u.downbeats.sum())}")
    print(f"olcu basi: haritadan vs syncpoint medyan fark {u.sync_err_ms.median():.1f} ms (kayit basina medyanlarin medyani)")
    print(f"tempo (vurus birimi) medyan {u.tempo.median():.0f} BPM, aralik {u.tempo.min():.0f}-{u.tempo.max():.0f}")
    gs = df[~df.split.str.startswith("gaps")]
    print(f"GuitarSet: {len(gs)} kayit, vurus {int(gs.beats.sum())}, olcu basi {int(gs.downbeats.sum())}")
    print(f"-> {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-matched", type=float, default=0.5, help="GAPS kaydi icin en az eslesen nota orani")
    a = ap.parse_args()
    main(a.min_matched)
