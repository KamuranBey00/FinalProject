"""
Katman 3.13 (README9 3.8e) — GAPS partisyonundan tel/perde etiketi + sol el parmak numarası.

Her GAPS önbellek kaydı (data/cache/gaps_{train,test}/<id>.npz) için partisyon TAB'ı hizalı MIDI notalarına
eşlenir (gtab.data.gaps_score.align) ve AYRI bir dosyaya yazılır (mevcut önbelleğe dokunulmaz):
    data/cache/gaps_tab/<id>.npz
        tab    (L, 6) int64  GuitarSet şeması: 0 = sessiz, fret + 1 (indeks 0 = pes E)
        known  (L,)   bool   karedeki TÜM notalar etiketli (değerlendirme/kayıp yalnız bu karelerde)
        finger (L, 6) int8   sol el parmağı (0 = boş tel, 1–4) | -1 = bilinmiyor
        notes  (N, 3) float  MIDI notaları (başlangıç sn, bitiş sn, perde); string/fret/finger (N,) | -1
    data/cache/gaps_tab/summary.csv   kayıt başına kapsama + kullanılır mı

Kullanılır kayıt: tekrar yapısı syncpoint'lerle tutarlı (açılmış ölçü sayısı = syncpoint ölçü sayısı) ve
notaların >= --min-labeled'i etiketli. Diğerleri yazılmaz (perde denetimi önbellekten aynen sürer).

Çalıştırma:
    python -m scripts.data.build_gaps_tab
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from gtab.config import FRAME_RATE
from gtab.core.instrument import STANDARD_6
from gtab.data.gaps import GAPS_DIR, GAPS_TAB_DIR
from gtab.data.gaps_score import align
from gtab.paths import CACHE_DIR

OUT_DIR = GAPS_TAB_DIR


def frame_labels(r, n_frames):
    """Nota düzeyi etiketler -> (tab, known, finger) kare dizileri (notes_to_rolls ile aynı yuvarlama)."""
    tab = np.zeros((n_frames, 6), np.int64)
    finger = np.full((n_frames, 6), -1, np.int8)
    unknown = np.zeros(n_frames, bool)
    for (st, en, _), s, f, g in zip(r["notes"], r["string"], r["fret"], r["finger"]):
        a = max(0, min(int(round(st * FRAME_RATE)), n_frames - 1))
        b = max(a + 1, min(int(round(en * FRAME_RATE)), n_frames))
        if s < 0:
            unknown[a:b] = True
        else:
            tab[a:b, s] = f + 1
            finger[a:b, s] = g
    return tab, ~unknown, finger


def main(min_labeled):
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "gaps_train", "*.npz")) +
                   glob.glob(os.path.join(CACHE_DIR, "gaps_test", "*.npz")))
    for k, f in enumerate(files, 1):
        tid = os.path.basename(f)[:-4]
        r = align(GAPS_DIR, tid, STANDARD_6.tuning)
        lab = r["string"] >= 0
        use = bool(r["consistent"] and lab.mean() >= min_labeled)
        with np.load(f) as d:
            frame = d["frame"] > 0
        tab, known, finger = frame_labels(r, len(frame))
        # tutarlılık: etiketli karelerde TAB'dan türeyen perde roll'u == önbellekteki perde roll'u
        roll = np.zeros_like(frame)
        for s in range(6):
            on = tab[:, s] > 0
            roll[np.flatnonzero(on), STANDARD_6.tuning[s] + tab[on, s] - 1 - STANDARD_6.tuning[0]] = True
        kn = known & frame.any(1)
        agree = (roll[kn] == frame[kn]).all(1).mean() if kn.any() else np.nan
        gl = r["finger"][lab]
        rows.append(dict(id=tid, split=os.path.basename(os.path.dirname(f)), use=use, consistent=r["consistent"],
                         shift=r["shift"], notes=len(lab), labeled=round(lab.mean(), 4),
                         known_frames=round(kn.sum() / max(frame.any(1).sum(), 1), 4), roll_agree=round(agree, 4),
                         fingered=round((gl >= 0).mean(), 4) if len(gl) else 0.0,
                         open_finger0=round(((gl == 0) == (r["fret"][lab] == 0))[gl >= 0].mean(), 4) if (gl >= 0).any() else np.nan,
                         dt_med_ms=round(1000 * np.median(np.abs(r["dt"])), 1) if len(r["dt"]) else np.nan))
        if use:
            np.savez_compressed(os.path.join(OUT_DIR, f"{tid}.npz"), tab=tab, known=known, finger=finger,
                                notes=np.array(r["notes"], np.float32), string=r["string"].astype(np.int8),
                                fret=r["fret"].astype(np.int8), finger_note=r["finger"].astype(np.int8))
        print(f"[{k}/{len(files)}] {tid} {'YAZILDI' if use else 'atlandi'}  tutarli={r['consistent']} "
              f"etiketli %{100 * lab.mean():.0f}  bilinen kare %{100 * rows[-1]['known_frames']:.0f}  "
              f"kapo/kayma {r['shift']:+d}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    u = df[df.use]
    w = u.notes
    print("\n" + "=" * 80)
    for sp in ("gaps_train", "gaps_test"):
        a, b = df[df.split == sp], u[u.split == sp]
        print(f"{sp:10s}: {len(b)}/{len(a)} kayit kullanilir | tekrar tutarsiz {int((~a.consistent).sum())} | "
              f"az etiketli {int((a.consistent & ~a.use).sum())}")
    print(f"kullanilan kayitlarda: etiketli nota %{100 * np.average(u.labeled, weights=w):.1f} | bilinen kare "
          f"%{100 * np.average(u.known_frames, weights=w):.1f} | perde roll uyumu %{100 * np.average(u.roll_agree, weights=w):.2f}")
    print(f"parmak numarali (etiketli notalarin) %{100 * np.average(u.fingered, weights=w * u.labeled):.1f} | "
          f"'parmak 0 <-> bos tel' tutarliligi %{100 * np.nanmean(u.open_finger0):.1f} | |dt| medyan {u.dt_med_ms.median():.1f} ms")
    print(f"kapo/perde kaymasi olan kayit: {int((u['shift'] != 0).sum())}")
    print(f"-> {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-labeled", type=float, default=0.5, help="kayit kullanilmak icin en az etiketli nota orani")
    a = ap.parse_args()
    main(a.min_labeled)
