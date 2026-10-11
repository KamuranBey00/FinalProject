"""
Katman 4 Faz 2A — GAPS'te sürekli F0 elde edilebilir mi? (SALT OKUNUR, sınırlı)

GAPS'te tel başına referans YOK; elde yalnız hizalı MIDI (karışık, polifonik) var. Bu script
üç soruyu ölçer, hiçbir dosyaya yazmaz:

  1. Monofonik / polifonik pasajlar mevcut veriyle ayrılabiliyor mu?
     Ölçüt: önbellekteki hizalı MIDI perde roll'u (`frame`) -> karede aktif perde sayısı.
  2. Önbellekteki CQT'den ince perde (vibrato ölçecek hassasiyette) okunabiliyor mu?
     Beklenen bin etrafında (±3 bin) tepe + parabolik enterpolasyon; CQT çözünürlüğü
     50 cent/bin (24 bin/oktav).
  3. (seçimlik) Gerçek bir F0 takipçisi (librosa pyin) sesten daha iyisini yapıyor mu?
     Yalnız birkaç kayıt ve kısa süre; GAPS sesi salt okunur açılır.

Protokol: yalnız gaps_fit + gaps_val (`gaps_training_files`), karantina çıkarılmış,
**gaps_test hiç açılmaz**.

Çalıştırma:
    python -m scripts.eval.probe_gaps_f0 --records 12
    python -m scripts.eval.probe_gaps_f0 --records 12 --pyin-records 2 --pyin-seconds 15
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from gtab.config import CQT_BINS_PER_OCTAVE, CQT_FMIN_HZ, FRAME_RATE, HOP_LENGTH, SAMPLE_RATE
from gtab.core.instrument import STANDARD_6
from gtab.data.gaps import GAPS_DIR, load_metadata
from gtab.data.labels import midi_range
from gtab.evaluation.protocol import gaps_training_files

CENTS_PER_BIN = 1200.0 / CQT_BINS_PER_OCTAVE                 # 50 cent
FMIN_MIDI = 69.0 + 12.0 * np.log2(CQT_FMIN_HZ / 440.0)       # C1 ~ 24
LO = midi_range(STANDARD_6)[0]


def poly_profile(frame: np.ndarray) -> Dict[str, float]:
    """Hizalı MIDI roll'u -> monofonik / polifonik kare profili."""
    k = frame.sum(1)
    active = k > 0
    n = int(active.sum())
    return {"kare": int(len(k)), "aktif": n,
            "mono": int((k == 1).sum()), "poli": int((k >= 2).sum()),
            "mono_oran": float((k == 1).sum() / max(n, 1)),
            "mono_sn": float((k == 1).sum() / FRAME_RATE)}


def cqt_cents(cqt: np.ndarray, frames: np.ndarray, pitches: np.ndarray,
              half_width: int = 3) -> np.ndarray:
    """
    Beklenen bin çevresinde tepe + parabolik enterpolasyon -> referans perdeden sapma (cent).
    Perde biliniyor varsayımıyla çalışır: ölçülen şey "ince perde çözünürlüğü", perde bulma değil.
    """
    out = []
    for t, p in zip(frames, pitches):
        centre = int(round((p - FMIN_MIDI) * CQT_BINS_PER_OCTAVE / 12.0))
        a, b = centre - half_width, centre + half_width + 1
        if a < 1 or b > cqt.shape[1] - 1:
            continue
        seg = cqt[t, a:b].astype(np.float64)
        j = int(np.argmax(seg)) + a
        y0, y1, y2 = cqt[t, j - 1], cqt[t, j], cqt[t, j + 1]
        den = y0 - 2 * y1 + y2
        delta = 0.0 if den == 0 else 0.5 * (y0 - y2) / den
        delta = float(np.clip(delta, -1.0, 1.0))
        out.append((j + delta - (p - FMIN_MIDI) * CQT_BINS_PER_OCTAVE / 12.0) * CENTS_PER_BIN)
    return np.array(out)


def pyin_cents(wav: Path, seconds: float, frames: np.ndarray, pitches: np.ndarray) -> Tuple:
    """librosa pyin ile F0 -> referans perdeden sapma (cent). Ses salt okunur açılır."""
    import librosa
    y, _ = librosa.load(wav, sr=SAMPLE_RATE, mono=True, duration=seconds)
    t0 = time.time()
    f0, voiced, _ = librosa.pyin(y, fmin=70.0, fmax=1200.0, sr=SAMPLE_RATE,
                                 frame_length=2048, hop_length=HOP_LENGTH)
    dur = time.time() - t0
    n = len(f0)
    keep = frames < n
    fr, pt = frames[keep], pitches[keep]
    ok = np.isfinite(f0[fr]) & voiced[fr]
    cents = (69.0 + 12.0 * np.log2(np.maximum(f0[fr][ok], 1e-9) / 440.0) - pt[ok]) * 100.0
    return cents, float(ok.mean() if len(ok) else 0.0), dur, len(y) / SAMPLE_RATE


def summarize(name: str, cents: np.ndarray) -> None:
    if len(cents) == 0:
        print(f"  {name}: ornek yok")
        return
    a = np.abs(cents)
    print(f"  {name}: n={len(cents):7d} | medyan |hata| {np.median(a):6.1f} cent | "
          f"p90 {np.percentile(a, 90):6.1f} | p99 {np.percentile(a, 99):7.1f} | "
          f"|hata|<=25c orani %{100 * (a <= 25).mean():.1f}")


def pick_records(n: int) -> List[Path]:
    fit, val = gaps_training_files()                         # metin yollar
    files = sorted(Path(p) for p in fit + val)
    step = max(1, len(files) // max(n, 1))
    return files[::step][:n]


def main(n_records: int, pyin_records: int, pyin_seconds: float, half_width: int) -> None:
    files = pick_records(n_records)
    print(f"Katman 4 Faz 2A — GAPS F0 incelemesi (salt okunur)\n"
          f"{len(files)} kayit (gaps_fit + gaps_val; gaps_test ACILMADI, karantina haric)\n")

    print("== 1) Monofonik / polifonik ayrim (hizali MIDI roll'u) ==")
    profiles, cqt_err, mono_store = [], [], {}
    for f in files:
        with np.load(f) as d:
            frame = d["frame"]
            cqt = d["cqt"].astype(np.float32)
        pr = poly_profile(frame)
        profiles.append(pr)
        k = frame.sum(1)
        mono = np.flatnonzero(k == 1)
        pitches = np.array([LO + int(np.argmax(frame[t])) for t in mono])
        mono_store[f.stem] = (mono, pitches)
        cqt_err.append(cqt_cents(cqt, mono, pitches, half_width))
    mono_oran = np.array([p["mono_oran"] for p in profiles])
    print(f"  aktif kare basina monofonik oran: medyan %{100 * np.median(mono_oran):.1f} | "
          f"min %{100 * mono_oran.min():.1f} | max %{100 * mono_oran.max():.1f}")
    print(f"  toplam: aktif {sum(p['aktif'] for p in profiles)} kare | "
          f"mono {sum(p['mono'] for p in profiles)} ({sum(p['mono_sn'] for p in profiles):.0f} sn) | "
          f"poli {sum(p['poli'] for p in profiles)}")
    print("  -> ayrim MUMKUN: olcut hizali MIDI referansi (ses degil); yalniz referans "
          "varken kullanilabilir.")

    print(f"\n== 2) Onbellekteki CQT'den ince perde (cozunurluk {CENTS_PER_BIN:.0f} cent/bin) ==")
    summarize("CQT tepe + parabolik", np.concatenate(cqt_err) if cqt_err else np.zeros(0))
    print("  NOT: referans perde tam sayi MIDI; 'hata' = takipci hatasi + gercek perde sapmasi "
          "(vibrato/akort). GAPS'te tel basina referans olmadigi icin ikisi AYRILAMAZ.")

    if pyin_records > 0:
        print(f"\n== 3) librosa pyin ({pyin_records} kayit x {pyin_seconds:.0f} sn) ==")
        meta = load_metadata().set_index("id")
        for f in files[:pyin_records]:
            rel = meta.audio_path.get(f.stem)
            wav = Path(GAPS_DIR) / str(rel)
            if not rel or not wav.exists():
                print(f"  {f.stem}: ses bulunamadi, atlandi")
                continue
            mono, pitches = mono_store[f.stem]
            cents, vrate, dur, audio_s = pyin_cents(wav, pyin_seconds, mono, pitches)
            summarize(f"pyin {f.stem}", cents)
            print(f"    voiced orani {100 * vrate:.0f}% | sure {dur:.1f} sn / {audio_s:.0f} sn ses "
                  f"({dur / max(audio_s, 1e-9):.2f}x gercek zaman)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=int, default=12, help="incelenecek GAPS kaydi")
    ap.add_argument("--pyin-records", type=int, default=0, help="pyin denenecek kayit sayisi")
    ap.add_argument("--pyin-seconds", type=float, default=15.0)
    ap.add_argument("--half-width", type=int, default=3, help="CQT tepe arama yarim genisligi (bin)")
    a = ap.parse_args()
    main(a.records, a.pyin_records, a.pyin_seconds, a.half_width)
