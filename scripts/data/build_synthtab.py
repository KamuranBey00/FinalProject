"""
Katman 3.8d — SynthTab -> önbellek (CQT + tel/fret 'tab' etiketi).

Okuma mantığı ve DOĞRULANMIŞ dosya yapısı: gtab/data/synthtab.py
Zip'ler açılmadan okunur. Varsayılan olarak yalnızca NAYLON tını ('luthier_*')
alınır ve standart akortta olmayan parçalar atlanır.

DİSK: ~73 GB boş -> --max-hours ile sınırla (30 sa CQT float16 ≈ 3.5 GB).

Çalıştırma:
    python -m scripts.data.build_synthtab --inspect --audio data/raw/synthtab/SynthTab_Dev.zip
    python -m scripts.data.build_synthtab --audio data/raw/synthtab/SynthTab_Dev.zip --split synthtab_dev
    # tam set: indirilen luthier zip'lerinin klasörü + etiket zip'i
    python -m scripts.data.build_synthtab --audio data/raw/synthtab/acoustic \
        --labels data/raw/synthtab/all_jams_midi_V2_60000_tracks.zip --max-hours 30
"""

import argparse
import io
import os
import re
from collections import Counter

import librosa
import numpy as np
import soundfile as sf

from gtab.config import SAMPLE_RATE, FRAME_RATE
from gtab.data.features import compute_cqt
from gtab.data.synthtab import index, string_notes, notes_to_tab
from gtab.paths import CACHE_DIR


def _safe(name):
    return re.sub(r"[^\w\-.]+", "_", name)[:150]


def load_audio(src, f):
    y, sr = sf.read(io.BytesIO(src.read(f)), dtype="float32", always_2d=True)
    return librosa.resample(y.mean(1), orig_sr=sr, target_sr=SAMPLE_RATE)


def inspect(audio_path, label_path, timbre):
    audio, labels = index(audio_path, label_path, timbre)
    print(f"ses ornegi (tini={timbre or 'hepsi'}): {len(audio)} | etiketli parca: {len(labels)}")
    print("tini dagilimi:", Counter(t for t, _ in audio).most_common())
    ok, why = 0, Counter()
    for (t, k) in sorted(audio):
        notes, info = string_notes(labels[k]) if k in labels else (None, "etiket yok")
        if notes is None:
            why[info.split(" ")[0]] += 1
        else:
            ok += 1
    print(f"kullanilabilir: {ok} | atlanan: {dict(why)}")
    if ok:
        t, k = next((t, k) for (t, k) in sorted(audio) if k in labels and string_notes(labels[k])[0])
        notes, tun = string_notes(labels[k])
        print(f"ornek: [{t}] {k}\n  akort {tun} | tel->nota {[len(notes[s]) for s in sorted(notes)]}")


def build(audio_path, label_path, timbre, max_hours, split):
    audio, labels = index(audio_path, label_path, timbre)
    out_dir = os.path.join(CACHE_DIR, split); os.makedirs(out_dir, exist_ok=True)
    total, skipped = 0.0, Counter()
    for i, (t, k) in enumerate(sorted(audio), 1):
        out = os.path.join(out_dir, _safe(f"{t}__{k}") + ".npz")
        if os.path.exists(out):                       # kaldığı yerden devam
            with np.load(out) as d:
                total += d["cqt"].shape[0] / FRAME_RATE / 3600
            continue
        if total >= max_hours:
            break
        notes, info = string_notes(labels[k]) if k in labels else (None, "etiket yok")
        if notes is None:
            skipped[info.split(" ")[0]] += 1
            continue
        try:
            cqt = compute_cqt(load_audio(*audio[(t, k)]))
        except Exception as e:                        # bozuk ses: atla, devam et
            skipped["ses hatasi"] += 1
            print(f"  atlandi {k}: {e}")
            continue
        tab = notes_to_tab(notes, cqt.shape[0])
        np.savez_compressed(out, cqt=cqt.astype(np.float16), tab=tab)
        total += cqt.shape[0] / FRAME_RATE / 3600
        print(f"[{i}/{len(audio)}] {t:16s} {k[:60]}  {cqt.shape[0] / FRAME_RATE:6.1f}s  toplam {total:.2f} sa")
    print(f"Bitti: {total:.2f} saat -> {out_dir} | atlanan: {dict(skipped)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="SynthTab ses klasoru ya da zip")
    ap.add_argument("--labels", default=None, help="jams/MIDI klasoru ya da zip (bos = --audio icinde)")
    ap.add_argument("--timbre", default="luthier", help="tini filtresi; 'luthier' = naylon, '' = hepsi")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--max-hours", type=float, default=30.0)
    ap.add_argument("--split", default="synthtab_train")
    a = ap.parse_args()
    if a.inspect:
        inspect(a.audio, a.labels, a.timbre or None)
    else:
        build(a.audio, a.labels, a.timbre or None, a.max_hours, a.split)
