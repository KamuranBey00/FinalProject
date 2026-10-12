"""
Katman 5 Adım 2 — Ses -> ölçülü TAB (MusicXML). Ritim yarı otomatik: kullanıcı yaklaşık BPM + ölçü imi verir.

Girdi: ses dosyası (wav / mp3 / flac ...) ya da önbellek .npz (cqt). Çıktı: .musicxml (nota portresi + TAB portresi;
MuseScore / Guitar Pro / AlphaTab açar). BPM, ölçü iminin vuruş birimine göre: x/4 çeyrek, x/2 ikilik, 6/8 - 9/8 - 12/8
noktalı çeyrek (bileşik), diğer x/8 sekizlik. Kaba tahmin yeterli (± %35 içinde rubato sesten izlenir).
Tel/perde: --mode ses (duyulan konum) | edisyon (klasik edisyon önerisi + sol el parmağı).

Çalıştırma:
    python -m scripts.app.transcribe kayit.wav --bpm 80 --meter 3/4
    python -m scripts.app.transcribe data/cache/gaps_test/<id>.npz --bpm 66 --meter 6/8 --mode edisyon --out cikti.musicxml
"""

import argparse
import os

import numpy as np

from gtab.data.features import cqt_from_file
from gtab.pipeline import load_models, transcribe
from gtab.render.musicxml import write_musicxml
from gtab.utils import get_device


def main(src, bpm, meter, mode, out):
    sig = tuple(int(x) for x in meter.split("/"))
    if src.endswith(".npz"):
        with np.load(src) as d:
            cqt = d["cqt"]
    else:
        cqt = cqt_from_file(src)
    m = load_models(get_device())
    r = transcribe(cqt, m, bpm, sig, mode)
    out = out or os.path.splitext(os.path.basename(src))[0] + ".musicxml"
    write_musicxml(out, r["measures"], r["notes"][:, 2].astype(int), r["string"], r["fret"], sig, r["beat_q"], bpm=bpm,
                   finger=r["finger"] if mode == "edisyon" else None, title=os.path.basename(src))
    n_pos = int((r["string"] >= 0).sum())
    print(f"{len(r['notes'])} nota ({n_pos} tel/perde konumlu) | {len(r['beats'])} vurus, "
          f"{len(r['downbeats'])} olcu basi | {len(r['measures'])} olcu {meter} | mod {mode} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="ses dosyasi ya da onbellek .npz")
    ap.add_argument("--bpm", type=float, required=True, help="yaklasik tempo (olcu iminin vurus birimine gore)")
    ap.add_argument("--meter", required=True, help="olcu imi, orn. 3/4, 2/2, 6/8")
    ap.add_argument("--mode", default="ses", choices=["ses", "edisyon"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(a.src, a.bpm, a.meter, a.mode, a.out)
