"""
Katman 5 Adım 2–3 — Ses -> ölçülü TAB (MusicXML). Ritim kullanıcı girdisiyle: yaklaşık BPM (yarı otomatik) ya da
vuruş dokunuşları (tap modu, daha doğru; README17) + ölçü imi.

Girdi: ses dosyası (wav / mp3 / flac ...) ya da önbellek .npz (cqt). Çıktı: .musicxml (nota portresi + TAB portresi;
MuseScore / Guitar Pro / AlphaTab açar). BPM, ölçü iminin vuruş birimine göre: x/4 çeyrek, x/2 ikilik, 6/8 - 9/8 - 12/8
noktalı çeyrek (bileşik), diğer x/8 sekizlik. Kaba tahmin yeterli (± %35 içinde rubato sesten izlenir).
Tel/perde: --mode ses (duyulan konum) | edisyon (klasik edisyon önerisi + sol el parmağı).
Ölçü çizgileri yanlış vuruşta başlıyorsa --shift k ile k vuruş ileri kaydırılır (ör. 3/4'te 1 ya da 2).
Tap modu: --taps dosya = kaydı dinlerken İLK ÖLÇÜ BAŞINDAN başlayarak her vuruşa dokunulan zamanlar (sn; satır, boşluk
ya da virgülle ayrılmış). Gecikme, kaçan / fazla dokunuş ve ölçü fazı otomatik düzeltilir; --bpm gerekmez.

Çalıştırma:
    python -m scripts.app.transcribe kayit.wav --bpm 80 --meter 3/4
    python -m scripts.app.transcribe data/cache/gaps_test/<id>.npz --bpm 66 --meter 6/8 --mode edisyon --out cikti.musicxml
    python -m scripts.app.transcribe kayit.wav --taps dokunuslar.txt --meter 3/4
"""

import argparse
import os

import numpy as np

from gtab.data.features import cqt_from_file
from gtab.decoding.taps import load_taps
from gtab.pipeline import load_models, transcribe
from gtab.render.musicxml import write_musicxml
from gtab.utils import get_device


def main(src, bpm, meter, mode, out, shift, taps_file):
    sig = tuple(int(x) for x in meter.split("/"))
    if src.endswith(".npz"):
        with np.load(src) as d:
            cqt = d["cqt"]
    else:
        cqt = cqt_from_file(src)
    taps = load_taps(taps_file) if taps_file else None
    m = load_models(get_device())
    r = transcribe(cqt, m, bpm, sig, mode, shift=shift, taps=taps)
    bpm = bpm or 60 / np.median(np.diff(r["beats"]))                    # tap modu: metronom işareti dokunuşlardan
    out = out or os.path.splitext(os.path.basename(src))[0] + ".musicxml"
    write_musicxml(out, r["measures"], r["notes"][:, 2].astype(int), r["string"], r["fret"], sig, r["beat_q"], bpm=bpm,
                   finger=r["finger"] if mode == "edisyon" else None, title=os.path.basename(src))
    n_pos = int((r["string"] >= 0).sum())
    print(f"{len(r['notes'])} nota ({n_pos} tel/perde konumlu) | {len(r['beats'])} vurus, "
          f"{len(r['downbeats'])} olcu basi | {len(r['measures'])} olcu {meter} | mod {mode} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="ses dosyasi ya da onbellek .npz")
    ap.add_argument("--bpm", type=float, default=None, help="yaklasik tempo (olcu iminin vurus birimine gore)")
    ap.add_argument("--meter", required=True, help="olcu imi, orn. 3/4, 2/2, 6/8")
    ap.add_argument("--mode", default="ses", choices=["ses", "edisyon"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--shift", type=int, default=0, help="olcu cizgilerini k vurus ileri kaydir")
    ap.add_argument("--taps", default=None, help="tap modu: vurus dokunuslari dosyasi (sn)")
    a = ap.parse_args()
    if a.bpm is None and a.taps is None:
        ap.error("--bpm ya da --taps gerekli")
    main(a.src, a.bpm, a.meter, a.mode, a.out, a.shift, a.taps)
