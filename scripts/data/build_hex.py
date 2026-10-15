"""
ArmaganDegisiklikler — GuitarSet hex pickup -> per-string spectra cache (auxiliary target for train_onset --hex-weight).

For every GuitarSet cache record (train, train_comp, val, val_comp) writes
    data/cache/hex/<track_id>.npz   spec (T, 6, 96) float16, T = mic cache frames
and prints a per-split sanity line: frame-count match and how often the labelled string's hex
channel is the loudest one on single-string frames (should be close to 1).
Existing caches are not touched. Needs audio_hex-pickup_debleeded under GUITARSET_DATA_HOME.

Run (project root):
    python -m scripts.data.build_hex
"""

import argparse
from pathlib import Path

import numpy as np

from gtab.config import GUITARSET_DATA_HOME
from gtab.data.hex import hex_string_spectra, string_energy_agreement
from gtab.paths import CACHE_DIR

SPLITS = ("train", "train_comp", "val", "val_comp")
OUT_DIR = Path(CACHE_DIR) / "hex"


def main(splits) -> None:
    import mirdata
    tracks = mirdata.initialize("guitarset", data_home=GUITARSET_DATA_HOME).load_tracks()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sp in splits:
        files = sorted((Path(CACHE_DIR) / sp).glob("*.npz"))
        agree, missing = [], 0
        for k, f in enumerate(files, 1):
            tid = f.stem
            hex_path = Path(tracks[tid].audio_hex_cln_path)
            if not hex_path.exists():
                missing += 1
                continue
            with np.load(f) as d:
                tab = d["tab"]
            spec = hex_string_spectra(hex_path, len(tab))
            np.savez_compressed(OUT_DIR / f"{tid}.npz", spec=spec)
            agree.append(string_energy_agreement(spec, tab))
            if k % 25 == 0 or k == len(files):
                print(f"  {sp} [{k}/{len(files)}]", flush=True)
        a = np.array([x for x in agree if x == x])
        print(f"{sp:10s}: {len(agree)}/{len(files)} kayit | hex dosyasi yok {missing} | "
              f"tek telli karelerde en guclu kanal = etiket teli: ort %{100 * a.mean():.1f}, en dusuk %{100 * a.min():.1f}")
    print(f"-> {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    main(ap.parse_args().splits)
