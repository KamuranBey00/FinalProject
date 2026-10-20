"""Test B: same pieces, different guitars. GAPS test pieces with score-derived string/fret labels are re-rendered
with each local guitar sample set; the TAB pipeline runs on every rendering and on the real recording, all scored
against the same ground truth (repo convention: mir_eval note match, onset +-50 ms, offsets ignored).
Run from the repo root: python -m scripts.eval.gv.test_b_resynth docs/results/gv/test_b_trimmed.csv
"""
import csv
import sys
from pathlib import Path

import mir_eval
import numpy as np
import soundfile as sf

from gtab.data.features import cqt_from_file
from gtab.pipeline import analyze, load_models
from gtab.utils import get_device
from scripts.eval.gv import samplers
from scripts.eval.gv.paths import GAPS_TAB as TAB, GAPS_TEST as TEST, ICHIYANAGI, LIBRARIES, VIEWER_FONT


def libraries() -> dict[str, list[samplers.Sample]]:
    gv = LIBRARIES  # fetched / unpacked by scripts/data/fetch_gv_data.py
    # __MACOSX/._* are macOS metadata stubs inside the zips, not audio
    wavs = lambda pat: sorted(p for p in gv.rglob("*.wav") if "__MACOSX" not in p.parts and pat(p))
    return {
        "freepats_viewer": samplers.from_viewer_font(VIEWER_FONT),
        "custom_classical_sf2": samplers.from_soundfont(gv / "Custom Classical Guitar.sf2"),
        "ichiyanagi_sf3": samplers.from_soundfont(ICHIYANAGI),
        "mg_soft_nylon": samplers.from_wavs(wavs(lambda p: "MG_NylonGuitar" in p.name)),
        "mdm_classical": samplers.from_wavs(wavs(lambda p: "MDM" in str(p))),
        "rjs_lofi_nylon": samplers.from_wavs(wavs(lambda p: p.name.startswith("Nylon_"))),
    }


def mir(notes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    iv = np.column_stack([notes[:, 0], np.maximum(notes[:, 1], notes[:, 0] + 0.01)])
    return iv, mir_eval.util.midi_to_hz(notes[:, 2])


def score(an: dict, ref: np.ndarray, ref_string: np.ndarray) -> dict[str, float]:
    est = an["notes"]
    if len(est) == 0:
        return {"P": 0.0, "R": 0.0, "F": 0.0, "string_acc": float("nan"), "n_est": 0}
    pairs = mir_eval.transcription.match_notes(*mir(ref), *mir(est), onset_tolerance=0.05, offset_ratio=None)
    tp = len(pairs)
    P, R = tp / len(est), tp / len(ref)
    known = [(i, j) for i, j in pairs if ref_string[i] >= 0]
    acc = np.mean([an["string"][j] == ref_string[i] for i, j in known]) if known else float("nan")
    return {"P": P, "R": R, "F": 2 * P * R / (P + R) if tp else 0.0, "string_acc": float(acc), "n_est": len(est)}


def main(out: Path) -> None:
    m, libs = load_models(get_device()), libraries()
    ids = sorted({p.stem for p in TEST.glob("*.npz")} & {p.stem for p in TAB.glob("*.npz")})
    tmp = out.parent / "render_tmp.wav"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "source", "P", "R", "F", "string_acc", "n_est", "n_ref"])
        w.writeheader()
        for k, tid in enumerate(ids):
            with np.load(TAB / f"{tid}.npz") as d:
                ref, ref_string = d["notes"].astype(float), d["string"].astype(int)
            with np.load(TEST / f"{tid}.npz") as d:
                cqt = d["cqt"]
            w.writerow({"id": tid, "source": "real_recording", **score(analyze(cqt, m, "ses"), ref, ref_string),
                        "n_ref": len(ref)})
            for name, lib in libs.items():
                sf.write(tmp, samplers.render(ref, lib), samplers.SR)
                w.writerow({"id": tid, "source": name, **score(analyze(cqt_from_file(str(tmp)), m, "ses"), ref,
                                                                 ref_string), "n_ref": len(ref)})
            f.flush()
            print(f"[{k + 1}/{len(ids)}] {tid}", flush=True)
    tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
