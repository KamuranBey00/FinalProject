"""Test A: single-note sample libraries (one plucked note per file) through the TAB pipeline.

Reference pitch is measured from the audio itself (pYIN median f0), not trusted from file names: sample libraries use
different octave conventions. For every file: did the model find exactly that note, an octave error, extra notes,
and is the chosen string/fret physically consistent (open string + fret == pitch)?
Run from the repo root: python -m scripts.eval.gv.test_a_single_notes data/raw/gv/libraries <out_csv>
"""
import csv
import re
import sys
from pathlib import Path

import librosa
import numpy as np

from gtab.core.instrument import STANDARD_6
from gtab.data.features import cqt_from_file
from gtab.pipeline import analyze, load_models
from gtab.utils import get_device

MODEL_RANGE = (40, 88)  # E2..E6, pipeline cannot report notes outside it


def reference_pitch(path: Path) -> float | None:
    y, sr = librosa.load(path, sr=22050, mono=True, duration=2.0)
    f0, voiced, _ = librosa.pyin(y, fmin=70, fmax=1400, sr=sr)
    f0 = f0[voiced & ~np.isnan(f0)]
    return float(np.median(librosa.hz_to_midi(f0))) if len(f0) >= 5 else None


def name_label(path: Path) -> str:
    m = re.search(r"([A-G]#?\d)", path.name) or re.search(r"Nylon_\d+_\d+_(\d+)_", path.name)
    return m.group(1) if m else ""


def summarize(an: dict, ref: int) -> dict:
    notes, string, fret = an["notes"], an["string"], an["fret"]
    pitches = notes[:, 2].astype(int) if len(notes) else np.array([], int)
    main = int(np.argmax(notes[:, 1] - notes[:, 0])) if len(notes) else -1  # longest note = the plucked one
    tuning = STANDARD_6.tuning
    consistent = all(s < 0 or tuning[s] + f == p for s, f, p in zip(string, fret, pitches))
    return {"n_notes": len(notes), "found_ref": int(ref in pitches), "main_pitch": int(pitches[main]) if main >= 0 else -1,
            "octave_err": int(main >= 0 and abs(pitches[main] - ref) == 12),
            "extra": int(sum(p != ref for p in pitches)),
            "string": int(string[main]) if main >= 0 else -1, "fret": int(fret[main]) if main >= 0 else -1,
            "consistent": int(consistent)}


def main(root: Path, out: Path) -> None:
    files = sorted(p for p in root.rglob("*.wav") if "Noises" not in str(p) and "__MACOSX" not in str(p))
    m = load_models(get_device())
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = None
        for p in files:
            ref = reference_pitch(p)
            row = {"library": p.relative_to(root).parts[0], "file": p.name, "label": name_label(p),
                   "ref_midi": "" if ref is None else round(ref, 2)}
            if ref is not None:
                row.update(summarize(analyze(cqt_from_file(str(p)), m, "ses"), int(round(ref))))
                row["in_range"] = int(MODEL_RANGE[0] <= round(ref) <= MODEL_RANGE[1])
            w = w or csv.DictWriter(f, fieldnames=["library", "file", "label", "ref_midi", "n_notes", "found_ref",
                                                   "main_pitch", "octave_err", "extra", "string", "fret",
                                                   "consistent", "in_range"])
            if f.tell() == 0:
                w.writeheader()
            w.writerow(row)
            print(row, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
