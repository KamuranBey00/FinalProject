"""Test C: recordings without ground truth. Checks the produced TAB against independent evidence:
onset agreement with librosa's onset detector, pitch-class profile vs the audio's chroma, and TAB rules in the
written MusicXML (one note per string per chord, open string + fret == pitch, fret range).
Run from the repo root: python -m scripts.eval.gv.test_c_sanity <audio> <xml_ses> <xml_edisyon>
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import librosa
import numpy as np

from gtab.core.instrument import STANDARD_6
from gtab.data.features import cqt_from_file
from gtab.pipeline import analyze, load_models
from gtab.utils import get_device

STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def audio_evidence(path: Path, notes: np.ndarray) -> dict[str, float]:
    y, sr = librosa.load(path, sr=22050, mono=True)
    ref_on = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=False)
    est_on = np.unique(np.round(notes[:, 0], 2))
    near = lambda a, b: np.mean([np.abs(b - t).min() < 0.05 for t in a]) if len(a) and len(b) else 0.0
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    est = np.bincount(notes[:, 2].astype(int) % 12, weights=notes[:, 1] - notes[:, 0], minlength=12)
    return {"onsets_confirmed_by_audio": near(est_on, ref_on), "audio_onsets_found": near(ref_on, est_on),
            "chroma_corr": float(np.corrcoef(chroma, est)[0, 1]), "pitch_range": f"{int(notes[:, 2].min())}-{int(notes[:, 2].max())}"}


def tab_rules(xml: Path) -> dict[str, float]:
    tuning, S = STANDARD_6.tuning, len(STANDARD_6.tuning)
    part = ET.parse(xml).getroot().findall("part")[1]  # P2 = TAB staff
    chords, bad_pitch, frets = [], 0, []
    for n in part.iter("note"):
        p, s, f = n.find("pitch"), n.find(".//string"), n.find(".//fret")
        if p is None or s is None:
            continue
        midi = 12 * (int(p.findtext("octave")) + 1) + STEP[p.findtext("step")] + int(p.findtext("alter") or 0)
        string = S - int(s.text)  # MusicXML string 1 = high e; ours 0 = low E
        bad_pitch += tuning[string] + int(f.text) != midi
        frets.append(int(f.text))
        if n.find("chord") is None:
            chords.append([])
        chords[-1].append(string)
    dup = sum(len(c) != len(set(c)) for c in chords)
    return {"tab_notes": len(frets), "pitch_mismatch": bad_pitch, "chords_with_string_clash": dup,
            "max_fret": max(frets), "fret_gt_12_pct": round(100 * np.mean(np.array(frets) > 12), 1)}


if __name__ == "__main__":
    audio, xml_ses, xml_ed = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    an = analyze(cqt_from_file(str(audio)), load_models(get_device()), "ses")
    print(audio.name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in audio_evidence(audio, an["notes"]).items()})
    for mode, x in (("ses", xml_ses), ("edisyon", xml_ed)):
        print(f"  {mode:8s}", tab_rules(x))
