"""README22 Step 2B check: "ses" (heard position) string accuracy with and without the chord constraint
(gtab.pipeline.heard_positions -> resolve_chord_strings), on GAPS pieces with score-derived string labels.
Notes are matched to the labels with mir_eval (onset +-50 ms, offsets ignored) as in test_b_resynth.score.
Run from the repo root: python -m scripts.eval.gv.chord_effect
"""
import os

import mir_eval
import numpy as np

import gtab.pipeline as pipeline
from gtab.data.gaps import gaps_files
from gtab.pipeline import analyze, load_models
from gtab.utils import get_device
from scripts.eval.gv.paths import GAPS_TAB, GAPS_TEST
from scripts.eval.gv.test_b_resynth import mir

CONSTRAINED = pipeline.resolve_chord_strings


def split_ids(split: str) -> list[str]:
    files = gaps_files("gaps_train")[1] if split == "gaps_val" else sorted(map(str, GAPS_TEST.glob("*.npz")))
    return [os.path.basename(f)[:-4] for f in files if (GAPS_TAB / os.path.basename(f)).exists()]


def accuracy(ids: list[str], split: str, m: dict) -> tuple[float, int]:
    # micro average over matched labelled notes: pieces differ a lot in length
    hits, n = 0, 0
    folder = GAPS_TEST if split == "gaps_test" else GAPS_TEST.parent / "gaps_train"
    for tid in ids:
        with np.load(GAPS_TAB / f"{tid}.npz") as d:
            ref, ref_s = d["notes"].astype(float), d["string"].astype(int)
        with np.load(folder / f"{tid}.npz") as d:
            an = analyze(d["cqt"], m, "ses")
        if not len(an["notes"]):
            continue
        pairs = mir_eval.transcription.match_notes(*mir(ref), *mir(an["notes"]), onset_tolerance=0.05,
                                                   offset_ratio=None)
        known = [(i, j) for i, j in pairs if ref_s[i] >= 0]
        hits += sum(an["string"][j] == ref_s[i] for i, j in known); n += len(known)
    return hits / max(n, 1), n


def main() -> None:
    m = load_models(get_device())
    for split in ("gaps_val", "gaps_test"):
        ids = split_ids(split)
        pipeline.resolve_chord_strings = lambda lattice, picks, groups: picks
        before, n = accuracy(ids, split, m)
        pipeline.resolve_chord_strings = CONSTRAINED
        after, _ = accuracy(ids, split, m)
        print(f"{split}: pieces {len(ids)}, notes {n}, ses string accuracy greedy {before:.4f} -> chord constraint "
              f"{after:.4f} ({after - before:+.4f})")


if __name__ == "__main__":
    main()
