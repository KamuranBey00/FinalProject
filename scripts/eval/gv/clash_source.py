"""Where do same-string clashes in the TAB come from? For note pairs that end up on the same string at the same
beat position, look at their real onset gap: < 30 ms = heard together (string-assignment problem),
larger = distinct notes squeezed into one rhythmic slot (quantization problem).
Run from the repo root: python -m scripts.eval.gv.clash_source [extra_audio ...]
"""
import collections
import sys
from pathlib import Path

import numpy as np

from gtab.data.features import cqt_from_file
from gtab.pipeline import load_models, transcribe
from gtab.utils import get_device
from scripts.eval.gv.paths import GAPS_TEST

BINS = ((0.03, "<30 ms (heard together)"), (0.1, "30-100 ms"), (0.25, "100-250 ms"), (np.inf, ">250 ms"))


def pairs(r: dict) -> list[tuple[float, bool]]:
    on, q, s, p = r["notes"][:, 0], np.asarray(r["q"]), r["string"], r["notes"][:, 2]
    out = []
    for i in range(len(on)):
        for j in range(i + 1, min(i + 12, len(on))):
            if s[i] >= 0 and s[i] == s[j] and abs(q[i] - q[j]) < 1e-9:
                out.append((abs(on[j] - on[i]), p[i] != p[j]))
    return out


def main(extra: list[Path]) -> None:
    m = load_models(get_device())
    sources = [(p.name, cqt_from_file(str(p))) for p in extra]
    sources += [(p.stem, np.load(p)["cqt"]) for p in sorted(GAPS_TEST.glob("*.npz"))]
    hist, string_err, n_notes, total = collections.Counter(), 0, 0, 0
    for name, cqt in sources:
        r = transcribe(cqt, m, 90.0, (4, 4), "ses")
        n_notes += len(r["notes"])
        for dt, diff_pitch in pairs(r):
            total += 1
            hist[next(label for lim, label in BINS if dt < lim)] += 1
            string_err += dt < 0.03 and diff_pitch
    print(f"sources {len(sources)}, notes {n_notes}, same-string pairs at the same beat position: {total}")
    for _, label in BINS:
        print(f"  real onset gap {label:24s}: {hist[label]}")
    print(f"heard together with different pitch (string-assignment error): {string_err}")


if __name__ == "__main__":
    main([Path(a) for a in sys.argv[1:]])
