"""README22 Step 5B analysis (no product change): can a rule tell sympathetic open-string ringing from real
open-string notes? Same notes, labels and matching as extra_notes / ghost_audit (note F1 definition).

Population: detected notes whose pitch is an open-string pitch (E2 A2 D3 G3 B3 E4).
  ghost  matches no label note      real  matches a label note
Trigger (runtime-available, DETECTED notes only): another detected note sounding at the onset whose pitch puts one
of its first partials on a partial of the open string: (trigger - open) in {0, 7, 12, 19, 24} semitones
(unison on another string, 3rd partial vs 2nd, octave, 12th, double octave).
Evidence: pitch-onset peak, energy rise (dB, as the segmenter), mean pitch probability, duration, fallback origin.
Candidate rules "drop an open-string note that has a trigger AND weak evidence" are counted as ghost / real removed.
Run from the repo root: python -m scripts.eval.gv.open_strings [--split gaps_val] [--limit N]
"""
import argparse

import mir_eval
import numpy as np

from gtab.config import FRAME_RATE
from gtab.core.instrument import STANDARD_6
from gtab.decoding.viterbi import pitch_matrix, segment
from gtab.evaluation.pitch_eval import load_split, roll_to_notes, to_mir
from gtab.pipeline import load_models
from gtab.utils import get_device

OPEN = set(STANDARD_6.tuning)
TRIGGER = {0, 7, 12, 19, 24}
RULES = [(peak, rise) for peak in (0.3, 0.5, 0.7) for rise in (3.0, 6.0, 10.0)]


def rows_of(item: tuple, sel: tuple) -> list[dict]:
    probs, ons, frame, onset, rise, offs, pons = item
    _, thr, ot, dec = sel
    n = min(len(probs), len(frame))
    kw = dict(rise=rise[:n], offsets=offs[:n], pitch_onsets=pons[:n])
    est = segment(probs[:n], ons[:n], thr, ot, STANDARD_6, **kw, **dec)
    onset_started = set(segment(probs[:n], ons[:n], thr, ot, STANDARD_6, **kw, **dict(dec, fallback=0)))
    pm, lo = pitch_matrix(probs[:n], STANDARD_6, dec.get("combine", "max"))
    ref = roll_to_notes(frame[:n] > 0, onset[:n] > 0, lo)
    pairs = mir_eval.transcription.match_notes(*to_mir(ref), *to_mir(est), onset_tolerance=0.05,
                                               offset_ratio=None) if len(ref) and len(est) else []
    matched = {j for _, j in pairs}
    out = []
    for j, (a, b, p) in enumerate(est):
        if p not in OPEN:
            continue
        k = p - lo
        trig = any(da <= a < db and dk != j and dp - p in TRIGGER and not (dp == p and da == a)
                   for dk, (da, db, dp) in enumerate(est))
        out.append(dict(kind="real" if j in matched else "ghost", trigger=trig, fallback=(a, b, p) not in onset_started,
                        peak=float(pons[a, k]), rise=float(rise[max(0, a - 1):a + 3, k].max()),
                        prob=float(pm[a:b, k].mean()), dur=(b - a) / FRAME_RATE))
    return out


def count(rows: list[dict]) -> tuple[int, int]:
    return sum(r["kind"] == "ghost" for r in rows), sum(r["kind"] == "real" for r in rows)


def report(split: str, rows: list[dict]) -> None:
    print(f"{split}: open-string-pitch notes ghost / real {count(rows)}")
    trig = [r for r in rows if r["trigger"]]
    print(f"  with a detected trigger sounding: ghost / real {count(trig)}; without: "
          f"{count([r for r in rows if not r['trigger']])}")
    print(f"  fallback origin: ghost / real {count([r for r in rows if r['fallback']])}")
    for kind in ("ghost", "real"):
        x = [r for r in trig if r["kind"] == kind]
        if x:
            q = lambda key: np.percentile([r[key] for r in x], [10, 25, 50, 75]).round(3).tolist()
            print(f"  triggered {kind:5s} n {len(x):5d}  onset peak p10/25/50/75 {q('peak')}  rise dB {q('rise')}  "
                  f"pitch prob {q('prob')}  dur s {q('dur')}")
    print("  rule 'trigger AND onset peak < x AND rise < y' removes ghost / real:")
    for peak, rise in RULES:
        print(f"    peak < {peak:.1f}, rise < {rise:4.1f} dB: "
              f"{count([r for r in trig if r['peak'] < peak and r['rise'] < rise])}")


def main(split: str, limit: int | None) -> None:
    m = load_models(get_device())
    data = load_split(split, m["model"], None, "crnn", m["device"], limit)
    report(split, [r for item in data for r in rows_of(item, m["sel"])])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="gaps_val")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    main(a.split, a.limit)
