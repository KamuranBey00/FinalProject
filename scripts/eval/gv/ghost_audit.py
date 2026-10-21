"""README22 Step 5 analysis (no product change): what are the ghosts that are not overtones, and how much of the
"ghost" count is label error? Same notes, labels and matching as extra_notes (note F1 definition).

For every unmatched detected note (ghost) not explained by an overtone of a sounding label note:
  timing <150 ms   a label note of the same pitch starts 50-150 ms away (matching tolerance is 50 ms)
  near 150-500 ms  same, 150-500 ms away (a larger timing / alignment slip)
  open string      its pitch is an open-string pitch and that string's label is silent (sympathetic ringing)
  label gap        no label onset of any pitch within 1 s (alignment hole: the model may be right)
  interval mod 12  otherwise, interval to the lowest sounding label note reduced to one octave
timing / near ghosts are split by their nearest same-pitch label note: '/fragment' = that label note is matched by
another detection (the ghost is an extra piece of a found note), '/displaced' = it is matched by nothing (the note was
found but off in time, or the label is: counted once as ghost and once as missed).
Timing: signed onset error (detected - label, ms) of matched notes and of the "timing <150 ms" ghosts, to see a bias.
Writes a random sample of 30 ghosts (recording, time, pitch, category) for listening / visual checks.
Run from the repo root: python -m scripts.eval.gv.ghost_audit [--split gaps_val] [--limit N]
"""
import argparse
import csv
import os
import random
from collections import Counter

import mir_eval
import numpy as np

from gtab.config import FRAME_RATE
from gtab.core.instrument import STANDARD_6
from gtab.data.gaps import gaps_files
from gtab.decoding.viterbi import segment
from gtab.evaluation.pitch_eval import load_split, roll_to_notes, to_mir
from gtab.paths import CACHE_DIR
from gtab.pipeline import load_models
from gtab.utils import get_device
from scripts.eval.gv.extra_notes import OVERTONES, host_interval
from scripts.eval.gv.paths import RESULTS

OPEN = set(STANDARD_6.tuning)
NEAR_FR, FAR_FR, GAP_FR = round(0.15 * FRAME_RATE), round(0.5 * FRAME_RATE), round(1.0 * FRAME_RATE)


def ids_of(split: str, limit: int | None) -> list[str]:
    files = gaps_files("gaps_train")[1] if split == "gaps_val" else sorted(
        os.path.join(CACHE_DIR, split, f) for f in os.listdir(os.path.join(CACHE_DIR, split)) if f.endswith(".npz"))
    return [os.path.basename(f)[:-4] for f in files][:limit]


def category(a: int, p: int, ref: list, matched_ref: set) -> str:
    same = sorted((abs(r[0] - a), i) for i, r in enumerate(ref) if r[2] == p)
    kind = lambda: "/fragment" if same[0][1] in matched_ref else "/displaced"
    if same and same[0][0] <= NEAR_FR:
        return "timing <150 ms" + kind()
    iv, _ = host_interval(a, p, ref)
    if iv in OVERTONES:
        return "overtone"
    if same and same[0][0] <= FAR_FR:
        return "near 150-500 ms" + kind()
    if p in OPEN and not any(ra <= a < rb and rp == p for ra, rb, rp in ref):
        return "open string"
    if not any(abs(ra - a) <= GAP_FR for ra, _, _ in ref):
        return "label gap"
    return "no host" if iv is None else f"interval mod 12 = {iv % 12}"


def piece(item: tuple, sel: tuple) -> tuple[list, list]:
    probs, ons, frame, onset, rise, offs, pons = item
    _, thr, ot, dec = sel
    n = min(len(probs), len(frame))
    est = segment(probs[:n], ons[:n], thr, ot, STANDARD_6, rise=rise[:n], offsets=offs[:n],
                  pitch_onsets=pons[:n], **dec)
    ref = roll_to_notes(frame[:n] > 0, onset[:n] > 0, STANDARD_6.pitch_range()[0])
    pairs = mir_eval.transcription.match_notes(*to_mir(ref), *to_mir(est), onset_tolerance=0.05,
                                               offset_ratio=None) if len(ref) and len(est) else []
    matched, matched_ref = {j for _, j in pairs}, {i for i, _ in pairs}
    errs = [(est[j][0] - ref[i][0]) * 1000 / FRAME_RATE for i, j in pairs]
    ghosts = []
    for j, (a, b, p) in enumerate(est):
        if j not in matched:
            cat = category(a, p, ref, matched_ref)
            if cat.startswith("timing <150 ms"):
                ra = min((r for r in ref if r[2] == p), key=lambda r: abs(r[0] - a))[0]
                errs_g = (a - ra) * 1000 / FRAME_RATE
            else:
                errs_g = None
            ghosts.append(dict(t=a / FRAME_RATE, pitch=p, cat=cat, err_ms=errs_g))
    return ghosts, errs


def main(split: str, limit: int | None) -> None:
    m = load_models(get_device())
    ids = ids_of(split, limit)
    data = load_split(split, m["model"], None, "crnn", m["device"], limit)
    ghosts, errs = [], []
    for tid, item in zip(ids, data):
        g, e = piece(item, m["sel"])
        ghosts += [dict(x, id=tid) for x in g]
        errs += e
    c = Counter(x["cat"] for x in ghosts)
    print(f"{split}: ghosts {len(ghosts)}")
    for k, v in c.most_common():
        print(f"  {k:24s} {v:6d}  {v / len(ghosts):6.1%}")
    q = lambda x: np.percentile(x, [10, 25, 50, 75, 90]).round(1).tolist()
    tg = [x["err_ms"] for x in ghosts if x["err_ms"] is not None]
    print(f"  onset error ms (detected - label) p10/25/50/75/90: matched {q(errs)}  timing ghosts {q(tg)}")
    print(f"  timing ghosts late / early: {sum(e > 0 for e in tg)} / {sum(e < 0 for e in tg)}")
    random.seed(0)
    sample = random.sample([x for x in ghosts if x["cat"] != "overtone"], 30)
    out = os.path.join(RESULTS, f"ghost_sample_{split}.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "t", "pitch", "cat", "err_ms"])
        w.writeheader()
        w.writerows(sorted(sample, key=lambda x: (x["id"], x["t"])))
    print(f"  sample of 30 non-overtone ghosts -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="gaps_val")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    main(a.split, a.limit)
