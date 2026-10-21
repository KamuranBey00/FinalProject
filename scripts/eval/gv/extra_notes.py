"""README22 Step 3 analysis (no product change): where do extra notes (detected, not in the labels) come from, and can
the model's own evidence tell an overtone ghost from a real note at the same interval?

Notes and labels exactly as the note F1 (gtab.evaluation.pitch_eval.track_scores: active selection, roll_to_notes
labels, mir_eval onset +-50 ms). For every detected note we look for a "host": a label note of another pitch that is
sounding at the detected onset. Interval = detected pitch - host pitch, taken from the lowest such host.
  ghost  detected note matches no label note
  real   detected note matches a label note (a real note that happens to sit at that interval above a sounding one)
Evidence per note (onset frame a, pitch p): pitch-onset peak, mean pitch probability over the note, energy rise (dB,
same as the segmenter), duration, and whether it starts with its host (< 50 ms) or later (in the host's decay).
Run from the repo root: python -m scripts.eval.gv.extra_notes [--split gaps_val] [--limit N]
"""
import argparse
from collections import Counter, defaultdict

import mir_eval
import numpy as np

from gtab.config import FRAME_RATE
from gtab.core.instrument import STANDARD_6
from gtab.decoding.viterbi import pitch_matrix, segment
from gtab.evaluation.pitch_eval import load_split, roll_to_notes, to_mir
from gtab.pipeline import load_models
from gtab.utils import get_device

OVERTONES = {12: "+12 (2nd partial)", 19: "+19 (3rd)", 24: "+24 (4th)", 28: "+28 (5th)", 31: "+31 (6th)"}
WITH_HOST_FR = round(0.05 * FRAME_RATE)


TIMING_FR = round(0.15 * FRAME_RATE)


def label(interval: int | None) -> str:
    if interval is None:
        return "no host sounding"
    if interval in OVERTONES:
        return OVERTONES[interval]
    if interval == -12:
        return "-12 (octave below)"
    if abs(interval) <= 2:
        return "+-1..2 (neighbour pitch)"
    return "other interval"


def host_interval(a: int, p: int, ref: list) -> tuple[int | None, int]:
    # label notes of another pitch sounding at frame a; an overtone relation to ANY of them wins (smallest partial),
    # else the lowest one -> (interval, its onset frame)
    hosts = sorted((rp, ra) for ra, rb, rp in ref if ra <= a < rb and rp != p)
    if not hosts:
        return None, a
    over = sorted((p - hp, ha) for hp, ha in hosts if p - hp in OVERTONES)
    return over[0] if over else (p - hosts[0][0], hosts[0][1])


def near_same_pitch(a: int, p: int, ref: list) -> bool:
    # a label note of the same pitch starts within 150 ms: a timing miss (> 50 ms), not an extra note
    return any(rp == p and abs(ra - a) <= TIMING_FR for ra, _, rp in ref)


def piece_rows(item: tuple, sel: tuple) -> list[dict]:
    probs, ons, frame, onset, rise, offs, pons = item
    _, thr, ot, dec = sel
    n = min(len(probs), len(frame))
    est = segment(probs[:n], ons[:n], thr, ot, STANDARD_6, rise=rise[:n], offsets=offs[:n],
                  pitch_onsets=pons[:n], **dec)
    # notes the 'fallback' rule added (no onset): the onset-started ones are identical with fallback off
    onset_started = set(segment(probs[:n], ons[:n], thr, ot, STANDARD_6, rise=rise[:n], offsets=offs[:n],
                                pitch_onsets=pons[:n], **dict(dec, fallback=0)))
    pm, lo = pitch_matrix(probs[:n], STANDARD_6, dec.get("combine", "max"))
    ref = roll_to_notes(frame[:n] > 0, onset[:n] > 0, lo)
    matched = set()
    if len(ref) and len(est):
        pairs = mir_eval.transcription.match_notes(*to_mir(ref), *to_mir(est), onset_tolerance=0.05, offset_ratio=None)
        matched = {j for _, j in pairs}
    rows = []
    for j, (a, b, p) in enumerate(est):
        iv, ha = host_interval(a, p, ref)
        k, kind = p - lo, "real" if j in matched else "ghost"
        cat = "timing (same pitch <150 ms)" if kind == "ghost" and near_same_pitch(a, p, ref) else label(iv)
        # runtime-available relation: is it an overtone of a DETECTED onset-started note sounding at frame a?
        det_host = any(da <= a < db and p - dp in OVERTONES for da, db, dp in onset_started)
        rows.append(dict(kind=kind, cat=cat, fallback=(a, b, p) not in onset_started, det_overtone=det_host,
                         with_host=iv is not None and abs(a - ha) <= WITH_HOST_FR,
                         peak=float(pons[a, k]), prob=float(pm[a:b, k].mean()),
                         rise=float(rise[max(0, a - 1):a + 3, k].max()), dur=(b - a) / FRAME_RATE))
    return rows


def report(rows: list[dict]) -> None:
    ghosts = [r for r in rows if r["kind"] == "ghost"]
    print(f"detected {len(rows)}, ghosts (unmatched) {len(ghosts)} ({len(ghosts) / max(len(rows), 1):.1%})")
    cats = Counter(r["cat"] for r in ghosts)
    real = Counter(r["cat"] for r in rows if r["kind"] == "real")
    print(f"  {'interval to sounding label note':28s} {'ghost':>6s} {'real':>6s}  ghost: with host / in decay")
    for c, v in cats.most_common():
        w = sum(r["with_host"] for r in ghosts if r["cat"] == c)
        print(f"  {c:28s} {v:6d} {real[c]:6d}  {w:5d} / {v - w}")
    by = defaultdict(list)
    for r in rows:
        if r["cat"] in OVERTONES.values():
            by[r["kind"]].append(r)
    for kind in ("ghost", "real"):
        x = by[kind]
        if x:
            q = lambda key: np.percentile([r[key] for r in x], [25, 50, 75]).round(3).tolist()
            print(f"  overtone-interval {kind:5s} n {len(x):5d}  onset peak {q('peak')}  pitch prob {q('prob')}  "
                  f"rise dB {q('rise')}  dur s {q('dur')}")


def origin_report(rows: list[dict]) -> None:
    fb = [r for r in rows if r["fallback"]]
    c = lambda x: (sum(r["kind"] == "ghost" for r in x), sum(r["kind"] == "real" for r in x))
    print(f"  origin: onset-started ghost/real {c([r for r in rows if not r['fallback']])}, "
          f"fallback (no onset) ghost/real {c(fb)}")
    print(f"  rule 'drop fallback notes on an overtone of a detected sounding note' removes ghost/real "
          f"{c([r for r in fb if r['det_overtone']])}")
    for kind in ("ghost", "real"):
        x = [r for r in fb if r["kind"] == kind]
        if x:
            q = lambda key: np.percentile([r[key] for r in x], [10, 25, 50, 75]).round(3).tolist()
            print(f"  fallback {kind:5s} n {len(x):5d}  pitch prob p10/25/50/75 {q('prob')}  dur s {q('dur')}  "
                  f"rise dB {q('rise')}")
    print(f"  same rule for onset-started notes would remove ghost/real "
          f"{c([r for r in rows if not r['fallback'] and r['det_overtone']])}")


def main(split: str, limit: int | None) -> None:
    m = load_models(get_device())
    data = load_split(split, m["model"], None, "crnn", m["device"], limit)
    rows = [r for item in data for r in piece_rows(item, m["sel"])]
    print(f"{split} ({len(data)} recordings), selection {m['sel'][3]}")
    report(rows)
    origin_report(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="gaps_val")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    main(a.split, a.limit)
