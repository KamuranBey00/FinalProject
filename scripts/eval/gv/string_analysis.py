"""Analysis only (no product change): why is the "ses" (heard position) string choice wrong, and what would a
one-note-per-string chord constraint change? GAPS test pieces with score-derived string labels, real recordings.
Per-note candidate scores are computed exactly like gtab.decoding.transitions.build_lattice (mean string prob).
Run from the repo root: python -m scripts.eval.gv.string_analysis
"""
import itertools

import mir_eval
import numpy as np

from gtab.core.instrument import STANDARD_6
from gtab.data.torch_dataset import _normalize
from gtab.decoding.viterbi import energy_rise, segment
from gtab.models.inference import predict_heads
from gtab.pipeline import load_models
from gtab.utils import get_device
from gtab.config import FRAME_RATE
from scripts.eval.gv.paths import GAPS_TAB as TAB, GAPS_TEST as TEST


def notes_and_scores(cqt: np.ndarray, m: dict) -> tuple[np.ndarray, list, list]:
    _, thr, ot, rules = m["sel"]
    probs, ons, offs, pons = predict_heads(m["model"], _normalize(cqt), m["device"], with_pitch_onset=True)
    segs = segment(probs, ons, thr, ot, STANDARD_6, rise=energy_rise(cqt, STANDARD_6), offsets=offs,
                   pitch_onsets=pons if rules.get("onset_source") == "pitch" else None, **rules)
    cands, scores = [], []
    for a, b, p in segs:
        c = STANDARD_6.pitch_to_positions(p)
        cands.append(c)
        scores.append(np.array([probs[a:max(a + 1, b), s, f + 1].mean() for s, f in c]) + 1e-6)
    notes = np.array([(a / FRAME_RATE, b / FRAME_RATE, p) for a, b, p in segs], float)
    return notes, cands, scores


def chord_groups(notes: np.ndarray) -> list[list[int]]:
    # notes sounding together: onset within 30 ms of the group's first note (same rule as the tab clash analysis)
    groups, start = [], None
    for k in np.argsort(notes[:, 0], kind="stable"):
        if start is None or notes[k, 0] - start > 0.03:
            groups.append([]); start = notes[k, 0]
        groups[-1].append(int(k))
    return groups


def joint(group: list[int], cands: list, scores: list) -> dict[int, int]:
    # best total log-score with distinct strings; groups are <= 6 notes x few candidates, brute force is exact
    opts = [range(len(cands[k])) for k in group]
    best, pick = -np.inf, None
    for combo in itertools.product(*opts):
        strings = [cands[k][c][0] for k, c in zip(group, combo)]
        if len(set(strings)) < len(strings):
            continue
        s = sum(np.log(scores[k][c]) for k, c in zip(group, combo))
        if s > best:
            best, pick = s, combo
    if pick is None:  # more notes than distinct strings possible: keep greedy
        pick = [int(np.argmax(scores[k])) for k in group]
    return {k: c for k, c in zip(group, pick)}


def main() -> None:
    m = load_models(get_device())
    stats = {"n": 0, "greedy_ok": 0, "joint_ok": 0, "adjacent": 0, "second_choice": 0, "greedy_clash": 0,
             "joint_clash": 0, "groups": 0}
    ids = sorted({p.stem for p in TEST.glob("*.npz")} & {p.stem for p in TAB.glob("*.npz")})
    for tid in ids:
        with np.load(TAB / f"{tid}.npz") as d:
            ref, ref_s = d["notes"].astype(float), d["string"].astype(int)
        notes, cands, scores = notes_and_scores(np.load(TEST / f"{tid}.npz")["cqt"], m)
        greedy = [int(np.argmax(s)) for s in scores]
        jsel = dict(greedy=None)
        jpick = {}
        for g in chord_groups(notes):
            jpick.update(joint(g, cands, scores))
            stats["groups"] += 1
            gs = [cands[k][greedy[k]][0] for k in g]
            js = [cands[k][jpick[k]][0] for k in g]
            stats["greedy_clash"] += len(gs) != len(set(gs))
            stats["joint_clash"] += len(js) != len(set(js))
        iv = lambda n: (np.column_stack([n[:, 0], np.maximum(n[:, 1], n[:, 0] + .01)]), mir_eval.util.midi_to_hz(n[:, 2]))
        for i, j in mir_eval.transcription.match_notes(*iv(ref), *iv(notes), onset_tolerance=0.05, offset_ratio=None):
            if ref_s[i] < 0:
                continue
            gt, g_s, j_s = ref_s[i], cands[j][greedy[j]][0], cands[j][jpick[j]][0]
            order = [cands[j][c][0] for c in np.argsort(-scores[j])]
            stats["n"] += 1
            stats["greedy_ok"] += g_s == gt
            stats["joint_ok"] += j_s == gt
            stats["adjacent"] += g_s != gt and abs(g_s - gt) == 1
            stats["second_choice"] += g_s != gt and len(order) > 1 and order[1] == gt
    n, wrong = stats["n"], stats["n"] - stats["greedy_ok"]
    print(f"matched labelled notes {n}")
    print(f"string accuracy: greedy (today) {stats['greedy_ok'] / n:.3f} | one-note-per-string {stats['joint_ok'] / n:.3f}")
    print(f"of the wrong ones: adjacent string {stats['adjacent'] / wrong:.1%}, true string = model's 2nd choice "
          f"{stats['second_choice'] / wrong:.1%}")
    print(f"chord groups {stats['groups']}: with a string clash greedy {stats['greedy_clash']}, joint {stats['joint_clash']}")


if __name__ == "__main__":
    main()
