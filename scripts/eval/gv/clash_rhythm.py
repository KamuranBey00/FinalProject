"""Analysis only (no product change): same-string clashes in the written TAB, classified against the GAPS score.

A clash = two notes written at the same rhythmic position (same tick after build_measures rounding) on the same
string. User input is simulated as in eval_score B (true median tempo x (1 +- 10 %), most common time signature).
Each clash is classified by what the score says about the two notes:
  chord     both match score notes at the same score position -> string assignment is wrong
  squeezed  both match score notes at different positions     -> rhythm quantization merged them
  extra     at least one matches no score note (same pitch: re-detection; else ghost / wrong pitch)
gaps_val (performer-disjoint part of gaps_train) is the selection set, gaps_test is reported only.
Run from the repo root: python -m scripts.eval.gv.clash_rhythm [--limit N]
"""
import argparse
import glob
import os
from collections import Counter

import numpy as np

from gtab.data.gaps import GAPS_DIR, gaps_files
from gtab.data.gaps_score import match_notes, time_signatures
from gtab.evaluation.rhythm_metrics import tempo_of
from gtab.models.rhythm_net import TICKS
from gtab.paths import CACHE_DIR
from gtab.pipeline import analyze, load_models, rhythm
from gtab.utils import get_device

RHY_DIR = os.path.join(CACHE_DIR, "rhythm")
ONSET_TOL, USER_ERR, CHORD_S = 0.05, 0.10, 0.03


def split_files(split: str) -> list[str]:
    files = (gaps_files("gaps_train")[1] if split == "gaps_val"
             else sorted(glob.glob(os.path.join(CACHE_DIR, split, "*.npz"))))
    return [f for f in files if os.path.exists(os.path.join(RHY_DIR, os.path.basename(f)))]


def clashes(q: np.ndarray, string: np.ndarray) -> list[tuple[int, int]]:
    tick = np.round(q * TICKS)
    out = []
    for t in np.unique(tick[~np.isnan(tick)]):
        idx = np.flatnonzero((tick == t) & (string >= 0))
        for k, i in enumerate(idx):
            out += [(i, j) for j in idx[k + 1:] if string[i] == string[j]]
    return out


def classify(i: int, j: int, det: np.ndarray, mi: np.ndarray, score_pos: np.ndarray) -> str:
    gap = "<30ms" if abs(det[i, 0] - det[j, 0]) < CHORD_S else ">=30ms"
    if mi[i] < 0 or mi[j] < 0:
        kind = "extra_same_pitch" if det[i, 2] == det[j, 2] else "extra_other_pitch"
    else:
        kind = "chord" if abs(score_pos[mi[i]] - score_pos[mi[j]]) < 1e-6 else "squeezed"
    return f"{kind} {gap}"


def run_piece(f: str, k: int, models: dict, mode: str) -> tuple[Counter, int]:
    tid = os.path.basename(f)[:-4]
    with np.load(f) as d:
        cqt = d["cqt"]
    with np.load(os.path.join(RHY_DIR, tid + ".npz")) as g:
        gb, gn, s = g["beats"], g["notes"], g["note_beat"]
    sig = Counter(time_signatures(os.path.join(GAPS_DIR, "musicxml", f"{tid}.xml"))).most_common(1)[0][0]
    an = analyze(cqt, models, mode)
    det = an["notes"]
    r = rhythm(an, models, sig, tempo_of(gb) * (1 + USER_ERR * (-1) ** k))
    to_det, _ = match_notes([(on, int(p)) for on, _, p in gn], [dict(pitch=int(p)) for _, _, p in det], det[:, 0],
                            ONSET_TOL)
    mi = np.full(len(det), -1)  # match_notes maps score -> detected; invert to detected -> score
    mi[to_det[to_det >= 0]] = np.flatnonzero(to_det >= 0)
    return Counter(classify(i, j, det, mi, s) for i, j in clashes(np.asarray(r["q"]), an["string"])), len(det)


def main(limit: int | None) -> None:
    models = load_models(get_device())
    for split in ("gaps_val", "gaps_test"):
        files = split_files(split)[:limit]
        for mode in ("ses", "edisyon"):
            total, notes, hit = Counter(), 0, 0
            for k, f in enumerate(files):
                c, n = run_piece(f, k, models, mode)
                total += c; notes += n; hit += bool(c)
            n_cl = sum(total.values())
            print(f"{split} {mode}: pieces {len(files)} (with clash {hit}), notes {notes}, clashes {n_cl}")
            for key, v in sorted(total.items(), key=lambda x: -x[1]):
                print(f"    {key:28s} {v:6d}  {v / max(n_cl, 1):6.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    main(ap.parse_args().limit)
