"""README22 Step 4 (measurement): string accuracy of the position modes on the same detected notes.
  ses w=x    heard position (gtab.pipeline.heard_positions) with the learned hand-position prior at weight x
             (product: x = 0); chord constraint included
  edisyon    edition suggestion (EditionNet), same notes
  hibrit t=x ses where the heard-position model is sure (log-prob gap between its 1st and 2nd string >= x),
             edisyon where it is not; a chord left with a string clash takes the edisyon strings as a whole
Labels: GAPS score-derived strings (data/cache/gaps_tab_rec) = the WRITTEN fingering (what edisyon imitates);
GuitarSet val (--dataset guitarset, its own 'val' calibration) = the string actually PLAYED (hexaphonic pickup), the
only ground truth for the heard position. Notes matched with mir_eval (onset +-50 ms).
Model outputs and notes are computed once per recording, so every variant is scored on identical notes.
gaps_val = selection, gaps_test = report.
Run from the repo root: python -m scripts.eval.gv.string_modes [--dataset gaps|guitarset] [--limit N]
"""
import argparse
import glob
import os

import mir_eval
import numpy as np

from gtab.config import FRAME_RATE
from gtab.core.instrument import STANDARD_6
from gtab.data.gaps import gaps_files
from gtab.data.torch_dataset import _normalize
from gtab.decoding.edition import order_notes
from gtab.decoding.transitions import build_lattice, notes_from_tab
from gtab.decoding.viterbi import energy_rise, segment
from gtab.models.edition_net import predict as edition_predict
from gtab.models.inference import predict_heads
from gtab.pipeline import heard_positions, load_models
from gtab.utils import get_device
from gtab.paths import CACHE_DIR
from scripts.eval.gv.paths import GAPS_TAB, GAPS_TEST
from scripts.eval.gv.test_b_resynth import mir

W_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
TAU_GRID = (0.5, 1.0, 1.5, 2.0, 3.0)


def split_items(split: str) -> list[tuple[str, str]]:
    if split == "val":
        return [(os.path.basename(f)[:-4], f) for f in sorted(glob.glob(os.path.join(CACHE_DIR, "val", "*.npz")))]
    files = gaps_files("gaps_train")[1] if split == "gaps_val" else sorted(map(str, GAPS_TEST.glob("*.npz")))
    return [(os.path.basename(f)[:-4], f) for f in files if (GAPS_TAB / os.path.basename(f)).exists()]


def labels(split: str, tid: str, d) -> tuple[np.ndarray, np.ndarray]:
    # -> (notes (N, 3) onset s, offset s, pitch; string (N,), 0 = low E, -1 = unknown)
    if split == "val":
        nt = notes_from_tab(d["tab"])
        return np.array([n[:3] for n in nt], float).reshape(-1, 3), np.array([n[3] for n in nt], int)
    with np.load(GAPS_TAB / f"{tid}.npz") as g:
        return g["notes"].astype(float), g["string"].astype(int)


def notes_of(cqt: np.ndarray, m: dict) -> tuple[list, np.ndarray, np.ndarray]:
    # same path as gtab.pipeline.analyze
    _, thr, ot, rules = m["sel"]
    probs, ons, offs, pons = predict_heads(m["model"], _normalize(cqt), m["device"], with_pitch_onset=True)
    segs = segment(probs, ons, thr, ot, STANDARD_6, rise=energy_rise(cqt, STANDARD_6), offsets=offs,
                   pitch_onsets=pons if rules.get("onset_source") == "pitch" else None, **rules)
    notes = np.array([(a / FRAME_RATE, b / FRAME_RATE, p) for a, b, p in segs], float).reshape(-1, 3)
    return segs, probs, notes


def strings_by_variant(segs: list, probs: np.ndarray, notes: np.ndarray, m: dict) -> dict[str, np.ndarray]:
    out = {}
    for w in W_GRID:
        out[f"ses w={w:g}"] = np.array([-1 if s is None else s for s, _ in heard_positions(segs, probs, m["tm"],
                                                                                      w_transition=w)])
    o, chord = order_notes(notes)
    s = np.full(len(notes), -1)
    s[o] = edition_predict(m["edition"], notes[o], chord, m["device"])[0]
    out["edisyon"] = s
    gap = confidence_gap(segs, probs, m)
    groups = np.split(o, np.flatnonzero(~chord)[1:])
    for tau in TAU_GRID:
        out[f"hibrit t={tau:g}"] = hybrid(out["ses w=0"], s, gap, tau, groups)
    out["_gap"] = gap
    return out


def confidence_gap(segs: list, probs: np.ndarray, m: dict) -> np.ndarray:
    # log-prob gap between the heard-position model's best and second-best string (inf: only one position)
    _, E, _ = build_lattice(segs, probs, m["tm"])
    return np.array([np.inf if len(e) < 2 else float(np.diff(np.sort(e)[-2:])[0]) for e in E])


def hybrid(ses: np.ndarray, ed: np.ndarray, gap: np.ndarray, tau: float, groups: list) -> np.ndarray:
    s = np.where((gap < tau) & (ed >= 0), ed, ses)
    for g in groups:
        if len(set(s[g])) < len(g):
            s[g] = ed[g]
    return s


def main(dataset: str, limit: int | None) -> None:
    m = load_models(get_device(), domain="val" if dataset == "guitarset" else "gaps_val")
    for split in (("val",) if dataset == "guitarset" else ("gaps_val", "gaps_test")):
        hits, n = {}, 0
        for tid, f in split_items(split)[:limit]:
            with np.load(f) as d:
                ref, ref_s = labels(split, tid, d)
                segs, probs, notes = notes_of(d["cqt"], m)
            if not len(notes):
                continue
            pairs = mir_eval.transcription.match_notes(*mir(ref), *mir(notes), onset_tolerance=0.05, offset_ratio=None)
            known = [(i, j) for i, j in pairs if ref_s[i] >= 0]
            n += len(known)
            var = strings_by_variant(segs, probs, notes, m)
            gap = var.pop("_gap")
            for tau in TAU_GRID:
                hits[f"  share t={tau:g}"] = hits.get(f"  share t={tau:g}", 0) + sum(gap[j] < tau for _, j in known)
            for name, s in var.items():
                hits[name] = hits.get(name, 0) + sum(s[j] == ref_s[i] for i, j in known)
        print(f"{split}: matched labelled notes {n}")
        for name, h in hits.items():
            what = "notes handed to edisyon" if name.startswith("  share") else "string accuracy"
            print(f"  {name:12s} {what} {h / max(n, 1):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["gaps", "guitarset"], default="gaps")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    main(a.dataset, a.limit)
