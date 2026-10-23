"""External test on our own recordings (docs/KAYIT_PROTOKOLU.md E01–E05 + the task sheet "Kayit_Gorevleri" T01, T02
blocks 1–6, given by the user 22 Oct 2026): TEST ONLY — never used for training,
threshold or checkpoint selection (they are the one set no decision has looked at).

The protocol fixes every note's string and fret, so these recordings give what GAPS cannot: the PLAYED string on a
classical guitar. There are no note times (metronome, played by hand, small slips such as a forgotten note), so the
model output is compared like the classtab tests (README20 / Test E-F): pitch sequences aligned with an LCS, chords
as one group; reported: share of expected notes found in order, share of model notes matched, and string agreement on
the matched notes. Protocol string numbers t1..t6 (t1 = high e) are converted to the pipeline's 0 = low E.
Audio: data/raw/gv/own_wav/<name>.wav (converted from the phone recordings; git-ignored).
Run from the repo root: python -m scripts.eval.gv.own_recordings [--ckpt X.pt --cal X.cal.json]
"""
import argparse
import glob
import re
from pathlib import Path

import numpy as np

from gtab.core.instrument import STANDARD_6
from gtab.data.features import cqt_from_file
from gtab.evaluation.pitch_eval import load_selection
from gtab.pipeline import analyze, load_models
from gtab.models.inference import load_model
from gtab.paths import ckpt_path
from gtab.utils import get_device
from scripts.eval.gv import classtab
from scripts.eval.gv.paths import RAW

WAV = RAW / "own_wav"
TUNING = STANDARD_6.tuning                     # low E .. high e


def pos(t: int, fret: int) -> tuple[int, int]:
    s = 6 - t                                  # protocol t1 (high e) .. t6 (low E) -> pipeline 5 .. 0
    return TUNING[s] + fret, s


def e01() -> list[list[tuple[int, int]]]:
    frets = list(range(6)) + list(range(5, -1, -1))
    return [[pos(t, f)] for t in range(6, 0, -1) for f in frets]


def e02() -> list[list[tuple[int, int]]]:
    spots = [(1, 0), (2, 5), (3, 9), (4, 14), (2, 0), (3, 4), (4, 9), (5, 14),
             (3, 0), (4, 5), (5, 10), (6, 15), (4, 0), (5, 5), (6, 10)]
    return [[pos(t, f)] for t, f in spots for _ in range(2)]


def e03() -> list[list[tuple[int, int]]]:
    up = [(5, 3), (4, 0), (4, 2), (4, 3), (3, 0), (3, 2), (2, 0), (2, 1), (2, 3), (1, 0), (1, 1), (1, 3)]
    return [[pos(t, f)] for t, f in up + up[-2::-1]]


E04_CHORDS = {  # bass (p) + strings 3, 2, 1 (i, m, a)
    "C": [(5, 3), (3, 0), (2, 1), (1, 0)], "Am": [(5, 0), (3, 2), (2, 1), (1, 0)],
    "Dm": [(4, 0), (3, 2), (2, 3), (1, 1)], "E": [(6, 0), (3, 1), (2, 0), (1, 0)],
    "G": [(6, 3), (3, 0), (2, 0), (1, 3)],
}


REPEATS = 2   # user, 22 Oct 2026: in the recordings every chord was played 2 times (protocol: 4)


def e04() -> list[list[tuple[int, int]]]:
    return [[pos(t, f)] for c in ("C", "Am", "Dm", "E", "G", "C") for _ in range(REPEATS) for t, f in E04_CHORDS[c]]


def e05() -> list[list[tuple[int, int]]]:
    chords = [[(6, 0), (1, 0)], [(5, 3), (3, 0), (2, 1)], [(5, 0), (4, 2), (3, 2), (2, 1)],
              [(6, 0), (5, 2), (4, 2), (3, 1), (2, 0), (1, 0)]]
    return [[pos(t, f) for t, f in c] for c in chords for _ in range(REPEATS)]


def t01() -> list[list[tuple[int, int]]]:
    # string 6 -> 1, frets 0..12 on each (78 notes)
    return [[pos(t, f)] for t in range(6, 0, -1) for f in range(13)]


T02_BLOCKS = {1: [(1, 0), (2, 5), (3, 9)], 2: [(2, 0), (3, 4), (4, 9)], 3: [(3, 2), (4, 7), (5, 12)],
              4: [(2, 3), (3, 7), (4, 12)], 5: [(1, 3), (2, 8), (3, 12)], 6: [(4, 2), (5, 7), (6, 12)]}


def t02(block: int) -> list[list[tuple[int, int]]]:
    # each position twice, then one beat rest (the rest is not a note)
    return [[pos(t, f)] for t, f in T02_BLOCKS[block] for _ in range(2)]


EXPECTED = {"E01": e01, "E02": e02, "E03": e03, "E04": e04, "E05": e05, "T01": t01}


AUTUMN = RAW / "own" / "autumn_leaves_tab.tsv"   # parsed from the guitartabmaker PDF (third-party: git-ignored)


def e15() -> list[list[tuple[int, int]]] | None:
    if not AUTUMN.exists():
        return None
    rows = np.loadtxt(AUTUMN, skiprows=1, dtype=int).reshape(-1, 3)
    return [[(int(p), int(s)) for _, p, s in rows[rows[:, 0] == o]] for o in np.unique(rows[:, 0])]


def expected_for(stem: str) -> list[list[tuple[int, int]]] | None:
    # file names: "E04-Arpej-...", "T01 - ...", "T02- Blok-1", "T02 - blok2", "E15-..." ...
    if stem.startswith("E15"):
        return e15()
    if stem.startswith("T02"):
        return t02(int(re.search(r"(\d)\s*$", stem).group(1)))
    f = EXPECTED.get(stem[:3])
    return f() if f else None


def tab_array(groups: list[list[tuple[int, int]]]) -> np.ndarray:
    # classtab.events layout: (order, pitch, string); a chord shares one order value
    return np.array([(k, p, s) for k, g in enumerate(groups) for p, s in g], float)


def models(ckpt: str | None, cal: str | None) -> dict:
    m = load_models(get_device())
    if ckpt:                                   # candidate model: acoustic + its decoding, rest of the stack unchanged
        m["model"] = load_model("crnn", m["device"], ckpt)[0]
        m["sel"] = load_selection(ckpt_path(cal))["gaps_val"]
    return m


def main(ckpt: str | None, cal: str | None) -> None:
    m = models(ckpt, cal)
    rows = []
    for wav in sorted(glob.glob(str(WAV / "*.wav"))):
        ex, groups = Path(wav).name[:3], expected_for(Path(wav).stem)
        if groups is None:                     # no reference for this file
            continue
        tab = tab_array(groups)
        cqt = cqt_from_file(wav)
        for mode in ("ses", "edisyon"):
            an = analyze(cqt, m, mode)
            r = classtab.compare(tab, an["notes"], an["string"])
            rows.append((ex, mode, r))
            print(f"{Path(wav).stem[:40]:40s} {mode:8s} expected {r['tab_notes']:3d} model {r['model_notes']:3d}  "
                  f"found {r['tab_found']:.3f}  matched {r['model_matched']:.3f}  string {r['string_agree']:.3f}",
                  flush=True)
    for mode in ("ses", "edisyon"):
        x = [r for _, md, r in rows if md == mode]
        for ex in sorted({e for e, _, _ in rows}):
            y = [r for e, md, r in rows if md == mode and e == ex]
            print(f"  {ex} {mode:8s} found {np.mean([r['tab_found'] for r in y]):.3f}  string "
                  f"{np.nanmean([r['string_agree'] for r in y]):.3f}")
        w = lambda k: np.average([r[k] for r in x], weights=[r["tab_notes"] for r in x])
        print(f"ALL {mode:8s} found {w('tab_found'):.3f}  matched {w('model_matched'):.3f}  string "
              f"{np.nanmean([r['string_agree'] for r in x]):.3f}  (weighted by expected notes; string = mean)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--cal")
    a = ap.parse_args()
    main(a.ckpt, a.cal)
