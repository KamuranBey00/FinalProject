"""README22 Step 2A: select the two-evidence re-attack gate (viterbi.segment_notes_onset reattack_hi / reattack_rise)
on gaps_val with the same rule as gtab.evaluation.pitch_eval._try: a candidate replaces the current selection only if
it raises the validation score (criterion 'mix' = (note F1 + frame F1) / 2, the one the active selection
was made with: its saved score 0.734 = (0.780 + 0.688) / 2, README13). gaps_test is evaluated afterwards for reporting; it never influences the choice.
Prints note P/R/F1, detected notes and the fast-repeat miss rate (IOI < 200 ms) for every candidate.
Run from the repo root: python -m scripts.eval.gv.calibrate_reattack [--limit N]
"""
import argparse

from gtab.evaluation.pitch_eval import evaluate_split, load_split, score
from gtab.pipeline import load_models
from gtab.utils import get_device

CRITERION = "mix"

GRID = [dict(reattack_hi=hi, reattack_rise=r) for hi in (0.0, 0.6, 0.7, 0.8) for r in (0.0, 2.0, 3.0, 4.0)
        if (hi, r) != (0.0, 0.0)]


def row(label: str, r: dict) -> str:
    p, rc, f = r["note"]
    n_rep, miss = r["rep_fast"]
    return (f"  {label:22s} mix {score(r, CRITERION):.4f}  note P {p:.4f} R {rc:.4f} F1 {f:.4f}  frame F1 "
            f"{r['frame'][2]:.4f}  fast-repeat miss {miss:.3f} (n {n_rep})")


def run(data: list, t0: float, o0: float, dec: dict) -> dict:
    return evaluate_split(data, thresholds=[t0], onset_thr=o0, dec=dec)[t0]


def main(limit: int | None) -> None:
    m = load_models(get_device())
    f_saved, t0, o0, d0 = m["sel"]
    val = load_split("gaps_val", m["model"], None, "crnn", m["device"], limit)
    base = run(val, t0, o0, d0)
    print(f"gaps_val current selection: score {score(base, CRITERION):.4f} (saved {f_saved:.4f})")
    print(row("current", base))
    best = (score(base, CRITERION), d0, "current")
    for g in GRID:
        r = run(val, t0, o0, dict(d0, **g))
        label = f"hi {g['reattack_hi']:.1f} rise {g['reattack_rise']:.0f} dB"
        print(row(label, r))
        if score(r, CRITERION) > best[0]:
            best = (score(r, CRITERION), dict(d0, **g), label)
    print(f"selected on gaps_val: {best[2]} (score {best[0]:.4f})")
    test = load_split("gaps_test", m["model"], None, "crnn", m["device"], limit)
    print("gaps_test (report only):")
    print(row("current", run(test, t0, o0, d0)))
    print(row(best[2], run(test, t0, o0, best[1])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    main(ap.parse_args().limit)
