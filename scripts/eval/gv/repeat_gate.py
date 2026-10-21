"""Analysis only (no product change): are same-pitch notes that land on the same TAB position real repeats or false
splits, and does the evidence the segmenter used (gtab.decoding.viterbi.segment_notes_onset gates: pitch-onset peak
>= reattack, energy rise >= rise_keep / rise_split, previous note already ended) tell them apart?

Pairs = consecutive detected notes of the same pitch, later one starting < MAX_IOI after the earlier one.
  real  both match score notes (pitch, +-50 ms) at different score positions
  false the later one matches no score note (the earlier one does)
Gate of the later note: 'ended' (previous note had stopped: not a re-attack decision), 'onset' (peak >= reattack),
'rise' (rise >= rise_keep), 'split' (added by rise_split). Same user-input simulation as clash_rhythm.
Run from the repo root: python -m scripts.eval.gv.repeat_gate [--split gaps_val] [--limit N]
"""
import argparse
import os
from collections import Counter, defaultdict

import numpy as np

from gtab.config import FRAME_RATE
from gtab.core.instrument import STANDARD_6
from gtab.data.gaps import GAPS_DIR
from gtab.data.gaps_score import match_notes, time_signatures
from gtab.data.torch_dataset import _normalize
from gtab.decoding.viterbi import energy_rise, pitch_matrix
from gtab.evaluation.rhythm_metrics import tempo_of
from gtab.models.inference import predict_heads
from gtab.models.rhythm_net import TICKS
from gtab.pipeline import analyze, load_models, rhythm
from gtab.utils import get_device
from scripts.eval.gv.clash_rhythm import ONSET_TOL, RHY_DIR, USER_ERR, split_files

MAX_IOI = 0.25


def evidence(cqt: np.ndarray, m: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, thr, _, rules = m["sel"]
    probs, _, _, pons = predict_heads(m["model"], _normalize(cqt), m["device"], with_pitch_onset=True)
    pm, _ = pitch_matrix(probs, STANDARD_6, rules.get("combine", "max"))
    cont = pm > thr * rules.get("off_ratio", 1.0)
    return pons, energy_rise(cqt, STANDARD_6), cont


def gate(a: int, p: int, pons: np.ndarray, rise: np.ndarray, cont: np.ndarray, rules: dict) -> str:
    if a == 0 or not cont[a - 1, p]:
        return "ended"
    if pons[a, p] >= rules.get("reattack", 0) > 0:
        return "onset"
    if rise[max(0, a - 1):a + 3, p].max() >= rules.get("rise_keep", 0) > 0:
        return "rise"
    return "split"


def pairs_of_piece(f: str, k: int, m: dict) -> list[dict]:
    tid = os.path.basename(f)[:-4]
    with np.load(f) as d:
        cqt = d["cqt"]
    with np.load(os.path.join(RHY_DIR, tid + ".npz")) as g:
        gb, gn, s = g["beats"], g["notes"], g["note_beat"]
    sig = Counter(time_signatures(os.path.join(GAPS_DIR, "musicxml", f"{tid}.xml"))).most_common(1)[0][0]
    an = analyze(cqt, m, "ses")
    det, rules, lo = an["notes"], m["sel"][3], STANDARD_6.pitch_range()[0]
    tick = np.round(np.asarray(rhythm(an, m, sig, tempo_of(gb) * (1 + USER_ERR * (-1) ** k))["q"]) * TICKS)
    to_det, _ = match_notes([(on, int(p)) for on, _, p in gn], [dict(pitch=int(p)) for _, _, p in det], det[:, 0],
                            ONSET_TOL)
    mi = np.full(len(det), -1)
    mi[to_det[to_det >= 0]] = np.flatnonzero(to_det >= 0)
    pons, rise, cont = evidence(cqt, m)
    out, last = [], {}
    for j in np.argsort(det[:, 0], kind="stable"):
        p = int(det[j, 2])
        i = last.get(p)
        last[p] = j
        if i is None or det[j, 0] - det[i, 0] >= MAX_IOI or mi[i] < 0:
            continue
        kind = "false" if mi[j] < 0 else ("real" if abs(s[mi[i]] - s[mi[j]]) > 1e-6 else None)
        if kind is None:
            continue
        a = int(round(det[j, 0] * FRAME_RATE))
        out.append(dict(kind=kind, same_tick=tick[i] == tick[j], ioi=det[j, 0] - det[i, 0],
                        gate=gate(a, p - lo, pons, rise, cont, rules), peak=float(pons[a, p - lo]),
                        rise=float(rise[max(0, a - 1):a + 3, p - lo].max())))
    return out


def report(rows: list[dict]) -> None:
    for scope, sel in (("all pairs (IOI < 250 ms)", rows), ("same TAB position", [r for r in rows if r["same_tick"]])):
        n = Counter(r["kind"] for r in sel)
        print(f"\n{scope}: real repeats {n['real']}, false splits {n['false']}")
        by = defaultdict(Counter)
        for r in sel:
            by[r["gate"]][r["kind"]] += 1
        for g, c in sorted(by.items()):
            print(f"    gate {g:6s} real {c['real']:5d}  false {c['false']:5d}")
        for kind in ("real", "false"):
            x = [r for r in sel if r["kind"] == kind]
            if x:
                q = lambda key: np.percentile([r[key] for r in x], [25, 50, 75]).round(3).tolist()
                print(f"    {kind:5s} IOI s {q('ioi')}  onset peak {q('peak')}  rise dB {q('rise')}")


def gate_grid(rows: list[dict]) -> None:
    # onset-gated pairs only: keep the later note if peak >= hi or rise >= r (today: hi = reattack, r = 0)
    x = [r for r in rows if r["gate"] == "onset"]
    print(f"\nonset-gate pairs {len(x)}: kept real / kept false for 'peak >= hi or rise >= r'")
    for hi in (0.6, 0.7, 0.8):
        cells = []
        for rmin in (2.0, 3.0, 4.0):
            keep = [r for r in x if r["peak"] >= hi or r["rise"] >= rmin]
            cells.append(f"r>={rmin:.0f}: {sum(r['kind'] == 'real' for r in keep):4d} / {sum(r['kind'] == 'false' for r in keep):4d}")
        print(f"    hi {hi:.1f}  " + "   ".join(cells))


def main(split: str, limit: int | None) -> None:
    m = load_models(get_device())
    rows = []
    for k, f in enumerate(split_files(split)[:limit]):
        rows += pairs_of_piece(f, k, m)
    print(f"{split}: selection rules {m['sel'][3]}")
    report(rows)
    gate_grid(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="gaps_val")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    main(a.split, a.limit)
