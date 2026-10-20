"""Check classtab.events against classtab's own MIDI files (made from the same tabs): the parsed pitch sequence must
be found, in order, in the MIDI. Also reports how many tabs parse at all and the pieces used in Test E/F.
Run from the repo root: python -m scripts.eval.gv.validate_parser data/raw/gv/classtab
"""
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pretty_midi

from scripts.eval.gv import classtab as c

PIECES = ["sor_op31_no01_in_c_andante", "tarrega_recuerdos_de_la_alhambra", "carcassi_op26no1_capricho_in_c",
          "carcassi_op26no3_capricho_in_em", "sor_op35_no17_moderato_in_d", "carcassi_op60_no06_study_in_c",
          "anon_el_noi_de_la_mare", "tarrega_lagrima", "tarrega_marieta_mazurka", "carcassi_op60_no07_study_in_am",
          "tarrega_estudio_en_forma_de_minuetto"]


def midi_agreement(tab: Path, mid: Path) -> tuple[float, float] | None:
    # (tab notes found in MIDI, MIDI notes found in tab): the second one drops if a tab is cut short
    ev = c.events(tab)
    if ev is None or not 30 <= len(ev) <= 2500:
        return None
    notes = sorted((n.start, n.pitch) for i in pretty_midi.PrettyMIDI(str(mid)).instruments if not i.is_drum
                   for n in i.notes)
    m = np.array(notes, float)
    if len(m) > 4000:
        return None
    ti, mi = c.sequence(ev[:, 0], ev[:, 1], 0.5), c.sequence(m[:, 0], m[:, 1], 0.03)
    k = len(c.lcs_pairs(ev[ti, 1].astype(int), m[mi, 1].astype(int)))
    return k / len(ev), k / len(m)


def main(d: Path) -> None:
    warnings.filterwarnings("ignore")
    tabs = sorted(d.glob("*.txt"))
    print(f"tabs with no parsed notes: {sum(c.events(p) is None for p in tabs)}/{len(tabs)}")
    for s in PIECES:
        ev = c.events(d / f"{s}.txt")
        print(f"  {s}: {0 if ev is None else len(ev)} notes")
    random.seed(0)
    pairs = [p for p in tabs if (d / f"{p.stem}.mid").exists()]
    r = np.array([x for p in random.sample(pairs, 120) if (x := midi_agreement(p, d / f"{p.stem}.mid")) is not None])
    for k, name in enumerate(("tab notes in MIDI", "MIDI notes in tab")):
        print(f"validation {name}: pieces {len(r)}, median {np.median(r[:, k]):.3f}, >=0.9 {np.mean(r[:, k] >= .9):.0%}")
    for s in ("tarrega_lagrima", "tarrega_marieta_mazurka"):
        print(f"  {s}: {np.round(midi_agreement(d / f'{s}.txt', d / f'{s}.mid'), 3)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
