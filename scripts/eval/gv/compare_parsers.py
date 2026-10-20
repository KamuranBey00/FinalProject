"""Current classtab parser vs the baseline used in Test E/F (labelled lines only). Accept the new rule only if
previously parsed tabs do not get worse against their own MIDI and newly parsed tabs agree with theirs.
The two versions are loaded as independent module objects (a reload would make them share globals).
Run from the repo root: python -m scripts.eval.gv.compare_parsers data/raw/gv/classtab
"""
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np

from scripts.eval.gv import validate_parser as v

HERE = Path(__file__).parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / "classtab.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def agreement(mod, d: Path, stem: str):
    v.c = mod
    return v.midi_agreement(d / f"{stem}.txt", d / f"{stem}.mid")


def summarize(label: str, keys: list[str], old, new, d: Path) -> None:
    rows = []
    for k in (k for k in keys if (d / f"{k}.mid").exists()):
        ro, rn = agreement(old, d, k), agreement(new, d, k)
        rows.append((*(ro if ro is not None else (np.nan, np.nan)), *(rn if rn is not None else (np.nan, np.nan))))
    if not rows:
        print(f"  {label}: no MIDI to compare")
        return
    r = np.array(rows, float)
    print(f"  {label} with MIDI ({len(r)}): tab-in-MIDI {np.nanmedian(r[:, 0]):.3f} -> {np.nanmedian(r[:, 2]):.3f}, "
          f"MIDI-in-tab {np.nanmedian(r[:, 1]):.3f} -> {np.nanmedian(r[:, 3]):.3f}, "
          f"worse tab-in-MIDI(>0.05) {int(np.nansum(r[:, 2] < r[:, 0] - 0.05))}")


def main(d: Path) -> None:
    warnings.filterwarnings("ignore")
    new, old = load("classtab_current"), load("classtab_baseline")
    rule = lambda l: bool(old.LINE.match(l)) and l.count("-") >= 5
    old.is_block = lambda block: all(rule(l) for l in block)
    stems = [p.stem for p in sorted(d.glob("*.txt"))]
    oe, ne = {k: old.events(d / f"{k}.txt") for k in stems}, {k: new.events(d / f"{k}.txt") for k in stems}
    print(f"unparsed: baseline {sum(x is None for x in oe.values())} -> new {sum(x is None for x in ne.values())}")
    changed = [k for k in stems if oe[k] is not None
               and (ne[k] is None or oe[k].shape != ne[k].shape or not np.array_equal(oe[k], ne[k]))]
    gained = [k for k in stems if oe[k] is None and ne[k] is not None]
    lost = [k for k in stems if oe[k] is not None and ne[k] is None]
    print(f"previously parsed and changed: {len(changed)} (now unparsed: {len(lost)}) | newly parsed: {len(gained)}")
    summarize("changed", changed, old, new, d)
    summarize("newly parsed", gained, old, new, d)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
