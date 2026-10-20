"""Re-measure MDM with the library's own SFZ note mapping as the reference (pYIN misread 9/37 samples by +12).
Test A part: single notes, reference = SFZ key. Test B part: 28 GAPS test pieces rendered with SFZ-rooted samples.
Run from the repo root: python -m scripts.eval.gv.rerun_mdm
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from gtab.data.features import cqt_from_file
from gtab.pipeline import analyze, load_models
from gtab.utils import get_device
from scripts.eval.gv import samplers, test_a_single_notes as A, test_b_resynth as B
from scripts.eval.gv.paths import LIBRARIES


def main() -> None:
    m = load_models(get_device())
    sfz = next(p for p in LIBRARIES.rglob("*.sfz") if "MDM" in p.name)  # ships with the library
    roots = samplers.sfz_roots(sfz.read_text(encoding="utf-8"))
    files = sorted(p for p in LIBRARIES.rglob("*.wav") if "MDM" in str(p) and "__MACOSX" not in p.parts)
    rows = [A.summarize(analyze(cqt_from_file(str(p)), m, "ses"), roots[p.name]) for p in files]
    found, octave = np.mean([r["found_ref"] for r in rows]), np.mean([r["octave_err"] for r in rows])
    main_ok = np.mean([r["main_pitch"] == roots[p.name] for r, p in zip(rows, files)])
    print(f"Test A MDM (SFZ reference): found {found:.3f}, longest note correct {main_ok:.3f}, octave error {octave:.3f}, "
          f"exactly one note {np.mean([r['n_notes'] == 1 for r in rows]):.3f}")
    lib = samplers.from_wavs(files, roots)
    ids = sorted({p.stem for p in B.TEST.glob("*.npz")} & {p.stem for p in B.TAB.glob("*.npz")})
    tmp, F, acc = LIBRARIES / "tmp_mdm.wav", [], []
    for tid in ids:
        with np.load(B.TAB / f"{tid}.npz") as d:
            ref, ref_s = d["notes"].astype(float), d["string"].astype(int)
        sf.write(tmp, samplers.render(ref, lib), samplers.SR)
        s = B.score(analyze(cqt_from_file(str(tmp)), m, "ses"), ref, ref_s)
        F.append(s["F"]); acc.append(s["string_acc"])
    tmp.unlink(missing_ok=True)
    print(f"Test B MDM (SFZ roots, trimmed): pieces {len(F)}, F {np.mean(F):.3f} (min {np.min(F):.3f}), "
          f"string_acc {np.nanmean(acc):.3f}")


if __name__ == "__main__":
    sys.exit(main())
