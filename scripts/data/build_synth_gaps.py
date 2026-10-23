"""README22 Stage 1b — timbre augmentation: GAPS training recordings re-played with other nylon-guitar sample sets.

For every gaps_fit recording (training part of gaps_train; validation and test get NO synthetic copy) the label notes
(the same frame / onset rolls the model is trained on, so the copy is aligned frame by frame) are rendered with a
sample set (scripts.eval.gv.samplers: nearest-root sample, pitch-shifted, cut at the label offset) at a random
per-note velocity, and the CQT is computed exactly like the cache (gtab.data.features.compute_cqt). Output follows the
alternative-copy layout of build_degraded, so `train_onset --degrade-root data/cache/synth` uses it unchanged:

  data/cache/synth/<library>/gaps_train/<id>.npz   (cqt float16)

Libraries: only sets whose license allows it (FreePats viewer font CC0, Ichiyanagi CC BY-NC 4.0 "works made with the
soundfont as a tool are free"). Existing files are skipped. Seeded: the same recording gets the same velocities.
Run from the repo root: python -m scripts.data.build_synth_gaps [--limit N]
"""
import argparse
import os
import zlib

import numpy as np

from gtab.core.instrument import STANDARD_6
from gtab.config import FRAME_RATE
from gtab.data.features import compute_cqt
from gtab.data.gaps import gaps_files
from gtab.evaluation.pitch_eval import roll_to_notes
from gtab.paths import CACHE_DIR
from scripts.eval.gv import samplers
from scripts.eval.gv.paths import ICHIYANAGI, RAW, VIEWER_FONT

OUT = os.path.join(CACHE_DIR, "synth")
VELOCITY = (0.3, 0.8)                       # uniform per note
PROGRESS = RAW / "progress_synth.txt"


def libraries() -> dict[str, list[samplers.Sample]]:
    return {"freepats": samplers.from_viewer_font(VIEWER_FONT), "ichiyanagi": samplers.from_soundfont(ICHIYANAGI)}


def synth_cqt(path: str, lib: list[samplers.Sample]) -> np.ndarray:
    with np.load(path) as d:
        frame, onset, T = d["frame"] > 0, d["onset"] > 0, d["cqt"].shape[0]
    notes = np.array([(a / FRAME_RATE, b / FRAME_RATE, p)
                      for a, b, p in roll_to_notes(frame, onset, STANDARD_6.pitch_range()[0])], float).reshape(-1, 3)
    rng = np.random.default_rng(zlib.crc32(os.path.basename(path).encode()))
    y = samplers.render(notes, lib, rng.uniform(*VELOCITY, len(notes))) if len(notes) else np.zeros(samplers.SR)
    c = compute_cqt(y.astype(np.float32))[:T]
    return np.pad(c, ((0, T - len(c)), (0, 0)), constant_values=c.min()) if len(c) < T else c


def main(limit: int | None) -> None:
    fit = gaps_files("gaps_train")[0][:limit]
    for name, lib in libraries().items():
        d = os.path.join(OUT, name, "gaps_train")
        os.makedirs(d, exist_ok=True)
        for k, path in enumerate(fit, 1):
            out = os.path.join(d, os.path.basename(path))
            if not os.path.exists(out):
                np.savez_compressed(out, cqt=synth_cqt(path, lib).astype(np.float16))
            with open(PROGRESS, "a", encoding="utf-8") as f:
                f.write(f"{name} {k}\n")
        print(f"{name}: {len(fit)} recordings -> {d}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    main(ap.parse_args().limit)
