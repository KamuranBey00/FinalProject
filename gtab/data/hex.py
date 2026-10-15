"""
ArmaganDegisiklikler — GuitarSet hexaphonic pickup -> per-string spectra (auxiliary training target).

Why: the A-string analysis (ArmaganDegisiklikler) showed the model picks strings mostly from the
training players' habits; the mic CQT alone does not make it separate string timbres. GuitarSet's
hex pickup records each string on its own channel, so "how the mic spectrum splits over the six
strings" can be supervised directly. Only used as a training target; inference stays mic-only.

Output convention: (n_frames, 6, n_bins // POOL) in [0, 1], same frame grid as data/cache/<split>.
String index 0 = low E (project convention); GuitarSet hex channel 0 = low E as well.
"""

from pathlib import Path

import librosa
import numpy as np

from gtab.config import CQT_BINS_PER_OCTAVE, CQT_FMIN_HZ, CQT_N_BINS, HOP_LENGTH, SAMPLE_RATE

POOL = 2                      # 192 CQT bins -> 96 (one bin per semitone): enough for harmonic envelopes
FLOOR_DB = 80.0               # same dynamic range as the mic input (_normalize: (dB + 80) / 80)


def hex_string_spectra(path: Path, n_frames: int) -> np.ndarray:
    """
    6-channel hex WAV -> per-string log-CQT, normalised like the model input.
    All six channels share one dB reference (the loudest bin of the whole recording), so the
    relative level between strings is kept; a per-channel reference would erase it.
    Frames are trimmed / zero-padded to n_frames (the mic cache length).
    """
    y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=False)       # (6, samples)
    if y.ndim != 2 or y.shape[0] != 6:
        raise ValueError(f"6 kanalli hex bekleniyor, gelen sekil {y.shape}: {path}")
    mags = np.stack([np.abs(librosa.cqt(ch, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, fmin=CQT_FMIN_HZ,
                                        bins_per_octave=CQT_BINS_PER_OCTAVE, n_bins=CQT_N_BINS))
                     for ch in y])                                     # (6, bins, frames)
    db = librosa.amplitude_to_db(mags, ref=np.max(mags))
    spec = np.clip((db + FLOOR_DB) / FLOOR_DB, 0.0, 1.0)
    spec = spec.reshape(6, CQT_N_BINS // POOL, POOL, -1).mean(2)      # (6, bins/POOL, frames)
    spec = spec.transpose(2, 0, 1)                                     # (frames, 6, bins/POOL)
    out = np.zeros((n_frames, 6, CQT_N_BINS // POOL), np.float16)
    n = min(n_frames, spec.shape[0])
    out[:n] = spec[:n]
    return out


def string_energy_agreement(spec: np.ndarray, tab: np.ndarray) -> float:
    """
    Sanity check: on frames where exactly one string is labelled, how often is that string's
    hex channel the loudest one. Low values mean misaligned frames or a wrong channel order.
    """
    one = (tab > 0).sum(1) == 1
    if not one.any():
        return float("nan")
    loud = spec[one].astype(np.float32).mean(2).argmax(1)
    return float((loud == (tab[one] > 0).argmax(1)).mean())
