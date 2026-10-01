"""
Katman 1 — Özellik çıkarımı: ses dalgası -> log-CQT.

Neden log-CQT?
- CQT frekans eksenini nota mantığıyla böler (Katman 0'da anlattık).
- Genliği dB'ye (log) çeviririz çünkü kulak da, model de sesi logaritmik algılar;
  zayıf ve güçlü notalar arası fark bu sayede dengelenir.

Çıktı konvansiyonu: (n_frames, n_bins)  -> yani "her satır bir zaman karesi".
Bu düzeni tüm projede koru; model girdileri buna göre şekillenir.
"""

import numpy as np
import librosa

from gtab.config import (
    SAMPLE_RATE, HOP_LENGTH, CQT_FMIN_HZ,
    CQT_BINS_PER_OCTAVE, CQT_N_BINS,
)


def load_audio(path: str) -> np.ndarray:
    """Diskteki sesi tek kanala indirip sabit sample rate'e getirir."""
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return y


def compute_cqt(y: np.ndarray) -> np.ndarray:
    """
    Ses dalgası -> log-CQT, şekil (n_frames, n_bins).
    """
    C = librosa.cqt(
        y,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        fmin=CQT_FMIN_HZ,
        bins_per_octave=CQT_BINS_PER_OCTAVE,
        n_bins=CQT_N_BINS,
    )
    C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)   # (n_bins, n_frames)
    return C_db.T.astype(np.float32)                        # -> (n_frames, n_bins)


def cqt_from_file(path: str) -> np.ndarray:
    return compute_cqt(load_audio(path))
