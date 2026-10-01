"""
Katman 1 — Etiket dönüşümü: NoteEvent listesi -> kare-seviye hedef matrisleri.

Model, sesi kare kare işler; o yüzden sembolik notalarımızı da kare-seviye
"piano roll"a çevirmemiz gerekir. İki hedef üretiyoruz:

  1) frame_roll : her karede hangi perde AKTİF  -> perde tahmini (multi-pitch)
  2) onset_roll : her karede hangi perde BAŞLADI -> nota başlangıcı tespiti

Bu ikisi ayrı ayrı önemli: bir nota uzun sürerken "aktif" kalır ama "başlangıç"
sadece ilk karede olur. Katman 4'te hammer-on/pull-off'u (vuruşsuz başlangıç)
tam olarak onset sinyalinin YOKLUĞUNDAN ayırt edeceğiz — yani bu ayrımı şimdiden
doğru kurmak ileriye yatırım.

Perde aralığı enstrümandan türetilir (sabit kodlanmaz -> farklı gitarlar için future-proof).
"""

import numpy as np

from .note_event import Transcription
from .instrument import Instrument, STANDARD_6
from .config import FRAME_RATE


def midi_range(instrument: Instrument = STANDARD_6):
    lo, hi = instrument.pitch_range()
    return lo, hi


def notes_to_rolls(
    transcription: Transcription,
    n_frames: int,
    instrument: Instrument = STANDARD_6,
    frame_rate: float = FRAME_RATE,
    onset_frames: int = 1,
):
    """
    -> (frame_roll, onset_roll), her ikisi de şekil (n_frames, n_pitches), float32.
    onset_frames: onset işaretinin kaç kare 'kalın' olacağı (model için biraz
    genişletmek eğitimi kolaylaştırır).
    """
    lo, hi = midi_range(instrument)
    n_pitches = hi - lo + 1
    frame_roll = np.zeros((n_frames, n_pitches), dtype=np.float32)
    onset_roll = np.zeros((n_frames, n_pitches), dtype=np.float32)

    for note in transcription.notes:
        p = note.pitch - lo
        if not (0 <= p < n_pitches):
            continue  # enstrüman aralığı dışıysa atla
        start = int(round(note.onset * frame_rate))
        end = int(round(note.offset * frame_rate))
        start = max(0, min(start, n_frames - 1))
        end = max(start + 1, min(end, n_frames))

        frame_roll[start:end, p] = 1.0
        onset_roll[start:min(start + onset_frames, n_frames), p] = 1.0

    return frame_roll, onset_roll


def n_pitches(instrument: Instrument = STANDARD_6) -> int:
    lo, hi = midi_range(instrument)
    return hi - lo + 1
