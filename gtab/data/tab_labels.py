"""
Katman 3 — Tel-bazlı etiketler: GuitarSet'in her-tel notaları -> (T, 6) fret sınıfı.

Katman 2'de hedef "hangi perde" idi. Şimdi hedef "hangi telde hangi fret".
GuitarSet bunu bize hazır veriyor (track.notes = tel adına göre sözlük),
ve tel sırası bizim tuning sıramızla aynı (index 0 = Low E).

Sınıf şeması (her tel için ayrı):
    0            = SESSİZ (o telde nota yok)
    1..num_frets+1 = fret 0..num_frets   (yani sınıf = fret + 1)
Bu yüzden tel başına sınıf sayısı = num_frets + 2.

"Her telde aynı anda en fazla bir nota" fiziksel gerçeği bu şemaya gömülü:
tel başına TEK bir sınıf (softmax) -> Katman 6'da akorlar bedavaya gelir.
"""

import numpy as np

from gtab.core.instrument import Instrument, STANDARD_6
from gtab.config import FRAME_RATE

# mirdata guitarset tel sırası (index 0 = Low E), bizim tuning ile ayni.
GUITARSET_ORDER = ["E", "A", "D", "G", "B", "e"]


def _to_midi(pitches, unit):
    if unit == "midi":
        return pitches
    if unit == "hz":
        import librosa
        return librosa.hz_to_midi(pitches)
    raise ValueError(f"Beklenmeyen pitch birimi: {unit!r}")


def n_tab_classes(instrument: Instrument = STANDARD_6) -> int:
    return instrument.num_frets + 2      # sessiz + fret 0..num_frets


def build_tab_targets(notes_dict, n_frames: int,
                      instrument: Instrument = STANDARD_6,
                      frame_rate: float = FRAME_RATE) -> np.ndarray:
    """
    notes_dict: {tel_adi: mirdata NoteData}. -> (n_frames, num_strings) int sınıf dizisi.
    """
    S = instrument.num_strings
    tab = np.zeros((n_frames, S), dtype=np.int64)     # 0 = sessiz

    skipped = 0
    for s in range(S):
        nd = notes_dict[GUITARSET_ORDER[s]]
        if nd is None:              # bu telde bu kayitta hic nota calinmamis
            continue
        midi_vals = _to_midi(nd.pitches, nd.pitch_unit)
        for (start, end), m in zip(nd.intervals, midi_vals):
            fret = int(round(m)) - instrument.tuning[s]
            if fret < 0 or fret > instrument.num_frets:
                skipped += 1
                continue
            a = max(0, int(round(start * frame_rate)))
            b = min(n_frames, int(round(end * frame_rate)))
            if b > a:
                tab[a:b, s] = fret + 1        # sınıf = fret + 1
    return tab


def tab_class_to_fret(c: int):
    """Sınıf -> fret (sessizse None)."""
    return None if c == 0 else c - 1
