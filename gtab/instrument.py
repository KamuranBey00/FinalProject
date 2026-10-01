"""
Katman 0 — Enstrüman soyutlaması.

"İleride farklı gitar modelleri için de geliştirmek istiyorum" dedin.
İşte o future-proofing tam burada: hiçbir yerde "6 tel", "standart akort" ya da
"24 perde" SABİT KODLANMAYACAK. Bunların hepsi bir Instrument nesnesinin
parametresi. 7 telli gitar, drop-D, bas, hatta ukulele -> sadece yeni bir preset.

Bu sınıf ayrıca tab'ın en zor problemini de çözer:
    pitch_to_positions()  ->  "bu ses telin/perdenin NERELERİNDE çıkar?"
Bir E4 notası birden çok yerde çalınabilir; string/fret ataması bu listeyi kullanır.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Instrument:
    name: str
    # Her açık telin MIDI nota numarası. İNDEKS 0 = EN PES (en kalın) TEL.
    # Bu konvansiyonu her yerde koru; yoksa tab ters çıkar.
    tuning: Tuple[int, ...]
    num_frets: int = 24

    @property
    def num_strings(self) -> int:
        return len(self.tuning)

    def position_to_pitch(self, string: int, fret: int) -> int:
        """(tel, perde) -> MIDI nota. Basit ama her yerde lazım."""
        return self.tuning[string] + fret

    def pitch_to_positions(self, pitch: int) -> List[Tuple[int, int]]:
        """
        Bir MIDI notasının bu enstrümanda çalınabileceği TÜM (tel, perde)
        kombinasyonları. Tab atamasının kalbi burasıdır.
        """
        positions = []
        for string_idx, open_pitch in enumerate(self.tuning):
            fret = pitch - open_pitch
            if 0 <= fret <= self.num_frets:
                positions.append((string_idx, fret))
        return positions

    def pitch_range(self) -> Tuple[int, int]:
        """Bu enstrümanın çalabildiği en pes ve en tiz MIDI nota."""
        lowest = min(self.tuning)
        highest = max(self.tuning) + self.num_frets
        return lowest, highest


# --- Hazır presetler ---
# MIDI referansı: C4 (orta do) = 60, E2 = 40.
STANDARD_6   = Instrument("standard_6",   (40, 45, 50, 55, 59, 64), 24)          # E2 A2 D3 G3 B3 E4
DROP_D_6     = Instrument("drop_d_6",      (38, 45, 50, 55, 59, 64), 24)          # D2 A2 D3 G3 B3 E4
SEVEN_STRING = Instrument("seven_string",  (35, 40, 45, 50, 55, 59, 64), 24)      # B1 E2 A2 D3 G3 B3 E4
BASS_4       = Instrument("bass_4",        (28, 33, 38, 43), 24)                  # E1 A1 D2 G2

PRESETS = {i.name: i for i in (STANDARD_6, DROP_D_6, SEVEN_STRING, BASS_4)}
