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

_NOTE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def parse_tuning(text: str, num_frets: int = 24) -> Instrument:
    """
    Katman 6 Adım 3 — kullanıcı akortu (pesten tize): "DADGBE" (6 telli; oktav standart açık telin en yakınından)
    ya da "D2 A2 D3 G3 B3 E4" (oktavlı; # / b işaretli). Bilinen bir preset'le aynıysa o preset döner (ör. DROP_D_6).
    """
    t = text.strip()
    if t.islower():                                          # "dadgbe": küçük b burada nota, bemol değil
        t = t.upper()
    toks = t.split()
    if " " not in t:                                         # "DADGBE": harf + isteğe bağlı # / b
        toks, i = [], 0
        while i < len(t):
            j = i + 1 + (i + 1 < len(t) and t[i + 1] in "#b")
            toks.append(t[i:j]); i = j
    out = []
    for k, tok in enumerate(toks):
        name = tok[0].upper()
        if name not in _NOTE:
            raise ValueError(f"akort anlasilmadi: {text!r} ({tok!r})")
        acc = tok[1:2] if tok[1:2] in ("#", "b") else ""
        pc = _NOTE[name] + {"#": 1, "b": -1, "": 0}[acc]
        octave = tok[1 + len(acc):]
        if octave:
            out.append(12 * (int(octave) + 1) + pc)
        else:
            if len(toks) != STANDARD_6.num_strings:
                raise ValueError(f"oktavsiz akort yalniz 6 tel icin: {text!r}")
            ref = STANDARD_6.tuning[k]                       # standart açık tele en yakın oktav (± 6 yarım ses)
            out.append(ref + (pc - ref + 6) % 12 - 6)
    tuning = tuple(out)
    for p in PRESETS.values():
        if p.tuning == tuning and p.num_frets == num_frets:
            return p
    return Instrument("tuning_" + "_".join(map(str, tuning)), tuning, num_frets)


def with_capo(instr: Instrument, capo: int) -> Instrument:
    """Kapo k: açık teller k yarım ses tiz, perdeler kapodan sayılır (TAB / MusicXML <capo> geleneği). k = 0 -> aynı nesne."""
    if capo == 0:
        return instr
    if not 0 < capo < instr.num_frets:
        raise ValueError(f"kapo 0..{instr.num_frets - 1} olmali: {capo}")
    return Instrument(f"{instr.name}_capo{capo}", tuple(p + capo for p in instr.tuning), instr.num_frets - capo)
