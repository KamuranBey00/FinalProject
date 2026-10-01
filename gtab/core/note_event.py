"""
Katman 0 — Ara veri şeması (PROJENİN EN KRİTİK KARARI).

Bütün pipeline şu tek temsilin etrafında döner:
    ses  ->  [NoteEvent, NoteEvent, ...]  ->  tab

Bu şemayı bir kez zengin tutarsak, her yeni katman (string tespiti, bend tespiti,
ritim, render) sadece NoteEvent'in bir alanını doldurur ya da okur -- hiçbir şeyi
kırmadan. O yüzden bugün henüz kullanmayacağımız alanları (string, fret, technique)
şimdiden koyuyoruz; Katman 2'de boş bırakırız, Katman 3-4'te doldururuz.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Technique(Enum):
    """Gitar teknikleri. Katman 4'te tespit edeceğiz; şimdilik hepsi NORMAL."""
    NORMAL = "normal"
    BEND = "bend"                 # ses sürekli yükselir, yeni vuruş yok
    BEND_RELEASE = "bend_release" # bend sonrası geri iner
    SLIDE_UP = "slide_up"         # parmak tel üstünde tize kayar
    SLIDE_DOWN = "slide_down"
    HAMMER_ON = "hammer_on"       # vuruşsuz, tize legato
    PULL_OFF = "pull_off"         # vuruşsuz, pese legato
    VIBRATO = "vibrato"           # merkez ses etrafında salınım
    PALM_MUTE = "palm_mute"       # timbral: kısılmış/boğuk
    HARMONIC = "harmonic"
    DEAD_NOTE = "dead_note"
    TAP = "tap"


@dataclass
class NoteEvent:
    onset: float                              # başlangıç, saniye
    offset: float                             # bitiş, saniye
    pitch: int                                # BAŞLANGIÇ MIDI notası

    # --- Katman 3'te doldurulacak (string/fret ataması) ---
    string: Optional[int] = None              # 0 = en pes tel; None = bilinmiyor
    fret: Optional[int] = None

    # --- Katman 4'te doldurulacak (teknik tespiti) ---
    technique: Technique = Technique.NORMAL
    # Tekniğe özel parametreler, örn: {"bend_semitones": 2.0} ya da {"target_fret": 7}
    technique_params: Dict = field(default_factory=dict)

    # --- Genel ---
    confidence: float = 1.0                   # modelin bu nota için güveni

    @property
    def duration(self) -> float:
        return self.offset - self.onset

    def has_position(self) -> bool:
        return self.string is not None and self.fret is not None


@dataclass
class Transcription:
    """Bir parçanın tam transkripsiyonu. Pipeline'ın her ucunda bu dolaşır."""
    notes: List[NoteEvent]
    instrument_name: str = "standard_6"
    tempo_bpm: Optional[float] = None         # Katman 5'te doldurulur
    beats: Optional[List[float]] = None       # vuruş zamanları (saniye), Katman 5

    def sort(self) -> "Transcription":
        self.notes.sort(key=lambda n: (n.onset, n.pitch))
        return self

    def duration(self) -> float:
        return max((n.offset for n in self.notes), default=0.0)
