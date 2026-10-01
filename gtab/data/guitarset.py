"""
GuitarSet yardımcıları (mirdata üstünden).

Track id örneği: "05_BN1-129-Eb_solo" -> oyuncu "05", tür "solo" | "comp".
"""

import librosa

from gtab.core.note_event import NoteEvent, Transcription


def is_solo(track_id: str) -> bool:
    return track_id.split("_")[-1] == "solo"


def is_comp(track_id: str) -> bool:
    return track_id.split("_")[-1] == "comp"


def player_of(track_id: str) -> str:
    return track_id.split("_")[0]


def notedata_to_midi(pitches, unit: str):
    """
    mirdata NoteData.pitches'i MIDI'ye getirir -- BİRİME BAKARAK.
    GuitarSet (note_midi namespace) zaten 'midi' verir; dokunmayız.
    Bilinmeyen birimde sessizce yanlış yapmaktansa HATA fırlatır.
    """
    if unit == "midi":
        return pitches
    if unit == "hz":
        return librosa.hz_to_midi(pitches)
    raise ValueError(f"Beklenmeyen pitch birimi: {unit!r}")


def track_to_transcription(track) -> Transcription:
    """GuitarSet track -> NoteEvent listesi (onset/offset/pitch)."""
    nd = track.notes_all      # mirdata NoteData: .intervals (Nx2 sn), .pitches, .pitch_unit
    midi_vals = notedata_to_midi(nd.pitches, nd.pitch_unit)
    events = [
        NoteEvent(onset=float(s), offset=float(e), pitch=int(round(m)))  # kesirli MIDI -> en yakın yarım ses
        for (s, e), m in zip(nd.intervals, midi_vals)
    ]
    return Transcription(notes=events, instrument_name="standard_6").sort()
