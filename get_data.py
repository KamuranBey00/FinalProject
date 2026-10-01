"""
Katman 0.5 — Veri edinme.

Elinde hiç veri yok, dedin. Sıfırdan gitar kaydı toplamak/etiketlemek aylar sürer.
Neyse ki bu iş için ALTIN STANDART veri seti hazır: GuitarSet.
- ~360 kayıt, gerçek gitar
- Hexafonik pickup ile kaydedilmiş: HER TEL AYRI KANAL
  => yani "hangi tel çalındı" bilgisi ETİKETLİ geliyor (Katman 3'ün altını doldurur)
- Nota başlangıç/bitiş + perde bilgisi JAMS formatında hazır

Bunu 'mirdata' kütüphanesi indirir ve okur; JAMS ayrıştırmayı senin yerine yapar.

Çalıştırma:
    pip install mirdata librosa
    python get_data.py

Not: mirdata'nın alan adları sürüme göre küçük farklar gösterebilir.
Bir track objesi elde ettikten sonra dir(track) ile alanları teyit et.
"""

import mirdata
import librosa  # yalnizca 'hz' yedek yolunda kullanilir

from gtab.note_event import NoteEvent, Transcription
from gtab.config import GUITARSET_DATA_HOME


def notedata_to_midi(pitches, unit: str):
    """Birime bakarak MIDI'ye getirir. GuitarSet zaten 'midi' verir."""
    if unit == "midi":
        return pitches
    if unit == "hz":
        return librosa.hz_to_midi(pitches)
    raise ValueError(f"Beklenmeyen pitch birimi: {unit!r}")


def download_guitarset():
    gset = mirdata.initialize("guitarset", data_home=GUITARSET_DATA_HOME)
    gset.download()          # ilk sefer birkaç GB indirir
    gset.validate()          # dosyalar tam mı diye kontrol
    return gset


def track_to_transcription(track) -> Transcription:
    """
    Bir GuitarSet track'ini bizim NoteEvent şemamıza çevirir.
    Katman 0 için sadece onset/offset/pitch dolduruyoruz; string/fret ve teknik
    sonraki katmanların işi (GuitarSet'te string bilgisi de var, Katman 3'te ekleriz).
    """
    nd = track.notes_all                 # mirdata NoteData: .intervals, .pitches, .pitch_unit
    midi_vals = notedata_to_midi(nd.pitches, nd.pitch_unit)
    events = []
    for (start, end), m in zip(nd.intervals, midi_vals):
        events.append(NoteEvent(onset=float(start), offset=float(end), pitch=int(round(m))))
    return Transcription(notes=events, instrument_name="standard_6").sort()


if __name__ == "__main__":
    gset = download_guitarset()
    tracks = gset.load_tracks()
    print(f"Toplam {len(tracks)} kayıt indirildi.")

    # İlk kaydı şemamıza çevirip göz atalım
    first_id = next(iter(tracks))
    tr = track_to_transcription(tracks[first_id])
    print(f"'{first_id}' -> {len(tr.notes)} nota, süre {tr.duration():.1f} sn")
    for n in tr.notes[:8]:
        print(f"  {n.onset:6.2f}-{n.offset:6.2f} sn  MIDI {n.pitch}")
