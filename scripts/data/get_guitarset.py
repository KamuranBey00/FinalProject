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
    pip install -r requirements.txt
    python -m scripts.data.get_guitarset

Not: mirdata'nın alan adları sürüme göre küçük farklar gösterebilir.
Bir track objesi elde ettikten sonra dir(track) ile alanları teyit et.
"""

import mirdata

from gtab.config import GUITARSET_DATA_HOME
from gtab.data.guitarset import track_to_transcription


def download_guitarset():
    gset = mirdata.initialize("guitarset", data_home=GUITARSET_DATA_HOME)
    gset.download()          # ilk sefer birkaç GB indirir
    gset.validate()          # dosyalar tam mı diye kontrol
    return gset




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
