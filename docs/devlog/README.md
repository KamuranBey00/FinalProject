# Gitar Tab Transkripsiyon — Katmanlı Yol Haritası

Ses/video → NoteEvent listesi → tab. Her katman bir öncekinin üstüne biner,
hiçbir şeyi kırmadan. Şu an **Katman 0** bitti.

## Katmanlar

- [x] **Katman 0 — Temel (BİTTİ).** Değiştirmesi pahalı 4 karar sabitlendi:
  - `gtab/config.py` — ses/CQT sabitleri (bir kez, ömür boyu)
  - `gtab/instrument.py` — enstrüman soyutlaması (farklı gitarlar için future-proof)
  - `gtab/note_event.py` — ara veri şeması (pipeline'ın belkemiği)
  - `get_data.py` — GuitarSet edinme (altın standart veri)

- [ ] **Katman 1 — Özellik + veri yükleyici.** Ses → CQT; GuitarSet'i şemaya
  bağlayan tam data loader; eğitim/doğrulama ayrımı.

- [ ] **Katman 2 — Monofonik uçtan uca.** CREPE ile pitch → NoteEvent → naif
  tab (en düşük pozisyon). İlk çalışan çıktı, kalite düşük olsa da.

- [ ] **Katman 3 — Gerçek string/fret.** GuitarSet'in tel etiketleriyle
  TabCNN/CRNN tarzı model. Artık pozisyon tahmin değil, gerçek.

- [ ] **Katman 4 — Teknikler.** Sürekli F0 eğrisi + onset zarfı → önce BEND,
  sonra slide, hammer/pull, vibrato. Senin asıl hedefin.

- [ ] **Katman 5 — Ritim + render.** Tempo/beat takibi, kuantalama; AlphaTab
  ile bend/slide sembollü gerçek tab görseli.

- [ ] **Katman 6 — Zorlaştırma.** Polifoni (akorlar), distortion dayanıklılığı,
  farklı gitarlar/akortlar.

## Kurulum
```bash
pip install mirdata librosa numpy
python get_data.py     # GuitarSet indirir (birkaç GB, ilk sefer)
```

## Tasarım ilkesi
Özellikleri değil, ARAYÜZLERİ önden sağlamlaştır. NoteEvent bugün yarı boş
(string/fret/technique = None); her katman bir alanı doldurur.
