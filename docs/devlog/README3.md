# Gitar Tab Transkripsiyon — Katmanlı Yol Haritası

Ses/video → NoteEvent listesi → tab. Her katman bir öncekinin üstüne biner.

## Katmanlar

- [x] **Katman 0 — Temel.** config, instrument, note_event, get_data (GuitarSet).
- [x] **Katman 1 — Özellik + veri.** features (log-CQT), labels, build_dataset (solo filtresi, oyuncuya göre bölme, .npz önbellek).
- [x] **Katman 2 — Monofonik uçtan uca.** PitchCNN (CQT→perde), decode, train; kare F1 **0.853** (eşik 0.90, oyuncu 05 val).
- [x] **Katman 3 — Gerçek string/fret.**
  - `gtab/tab_labels.py` — tel-bazlı fret sınıfı etiketleri
  - `build_tab_labels.py` — önbelleğe 'tab' etiketi ekler (CQT'yi yeniden hesaplamaz)
  - `gtab/torch_dataset.py` — GuitarSetTab
  - `gtab/model.py` — TabCNN (6 tel × sınıf, tel başına softmax)
  - `gtab/decode.py` — tab_to_transcription (gerçek tel/fret)
  - `train_tab.py` — eğitim + pitch F1 & tab F1

- [ ] **Katman 4 — Teknikler.** Sürekli F0 + onset → bend, slide, hammer/pull, vibrato.
- [ ] **Katman 5 — Ritim + render.** Tempo/beat, kuantalama, AlphaTab görsel tab.
- [ ] **Katman 6 — Zorlaştırma.** Polifoni (akor), distortion, farklı gitarlar.

## Katman 3 çalıştırma
```bash
python build_tab_labels.py            # onbellege 'tab' ekler (hizli)
python train_tab.py --epochs 20
python train_tab.py --demo data/cache/val/05_Rock1-130-A_solo.npz
```

## İki metrik
- **pitch F1**: tab→perde; Katman 2 (0.853) ile karşılaştırılabilir.
- **tab F1**: (tel, fret) token; daha katı. tab F1 < pitch F1; fark = tel atama kalitesi.
