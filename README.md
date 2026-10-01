# Gitar Tab Transkripsiyonu

Bu proje, bir **bitirme projesi (final project)** kapsamında geliştirilmektedir.
Amaç: elektro gitar ses kaydını otomatik olarak **tab (tablature)** notasyonuna
çeviren uçtan uca bir derin öğrenme sistemi kurmak — hangi an, hangi telde,
hangi perdenin çalındığını ses dalgasından tahmin etmek.

Sistem, ses → CQT özellik çıkarımı → CNN/CRNN tabanlı tel-fret sınıflandırması →
ASCII tab render şeklinde katmanlı bir pipeline olarak inşa edildi; her katman
bir öncekini kırmadan üzerine eklendi.

## İçindekiler
- [Pipeline](#pipeline)
- [Mimari ve teknik detaylar](#mimari-ve-teknik-detaylar)
- [Sonuçlar](#sonuçlar)
- [Repo yapısı](#repo-yapısı)
- [Kurulum](#kurulum)
- [Çalıştırma](#çalıştırma)
- [Veri seti ve eğitilmiş modeller](#veri-seti-ve-eğitilmiş-modeller)
- [Geliştirme günlüğü (katman katman)](#geliştirme-günlüğü-katman-katman)
- [Yol haritası](#yol-haritası)

## Pipeline

```
ses (.wav)
   │  librosa
   ▼
log-CQT  (192 bin, oktav başına 24 bin, hop=512, sr=22050)
   │  CNN gövdesi (iki Conv2d+BN+ReLU blok + MaxPool + Dropout)
   ▼
┌─────────────────────────────┬─────────────────────────────────┐
│ PitchCNN                    │ TabCNN / TabCRNN                │
│ çok-etiketli perde logit'i  │ 6 tel × 26 sınıf (fret 0-24 +    │
│ (hangi MIDI notaları çalıyor)│ sessizlik), tel başına softmax  │
└─────────────────────────────┴─────────────────────────────────┘
   │
   ▼
decode: frame → nota (onset/offset birleştirme) → tel/fret ataması
   │  zamansal düzeltme (mode-filter smoothing) + sessizlik-marjı taraması
   ▼
ASCII tab çıktısı
```

## Mimari ve teknik detaylar

**Ses ön-işleme** ([gtab/config.py](gtab/config.py), [gtab/features.py](gtab/features.py))
- Örnekleme hızı: 22050 Hz, hop uzunluğu: 512 (~23 ms kare çözünürlüğü, ~43 kare/sn)
- Girdi temsili: **Constant-Q Transform (log-CQT)** — C1'den (32.7 Hz) başlayıp 8
  oktav (192 bin), oktav başına 24 bin (yarım ses başına 2 bin — bend gibi
  mikroton geçişlerini yakalamak için)

**Ara veri şeması** ([gtab/note_event.py](gtab/note_event.py)) — `NoteEvent`
(onset, offset, pitch, string, fret, technique) ve `Transcription`; pipeline'ın
tüm katmanlarının paylaştığı tek kaynak.

**Modeller** ([gtab/model.py](gtab/model.py))
| Model | Girdi | Çıktı | Kullanım |
|---|---|---|---|
| `PitchCNN` | CQT penceresi (context=9 kare) | çok-etiketli perde logit'i | Katman 2 — sadece perde |
| `TabCNN` | CQT penceresi (context=9 kare) | (6 tel × 26 sınıf) logit, tel başına softmax | Katman 3 — gerçek tel/fret |
| `TabCRNN` | CQT sekansı (CNN + 2 katman BiLSTM, çift yönlü) | (L × 6 tel × 26 sınıf) | Katman 3.5 — zamansal tutarlılık, tel-sıçramasını azaltmak |

Ortak CNN gövdesi: `Conv2d(1→32) → BN → ReLU → Conv2d(32→64) → BN → ReLU →
MaxPool2d(2) → Dropout(0.25)`. TabCRNN'de zaman ekseni korunur (sadece frekans
ekseni havuzlanır), ardından `Linear → BiLSTM(hidden=128, 2 katman) → Linear`
ile zamansal bağlam eklenir.

**Eğitim** ([train.py](train.py), [train_tab.py](train_tab.py), [train_crnn.py](train_crnn.py))
- Adam optimizer, cosine learning-rate decay, sabit seed (tekrarlanabilirlik)
- Sınıf dengesizliği için ayarlanabilir class-weighting (`inv`/`sqrt`, cap'li)
- **Sessizlik-marjı taraması** (`--sweep`): retrain gerektirmeden precision/recall
  dengesini tek eğitilmiş modelden ayarlamanın yolu
- **Zamansal düzeltme** (mode-filter smoothing): ardışık kareler arasında
  hayalet-nota (tek karelik yanlış tahmin) temizliği
- İki ayrı metrik: **pitch F1** (sadece hangi nota çalındı) ve daha katı
  **tab F1** (tam olarak hangi tel + hangi fret) — aradaki fark tel atama
  kalitesini gösterir

## Sonuçlar

| Katman | Model | Metrik | Sonuç |
|---|---|---|---|
| 2 — Monofonik perde | PitchCNN | kare-seviye F1 (eşik 0.90, oyuncu 05 val) | **0.853** |
| 3 — Gerçek tel/fret | TabCNN | tab F1 tavanı | **~0.586** (düşük precision, orta-nota tel sıçramasından) |
| 3.5 — Zamansal | TabCRNN | — | mimari BiLSTM ile tel-sıçramasını kaynağında azaltmayı hedefliyor; tam veri/GPU üzerinde nihai sayılar henüz raporlanmadı |

Detaylı gerekçeler ve ara deneyler için [docs/devlog/](docs/devlog/) klasörüne bakın.

## Repo yapısı

```
gtab/                  çekirdek kütüphane (config, features, labels, model, decode, ...)
build_dataset.py       GuitarSet -> CQT + frame/onset roll -> .npz önbellek (data/cache)
build_tab_labels.py    önbelleğe tel/fret etiketi ekler
get_data.py            GuitarSet indirme (mirdata)
train.py               Katman 2 eğitimi (PitchCNN)
train_tab.py           Katman 3 eğitimi (TabCNN) + sweep + demo
train_crnn.py          Katman 3.5 eğitimi (TabCRNN) + sweep + demo
sweep_threshold.py     Katman 2 için eşik taraması
data/cache/             önbelleğe alınmış eğitim/doğrulama özellikleri (.npz)
pitchcnn.pt / tabcnn.pt / tabcrnn.pt   eğitilmiş model ağırlıkları
docs/devlog/            katman katman geliştirme günlüğü (eski README sürümleri)
```

## Kurulum

```bash
pip install -r requirements.txt
```

Gereksinimler: `torch`, `numpy`, `librosa`, `mirdata` (bkz. [requirements.txt](requirements.txt)).

## Çalıştırma

```bash
# (yalnızca ham veriden yeniden başlanıyorsa gerekli — bkz. aşağıdaki veri notu)
python get_data.py
python build_dataset.py
python build_tab_labels.py

# Katman 2 — perde modeli
python train.py --epochs 15
python train.py --demo data/cache/val/05_Rock1-130-A_solo.npz

# Katman 3 — tel/fret modeli
python train_tab.py --epochs 40
python train_tab.py --sweep
python train_tab.py --demo data/cache/val/05_Rock1-130-A_solo.npz --margin 0.10 --smooth 5

# Katman 3.5 — CRNN
python train_crnn.py --epochs 30
python train_crnn.py --demo data/cache/val/05_Rock1-130-A_solo.npz --margin 0.1 --smooth 5
```

## Veri seti ve eğitilmiş modeller

**Veri seti:** [GuitarSet](https://guitarset.weebly.com/) — ~360 kayıt, hexafonik
pickup ile kaydedilmiş gerçek gitar (her tel ayrı kanal, bu sayede "hangi tel
çalındı" bilgisi etiketli), nota onset/offset + perde JAMS formatında. `mirdata`
kütüphanesi ile indirilir ([get_data.py](get_data.py)).

- Ham ses dosyaları **bu repoya dahil değildir** (GuitarSet birkaç GB ve ayrı bir
  lisansla dağıtılan genel bir veri seti olduğu için); `python get_data.py`
  çalıştırıldığında `GUITARSET_DATA_HOME` ortam değişkeniyle belirtilen yola
  otomatik indirilir.
- Bu repoda yer alan **[data/cache/](data/cache/)** klasörü, GuitarSet'ten
  `build_dataset.py` ve `build_tab_labels.py` ile üretilmiş, modelleri eğitmek
  için doğrudan kullanılabilen **önbelleğe alınmış CQT özellikleri + etiketleri**
  içerir (oyuncuya göre train/val ayrımı yapılmış `.npz` dosyaları). Bu sayede
  GuitarSet'i yeniden indirip işlemeden de eğitim/değerlendirme tekrarlanabilir.
- **[pitchcnn.pt](pitchcnn.pt)**, **[tabcnn.pt](tabcnn.pt)**, **[tabcrnn.pt](tabcrnn.pt)**
  — ilgili katmanlarda eğitilmiş, doğrudan yüklenebilir model ağırlıkları.

## Geliştirme günlüğü (katman katman)

Projenin katman katman nasıl geliştiği, her katmanda alınan kararlar ve
gerekçeleri [docs/devlog/](docs/devlog/) klasöründe korunmaktadır:

- [docs/devlog/README.md](docs/devlog/README.md) — Katman 0-1 (temel + özellik/veri yükleyici)
- [docs/devlog/README1.md](docs/devlog/README1.md) — Katman 1 tamam
- [docs/devlog/README2.md](docs/devlog/README2.md) — Katman 2 (monofonik uçtan uca)
- [docs/devlog/README3.md](docs/devlog/README3.md) — Katman 3 (gerçek tel/fret)
- [docs/devlog/README4.md](docs/devlog/README4.md) — Katman 3 konsolidasyon + denetim düzeltmeleri
- [docs/devlog/README5.md](docs/devlog/README5.md) — Katman 3.5 (CRNN)

## Yol haritası

- [x] Katman 0 — Temel (config, instrument, note_event, veri edinme)
- [x] Katman 1 — Özellik çıkarımı + veri yükleyici
- [x] Katman 2 — Monofonik uçtan uca (PitchCNN)
- [x] Katman 3 — Gerçek tel/fret (TabCNN)
- [x] Katman 3.5 — Zamansal model (TabCRNN)
- [ ] Katman 4 — Teknikler: sürekli F0 eğrisi + onset zarfından bend, slide,
      hammer-on/pull-off, vibrato tespiti
- [ ] Katman 5 — Ritim + render: tempo/beat takibi, kuantalama, AlphaTab ile
      bend/slide sembollü gerçek tab görseli
- [ ] Katman 6 — Zorlaştırma: polifoni (akorlar), distortion dayanıklılığı,
      farklı gitar/akort desteği
