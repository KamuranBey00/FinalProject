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

**Ses ön-işleme** ([gtab/config.py](gtab/config.py), [gtab/data/features.py](gtab/data/features.py))
- Örnekleme hızı: 22050 Hz, hop uzunluğu: 512 (~23 ms kare çözünürlüğü, ~43 kare/sn)
- Girdi temsili: **Constant-Q Transform (log-CQT)** — C1'den (32.7 Hz) başlayıp 8
  oktav (192 bin), oktav başına 24 bin (yarım ses başına 2 bin — bend gibi
  mikroton geçişlerini yakalamak için)

**Ara veri şeması** ([gtab/core/note_event.py](gtab/core/note_event.py)) — `NoteEvent`
(onset, offset, pitch, string, fret, technique) ve `Transcription`; pipeline'ın
tüm katmanlarının paylaştığı tek kaynak.

**Modeller** ([gtab/models/nets.py](gtab/models/nets.py))
| Model | Girdi | Çıktı | Kullanım |
|---|---|---|---|
| `PitchCNN` | CQT penceresi (context=9 kare) | çok-etiketli perde logit'i | Katman 2 — sadece perde |
| `TabCNN` | CQT penceresi (context=9 kare) | (6 tel × 26 sınıf) logit, tel başına softmax | Katman 3 — gerçek tel/fret |
| `TabCRNN` | CQT sekansı (CNN + 2 katman BiLSTM, çift yönlü) | (L × 6 tel × 26 sınıf) | Katman 3.5 — zamansal tutarlılık, tel-sıçramasını azaltmak |

Ortak CNN gövdesi: `Conv2d(1→32) → BN → ReLU → Conv2d(32→64) → BN → ReLU →
MaxPool2d(2) → Dropout(0.25)`. TabCRNN'de zaman ekseni korunur (sadece frekans
ekseni havuzlanır), ardından `Linear → BiLSTM(hidden=128, 2 katman) → Linear`
ile zamansal bağlam eklenir.

**Eğitim** ([scripts/train/](scripts/train/): train_pitch, train_tab, train_crnn, train_domain)
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
| 3.5 — Zamansal | TabCRNN | tab F1 (argmax + marj) | 0.634 |
| 3.6 — Tel ataması | TabCRNN + nota başına tek pozisyon | tab F1 | 0.664 (el yapımı / öğrenilen önsel katkı vermedi) |
| 3.7 — Veri genişletme | TabCRNN + GuitarSet comp (`tabcrnn_comp.pt`) | tab F1 / oracle tel doğruluğu | **0.667** / 0.762 |
| 3.8 — Klasik gitar | TabCRNN + GAPS perde ince ayarı (`tabcrnn_gaps.pt`) | GuitarSet tab F1 / GAPS nota F1 | 0.697 / 0.360 (önce 0.255) — [README9](docs/devlog/README9.md) |
| 3.9 — Tel/perde sesi | + onset kafası + harmonik istifleme (`tabcrnn_onset_h.pt`) | GuitarSet tab F1 / oracle tel / akorlu nota F1 | **0.739** / 0.850 / 0.724 — [README10](docs/devlog/README10.md) |

Detaylı gerekçeler ve ara deneyler için [docs/devlog/](docs/devlog/) klasörüne bakın.
Yeni katkıda bulunanlar için dosya rehberi ve çalışma kuralları: [CONTRIBUTING.md](CONTRIBUTING.md).

## Repo yapısı

```
gtab/                       çekirdek kütüphane (yalnızca import edilir)
  config.py, paths.py       ses/CQT sabitleri, proje yolları
  core/                     instrument (akort, tel/perde), note_event (ortak nota şeması)
  data/                     features (CQT), labels, tab_labels, torch_dataset,
                            guitarset / gaps / synthtab (veri seti okuyucuları)
  models/                   nets (PitchCNN, TabCNN, TabCRNN), losses, inference
  decoding/                 decode (çözümleme, ASCII tab), viterbi, transitions
  evaluation/               metrics
scripts/                    çalıştırılan dosyalar (python -m scripts.<klasör>.<ad>)
  data/                     get_guitarset, build_guitarset, build_tab_labels, build_comp,
                            get_gaps, build_gaps, build_synthtab
  train/                    train_pitch (K2), train_tab (K3), train_crnn (K3.5–3.7), train_domain (K3.8)
  eval/                     sweep_threshold (K2), eval_viterbi (K3.6), eval_hmm (K3.6b), eval_pitch (K3.8)
checkpoints/                *.pt model ağırlıkları + transitions.npz
data/cache/                 önbellek (.npz): train, val, train_comp, val_comp, gaps_*, synthtab_*  -- git'e girmez
data/raw/                   ham veri (GAPS, SynthTab zip'leri)                                  -- git'e girmez
docs/devlog/                katman katman geliştirme günlüğü
inputs/                     proje sunumu ve katman karar belgeleri
```

Eski düz yapıdan yeni yollara geçiş (devlog'lardaki eski komutlar için):
`train.py → scripts/train/train_pitch.py`, `get_data.py → scripts/data/get_guitarset.py`,
`build_dataset.py → scripts/data/build_guitarset.py`, diğer script'ler aynı adla
`scripts/{data,train,eval}/` altında; `*.pt` dosyaları `checkpoints/` altında.

## Kurulum

```bash
pip install -r requirements.txt
```

Tüm komutlar **proje kökünden** `python -m` ile çalıştırılır (kurulum gerekmez).
`--ckpt` gibi argümanlara çıplak dosya adı verilirse `checkpoints/` altında aranır.

## Çalıştırma

```bash
# Veri (yalnızca ham veriden yeniden başlanıyorsa)
python -m scripts.data.get_guitarset
python -m scripts.data.build_guitarset
python -m scripts.data.build_tab_labels
python -m scripts.data.build_comp                 # Katman 3.7: comp kayıtları

# Katman 2 — perde modeli
python -m scripts.train.train_pitch --epochs 15
python -m scripts.eval.sweep_threshold

# Katman 3 — TabCNN
python -m scripts.train.train_tab --epochs 40
python -m scripts.train.train_tab --sweep

# Katman 3.5–3.7 — TabCRNN (+comp)
python -m scripts.train.train_crnn --epochs 30 --splits train,train_comp --ckpt tabcrnn_comp.pt
python -m scripts.eval.eval_hmm --model crnn --ckpt tabcrnn_comp.pt
python -m scripts.train.train_crnn --demo data/cache/val/05_Rock1-130-A_solo.npz --ckpt tabcrnn_comp.pt --margin 0.8

# Katman 3.8 — klasik gitar (ayrıntı: docs/devlog/README9.md)
python -m scripts.data.get_gaps --splits train test
python -m scripts.data.build_gaps --disjoint
python -m scripts.eval.eval_pitch --ckpt tabcrnn_comp.pt --splits gaps_test val val_comp
```

## Veri seti ve eğitilmiş modeller

**Veri seti:** [GuitarSet](https://guitarset.weebly.com/) — ~360 kayıt, hexafonik
pickup ile kaydedilmiş gerçek gitar (her tel ayrı kanal, bu sayede "hangi tel
çalındı" bilgisi etiketli), nota onset/offset + perde JAMS formatında. `mirdata`
kütüphanesi ile indirilir ([scripts/data/get_guitarset.py](scripts/data/get_guitarset.py)).

- Ham ses dosyaları **bu repoya dahil değildir** (GuitarSet birkaç GB ve ayrı bir
  lisansla dağıtılan genel bir veri seti olduğu için); `python -m scripts.data.get_guitarset`
  çalıştırıldığında `GUITARSET_DATA_HOME` ortam değişkeniyle belirtilen yola
  otomatik indirilir.
- **Veri setleri ve önbellek repoda yok** (`data/` `.gitignore`'da). Kaynaklar:
  GuitarSet (mirdata), GAPS v1.1 (https://huggingface.co/datasets/xavriley/GAPS),
  SynthTab (https://github.com/yongyizang/SynthTab). Önbellek, `scripts/data/`
  altındaki script'lerle yeniden üretilir (komutlar yukarıda ve devlog'larda).
- **[checkpoints/](checkpoints/)** — ilgili katmanlarda eğitilmiş, doğrudan yüklenebilir model
  ağırlıkları (`pitchcnn.pt`, `tabcnn.pt`, `tabcrnn.pt`, `tabcrnn_comp.pt` = güncel en iyi, ...).

## Geliştirme günlüğü (katman katman)

Projenin katman katman nasıl geliştiği, her katmanda alınan kararlar ve
gerekçeleri [docs/devlog/](docs/devlog/) klasöründe korunmaktadır:

- [docs/devlog/README.md](docs/devlog/README.md) — Katman 0-1 (temel + özellik/veri yükleyici)
- [docs/devlog/README1.md](docs/devlog/README1.md) — Katman 1 tamam
- [docs/devlog/README2.md](docs/devlog/README2.md) — Katman 2 (monofonik uçtan uca)
- [docs/devlog/README3.md](docs/devlog/README3.md) — Katman 3 (gerçek tel/fret)
- [docs/devlog/README4.md](docs/devlog/README4.md) — Katman 3 konsolidasyon + denetim düzeltmeleri
- [docs/devlog/README5.md](docs/devlog/README5.md) — Katman 3.5 (CRNN)
- [docs/devlog/README6.md](docs/devlog/README6.md) — Katman 3.6 (Viterbi tel ataması, el yapımı maliyet)
- [docs/devlog/README7.md](docs/devlog/README7.md) — Katman 3.6b (öğrenilen geçiş modeli + teşhis)
- [docs/devlog/README8.md](docs/devlog/README8.md) — Katman 3.7 (veri genişletme: comp + çoğaltma)
- [docs/devlog/README9.md](docs/devlog/README9.md) — Katman 3.8 (klasik gitar: GAPS + SynthTab yol haritası)
- [docs/devlog/README10.md](docs/devlog/README10.md) — Katman 3.9 (tel/perde sesini öğrenmek: hata analizi, onset, harmonik istifleme)
- [docs/devlog/README11.md](docs/devlog/README11.md) — Katman 3.10 (polifonide perde: hayalet notalar ve kaçan perdeler)

## Yol haritası

- [x] Katman 0 — Temel (config, instrument, note_event, veri edinme)
- [x] Katman 1 — Özellik çıkarımı + veri yükleyici
- [x] Katman 2 — Monofonik uçtan uca (PitchCNN)
- [x] Katman 3 — Gerçek tel/fret (TabCNN)
- [x] Katman 3.5 — Zamansal model (TabCRNN)
- [x] Katman 3.6–3.7 — Tel ataması (Viterbi / öğrenilen önsel) + veri genişletme (comp)
- [ ] Katman 3.8 — Klasik gitar alan uyarlaması: GAPS + SynthTab nylon
      (sunum Phase 3–4; bkz. [docs/devlog/README9.md](docs/devlog/README9.md))
- [x] Katman 3.9 — Tel/perde sesini öğrenmek: onset + harmonik istifleme
      (bkz. [docs/devlog/README10.md](docs/devlog/README10.md))
- [ ] Katman 3.10 — Polifonide perde: hayalet notalar ve kaçan perdeler
      (bkz. [docs/devlog/README11.md](docs/devlog/README11.md)) — fingerstyle (GAPS) ölçütü sağlandı,
      tab F1 0.760 (`tabcrnn_rep_off.pt`); akor ölçütü eksik, kapanış kararı bekleniyor
- [ ] Katman 4 — Teknikler: sürekli F0 eğrisi + onset zarfından bend, slide,
      hammer-on/pull-off, vibrato tespiti
- [ ] Katman 5 — Ritim + render: tempo/beat takibi, kuantalama, AlphaTab ile
      bend/slide sembollü gerçek tab görseli
- [ ] Katman 6 — Zorlaştırma: polifoni (akorlar), distortion dayanıklılığı,
      farklı gitar/akort desteği
