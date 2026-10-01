# Katman 3.9 — Tel ve perde sesini daha iyi öğrenmek (polifoniye hazırlık)

## Hedef
Nihai hedef polifoni. Ona geçmeden önce modelin (a) hangi perdenin çaldığını
ve nerede başladığını, (b) hangi telde çalındığını sağlam öğrenmesi. Arkada
boşluk bırakmadan ilerlenir: her adım ölçülür; ciddi iyileşme olmadan bir
sonraki kademeye (Katman 4 / polifoni) geçilmez.

Başlangıç noktası: `tabcrnn_gaps.pt` (Katman 3.8c) — GuitarSet tab F1 0.697,
oracle tel doğruluğu 0.797, GAPS nota F1 0.360.

## Adım 0 — Hata analizi (`scripts/eval/diagnose_strings.py`)
Eğitim yok, sadece ölçüm. `tabcrnn_gaps.pt`, GuitarSet 05 (solo = val,
akor eşliği = val_comp).

### A) Kayıp nereden geliyor? (uçtan uca, greedy, eşik 0.8)
| | solo (val) | polifonik (val_comp) |
|---|---|---|
| gerçek hücre: doğru | 72.3% | 71.5% |
| gerçek hücre: perde bulundu, **tel yanlış** | 17.5% | **5.4%** |
| gerçek hücre: **perde bulunamadı** | 10.2% | **23.1%** |
| yanlış pozitif: perde doğru, yanlış tel | 48.3% | 18.6% |
| yanlış pozitif: **olmayan perde (hayalet)** | 51.7% | **81.4%** |

→ **Polifonide darboğaz tel değil, perde/nota bulma.** Solo'da ikisi yarı yarıya.

### B) Tel karışıklığı (oracle notalar)
- Tel doğruluğu: solo **0.797**, polifonik **0.903** (akor kalıpları teli kolaylaştırıyor).
- Hataların **%98–99'u komşu tel**: aynı perde, 5 fret (G–B arası 4) öte.
- Solo'da en zayıf: **A teli 0.60** (çoğu D'ye kayıyor), B→G, e→B.
  Hataların %53'ünde (polifonik: %71) model kola doğru daha yüksek perdeyi seçiyor.
- Perde bölgesine göre: 5–9 en iyi (0.83 / 0.94); 1–4 ve 10–14 zayıf.

### C) Tel bilgisi notanın neresinde? (oracle, çok adaylı notalar)
| teli şuradan seç | solo | polifonik |
|---|---|---|
| notanın tamamı | **0.797** | **0.903** |
| ilk 1 / 3 / 5 kare (vuruş) | 0.781 / 0.788 / 0.797 | 0.894 / 0.899 / 0.901 |
| vuruş hariç orta | 0.797 | 0.903 |

→ **Hipotez çürütüldü:** teli vuruş anından seçmek kazandırmıyor. Onset çıkışının
asıl değeri tel değil, **nota bulma ve hayalet notaları elemek** (A'daki %81).

### D) Güven
- Solo: yanlış seçimlerin medyan marjı 0.09 (doğrularda 0.25); yalnızca %12'si emin.
  → Hatalar **kararsızlık**: model komşu telleri ayıracak ses özelliğini yeterince
  göremiyor → Adım 2 (zengin tını özelliği) tam hedef.
- Polifonik: yanlışların %65'i emin → orada sistematik (akor kalıbı) hatalar.

## Düzeltilmiş plan
| Adım | İş | Hedeflediği kayıp | Sunumdaki karşılığı |
|---|---|---|---|
| **1** | **Onset çıkışı + onset'e dayalı nota bulma** (Onsets & Frames ilkesi: onset yoksa nota başlamaz; aynı perdede yeni onset = yeni nota) | perde bulunamadı %10–23, hayalet nota %52–81, tekrar eden notaların birleşmesi | Modül 2: "Pitch + onset + offset" |
| **2** | **Harmonik istifleme (HCQT etkisi, önbellek yeniden üretilmeden)** | komşu tel karışıklığı (solo'da tel hatası %17.5, kararsız) | Modül 1: "CQT / spektrogram veya learned representation" |
| (hazır) | Öğrenilen geçiş önseli, w_tr=0.25 | solo'daki kararsız tel seçimleri (oracle 0.797 → 0.819) | Modül 4: playability |

## "Ciddi iyileşme" ölçütü (Katman 3.9 çıkışı)
`tabcrnn_gaps.pt` tabanına göre, tek seed gürültüsü (±0.01) üstünde:
- GuitarSet solo tab F1 **≥ 0.72** (şimdi 0.697)
- GuitarSet val_comp nota F1 **≥ +0.10** (şimdi 0.463)
- GAPS test nota F1 **≥ +0.05** (şimdi 0.360)
- Hiçbir ana metrikte −0.01'den fazla kayıp yok.

## Adım 1 — Onset çıkışı (uygulandı)
- `TabCRNNOnset` (`gtab/models/nets.py`): TabCRNN gövdesi + tab kafası AYNI, üstüne
  tel başına onset kafası. Checkpoint'te `"onset": True` → `load_model` otomatik tanır;
  `forward()` yine tab döndürür, yani tüm eski değerlendirme kodu çalışır.
- Onset hedefi önbellekten türetilir (`string_onsets`, yeniden üretim yok): telde sınıf
  değişimi **veya** perde onset işareti o telin perdesinde → tekrar eden notalar da
  yakalanır (comp: sınıf değişimiyle 1597, bu yolla 1772 ≈ perde onset 1768).
- GAPS için perde-onset olasılığı noisy-OR ile türetilir (onset_tel × P_tel(perde)).
- Çözümleme (`segment_notes_onset`): nota yalnızca onset ile başlar (hayalet nota
  elenir), aynı perdede yeni onset = yeni nota (tekrarlar birleşmez).
- Eşikler doğrulamada seçilir: eğitim onset eşiğini kaydeder; `eval_pitch` perde ve
  onset eşiğini GuitarSet val'de **nota F1** ile birlikte seçer.
- Duman testinde görülen ve düzeltilen iki sorun:
  (1) sıfır bias ile onset kafası her yerde ~0.5 tahminle başlıyordu → bias önsel
  orana (%5.4 → logit −2.94) göre başlatıldı; (2) sabit 0.5 eşiğiyle onset F1 hep 0
  görünüyordu → seçim ölçütü eşik taramalı yapıldı + `--pos-weight` (varsayılan 3).
- **Dikkat:** yarım eğitilmiş onset kafasıyla (F1 0.26) onset'li çözümleme comp'ta
  nota F1'i düşürdü. Karar tam eğitim sonrası, iki çözümleme yan yana ölçülerek
  verilecek (`--onset-thr -1` = eski çözümleme).

## Adım 2 — Harmonik istifleme (uygulandı)
- HCQT'yi ayrı hesaplayıp önbellekleri yeniden üretmek yerine (saatler sürerdi)
  **harmonik istifleme** (`HarmonicStack`, Basic Pitch yaklaşımı): log-frekanslı
  CQT, h. harmoniğe denk gelen bin kadar kaydırılıp kanal yapılır
  (h = 0.5, 1, 2, 3, 4, 5 → −24, 0, 24, 38, 48, 56 bin). Mevcut önbellekle çalışır.
- Giriş katmanı genişletilirken h=1 kanalı eski ağırlığı alır, diğerleri 0 →
  yeni model başlangıçta eskisiyle **birebir aynı** çıktıyı verir (fark 0.0, test edildi).
  Yapılanlar atılmıyor; eğitim harmonik kanalları kullanmayı öğreniyor.
- Not: daha ince frekans çözünürlüğü (36 bin/oktav) önbellek yeniden üretimi
  gerektirir; Adım 2 sonucu yetersiz kalırsa sıradaki aday.

## Çalıştırma (sırayla)
```bash
# Adım 1 — onset kafası (tabcrnn_gaps.pt'den)
python -m scripts.train.train_onset --init tabcrnn_gaps.pt --epochs 15 --ckpt tabcrnn_onset.pt

# Adım 1 ölçüm: onset çözümlemesi vs eski çözümleme (aynı model)
python -m scripts.eval.eval_pitch --ckpt tabcrnn_onset.pt --splits gaps_test val val_comp
python -m scripts.eval.eval_pitch --ckpt tabcrnn_onset.pt --splits gaps_test val val_comp --onset-thr -1
python -m scripts.eval.eval_hmm --model crnn --ckpt tabcrnn_onset.pt
python -m scripts.eval.diagnose_strings --ckpt tabcrnn_onset.pt --splits val val_comp

# Adım 2 — + harmonik istifleme (Adım 1'in sonucundan)
python -m scripts.train.train_onset --init tabcrnn_onset.pt --harmonics default --epochs 15 --ckpt tabcrnn_onset_h.pt
python -m scripts.eval.eval_pitch --ckpt tabcrnn_onset_h.pt --splits gaps_test val val_comp
python -m scripts.eval.eval_hmm --model crnn --ckpt tabcrnn_onset_h.pt
python -m scripts.eval.diagnose_strings --ckpt tabcrnn_onset_h.pt --splits val val_comp
```

## Karşılaştırma tablosu (doldurulacak)
| Model | GS solo tab F1 | oracle tel (solo) | GS val nota F1 | val_comp nota F1 | GAPS nota F1 | GAPS kare F1 |
|---|---|---|---|---|---|---|
| tabcrnn_gaps (taban) | 0.697 | 0.797 | 0.876 | 0.463 | 0.360 | 0.429 |
| tabcrnn_onset — onset çözümleme | | | | | | |
| tabcrnn_onset — eski çözümleme | | | | | | |
| tabcrnn_onset_h | | | | | | |

## Durum
- [x] Adım 0 — hata analizi
- [x] Adım 1 — onset çıkışı: kod + testler (tam eğitim bekleniyor)
- [x] Adım 2 — harmonik istifleme: kod + testler (tam eğitim bekleniyor)
