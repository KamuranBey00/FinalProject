# Proje sözlüğü ve başvuru bilgileri

> Gitar Tab Transkripsiyonu projesi için tek sayfalık başvuru belgesi.
> Hedef okur: gitarı bilen, mühendislik geçmişi olan ama ses işleme ve
> derin öğrenme terimlerine her gün bakmayan biri.
> Önerilen konum: `docs/SOZLUK.md`. Sayılar 8 Ekim 2026 itibarıyla
> (taban model: `tabcrnn_rep_off.pt`); yeni katmanlarda §5 ve §7 güncellenmeli.

## İçindekiler
1. [Projenin sabitleri](#1-projenin-sabitleri)
2. [Dönüşüm tabloları](#2-dönüşüm-tabloları)
3. [Gitar ↔ model eşlemesi](#3-gitar--model-eşlemesi)
4. [Veri setleri ve bölmeler](#4-veri-setleri-ve-bölmeler)
5. [Modeller ve güncel sayılar](#5-modeller-ve-güncel-sayılar)
6. [Metrikler](#6-metrikler)
7. [Terminal çıktısını okuma rehberi](#7-terminal-çıktısını-okuma-rehberi)
8. [Sözlük](#8-sözlük)
9. [Gitar teknikleri (Katman 4–5 için)](#9-gitar-teknikleri-katman-45-için)
10. [Denenmiş ve işe yaramamış olanlar](#10-denenmiş-ve-işe-yaramamış-olanlar)
11. [Kaynaklar](#11-kaynaklar)

---

## 1. Projenin sabitleri
Kaynak: `gtab/config.py`. Bunlardan biri değişirse **tüm modeller ve önbellekler geçersiz olur.**

| Sabit | Değer | Anlamı |
|---|---|---|
| Örnekleme hızı | 22050 Hz | Nyquist ~11 kHz; gitar için yeterli |
| Hop | 512 örnek | iki analiz karesi arası adım |
| Kare süresi | **23.2 ms** | 512 / 22050 |
| Kare hızı | **43.07 kare/sn** | |
| CQT en pes bin | 32.70 Hz (C1) | gitarın en pesinden (E2) iki oktav altı |
| Bin/oktav | 24 | yarım ses başına 2 bin (yarım perde çözünürlüğü) |
| Bin sayısı | 192 | 8 oktav, C1 – ~8.1 kHz |
| CQT Q faktörü | ≈ 34.1 | analiz penceresi süresi ≈ Q / f |
| Normalizasyon | (dB + 80) / 80, [0, 1]'e kırpılır | model girişi |
| Eğitim parçası | 200 kare (~4.6 sn) | BiLSTM eğitimde bu uzunlukta sekans görür |
| Çıkarım parçası | 2000 kare + 100 kare bağlam | uzun kayıtlar örtüşmeli pencerelerle işlenir |
| Nota eşleme toleransı | 50 ms | mir_eval; nota sonu (offset) sayılmaz |

## 2. Dönüşüm tabloları

### 2a. Kare ↔ milisaniye
| kare | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 20 | 43 |
|---|---|---|---|---|---|---|---|---|---|---|
| ms | 23 | 46 | 70 | 93 | 116 | 139 | 186 | 232 | 464 | ~1000 |

Pratik ezber:
- **<100 ms ≈ 4 kareden kısa.**
- **Refrakter 6 kare ≈ 140 ms.**
- **Yedek kural 10 kare ≈ 230 ms.**

### 2b. Tempo → vuruş aralığı (IOI), ms (kare)
| BPM | 4'lük | 8'lik | 8'lik triole | 16'lık | 16'lık sekstole | 32'lik |
|---|---|---|---|---|---|---|
| 60 | 1000 (43) | 500 (21.5) | 333 (14.4) | 250 (10.8) | 167 (7.2) | 125 (5.4) |
| 80 | 750 (32) | 375 (16.1) | 250 (10.8) | 188 (8.1) | 125 (5.4) | 94 (4.0) |
| 100 | 600 (26) | 300 (12.9) | 200 (8.6) | 150 (6.5) | 100 (4.3) | 75 (3.2) |
| 120 | 500 (21.5) | 250 (10.8) | 167 (7.2) | 125 (5.4) | 83 (3.6) | 62 (2.7) |
| 150 | 400 (17.2) | 200 (8.6) | 133 (5.7) | 100 (4.3) | 67 (2.9) | 50 (2.2) |

Okuma örneği: 120 BPM'de 16'lık sekstole tremolo, notalar arası **83 ms ≈ 3.6 kare**.
Bu, projedeki "<100 ms tekrar" kategorisine düşer.

### 2c. Nota ↔ MIDI ↔ Hz ↔ CQT bini ↔ analiz penceresi
Formüller:
- f = 440 · 2^((MIDI − 69) / 12)
- CQT bini = 2 · (MIDI − 24)
- pencere ≈ 34.1 / f saniye

| Nota | MIDI | Hz | CQT bini | Pencere (ms) | Gitarda |
|---|---|---|---|---|---|
| E2 | 40 | 82.41 | 32 | ~414 | 6. tel boş |
| A2 | 45 | 110.00 | 42 | ~310 | 5. tel boş |
| D3 | 50 | 146.83 | 52 | ~232 | 4. tel boş |
| G3 | 55 | 196.00 | 62 | ~174 | 3. tel boş |
| B3 | 59 | 246.94 | 70 | ~138 | 2. tel boş |
| E4 | 64 | 329.63 | 80 | ~104 | 1. tel boş |
| A4 | 69 | 440.00 | 90 | ~78 | 1. tel 5. perde |
| E5 | 76 | 659.26 | 104 | ~52 | 1. tel 12. perde |
| E6 | 88 | 1318.51 | 128 | ~26 | 1. tel 24. perde (modelin üst sınırı) |

Yüksek frekans bölgesi (vuruş geçişlerinin izi burada):

| Frekans | 500 Hz | 1 kHz | 2 kHz | 4 kHz | 8 kHz |
|---------|--------|--------|-------|-------|-------|
| CQT bini | 94    |   118 | 142   | 166   | 190   |
| Pencere | 68 ms  | 34 ms  | 17 ms  | 8.5 ms | 4.3 ms|

> **Neden önemli:** basta pencere 400 ms'ye yaklaşıyor, yani CQT bas notanın
> vuruş anını "bulanık" görür. Tizde ve 2 kHz üstünde pencere birkaç ms'ye iner;
> parmak/tırnak geçişi orada keskin görünür. README12 Adım 2'deki
> "geniş bant akı" fikri bu tabloya dayanıyor.

### 2d. Harmonikler
| Harmonik | Temele göre aralık | Yarım ses | CQT kaydırması (bin) | Doğal flajolet perdesi |
|---|---|---|---|---|
| 0.5 (alt) | bir oktav aşağı | −12 | −24 | — |
| 1 (temel) | — | 0 | 0 | boş tel |
| 2 | oktav | +12 | +24 | 12 |
| 3 | oktav + tam beşli | +19.02 | +38 | 7, 19 |
| 4 | iki oktav | +24 | +48 | 5, 24 |
| 5 | iki oktav + büyük üçlü | +27.86 | +56 | 4, 9, 16 |
| 6 | iki oktav + beşli | +31.02 | +62 | 3.2 civarı |

Bu tablonun projede iki kullanımı var:
- `HarmonicStack` girişte bu kaydırmaları kullanıyor (0.5, 1, 2, 3, 4, 5).
- Tanı çıktısındaki "oktav" ve "5'li/12'li harmonik" hata türleri tam olarak
  2. ve 3. harmoniğe karışan notalar.

## 3. Gitar ↔ model eşlemesi
| Konu | Değer |
|---|---|
| Akort (`STANDARD_6`) | MIDI (40, 45, 50, 55, 59, 64) = E2 A2 D3 G3 B3 E4 |
| Tel indeksi | **0 = en kalın (6. tel, E2)** … 5 = en ince (1. tel, E4). ASCII tab'da tiz tel üstte basılır. |
| Perde aralığı | 0–24 |
| Perde (MIDI) aralığı | 40–88 → **49 perde** |
| Tel başına sınıf | **26** = sessiz + perde 0…24 (sınıf = perde + 1) |
| Bir perdenin pozisyon sayısı | 1–6. Örneğin E2 yalnız 1 yerde; E4 6 yerde (6. telin 24. perdesi dahil). |
| Diğer hazır akortlar | `DROP_D_6`, `SEVEN_STRING`, `BASS_4` (`gtab/core/instrument.py`) |
| Ölçülmüş tel hatası deseni | hataların ~%99'u **komşu tel**; en zayıf tel tarihsel olarak A (5. tel) |

## 4. Veri setleri ve bölmeler
| Ad (önbellek klasörü) | İçerik | Rolü |
|---|---|---|
| `train` | GuitarSet, oyuncu 00–04, solo, 150 kayıt | eğitim (tab etiketi) |
| `val` | GuitarSet, **oyuncu 05**, solo, 30 kayıt | **ana doğrulama**; tab F1 hep burada |
| `train_comp` | GuitarSet, oyuncu 00–04, akorlu eşlik | eğitim |
| `val_comp` | GuitarSet, oyuncu 05, akorlu eşlik, 30 kayıt | polifoni ölçümü |
| `gaps_train` | GAPS, icracı-ayrık eğitim (203 kayıt) | eğitim (yalnız perde etiketi) |
| `gaps_val` | `gaps_train` içinden ayrılan 28 kayıt | GAPS için eşik/kural seçimi |
| `gaps_test` | GAPS resmi test, 30 kayıt | **yalnız raporlama** |
| `synthtab_*` | SynthTab naylon (luthier) alt kümesi | ön-eğitim adayı |

Veri setlerinin kendisi:
- **GuitarSet**
  - 360 kayıt: 6 gitarist, her biri solo ve eşlik.
  - Çelik telli akustik gitar.
  - Hexafonik manyetik (her tel ayrı kanal) → tel etiketi gerçek. Model yine de mikrofon kaydını dinliyor.
- **GAPS v1.1**
  - 300+ gerçek klasik gitar performansı, 48 kHz stereo (yükleyici 22050 Hz mono'ya indirir).
  - Etiket: partisyondan sese hizalanmış MIDI. Nota **başlangıçları** iyi; **bitişleri** partisyondaki nota değeri kadar, telin gerçekte ne kadar çınladığını göstermiyor.
  - Yanında MusicXML (tab + sol el parmak numarası) ve `.match` hizalama dosyaları var.
  - Resmi bölme 270 / 30 / 101 (atanmamış). 12 icracı hem train'de hem test'te olduğu için `--disjoint` ile 39 kayıt eğitimden çıkarılıyor.
- **SynthTab**
  - Sentetik gitar; `luthier_*` = Ample Guitar L (klasik naylon).
  - Tam set ~2 TB. Dev setinde kullanılabilir 4 naylon parça var (0.18 saat).

Değerlendirme kuralları (değiştirilmez):
- Ayar seçimi doğrulamada yapılır; test kümesine bakılmaz.
- GAPS icracı-ayrık değerlendirilir.
- Tek seed farkları ±0.01 gürültü sayılır.

## 5. Modeller ve güncel sayılar

### 5a. Checkpoint'ler (`checkpoints/`)
| Dosya | Katman | Ne eklendi |
|---|---|---|
| `pitchcnn.pt` | 2 | yalnız perde (tel yok) |
| `tabcnn.pt` | 3 | tel + perde, kare bazlı CNN |
| `tabcrnn.pt` | 3.5 | + BiLSTM (zamansal bağlam) |
| `tabcrnn_base.pt`, `tabcrnn_comp.pt`, `tabcrnn_aug.pt` | 3.7 | taban / + akorlu kayıtlar / + CQT çoğaltma (zararlı çıktı) |
| `tabcrnn_gaps.pt` | 3.8 | + GAPS klasik gitar ince ayarı |
| `tabcrnn_onset.pt` | 3.9 | + onset kafası |
| `tabcrnn_onset_h.pt` | 3.9 | + harmonik istifleme |
| `tabcrnn_poly.pt` | 3.10 | + keskin onset, GuitarSet perde kaybı, kısa nota ağırlığı |
| **`tabcrnn_rep_off.pt`** | 3.10 | **güncel taban:** + offset kafası, hızlı tekrar onset ağırlığı, GAPS perde ağırlığı ×2 |
| `transitions.npz` | 3.6b | öğrenilen tel geçiş modeli (güçlü modellerde katkısız) |

### 5b. Güncel taban (`tabcrnn_rep_off.pt`)
| Ölçü | Değer |
|---|---|
| GuitarSet solo **tab F1** (greedy / Viterbi) | **0.760 / 0.760** |
| Oracle tel doğruluğu | 0.862 |
| GAPS test nota / kare F1 (@100 ms) | 0.689 / 0.675 (0.703) |
| GuitarSet val nota / kare F1 | 0.915 / 0.871 |
| val_comp nota / kare F1 | 0.769 / 0.817 |
| gaps_val kaçan / hayalet | %28.3 / %33.7 |
| val_comp kaçan / hayalet | %18.1 / %27.5 |
| <100 ms aynı perde tekrarı kaçma (GAPS / val_comp) | %75.0 (n=647) / %72.5 (n=80) |

Gelişim (GuitarSet solo tab F1):
0.586 TabCNN → 0.634 CRNN → 0.667 +comp → 0.697 +GAPS → 0.739 +onset+harmonik
→ 0.743 kurallar → 0.748 poly → **0.760** rep_off.

### 5c. Seçilmiş çözümleme ayarları
| Parametre | GuitarSet (val'de seçildi) | GAPS (gaps_val'de seçildi) |
|---|---|---|
| perde eşiği | 0.7 | 0.3 |
| onset eşiği | 0.2 | 0.1 |
| histerezis (off_ratio) | 1.0 (kapalı) | 0.5 |
| refrakter | 6 kare | 6 kare |
| yedek kural | 10 kare | 10 kare |
| tepe başlangıcı | açık | açık |
| yeniden vuruş | 0.4 | 0.3 |
| enerji kabul / bölme | 6 / 0 dB | 6 / 8 dB |
| offset eşiği | 0.5 | kapalı |

## 6. Metrikler
| Ad | Tanım | Gitar dilinde |
|---|---|---|
| **TP / FP / FN** | doğru bulunan / uydurulan / kaçırılan | |
| **Precision (P)** | TP / (TP + FP) | "çaldı" dediklerimin kaçı gerçekten çalındı |
| **Recall (R)** | TP / (TP + FN) | çalınanların kaçını yakaladım |
| **F1** | 2PR / (P + R) | ikisinin dengesi; ikisinden biri düşükse F1 düşer |
| **Nota F1** | perde doğru ve başlangıç ±50 ms içinde; nota sonu sayılmaz | tab okuyanın gördüğü: hangi nota, ne zaman |
| **@100ms** | aynı ayarla tolerans 100 ms | 50 ms'den çok farkı yoksa hata zamanlamada değil |
| **Kare F1** | her 23 ms'lik karede çalan perdelerin doğruluğu | sustain dahil |
| **Tab F1** | kare düzeyinde tel ve perde birlikte doğru | en katı ölçü; ana proje metriği |
| **Oracle tel doğruluğu** | gerçek notalar verildiğinde doğru tel seçme oranı | saf tel seçimi becerisi |
| **Polifoni kırılımı** | aynı anda 1 / 2 / 3 / 4+ nota çalan karelerde ayrı kare F1 | tek nota ve akor ayrı ayrı |
| **mix ölçütü** | (nota F1 + kare F1) / 2; eşik seçiminde kullanılır | hem vuruş hem sustain önemli |
| **AUC** | bir özelliğin iki durumu ayırma gücü; 1.0 kusursuz, 0.5 yazı tura | |
| **@FA5%** | %5 yanlış alarma razı olunca yakalama oranı | |

Hızlı sağlama: kaçan %k ve hayalet %h ise R = 1 − k, P = 1 − h, F1 = 2PR/(P+R).
Örnek gaps_val: R 0.717, P 0.663 → F1 ≈ 0.689.

## 7. Terminal çıktısını okuma rehberi

### 7a. `eval_pitch` — eşik tablosu ve özet
- **Eşik tablosu:** eşik büyüdükçe P artar, R düşer. `<- secilen` satırı doğrulama kümesinde seçilen ayardır; raporlanan kümenin en iyisi olmak zorunda değildir.
- **Özet tablo sütunları:**
  - `split`: raporlanan küme.
  - `secim`: eşiklerin seçildiği küme.
  - `cozumleme onset@x`: nota onset ile başlatılıyor, onset eşiği x.
  - `esik`: perde eşiği.
  - `nota F1`, `@100ms`, `kare F1`, `polifoni 1/2/3/4+`: §6'daki tanımlar.

### 7b. `diagnose_pitch` — bölümler
| Bölüm | Soru | Kategoriler |
|---|---|---|
| Özet satır | ne kadar kaçtı, ne kadar uyduruldu | kaçan = 1 − R; hayalet = 1 − P |
| **A** | kaçanlar neden kaçtı | perde hiç aktif değil (model duymadı) · oktav/harmonik seçilmiş · onset kaçtı (perde duyuldu, nota başlatılmadı) · onset zamanı kaymış (>50 ms; hem kaçan hem hayalet sayılır) |
| **A'** | hangi gruplar daha çok kaçıyor | süre (kısa <5 kare) · register (pes <52, orta, tiz 64+) · polifoni · tekrar |
| **B** | hayaletler gerçek notalarla nasıl ilişkili | aynı perde (çift tetik) · oktav · 5'li/12'li harmonik · komşu ses · diğer aralık · sessizlikte |
| **C** | kare kaybı notanın neresinde | baş %20 / orta / son %20 (son yüksekse sönen kuyruk kesiliyor) |
| **D** | hayaletlerin alt türü | fragman (gerçek nota bulunmuş, model ikiye bölmüş) · kayık başlangıç · nota bittikten sonra · farklı perde; kaynak (onset mi yedek kural mı); onset tepe değerleri ve "tepe eşiği" ne-olurdu tabloları |
| **E** | hızlı aynı perde tekrarları | IOI'ye göre kaçma oranı; **gömülü** = önceki tahmin notasının içinde kalmış; enerji yükselişi (dB) ve dB eşik tablosu |

**"Ne-olurdu" tablosu nasıl okunur:**
- `>=0.5: hayalet -%87 / dogru -%67` → vuruş gücü 0.5'in altındakileri atarsak
  hayaletlerin %87'si gider ama doğru notaların %67'si de gider. Kötü takas.
- dB tablosunda bir satır "bu eşiği koysak gömülü tekrarın yüzde kaçı yakalanır,
  süren notada yüzde kaç yanlış alarm olur".

### 7c. Bu sayı yüksekse nereye bakmalı
| Belirti | Muhtemel kaynak | Çözüm yeri |
|---|---|---|
| "perde hiç aktif değil" yüksek | akustik model duymuyor | eğitim / veri |
| oktav / harmonik yüksek | harmonik karışıklığı | eğitim (harmonik istifleme, sert negatif) |
| "onset kaçtı" yüksek | onset kafası zayıf | yedek kural (çözümleme) ya da onset eğitimi |
| "zamanı kaymış" yüksek ve @100ms farkı küçük | nota parçalanması / geç başlangıç | çözümleme (tepe, yeniden vuruş) |
| fragman yüksek | çalan nota ikiye bölünüyor | çözümleme (yeniden vuruş, enerji, strum gruplama) |
| son %20 kare kaybı yüksek | kuyruk kesiliyor | histerezis, offset kafası (GAPS'te etiket sınırı) |
| hızlı tekrar "gömülü" yüksek | iki vuruş tek notaya birleşiyor | çözümleme tavanı (README12 Adım 1) |
| oracle tel düşük, perde F1 yüksek | tel seçimi zayıf | tel tarafı (tını özellikleri, veri) |

## 8. Sözlük

### Ses ve temsil
- **Kare (frame):** 23.2 ms'lik analiz dilimi. Model zamanı kare kare görür.
- **CQT (Constant-Q Transform):** frekans eksenini perdelere hizalı bölen spektrum analizörü. Yarım ses başına 2 bin. Pesde uzun, tizde kısa pencere kullanır (§2c).
- **log-CQT / dB:** genliğin logaritmik ölçeği, kulağın algısına yakın. +6 dB ≈ genlik ×2.
- **Harmonik:** telin temel frekansın katlarında verdiği üst sesler. Tınıyı ve tel kimliğini taşır (§2d).
- **Harmonik istifleme (HarmonicStack):** spektrumun kaydırılmış kopyaları kanal olarak üst üste konur; notanın tüm harmonikleri aynı hizaya gelir. Ön hesap ya da yeni önbellek gerektirmez.
- **İnharmonisite:** gerçek telde üst harmoniklerin tam katlardan hafif sapması. Kalın sarımlı tel ile ince düz tel farklı sapar; tel ayrımının ipuçlarından biri.
- **Onset:** notanın başladığı an (vuruş, koparma).
- **Offset:** notanın bittiği an (susturma ya da sönme).
- **IOI (inter-onset interval):** iki vuruş başlangıcı arası süre (§2b).
- **Enerji yükselişi (rise):** bir perdenin harmonik enerjisinin son birkaç kareye göre kaç dB sıçradığı. Yeniden çekilen tel sıçratır, sadece sönen tel sıçratmaz.
- **Geniş bant akı (spectral flux):** tüm frekanslardaki ani enerji artışının toplamı; perdeden bağımsız "bir şey vuruldu" kanıtı.
- **Gömülü tekrar:** önceki nota sönmeden gelen aynı perde vuruşu. Spektrumda yalnızca küçük bir sıçrama bırakır.

### Model
- **Perde ve tel/perde ayrımı:** aynı perde (ör. E4) gitarda birden fazla yerde çalınabilir. "Hangi perde" kolay soru, "hangi telde" zor soru.
- **CNN:** spektrumdaki yerel desenleri okuyan katmanlar; modelin "kulağı".
- **BiLSTM:** zaman boyunca ileri ve geri bakan tekrarlayan ağ; modelin "müzikal hafızası".
- **CRNN / TabCRNN:** CNN + BiLSTM.
- **TabCRNNOnset:** + onset kafası (+ isteğe bağlı offset kafası).
- **Kafa (head):** gövdenin üstündeki çıkış katmanı.
  - tab kafası: tel başına 26 sınıf;
  - onset kafası: "bu karede bu tele vuruldu";
  - offset kafası: "bu karede bu teldeki nota bitti".
- **Softmax (tel başına):** her tel için 26 sınıfın olasılıkları toplamı 1. Fiziksel kural: bir tel aynı anda tek perdede çalar.
- **Logit / sigmoid:** ham çıkış ve onun 0–1 olasılığa çevrilmiş hâli.
- **max ve noisy-OR:** bir perdenin olasılığını tellerden birleştirme.
  - max: en emin teli alır.
  - noisy-OR: 1 − Π(1 − p_tel), yani "en az bir telde çalıyor".
  - Eğitim noisy-OR, çözümleme max kullanıyor (README12 §1b).
- **Checkpoint (.pt):** eğitilmiş ağırlıkların kaydı; modelin "preset"i.
- **Sıcak başlatma (warm start):** yeni modeli eski checkpoint'ten başlatmak. Yeni kafalar önsel değerle başlar, eski davranış korunur.

### Eğitim
- **Epoch:** eğitim verisinin bir kez baştan sona geçilmesi.
- **Kayıp (loss):** modelin hatasının eğitimde küçültülen sayısal ölçüsü.
  - CE (cross-entropy): sınıflar için.
  - BCE: evet/hayır çıkışları (onset, perde) için.
- **pos_weight / sınıf ağırlığı:** seyrek pozitifleri (ör. onset kareleri ~%5) kayıpta güçlendirmek.
- **Keskin onset hedefi (`--onset-soft`):** onset karesi 1, sonraki kare yumuşak değer (0.3). Eski `dilate=2` iki kareyi de 1 yapıyordu.
- **Tekrar ağırlığı (`--repeat-weight`):** hızlı aynı perde tekrarının onset karesine ek kayıp ağırlığı.
- **Kısa nota ağırlığı:** 5 kareden kısa notaların karelerine ek ağırlık.
- **Perde ağırlığı (`--pitch-weight`):** GAPS perde kaybının çarpanı.
- **Seed:** rastgeleliğin başlangıç değeri. Aynı eğitimi farklı seed'le tekrarlamak ±0.01 oynatır.
- **Aşırı öğrenme (overfit):** eğitimde iyileşip doğrulamada kötüleşmek. Oyuncu-ayrık doğrulama bunu yakalamak için var.
- **Çoğaltma (augmentation):** eğitim verisini yapay çeşitlendirmek. CQT'de perde kaydırma denendi, zarar verdi (§10).

### Çözümleme (olasılıklardan notaya)
- **Perde eşiği:** "bu perde çalıyor" demek için gereken olasılık. Noise gate eşiğinin karşılığı.
- **Onset eşiği (onset@x):** nota başlatmak için gereken vuruş olasılığı.
- **Onset tabanlı çözümleme (Onsets & Frames ilkesi):** nota yalnızca vuruşla başlar, perde aktif kaldıkça sürer.
- **Kare-eşik çözümleme:** eski yöntem; aktif ardışık kareleri nota sayar, tekrarları birleştirir.
- **Tepe (peak):** notayı vuruş eğrisinin en yüksek karesinden başlatmak.
- **Histerezis (off_ratio):** başlatma ve sürdürme eşikleri farklı (ör. 0.3 ve 0.15); sönen kuyruk kesilmez. Gate'teki hysteresis ile aynı mantık.
- **Refrakter:** nota başladıktan sonra belli sayıda kare aynı perdede yeni vuruşu yok saymak. Çift tetiği önler ama hızlı tremoloyu da yutar.
- **Yedek kural (fallback):** vuruş görülmese bile uzun süre aktif kalan perdeyi nota saymak.
- **Yeniden vuruş (reattack):** nota çalarken yeni nota açmak için gereken minimum vuruş gücü.
- **Enerji kabul / bölme (rise_keep / rise_split):** yeniden vuruşu kabul etmek ya da süren notayı bölmek için gereken dB sıçraması.
- **Yerel tepe seçici (önerilen, README12):** vuruş adaylarını eşik aşan koşu yerine yerel maksimum + vadi derinliğiyle seçmek.
  - **Vadi derinliği (prominence):** iki tepe arasında eğrinin ne kadar inmesi gerektiği.
- **Fragman:** tek gerçek notanın model tarafından ikiye bölünmüş parçası.
- **Strum gruplama (önerilen):** birkaç kare içinde çok telde gelen vuruşları tek tıngırtı olayı saymak.
- **Greedy ve Viterbi:**
  - greedy: her notaya teli tek tek seçmek;
  - Viterbi: tüm pasaja bakıp en olası el yolunu seçmek;
  - `w_transition = 0` iken ikisi aynı. Güncel modelde Viterbi katkısız.
- **Kalibrasyon:** eşik ve kuralların doğrulama kümesinde aranıp seçilmesi; her alan (GuitarSet / GAPS) kendi doğrulamasında.

### Ölçüm ve deney düzeni
- **Doğrulama (val) ve test:** ayarlar val'de seçilir, test sadece rapor içindir. Teste bakarak ayar seçmek "sızıntı"dır.
- **İcracı-ayrık (performer-disjoint):** test icracıları eğitimde hiç yok; model hiç duymadığı gitaristle sınanır.
- **Alan (domain) farkı:** çelik telli (GuitarSet) ve naylon (GAPS) arasındaki ses farkı. Alan-içi kalibrasyon bunun için var.
- **Etiket gürültüsü:** etiketlerin kendisindeki hata (ör. GAPS nota bitişleri).
- **Hayalet nota (bu projede):** **yanlış pozitif**; olmayan ya da ikiye bölünmüş nota. Müzikteki "ghost note" ile karıştırma (§9).
- **Kaçan nota:** yanlış negatif.
- **Çıkış ölçütü / "ciddi iyileşme":** bir katmanın kapanması için önceden yazılan sayısal hedef.

### Yazılım süreci
- **Duman testi (smoke test):** yeni devreye ilk kez güç verip duman çıkıyor mu bakmak gibi; "bariz bir şey bozulmadı mı" kontrolü. Performans ölçmez. Bu projede iki biçimi var:
  - kod değişikliğinden sonra mini veriyle uçtan uca çalıştırma ve 1 epoch'luk eğitim;
  - sentetik sesle davranış kontrolü (README12 §2). Mutlak sayılar gerçek veriye taşınamaz, yalnız aynı model çıktısı üzerinde A–B karşılaştırması anlamlıdır.
- **Regresyon testi ("birebir aynı" kontrolü):** yeni kod varsayılan ayarlarla eski modelde eski sonucu birebir vermeli.
- **Birim testi:** tek bir fonksiyonu elle kurulmuş küçük örnekle doğrulamak (ör. 70 ms'lik iki vuruş → 2 nota).
- **Ablasyon:** bir bileşeni kapatıp ya da sabitleyip etkisini ayrıca ölçmek.
- **Önbellek (cache, `.npz`):** CQT ve etiketlerin bir kez hesaplanıp saklandığı dosyalar. Git'e girmez.
- **Dal (branch):** her katman ayrı git dalında (`katman-<no>-<konu>`).

## 9. Gitar teknikleri (Katman 4–5 için)
| Terim | Ne | Seste/modelde iz |
|---|---|---|
| **Ghost note (müzikte)** | susturulmuş ya da çok hafif, perdesiz/perküsif nota | perde neredeyse yok, kısa geniş bant tık; şemada `DEAD_NOTE` |
| **Tremolo (klasik)** | p-a-m-i ile aynı tiz notanın hızlı tekrarı + bas | 60–100 ms IOI aynı perde; projedeki "<100 ms tekrar" |
| **Rasgueado** | parmakların sırayla açılarak tıngırtı | çok telde yakın, art arda vuruşlar |
| **Arpej** | akor notalarının sırayla çalınması | farklı tellerde ardışık onset'ler, notalar birbirine binerek çınlar |
| **Apoyando / tirando** | dayanarak / serbest vuruş | apoyando daha güçlü, daha keskin geçiş |
| **p-i-m-a** | baş, işaret, orta, yüzük parmağı (sağ el) | GAPS MusicXML'de parmak bilgisi var |
| **Let ring** | notaları birbirinin üstüne çınlamaya bırakmak | kare düzeyinde polifoni artar |
| **Palm mute** | elle köprü yanında boğuk çalma | kısa sönüm, zayıf üst harmonikler |
| **Hammer-on / pull-off (legato)** | vuruşsuz perde değişimi | perde değişir ama **onset yok** ya da zayıf |
| **Slide (glissando)** | parmak telde kayar | perde sürekli değişir, ara perdelerden geçer |
| **Bend** | teli çekerek perdeyi yükseltmek | F0 sürekli yükselir, yeni onset yok; yarım ses = 2 CQT bini |
| **Vibrato** | perde etrafında periyodik salınım | 5–7 Hz civarı küçük F0 dalgalanması |
| **Doğal flajolet** | düğüm noktasına dokunarak harmonik | perde yerine 2./3./4. harmonik duyulur (§2d) |
| **Yapay flajolet** | perdeye basıp 12 perde üstte dokunmak | basılan notanın oktavı |
| **Sustain / sönüm** | notanın çınlama süresi | naylon çelikten kısa; tizde daha kısa |

## 10. Denenmiş ve işe yaramamış olanlar
| Deneme | Sonuç | Katman |
|---|---|---|
| El yapımı "en az el hareketi" Viterbi maliyeti | gerçek veride katkı yok | 3.6 |
| Öğrenilen tel geçiş önseli (güçlü modellerde) | katkı yok (`w_tr = 0` seçiliyor) | 3.6b, 3.9+ |
| CQT'de perde kaydırmalı çoğaltma | tab F1 −0.06 | 3.7 |
| Teli yalnız vuruş anından seçmek | hipotez çürütüldü | 3.9 |
| Kısa pencere girişi / hop 256 | hızlı tekrar ayrımı artmadı (AUC ~0.89) | 3.10 |
| GAPS'te offset eşiği | seçilmiyor (etiket bitişleri partisyon hizalı) | 3.10 |
| Refrakter penceresi (hızlı tekrarlar için) | ≤140 ms gerçek tekrarları siliyor; README12'de yeniden ele alınıyor | 3.10 |

## 11. Kaynaklar
- GuitarSet — https://guitarset.weebly.com/ (mirdata ile indirilir)
- GAPS v1.1 — https://huggingface.co/datasets/xavriley/GAPS (MIT)
- SynthTab — https://github.com/yongyizang/SynthTab
- Proje içi: `README.md`, `CONTRIBUTING.md`, `CLAUDE.md`, `docs/devlog/README*.md`, `inputs/Klasik_Gitar_AI_Proje_Sunumu.pptx`
