# Katman 3.11 — Hızlı tekrarlar: tepe tabanlı çözümleme, noisy-OR ve perde-onset

> Durum: **plan + duman testi** (8 Ekim 2026). Kod henüz yazılmadı.
> Taban: `checkpoints/tabcrnn_rep_off.pt` (Katman 3.10 Adım 3c).
> Bu belge, Katman 3.10'un açık kalan iki maddesini hedefler: <100 ms aynı perde
> tekrarları (GAPS'te %75 kaçıyor) ve val_comp çıkış ölçütü (0.780 / 0.830).
> Katman 3.10'u kapatıp burayı 3.11 olarak mı açacağınız, yoksa 3.10'un Adım 4'ü
> olarak mı yürüteceğiniz kullanıcı kararıdır. İçerik ikisinde de aynıdır.

## 0. Terim ve ölçek

- Devlog'larda **"hayalet nota" = yanlış pozitif** (olmayan ya da ikiye bölünmüş nota).
- Hedeflenen **%75 kaçış**, GAPS'te **<100 ms aralıklı aynı perde tekrarlarıdır**
  (tremolo, hızlı tekrar). Bu belge onlara **"hızlı tekrar"** der.
- **Ölçek:** GAPS notalarının %17.9'u hızlı tekrar, bunların %13'ü <100 ms.
  Yani <100 ms tekrarlar notaların yaklaşık **%2.3**'ü. Hepsi yakalansa nota F1'e
  etkisi yaklaşık +0.01–0.02 olur.
- Yine de düzeltmeye değer: tremolo klasik gitarın imza tekniği ve hata gözle çok
  görünür. Ayrıca aynı çözümleme değişikliği 100–200 ms tekrarlara (GAPS kaçma %55)
  ve akor tekrarlarına (val_comp notalarının %38'i) da dokunuyor.
- **GAPS'teki en büyük kayıplar başka yerde:** kaçanların %45'inde perde hiç aktif
  değil, hayaletlerin %53'ü yanlış perde. Bunlar bu katmanın değil, eğitim ve veri
  tarafının konusu (bkz. §7).

## 1. Teşhis: kod okumasından iki yapısal bulgu

### 1a. Çözümleme, birbirine yakın vuruşları yapısal olarak birleştiriyor
`segment_notes_onset` (`gtab/decoding/viterbi.py`) nota başlangıcını, onset eşiğini
aşan **koşunun yükselen kenarından** alıyor. GAPS'te seçilen eşik 0.1. İki yakın
vuruş arasındaki vadi neredeyse hiç 0.1'in altına inmediği için iki onset tek koşuya
birleşiyor ve tek nota başlatılıyor.

Buna ek olarak, <100 ms'lik bir tekrar (≤4 kare) şu dört kapının **hepsine** takılıyor:

| Kapı | Neden eliyor |
|---|---|
| koşu tabanlı başlangıç (onset@0.1) | vadi eşiğin altına inmiyor → iki onset tek koşu |
| `refractory = 6` (~140 ms) | önceki nota sürerken gelen her tekrarı siliyor |
| `rise_keep = 6` dB | gömülü tekrarlarda ölçülen medyan yükseliş 3.6 dB |
| `rise_split` içindeki sabit `> 3` kare koşulu | mevcut başlangıca 3 kareden yakın adayı atıyor |

Tekrar ağırlıklı eğitimin (3c) <100 ms kaçmayı hiç oynatmaması (%76 → %75) bu
yapısal tavanla tutarlı.

Elle kurulmuş senaryoda doğrulandı: model 70 ms arayla iki net tepe veriyor
(0.85 / 0.75, vadi 0.25). Mevcut GAPS seçimi tek nota çıkarıyor (refrakter 0 olsa
bile). Yerel tepe seçici iki nota çıkarıyor.

### 1b. Eğitim–çıkarım uyumsuzluğu: noisy-OR ve max
- **Eğitim:** GAPS perde ve perde-onset olasılığı teller üzerinden **noisy-OR**
  ile hesaplanıyor (`gtab/models/losses.py`: `pitch_probs`, `pitch_onset_probs`).
- **Çözümleme:** `pitch_matrix` ve `pitch_onset_matrix` **max** alıyor.

GAPS'te yalnızca perde etiketi olduğu için model olasılığı birden fazla tele
dağıtmaktan çekinmez. Örneğin iki tele 0.45'er dağılmış bir perde max ile 0.45,
noisy-OR ile 0.70 görünür. Bu fark, zayıf onset'leri ve "perde aktif değil"
kaçışlarını büyütüyor olabilir.

## 2. Duman testi (sentetik ses, gerçek model)

**Kurulum:** repo kodu ve `tabcrnn_rep_off.pt`, CPU. Veri seti yok; ses sentetik
üretildi:
- modal naylon tel modeli: inharmonik kısmi sesler, koparma konumu, parmak tıkı,
  gövde rezonansları (98 / 205 / 400 Hz), kısa oda yankısı;
- aynı tel yeniden çekilince eski titreşim ~8 ms'de söner.

**Sahne (her seed'de 132 nota, 23.8 sn):**
- klasik tremolo (p-a-m-i: bas + aynı tiz nota üç kez), IOI 60 / 75 / 90 / 110 / 140 / 200 ms;
- düz tekrar dizileri, IOI 60 / 80 / 100 / 150 ms;
- tekrarsız melodi (kontrol).

**Mevcut GAPS seçimi:** onset@0.1, eşik 0.3, tepe, histerezis 0.5, refrakter 6,
yedek 10, yeniden vuruş 0.3, enerji 6/8.

**Prototip:** yerel tepe seçici (eşik 0.1, en az 2 kare ara, vadi ≥ 0.1) + noisy-OR.
Nota uzatma, histerezis ve yedek kural mevcutla aynı.

**Eşleme:** mir_eval, 50 ms onset toleransı.

### Sonuçlar (3 seed ortalaması; parantez içinde seed 0 / 1 / 2)

| | mevcut GAPS (A) | A, refrakter 0 (B) | mevcut + noisy-OR (C) | **yerel tepe + noisy-OR (E)** |
|---|---|---|---|---|
| <100 ms tekrar yakalama | %14 (17/6/19) | %20 (28/11/22) | %15 (17/6/22) | **%33** (39/17/44) |
| 100–200 ms yakalama | %16 (28/14/7) | %21 (31/14/17) | %23 (31/29/10) | **%51** (72/43/37) |
| tekrarsız yakalama | %65 | %65 | %65 | %68 |
| nota P | 0.53 | 0.53 | 0.49 | 0.43 |
| nota F1 | 0.456 | 0.473 | 0.455 | **0.480** |

**Tepe görünürlüğü (Adım 0'daki tanı, noisy-OR):** <100 ms tekrarlarda ±1 kare
içinde ayrı bir onset tepesi %30 (36 / 11 / 42), tekrarsız notalarda %58.

### Yorum
1. **Çözümleme tavanı gerçek.** Mevcut çözümleme, modelin zaten görünür kıldığı
   <100 ms tepelerinin yarısını atıyor (görünür %30, yakalanan %14). Yerel tepe
   seçici görünür tepelerin neredeyse tamamını notaya çeviriyor (%33).
   100–200 ms'de kazanç daha büyük: %16 → %51.
2. **noisy-OR tek başına yetmiyor** (C ≈ A): koşu tabanlı çözümleme vadileri yine
   birleştiriyor. Yerel tepe seçiciyle birlikte, 100–200 ms görünürlüğünü artırarak
   işe yarıyor. İkisi birlikte seçilmeli.
3. **Bedel precision** (0.53 → 0.43). Seed 0'da vadi 0.25 ile precision 0.50'ye döndü ama
   yakalama düştü. Vadi derinliği doğrulamada seçilmeli, sabitlenmemeli.
4. **Kalan kayıp model tarafında.** <100 ms tekrarların ~%70'inde model ayrı bir
   tepe hiç vermiyor; seed 1'de bu oran %90. Çözümleme bunu kurtaramaz. Adım 4
   (eğitim) gerekecek gibi görünüyor, ama karar gerçek veride Adım 0 ölçümüyle
   verilmeli.

**Sınırlar:** Model sentetik sesi gerçek kayıttan daha kötü duyuyor (genel nota F1
~0.44–0.52, GAPS gerçek kayıtta 0.689). Mutlak sayılar GAPS'e taşınamaz; anlamlı olan,
**aynı model çıktısı üzerinde çözümlemelerin karşılaştırması**. Akorlar (strum) bu
testte yoktu.

## 3. Plan

Her adımda varsayılan değerler eski davranışı **birebir** korur. Yeni parametreler
doğrulamada seçilir: GAPS için gaps_val, GuitarSet için val. Test setleri yalnızca
raporlama içindir.

### Adım 0 — Ölçüm: tepe görünürlüğü (eğitim yok)
`diagnose_pitch`'e yeni bölüm **F** eklenir. Her gerçek nota için, perde-onset
eğrisinde ±1 kare içinde (eşik + vadi derinliği) yerel tepe var mı? Ayrıca:
- IOI kırılımı: <100 / 100–200 / ≥200 ms / tekrarsız;
- max ve noisy-OR karşılaştırması;
- "görünür ama çözümlemenin attığı" oran = çözümleme tavanı.

**Karar dalı (gaps_val, <100 ms tekrarlar):**
- görünürlük ≥ %60 → sorun büyük ölçüde çözümlemede; Adım 1–2 yeterli olabilir;
- görünürlük ≤ %30 → model tekrarı görmüyor; Adım 1–2'ye ek olarak Adım 4 gerekli;
- arada → ikisi de.

### Adım 1 — Tepe tabanlı çözümleme + noisy-OR (eğitim yok)
1. **Yerel tepe seçici:** onset adayı = yerel maksimum ve iki koşul:
   - en az `min_dist` kare ara (önerilen 2);
   - önceki tepeden bu yana eğri en az `prominence` kadar inmiş olmalı.
   Bu, "düşük eşikle her şeyi duy" ile "yakın vuruşları ayır"ı birbirinden
   bağımsız ayarlanabilir yapar.
2. **noisy-OR birleştirme:** perde ve perde-onset matrisleri için `max` ya da
   `noisyor`, eğitimle uyumlu.
3. **Sabitlerin parametreleşmesi:** `rise_split` içindeki `> 3` kare koşulu
   `min_dist`'e bağlanır. Refrakter penceresi vadi derinliği ile birlikte aranır
   (refrakter 0 adayı her zaman denenir).
4. **Kalibrasyon araması:**
   - birleştirme ∈ {max, noisyor} × başlangıç ∈ {koşu, yerel tepe};
   - yerel tepe için prominence ∈ {0.05, 0.1, 0.15, 0.25} × min_dist ∈ {1, 2, 3};
   - mevcut kurallarla (histerezis, yedek, yeniden vuruş, enerji) birlikte.

Testler:
- varsayılanda eski çözümlemeyle birebir aynı sonuç (`eval_pitch --no-dec-search --limit 3` regresyonu);
- elle kurulmuş 70 ms senaryosunda 2 nota;
- noisy-OR'un `losses.pitch_onset_probs` ile sayısal eşitliği.

### Adım 2 — Enerji kanıtını güçlendirmek (eğitim yok, önbellek yeniden üretilmez)
1. **Sönüm telafili yükseliş:** `energy_rise` şu an son 3 karenin minimumuna göre
   ölçüyor. Önceki vuruş hâlâ tepedeyken gömülü tekrarın yükselişi küçük çıkıyor
   (medyan 3.6 dB). Bunun yerine beklenen sönüme (son karelerdeki eğim) göre
   sapma ölçülür.
2. **Geniş bant yüksek frekans akısı:** CQT'nin ~2 kHz üstü binlerinde pencere kısa
   (~10 ms). Tırnak ya da parmağın tele değdiği an, perde zaten çalıyor olsa bile
   burada geniş bantlı bir iz bırakır. Mevcut el yapımı özellikler yalnızca perdenin
   kendi harmoniklerine baktığı için bu kanıtı hiç görmedi.
3. **Önce ölç:** ikisi de `diagnose_onset_features`'a yeni satır olarak eklenir
   (AUC ve @FA5%, <100 ms / pes kırılımı). Mevcut CQT k=3 satırını (GAPS AUC 0.889)
   belirgin geçmezse çözümlemeye alınmaz.

### Adım 3 — Akorlar (val_comp: +0.011 / +0.013 eksik)
1. **Tıngırtı (strum) gruplama:** akorda teller 5–40 ms arayla vurulur. Birkaç kare
   içinde ≥3 telde gelen onset tepeleri tek bir vuruş olayı sayılır.
   - Vuruş olayı varken çalan aynı perde yeniden başlatılır.
   - Vuruş olayı yokken tek bir telde beliren zayıf onset parça (fragman) sayılıp
     yok sayılır. Akor hayaletlerinin %67'si bu tür aynı perde parçaları.
2. **Seçim parçası (kullanıcı onayı gerekli):** kurallar şu an hızlı tekrarın
   yalnızca %7 olduğu GuitarSet solo val'de seçiliyor. Bu yüzden akora yarayan
   kurallar hiç seçilmiyor. Öneri: val_comp'u kayıt bazında ikiye bölmek,
   yarısını seçimde, yarısını raporda kullanmak. Bu bir değerlendirme kuralı
   değişikliği olduğu için onay olmadan yapılmaz.

### Adım 4 — Eğitim (yalnız Adım 0 gerekli gösterirse)
1. **Vadi hedefi:** iki yakın aynı perde onset'i arasındaki karelere yüksek
   ağırlıklı "onset yok" hedefi. Mevcut tekrar ağırlığı yalnızca tepeyi
   ödüllendiriyor; çözümlemenin ihtiyaç duyduğu dibi öğretmiyor.
2. **`--onset-soft` düzeltmesi:** sonraki onset 2 kare içindeyse soft komşu değeri
   uygulanmaz. Şu anki hâliyle 2 karelik aralıklarda vadiyi dolduruyor.
3. **Parça düzeyinde fazla örnekleme:** <100 ms tekrar içeren 200 karelik parçalar
   daha sık örneklenir. 2 kareye ×3 kayıp ağırlığı, seyrek bir durumda fazla seyreliyor.
4. İnce ayar `tabcrnn_rep_off.pt`'den, yeni dosyaya (ör. `tabcrnn_rep_valley.pt`).

### Adım 5 — Mimari (Adım 1–4'ten sonra, gerekirse): doğrudan perde-onset kafası
- Ana ölçüt GAPS ama onset kafası tel başına çalışıyor ve GAPS'te yalnızca noisy-OR
  çarpımı üzerinden dolaylı öğreniyor. Klasik gitarda onset'in zayıf kalmasının
  olası nedeni bu (Katman 3.9'da onset F1 0.44).
- Ek bir **perde-onset kafası** (49 perde) eklenir; hem GuitarSet'te hem GAPS'te
  doğrudan denetlenir. Nota bulma bu kafayla yapılır, tel seçimi mevcut tab kafasında
  kalır (Onsets & Frames yapısı).
- Sıcak başlatma: mevcut ağırlıklar korunur, yeni kafa önsel bias ile başlar.

## 4. Etkilenecek dosyalar (öngörü)

| Dosya | Adım | Değişiklik |
|---|---|---|
| `gtab/decoding/viterbi.py` | 1, 2, 3 | yerel tepe seçici, noisy-OR birleştirme, sönüm telafili yükseliş, geniş bant akı, strum gruplama |
| `gtab/evaluation/pitch_eval.py` | 1, 3 | kalibrasyon araması, seçim parçası |
| `scripts/eval/diagnose_pitch.py` | 0 | bölüm F (tepe görünürlüğü) |
| `scripts/eval/diagnose_onset_features.py` | 2 | iki yeni özellik satırı |
| `gtab/data/tab_labels.py`, `gtab/models/losses.py`, `scripts/train/train_onset.py` | 4 | vadi hedefi, soft düzeltmesi, örnekleme |
| `gtab/models/nets.py` | 5 | perde-onset kafası |

## 5. Karar kuralları

**Adım 1–2 kabulü (eğitimsiz, tabcrnn_rep_off):**
- GAPS <100 ms kaçma %75 → **≤ %55** ve 100–200 ms kaçma %55 → **≤ %40**;
- GAPS nota F1 ≥ 0.679 (taban 0.689, −0.01'e kadar);
- GS solo tab F1 ≥ 0.750 (taban 0.760, −0.01'e kadar).

**Adım 4 kabulü (eğitim):** Adım 1–2 sonrası değerlere göre <100 ms kaçma −10 puan
ve GAPS nota F1 +0.01; GS tab F1 ≥ 0.750.

**Katman çıkışı ("ciddi iyileşme"):**
- GAPS <100 ms kaçma ≤ %40, 100–200 ms ≤ %35;
- GAPS nota F1 ≥ 0.709 (+0.02);
- val_comp nota / kare ≥ 0.780 / 0.830 (3.10'dan devreden ölçüt);
- GS solo tab F1 ≥ 0.750.

Tek seed farkları ±0.01 gürültü sayılır.

## 6. Denenmiş olanlarla çakışma kontrolü

| Daha önce | Bu plan |
|---|---|
| Hop 256 ve kısa pencere girişi: ayrımı artırmadı (AUC ~0.89) | Önerilmiyor. Adım 2'deki geniş bant akı farklı bir kanıt (perde harmonikleri değil, yüksek bant geçişi) ve önce ölçülecek. |
| Tekrar onset ağırlığı (3c): <100 ms değişmedi | Teşhis: çözümleme tavanı + vadi öğretilmemesi. Adım 1 ve 4.1 bunu hedefliyor. |
| Refrakter penceresi | Kaldırılmıyor; vadi derinliği ile birlikte aranıyor, refrakter 0 adayı her zaman deneniyor. |
| GAPS'te offset eşiği seçilmiyor (etiket bitişleri) | Bu plan offset'e dayanmıyor. |

## 7. Bu katmanın kapsamı dışında (not)
GAPS'teki asıl büyük kayıplar şunlar:
- kaçanların %45'inde perde hiç aktif değil;
- hayaletlerin %53'ü yanlış perde (harmonik, oktav, komşu ses).

Bunlar akustik model ve veri meselesi (SynthTab naylon ön-eğitimi, 36 bin/oktav,
GAPS partisyon TAB etiketleri). Katman 4'e geçmeden önce ayrı bir adım olarak
değerlendirilmeli.

## 8a. Uygulama (8 Ekim 2026) — Adım 0, 1 ve 2'nin ölçüm kısmı kodlandı

Yol planı ve sıra:

| Sıra | Adım | Durum | Ne zaman |
|---|---|---|---|
| 1 | Adım 0 — tepe görünürlüğü (`diagnose_pitch` bölüm F) | **kodlandı, ölçüm bekleniyor** | şimdi |
| 2 | Adım 2 ölçümü — `cqt_decay`, `hf_flux` satırları (`diagnose_onset_features --fast`) | **kodlandı, ölçüm bekleniyor** | şimdi (Adım 0 ile paralel) |
| 3 | Adım 1 — yerel tepe + noisy-OR kalibrasyonu (`eval_pitch`, `eval_hmm`) | **kodlandı, ölçüm bekleniyor** | şimdi |
| 4 | Adım 2 çözümlemeye alma | bekliyor | yalnız Adım 2 ölçümü CQT k=3'ü (GAPS AUC 0.889) belirgin geçerse |
| 5 | Adım 3 — strum gruplama + val_comp seçim parçası | bekliyor | **kullanıcı kararı** (değerlendirme kuralı) |
| 6 | Adım 4 — eğitim (vadi hedefi, soft düzeltmesi, örnekleme) | bekliyor | Adım 0 <100 ms görünürlüğü < %60 ise |
| 7 | Adım 5 — perde-onset kafası | bekliyor | Adım 1–4 sonrası gerekirse |

Kod:
- `gtab/decoding/viterbi.py`: `pick_peaks` (yerel tepe; `min_dist`, `prominence`), `pitch_matrix` /
  `pitch_onset_matrix` `combine='max'|'noisyor'`, `segment_notes_onset(peak_pick, min_dist, prominence)`;
  `rise_split`'in sabit 3 karesi `peak_pick` iken `min_dist`. `segment(..., combine=...)` dec üzerinden.
- `gtab/evaluation/pitch_eval.py`: `calibrate_peaks` — 3.10 seçiminin üstüne (a) noisy-OR, (b) `PEAK_GRID`
  (vadi {0.05, 0.1, 0.15, 0.25} × ara {1, 2, 3}) × refrakter {seçili, 0}, (c) yerel tepe seçildiyse
  yeniden vuruş / enerji kabul kapılarını kapatma, (d) onset eşiği {0.05…0.3} ve perde eşiği ±0.1.
  Her aday yalnızca doğrulama skorunu artırırsa alınır (3.10 seçimi her zaman aday).
  `eval_pitch` / `diagnose_pitch` `--no-peak-search` = 3.10 seçimi birebir.
- `scripts/eval/diagnose_pitch.py`: bölüm **F** (max / noisy-OR, vadi 0.10 / 0.25; IOI kırılımı;
  "kaçanlardan görünür" = çözümleme tavanı), `--limit`.
- `scripts/eval/eval_hmm.py`: `--combine --peak-pick --prominence --min-dist`.
- `scripts/eval/diagnose_onset_features.py`: `cqt_decay`, `hf_flux`, `decay+hf`; `--fast` (yalnız önbellek).

Testler:
- 70 ms senaryosu (tepeler 0.85 / 0.75, vadi 0.25): mevcut GAPS seçimi 1 nota; yerel tepe + refrakter 6 yine
  1 nota; **yerel tepe + refrakter 0 → 2 nota** ✓ (refrakter kapısı birlikte aranmalı — kalibrasyonda var);
  vadi 0.6 → 1 nota ✓.
- noisy-OR = `losses.pitch_onset_probs` / `pitch_probs` (maks fark 2e-7) ✓.
- Regresyon: `--no-dec-search --limit 3` (tabcrnn_poly) 0.557/0.634, 0.867/0.841 = HEAD ✓;
  tam kalibrasyon `--no-peak-search --limit 2` (tabcrnn_rep_off) HEAD (3e75366) ile birebir aynı seçim ve skor ✓.
- `diagnose_onset_features --fast` (val_comp, 3 kayıt) uçtan uca ✓.
- Duman (anlamlı değil, küçük örnek): `eval_pitch --limit 2` gaps_val noisy-OR'u seçti (doğrulama 0.629 → 0.641),
  yerel tepe seçilmedi; `diagnose_pitch --limit 3 --no-peak-search` bölüm F uçtan uca ✓
  (gaps_val tekrarsız notalarda görünürlük max %72 / noisy-OR %76).

### Çalıştırma (Adım 0 + 2 ölçümü + Adım 1)
```bash
# Adım 0: tepe görünürlüğü, 3.10 çözümlemesiyle (çözümleme tavanı) -> bölüm F
python -m scripts.eval.diagnose_pitch --ckpt tabcrnn_rep_off.pt --splits gaps_val val_comp --criterion mix --no-peak-search
# Adım 2 ölçümü (hızlı; ses okumaz)
python -m scripts.eval.diagnose_onset_features --splits gaps_val val_comp --fast
# Adım 1: yerel tepe + noisy-OR kalibrasyonu (3.10'dan ~30 ek aday; daha uzun sürer)
python -m scripts.eval.eval_pitch --ckpt tabcrnn_rep_off.pt --splits gaps_test val val_comp --criterion mix
python -m scripts.eval.diagnose_pitch --ckpt tabcrnn_rep_off.pt --splits gaps_val val_comp --criterion mix
# tab F1: [val] secim satırındaki değerlerle (yerel_tepe=True ise --peak-pick --prominence .. --min-dist ..)
python -m scripts.eval.eval_hmm --model crnn --ckpt tabcrnn_rep_off.pt --onset-thr <..> --off-ratio <..> --refractory <..> --fallback <..> [--peak] --reattack <..> --rise-keep <..> --rise-split <..> --offset-thr <..> --combine <..> [--peak-pick --prominence <..> --min-dist <..>]
```
Karar: §5 (Adım 1–2 kabulü) ve §3 Adım 0 karar dalı.

## 8b. Sonuçlar (8 Ekim 2026, tabcrnn_rep_off)

### Adım 0 — tepe görünürlüğü (3.10 çözümlemesiyle)
| gaps_val | n | max / vadi 0.10 | noisy-OR / 0.10 | noisy-OR / 0.25 | kaçan | kaçanlardan görünür |
|---|---|---|---|---|---|---|
| <100 ms | 647 | 60.7% | **66.6%** | 46.4% | 75.0% | **57.9%** |
| 100–200 ms | 1020 | 62.8% | 65.8% | 48.5% | 55.1% | 40.2% |
| ≥200 ms | 3444 | 76.3% | 79.8% | 62.5% | 26.9% | 34.5% |
| tekrarsız | 23380 | 73.8% | 76.9% | 59.3% | 26.0% | 24.3% |

val_comp <100 ms görünürlük %76, kaçanlardan görünür %71. **Karar dalı: <100 ms görünürlük %66.6 ≥ %60 →
sorun büyük ölçüde çözümlemede; Adım 4 (eğitim) şimdilik gerekmiyor.** noisy-OR görünürlüğü her grupta
max'tan 3–6 puan yüksek.

### Adım 2 — yeni enerji özellikleri (AUC, gaps_val / val_comp)
cqt k=3 0.889 / 0.893; cqt_k1 0.896 / 0.920; **cqt_decay 0.848 / 0.872; hf_flux 0.843 / 0.916;
decay+hf 0.884 / 0.935**. GAPS'te ikisi de mevcut CQT'nin **altında** → çözümlemeye alınmadı (kural §3 Adım 2).
Not: akorlarda decay+hf en iyi (0.935; tıngırtının geniş bant izi) → Adım 3 (strum) için aday kanıt.

### Adım 1 — yerel tepe + noisy-OR kalibrasyonu
Seçim: gaps_val → noisy-OR, yerel tepe (vadi 0.1, ara 2), onset@0.05, eşik 0.4, refrakter 6, yeniden_vuruş **0**,
enerji 6/8, offset yok. val → noisy-OR, yerel tepe (vadi 0.05, ara 3), onset@0.2, eşik 0.7, refrakter 6,
yeniden_vuruş **0**, enerji 6/0, offset 0.5.

| nota / kare F1 | 3.10 (rep_off) | **Adım 1** |
|---|---|---|
| gaps_test | 0.689 / 0.675 (P 0.629, R 0.761) | **0.736 / 0.676** (P 0.731, R 0.741) |
| GS val | 0.915 / 0.871 | 0.922 / 0.873 |
| val_comp | 0.769 / 0.817 (P 0.725, R 0.819) | 0.768 / 0.822 (P 0.834, R 0.711) |

| hızlı tekrar kaçma | gaps_val 3.10 → Adım 1 | val_comp 3.10 → Adım 1 |
|---|---|---|
| tümü | 38.6% → **43.0%** | 15.2% → **42.9%** |
| <100 ms | 75.0% → 78.5% | 72.5% → 85.0% |
| 100–200 ms | 55.1% → 64.1% | 32.5% → 59.3% |

Yorum:
- GAPS nota F1 **+0.047** (0.736; katman çıkış hedefi 0.709 aşıldı) — ama kazanç **precision**'dan
  (hayalet 10376 → 6252, "farklı perde" hayaletleri dahil), tekrarlardan değil.
- **Tekrarlar kötüleşti.** Neden: kalibrasyon (c) aşaması yeniden vuruş eşiğini 0'a çekti (precision kazancı),
  refrakter 6 kaldı. Böylece nota sürerken gelen tekrar yalnız `enerji ≥ 6 dB` ile kabul ediliyor; gömülü
  tekrarların medyan yükselişi 3.3 dB → çoğu reddediliyor (val_comp kaçanların %79'u önceki notaya gömülü;
  ≥200 ms kaçanların %86'sının tepesi görünür). Yerel tepe seçicinin ürettiği adaylar eski koşu-tabanlı
  kapılardan (refrakter, enerji kabul) geçemiyor.
- Ölçüt (mix) tüm notalara bakıyor; tekrarlar notaların küçük bir kısmı olduğundan, kalibrasyon tekrarları
  precision'a feda etti.
- Adım 1–2 kabulü (§5): GAPS nota ≥ 0.679 ✓; <100 ms ≤ %55 ✗ (78.5), 100–200 ms ≤ %40 ✗ (64.1);
  GS solo tab F1 ≥ 0.750 ✓.

**GS solo tab F1 (eval_hmm, val seçimiyle: noisy-OR, yerel tepe vadi 0.05 / ara 3, yeniden_vuruş 0):**
greedy **0.760** (eşik 0.8), Viterbi 0.760 (w_tr 0) — 3.10 ile aynı (0.760 / 0.760); oracle tel 0.862.
P 0.747 / R 0.773 (3.10: 0.741 / 0.779) → tel tarafı nötr, precision lehine küçük kayma.

**Karar (8 Ekim):** Adım 1 çözümlemesi **GAPS için kabul** (nota F1 +0.047, hiçbir ölçüt düşmedi; tab F1 nötr).
Katmanın hedefi olan hızlı tekrarlar ise kötüleşti → Adım 1b gerekli. Mevcut çıkış ölçütleri (§5):
GAPS nota ≥ 0.709 ✓ (0.736), GS tab F1 ≥ 0.750 ✓, <100 ms ≤ %40 ✗ (78.5), 100–200 ms ≤ %35 ✗ (64.1),
val_comp ≥ 0.780 / 0.830 ✗ (0.768 / 0.822).

### Önerilen sıradaki adım: Adım 1b — yerel tepeye uygun tekrar kapıları (eğitimsiz)
Sorun: yerel tepe seçicinin bulduğu tekrar adayları eski koşu-tabanlı kapılardan (refrakter 6, yeniden_vuruş 0
→ yalnız enerji ≥ 6 dB) geçemiyor; gömülü tekrarların medyan enerji yükselişi 3.3 dB.
1. **Vadi kanıtı:** nota sürerken gelen tepe, önceki tepeyle arasındaki vadi derinliği ≥ `re_valley` ise kabul
   (yerel tepe zaten bunu ölçüyor; enerji/onset tepe değeri yerine).
2. **Ortak arama:** yerel tepe seçiliyken refrakter {0, 3, 6} × yeniden_vuruş {0, 0.2, 0.3} × enerji_kabul {0, 3, 6}
   × re_valley {0, 0.15, 0.25} birlikte (tek tek değil).
3. **Seçim ölçütü (değerlendirme kuralı değil, aday filtresi):** doğrulama skoru en iyinin en çok 0.01 altında
   kalan adaylar arasından doğrulamada hızlı tekrar kaçma oranı en düşük olan. Böylece bugünkü precision
   kazancı korunur, tekrarlar öne çıkar.
Kabul: §5 Adım 1–2 kuralı (<100 ms ≤ %55, 100–200 ms ≤ %40, GAPS nota ≥ 0.726, GS tab F1 ≥ 0.750).
Tutmazsa → Adım 4 (vadi hedefli eğitim).

## 8c. Adım 1b — vadi kanıtlı tekrar kapıları (9 Ekim 2026, kodlandı)

Kod:
- `segment_notes_onset(re_valley=...)` (yalnız `peak_pick`): nota sürerken gelen tepe, önceki tepeyle arasındaki
  vadi derinliği (alçak tepe − vadi dibi) ≥ `re_valley` ise yeni nota. Yeniden vuruş ve enerji kabul kapılarına
  **VEYA** ile eklenir; 0 = kapalı (Adım 1 birebir).
- `track_scores` / `aggregate`: hızlı tekrar sayımı (önceki aynı perde notası ≤ 3 kare önce bitti; IOI <100 ve
  100–200 ms) → `rep`, `rep_fast` (diagnose_pitch bölüm E ile aynı tanım).
- `calibrate_repeats` (Adım 1'den sonra, yalnız yerel tepe seçildiyse): `REP_GRID` = refrakter {0, 3, 6} ×
  yeniden_vuruş {0, 0.3} × enerji_kabul {0, 6} × re_valley {0, 0.15, 0.25} (36 aday + mevcut seçim) **birlikte**
  aranır. Doğrulama skoru en iyinin en çok `--rep-tol` (0.01) altında kalan adaylar arasından doğrulamadaki
  hızlı tekrar (IOI < 200 ms) kaçma oranı en düşük olan seçilir. Test setine bakılmaz; değerlendirme kuralı değişmez.
  `--no-rep-search` = Adım 1 seçimi birebir; `--rep-tol 0` = saf skorla ortak arama.
- `eval_pitch` özet tablosunda test setleri için **tekrar kaçma <100 / 100–200 ms** sütunu; seçim satırında
  `vadi_kaniti`. `eval_hmm --re-valley`. `diagnose_pitch --no-rep-search --rep-tol`.

Testler:
- 70 ms senaryosu, Adım 1 GAPS kapıları (yeniden_vuruş 0, enerji 6) + gömülü tekrar (enerji yok): vadi kanıtı yokken
  1 nota; **re_valley 0.25 → 2 nota** ✓; re_valley 0.6 → 1 nota ✓; peak_pick kapalıyken etkisiz ✓.
- Tekrar sayımı = diagnose_pitch bölüm E (gaps_val 3 kayıt: <100 ms n=3 %66.7, 100–200 ms n=11 %36.4) ✓.
- `calibrate_repeats` duman (gaps_val 3 kayıt, Adım 1 GAPS seçimi): skor 0.684 → 0.680 (tol içinde), hızlı tekrar
  kaçma %42.9 → %35.7 (refrakter 6 → 3); tol=0 → Adım 1 seçimi korunuyor ✓. (Örnek küçük: 14 tekrar.)
- Regresyon: `eval_pitch --limit 2` (yerel tepe seçilmeyen durum) önceki duman ile birebir (0.666/0.668, 0.893/0.854) ✓.

### Çalıştırma (Adım 1b)
```bash
python -m scripts.eval.eval_pitch --ckpt tabcrnn_rep_off.pt --splits gaps_test val val_comp --criterion mix
python -m scripts.eval.diagnose_pitch --ckpt tabcrnn_rep_off.pt --splits gaps_val val_comp --criterion mix
# tab F1: [val] secim satırındaki değerlerle
python -m scripts.eval.eval_hmm --model crnn --ckpt tabcrnn_rep_off.pt --onset-thr <..> --off-ratio <..> --refractory <..> --fallback <..> [--peak] --reattack <..> --rise-keep <..> --rise-split <..> --offset-thr <..> --combine <..> --peak-pick --prominence <..> --min-dist <..> --re-valley <..>
```
İsteğe bağlı ablasyon (tekrar odaklı seçimin katkısı): aynı eval_pitch komutuna `--rep-tol 0`.

Kabul (§5 Adım 1–2): GAPS <100 ms kaçma ≤ %55, 100–200 ms ≤ %40, GAPS nota F1 ≥ 0.726 (Adım 1: 0.736 − 0.01),
GS tab F1 ≥ 0.750. Tutmazsa → Adım 4 (vadi hedefli eğitim).

## 8d. Adım 1b — sonuçlar (9 Ekim 2026)
(Kullanıcı koşusu `cihaz: cpu` — eski .venv; sayılar cihazdan bağımsız. .venv'e CUDA torch kuruldu.)

Seçim: gaps_val → refrakter **0**, yeniden_vuruş 0, enerji_kabul 6, **vadi_kanıtı 0 (seçilmedi)**, yerel tepe (0.1 / 2);
val → refrakter **0**, yeniden_vuruş **0.3**, enerji_kabul **0**, vadi_kanıtı 0, yerel tepe (0.05 / 3).

| nota / kare F1 | 3.10 | Adım 1 | **Adım 1b** |
|---|---|---|---|
| gaps_test | 0.689 / 0.675 | 0.736 / 0.676 | 0.733 / 0.676 |
| GS val | 0.915 / 0.871 | 0.922 / 0.873 | 0.908 / 0.874 |
| val_comp | 0.769 / 0.817 | 0.768 / 0.822 | 0.766 / 0.822 (P 0.699, R 0.845) |

| hızlı tekrar kaçma <100 / 100–200 ms | 3.10 | Adım 1 | **Adım 1b** |
|---|---|---|---|
| gaps_val (diagnose) | 75.0 / 55.1 | 78.5 / 64.1 | 72.6 / 62.9 |
| gaps_test (yeni sütun) | — | — | 76.2 / 43.8 |
| val_comp | 72.5 / 32.5 | 85.0 / 59.3 | **22.5 / 17.7** |

Doğrulama (gaps_val 5 kayıt, GAPS seçimi + re_valley):
| re_valley | skor | nota P | nota F1 | tekrar kaçma <100 / 100–200 |
|---|---|---|---|---|
| 0 (seçilen) | 0.699 | 0.760 | 0.727 | 66.7 / 57.7 |
| 0.15 | 0.651 | 0.558 | 0.629 | 55.6 / 42.0 |
| 0.25 | 0.674 | 0.647 | 0.677 | 55.6 / 51.8 |
| 0.35 | 0.688 | 0.706 | 0.704 | 66.7 / 54.4 |

Yorum:
- **GAPS (fingerstyle):** vadi kanıtı tekrarları yakalıyor (100–200 ms −16 puan) ama precision çöküyor (0.76 → 0.56):
  modelin onset eğrisinde **sahte bölünmelerin (fragman) vadisi gerçek tekrarın vadisine benziyor**. Çözümleme
  ikisini ayıramıyor → tol (0.01) içinde kalan aday yok, GAPS tekrarları neredeyse değişmedi. Eğitimsiz yol GAPS
  için tükendi.
- **GuitarSet akor:** tekrar kaçma büyük ölçüde düştü (<100 ms 72.5 → 22.5, 100–200 ms 32.5 → 17.7; kapı
  gevşedi, refrakter 0, enerji kapısı kalktı) ama akor hayaletleri arttı (P 0.834 → 0.699, fragman 1225 → 1507);
  nota F1 değişmedi. GS val nota F1 −0.014 (0.922 → 0.908; tol mix skorunda −0.006).
- Kabul (§5): GAPS nota ≥ 0.726 ✓ (0.733); GAPS <100 ms ≤ %55 ✗, 100–200 ms ≤ %40 ✗; GS tab F1 bekleniyor.

**GS solo tab F1 (val seçimiyle, refrakter 0, yeniden_vuruş 0.3, enerji 0):** greedy **0.760**, Viterbi 0.760
(P 0.747 / R 0.774) — Adım 1 ve 3.10 ile aynı → ≥ 0.750 ✓. Kare düzeyinde tel ataması etkilenmedi; GS val nota F1
düşüşü (−0.014) tab F1'e yansımadı. **Karar: 1b seçimi kabul** (akor tekrarlarında büyük kazanç, tab F1 nötr;
GAPS seçimi Adım 1 ile fiilen aynı).

**Karar dalı → Adım 4 (eğitim).** Gerekçe ölçüldü: sorun artık çözümleme kapıları değil, modelin gerçek tekrarı
sahte bölünmeden ayıran bir onset vadisi vermemesi. Adım 4.1 (iki yakın aynı perde onset'i arasına "onset yok"
hedefi + fragman yerlerinde negatif) tam bunu öğretir.

## 8e. Adım 4 — vadi hedefli eğitim (9 Ekim 2026, kodlandı)

Gerekçe (§8d): çözümleme gerçek tekrarı sahte bölünmeden ayıramıyor; ikisinin onset vadisi aynı görünüyor.
Model ikisini ancak ikisini de görürse ayırmayı öğrenir → iki negatif türü + örnekleme (kullanıcı planı):

| Bileşen | Ne yapar | Kod |
|---|---|---|
| **Vadi hedefi** `--valley-weight` | hızlı aynı perde tekrarında (önceki nota ≤ 3 kare önce bitti, iki onset arası ≤ 8 kare ~186 ms) iki onset ARASINDAKİ karelere "vuruş yok" (hedef 0, soft komşu değeri de silinir) × ağırlık | `tab_labels.valley_frames` |
| **Fragman negatifleri** `--frag-weight` | init modelin (`tabcrnn_rep_off`) eğitim verisinde etiket notası sürerken tepe verdiği ama etikette **±2 karede vuruş olmayan** yerlere (tepe ±1 kare) "vuruş yok" × ağırlık; eğitim başında bir kez taranır, önbellek yazılmaz | `tab_labels.fragment_frames`, `train_onset.fragment_fns` |
| **Örnekleme** `--oversample` | hızlı tekrar içeren 200 karelik parçalar bu kat sık örneklenir (`WeightedRandomSampler`, epoch boyu aynı) | `train_onset.make_loader`, `ds.fast` |

GuitarSet'te tel düzeyinde (tel onset eğrisi), GAPS'te perde düzeyinde (noisy-OR perde-onset eğrisi) uygulanır.
Varsayılanlar (1) = önceki eğitim, birebir.

Testler:
- Birim: vadi yalnız hızlı aynı perde tekrarının iki onset'i arasında (IOI 12 kare ve farklı perde hariç) ✓;
  fragman yalnız nota içindeki, etiket onset'ine > 2 kare uzak tepede (onset'e yakın tepe ve sessizlik hariç) ✓.
- **Varsayılan = HEAD (0b9fc20):** GuitarSetSeq (val_comp) ve PitchSeq (gaps_val 3 kayıt) onset hedefleri,
  onset ağırlıkları, kısa nota ağırlıkları birebir ✓.
- Vadi karelerinde hedef 0 ✓ (val_comp 2323 vadi karesi). Fragman oranı: train_comp[:20] aktif tel karelerinin
  %6.0'ı (~1500 tepe / 6098 onset), gaps_train[:5] aktif perde karelerinin %5.3'ü (~1080 tepe / 3072 onset).
  Hızlı tekrar içeren parça: val %15, gaps_test %8.
- 1 epoch duman eğitimi (küçük veri, tüm seçenekler açık) uçtan uca ✓.

**Risk:** GAPS'te fragman tepeleri onset sayısının ~%35'i; partisyonda eksik kalmış gerçek tekrarlar da negatif
olabilir (±2 kare toleransı yalnız zamanlama kaymasını kapsar). Precision düşer ve recall artarsa (ya da GAPS
tekrarları kötüleşirse) ilk ayar `--frag-weight` düşürmek.

### Çalıştırma (Adım 4)
```bash
python -m scripts.train.train_onset --init tabcrnn_rep_off.pt --epochs 15 --ckpt tabcrnn_rep_valley.pt --onset-soft 0.3 --gs-pitch-weight 1.0 --short-frames 5 --short-weight 2.0 --repeat-weight 3 --pitch-weight 2 --offset-weight 1 --valley-weight 5 --frag-weight 3 --oversample 3
python -m scripts.eval.eval_pitch --ckpt tabcrnn_rep_valley.pt --splits gaps_test val val_comp --criterion mix
python -m scripts.eval.diagnose_pitch --ckpt tabcrnn_rep_valley.pt --splits gaps_val val_comp --criterion mix
# tab F1: [val] secim satırındaki değerlerle
python -m scripts.eval.eval_hmm --model crnn --ckpt tabcrnn_rep_valley.pt --onset-thr <..> --off-ratio <..> --refractory <..> --fallback <..> [--peak] --reattack <..> --rise-keep <..> --rise-split <..> --offset-thr <..> --combine <..> --peak-pick --prominence <..> --min-dist <..> --re-valley <..>
```

Karar (§5 Adım 4, taban = rep_off + 1b çözümlemesi):
- Kabul: GAPS <100 ms kaçma **−10 puan** (gaps_val 72.6 → ≤ 62.6; gaps_test 76.2 → ≤ 66.2) **ve** GAPS nota F1
  **+0.01** (0.733 → ≥ 0.743); GS solo tab F1 ≥ 0.750.
- Ardından Katman 3.11 kapanışı ve **Katman 4'e geçiş** (kullanıcı kararı, 9 Ekim): Adım 4 sonucuyla katman çıkış
  ölçütleri (§5) raporlanır; sağlanmayanlar açık madde olarak yazılır. Adım 5 (perde-onset kafası) yalnız gerekirse,
  Katman 4'ten sonra.

## 8f. Adım 4 — sonuçlar (9 Ekim 2026, `tabcrnn_rep_valley.pt`)

Eğitim: 15 epoch; hızlı tekrar içeren parça GuitarSet 934/2110, GAPS 1542/8785 (×3 örneklendi). Seçim skoru
0.564 (init) → 0.588; GS tab F1 0.692 → 0.704, GAPS kare F1 0.612 → 0.633, GAPS onset F1 0.325 → 0.375
(eğitim içi ölçüler).

Seçim: gaps_val → **koşu tabanlı** (yerel tepe seçilmedi), onset@0.1, eşik 0.4, refrakter 6, yeniden_vuruş 0.3,
enerji 6/8, noisy-OR. val → yerel tepe (0.05 / 3), onset@0.05, eşik 0.8, refrakter 0, yeniden_vuruş 0.3,
enerji_böl 8, offset 0.7, noisy-OR.

| | rep_off + 1b | **rep_valley** | fark |
|---|---|---|---|
| gaps_test nota / kare F1 | 0.733 / 0.676 (P 0.725 R 0.742) | 0.720 / 0.684 (P 0.666 R 0.784) | −0.013 / +0.008 |
| gaps_test tekrar kaçma <100 / 100–200 | 76.2 / 43.8 | 77.7 / **36.8** | +1.5 / **−7.0** |
| gaps_val tekrar kaçma <100 / 100–200 / ≥200 | 72.6 / 62.9 / 30.1 | 73.0 / **52.7** / **24.2** | 0 / **−10** / −6 |
| gaps_val kaçan / hayalet | 29.7% / 25.0% | **25.9%** / 30.3% | recall ↑, precision ↓ |
| GS val nota / kare | 0.908 / 0.874 | **0.916 / 0.880** | +0.008 / +0.006 |
| val_comp nota / kare | 0.766 / 0.822 | **0.783** / 0.818 | +0.017 / −0.004 |
| GS solo tab F1 (greedy / Viterbi) | 0.760 / 0.760 | **0.769 / 0.771** | **+0.009 / +0.011** |
| oracle tel doğruluğu | 0.862 | 0.869 | +0.007 |
| tepe görünürlüğü <100 ms (gaps_val, noisy-OR) | 66.5% | 65.1% | değişmedi |

Yorum:
- **100–200 ms tekrarlar düzeldi** (gaps_val −10, gaps_test −7 puan), ≥200 ms de (−6). Vadi + örnekleme bu
  bandda çalıştı.
- **<100 ms tekrarlar hiç değişmedi** — ne çözümlemede ne eğitimde; tepe görünürlüğü de aynı (%65). Bu bant
  (≤ 4 kare) için model ayrı tepe üretmeyi öğrenmedi. Aday nedenler: GAPS partisyon etiketlerinin zamanlaması
  (eksik / kayık tekrarlar) ve 23 ms kare ile tel düzeyindeki dolaylı onset öğrenimi. Kalan aday: Adım 5
  (doğrudan perde-onset kafası) — Katman 4'ten sonra, gerekirse.
- GAPS'te recall arttı, precision düştü (fragman negatiflerine rağmen hayalet arttı); nota F1 −0.013 (gürültü
  sınırının biraz dışında), kare F1 +0.008; mix ölçütü ≈ aynı (0.7045 → 0.702).
- **Proje ana ölçütü tab F1 en iyi değerinde (0.771)**; val_comp nota F1 ilk kez hedefi (0.780) geçti.

Kabul (§5 Adım 4): <100 ms −10 puan ✗; GAPS nota +0.01 ✗ (−0.013); GS tab F1 ≥ 0.750 ✓ → **kabul kuralı
sağlanmadı.**

Katman 3.11 çıkış ölçütleri (§5): GAPS <100 ms ≤ %40 ✗ (77.7) | 100–200 ms ≤ %35 ✗ (36.8, yakın) |
GAPS nota ≥ 0.709 ✓ (0.720) | val_comp ≥ 0.780 / 0.830 → nota ✓ 0.783, kare ✗ 0.818 | GS tab F1 ≥ 0.750 ✓ (0.771).

**Öneri:** Katman 3.11 bu durumla kapanır (kullanıcı kararı: sonuçtan sonra Katman 4). Katman 4'e taşınacak taban
için öneri `tabcrnn_rep_valley.pt` (tab F1, GS nota/kare, val_comp nota, 100–200 ms tekrarlar daha iyi; GAPS nota
−0.013 tek kayıp). Alternatif: GAPS nota F1 öncelikliyse `tabcrnn_rep_off.pt` + 1b. İki checkpoint de korunuyor.
Açık maddeler Katman 4'e devreder: <100 ms aynı perde tekrarları; val_comp kare F1 (0.818 / 0.830).

## 8. Durum
- [x] Kod okuma teşhisi (1a, 1b)
- [x] Sentetik duman testi (3 seed): çözümleme tavanı doğrulandı, yerel tepe + noisy-OR <100 ms yakalamayı ~2.4×, 100–200 ms'yi ~3× artırdı
- [x] Adım 0 — bölüm F kodu + testler
- [x] Adım 0 — tepe görünürlüğü ölçümü: <100 ms %66.6 (≥ %60) → çözümleme dalı; eğitim şimdilik gerekmiyor
- [x] Adım 1 — yerel tepe seçici + noisy-OR + kalibrasyon: kod + testler + regresyon
- [x] Adım 1 — ölçüm: GAPS nota 0.689 → 0.736 (precision), tekrarlar kötüleşti (kapılar)
- [x] Adım 1 — GS tab F1 (eval_hmm): 0.760 / 0.760 (nötr) → Adım 1 çözümlemesi GAPS için kabul
- [x] Adım 1b — vadi kanıtı + ortak arama + tekrar odaklı seçim: kod + testler + regresyon
- [x] Adım 1b — ölçüm: GAPS tekrarları değişmedi (vadi kanıtı precision'ı çökertiyor), akor tekrarları büyük düşüş
- [x] Adım 1b — GS tab F1 0.760 (nötr) → 1b seçimi kabul
- [x] Adım 4 — vadi hedefi + fragman negatifleri + örnekleme: kod + testler (§8e)
- [x] Adım 4 — eğitim + ölçüm: tab F1 0.771 (en iyi), 100–200 ms tekrarlar −10 puan, <100 ms değişmedi, GAPS nota −0.013 → kabul kuralı sağlanmadı (§8f)
- [x] Katman 3.11 kapanışı: kullanıcı Katman 3.12'ye (akustik katman, README13) geçti; taban `tabcrnn_rep_valley.pt`
- [x] Adım 2 — `cqt_decay`, `hf_flux` ölçüm satırları: kod
- [x] Adım 2 — AUC ölçümü: GAPS'te mevcut CQT'nin altında → çözümlemeye alınmadı
- [ ] Adım 3 — strum gruplama; seçim parçası kararı (kullanıcı)
- [ ] Adım 4 — eğitim (koşullu)
- [ ] Adım 5 — perde-onset kafası (koşullu)
