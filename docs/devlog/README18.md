# Katman 3.13 — P3: Fiziksel Çalınabilirlik Analizi

> **Durum (11 Ekim 2026):** P3 tamamlandı ve **son doğrulamadan geçti**. Yalnız ölçüm: model,
> decoder, checkpoint, önbellek ve veri değişmedi; hiçbir kısıt uygulanmadı, model seçimi
> yapılmadı. Dal `Armagan-katman-4`, HEAD `e25b5fa`; commit/push yapılmadı. Sıradaki (onay
> bekliyor): **P4** — akor içi parmak kısıtı ve el pozisyonu çıktısı.
> Önceki fazlar: `README16.md` (P1), `README17.md` (P2).

## 1. Başlangıç durumu

| | Değer |
|---|---|
| Parmaklandırma modeli | `checkpoints/edition_seq.pt` (Katman 3.13 Adım 2b) — değiştirilmedi |
| Test tabanı | 129 test (P2 sonunda), hepsi geçiyor |
| Çalınabilirlik denetimi | **yok** — sert kısıt yalnız enstrüman aralığı, akor içi tel teklik, boş tel ⇔ parmak 0 |
| Sunumun istediği metrikler (slayt 12) | "rule violation, hand movement cost, stretch / position shifts" — **hiçbiri ölçülmemişti** |

## 2. Metrik tasarım kararları ve gerekçeleri

**K1 — Hüküm değil, sayım.** Metrikler akoru reddetmez, düzeltmez; sayar ve sınıflandırır.
Gerekçe: klasik gitarda "imkânsız" ile "zor" sınırı icracıya ve el yapısına bağlı.

**K2 — Üç sertlik düzeyi karıştırılmaz.** KESİN fiziksel sorun (A) · ERGONOMİK RİSK (C) ·
YAKLAŞIK ölçü (D).

**K3 — Barre hata değil.** Aynı parmak + aynı perde + çok tel = barre adayı (E); geçerliliği
belirsiz durumlar ("teller bitişik değil", "arada daha düşük perde var") ayrıca işaretlenir.

**K4 — Boş teller ve eksik etiketler hesaba girmez.** Boş tel parmak tutmaz (parmak 0); konumu
(`perde < 0`) ya da parmağı bilinmeyen nota **tahmin edilmez** → o akor "değerlendirilemez".

**K5 — Eşik dayatma yok.** El açıklığı 3/4/5/6 yarım ses eşiklerinde **dağılım** olarak verilir.

**K6 — Uzun boşluk serbest taşıma.** Pozisyon değişimi `LONG_GAP_SEC = 0,5 sn`'den uzun
aralıklarda ayrı sayılır.

**K7 — ZAMAN ÖRTÜŞMESİ zorunlu (son doğrulamada eklendi).** `order_notes` akorları yalnız
**başlangıç** yakınlığına göre gruplar (≤ 30 ms); başlangıcı yakın iki nota aynı anda basılı
olmak zorunda değil. Fiziksel ihlal ancak notalar birlikte tutuluyorken anlamlıdır, bu yüzden
A/B/C/E yalnız **zamanda örtüşen çiftlerde** (`min(bitiş) − max(başlangıç) > 0`) hesaplanır.
Gruplama kodu değiştirilmedi; örtüşme ölçüm modülünde denetleniyor. Örtüşme gözetmeyen eski
tanım karşılaştırma için `group_span()` / `group_finger_conflicts()` olarak korunuyor.

**K8 — Üç ayrı payda (son doğrulamada netleştirildi).** Yöntemlerin kapsamı farklı olduğu için
her metrik üç blokta verilir: **kendi kapsamı** (paydalar farklı) · **editör alt kümesi** ·
**gerçek kesişim** (üç yöntemin de değerlendirilebilir olduğu akorlar; PAYDA-1 aynı).
A/B/C/E oranlarının paydası **PAYDA-3** = örtüşen basılı çifti olan akorlar (tek notalı akorlarda
bu metrikler tanımsızdır).

## 3. Geliştirilen kod ve testler

| Dosya | Satır | İçerik |
|---|---|---|
| `gtab/decoding/playability.py` | 327 | `ChordShape` (zamanlarla), `overlap` / `overlap_pairs` / `nonoverlap_pairs`; `finger_conflicts` (A), `hand_span` (B), `finger_order_inversions` (C), `hand_position` / `position_spread` / `position_shifts` (D), `barre_detail` (E); eski tanımlar `group_span` / `group_finger_conflicts`; `evaluate_shapes` + `PlayabilityReport.rates()` |
| `scripts/eval/eval_playability.py` | 245 | Üç yöntem × üç payda; örtüşme tanılaması; aşırı açıklık doğrulaması; en sorunlu akorlar. Hiçbir şey yazmaz |
| `tests/test_playability.py` | 285 | **41 test** (sentetik; veri seti/model gerektirmez) |

**Bağımsızlık:** `edition.py` / `edition_net.py` bu modülü import etmiyor; ürün çıkarım yolu
değişmedi. Modül yalnız `transitions.LONG_GAP_SEC` sabitini okuyor.

**Testler neyi sabitliyor:** örtüşmeyen aynı parmağın çakışma sayılmaması; sınırda (örtüşme = 0)
çakışma olmaması; kısmi örtüşmede yalnız örtüşen çiftin sayılması; eski grup tanımının ayrı
korunması; boş telin hesaba girmemesi; barre'nin çakışma sayılmaması ve örtüşmeyen barre'nin
sayılmaması; aynı parmağın üç perdesinde 3 çakışan çift (grup tanımı 2); açıklığın yalnız basılı
notalardan; ters sıranın barre çiftini saymaması; uzun boşluğun ayrılması; eksik parmak **ve**
eksik konumun akoru değerlendirilemez yapması; gerçek veri biçiminde (başlangıç, **bitiş**, perde)
örtüşmeyen grup; pay/payda oranları; determinizm; boş girdi.

## 4. Deney koşulları

- **Split:** `gaps_val`. `gaps_test` script tarafından **reddediliyor**
  (`protocol.assert_may_select` → "secim/kalibrasyonda kullanilamaz"). Dosya listesi
  `gaps_training_files()` → karantina `395_w41wc` düşer.
- **Veri:** 28 `gaps_val` kaydından **15'i** `gaps_tab` etiketli → **15.982 nota, 12.842 akor grubu**.
  Grup büyüklüğü: 10.782 tek notalı, 1.300 ikili, 524 üçlü, 162 dörtlü, 68 beşli, 3 altılı,
  3 grup 6'dan büyük.
- **Yöntemler:** `model` (BiLSTM, konumu kendi seçer) · `naif` (en düşük perde +
  `finger_baseline`) · `editor` (partisyon; **kusursuz varsayılmadı**).
- **Paydalar:** editörün parmağı tam bilinen akor **3.419 (%26,6)**;
  **gerçek kesişim (üç yöntem de değerlendirilebilir) = 2.963 akor.**

## 5. Üç yöntemin karşılaştırmalı sonuçları

### 5.1 Gerçek kesişim — ana tablo (2.963 akor; PAYDA-1 üç yöntemde aynı)

| metrik | model | naif | editor |
|---|---|---|---|
| PAYDA-1 değerlendirilebilir akor | 2.963 | 2.963 | 2.963 |
| PAYDA-2 basılı nota | 3.257 | 3.180 | 3.264 |
| PAYDA-3 örtüşen basılı çifti olan akor | 223 | 186 | 232 |
| örtüşen çift / örtüşmeyen çift | 366 / 1 | 247 / 1 | 371 / 1 |
| **A) çakışma (örtüşen çift; KESİN)** | **22** | **25** | **0** |
| **A) çakışmalı akor oranı (PAYDA-3)** | **%8,52** | **%13,44** | **%0,00** |
| B) açıklık medyan / maks | 1 / 6 | 2 / 9 | 1 / **3** |
| B) açıklık ≥ 3 yarım ses | %14,80 | %36,02 | %6,90 |
| B) açıklık ≥ 4 yarım ses | %3,59 | **%27,42** | **%0,00** |
| B) açıklık ≥ 6 yarım ses | %0,90 | %12,37 | %0,00 |
| C) ters parmak sırası | 5 (%2,24) | 6 (%3,23) | **0 (%0,00)** |
| E) barre (temiz / belirsiz) | 20 / 4 | 13 / 26 | 3 / 12 |
| D) geçiş / kayma (\|Δh\| ≥ 1) | 2.949 / 1.612 | 2.949 / 939 | 2.949 / 1.391 |
| D) ortalama hareket (yarım ses) | 0,99 | 0,94 | 0,94 |
| D) uzun boşluklu geçiş | 764 | 764 | 764 |

### 5.2 Kendi kapsamı (paydalar FARKLI — kesişimle karıştırılmamalı)

| metrik | model | naif | editor |
|---|---|---|---|
| PAYDA-1 değerlendirilebilir akor | 9.582 | 9.127 | 3.419 |
| etiketi eksik → değerlendirilemez | 1 | 1 | 7.287 |
| basılı notası yok (yalnız boş tel) | 3.259 | 3.714 | 2.136 |
| PAYDA-3 örtüşen çifti olan akor | 1.260 | 1.130 | 235 |
| A) çakışma | 126 (%9,21) | 161 (%13,45) | 0 (%0,00) |
| B) açıklık medyan / maks | 1 / **22** | 1,5 / 13 | 1 / 3 |
| C) ters parmak sırası | 104 (%6,43) | 42 (%3,36) | 0 |
| E) barre (temiz / belirsiz) | 106 / 66 | 181 / 145 | 3 / 12 |

Editör alt kümesi (3.419 akor) sonuçları kesişimle neredeyse aynı (model A = 22, naif 25,
editör 0); fark yalnız paydalarda (model 3.043, naif 2.963, editör 3.419 değerlendirilebilir akor).

### 5.3 Yorum

1. **Editörün parmaklandırması kurallarımızı hiç ihlal etmiyor:** 0 çakışma, 0 ters sıra,
   örtüşen tanımda açıklık en fazla **3** yarım ses. Kural kümesi **fazla katı değil** — gerçek
   editöryal pratikle uyumlu. (Editör kusursuz varsayılmadı; ölçüm gösterdi.)
2. **Model fiziksel olarak imkânsız akorlar üretiyor:** kesişimde 22 çakışan çift, örtüşen çifti
   olan akorların **%8,52'sinde**. Sebebi kodda görünüyor: parmaklar akor düzeyinde değil nota
   başına argmax ile seçiliyor (`edition_net.py:114`); tellerde var olan çakışma çözümünün
   (`resolve_chords`) parmak karşılığı yok. **P4'ün ana hedefi budur.**
3. **Açıklıkta model editöre yakın, naif çok uzak:** ≥4 yarım ses oranı editörde %0, modelde
   %3,59, naifte **%27,42**. "En düşük perde" kuralı el şeklini hiç gözetmiyor.
4. **Modelin en kötü şekilleri editörün etiketlemediği bölgede:** kendi kapsamında açıklık
   maksimumu **22** yarım ses; kesişimde 6. Yani kesişim sayıları modelin **iyimser** tarafıdır.
5. **Hareket maliyeti:** kesişimde üç yöntem aynı 2.949 geçişte ölçüldü; kayma sayısı model 1.612,
   editör 1.391, naif 939; ortalama hareket 0,99 / 0,94 / 0,94. Model editörden biraz fazla taşıyor.
6. **Ters sıra ayırt edici değil** (5–6 akor); asıl sinyal A ve B'de.
7. **Barre belirsizliği kuralı temkinli:** editörün 15 barre adayının 12'si "belirsiz" çıkıyor
   (kısmi barre = bitişik olmayan teller). Kural gevşetilmeli (bölüm 7).

### 5.4 Örtüşme tanılaması (madde 4)

| ölçüt | model | naif | editor |
|---|---|---|---|
| örtüşmeyen çift oranı (tüm basılı çiftlerde) | %1,99 | %2,27 | %0,27 |
| örtüşmeyen çifti olan akor | 37 | 35 | 1 |
| A) çakışma: örtüşen → grup tanımı | 126 → 120 | 161 → 156 | 0 → 0 |
| B) maks açıklık: örtüşen → grup tanımı | 22 → 22 | 13 → 13 | **3 → 5** |

**Sonuç:** 30 ms gruplaması, başlangıcı yakın ama aynı anda basılmayan notaları aynı akora
koyuyor; bu çiftler basılı çiftlerin %0,3–2,3'ü. Etkisi en net editörde görülüyor: grup tanımında
maksimum açıklık 5, örtüşen tanımda **3** — yani editörün "5 yarım ses açıklık" diye görünen
akoru aslında ardışık çalınan iki notaydı. Model tarafında aşırı açıklıklar örtüşme filtresinden
sonra da duruyor (22 → 22), yani **gerçek**. A sütunundaki 126 → 120 farkı örtüşmeden değil
sayım şeklinden geliyor: örtüşen tanım çakışan **çiftleri**, grup tanımı "farklı perde sayısı − 1"
sayar (3 notalı durumda 3'e karşı 2).

### 5.5 Aşırı açıklık doğrulaması (madde 5)

Dört örnek ham notalarla denetlendi; **hepsi gerçek model çıktısı, değerlendirme hatası değil**:

| kayıt | grup başı | açıklık | örtüşme | durum |
|---|---|---|---|---|
| `275_Lw1wc` | 123,582 sn | 22 | 10 örtüşen çift, 0,54–0,61 sn | gerçek |
| `214_z41wc` | 107,968 sn | 19 | 10 örtüşen çift, 1,92 sn | gerçek |
| `275_Lw1wc` | 99,789 sn | 19 | 5 örtüşen çift (grupta örtüşmeyenler de var) | gerçek |
| `275_Lw1wc` | 123,057 sn | 19 | 10 örtüşen çift, ~0,52 sn | gerçek |

Örnek (123,582 sn): perde 45→tel 1 perde 0 (boş), 52→tel 2 perde 2, 57→tel 3 perde 2,
61→tel 4 perde 2, **64→tel 0 perde 24**, 69→tel 5 perde 5. Altı nota 0,54 sn boyunca birlikte
tutuluyor. Model, E4'ü (64) 5. telde açık çalmak yerine **0. telin 24. perdesine** atamış; 69
zaten 5. teli kullandığı için akor içi tel teklik kısıtı 64'ü en pes tele itmiş. Yani aşırı
açıklık, **tel teklik kısıtı + açıklık kısıtının yokluğu** birleşiminden doğuyor.

**Ek bulgu (tel teklik kısıtının sınırı):** 12.842 grubun 3'ü 6'dan fazla nota içeriyor
(30 ms içine sıkışan arpejler). Bu durumda `resolve_chords` her notaya ayrı tel veremiyor ve
argmax'a düşüyor: **3 grupta aynı tel iki notaya verilmiş (%0,02)**; bu çiftlerin
**hiçbiri zamanda örtüşmüyor**, yani fiziksel olarak imkânsız bir durum üretmiyor.

## 6. Karşılaşılan sorunlar ve çözümleri

1. **`finger_baseline` çalınamayan perdede çöküyordu.** Yeni script naif konumu `(-1,-1)`
   veriyordu; mevcut fonksiyon `None` bekliyor (`ValueError: min() arg is an empty sequence`).
   **Çözüm:** script `eval_edition.naive` sözleşmesine uyduruldu. Mevcut kod değişmedi.
2. **Konumu bilinmeyen nota akoru "değerlendirilebilir" gösteriyordu.** `ChordShape.n_unknown`
   artık `perde < 0` durumunu da sayıyor; iki test sabitliyor.
3. **"Ortak alt küme" gerçek kesişim değildi (son doğrulamada bulundu).** İlk sürüm yalnız
   editörün değerlendirilebilir olduğu akorları alıyordu; model ve naif o akorların bir kısmında
   düşüyordu (3.043 / 2.963 / 3.419) → satırlar farklı paydalarla karşılaştırılıyordu.
   **Çözüm:** üç yöntemin DE değerlendirilebilir olduğu **gerçek kesişim** (2.963 akor) ayrı blok
   olarak eklendi; kendi kapsamı ve editör alt kümesi ayrı ayrı korundu.
4. **Zaman örtüşmesi gözetilmiyordu (son doğrulamada bulundu).** Metrikler 30 ms ile gruplanmış
   ama aynı anda basılmayan notaları da fiziksel ihlal sayabiliyordu. **Çözüm:** K7 — A/B/C/E
   yalnız örtüşen çiftlerde; eski tanım karşılaştırma için korundu; 6 yeni test.
5. **Oran paydası yanlıştı.** Çakışma/ters sıra oranları tüm değerlendirilebilir akorlara
   bölünüyordu (tek notalı akorlar dahil), oranı ~13 kat seyreltiyordu. **Çözüm:** PAYDA-3
   (örtüşen basılı çifti olan akorlar) tanımlandı ve tablolarda açıkça yazıldı.

## 7. Ölçümlerin sınırlamaları

1. **PAYDA-3 küçük:** kesişimde yalnız 223–232 akor örtüşen basılı çift içeriyor (GAPS'te
   repertuvar ağırlıkla tek sesli). Oranların güven aralığı geniştir.
2. **El pozisyonu yaklaşıktır** (`perde − parmak + 1` medyanı); bilek açısı, parmak uzunluğu
   modellenmiyor. D metrikleri grup düzeyinde, örtüşme ayrımı yapılmıyor.
3. **Barre belirsizlik kuralı fazla temkinli** (editör 3 temiz / 12 belirsiz): kısmi barre
   klasik icrada yaygın; kural gevşetilmeli.
4. **A) iki sayım şekli var:** örtüşen çift sayısı (ana) ve grup tanımı ("farklı perde − 1").
   3+ notalı durumlarda farklı sonuç verirler; karşılaştırırken tanım belirtilmeli.
5. **Kapsam asimetrisi giderilemez:** editörün parmağı akorların %26,6'sında tam. Modelin en kötü
   şekilleri ölçümsüz bölgede yoğunlaşıyor → kesişim sayıları iyimser.
6. **Tek split (gaps_val, 15 kayıt).** `gaps_test` açılmadı.
7. **Sağ el, bilek, legato/sustain modellenmiyor**; yalnız sol el geometrisi.
8. **Ters sıra ve açıklık "imkânsızlık" değil**; yalnız dağılım ve risk göstergesi.

## 8. P4 için öneriler

1. **Akor içi parmak kısıtı (ana iş).** `resolve_chords` mantığının parmağa taşınması: örtüşen
   notalarda bir parmak yalnız bir perdede (aynı perdede çok tel = barre serbest). Hedef:
   kesişimde A = 22 → **0**; `gaps_val` parmak doğruluğu (0,675) **düşmemeli**.
2. **Tel teklik kısıtı ile açıklık birlikte ele alınmalı.** 5.5'teki örnek gösteriyor: tel teklik
   kısıtı tek başına, açıklık kısıtı olmadan, perde 24 gibi saçma konumlar üretiyor. Önce yumuşak
   açıklık cezası denenmeli (editör dağılımı üst sınır: maks 3, ≥4 oranı %0).
3. **Barre belirsizlik kuralı gevşetilmeli** (kısmi barre geçerli sayılsın).
4. **El pozisyonu çıktısı** — `hand_position` hazır; `NoteEvent`'e taşınması P6.
5. **P4 kabul ölçütü önerisi:** kesişimde A = 0, B ≥4 oranı editöre yaklaşsın (≤ %1),
   `gaps_val` tel/parmak/ürün değerleri (0,766 / 0,675 / 0,382) düşmesin, tüm testler geçsin,
   çıkarım regresyonu farksız kalsın.
6. **Kural kalibrasyonu tamam sayılabilir:** editör A ve C'de 0 ihlal verdi → bu iki kural
   fazla katı değil. Açıklık için "imkânsızlık" eşiği hâlâ yok; editör dağılımı referans alınmalı.

## Çalıştırma

```bash
python -m scripts.eval.eval_playability                       # gaps_val, 3 yontem x 3 payda
python -m scripts.eval.eval_playability --verify-span 12      # asiri aciklik dogrulamasi
python -m unittest discover -s tests -t . -p "test_playability.py"
```
