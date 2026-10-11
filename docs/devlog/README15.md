# Katman 4 Faz 0 — Ölçüm ve test protokolü (teknik tespitinden ÖNCE)

> Durum (11 Ekim 2026): **Faz 0 tamamlandı.** Eğitim yok, model yok, veri indirme yok; mevcut kodun hiçbir satırı
> değişmedi (tüm değişiklik yeni dosya). Taban aynen: `checkpoints/tabcrnn_deep2.pt` + `edition_seq.pt`.
> Dal `katman-4-gaps-validation`, HEAD `72295e6`. Sıradaki (onay bekliyor): Faz 1 — teknik kapsamı + etiket envanteri.

## Neden bu faz
Katman 4 = **gitar çalma tekniklerinin tespiti** (sunum Modül 6 "Technique Recognition", Phase 5; depo yol
haritasında "Katman 4 — Teknikler"). Katman 3.13 kapanış denetiminde (11 Ekim) ölçüm tarafında üç açık bulundu;
bunlar kapatılmadan teknik sonuçları savunulamaz:

1. **GuitarSet'te bağımsız test yok.** Oyuncu 05 aynı anda (a) `train_onset` seçim skorunun bileşeni, (b) eşik/
   çözümleme kalibrasyonunun yapıldığı set, (c) manşet sayıların (tab F1 0.777, nota F1 0.924) raporlandığı set.
   Bu sayılar **validation**'dır; bugüne kadar bu ayrım metinlerde her yerde yazılı değildi.
2. **gaps_test bir kez karar kuralında kullanıldı.** README14 Adım 2 kabul kuralı "gaps_test tel ≥ naif + 0.05"
   biçimindeydi. Ağırlıklar test'e uydurulmadı (seçim gaps_val'de), ama kabul/ret kararı test setine bakılarak
   verildi. Katman 4'te tekrarlanmaması gerekiyor.
3. **Otomatik test yok.** 14 katmanlık geçmişte "regresyon birebir" kontrolleri elle yapıldı; `pytest`/`unittest`
   dosyası yoktu. Katman 4 pipeline'a yeni bir modül ekleyeceği için "eski yolu kırmama" kuralını mekanik
   doğrulayacak bir ağ gerekiyordu.

Devralınan ve Faz 1'in çözmesi gereken açık: **teknik etiketi yok.** Ölçüldü (11 Ekim, 300 GAPS MusicXML dosyası):
`slide` 0, `glissando` 0, `harmonic` 0, `trill-mark` 0, `slur` 4, `tremolo` 1 dosya; yalnız artikülasyon/dinamik var
(`staccato` 48, `accent` 93, `dynamics` 226). GuitarSet JAMS'te teknik anotasyonu yok ama **tel başına
`pitch_contour` + `note_midi`** var (hex pickup) — türetilmiş (sözde) etiketin tek gerçekçi kaynağı. Bu yüzden
Faz 0'ın ölçüt modülü sınıf listesini **sabitlemez**, parametre olarak alır.

## Kapsam
| Bu fazda var | Bu fazda YOK |
|---|---|
| Veri protokolünün koda bağlanması (izin kapıları) | Teknik sınıf listesinin sabitlenmesi (Faz 1) |
| Teknik tanıma ölçüt modülü (P/R/F1, macro, confusion) | F0 eğrisi / özellik çıkarımı (Faz 2) |
| Otomatik test altyapısı (80 test) | Eğitim, model, checkpoint (Faz 4) |
| D1 referans tablosu + çıkarım parmak izi | Mevcut script'lerin değiştirilmesi |
| Seçim / early stopping / aşırı öğrenme kuralları | Veri indirme, önbellek yenileme |

## 1. Veri protokolü (`gtab/evaluation/protocol.py`)
Üç izin ayrı tutulur: **may_train** (ağırlık güncellemesi), **may_select** (eşik/hiperparametre/epoch seçimi),
**may_report** (bir sayı olarak raporlanabilir). "Bağımsız test" ayrı bir alandır.

| split | içerik | eğitim | seçim | rapor | bağımsız test | rapor etiketi |
|---|---|---|---|---|---|---|
| `train`, `train_comp` | GuitarSet oyuncu 00–04 (150 + 150) | ✓ | ✗ | ✗ | ✗ | eğitim verisi |
| `val`, `val_comp` | GuitarSet oyuncu 05 (30 + 30) | ✗ | ✓ | ✓ | **✗** | **validation (bağımsız test değil)** |
| `val_half_a/b` | oyuncu 05, parça-ayrık yarılar (14 / 16) | ✗ | ✗ | ✗ | ✗ | keşif / duyarlılık analizi |
| `gaps_fit` | gaps_train'in eğitim kısmı (203 kayıt, 149 icracı; karantina sonrası **202**) | ✓ | ✗ | ✗ | ✗ | eğitim verisi |
| `gaps_val` | gaps_train'in doğrulama kısmı (28 kayıt, 22 icracı) | ✗ | ✓ | ✓ | ✗ | validation |
| `gaps_test` | resmi GAPS test (30 kayıt, 27 icracı) | ✗ | **✗** | ✓ | **✓** | bağımsız test (icracı-ayrık) |
| `gaps_skipped` | test icracılarıyla örtüşen 39 train kaydı | ✗ | ✗ | ✗ | ✗ | kullanılmaz |

### Yasak kurallar (koda bağlı)
- `assert_may_select("gaps_test")` → **ProtocolError**. Katman 4'ün seçim yapan her yolu bu kapıdan geçer
  (`EpochSelector` kurucusunda otomatik).
- `assert_may_train` eğitim split listesini doğrular: `val`, `val_comp`, `gaps_val`, `gaps_test`, `gaps_skipped`,
  `val_half_*` eğitimde kullanılamaz.
- Protokolde **tanımsız bir split adı** da hata verir (yazım hatasıyla sessizce kural dışına çıkılamaz).
- **Oyuncu 05 alt bölmeleri** (`val_half_a/b`) yalnız keşif/duyarlılık analizidir: `may_select = may_report = False`.
  Yani "bir yarıda eşik seç, diğer yarıda test de" yolu protokolce kapalı — bu bölme bağımsız test ÜRETMEZ
  (oyuncu 05 zaten model seçiminde kullanıldı; bağımsız bir GuitarSet oyuncusu mevcut değil).
- Raporlanan her sayının yanında `report_label(split)` zorunlu: "validation (bağımsız test değil)" /
  "bağımsız test (icracı-ayrık)" / "keşif".

### GAPS 39 kayıt — ve Faz 0'da bulunan üç ek bulgu
`build_gaps --disjoint` test icracılarının train kayıtlarını önbelleğe hiç almıyor (270 → 231). Doğrulandı:
**39 kayıt hesaplandı, önbellekte 0 tanesi var, 0 tanesinin TAB etiketi üretilmiş.** Faz 0'da üç yeni kontrol eklendi:

1. **Ad yazım varyantı.** `build_gaps` ham ad eşleşmesi kullanıyor; GAPS metadata'sında aynı kişi farklı
   yazılabiliyor — ölçüldü: `Ana Vidovic` / `Ana Vidović`, `Tengyue "TY" Zhang` / `Tengyue "Ty" Zhang`.
   `normalize_performer` (aksan kaldır, küçük harf, noktalama at, kelimeleri sırala) ile yeniden hesaplandı:
   **39 = 39, normalize ile yeni yakalanan kayıt yok** → mevcut split varyant yüzünden sızıntı taşımıyor.
   Varyant eşleşmesi bundan sonra standart (test: `test_yazim_varyanti_yakalanir`).
2. **Kanal düzeyi örtüşme.** Ham metadata'da **14** YouTube kanalı hem train hem test'te geçiyor; bu kanallarda
   **88** train kaydı var ve bunların 37'si dışlanmış. Kanallar sayıldı (normalize ad, train + test):
   **11 kişisel kanal** (kanalda tek icracı: Damien Kelly, Edson Lopes, Guitars International – CICGF,
   Maria Linnemann, Massimo Agostinelli, Samantha Muir, Sanja Plohl, Stephanie Jones, Thu Le, Yoo Sik Ro,
   davidrussellguitar) → **hepsinin train kayıtları tamamen dışlanmış (eğitimde kalan 0)**; **3 çok icracılı
   platform kanalı** (`Guitar Salon International` 30 icracı / 37 train kaydı / 35'i eğitimde,
   `SiccasGuitars` 21 icracı / 15 / 15, `PARTITA STUDIO` 2 icracı — test "Han Eun", train "Xavier Jara" / 1 / 1)
   → aynı kanal aynı icracı değil, örtüşme beklenen ve zararsız. 11 + 3 = 14 ✓.
   Kural koda geçti: *kanal train+test'te ve tek icracılıysa o kanalın tüm train kayıtları dışlanmış olmalı*
   → `channel_leakage_suspects`, bugün **ihlal yok**.
   İki kontrol birbirini tamamlıyor: 39 yasaklı kaydın **37'si** ortak kanallarda, **2'si farklı kanal adı
   altında** (`327_6f1wc` → "Matteo Carcassi Studies" / Edson Lopes, `374_3w1wc` → "Ignacio Barra" /
   David Russell). Yani yalnız kanala bakan bir kontrol bu 2 kaydı kaçırırdı; ad eşleşmesi yakaladı.
3. **Adsız 3 kayıt → ikisi doğrulandı, biri karantinaya alındı.** `292_8Sswc`, `309_DV1wc`, `395_w41wc`
   kayıtlarında `performer_name` boş; test icracısı olup olmadıkları ADLA doğrulanamıyor. Kanal adıyla bakıldı:
   `292_8Sswc` → "Armen Doneyan", `309_DV1wc` → "Denian Arcoleo"; ikisi de test kanalları arasında **değil**
   → temiz. `395_w41wc`'de `performer_name`, `channel_name`, `ss_artist`, `video_title` alanlarının **hepsi boş**
   (elde yalnız `yt_id = 2H53_rwralE`) → kimlik doğrulanamıyor, **karantinaya alındı** (aşağıda).
   Not: `gaps_files` icracı alanını `str()` ile okuduğu için üç adsız kayıt tek bir sözde icracı ("nan") sayılıyor;
   bu yüzden fit/val arasına dağılmıyorlar — üçü de `gaps_fit` tarafında.

### Karantina ve Faz 1+ için zorunlu kapı
`QUARANTINE_GAPS_IDS = ("395_w41wc",)` — kimliği doğrulanamayan kayıt. Önbellekten **silinmedi** (mevcut
sonuçların yeniden üretilebilirliği bozulmasın; `data/cache/gaps_train/395_w41wc.npz` yerinde), yalnız yeni
eğitim/seçim listelerinden çıkarılıyor. Kaydın `gaps_tab` TAB etiketi zaten yok (`summary.csv`: `use=False`,
tekrar yapısı tutarsız, notaların %39'u etiketli) → edisyon modelini etkilemiyor; etkisi yalnız perde
denetimiydi ve `gaps_fit` tarafındaydı.

**Katman 4'te yazılacak her yeni eğitim/değerlendirme script'i şu kapıları kullanacak** (mevcut
`train_onset` / `eval_pitch` / `eval_hmm` / `train_edition_seq` bilinçli olarak değiştirilmedi):

```python
from gtab.evaluation.protocol import (assert_may_report, assert_may_select, assert_may_train,
                                      gaps_training_files, report_label)
from gtab.evaluation.selection import EpochSelector

fit, val = gaps_training_files()          # gaps_files YERINE: split dogrulanir + karantina dusulur
assert_may_train(["train", "train_comp"]) # GuitarSet egitim split'leri
sel = EpochSelector("gaps_val")           # secim kapisi; "gaps_test" verilirse ProtocolError
...
assert_may_report("gaps_test")            # yalniz kapanista, bir kez
print(f"GAPS nota F1 {x:.3f} [{report_label('gaps_test')}]")
```

Kurallar: (1) eğitim dosya listesi `gaps_training_files()` / `assert_may_train` ile alınır, (2) her eşik ve
epoch seçimi `EpochSelector` ya da `assert_may_select` üzerinden gider, (3) raporlanan her sayının yanına
`report_label(split)` yazılır, (4) `gaps_test` yalnız faz kapanışında bir kez okunur, (5) yeni ölçüt eklenirse
`docs/baselines/d1_baseline.json`'a **grup** ve **koşullu/oracle** alanlarıyla eklenir.

## 2. Teknik tanıma ölçütleri (`gtab/evaluation/technique_eval.py`)
**Değerlendirme birimi: nota olayı.** Teknik nota olayına atfedilir; kare düzeyi teknik metriği tanımlanmadı
(vibrato/legato nota boyunca sürer; karede sayılırsa uzun notalar metriği domine eder).

**Eşleştirme:** gerçek ↔ tahmin notaları **aynı perde + |Δonset| ≤ 50 ms**, bire bir, en küçük |Δt| önce
(`mir_eval` nota toleransı ve `eval_edition` uçtan uca eşlemesiyle aynı kural). Sınır dahil (kayan nokta payı 1e-9).

**İki mod (karıştırılmaz):**
- `eslenmis` — yalnız eşleşen notalar: teknik modülünün kendi doğruluğu.
- `uctan-uca` — eşleşmeyen gerçek nota ilgili sınıfa FN, eşleşmeyen tahmin ilgili sınıfa FP; confusion matrix'te
  son satır/sütun `yok`. Ürün sayısı budur (nota bulma hatası dahil).

**Metrikler:** sınıf başına P / R / F1 / support, **macro-F1 (ANA ÖLÇÜT)**, `macro_f1_no_normal`,
`macro_f1_strict`, weighted-F1 ve (C+1)×(C+1) confusion matrix. NORMAL sınıfı baskın olacağı için weighted-F1 ve
doğruluk yanıltıcıdır; teknik kararları `macro_f1_no_normal` ile birlikte okunur. Kabul kuralları
`majority_baseline` (her notaya NORMAL) tabanına göre kurulur.

**Boş sınıf kuralı** (raporda aynen bu adlar kullanılır):

| durum | P | R | F1 | ana macro-F1 | nerede görünür |
|---|---|---|---|---|---|
| support > 0 | tp/(tp+fp), hiç tahmin yoksa 0 | tp/support | normal | **girer** | tabloda `olculdu` |
| support = 0, tahmin yok | — | — | — | girmez | `tanimsiz` listesi |
| support = 0, tahmin var ("hayalet") | 0 | tanımsız | 0 | girmez | `hayalet` listesi + `macro_f1_strict` |

**Etiket kaynağı zorunlu:** `gercek` (hex pickup / insan etiketi) ve `turetilmis` (kuraldan üretilmiş sözde etiket)
rapor başlığında taşınır; `assert_not_mixed` ikisini aynı tabloda raporlamayı engeller. Faz 3'ün sözde etiketleri
"ground truth" olarak sunulmayacak.

## 3. Otomatik test altyapısı (`tests/`)
Harici bağımlılık eklenmedi: **stdlib `unittest`** (`requirements.txt`'te pytest yok ve eklenmedi). Testler
yalnız `numpy` kullanıyor — torch olmadan da geçiyor (doğrulandı), yani veri seti, GPU ve checkpoint gerekmez.

```bash
python -m unittest discover -s tests -t .        # tek komut, ~0.01 sn
```

| dosya | test | kapsam |
|---|---|---|
| `tests/test_protocol.py` | 25 | izin kapıları, bağımsız test tekliği, keşif yarıları, split kesişimi, icracı örtüşmesi, 39 kayıt, yazım varyantı, adsız kayıt, kanal düzeyi sızıntı, karantina + eğitim listesi kapısı, yarı bölmesinin determinizmi |
| `tests/test_technique_eval.py` | 21 | eşleştirme toleransı (sınır dahil/dışında), bire bir eşleme, elle hesaplanmış P/R/F1, macro/weighted, boş + hayalet sınıf, confusion matrix (şekil, sayım, `yok` satır/sütunu), etiket kaynağı korumaları, çoğunluk tabanı |
| `tests/test_baselines.py` | 12 | tablo bütünlüğü, D1 çekirdek değerlerinin birebirliği, split cinsi etiketi, tolerans sınırları, metrik grubu karıştırma koruması |
| `tests/test_fingerprint.py` | 11 | parmak izi istatistikleri, kafa eklenmesi/kaldırılması, kurulum farkı, histogram kayması, tolerans içi oynama |
| `tests/test_selection.py` | 11 | seçim kapısı, en iyi epoch, başlangıç modelinin korunması, min_delta, early stopping, aşırı öğrenme işareti |
| **toplam** | **80** | — |

Gerçek diskteki önbelleği denetleyen kısım testlerde değil, ayrı bir script'tedir (aşağıda): birim testler veri
seti olmadan her makinede çalışsın diye.

## 4. Regresyon koruması
**(a) D1 referans tablosu** — `docs/baselines/d1_baseline.json` + `gtab/evaluation/baselines.py`.
Her satır: değer, veri seti, split, **split cinsi** (protokolden), **birim**, **koşullu/oracle bilgi**,
çözümleme, metrik **grubu**, kaynak devlog.

| metrik | değer | split | cins | birim / koşul |
|---|---|---|---|---|
| `gs_solo_note_f1` | 0.924 | `val` | validation | nota olayı; sesten uçtan uca |
| `gs_poly_note_f1` | 0.807 | `val_comp` | validation | nota olayı; sesten uçtan uca |
| `gs_solo_tab_f1` | 0.777 | `val` | validation | kare düzeyi (tel, perde) token |
| `gaps_note_f1` | 0.779 | `gaps_test` | **bağımsız test** | nota olayı; sesten uçtan uca |
| `gaps_edition_string_acc` | 0.815 | `gaps_test` | **bağımsız test** | nota; **ORACLE NOTA**, ses yok (edisyon modu) |
| `gaps_edition_finger_acc` | 0.636 | `gaps_test` | **bağımsız test** | nota; **ORACLE KONUM**, yalnız parmak |

Çekirdeğe ek olarak karıştırmayı önlemek için bağlam satırları da tabloda: `gs_oracle_string_acc` 0.878,
`gaps_audio_oracle_string_acc` 0.610 (gaps_val), `gaps_naive_string_acc` 0.745, `gaps_finger_baseline_acc` 0.566,
`gaps_edition_product_acc` 0.434, `gaps_e2e_string_acc` 0.655, `gaps_e2e_full_finger_acc` 0.331,
`gaps_note_coverage` 0.817.

**Toleranslar:** `yeniden-kosu` **0.002** (aynı checkpoint + aynı kod; README14'te ölçülen GPU iç oynaması),
`yeniden-egitim` **0.01** (CLAUDE.md "tek seed farkı gürültüdür"). **Grup kuralı:** `assert_comparable` farklı
gruptaki sayıların kıyaslanmasını engelliyor — 0.815 (oracle nota + edisyon) ile 0.655 (uçtan uca) ya da nota F1
ile tab F1 aynı tabloda karşılaştırılamaz.

**(b) Çıkarım parmak izi** — `gtab/evaluation/fingerprint.py` + `docs/baselines/inference_fingerprint.json`.
`tabcrnn_deep2.pt` ile `val`'in ilk 2 kaydında kafa başına (ortalama, std) + tab argmax sınıf histogramı.
Tam metrik koşusu saatler sürüyor; parmak izi saniyeler sürüyor ve kare düzeyinde karşılaştırdığı için metrikten
daha hassas. Cihaz varsayılanı **CPU** (deterministik; GPU koşudan koşuya ±0.002 oynuyor). Kaydedilen referans:

| kayıt | kare | tab güveni | onset ort. | perde-onset ort. |
|---|---|---|---|---|
| `05_BN1-129-Eb_solo` | 962 | 0.969301 | 0.014073 | 0.002035 |
| `05_BN1-147-Gb_solo` | 844 | 0.962936 | 0.010773 | 0.002144 |

Not: tab kafası softmax olduğu için tüm olasılıkların ortalaması 1/26 sabittir (ilk denemede bu yakalandı) →
parmak izinde ortalama yerine **güven** (argmax olasılığı) kullanılıyor.

**Faz 4'ün kuralı:** teknik modülü kapalıyken `check_inference_regression` farksız geçmeli; geçmiyorsa yeni modül
eski çıkarım yolunu kırmıştır. Faz 0'da bu komut iki kez çalıştırıldı ve ikinci koşu referansla **birebir** geldi.

## 5. Aşırı öğrenme önlemleri (`gtab/evaluation/selection.py`)
Eğitim yapılmadı; kurallar koda bağlandı. `EpochSelector`:
- Kurucu **seçim split'ini protokolden doğrular** → `gaps_test` ile kurulamaz.
- Seçim ölçütü teknik tanımada **macro-F1**; checkpoint yalnız ölçüt iyileşince yazılır.
- **Epoch 0 (başlangıç modeli) de değerlendirilir**; hiçbir epoch geçemezse `kept_initial()` → taban korunur
  (Katman 3.13 Adım 1'de `tabcrnn_gtab.pt` ile yaşanan durumun kurallaştırılmış hali).
- `min_delta` ile gürültü eşiği, `patience` ile early stopping.
- `overfitting(window=3)`: eğitim kaybı sürekli düşerken doğrulama ölçütü iyileşmiyorsa uyarır (README14 Adım
  2b'de parmak kaybının 10. epoch'tan sonra yaptığı şey).
- Günlük satırı eğitim kaybı + doğrulama kaybı + ölçütü birlikte yazar.

**Sınıf dengesizliği için yeni kod yazılmadı:** mevcut `GuitarSetSeq/PitchSeq.class_weights` (`inv` / `sqrt`,
`cap`) ve `train_onset --oversample` yeniden kullanılacak; teknik tarafında ana koruma macro-F1'in ölçüt olması.

## Eklenen dosyalar (mevcut dosyaların hiçbiri değişmedi)
| dosya | satır | görev |
|---|---|---|
| `gtab/evaluation/protocol.py` | 266 | split rolleri, izin kapıları, sızıntı kontrolleri (ad varyantı, kanal), karantina + `gaps_training_files`, oyuncu 05 keşif yarıları |
| `gtab/evaluation/technique_eval.py` | 220 | nota eşleştirme, sınıf metrikleri, macro/weighted, confusion matrix, rapor |
| `gtab/evaluation/baselines.py` | 102 | D1 tablosu okuma/doğrulama, tolerans karşılaştırma, grup koruması |
| `gtab/evaluation/fingerprint.py` | 80 | çıkarım parmak izi üretimi ve karşılaştırması |
| `gtab/evaluation/selection.py` | 104 | epoch seçimi, early stopping, aşırı öğrenme işareti |
| `scripts/eval/check_protocol.py` | 185 | gerçek önbellekte protokol/sızıntı denetimi (salt okunur) |
| `scripts/eval/check_inference_regression.py` | 90 | parmak izi kaydet/karşılaştır |
| `docs/baselines/d1_baseline.json` | — | D1 referans değerleri (tek kaynak) |
| `docs/baselines/inference_fingerprint.json` | — | çıkarım parmak izi referansı |
| `tests/__init__.py`, `tests/test_*.py` | 6 dosya, 701 | 80 otomatik test |
| `docs/devlog/README15.md` | — | bu kayıt |

Kök `README.md` ve geçmiş devlog'lar **bilinçli olarak değiştirilmedi** (kullanıcı talimatı: değişiklikler küçük
ve Katman 4'e özel kalsın). Yol haritası/devlog listesi güncellemesi Katman 4 kapanışına bırakıldı.

## Çalıştırma
```bash
# 1) birim testler (veri seti / GPU gerekmez, ~0.01 sn)
python -m unittest discover -s tests -t .

# 2) gerçek önbellekte protokol + sızıntı denetimi (salt okunur, ~3 sn)
python -m scripts.eval.check_protocol

# 3) çıkarım yolu regresyonu (CPU, ~10 sn); referans Faz 0'da kaydedildi
python -m scripts.eval.check_inference_regression
```

## Sonuçlar (11 Ekim 2026; son kontrol koşusu dahil)
- **Birim testler: 80/80 geçti** (0.013 sn). torch kaldırıldığında da 80/80 → bağımlılık yalnız numpy.
  Dağılım: protokol 25, teknik ölçütleri 21, referans tablo 12, parmak izi 11, seçim 11.
- **Protokol denetimi: 17/17 geçti.** GuitarSet kayıt/icracı kesişimi 0; GAPS train(231)/test(30) ortak icracı 0;
  yasaklı 39 kayıt önbellekte 0 ve TAB etiketi 0; yazım varyantı ile yeni yakalanan kayıt yok (39 = 39); kanal
  düzeyi ihlal yok (11 kişisel kanalın tamamı dışlanmış); adsız 3 kayıttan 2'si kanal adıyla temiz, 1'i
  karantinada; karantina kaydı eğitim/seçim listelerinde yok ama önbellekte duruyor (231 → denetimli 230;
  `gaps_fit` 203 → 202, `gaps_val` 28 değişmedi); `gaps_fit`/`gaps_val` ortak icracı 0; `gaps_tab` 165 = 142
  train + 23 test; oyuncu 05 keşif yarıları parça-ayrık (14 / 16); veri envanteri beklenen sayılarda.
- **Kod denetimi (son kontrol):** 13 yeni dosyada sözdizimi hatası ve kullanılmayan import yok (AST taraması +
  `compileall`); beş yeni `gtab.evaluation` modülü tek tek import edildi.
- **Çıkarım parmak izi:** referans kaydedildi, ikinci koşu farksız ("REGRESYON YOK").
- **D1 taban değerleri korundu:** tabloda yazılı 6 çekirdek değer devlog kayıtlarıyla birebir; testler bu
  değerleri elle yazılı olarak da doğruluyor (tablo değiştirilirse test düşer). **Bu fazda hiçbir metrik yeniden
  ölçülmedi** — Faz 0 ölçüm yapmadı, ölçümün kurallarını kurdu.

## Açık riskler
1. **GuitarSet'te bağımsız test hâlâ yok** ve bu faz onu yaratamaz: 6 oyuncudan 5'i eğitimde, 1'i kalibrasyonda.
   Çözüm ancak yeniden eğitimle (bir oyuncuyu her şeyden çıkararak) olur — maliyetli, Katman 4 kapsamı dışı.
   Bu yüzden kural: GuitarSet sayıları **validation** etiketiyle raporlanır, teknik kabul kriterleri GuitarSet
   validation + GAPS gaps_val üzerine kurulur, `gaps_test` kapanışta bir kez okunur.
2. **Teknik etiketi yok** (yukarıdaki MusicXML taraması). Faz 3'ün sözde etiketleri hem eğitim hem ölçüm
   referansı olursa metrik şişer; bu yüzden Faz 5'te küçük bir elle etiketli küme zorunlu tutuldu.
3. **`395_w41wc` karantinada, ama karantina yalnız yeni kodda etkili.** Kayıt önbellekte duruyor; eski
   script'ler (`train_onset`, `train_edition_seq`) `gaps_files`'ı doğrudan çağırdığı için onları etkilemez —
   yani mevcut `tabcrnn_deep2.pt` bu kaydı perde denetimi olarak görmüş olabilir (`gaps_fit` tarafında).
   Geçmiş değiştirilmiyor; Katman 4'ün yeni eğitimleri kaydı görmeyecek. Daha genel kalan risk: adı DOLU olup
   yanlış/eksik yazılmış bir icracı kimliği ilkesel olarak doğrulanamaz (normalize ad + kanal kontrolü bugün
   temiz sonuç veriyor).
4. **Protokol kapıları yalnız yeni kodda zorunlu.** Mevcut `train_onset` / `eval_pitch` / `eval_hmm` /
   `train_edition_seq` bilinçli olarak değiştirilmedi (davranışı kırmamak için); eski yolların protokol ihlali
   hâlâ mümkün. Katman 4'ün yazacağı her yeni eğitim/ölçüm yolu yukarıdaki kapıları çağırmak zorunda; kapıların
   mekanik zorlayıcısı yok, kural devlog + kod yorumlarıyla taşınıyor.
5. **Parmak izi 2 GuitarSet kaydıyla sınırlı** (hız için). GAPS yolundaki bir regresyonu yakalamaz; Faz 4'te
   `--split gaps_val --files 1` ile genişletilmesi önerilir.

## Faz 0 kabul kriterleri
| kriter | durum |
|---|---|
| Veri protokolü belgelenmiş ve koda bağlanmış; test seti seçimde kullanılamıyor | ✓ (izin kapıları + 25 test) |
| GuitarSet oyuncu 05 sonuçları validation olarak etiketli; alt bölmeler keşif sayılıyor | ✓ |
| 39 GAPS kaydı eğitim/kalibrasyona girmiyor (+ varyant ve kanal kontrolü) | ✓ (17/17 denetim) |
| Kimliği doğrulanamayan kayıt (`395_w41wc`) yeni eğitim/seçimden çıkarıldı; önbellek değişmedi | ✓ (karantina) |
| Faz 1+ script'leri için protokol kapılarının kullanımı dokümante | ✓ ("Karantina ve Faz 1+ için zorunlu kapı") |
| Teknik ölçütleri (P/R/F1/support, macro, weighted, confusion, boş sınıf, etiket kaynağı, birim/tolerans) tanımlı | ✓ |
| Tek komutla çalışan, dataset gerektirmeyen otomatik test altyapısı | ✓ (80 test, stdlib) |
| D1 referans tablosu + toleranslar + metrik karıştırma koruması | ✓ |
| Teknik modülü kapalıyken eski çıkarım yolu değişmemiş | ✓ (parmak izi farksız; mevcut kod hiç değişmedi) |
| Aşırı öğrenme / seçim / early stopping kuralları hazır (eğitim yok) | ✓ |
| Checkpoint, önbellek ve ham veri korunmuş | ✓ (yalnız yeni dosya; `git status` takipli dosyada değişiklik yok) |

## Durum
- [x] Faz 0 — veri protokolü, teknik ölçütleri, test altyapısı, D1 referansı, regresyon parmak izi, seçim kuralları
- [x] Faz 0 son kontrol (11 Ekim) — `395_w41wc` karantinası, kanal sayımı düzeltmesi (11 + 3 = 14), yeni
      script'ler için kapı dokümantasyonu, kod denetimi; 80 test + 17 protokol kontrolü + regresyon farksız
- [ ] Faz 1 — teknik kapsamı + etiket envanteri (hangi teknik ölçülebilir, hangi etiket kaynağı, kaç örnek)
- [ ] Faz 2 — sürekli F0 altyapısı (GuitarSet hex `pitch_contour`, GAPS mono F0)
- [ ] Faz 3 — kural tabanlı tespit + sözde etiket üretimi (GuitarSet)
- [ ] Faz 4 — mikrofondan teknik sınıflandırıcı (eğitim; macro-F1 ölçütü, `EpochSelector`)
- [ ] Faz 5 — klasik gitarda doğrulama (GAPS + elle etiketli küçük küme)
- [ ] Faz 6 — entegrasyon, regresyon, kök `README.md` / yol haritası güncellemesi, Katman 4 kapanışı
