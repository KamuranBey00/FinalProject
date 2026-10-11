# Katman 4 — Faz 2A: Sürekli F0 Altyapısı ve Kontur Temizliği

> **Durum (11 Ekim 2026):** Faz 2A tamamlandı. Eğitim yok, veri indirme yok; mevcut önbellek, checkpoint
> ve model kodu değişmedi. Yeni önbellek yalnız `data/cache/contour_guitarset/` (18 MB). Taban aynen
> `checkpoints/tabcrnn_deep2.pt`. Dal `Armagan-katman-4`, HEAD `b988d82`; commit/push yapılmadı.
> Sıradaki (onay bekliyor): **Faz 2B** — insan doğrulaması ve etiket güvenilirliği.

**Terimler (ilk geçtikleri yerde):** *F0* = temel frekans, bir notanın algılanan perdesi (Hz).
*Kontur* = F0'ın zaman içindeki sürekli izi. *Cent* = yarım sesin yüzde biri (1 yarım ses = 100 cent);
perde sapmalarını ölçmenin standart birimi. *Vibrato* = perdenin merkez etrafında periyodik salınımı.
*Ligado (hammer-on / pull-off)* = ikinci notanın yeni bir sağ el vuruşu olmadan çıkması. *Glide* =
sürekli perde kayması (slide ve bend bu başlık altında birleşik). *CQT* = sabit-Q dönüşümü, projenin
model girdisi olan logaritmik frekans spektrumu.

---

## 1. Faz 2A'ya başlamadan önce elimizde ne vardı?

| Varlık | Durum |
|---|---|
| Nota katmanı | `tabcrnn_deep2.pt` + kalibrasyon; GuitarSet solo nota F1 0,924 (validation), GAPS nota F1 0,779 (test) |
| Ölçüm protokolü (Faz 0) | Split izin kapıları, D1 referans tablosu, çıkarım parmak izi, 80 otomatik test |
| Teknik envanteri (Faz 1) | GuitarSet'te **teknik etiketi yok**; tel başına `pitch_contour` var. Aday sayıları: VIBRATO 3.986 (%6,38), HAMMER_ON 2.019 (%3,23), PULL_OFF 1.896 (%3,03), GLIDE 408 (%0,65), NORMAL ≈ 54.167 (%86,7) |
| GAPS sembolik etiketleri | `slur` (ligado) 40 bağ / 3 kayıt, `tremolo` 34 / 1 kayıt; slide, glissando, bend, hammer-on, pull-off, harmonic = **0** |
| Kapsam kararı (Faz 1) | A: VIBRATO + HAMMER_ON + PULL_OFF + NORMAL · B: SLIDE/GLIDE ek etiket gerektirir · C: BEND ve HARMONİK kapsam dışı |

Yani teknik tespitinin **tek girdisi** sürekli F0'dı ve o F0'ın kalitesi hiç denetlenmemişti.

## 2. Faz 1'de hangi güvenilirlik sorunlarını tespit ettik?

1. **Fiziksel olarak imkânsız perde hareketi.** Nota içi net perde kayması en fazla **2.216 cent**
   (22 yarım ses) ölçüldü (Faz 1'in 60 kayıtlık duyarlılık alt kümesinde; Faz 2A aynı koşulu 360 kaydın
   tamamında ölçtü → 3.377 cent, §10). Gitarda tek bir nota boyunca böyle bir hareket mümkün değil
   → kontur verisinde hata var.
2. **Eşiklere aşırı duyarlılık.** Vibrato aday sayısı eşik setine göre iki kat değişiyordu
   (icracı 00'da gevşek 914 / varsayılan 736 / sıkı 517).
3. **Aday ≠ gerçek etiket.** Sayılar bir kuralın çıktısıydı; hiçbiri dinlenerek doğrulanmamıştı.
4. **Slide ve bend ayrışmıyor.** 408 glide adayının 206'sı tam yarım sese iniyor (slide profili),
   202'si inmiyor (bend/portamento profili) — kontur tek başına ayırmaya yetmiyor.
5. **GAPS'te sol el tekniği denetimi yok** (yukarıdaki sıfırlar) → klasik gitarda doğrulama yolu açık değil.

## 3. Bu sorunlar düzeltilmeden model eğitilse ne olurdu?

Dört somut yanlış sonuç riski vardı:

- **Aday havuzu, geçerliliği şüpheli örneklere dayanıyordu.** Faz 1'in kuralları Faz 2A'nın
  geçerlilik filtresiyle (sıfır frekans + tel aralığı dışı örnekler ayıklanmış) yeniden koşuldu.
  **Geçerlilik filtresi sonrası aday sayısı: GLIDE 408 → 389** (−19, sınıfın %4,7'si),
  **VIBRATO 3.986 → 3.912** (−74, %1,9). Bu, elenen adayların "yanlış teknik" olduğunu
  **kanıtlamaz**; yalnız şunu gösterir: bu adayların aday olması, artık geçersiz saydığımız
  örneklere bağlıydı. Notanın gerçekte teknik içerip içermediği ancak dinlemeyle (Faz 2B) bilinir.
  Risk şu: en küçük sınıfın yirmide biri, dayanağı denetlenmemiş örneklerden geliyordu.
- **Baskın sınıf kirlenirdi.** Kuralın karar veremediği notalar sessizce NORMAL sayılsa, NORMAL
  (notaların %86,7'si) içine gerçek teknikler karışır; sınıflandırıcı "teknik yok" demeyi öğrenir ve
  *macro-F1* (sınıf başına F1'lerin ortalaması; baskın sınıfı ödüllendirmeyen ölçüt) olduğundan iyi görünür.
- **Metrik kendi kendini onaylardı.** Kuralla üretilen etikete karşı ölçüm yapılırsa, model yalnız
  kuralı taklit ettiği ölçüde yüksek skor alır; teknik doğruluğu hakkında hiçbir şey söylemez.
- **Klasik gitar iddiası dayanaksız kalırdı.** GAPS'te etiket olmadığı için, GuitarSet (çelik telli
  akustik) üzerinde elde edilen sayılar klasik gitar doğruluğu gibi sunulma riski taşıyordu.

## 4. Faz 2A'da hangi işlemleri hangi sırayla yaptık?

1. **Başlangıç güvenliği:** branch/HEAD/`git status` doğrulandı; disk alanı kontrol edildi (21 GB boş).
2. **Ham veri incelemesi:** 360 JAMS dosyasında örnek adımı, zaman ekseni, voiced alanı, Hz aralığı,
   geçersiz değerler, `index` alanı ve tel kimliği ölçüldü (bulgular §6).
3. **Modül yazımı:** `gtab/data/contour.py` + `scripts/data/build_contour.py` (mevcut koda dokunulmadı).
4. **Boyut provası:** `--dry-run` ile 10 kayıt → 105 KB/kayıt (≈ 38 MB tahmin) → yazmaya karar verildi.
5. **Önbellek üretimi:** 360 kayıt → `data/cache/contour_guitarset/` (18 MB, 360 npz + `summary.csv`).
6. **Faz 1 artefaktının kaynağı aranması:** nota içi net kayma dağılımı üç koşulda karşılaştırıldı (§10).
7. **Testler:** 24 yeni test yazıldı; ardından Faz 0/1 testleri, protokol kontrolü ve çıkarım regresyonu.
8. **GAPS incelemesi:** `scripts/eval/probe_gaps_f0.py` ile mono/poli ayrımı, CQT'den ince perde, pyin denemesi.
9. **Commit öncesi kod incelemesi:** iki kalite sorunu arandı, kırmızı testlerle yeniden üretildi ve
   düzeltildi (§5.1); 360 kayıtlık sayımlar yeniden hesaplandı.
10. **Dokümantasyon:** bu kayıt.

**Protokol uyumu (Faz 0 kuralları):** `gaps_test` hiç açılmadı, `395_w41wc` karantinada kaldı, GAPS
dosya listeleri `protocol.gaps_training_files()` ile alındı; GuitarSet ses dosyaları hiç okunmadı
(kontur JAMS'te hazır); `probe_gaps_f0.py` hiçbir dosyaya yazmıyor.

## 5. Oluşturulan kod dosyaları ve görevleri

| dosya | satır | görev |
|---|---|---|
| `gtab/data/contour.py` | 305 | Çekirdek kütüphane: Hz→cent (0/NaN güvenli), telin fiziksel perde aralığı, segment kimliği, örnek bayrakları, nota–kontur hizalaması, nota durumu, kalite istatistiği, `fallback_label` (UNKNOWN sözleşmesi) |
| `scripts/data/build_contour.py` | 146 | GuitarSet JAMS → `data/cache/contour_guitarset/` (kayıt başına npz + `summary.csv`); `--dry-run` ile yazmadan boyut/kalite raporu |
| `scripts/eval/probe_gaps_f0.py` | 162 | GAPS incelemesi (salt okunur): monofonik/polifonik ayrım, önbellek CQT'sinden ince perde, `librosa.pyin` denemesi |
| `tests/test_contour.py` | 293 | 29 test; tamamı sentetik veriyle çalışır, GuitarSet/GAPS gerektirmez |

Yeni modül bağımsızdır: model, çözümleme ve eğitim kodunu import etmez; mevcut dosyaların hiçbiri değişmedi.

### 5.1 Commit öncesi kod incelemesi: iki düzeltme

**1) Kapsama tanımı düzeltildi.** `coverage`, notanın penceresine düşen **tüm** örneklerden
hesaplanıyordu; 0 Hz ve tel aralığı dışı örnekler oranı şişiriyordu. Kapsama bir güvenilirlik
kapısı olduğu için ("notanın ne kadarını gerçekten ölçebiliyoruz?") doğru tanım **geçerli örnekli
süre / nota süresi**'dir; geçersiz örnek hiçbir perde bilgisi taşımaz. Düzeltme öncesi, konturunun
yarısı geçersiz olan bir nota kapsama kapısından geçip sessizce `RELIABLE` olabiliyordu.
**Etki:** `RELIABLE` 62.373 → **62.254**, `SUSPECT` 26 → **145** (+119 nota, 69 kayıt);
`INVALID` 77 ve `EMPTY` 0 değişmedi, örnek düzeyi sayımlar da değişmedi (bayraklara dokunulmadı).

**2) Konturu olmayan telin notaları artık atlanmıyor.** `quality_stats()` ve `pack()` yalnız
`pitch_contour` bulunan telleri dolaşıyordu; bir telde `note_midi` olup kontur olmasa o telin
notaları sayımdan ve önbellekten **sessizce** düşerdi — görünmeyen nota UNKNOWN bile olamaz.
Artık `iter_strings()` iki anahtar kümesinin birleşimini dolaşıyor; konturu olmayan tel boş kontur
alıyor ve notaları `EMPTY` (→ UNKNOWN) oluyor. **Etki:** GuitarSet sayılarında değişiklik yok —
360 kaydın tamamında her telde `pitch_contour` anotasyonu var, yani hata **gizli** kalmıştı.

İkisi de önce sentetik kırmızı testlerle yeniden üretildi (4 başarısız test), sonra düzeltildi;
testler kalıcı olarak `tests/test_contour.py` içinde (`TestCoverageDefinition`,
`TestStringWithoutContour`).

## 6. Geçersiz ve şüpheli örnekler nasıl tespit edildi?

Önce ham veride ne olduğu ölçüldü:

| bulgu | değer |
|---|---|
| Örnek adımı | 5,805 ms (256/44100); adımların %98'i tam bu değerde |
| `voiced` alanı | veride **her zaman True**; sessizlik örnek olarak yok, zaman boşluğu olarak var |
| `frequency == 0` örnekleri | **var** (voiced=True olmasına rağmen) → logaritma alınamaz |
| `value.index` alanı | takipçinin segment kimliği: her zaman boşluğunda değişiyor (3.236/3.236), boşluk olmadan da 1.435 kez değişiyor → takipçinin izi kaybettiği yerler |
| `data_source` ↔ tel | "0".."5" = `STANDARD_6.tuning[0..5]` ile birebir (src 0'ın en pes notası 39,8 ≈ E2 = 40) → yeniden indeksleme gerekmez |
| Tel aralığı dışı örnek | **var**; ör. 5. telde 61,7 Hz = açık telin ~29 yarım ses altı |
| Zaman ekseni | geriye gidiş 0, tekrarlı damga 0, NaN/inf 0 |

Buna dayanarak örnek başına **bit maskesi** üretiliyor — hiçbir değer düzeltilmiyor:

| bayrak | koşul | sonuç |
|---|---|---|
| `F_ZERO` | `frequency <= 0` | geçersiz (cent = NaN) |
| `F_NONFINITE` | NaN / inf | geçersiz |
| `F_RANGE` | perde, telin fiziksel aralığı dışında (açık tel − 0,5 … açık tel + 24 + 0,5 yarım ses) | geçersiz |
| `F_UNVOICED` | `voiced == False` | geçersiz (bu veride görülmedi) |
| `F_JUMP` | **aynı segment içinde** ardışık iki örnek arası \|Δcent\| > 600 | **şüpheli**, silinmez |
| `F_SEGSTART` | zaman boşluğu (> 20 ms) veya `index` değişimi | bilgi amaçlı |

İki tasarım kararı:

- **Oktav sıçraması yalnız segment içinde aranır.** Segment sınırındaki büyük fark takipçi
  kesintisidir, gerçek perde hareketi değil; orada sıçrama aramak yanlış pozitif üretir.
- **Sıçrama ham cent üzerinden hesaplanır**, yani geçersiz bir örneğin yol açtığı sıçrama da
  işaretlenir. 17 sıçramanın yalnız 4'ü geçerli örnekler arasında; 13'ü geçersiz örnek kaynaklı.
  İkisi de notayı şüpheli yapar — muhafazakâr yön bilinçli.

**Gerçek hareket neden kesilmiyor:** 600 cent / 5,8 ms ≈ 103 yarım ses/saniye; gerçek bir slide bunun
yirmi katı altında, vibrato ise ±100 cent mertebesinde salınır. Sentetik vibratonun işaretlenmediği
testle sabitlendi (§9).

## 7. NORMAL ve UNKNOWN ayrımı nasıl korundu?

Nota düzeyi durum, sırayla değerlendirilir: `EMPTY` (hiç geçerli örnek yok) → `INVALID` (geçerli örnek
< 10) → `SUSPECT` (sıçrama var **veya** geçerli örnek kapsaması < 0,6 **veya** \|net kayma\| > 700 cent)
→ `RELIABLE`. Kapsama **yalnız geçerli örnekler** üzerinden hesaplanır (§5.1).

**`RELIABLE` ne demek, ne demek değil:** yalnızca bu fazın F0 kalite kontrollerinden (geçerli örnek
sayısı, kapsama, sıçrama bayrağı, net kayma sınırı) geçtiği anlamına gelir. Notanın teknik etiketinin
doğrulandığı anlamına **gelmez** — bu fazda hiçbir etiket üretilmedi ve hiçbir örnek dinlenmedi.

Sözleşme koda bağlandı: `fallback_label(status)` yalnız `RELIABLE` için `None` döndürür (kararı Faz 3
verecek), **diğer her durumda `UNKNOWN`** döndürür ve hiçbir koşulda `NORMAL` döndürmez. Bu, iki ayrı
test tarafından sabitlendi; ayrıca `UNKNOWN` sabitinin `technique_eval.NORMAL` ile eşit olmadığı da
test edilir. Bu fazda hiçbir teknik etiketi (vibrato / ligado / glide) **üretilmedi**.

## 8. GuitarSet ve GAPS üzerinde ne ölçtük, ne öğrendik?

### 8.1 GuitarSet — 3.214.786 kontur örneği, 62.476 nota

| sınıf | adet | oran | etkilenen kayıt |
|---|---|---|---|
| geçerli | 3.194.079 | %99,356 | — |
| `frequency == 0` → geçersiz | 15.018 | %0,467 | 338/360 |
| tel aralığı dışı → geçersiz | 5.689 | %0,177 | 279/360 |
| NaN / inf, `voiced=False` | 0 | %0 | 0 |
| sıçrama (şüpheli, silinmedi) | 17 | %0,001 | 16/360 |
| segment başı (bilgi) | 62.476 | %1,94 | 360/360 |

En kötü kayıtlar: `04_SS3-98-C_comp` geçerli örnek oranı %95,1 (1.027 sıfır örnek);
`02_SS3-98-C_comp` 222 aralık dışı örnek; `02_SS1-68-E_comp` 2 sıçrama.

| nota durumu | adet | oran | anlamı |
|---|---|---|---|
| `RELIABLE` | 62.254 | %99,64 | F0 kalite kontrollerinden geçti → Faz 3'te teknik kararı verilebilir (teknik etiketi doğrulanmış değil) |
| `SUSPECT` | 145 | %0,23 | kapsama < 0,6: 116 · kapsama + net: 1 · \|net\| > 700: 9 · sıçrama bayraklı: 19 |
| `INVALID` | 77 | %0,12 | geçerli örnek < 10 |
| `EMPTY` | 0 | %0 | — |
| **UNKNOWN olacak** | **222** | **%0,36** | `RELIABLE` olmayan her nota |

Nota sayısı Faz 1'in sayısıyla birebir aynı (62.476) — iki bağımsız kod yolunun çapraz doğrulaması.
Şüpheli/geçersiz notalar 360 kaydın 69'una dağılmış, tek kayıtta yığılma yok. `SUSPECT` notaların
kapsama medyanı 0,51 (en düşük 0,11): yani notanın yarısı ölçülemiyor.

### 8.2 GAPS — 6 kayıt (gaps_fit + gaps_val), salt okunur

| soru | sonuç |
|---|---|
| Monofonik/polifonik pasaj ayrımı mümkün mü? | **Evet, ama yalnız referansla.** Aktif karelerin monofonik oranı: medyan %22,5 (min %9,4 – maks %55,3); 8.356 mono kare ≈ 194 sn (aktif 33.548, polifonik 25.192). Ölçüt hizalı MIDI'dir, ses değil. |
| Önbellekteki CQT'den ince perde okunabilir mi? | Beklenen bin çevresinde tepe + parabolik enterpolasyon: medyan \|hata\| **5,8 cent**, p90 41,3, p99 200,0; %86,1'i ≤ 25 cent (n = 8.356). Çözünürlük 50 cent/bin. |
| Bağımsız F0 takipçisi (pyin) bu kurulumda işe yarıyor mu? | **Hayır, tutarsız.** `003_HSswc`: medyan 22,5 cent, %98,9 ≤ 25 cent. `064_sV1wc`: medyan 7,5 cent ama p90 **1.902 cent** (oktav hataları), yalnız %63,2 ≤ 25 cent. Maliyet 0,22–0,48× gerçek zaman. (2 kayıt × 20 sn; varsayılan parametreler.) |

**Öğrenilen:** **Denediğimiz iki yol da güvenilir çıkmadı** — mevcut önbellek CQT'si ve varsayılan
parametreli pyin. Sebebi ölçüldü: MIDI referansına göre "monofonik" olan kareler akustik olarak
monofonik değil; naylon gitarda önceki notaların sustain'i ve sempatik rezonans sürüyor, pyin bu
yüzden oktav hatası yapıyor. CQT'nin 5,8 cent'lik medyan hatası ise **perdenin bilindiği
varsayımıyla** elde edildi (yalnız beklenen bin çevresi arandı), yani bağımsız bir perde ölçümü
değil; ayrıca referans tam sayı MIDI olduğundan bu "hata", takipçi hatası ile gerçek perde sapmasını
(vibrato, akort) birbirine karıştırıyor. GAPS'te tel başına referans olmadığı için ikisini ayıracak
ve bir F0 çıktısını bağımsız sınayacak ölçüt de yok.

**Bu bir imkânsızlık kanıtı değildir.** Denenmemiş yollar var: çok sesli F0 tahmini, kaynak
ayrıştırma, daha yüksek frekans çözünürlüğüyle yeniden analiz, pyin parametre ayarı. Bu fazda
hiçbiri denenmedi — "GAPS'ten F0 çıkarılamaz" demiyoruz, "elimizdeki iki yolla güvenilir çıkmadı"
diyoruz.

**Pratik karşılığı:** GAPS, Faz 5'te **nitel doğrulama** ve referans-monofonik pasajlarda sınırlı
ölçüm için kullanılabilir; bugünkü yöntemlerle GAPS üzerinde sayısal teknik doğruluğu iddia edilemez.

## 9. Testlerde neleri doğruladık?

| koşu | sonuç |
|---|---|
| Yeni kontur testleri | **29/29 geçti** |
| Tüm birim testler (Faz 0 + 1 + 2A) | **109/109 geçti** (0,30 sn; yalnız numpy/scipy) |
| Veri protokolü kontrolü | **17/17 geçti** |
| Çıkarım regresyonu | **REGRESYON YOK** (referansla birebir) |
| Yeni dosya denetimi | 4 dosyada sözdizimi hatası ve kullanılmayan import yok |

Doğrulanan davranışlar: sabit frekans dönüşümü (440 Hz → 6900 cent, 196 Hz → 5500,02 cent); **sentetik
vibratonun korunması** (±50 cent salınım bozulmuyor, sıçrama sayılmıyor, nota `RELIABLE` kalıyor);
oktav sıçramasının yakalanması (değer değiştirilmeden); segment sınırındaki büyük farkın sıçrama
sayılmaması; 0 Hz / NaN / inf / negatif Hz'in uyarı üretmeden NaN'a gitmesi; tel aralığı dışı örneğin
bayraklanması; iki telin birbirinin örneğini almaması ve tel/nota kimliğinin korunması; `UNKNOWN`'ın
hiçbir koşulda `NORMAL` olmaması; aynı girdide iki koşunun bit düzeyinde aynı sonucu vermesi.

Bir test ilk koşuda düştü: beklenen cent değerini yanlış hesaplamıştım (196 Hz ≈ 5500 cent, 5498 değil).
Kod doğruydu, test düzeltildi.

## 10. Hangi sorunlar çözüldü, hangileri çözülmedi?

**Çözüldü — Faz 1'in 2.216 cent'lik artefaktının kaynağı bulundu: tel aralığı dışı örnekler.**
Nota içi \|net kayma\| dağılımı:

| yöntem | n | p99 | p99,9 | maks | ≥ 700 cent |
|---|---|---|---|---|---|
| 0 Hz örnekler atılmazsa (hipotetik) | 62.476 | 4.712 | 6.803 | 7.800 | 965 |
| **Faz 1** (0 Hz atılır, tel aralığı kontrolü yok) | 62.476 | 331 | 699 | **3.377**¹ | 62 |
| **Faz 2A** (geçersizler bayraklı + ayıklanmış) | 62.476 | — | — | — | **10** |
| Faz 2A, yalnız `RELIABLE` notalar | 62.254 | 289 | 497 | 699 | 0 |

¹ Faz 1 bu koşulu 60 kayıtlık alt kümede ölçüp 2.216 cent bulmuştu; 360 kaydın tamamında maksimum
3.377 cent. Aynı yöntem, daha büyük örneklem.

Tel aralığı kontrolü, 700 cent'i aşan 62 notanın 52'sini ortadan kaldırdı; kalan 10'u `SUSPECT`.
(`RELIABLE` satırındaki 699 cent üst sınırı tanım gereğidir, eşik 700.) Ayrıca çözülenler: tel kimliği
artık varsayım değil ölçüm; geçersiz örnekler açıkça işaretli; UNKNOWN/NORMAL ayrımı kodda ve testte
sabit; GAPS'te denenen iki F0 yolunun sınırı ölçülmüş durumda.

**Çözülmedi:**

1. **Eşikler doğrulanmadı.** Kullanılan değerler ve dayanakları:

   | eşik | değer | gerekçe | durum |
   |---|---|---|---|
   | `JUMP_CENTS` | 600 cent / 5,8 ms | ≈ 103 yarım ses/sn; gerçek slide'ın çok üstü | **doğrulanmadı**; \|Δcent\| dağılımı: ≥100c 1.164, ≥300c 4, ≥600c 4 (n = 3,13 M), maks 1.319 |
   | `NET_CENTS` | 700 cent | nota içi 7 yarım ses kayma beklenmez | **doğrulanmadı**; 10 notayı şüpheli yaptı |
   | `RANGE_PAD` | 0,5 yarım ses | akort sapması + hafif bend payı | fiziksel kısıt (eşik değil) |
   | `GAP_SEC` | 20 ms | adımın ~3,4 katı | ölçümle desteklendi |
   | `MIN_SAMPLES` / `MIN_COVERAGE` | 10 örnek / 0,6 | kontur şekli için alt sınır | **doğrulanmadı**; kapsama 117 notayı şüpheli yaptı (§5.1) |

2. **%99,64 `RELIABLE`, ne "kontur kusursuz" ne "teknik etiketi doğru" demek.** Kurallar yalnız
   fiziksel imkânsızlıkları yakalıyor; takipçinin 50 cent mertebesindeki ince hataları bu kurallarla
   görünmez ve nota gerçekten vibrato/ligado içeriyor mu sorusuna bu durum cevap vermez.
3. **Aday etiketlerin doğruluğu hâlâ bilinmiyor** — hiçbir örnek dinlenmedi.
4. **Slide/bend ayrımı açık.** Kontur tek başına ayırmıyor (Faz 1: 206 / 202 kırılımı).
5. **GuitarSet çelik telli akustik**; kontur istatistikleri klasik gitar için geçerli sayılamaz.
6. **GAPS'te denenen F0 yolları güvenilir çıkmadı** (§8.2) → klasik gitarda sayısal teknik
   doğrulaması şimdilik yapılamıyor; başka yöntemler denenmediği için kapı kapalı değil.

## 11. Faz 2B'de ne yapmalıyız, neden?

**Neden:** Faz 2A, F0'ın *fiziksel olarak imkânsız* kısmını ayırdı; ama "kuralın bulduğu vibrato
gerçekten vibrato mu?" sorusuna veri cevap veremiyor. Bu cevap olmadan Faz 4'ün skorları yalnız kuralı
ölçer. Dolayısıyla Faz 2B'nin çıktısı, teknik sınıflandırıcısının **üst sınırını** belirler.

**Yapılacaklar:**

1. **Tabakalı örneklem** — icracı 00–04'ten ≈ 40 vibrato + 40 ligado + 30 glide + 30 NORMAL adayı.
   Çerçeve hazır: `summary.csv` ve nota başına `nstat` / `ncov` / `nnet` dizileri.
2. **Dinleme yardımcısı** — kontur JAMS'ten geldiği için ses hiç okunmadı; `audio_mono-mic`'ten nota
   penceresi kesip dinletecek küçük bir araç gerekiyor.
3. **Ölçülecek değerler** — kural başına precision (dinlemeye karşı), %10 tekrar etiketlemeyle
   öz-tutarlılık, sınıf başına güven aralığı.
4. **Eşiklerin sabitlenmesi** — `JUMP_CENTS`, `NET_CENTS`, `MIN_SAMPLES`, `MIN_COVERAGE` ve Faz 1'in
   vibrato/glide eşikleri ancak bu doğrulamadan sonra kesinleşecek.
5. **UNKNOWN sözleşmesinin devri** — Faz 3 yalnız `RELIABLE` notalara teknik kararı verecek;
   `fallback_label()` bunu zorunlu kılıyor.

Faz 2B eğitim yapmaz; ölçüm ve doğrulama fazıdır.

---

## Çalıştırma

```bash
python -m scripts.data.build_contour --dry-run        # boyut + kalite raporu, yazmaz
python -m scripts.data.build_contour                  # 360 kayit -> data/cache/contour_guitarset (18 MB)
python -m scripts.eval.probe_gaps_f0 --records 6      # GAPS incelemesi (pyin icin --pyin-records 2)
python -m unittest discover -s tests -t .             # 104 test
```

## Durum

- [x] Faz 2A — kontur altyapısı, bayraklama, nota durumları, GuitarSet önbelleği, GAPS incelemesi, 24 test
- [ ] Faz 2B — insan doğrulaması ve etiket güvenilirliği (kural precision'ı, öz-tutarlılık)
- [ ] Faz 3 — kural tabanlı tespit + sözde etiket üretimi (UNKNOWN sözleşmesiyle)
- [ ] Faz 4 — mikrofondan teknik sınıflandırıcı (macro-F1, `EpochSelector`)
- [ ] Faz 5 — klasik gitarda doğrulama (elle etiketli küçük küme; GAPS nitel)
- [ ] Faz 6 — entegrasyon, regresyon, kök `README.md` güncellemesi, Katman 4 kapanışı
