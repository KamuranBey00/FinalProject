# Katman 4 Faz 1 — Teknik kapsamı ve etiket envanteri

> Durum (11 Ekim 2026): **Faz 1 tamamlandı.** Eğitim yok, indirme yok, önbellek/checkpoint değişmedi.
> Taban aynen `checkpoints/tabcrnn_deep2.pt`. Faz 0 protokolü uygulandı: `gaps_test` hiç okunmadı,
> `395_w41wc` karantinada. Sıradaki (onay bekliyor): **Faz 2A** — F0 temizliği ve otomatik testler (bölüm 10).

## Amaç

Katman 4 teknik tespiti yapacak. Önce şu soruyu kanıta bağlamak gerekiyordu: **hangi teknik için
elimizde etiket var?** Bu faz yalnız ölçüm yapar; sınıf listesini ve hangi tekniğin kapsamda
kalacağını belirler.

Ölçüm aracı: `scripts/eval/inventory_technique.py` (salt okunur, ~4 dk). İki kaynak tarandı —
GuitarSet'in 360 JAMS dosyası ve GAPS'in MusicXML partisyonları.

## 1. GuitarSet: doğrudan teknik etiketi yok

360 JAMS dosyasının tamamında namespace sayımı:

| namespace | adet | içerik |
|---|---|---|
| `pitch_contour` | 2160 (6 × 360) | tel başına sürekli F0, 5.8 ms adım, Hz + `voiced` bayrağı |
| `note_midi` | 2160 (6 × 360) | tel başına nota (başlangıç, süre, **kesirli** MIDI) |
| `beat_position` | 360 | vuruş/ölçü konumu (Katman 5'in verisi) |
| `tempo` | 360 | tempo |
| `chord` | 720 | akor (biri yarı otomatik + elle doğrulanmış) |
| `key_mode` | 360 | ton |

**Teknik namespace'i yok.** Yani GuitarSet'te vibrato/slide/ligado/armonik için gerçek etiket
bulunmuyor; elimizdeki tek sinyal tel başına sürekli F0'dır. Yerel ses yalnız `audio_mono-mic`
(1.6 GB); hex pickup sesi indirilmemiş — etiket hex kanalından türetilip giriş mono mikrofon
olacağı için bu, istenen kurulumla uyumlu.

## 2. GuitarSet konturundan türetilen aday olaylar

62.476 nota, 6 icracı. Kurallar aşağıda; **bunlar aday sayılarıdır, gerçek etiket değildir.**

| teknik | aday | notaların % | icracı | kayıt (360'ta) |
|---|---|---|---|---|
| VIBRATO | 3.986 | %6,38 | 6/6 | 354 |
| HAMMER_ON | 2.019 | %3,23 | 6/6 | 304 |
| PULL_OFF | 1.896 | %3,03 | 6/6 | 318 |
| GLIDE (slide + bend birleşik) | 408 | %0,65 | 6/6 | 191 |
| HARMONİK | — | — | — | türetilemez (tınısal; konturda izi yok) |
| NORMAL (kalan) | ≈ 54.167 | ≈ %86,7 | 6/6 | 360 |

İcracı kırılımı (nota / VIBRATO / GLIDE / HAMMER_ON / PULL_OFF):

| icracı | nota | VIBRATO | GLIDE | HAMMER_ON | PULL_OFF |
|---|---|---|---|---|---|
| 00 | 13.223 | 736 | 82 | 579 | 415 |
| 01 | 11.268 | 620 | 40 | 278 | 295 |
| 02 | 9.659 | 725 | 84 | 381 | 423 |
| 03 | 9.358 | 627 | 63 | 173 | 217 |
| 04 | 10.253 | 674 | 81 | 399 | 362 |
| 05 (kalibrasyon) | 8.715 | 604 | 58 | 209 | 184 |

Sınıflar kesişebilir (bir nota hem ligado hedefi hem vibrato adayı olabilir); NORMAL kalan olarak
hesaplandığı için yaklaşıktır.

### Kullanılan kurallar

Nota içi kontur sapması (cent, notanın medyanına göre), kenar korumalı 5 örneklik kayan ortalama:

- **GLIDE** — net kayma ≥ 100 cent **ve** tek yönlülük (|net| / toplam değişim) ≥ 0,8.
- **VIBRATO** — tepe-tepe (p95−p5) ≥ 40 cent, ≥ 4 yön değişimi, salınım hızı 3–9 Hz.
- **HAMMER_ON / PULL_OFF** — aynı telde ardışık iki nota; arada ≤ 50 ms boşluk, 1–5 yarım ses
  aralık, sınırda kontur kesintisiz (örnekler arası boşluk ≤ 30 ms). Yön: Δperde > 0 → hammer-on.
- Tüm kurallar ≥ 0,25 sn süreli ve ≥ 10 kontur örneği olan notalara uygulanır.

### Eşiklerin ne kadar belirleyici olduğu (icracı 00, 13.223 nota)

| eşik seti | VIBRATO | GLIDE |
|---|---|---|
| gevşek (30 cent / 75 cent) | 914 | 105 |
| **varsayılan (40 / 100)** | **736** | **82** |
| sıkı (60 cent / 150 cent) | 517 | 54 |

Vibrato sayısı eşikle ~2 kat değişiyor → mutlak sayı değil, sıralama anlamlı. Nota içi hareketin
dağılımı (6.031 nota): |net kayma| medyan 5,9 cent, p90 69 cent, p99 299 cent; tepe-tepe salınım
medyan 16,8 cent, p90 114 cent. Yani tipik nota perdece durağan, kuyruk ince — GLIDE'ın az
çıkması eşik seçiminden değil veriden geliyor.

### Güvenilirlik ve kalan kuşkular

- **Aday ≠ gerçek etiket.** Kural ne üretirse onu ölçersek metrik şişer; Faz 5'teki dinleme
  doğrulaması zorunlu.
- JAMS'te annotator/curator alanları **boş**; konturun doğruluğu hex kanalı üzerindeki perde
  takipçisine bağlı ve bu seansta doğrulanmadı.
- Kontur oktav hatası kanıtı: nota içi |net kayma| maksimumu **2.216 cent** (22 yarım ses) — bu
  fiziksel değil. Faz 2'de oktav/atlama filtresi şart.
- GuitarSet **çelik telli akustik**; buradan öğrenilen teknik sayıları klasik gitar doğruluğu
  olarak sunulamaz.
- GLIDE içinde **slide ve bend ayrışmıyor**: ikisi de sürekli perde kayması. Tek ipucu, kaymanın
  tam yarım ses katına inip inmediği — 408 adayın **206'sı** yarım sese iniyor (slide profili),
  **202'si** inmiyor (bend/portamento profili). Bu ayrım tek başına sınıflandırma için zayıf.

## 3. GAPS: sembolik teknik etiketi pratikte yok

Taranan: **230 kayıt** (gaps_fit 202 + gaps_val 28), 170 icracı, 386.296 nota.
`gaps_test` (30 kayıt) protokol gereği **okunmadı**; 39 icracı-örtüşmeli kayıt önbellekte yok;
`395_w41wc` karantinada. (300 MusicXML = 230 tarandı + 30 test + 39 dışlanan + 1 karantina.)

| etiket | işaret | kayıt | yorum |
|---|---|---|---|
| `slur` (ligado) | 80 öğe = **40 bağ** | **3** | 282_Gw1wc (18), 356_6k1wc (14), 005_dSswc (8) |
| `tremolo` | 34 | **1** | 049_Fc1wc |
| `slide`, `glissando`, `bend`, `hammer-on`, `pull-off` | **0** | 0 | hiç yok |
| `harmonic` | **0** | 0 | hiç yok |
| `trill-mark`, `mordent`, `turn`, `tenuto` | **0** | 0 | hiç yok |
| `staccato` | 3.660 | 39 | sağ el artikülasyonu, sol el tekniği değil |
| `accent` | 3.726 | 69 | dinamik |
| `fermata` | 960 | 102 | süre işareti |

Sonuç: klasik gitar veri setinde **sol el tekniği denetimi yok**. Klasik notasyonda ligado zaten
`slur` ile yazılır, ama kayıtların yalnız %1,3'ünde var (3/230) — eğitim için kullanılamaz,
en iyi hâlde küçük bir sağlama kümesi.

## 4. Harici veri havuzu: doğrulanmamış

GOAT, IDMT-SMT-GUITAR, Guitar-TECHS ve AG-PT-set **yalnız sunumun 8. slaydında** geçiyor;
repoda (kod, doküman, indirme script'i) **tek referans yok**, yerel kopya yok. Slaytın kendisi de
bu kaynakların lisans ve anotasyon kalitesi açısından "final eğitim öncesi tekrar doğrulanacağını"
yazıyor. Bu yüzden içerikleri hakkında kesin bir şey söylenemez; Faz 1 bunları hesaba katmadı.

## 5. Teknik başına karar tablosu

| teknik | gerçek etiket? | türetilebilir? | örnek | icracı | klasik uygunluk | güvenilirlik | dengesizlik / overfit riski | ek veri gerekli? |
|---|---|---|---|---|---|---|---|---|
| VIBRATO | yok | ✓ kontur salınımı | 3.986 | 6 | **yüksek** (klasik icranın temel tekniği) | orta (eşiğe duyarlı, dinleme yok) | orta: notaların %6'sı | dinleme doğrulaması |
| HAMMER_ON | yok | ✓ aynı tel + kesintisiz kontur | 2.019 | 6 | **yüksek** (ligado) | orta-iyi (kural fiziksel) | orta: %3,2 | dinleme doğrulaması |
| PULL_OFF | yok | ✓ aynı yol, ters yön | 1.896 | 6 | **yüksek** (ligado) | orta-iyi | orta: %3,0 | dinleme doğrulaması |
| SLIDE | yok | kısmen (GLIDE içinde) | ≤ 206 | 6 | orta (klasikte seyrek) | **düşük** (bend'den ayrışmıyor) | **yüksek**: %0,3, 191 kayda yayılmış | evet: elle etiket |
| BEND | yok | hayır (slide ile aynı iz) | ≤ 202 | 6 | **düşük** (naylon telde fiziksel olarak zor) | **düşük** | **yüksek** | evet |
| HARMONİK | yok | hayır (tınısal) | 0 | 0 | orta-yüksek (klasikte kullanılır) | — | — | evet: elle etiket / ek veri seti |
| NORMAL | dolaylı (kalan) | ✓ | ≈ 54.167 | 6 | — | kuralların doğruluğuna bağlı | **baskın sınıf %86,7** | hayır |

## 6. Kapsam kararı (üç grup)

**A. Mevcut verilerle geliştirilebilir** — VIBRATO, HAMMER_ON, PULL_OFF, NORMAL.
Üçü de 6 icracıda ve 300'den fazla kayıtta geçiyor, kural fiziksel olarak savunulabilir, sayılar
eğitim için yeterli. Faz 3'ün sözde etiketleri bu dört sınıfla üretilecek.

**B. Ek etiket/veri gerektirir** — SLIDE (ve genel olarak GLIDE).
408 aday az, 191 kayda dağılmış ve bend'den ayrışmıyor. Ayrıca her sınıf için klasik gitarda
geçerlilik kanıtı yok (GAPS'te etiket yok) → Faz 5'te küçük elle etiketli küme zorunlu.

**C. Şimdilik kapsam dışı** — BEND ve HARMONİK.
Bend naylon telde fiziksel olarak marjinal ve konturda slide'dan ayrışmıyor; armonik tınısal bir
olay, konturda izi yok ve iki veri setinde de sıfır etiket var. İkisi de `Technique` enum'unda
tanımlı kalır ama Faz 3–4'te üretilmez; `technique_eval` boş sınıfları "tanımsız" olarak raporlar.

## 7. Sınıf dengesizliği ve overfitting

- NORMAL %86,7 → doğruluk ve weighted-F1 anlamsız; Faz 0'da ana ölçüt zaten **macro-F1**
  (+ `macro_f1_no_normal`). Taban karşılaştırması `majority_baseline` ile yapılacak.
- Sınıf ağırlığı için yeni kod gerekmiyor: mevcut `class_weights` (`inv` / `sqrt`, `cap`) ve
  `train_onset --oversample` yeniden kullanılacak.
- **Asıl risk ölçümde:** sözde etiketler kuralın kendisinden geliyor; aynı kuralla ölçülürse model
  yalnız kuralı öğrenip yüksek skor alır. Faz 4'ün raporu, Faz 5'in dinleme doğrulamalı alt
  kümesine karşı verilecek.
- İcracı dağılımı dengesiz (HAMMER_ON: icracı 00'da 579, icracı 03'te 173) → icracı-ayrık bölme
  (00–04 eğitim, 05 kalibrasyon) korunacak; icracı başına sonuç ayrı raporlanacak.

## 8. Protokol uyumu

- `gaps_test` hiç açılmadı; envanter kararları yalnız `gaps_fit` + `gaps_val` ve GuitarSet
  00–04 üzerinden verildi (oyuncu 05 sayıları bilgi amaçlı ayrı satırda).
- `395_w41wc` karantinada; dosya listesi `gaps_training_files()` ile alındı.
- Ölçüm script'i hiçbir dosyaya yazmıyor; önbellek, checkpoint ve ham veri değişmedi.

## 9. Faz 2 önerileri (sürekli F0 altyapısı)

1. **Kontur önbelleği** — `gtab/data/contour.py` + `scripts/data/build_contour.py`:
   GuitarSet için JAMS `pitch_contour`'u (sesten yeniden çıkarmaya gerek yok), GAPS için mono F0.
   Yeni dizin (`data/cache/contour_*`); mevcut önbellek dokunulmaz.
2. **Oktav/atlama filtresi** — ardışık kontur örnekleri arasında > 600 cent sıçrama veya nota içi
   > 700 cent net kayma → aday reddi. Gerekçe: ölçülen 2.216 cent maksimum.
3. **Zaman çözünürlüğü kararı** — kontur 5,8 ms, model karesi 23,2 ms. Teknik kararı nota düzeyinde
   verildiği için (Faz 0 birimi) kontur kendi çözünürlüğünde tutulup nota başına özetlenecek;
   kareye indirgeme yapılmayacak.
4. **GAPS mono F0'ın sınırı ölçülecek** — GAPS'te tel başına referans yok; polifonik pasajlarda
   mono F0 güvenilmez. Faz 2'nin kabul ölçütü: hizalı MIDI perdesine göre medyan sapma ≤ 25 cent
   ve yalnız monofonik pasajlarda kullanım.
5. **Dinleme doğrulama örneklemi hazırlanacak** — icracı 00–04'ten tabakalı örnek
   (≈ 40 VIBRATO + 40 ligado + 30 GLIDE + 30 NORMAL), kullanıcı dinler. Bu, Faz 4'ün tavanını
   belirler ve Faz 1 aday sayılarının ne kadarının gerçek olduğunu söyler.
6. **Faz 3 için sınıf listesi önerisi** — `VIBRATO`, `HAMMER_ON`, `PULL_OFF`, `NORMAL`
   (GLIDE yalnız ölçüm/tanılama olarak taşınır, eğitim sınıfı değil). Hammer-on ve pull-off'un
   ayrı tutulması öneriliyor: yön Δperde'nin işaretinden geliyor ve ikisi de ≥ 1.800 örnek.

## 10. Faz 1 sonrası güvenilirlik kararları

Faz 1'in sayıları aday sayılarıdır; aşağıdaki kararlar Faz 2 ve sonrasını bağlar.

- **Aday ≠ gerçek etiket.** Bölüm 2'deki sayılar kural çıktısıdır. Her raporda etiket kaynağı
  `turetilmis` olarak taşınacak ve `gercek` (insan/hex doğrulamalı) etiketle aynı tabloda
  gösterilmeyecek (`technique_eval.assert_not_mixed`).
- **Hammer-on / pull-off yalnız perde yönüyle belirlenemez.** Δperde'nin işareti yalnızca *yönü*
  verir; olayın ligado olup olmadığı ayrı kanıt ister (konturun sınırda kesintisiz olması +
  yeni vuruş bulunmaması). İki kanıt çelişirse sınıf atanmaz.
- **NORMAL ile UNKNOWN ayrı tutulacak.** NORMAL = tekniksiz olduğu kanıtlanmış nota;
  UNKNOWN = kuralın karar veremediği nota (kısa süre, az kontur örneği, oktav artefaktı, çelişen
  kanıt). UNKNOWN eğitim hedefi olmayacak, metrikte ayrı satır olarak raporlanacak.
- **Şüpheli örnekler otomatik NORMAL yapılmayacak.** Aksi hâlde baskın sınıf kirlenir ve macro-F1
  olduğundan iyi görünür.
- **Az ama doğru etiket önceliklidir.** Eşikler kapsamı (recall) değil doğruluğu (precision)
  önceleyecek biçimde sıkı tarafta seçilecek; gevşetme ancak Faz 2B'nin doğrulama sonucundan sonra
  tartışılacak.

**Faz 2A — F0 temizliği ve otomatik testler.** Kontur önbelleği, oktav/atlama filtresi (bölüm 2'de
ölçülen 2.216 cent kanıtı), kontur boşluk kuralı, UNKNOWN üretimi; sentetik vibrato/glissando
sinyalleriyle birim testler. Eğitim yok, ölçüm yok.

**Faz 2B — insan doğrulaması ve etiket güvenilirliği.** Tabakalı örneklem (icracı 00–04),
kullanıcı dinlemesi, kural başına precision, %10 tekrar etiketlemeyle öz-tutarlılık. Çıkan değer
Faz 4'ün üst sınırıdır; bu ölçüm yapılmadan teknik doğruluğu raporlanmayacak.

## Çalıştırma

```bash
python -m scripts.eval.inventory_technique                  # tam tarama (~4 dk)
python -m scripts.eval.inventory_technique --limit 20       # hizli deneme
```

## Durum

- [x] Faz 1 — GuitarSet namespace envanteri, kontur tabanlı aday sayımı, eşik duyarlılığı,
      GAPS MusicXML teknik etiket sayımı, harici kaynakların doğrulanması, kapsam kararı
- [ ] Faz 2A — F0 temizliği, oktav/atlama filtresi, UNKNOWN üretimi, otomatik testler
- [ ] Faz 2B — insan doğrulaması ve etiket güvenilirliği (kural precision'ı, öz-tutarlılık)
- [ ] Faz 3 — kural tabanlı tespit + sözde etiket üretimi
- [ ] Faz 4 — mikrofondan teknik sınıflandırıcı (macro-F1, `EpochSelector`)
- [ ] Faz 5 — klasik gitarda doğrulama (elle etiketli küçük küme)
- [ ] Faz 6 — entegrasyon, regresyon, kök `README.md` güncellemesi, Katman 4 kapanışı
