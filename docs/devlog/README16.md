# Katman 3.13 Devamı — Sol El Parmaklandırması / P1 İnceleme ve Yeniden Planlama

> **Durum (11 Ekim 2026):** P1 (inceleme) tamamlandı, P2 yapıldı (günlüğü: `README17.md`).
> Kod davranışı, model ve veri değişmedi. Dal `Armagan-katman-4`, HEAD `b988d82`.
> Bu dosya, teknik tanıma günlüğünün yerini aldı; eski içerik arşivde (bölüm 8).

## 1. Neden teknik tanıma beklemeye alındı

Katman 4 Faz 1 (teknik etiket envanteri) tek bir soruyu araştırdı ve kritik bir sonuçla döndü:
**teknik etiketi yok.** GuitarSet'in 360 JAMS dosyasında teknik namespace'i bulunmuyor; GAPS'in
230 partisyonunda slide / glissando / bend / hammer-on / pull-off / harmonic işaretlerinin tamamı
sıfır (yalnız 3 kayıtta `slur`, 1 kayıtta `tremolo`). Ayrıntılı envanter ve Faz 2A'nın GAPS F0
ölçümleri arşivdeki günlüklerde (bölüm 8).

Sonucu: teknik tespiti ancak **kuraldan türetilmiş sözde etiketlerle** yapılabilir, doğruluğu
yalnız insan dinlemesiyle sınanabilir ve **klasik gitarda sayısal bir doğruluk iddia edilemez.**

Buna karşılık **parmaklandırmada gerçek etiket var**: GAPS partisyonlarında 165 kayıt,
159.647 nota, bunların **38.828'i (%24,3) parmak numaralı**. Aynı emek veri-zengin bir hedefe
yöneltilebilir. Teknik tanıma bu nedenle **durdurulmadı, beklemeye alındı**; Faz 0–2A çıktıları
yerinde ve arşivli (bölüm 8).

## 2. Katman 4'ün kapsamı değişmedi

Belgelerin tamamı Katman 4'ü **teknik tanıma** olarak tanımlıyor ve bu tanım korunuyor:

| Kaynak | İfade |
|---|---|
| `docs/devlog/README.md:23-24` | "Katman 4 — Teknikler. Sürekli F0 eğrisi + onset zarfı → önce BEND, sonra slide, hammer/pull, vibrato." |
| kök `README.md:226-227` | aynı tanım (güncel yol haritası) |
| `gtab/core/note_event.py:19, 44` | "Gitar teknikleri. **Katman 4'te tespit edeceğiz**" |
| `gtab/config.py:37-39` | "Teknik tespiti için sürekli F0 … Katman 4'te (bend/slide) hayati" |
| `gtab/data/labels.py:10-13` | "Katman 4'te hammer-on/pull-off'u onset **yokluğundan** ayırt edeceğiz" |

Parmaklandırma Katman 4 **değildir**; sunumda ayrı bir modül (Modül 7 "Fingering Optimizer",
slayt 9) ve Phase 5'in parçası (slayt 10). Slayt 13 hem "Playability + left-hand fingering"
hem "Selected classical techniques" kalemlerini **"Başarırsa ekle"** kolonunda tutuyor; ikisi de
çekirdek hedef değil. Çekirdek hedef (slayt 13 "Minimum güçlü başarı" = polifonik nota
transkripsiyonu + anlamlı tel/perde TAB + objektif benchmark) **zaten sağlandı**
(GAPS nota F1 0,779; GuitarSet tab F1 0,777).

Bu yüzden parmaklandırma çalışması **Katman 4 olarak numaralanmıyor**; Katman 3.13'ün devamı
olarak yürütülüyor. (Tek belge çelişkisi: `README9.md:41` parmaklandırma verisini "Katman 7'ye
girdi" diye işaretliyor, ama "Katman 7" hiçbir yerde tanımlı değil; yol haritası Katman 6'da
bitiyor.)

## 3. Parmaklandırma Katman 3.13'te zaten var

`README14.md:112` açıkça yazıyor: *"Sunumun 4. (çalınabilirlik) ve 7. (parmaklandırma) modülleri
tek katmanda."* Katman 3.13 Adım 2'de **"klasik edisyon önerisi"** adıyla sembolik bir model
kabul edildi (`checkpoints/edition_seq.pt`). Sıfırdan başlamıyoruz; mevcut bir taban var.

## 4. P1 incelemesinin bulguları

**Girdi** (`gtab/models/edition_net.py:99-114`, `gtab/decoding/edition.py:28-42`): model **sesi
hiç görmüyor.** Girdi yalnız nota dizisi `(başlangıç, bitiş, perde)` + akor bayrağı. Türetilen
özellikler: perde gömmesi (E2–E6), akor bayrağı/boyu/sıra, başlangıç aralığı, süre, perde farkı,
o perdenin çalınabildiği geçerli teller. Akor = başlangıçları ≤ **30 ms** (`edition.py:22`), akor
içinde perde artan.

**Çıktı** (`edition_net.py:113-114`): **tel** (0–5), **perde** (pitch − açık tel), **parmak**
(0 = boş tel, 1–4). **El pozisyonu çıktı olarak üretilmiyor**; yalnız reddedilen log-lineer
modelin iç özelliğiydi ("işaret parmağının perdesi = perde − parmak + 1", `edition.py:16, 118`).

**Etiketlerin okunması:** `gtab/data/gaps_score.py:68-78` MusicXML `notations/technical/fingering`
alanını okur; `:110-125` parmağı nota portresinden `(ölçü, tick, perde)` anahtarıyla TAB notasına
taşır; `:215-250` `align()` partisyonu syncpoint + kapo + ±80 ms ince eşleme ile hizalı MIDI'ye
bağlar (tel = partisyon TAB'ı, perde = MIDI perdesi − açık tel).
`scripts/data/build_gaps_tab.py` `data/cache/gaps_tab/` altına yazar. Doğrulama: perde roll uyumu
%98,93, **"parmak 0 ⇔ boş tel" tutarlılığı %99,4**, eşleşme medyanı 16 ms.

**Eğitim** (`scripts/train/train_edition_seq.py:88-110`): yalnız **`gaps_fit`** (127 kayıt,
125 bin nota); 256 notalık kesitler, Adam `lr=1e-3`, batch 16, 30 epoch, seed 1, ~2 dk. Seçim
**`gaps_val`**'de (tel + parmak ortalaması) → epoch 17. `gaps_test` eğitimde ve seçimde
kullanılmadı. Parmak kaybı yalnız `(tel etiketli) & (parmak ≥ 1) & (perde > 0)` notalarda.

## 5. Üç sonucun koşulları (karıştırılmamalı)

Üçü de **gaps_test** (nihai rapor split'i) ve hepsi aynı nota alt kümesinde: partisyonda tel
etiketli, **basılı** (boş tel hariç), parmak numaralı notalar — gaps_test'te 4.466 nota.
Metrik tanımları `scripts/eval/eval_edition.py:171-240`.

| Sonuç | Koşul | Ne ölçüyor |
|---|---|---|
| **%63,6** | **Konum ORACLE**: partisyonun tel/perdesi verilir, yalnız parmak tahmin edilir (`eval_edition.py:191`) | Parmak seçiminin tek başına kalitesi. Taban ("perde − pozisyon + 1") aynı koşulda %56,6 |
| **%43,4** | Konumu **model seçer**; doğru = **tel + perde + parmak üçü birden** | Ürün doğruluğu, gerçek notalar verilmişken. Aynı konumda taban %37,6 |
| **%33,1** | **Uçtan uca**: notalar `deep2` ile sesten bulunur, sonra konum + parmak; tüm parmaklı notalarda (`eval_edition.py:205-240`) | Gerçek ürün zinciri. Bulunan notalarda %39,7; deep2 etiketli notaların %81,7'sini buluyor |

Ayrıca tel uyumu (boş teller dahil, tel etiketli tüm notalar): gaps_test **0,815** (naif 0,745).

## 6. Eksikler ve riskler (P1)

1. **Entegrasyon yok.** `edition` modüllerini yalnız `scripts/eval/eval_edition.py` ve iki eğitim
   script'i import ediyor. Ses → TAB üretim yolu (`gtab/decoding/decode.py`,
   `gtab/models/inference.py`, `scripts/eval/eval_hmm.py`) parmaklandırmayı **tanımıyor**.
2. **Ortak şema parmak taşımıyor.** `gtab/core/note_event.py:36-50`'de `string`, `fret`,
   `technique` var; **parmak alanı yok**. `render_ascii_tab` da parmak göstermiyor.
3. **Fiziksel çalınabilirlik denetlenmiyor.** Sert kısıtlar yalnız: enstrüman aralığı
   (`core/instrument.py:pitch_to_positions`), akor içi tel teklik (`edition_net.py:82-96`),
   boş tel ⇔ parmak 0 (`:75-78`). **Yok:** el açıklığı sınırı, parmak sırası tutarlılığı,
   **akor içi aynı parmak yasağı** (parmaklar akor düzeyinde değil nota başına argmax ile
   seçiliyor, `edition_net.py:114`), ihlal / hareket maliyeti metrikleri (sunum slayt 12).
4. **Yumuşak maliyetler yalnız reddedilen modelde açık.** El pozisyonu değişimi, aynı parmakla
   kayma, barre, uzun boşluk `edition.py:103-126`'da açık özellik; kabul edilen BiLSTM'de tamamı
   veriden örtük → sunumun "açıklanabilir optimizasyon" iddiası (slayt 11) şu an karşılıksız.
5. **Referans mutlak doğru değil.** Partisyonda parmak işaretli notalar editörlerin "belirgin
   olmayan" bulduğu yerler; tek doğru parmaklandırma yok (slayt 10). Uzman değerlendirmesi hiç
   yapılmadı.
6. **Testsizlik** — P1 denetiminde parmaklandırma zincirinin hiçbir otomatik testi yoktu
   (**P2'de kapatıldı**).

## 7. P2–P6 yol haritası

| Faz | İş | Kabul ölçütü |
|---|---|---|
| **P2** ✅ | Mevcut davranışı sabitleyen testler + checkpoint doğrulaması + yayımlanmış metriklerin yeniden üretimi (davranış değişikliği yok) | `gaps_val` tel/parmak/ürün değerleri birebir üretildi; mevcut testler bozulmadı |
| **P3** | **Çalınabilirlik denetimi (yalnız ölçüm):** el açıklığı, akor içi parmak çakışması, parmak sırası, pozisyon kayması → ihlal oranı + hareket maliyeti | Mevcut model / naif taban / partisyon (editör) ihlal oranları yan yana raporlanır |
| **P4** | **Akor içi parmak kısıtı** (telde var olan çakışma çözümünün parmağa taşınması) + el pozisyonu çıktısı | İhlal oranı düşer, `gaps_val` parmak doğruluğu düşmez |
| **P5** | **Açıklanabilir optimizasyon** (DP / beam search; slayt 11) ile BiLSTM'in karşılaştırılması | Açıklanabilir model ≥ BiLSTM − 0,02 ise tercih edilir |
| **P6** | **Entegrasyon:** `NoteEvent`'e parmak / el pozisyonu, TAB render'ında gösterim, uçtan uca ürün sayısı + uzman değerlendirmesi | Eski çıkarım yolu parmak izi farksız; uzman kabul oranı raporlanır |

Tüm fazlarda Faz 0 protokolü geçerli: eğitim `gaps_fit`, geliştirme/seçim `gaps_val`,
**`gaps_test` yalnız kapanışta bir kez**, `395_w41wc` karantinada (`gtab/evaluation/protocol.py`).

## 8. Arşivlenen günlükler ve korunan çıktılar

Teknik tanıma çalışmasının günlükleri arşive taşındı; **hiçbir kod, veri veya ölçüm silinmedi.**

| Arşiv dosyası | Eski adı | İçerik |
|---|---|---|
| `docs/devlog/archive/README16_katman4_faz1_teknik_envanteri.md` | README16 | Katman 4 Faz 1 — teknik kapsamı ve etiket envanteri |
| `docs/devlog/archive/README17_katman4_faz2a_f0_altyapisi.md` | README17 | Katman 4 Faz 2A — sürekli F0 altyapısı ve kontur temizliği |

Kopyalar `sha256` ve `cmp` ile **birebir** doğrulandı. `README15.md` (Faz 0 — ölçüm ve test
protokolü) **değiştirilmedi**; hâlâ geçerli ve bu çalışmanın da protokol kaynağı.
README1–README14 dosyalarına dokunulmadı.

Beklemeye alınan teknik tanıma kodu ve verisi yerinde: `gtab/evaluation/technique_eval.py`,
`gtab/data/contour.py`, `scripts/data/build_contour.py`, `scripts/eval/inventory_technique.py`,
`scripts/eval/probe_gaps_f0.py`, `data/cache/contour_guitarset/` (18 MB) ve ilgili testler.
Hedef-bağımsız ölçüm altyapısı (`gtab/evaluation/{protocol,baselines,fingerprint,selection}.py`)
bu çalışmada da kullanılıyor.

> Arşivdeki README17'de bilinen bir sayı hatası var (satır 189): "kaydın 69'una dağılmış"
> ifadesindeki 69 yanlış etiketlenmişti ("SUSPECT sayısı **artan** kayıt"). Doğrusu ölçüldü:
> ≥1 SUSPECT **84**, ≥1 INVALID/EMPTY **57**, birleşim **121** kayıt. Arşiv tarihsel kayıt
> olduğu için düzeltilmedi.
