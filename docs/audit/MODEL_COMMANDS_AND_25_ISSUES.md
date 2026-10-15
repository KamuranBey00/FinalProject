# Model Çalıştırma Komutları ve 25 Sorunun Açıklaması

> 14 Ekim 2026 · `Armagan-katman-4` @ `546c7bd`
> Komutların tamamı script'lerin gerçek `argparse` tanımlarından doğrulandı; uydurulmuş parametre yok.
> Hiçbir komut çalıştırılmadı, kod/checkpoint değiştirilmedi.

---

# 1. MODELİ NASIL ÇALIŞTIRACAĞIZ?

## 1.1 Dokuz soruya kısa cevap

| Soru | Cevap |
|---|---|
| **Ana model hangisi?** | `TabCRNNOnset` — derin sürüm (CNN + 2 artık blok + BiLSTM, dört çıkış kafası: tab, onset, offset, perde-onset) |
| **Checkpoint nerede?** | `checkpoints/tabcrnn_deep2.pt` + ayar dosyası `checkpoints/tabcrnn_deep2.cal.json`. Parmaklandırma için `checkpoints/edition_seq.pt`, tel geçişi için `checkpoints/transitions.npz` |
| **Eğitimi hangi dosya başlatıyor?** | Ses modeli: `scripts/train/train_onset.py` · Parmaklandırma: `scripts/train/train_edition_seq.py` |
| **Hangi datasetler eğitimde?** | `--tab-splits train,train_comp` (GuitarSet oyuncu 00–04, 150+150 kayıt) ve `--pitch-splits gaps_train` (GAPS; script bunu kendi içinde `gaps_fit`/`gaps_val` olarak ikiye bölüyor — `train_onset.py:195`) |
| **Validation hangi kodla?** | Eğitim sırasında otomatik: `train_onset.py` her epoch'ta doğrulama skorunu hesaplıyor ve **yalnız skor iyileşirse** kaydediyor (satır 329-330). Epoch 0 (başlangıç modeli) da yarışıyor (satır 259) → hiçbiri geçemezse taban korunuyor |
| **Nota doğruluğunu hangi kod ölçüyor?** | `scripts/eval/eval_pitch.py` — nota ve kare F1, polifoni kırılımı, tekrar kaçma |
| **Tel/perde doğruluğunu hangi kod ölçüyor?** | `scripts/eval/diagnose_strings.py` (tel karışıklık matrisi, tel başına doğruluk) · GAPS tarafı için `scripts/eval/eval_gaps_tab.py` |
| **Genel TAB başarısını hangi kod ölçüyor?** | `scripts/eval/eval_hmm.py` — kare düzeyinde tab F1 (ana GuitarSet ölçümü), üç yöntemi yan yana karşılaştırıyor |
| **Önceki modelle nasıl karşılaştıracağız?** | `docs/baselines/d1_baseline.json` referans değerleri tutuyor; `gtab/evaluation/baselines.py` içindeki `assert_comparable` farklı metrik gruplarının yan yana konmasını engelliyor. Toleranslar: aynı checkpoint + aynı kod **0,002**, yeniden eğitim **0,01** |

## 1.2 Çalıştırma sırası

| Sıra | Çalıştırılacak dosya | Ne yapıyor? | Kullanılacak veri |
|---|---|---|---|
| 1 | `tests/` (unittest) | 170 testi koşar; kodun bozulmadığını doğrular | veri gerekmez |
| 2 | `scripts/eval/check_protocol.py` | Veri sızıntısı ve karantina denetimi (17 kontrol) | önbellek + GAPS metadata |
| 3 | `scripts/eval/check_inference_regression.py` | Modelin çıktısının bit düzeyinde değişmediğini kontrol eder | `val` ilk 2 kayıt |
| 4 | `scripts/eval/diagnose_strings.py` | **Tel başına doğruluk + karışıklık matrisi** (S-01'in ölçümü) | `val`, `val_comp` |
| 5 | `scripts/eval/diagnose_pitch.py` | Kaçan/hayalet notaları nedenine göre kırar (S-04, S-06'nın ölçümü) | `gaps_val`, `val_comp` |
| 6 | `scripts/eval/eval_pitch.py` | Nota ve kare F1 + polifoni kırılımı | `gaps_val`, `val`, `val_comp` |
| 7 | `scripts/eval/eval_hmm.py` | **Ana TAB başarısı** (kare düzeyi tab F1) | `val` |
| 8 | `scripts/eval/eval_gaps_tab.py` | GAPS'te tel doğruluğu (oracle + naif karşılaştırması) | `gaps_val` |
| 9 | `scripts/eval/eval_playability.py` | Parmaklandırmanın fiziksel çalınabilirliği | `gaps_val` |
| 10 | `scripts/train/train_onset.py` | **Ses modeli eğitimi** (yeni checkpoint üretir) | `train`, `train_comp`, `gaps_fit` |
| 11 | `scripts/train/train_edition_seq.py` | Parmaklandırma modeli eğitimi (~2 dk) | `gaps_fit` (`gaps_tab` etiketleri) |

## 1.3 Komutlar

### Adım 1-3 — Regresyon kapısı (önce bunlar, hepsi hızlı)

```bash
.venv/Scripts/python -m unittest discover -s tests -t .
```
170 testi koşar; `.venv` ile çalıştırılmalı çünkü sistem Python'ında `mir_eval` paketi yok (S-23).

```bash
python -m scripts.eval.check_protocol
```
Gerçek önbellekte veri sızıntısı, karantina ve split rollerini denetler; hiçbir şey yazmaz.

```bash
python -m scripts.eval.check_inference_regression
```
`tabcrnn_deep2.pt`'ın çıktısını kayıtlı parmak iziyle karşılaştırır; "REGRESYON YOK" beklenir.

### Adım 4-5 — Ölçüm (eğitim yok; önce bunlar yapılmalı)

```bash
python -m scripts.eval.diagnose_strings --ckpt tabcrnn_deep2.pt --splits val val_comp --threshold 0.8
```
Tel karışıklık matrisini ve tel başına doğruluğu çıkarır; eşik 0,8 eski ölçümle (README10) karşılaştırılabilir olsun diye.

```bash
python -m scripts.eval.diagnose_strings --ckpt tabcrnn_deep2.pt --splits val val_comp --threshold 0.7
```
Aynı ölçüm, bu kez GuitarSet için seçilmiş gerçek eşikle (0,7).

```bash
python -m scripts.eval.diagnose_pitch --ckpt tabcrnn_deep2.pt --load-cal checkpoints/tabcrnn_deep2.cal.json --splits gaps_val val_comp
```
Kaçan notaları nedenine göre kırar ("perde hiç aktif değil", onset kaçtı, oktav seçildi, fragman).

### Adım 6-9 — Doğruluk ölçümleri

```bash
python -m scripts.eval.eval_pitch --ckpt tabcrnn_deep2.pt --load-cal checkpoints/tabcrnn_deep2.cal.json --splits gaps_val val val_comp
```
Nota ve kare F1'i ölçer. **`--splits` mutlaka verilmeli** — script'in varsayılanı `gaps_test val val_comp` ve test setini açar.

```bash
python -m scripts.eval.eval_hmm --model crnn --ckpt tabcrnn_deep2.pt \
    --threshold 0.7 --onset-thr 0.2 --off-ratio 1.0 --refractory 0 --fallback 10 \
    --peak --reattack 0.3 --rise-keep 0.0 --rise-split 8.0 --offset-thr 0.7 \
    --combine noisyor --peak-pick --prominence 0.25 --min-dist 2 --re-valley 0.0 \
    --onset-source pitch
```
Ana TAB başarısını (tab F1) ölçer; `eval_hmm` ayar dosyasını okuyamadığı için bütün parametreler `tabcrnn_deep2.cal.json`'daki `val` bölümünden elle yazıldı.

```bash
python -m scripts.eval.eval_gaps_tab --ckpt tabcrnn_deep2.pt --load-cal checkpoints/tabcrnn_deep2.cal.json --splits gaps_val
```
GAPS'te tel doğruluğunu ölçer. **`--splits gaps_val` mutlaka verilmeli** — varsayılanı `gaps_val gaps_test`.

```bash
python -m scripts.eval.eval_playability --split gaps_val
```
Parmaklandırmanın fiziksel çakışma, el açıklığı ve pozisyon ölçümlerini üç yöntem için karşılaştırır.

### Adım 10-11 — Eğitim (onay verdikten sonra)

```bash
python -m scripts.train.train_onset --init tabcrnn_deep2.pt --ckpt <YENI_AD>.pt \
    --tab-splits train,train_comp --pitch-splits gaps_train --deep 2 \
    --pitch-onset-weight 1 --offset-weight 1 --epochs <N> --lr 1e-4
```
Mevcut modelden başlayıp yeni bir checkpoint üretir. **`--init` ve `--ckpt` mutlaka verilmeli:** varsayılanları `tabcrnn_gaps.pt` ve `tabcrnn_onset.pt` ve ikinci dosya **diskte var** — yazılmazsa eski bir checkpoint'in üzerine yazar.

```bash
python -m scripts.train.train_edition_seq --out <YENI_AD>.pt --epochs 30 --seed 1
```
Parmaklandırma modelini yalnız `gaps_fit` üzerinde eğitir, en iyi epoch'u `gaps_val`'de seçer (~2 dk).

### Yeni checkpoint sonrası: kalibrasyon

```bash
python -m scripts.eval.eval_pitch --ckpt <YENI_AD>.pt --splits gaps_val val val_comp --save-cal checkpoints/<YENI_AD>.cal.json
```
Yeni model için eşik ve çözümleme kurallarını doğrulama setinde arar ve kaydeder (~1 saat; `deep2`'nin ayar dosyası kullanılmaz, yoksa iki modelin ayarı karışır).

## 1.4 Eğitim / validation / test ayrımı

| | Hangi split | Ne için |
|---|---|---|
| **Eğitim** | `train`, `train_comp` (GuitarSet 00-04) · `gaps_fit` (GAPS 202 kayıt) | Model ağırlıkları güncellenir |
| **Validation** | `val`, `val_comp` (GuitarSet oyuncu 05) · `gaps_val` (28 kayıt) | Epoch seçimi, eşik seçimi, hata analizi, geliştirme sırasında raporlama |
| **Test** | `gaps_test` (30 kayıt, 27 icracı) | **KAPALI.** Yalnız bir faz tamamen kapandığında, bir kez, rapor için |

**`gaps_test`'i yanlışlıkla açabilecek üç yer (dikkat):**

| Script | Tehlike | Korunma |
|---|---|---|
| `eval_pitch.py` | varsayılan `--splits` = `gaps_test val val_comp` | `--splits gaps_val val val_comp` yaz |
| `eval_gaps_tab.py` | varsayılan `--splits` = `gaps_val gaps_test` | `--splits gaps_val` yaz |
| `eval_edition.py` | **split seçeneği yok; tasarımı gereği `gaps_test` okuyor** | Geliştirme sırasında **çalıştırma**. `gaps_val` karşılığı için `tests/test_edition.py` içindeki yeniden üretim testini kullan |

`eval_playability.py` kendini koruyor: `gaps_test` verilirse reddediyor. `diagnose_strings.py` ve
`diagnose_pitch.py` yalnız verdiğin split'i okuyor.

## 1.5 Karşılaştırma için referans değerler

`docs/baselines/d1_baseline.json`'dan — hangisinin validation hangisinin test olduğu ayrı yazılı:

| Ölçüm | Değer | Split | Cinsi |
|---|---|---|---|
| GuitarSet solo tab F1 | 0,777 | `val` | validation |
| GuitarSet solo nota F1 | 0,924 | `val` | validation |
| GuitarSet akorlu nota F1 | 0,807 | `val_comp` | validation |
| GAPS nota F1 | 0,779 | `gaps_test` | **bağımsız test** |
| GuitarSet oracle tel doğruluğu | 0,878 | `val` | validation |
| GAPS ses tabanlı oracle tel | 0,610 | `gaps_val` | validation |
| GAPS naif tel kuralı | 0,745 | `gaps_test` | bağımsız test |
| Parmak doğruluğu (konum verilerek) | 0,636 | `gaps_test` | bağımsız test |

---

# 2. 25 SORUNUN AÇIKLAMASI

**S-01 — A telinde düşük doğruluk**

**Sorun ne?** Model bazı A teli notalarını başka telde (çoğunlukla D telinde) çalınmış gibi gösteriyor. Elimizdeki %60 sayısı iki model öncesine ait; `CLAUDE.md`'de yazan %66 ise hiçbir kayda dayanmıyor. Güncel modelde tel başına doğruluk **hiç ölçülmedi**.
**Neden önemli?** Doğru notayı bulsak bile TAB'da yanlış tel ve perde gösteririz; gitarist yanlış yerden çalar.
**Nasıl düzeltebiliriz?** Önce güncel modelle ölçeriz (§1.3 Adım 4). Sorun duruyorsa en ucuzdan başlarız: eğitim verisinde tellerin dengeli temsil edilip edilmediğini kontrol etmek, sonra modelin tel tınısını daha iyi duyması için frekans çözünürlüğünü artırmak.
**Kazanç?** TAB'daki yanlış tel seçimleri azalır, tab F1 yükselir.
**Durum:** Önce ölçüm gerekiyor.

**S-02 — GAPS'te tel tahmini basit kuraldan kötü**

**Sorun ne?** Klasik gitar kayıtlarında model, teli sesten tahmin ederken "en kalın uygun teli seç" gibi basit bir kuraldan daha kötü sonuç veriyor (0,61'e karşı 0,70). Aynı model çelik telli gitarda çok daha iyi (0,88).
**Neden önemli?** Projenin ana hedefi klasik gitar; orada "duyulan tel" bilgisi güvenilmez.
**Nasıl düzeltebiliriz?** Şimdilik ürün bunu zaten aşıyor: kullanıcıya iki ayrı mod sunuluyor ve klasik gitarda "edisyon önerisi" modu kullanılıyor (tel 0,815). Kalıcı çözüm S-01 ile aynı yoldan geçer.
**Kazanç?** Ses tabanlı modun klasik gitarda da kullanılabilir hâle gelmesi.
**Durum:** S-01'e bağlı. Kısmen hafifletildi.

**S-03 — Notayı bulmak ile teli bulmak arasındaki fark**

**Sorun ne?** Model notanın hangi perde olduğunu %92 doğru buluyor ama tel+perde birlikte doğru olma oranı %78'de kalıyor. Aradaki fark tamamen tel seçiminden geliyor.
**Neden önemli?** Bu, S-01 ve S-02'nin toplam etkisini gösteren bir gösterge; ayrı bir hata değil.
**Nasıl düzeltebiliriz?** Ayrı bir iş yok; S-01 ve S-02 çözülürse bu fark kapanır.
**Kazanç?** —
**Durum:** Başka işe bağlı (S-01, S-02).

**S-04 — "Model notayı hiç duymuyor"**

**Sorun ne?** Kaçırılan notaların yaklaşık üçte birinde model o notayı hiçbir anda "burada bu ses var" diye işaretlemiyor — yani sinyal modelin çıkışında hiç yok. Önemli: bu, "o nota eğitimde hiç görülmedi" demek **değil**; modelin duyarlılığıyla ilgili.
**Neden önemli?** Bu notalar sonradan hiçbir kuralla kurtarılamaz; çözümleme ayarlarıyla düzeltilemez.
**Nasıl düzeltebiliriz?** İlk iş ölçüm: bu oran son kez mevcut modelden önce ölçüldü, güncel değeri bilmiyoruz (§1.3 Adım 5). Ayrıca eğitim verisinde hangi notaların az geçtiğini sayan küçük bir ölçüm yazmak faydalı olur.
**Kazanç?** Kaçan nota sayısı azalır; TAB daha eksiksiz olur.
**Durum:** Önce ölçüm gerekiyor.

**S-05 — Çok hızlı tekrarlanan notalar (100 ms'den kısa)**

**Sorun ne?** Aynı notayı çok hızlı üst üste çalınca (klasik gitarda tremolo) model bunları tek uzun nota olarak görüyor; klasik gitar kayıtlarında bu tekrarların yarısından fazlası kaçıyor.
**Neden önemli?** Tremolo klasik gitar repertuvarında yaygın; TAB'da nota sayısı ve ritim yanlış çıkar.
**Nasıl düzeltebiliriz?** Ayar değiştirerek denenebilecek her yol tükendi (ölçüldü). Denenmemiş bir fikir var: modele "nerede vuruş var" diye sormak yerine "bu aralıkta kaç vuruş var" diye sormak — bu daha kaba bir soru olduğu için 23 ms'lik zaman çözünürlüğüne daha uygun.
**Kazanç?** Hızlı pasajlar doğru nota sayısıyla çıkar.
**Durum:** Eğitim gerektirir; S-06 ile birlikte ele alınmalı.

**S-06 — Akorların eksik veya bölünmüş çıkması**

**Sorun ne?** Model bazen tek bir notayı ikiye bölüyor, bazen akorun yalnız bir kısmını buluyor. Bunun nedeni S-05 ile aynı: ses eğrisindeki küçük bir çukur hem "yeni vuruş" hem "aynı notanın içindeki gürültü" olabiliyor ve ikisi ayırt edilemiyor.
**Neden önemli?** TAB'da olmayan notalar görünüyor veya akorlar eksik yazılıyor.
**Nasıl düzeltebiliriz?** Ucuz bir ölçümle başlanabilir: bu bölünmeler tek sesli yerlerde mi, akorlarda mı daha sık? Fark varsa kuralı akor durumuna göre değiştirmek eğitim gerektirmez.
**Kazanç?** Hem yanlış notalar hem eksik akorlar azalır.
**Durum:** Önce ölçüm gerekiyor (S-05 ile birlikte).

**S-07 — Akorlu kayıtlarda nota bulma zorluğu**

**Sorun ne?** Aynı anda birden fazla nota çalındığında modelin kaçırma oranı tek sesli pasajlara göre iki kattan fazla artıyor. Darboğaz tel seçimi değil, notanın kendisini bulmak.
**Neden önemli?** Klasik gitar polifonik; akorlar ve iç sesler projenin ana hedefinin parçası.
**Nasıl düzeltebiliriz?** Ayrı bir iş değil; S-04 ve S-06 çözülürse bu da düzelir. Takip etmek için akorlu kayıtlardaki ölçüm (`val_comp`) her denemede raporlanmalı.
**Kazanç?** Akor pasajlarında daha eksiksiz TAB.
**Durum:** Başka işe bağlı (S-04, S-06).

**S-08 — Akor gruplamasında zaman örtüşmesi gözetilmiyor**

**Sorun ne?** Kod, başlangıçları 30 ms içinde olan notaları aynı akor sayıyor. Ama arpejde notalar sırayla çalınır; başlangıçları yakın olsa da aynı anda basılı olmayabilirler.
**Neden önemli?** Parmaklandırma tarafında olmayan bir "imkânsızlık" üretiyor ve gereksiz kısıtlar devreye giriyor.
**Nasıl düzeltebiliriz?** Ölçüm tarafı zaten düzeltildi (artık yalnız gerçekten üst üste basılı notalar sayılıyor). Kalan iş, aynı düzeltmeyi kısıt tarafına da taşımak — S-12/S-13 ile aynı çalışmada yapılır.
**Kazanç?** Parmak kısıtları daha doğru yerlerde devreye girer.
**Durum:** Başka işe bağlı (S-12).

**S-09 — Parmak ve ritim bilgisi ortak veri yapısında yok**

**Sorun ne?** Projenin ortak nota kaydında (`NoteEvent`) başlangıç, bitiş, perde, tel ve fret alanları var; **parmak numarası ve ritim bilgisi yok**. Bu yüzden iki farklı yerde ayrı ayrı yan kanallar kuruldu ve iki ayrı "ses → TAB" komut dosyası ortaya çıktı.
**Neden önemli?** Hangisinin ürün olduğu belli değil; iki dosya ayrı ayrı bakım istiyor; parmak bilgisi TAB çıktısına taşınamıyor.
**Nasıl düzeltebiliriz?** Ortak kayda üç alan eklenir (parmak, el pozisyonu, vuruş konumu) ve boş değerle başlatılır — böylece mevcut davranış hiç değişmez. Sonra tek komut dosyası ana yol seçilir.
**Kazanç?** Tek ürün çıktısı; parmak bilgisi hem metin TAB'a hem nota dosyasına girebilir.
**Durum:** Hemen düzeltilebilir (düşük risk), ama S-22 birleştirmesiyle birlikte yapılması daha mantıklı.

**S-10 — Metin TAB'da parmak gösterilmiyor**

**Sorun ne?** Terminalde çıkan 6 telli TAB yalnız tel ve perde gösteriyor, parmak numarasını göstermiyor.
**Neden önemli?** Küçük bir eksik; nota dosyası (MusicXML) tarafında parmak zaten yazılıyor.
**Nasıl düzeltebiliriz?** S-09'daki alan eklendikten sonra tek bir çizim fonksiyonuna parmak satırı eklenir.
**Kazanç?** Terminal çıktısı da kullanışlı hâle gelir.
**Durum:** Başka işe bağlı (S-09). Sonraya bırakılabilir.

**S-11 — GuitarSet sonuçları test değil, doğrulama sonucu**

**Sorun ne?** GuitarSet'te 6 gitaristten 5'i eğitimde, 1'i (oyuncu 05) hem ayar seçiminde hem sonuç raporlamada kullanılıyor. Yani 0,777 ve 0,924 gibi manşet sayılar **doğrulama** sonucudur, bağımsız test değil. Kök README bunu söylemiyor.
**Neden önemli?** Tez savunmasında "bu test sonucu mu?" sorusu gelir. Cevabımız dürüst ve hazır, ama belgede yazılı değil.
**Nasıl düzeltebiliriz?** En kolay yol: kök README'deki sonuç tablosuna "bu sayı doğrulama mı test mi" diye bir kolon eklemek. Gerçek bağımsız test üretmek ancak bir gitaristi her şeyden çıkarıp sıfırdan eğitmekle olur — pahalı ve bu aşamada gereksiz.
**Kazanç?** Sayılar yanlış okunmaz; tez savunulabilir olur.
**Durum:** Hemen düzeltilebilir (yalnız belge).

**S-12 — Model çalınamayan parmaklandırma öneriyor**

**Sorun ne?** Model aynı parmağı, aynı anda basılı olan iki ayrı perdeye atayabiliyor — bu fiziksel olarak imkânsız. Üst üste basılı not içeren akorların %8,5'inde oluyor. Karşılaştırma: aynı kuralları gerçek nota editörlerinin parmaklandırmasına uyguladığımızda ihlal oranı **%0**.
**Neden önemli?** Sunumda "fiziksel kuralları ihlal etmeyen öneri" sözü verildi; şu an bu söz karşılıksız.
**Nasıl düzeltebiliriz?** Parmak seçimi şu an her nota için tek tek yapılıyor; akorun bütününe bakacak şekilde değiştirilmeli. Tellerde aynı sorun için zaten çalışan bir çözüm var, onun mantığı parmağa taşınabilir.
**Kazanç?** Önerilen parmaklandırma gerçekten çalınabilir olur.
**Durum:** Hemen çalışılabilir, **ama S-13 ile birlikte** (ayrı yapılırsa biri diğerini bozuyor). S-09'un ardından.

**S-13 — El açıklığı sınırı yok**

**Sorun ne?** Modelin ürettiği bazı akorlarda parmakların kapsadığı mesafe 22 perdeye kadar çıkıyor; insan eli için imkânsız. Editörlerin parmaklandırmasında bu mesafe en fazla 3 perde.
**Neden önemli?** Tek saçma konum bütün akoru çalınamaz yapıyor.
**Nasıl düzeltebiliriz?** Açıklık için yumuşak bir ceza eklenmeli; üst sınır için editörlerin dağılımı referans alınabilir. Ölçüm, bu sorunun "bir tel aynı anda iki nota çalamaz" kuralının tek başına uygulanmasından doğduğunu gösteriyor — bu yüzden iki kısıt birlikte eklenmeli.
**Kazanç?** Akorlar insan eliyle çalınabilir hâle gelir.
**Durum:** S-12 ile aynı çalışmada yapılmalı.

**S-14 — Barre kuralı fazla temkinli; el pozisyonu çıktıya girmiyor**

**Sorun ne?** Barre (bir parmakla aynı perdede birden çok tele basmak) ölçümünde kodumuz, teller yan yana değilse "emin değilim" diyor; oysa kısmi barre klasik gitarda normal. Ayrıca el pozisyonu hesaplanıyor ama kullanıcıya gösterilmiyor.
**Neden önemli?** Ölçüm barre'yi olduğundan sorunlu gösteriyor; el pozisyonu bilgisi kullanıcıya ulaşmıyor.
**Nasıl düzeltebiliriz?** Barre kuralını gevşetmek küçük bir değişiklik. El pozisyonu, S-09'daki alan eklemesiyle çıktıya taşınabilir.
**Kazanç?** Daha gerçekçi ölçüm; kullanıcıya el pozisyonu önerisi.
**Durum:** S-12/S-09 ile birlikte yapılabilir. Sonraya bırakılabilir.

**S-15 — Sentetik veriyle ön eğitim hiç denenmedi**

**Sorun ne?** Sunumda tezin ana araştırma sorusu şu: "büyük ölçekli sentetik naylon gitar verisiyle ön eğitim, gerçek klasik gitarda başarıyı ne kadar artırır?" Bu deney hiç yapılmadı. Bir ara "modelin kapasitesi yeterli, gerek kalmadı" denildi — ama kapasite sorusu ile araştırma sorusu aynı soru değil. Ayrıca sentetik verinin kendisi artık diskte yok.
**Neden önemli?** Tezin özgünlük iddiası bu deneye dayanıyor; şu an deneysel desteği yok.
**Nasıl düzeltebiliriz?** Önce ucuz bir kapı: S-04'teki kapsam ölçümü. Kaçan notalar gerçekten eğitimde az görülmüş notalardan geliyorsa deney mantıklı; gelmiyorsa hipotez pahalı deney yapılmadan zayıflar. Desteklenirse veri yeniden indirilir ve dört satırlık bir karşılaştırma tablosu doldurulur (kabul kuralı eski bir devlog'da zaten yazılı).
**Kazanç?** Tezin ana sorusu cevaplanır — olumsuz sonuç bile geçerli bir bulgudur.
**Durum:** S-04'ün ölçümüne bağlı. Veri indirme gerektirir.

**S-16 — Test seti bir kez karar için kullanıldı**

**Sorun ne?** Geçmişte bir karar (parmaklandırma modelini kabul etmek) test setine bakılarak verildi. Model ayarları test setine uydurulmadı, ama "kabul/ret" kararı oradan geldi.
**Neden önemli?** O modelin raporlanan test sayıları bir miktar iyimser okunmalı.
**Nasıl düzeltebiliriz?** **Bu geri alınamaz — düzeltilecek bir kod hatası değil, geçmişte kalmış bir süreç kaydı.** Yapılacak tek şey tezde bunu dipnot olarak belirtmek. Tekrarı zaten koda bağlanan kurallarla engellendi.
**Kazanç?** Dürüst raporlama.
**Durum:** Geçmişte kalmış; yalnız belgelenecek.

**S-17 — Koruma kuralları yalnız yeni kodda zorunlu**

**Sorun ne?** "Test setini ayar seçiminde kullanma" gibi kurallar koda bağlandı, ama yalnız sonradan yazılan dosyalarda. Eski eğitim ve ölçüm dosyaları bu kontrollerden geçmiyor.
**Neden önemli?** İlke olarak yanlış bir komut hâlâ test setini açabilir (§1.4'teki üç tehlike).
**Nasıl düzeltebiliriz?** Eski dosyalara dokunmak riskli — davranışları değişirse mevcut sonuçlar tekrarlanamaz hâle gelir. Bu yüzden önerilen: eski dosyalara dokunmamak, komutları verirken `--splits` parametresini her zaman açıkça yazmak, **yeni** yazılan her dosyada kontrolleri kullanmak.
**Kazanç?** Kazara sızıntı riski azalır.
**Durum:** Bilinçli olarak kabul edilmiş risk; düzeltme önerilmiyor.

**S-18 — Karantinaya alınan kayıt eski eğitim dosyalarında etkisiz**

**Sorun ne?** Kimliği doğrulanamayan bir GAPS kaydı (`395_w41wc`) karantinaya alındı, ama bu yalnız yeni kodda geçerli. Eski eğitim dosyası kayıt listesini doğrudan okuduğu için mevcut model bu kaydı görmüş olabilir.
**Neden önemli?** Etkisi küçük: o kaydın parmak/tel etiketi zaten üretilmedi, yalnız perde bilgisi kullanılmış olabilir.
**Nasıl düzeltebiliriz?** Geçmiş model yeniden eğitilmeyecek. Bundan sonraki her eğitimde karantinayı düşüren fonksiyon kullanılır.
**Kazanç?** Yeni modeller temiz veriyle eğitilir.
**Durum:** Geçmişte kalmış; ileriye dönük kural yeterli.

**S-19 — Etiketleri üreten kodun testi yok**

**Sorun ne?** Nota kâğıdındaki parmak numaralarını ve ritmi sese bağlayan 255 satırlık kodun (`gaps_score.py`) hiç otomatik testi yok. Doğruluğu yalnız toplu istatistiklerle kontrol edildi (%98,9 uyum gibi).
**Neden önemli?** Bütün parmaklandırma ve ritim etiketleri bu koddan geliyor. Buradaki sessiz bir hata tüm zinciri bozar ve istatistikler bunu yakalamayabilir.
**Nasıl düzeltebiliriz?** Elle hazırlanmış küçük bir nota kâğıdı örneği (içinde tekrar işareti, kapo ve parmak numarası olan) ile beklenen çıktıyı kontrol eden bir test yazmak.
**Kazanç?** Etiketlere güven; ileride bu koda dokunurken emniyet ağı.
**Durum:** Hemen düzeltilebilir (yalnız test ekler, kodu değiştirmez).

**S-20 — Vuruş takibi hedefin altında**

**Sorun ne?** Ses kaydından tempo ve vuruşları otomatik bulma hedefi tutmadı (0,63; hedef 0,75). Bu yüzden ürün şu an yarı otomatik: kullanıcı yaklaşık tempoyu ve ölçü sayısını (ör. 3/4) kendisi giriyor. Buna rağmen notaların ölçü içindeki yeri %61 doğru.
**Neden önemli?** Nota kâğıdında ölçü çizgileri kayabiliyor; nota değerleri doğru olsa bile yanlış yerde görünüyor.
**Nasıl düzeltebiliriz?** Hazırda bir yedek fikir var: kullanıcının müzik çalarken vuruşa dokunması ("tap" modu). Bu, otomatik takibin en zayıf halkasını atlar.
**Kazanç?** Ölçü çizgileri doğru yere oturur; nota kâğıdı okunabilir olur.
**Durum:** Sonraya bırakılabilir; yarı otomatik mod şimdilik çalışıyor. S-22'ye bağlı (kod diğer dalda).

**S-21 — Nota dosyası MuseScore'da hiç açılıp kontrol edilmedi**

**Sorun ne?** MusicXML yazıcısı teknik olarak doğrulandı (nota kâğıdını yazıp geri okuduğumuzda **birebir** aynı sonucu veriyor), ama dosya MuseScore 4'te hiç açılıp göz ile kontrol edilmedi. Teknik işaretler (bend, kaydırma, legato) ve web görüntüleyici de yok.
**Neden önemli?** Biçimin geçerli olması notaların doğru tahmin edildiğini kanıtlamaz — bunlar iki ayrı şey. Görsel kontrol yalnız birincisini doğrular.
**Nasıl düzeltebiliriz?** Dal birleştirildikten ve iki önbellek kurulduktan sonra bir kayıt için dosya üretilip MuseScore'da açılır; 9 maddelik kısa bir kontrol listesi uygulanır (nota portresi, TAB portresi, ölçü çizgileri, bağlar, parmak numaraları, tempo).
**Kazanç?** Ürünün gerçekten kullanılabilir olduğunu görmek.
**Durum:** S-22'ye bağlı.

**S-22 — İki dal birbirinden ayrı ilerledi**

**Sorun ne?** Ritim ve nota kâğıdı (MusicXML) kodu başka bir dalda yazıldı. İki dal aynı gün paralel ilerlediği için aynı dosya adlarını (`README15`, `README16`) farklı içerikle kullanıyorlar. Ayrıca diğer dalda hiç test yok ve ritmin ihtiyaç duyduğu iki ara dosya (808 MB) bu bilgisayarda yok.
**Neden önemli?** Dikkatsiz bir birleştirme **devlog kaybına** yol açabilir — devlog projenin tek tarihsel kaydı. Ayrıca birleştirmeden önce bir ölçüm aracı (`diagnose_strings.py`) diğer dalda arşive taşınmış durumda.
**Nasıl düzeltebiliriz?** Sırayla: (1) diğer dalın iki devlog dosyası farklı numaraya alınır, (2) S-01 ölçümü arşivlenme öncesi yapılır, (3) birleştirme yapılır ve iki test dosyasının kırılacağı yer elle düzeltilir, (4) eksik ara dosyalar üretilir.
**Kazanç?** Tek proje, tek ürün çıktısı; nota kâğıdı ve ritim ana dala gelir.
**Durum:** Hemen başlanabilir; ama **sıra önemli** — S-01 ölçümü birleştirmeden önce.

**S-23 — Bir test yanlış ortamda hata veriyor**

**Sorun ne?** Testlerden biri, bir yardımcı paket (`mir_eval`) kurulu değilse "atlandı" demek yerine hata veriyor. Doğru ortamda (`.venv`) 170 testin hepsi geçiyor.
**Neden önemli?** Yanlış yorumlayıcıyla test koşulduğunda gerçek bir hata sanılıyor; denetimde bu yaşandı.
**Nasıl düzeltebiliriz?** Testin başındaki kontrol listesine bu paketin adı da eklenir — tek satırlık değişiklik.
**Kazanç?** Testler her ortamda doğru davranır.
**Durum:** Hemen düzeltilebilir (en kolay iş).

**S-24 — Ana README iki model geride**

**Sorun ne?** Projenin ilk okunan dosyası (kök `README.md`) ve katkı rehberi, güncel modeli değil iki model öncesini gösteriyor; yol haritasındaki kutular da güncel değil. Kurulum satırının biri bozuk, `requirements.txt` yanlış Python sürümü yazıyor.
**Neden önemli?** Danışman veya jüri ilk bu dosyayı okuyor; tezdeki sayılarla repo sayıları uyuşmuyor görünüyor.
**Nasıl düzeltebiliriz?** Tek seferde güncelleme; sayıların kaynağı olarak `docs/baselines/d1_baseline.json` kullanılır. S-11'in kolonu da aynı anda eklenir.
**Kazanç?** Proje dışarıdan doğru görünür; en düşük maliyetli kazanç.
**Durum:** Hemen düzeltilebilir (yalnız belge).

**S-25 — Rastgelelik payı hiç ölçülmedi**

**Sorun ne?** Projede "aynı eğitimi tekrar çalıştırsak sonuç ±0,01 oynar" varsayımı var ve kabul kararları buna dayanıyor. Ama eğitim hiç ikinci kez (farklı rastgelelik değeriyle) çalıştırılmadı, yani bu varsayım ölçülmedi.
**Neden önemli?** Sınırda kalan kararların (ör. 0,013 fark) istatistiksel dayanağı zayıf.
**Nasıl düzeltebiliriz?** Bir modeli farklı rastgelelik değeriyle bir kez daha eğitip farkı ölçmek. Not: ses modeli eğitim dosyasında bunun için hazır bir seçenek **yok** (parmaklandırma dosyasında `--seed` var) — ses modelinde küçük bir ekleme gerekir.
**Kazanç?** Kabul eşiklerinin güvenilirliği; tezde savunulabilir bir sayı.
**Durum:** Bir eğitim koşusu gerektirir; sonraya bırakılabilir.

---

# 3. ÖNCELİK TABLOSU

| Öncelik | Sorun | Neden önce? |
|---|---|---|
| **1 — Kolay ve risksiz** | **S-23** | Tek satır; hiçbir şeyi bozmaz, test altyapısını düzeltir |
| **1** | **S-24, S-11** | Yalnız belge; kodla ilgisi yok, en yüksek görünür kazanç |
| **2 — Ölçümden önce yapılmalı** | **S-22 (1. adım)** | Diğer dalın devlog adlarını değiştirmek; **bilgi kaybı riskini kapatır** |
| **3 — Önce ölçmemiz gerekenler** | **S-01, S-04** | Bu iki ölçüm yapılmadan hangi eğitim denemesinin anlamlı olduğunu bilmiyoruz. **S-01, dal birleştirmesinden önce yapılmalı** (ölçüm aracı diğer dalda arşive taşınmış) |
| **3** | **S-06** (ölçüm kısmı) | Ucuz; S-05'in yönünü belirler |
| **4 — Eğitim öncesi altyapı** | **S-19** | Bütün parmak/ritim etiketlerinin kaynağı; test eklemek kodu değiştirmez |
| **4** | **S-09** | Ortak veri yapısına alan eklemek; S-10, S-12, S-14 ve tek ürün çıktısının ön koşulu |
| **5 — Birleştirme** | **S-22 (2-4. adım)** | Ritim ve nota kâğıdı ana dala gelir; S-20, S-21'in ön koşulu |
| **6 — Geliştirme** | **S-12 + S-13 + S-08 + S-14** | Parmaklandırma kısıtları; **dördü tek çalışmada** yapılmalı |
| **6** | **S-05** (eğitim kısmı) | S-06'nın ölçümünden sonra |
| **7 — Doğrulama** | **S-21** | Birleştirme + önbellek sonrası |
| **8 — Araştırma** | **S-15** | S-04'ün kapsam ölçümüne bağlı; veri indirme gerektirir |
| **8** | **S-20, S-25** | Yedek fikirler hazır; acil değil |
| **Düzeltilmeyecek** | **S-16, S-17, S-18** | Geçmişte kalmış süreç kayıtları; kod hatası değil. Yalnız tezde belirtilir |
| **Başka işe bağlı** | **S-02, S-03, S-07, S-10** | Kendi başına iş değil; S-01/S-04/S-09 çözülünce kapanırlar |

## Tarihsel sırayla ilerleme uyarıları

README 1'den başlayarak ilerlemek istediğini belirttin. Çoğu erken README'nin sorunu zaten kapanmış
durumda, bu yüzden tarihsel sıra pratikte **S-01 (README10) → S-04/S-05/S-06 (README11-12) →
S-12/S-13/S-14 (README18) → S-20/S-21 (Katman 5)** şeklinde ilerler. Bu sırada iki yerde dikkat
gerekiyor:

| Uyarı | Ayrıntı |
|---|---|
| **S-01 ölçümü birleştirmeden önce** | Ölçüm aracı `diagnose_strings.py`, diğer dalda `archive/` altına taşınmış. Birleştirmeyi önce yaparsak araç "arşiv kodu" sayılır |
| **S-12 ve S-13 birlikte** | Ölçüm gösteriyor ki "bir tel bir nota" kuralını tek başına uygulamak saçma konumlar üretiyor (24. perde gibi). Parmak kısıtını açıklık cezası olmadan eklemek sonucu kötüleştirebilir |
| **S-09 diğer işlerden önce** | Ortak veri yapısına alan eklemek; S-10, S-12 ve S-14 bu olmadan yarım kalır. Risk düşük: alanlar boş değerle başlarsa mevcut davranış hiç değişmez (çıkarım parmak izi testi bunu doğrular) |
| **S-05/S-06 eğitim gerektirir** | Ayar değiştirerek denenebilecek her yol tükendi (ölçüldü). Buradan sonrası yeni bir eğitim koşusu demek |
| **Her eğitim yeni dosya adı ister** | `train_onset.py`'ın varsayılan çıktı adı diskte var olan bir dosya; `--ckpt` yazılmazsa eski checkpoint'in üzerine yazar |
