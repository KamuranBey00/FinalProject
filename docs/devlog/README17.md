# P2 — Parmaklandırma Test Altyapısı ve Başlangıç Doğrulaması

> **Durum (11 Ekim 2026):** P2 tamamlandı. **Mevcut davranış değiştirilmedi**: model, checkpoint,
> önbellek, eğitim ve değerlendirme kodu aynı; yalnız test dosyası eklendi. Dal
> `Armagan-katman-4`, HEAD `b988d82`; commit/push yapılmadı. Sıradaki (onay bekliyor): **P3** —
> çalınabilirlik denetimi (yalnız ölçüm). P1 incelemesi ve yol haritası: `README16.md`.

## 1. Başlangıç durumu

| | Değer |
|---|---|
| Parmaklandırma modeli | `checkpoints/edition_seq.pt` (Katman 3.13 Adım 2b, kabul edilmiş) |
| Parmaklandırma zincirinin test kapsamı | **0 test** (P1 bulgusu: `edition.py`, `edition_net.py`, `gaps_score.py`, `build_gaps_tab.py`, `eval_edition.py`, `train_edition*.py` hiç test edilmemiş) |
| Mevcut test sayısı | 109 (Faz 0 + Faz 2A; 6 dosya) |
| Yayımlanmış metriklerin yeniden üretilebilirliği | **bilinmiyordu** — kabul kararı tek koşuya dayanıyordu |

## 2. Eklenen testler — `tests/test_edition.py` (20 test, 269 satır)

Testler **mevcut davranışı doğrular**; hiçbir fiziksel kısıt eklenmedi, model dokunulmadı.
İki grup: sentetik (veri/checkpoint/GPU gerektirmez) ve kapılı (`skipUnless`).

| Sınıf | Test | Ne doğruluyor |
|---|---|---|
| `TestOrderNotes` | 3 | Akor gruplaması: 30 ms toleransı içinde/dışında; akor içinde perde artan sıra (`edition.order_notes`) |
| `TestFingerCandidates` | 4 | `FingerModel.cands`: boş tel → yalnız 0, basılı → yalnız 1–4; `finger_baseline` boş telde 0, "perde − pozisyon + 1" ve 1–4 kırpması |
| `TestLabelSequence` | 1 | `label_sequence()` sentetik npz'den sıra + `string`/`fret`/`finger_note` okuma, akor bayrağı |
| `TestLogLinearDecode` | 2 | 2a log-lineer `decode()`: çıktı aralıkları (tel 0–5, fret = perde − açık tel, 0 ≤ fret ≤ 24, boş tel ⇔ parmak 0) ve `fixed` konumun korunması |
| `TestEditionNetCheckpoint` (kapılı) | 7 | Checkpoint yüklenip tahmin üretiyor; **boş tel → parmak 0**; basılı notada parmak 1–4; **akor içinde tel tekliği**; konum adaylarının geçerliliği (24 rastgele perde); **determinizm** (aynı girdi → birebir aynı çıktı); `fixed` konum |
| `TestGapsValReproduction` (kapılı) | 3 | Kayıt/nota sayısı + **gaps_val metriklerinin yeniden üretimi** (aşağıda) |

Kapı koşulu: `importlib.util.find_spec("torch")` + `checkpoints/edition_seq.pt` (+ reprodüksiyon
için `data/cache/gaps_tab/`). Koşul sağlanmazsa testler **atlanır**, hata vermez.

**Protokol uyumu:** reprodüksiyon yalnız **`gaps_val`** üzerinde; dosya listesi
`protocol.gaps_training_files()` ile alınıyor (karantina `395_w41wc` düşülür).
**`gaps_test` hiç okunmadı** — yeni deney veya model seçimi yapılmadı.

## 3. Çalıştırılan komutlar

```bash
# yalnız yeni testler
python -m unittest discover -s tests -t . -p "test_edition.py"
# tüm suite (regresyon kontrolü)
python -m unittest discover -s tests -t .
# protokol ve çıkarım regresyonu
python -m scripts.eval.check_protocol
python -m scripts.eval.check_inference_regression
```

## 4. Test sonuçları

| Koşu | Sonuç |
|---|---|
| `test_edition.py` | **20/20 geçti** |
| Tüm birim testler (Faz 0 + Faz 2A + P2) | **129/129 geçti** (5–9 sn; torch yüklemesine göre değişir) |
| Veri protokolü kontrolü | **17/17 geçti** |
| Çıkarım regresyonu (`tabcrnn_deep2`) | **REGRESYON YOK** — parmak izi referansla birebir |

Dosya başına: `test_protocol` 25, `test_contour` 29, `test_technique_eval` 21, **`test_edition` 20**,
`test_baselines` 12, `test_fingerprint` 11, `test_selection` 11.

## 5. Model doğrulama sonuçları

- `checkpoints/edition_seq.pt` CPU'da sorunsuz yükleniyor (`EditionNet` state_dict uyumlu) ve
  tahmin üretiyor.
- **Sert kısıtlar tutuyor:** boş tel (E2 → tek geçerli konum) parmak **0**; basılı notalarda
  parmak **1–4**; akor içinde üç notaya **üç farklı tel**; her notada
  `fret = perde − açık tel` ve `0 ≤ fret ≤ 24`.
- **Determinizm:** aynı girdiyle iki çağrı bit düzeyinde aynı sonucu veriyor (dropout `eval`
  modunda kapalı).
- `fixed` parametresi verilen konumu koruyor (parmak ölçümünün dayandığı davranış).

## 6. Önceki metriklerin yeniden üretilebilirliği

**`gaps_val` (geliştirme split'i) — 15 etiketli kayıt, 15.982 nota, 3.851 parmaklı basılı nota:**

| Metrik | README14 (gaps_val) | P2 ölçümü | Durum |
|---|---|---|---|
| Tel uyumu (dizi modeli) | 0,766 | **0,766** | birebir |
| Parmak (partisyon konumu verilerek) | 0,675 | **0,675** | birebir |
| Ürün (tel + perde + parmak) | 0,382 | **0,382** | birebir |

Nota sayısı da kayıtla uyuşuyor (README14: "parmaklı nota: gaps_val 3851"). Metrik tanımları
`scripts/eval/eval_edition.py`'den alındı (`pooled`, `string_acc` doğrudan import edilerek), parmak
sayımı aynı dosyanın 2. bölümündeki döngüyle birebir çoğaltıldı.

**Yeniden üretilmeyen:** `gaps_test` sayıları (tel 0,815 · parmak 0,636 · ürün 0,434 · uçtan uca
0,331). **Bilinçli olarak çalıştırılmadı** — protokol gereği test split'i yalnız kapanışta bir kez
okunur. Bu değerler hâlâ README14'ün tek koşuluk kaydıdır ve P2 bunları doğrulamamıştır.

## 7. Karşılaşılan sorunlar

1. **Testin kendi hatası (çözüldü).** `pooled` fonksiyonu sınıf niteliğine atandığı için bound
   method oldu ve `self` fazladan argüman olarak geçti (`TypeError: pooled() takes 2 positional
   arguments but 3 were given`). Metrik fonksiyonu artık test metodunun içinde import ediliyor.
   Model veya ölçüm kodu etkilenmedi.
2. **Kapı koşulu eksikti (çözüldü).** Kapı yalnız dosya varlığını kontrol ediyordu; torch'un
   kurulu olmadığı bir ortamda iki sınıf hata verirdi. Kapıya `find_spec("torch")` eklendi →
   torch yoksa 10 kapılı test atlanır. (Not: daha önce kullandığım "torch'u yapay olarak devre
   dışı bırakma" denemesi gerçek bir kurulum senaryosu değildir; o yapay koşuda bu testler
   atlanmaz, hata verir. Gerçek ortam kontrolü `find_spec` ile yapılıyor.)
3. Ölçüm tarafında sürpriz çıkmadı: üç `gaps_val` değeri ilk koşuda birebir geldi.

## 8. Çözülmeyen eksikler

1. **`gaps_test` doğrulanmadı** (yukarıda; protokol kararı).
2. **Uçtan uca zincir test dışı** — `deep2` notalarından parmaklandırmaya giden yol
   (`eval_edition.py` 3. bölüm) testlenmedi; ses modelini ve çözümlemeyi gerektiriyor.
3. **Etiket üretimi test dışı** — `gaps_score.py` (MusicXML hizalama) ve `build_gaps_tab.py` hâlâ
   testsiz; P2 yalnız `label_sequence()` okuma yolunu sentetik npz ile test ediyor.
4. **Log-lineer modelin eğitimi (`fit`) test dışı** — yalnız sıfır ağırlıklı `decode()` duman testi var.
5. **`eval_edition.py` bütün olarak test dışı** — `main()` gaps_test'i okuduğu için test içinde
   çağrılmıyor; yalnız metrik yardımcıları kullanıldı.
6. **Fiziksel çalınabilirlik hâlâ denetlenmiyor** — P2'nin kapsamı dışı (P3).
7. **Entegrasyon eksikleri duruyor** — `NoteEvent`'te parmak alanı yok, ses → TAB yolu
   parmaklandırmayı tanımıyor (README16 bölüm 6).

## 9. P3 öncesinde yapılması gerekenler

1. **Kural kümesi ve eşikler kararlaştırılmalı:** el açıklığı (kaç perde?), parmak sırası
   tutarlılığı, akor içi aynı parmak yasağı, pozisyon kayması tanımı. Eşikler **gaps_val**'de
   ölçülecek, `gaps_test` kapalı kalacak.
2. **Metrik adları sunumla eşlenmeli** (slayt 12): "rule violation", "hand movement cost",
   "stretch / position shifts".
3. **Referans olarak editörün kendi parmaklandırması ölçülmeli:** partisyonun kendi çözümü bizim
   kurallarımızı ne sıklıkta ihlal ediyor? Bu, kuralların kalibrasyonu için zorunlu — kural
   editörü de ihlal ediyorsa kural fazla katıdır.
4. **Karşılaştırma tabanları:** naif konum + taban parmak, dizi modeli, partisyon (editör).
5. P3 **yalnız ölçüm** fazıdır: model veya kısıt davranışı değişmeyecek; değişiklik P4'te.
