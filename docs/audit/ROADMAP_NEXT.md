# Önerilen Geliştirme Sırası ve Deney Planı

> Denetim: 14 Ekim 2026 · `Armagan-katman-4` @ `546c7bd`
> Bu belge **öneri**dir; hiçbir adım uygulanmadı. Her iş için gerekçe, başarı ölçütü ve
> doğrulama yöntemi verildi. Sorun ID'leri `ISSUE_REGISTER.md`'ye atıf yapar.

## Sıralama ilkesi

Üç ilkeye göre sıralandı:
1. **Bilgi kaybı riski olan iş en önce** (takipsiz dosya, çakışan devlog adları) — geri alınamaz.
2. **Ölçüm, çözümden önce** — projenin kendi kuralı (`CONTRIBUTING.md` kural 3). Ucuz ölçümler, pahalı eğitimlerden önce.
3. **Tezde görünür olan iş, teknik borçtan önce** — teslim tarihi varsa vitrin dokümanı ve araştırma sorusu kodun temizliğinden önemli.

| # | İş | Tür | Maliyet | Risk | Öncelik |
|---|---|---|---|---|---|
| 0 | `scripts/transcribe.py` yedeği | koruma | dakikalar | — | **İLK** |
| 1 | İki dalın birleştirilmesi | entegrasyon | 1-2 gün | yüksek | **KRİTİK** |
| 2 | Üç ucuz ölçüm (A-01, B-01, B-04) | ölçüm | saatler | yok | **KRİTİK** |
| 3 | Vitrin dokümanları + künye | dokümantasyon | yarım gün | yok | **KRİTİK** |
| 4 | P4 — akor içi parmak + açıklık kısıtı | geliştirme | 1-2 gün | orta | YÜKSEK |
| 5 | MuseScore 4 görsel doğrulama | doğrulama | saatler | yok | YÜKSEK |
| 6 | SynthTab araştırma sorusu kararı | araştırma | değişken | — | YÜKSEK |
| 7 | Akor fragmanı / tekrar deneyi | araştırma | 1-2 gün | orta | ORTA |
| 8 | Test boşlukları | kalite | 1 gün | yok | ORTA |
| 9 | Ek akustik datasetler | araştırma | — | — | EN SON |

---

## İş 0 — `scripts/transcribe.py` korunması (İLK, dakikalar)

**Neden.** 221 satırlık ürün giriş noktası **takipsiz** (`git status` → `?? scripts/transcribe.py`).
Yalnız bu makinede var, yedeği yok. Birleştirme sırasında bir hata (`git clean`, checkout) dosyayı
siler ve geri gelmez.

**Yapılacak (kullanıcı onayı ile).** Üç seçenek, önerilen sırayla:
1. Mevcut dala commit et (`git add scripts/transcribe.py && git commit`). Birleştirmede çakışır ama kaybolmaz.
2. Dal dışına kopyala (ör. `_backup/` dizini — proje kökünde zaten var).
3. İkisi birden.

**Başarı ölçütü.** Dosya en az bir git objesi veya dizin dışı kopya olarak mevcut.

> **Not:** Denetim bu dosyaya dokunmadı; yalnız `py_compile` ile sözdizimsel sağlamlığını
> doğruladı. Davranışı hiç çalıştırılmadı (bu makinede test ses dosyası yok; `inputs/` altında
> yalnız pptx/docx var, ses olarak yalnız `data/raw/gaps_hf/audio/*.wav` mevcut).

---

## İş 1 — İki dalın birleştirilmesi (KRİTİK)

**Neden.** Katman-5 dalı projenin en büyük eksik parçalarını zaten çözmüş durumda: **MusicXML
yazıcısı** (gidiş-dönüş birebir 1,0000), **ritim/kuantalama** (oracle IOI 0,933), **uçtan uca
pipeline** ve **383 satırlık sözlük**. Mevcut dal ise ölçüm altyapısını (170 test, protokol
kapıları, çalınabilirlik metrikleri) taşıyor. İkisi birleşmedikçe:
- MusicXML üretilemiyor (mevcut dalda `gtab/render/` yok),
- ritim kodu test ağına girmiyor,
- iki rakip giriş noktası ayrı bakım istiyor (`E-01`).

**Birleştirme yönü.** `katman-5-ritim` → mevcut dal (`Armagan-katman-4`) ya da **üçüncü bir
entegrasyon dalı**. Gerekçe: ölçüm altyapısı (`gtab/evaluation/protocol.py` vb. + `tests/`) katman-5'te
**hiç yok**; ters yönde birleştirme onu kaybetme riski taşır.

> **Not:** Bu denetim merge/rebase **yapmadı** (talimat gereği). Aşağıdaki liste çakışmaların
> envanteridir, uygulama planı değil.

### 1a. Adım 1 — Devlog adlarını çöz (ÖNCE, kodsuz)

`E-04`: iki dal `README15.md` ve `README16.md` adlarını **farklı içerikle** kullanmış.

| Dosya | Mevcut dal | Katman-5 | Öneri |
|---|---|---|---|
| `README15.md` | Katman 4 Faz 0 (protokol) | Katman 5 ritim | katman-5 → **`README19.md`** |
| `README16.md` | P1 parmaklandırma | Katman 5 Adım 2 (MusicXML) | katman-5 → **`README20.md`** |
| `README17.md`, `README18.md` | P2, P3 | yok | değişmez |

Yeniden adlandırma katman-5 tarafında, birleştirmeden önce yapılır; iç çapraz referanslar da
güncellenir (katman-5 `README16.md` içinde "[README15](README15.md)" atfı var).

**Başarı ölçütü.** Birleştirme sonrası `docs/devlog/` içinde her numara **tek** dosya ve tek konu;
`grep -rn "README1[5-9]\|README20" docs/` ile tüm çapraz referanslar tutarlı; hiçbir devlog içeriği
kaybolmamış (`git log --follow` ile izlenebilir).

### 1b. Adım 2 — Kod çakışmaları

Doğrulanmış çakışma listesi (`git diff --stat HEAD origin/katman-5-ritim`, 104 dosya):

| Dosya | Çakışma türü | Çözüm notu |
|---|---|---|
| `gtab/decoding/edition.py` | 212 satır değişti (katman-5 log-lineer modeli `archive/gtab/decoding/edition_loglinear.py`'a taşıdı) | **`tests/test_edition.py::TestLogLinearDecode` (2 test) bu kodu import ediyor → kırılır.** Ya import yolu güncellenir ya testler arşiv yoluna taşınır |
| `gtab/data/gaps_score.py` | 20 satır (katman-5: `align` → `order`, `lens`, `score_x`; `time_signatures`) | Katman-5 bu değişikliği **9 kayıtta `gaps_tab` ile birebir** doğruladı → katman-5 sürümü alınabilir |
| `gtab/models/inference.py` | 8 satır | elle incele |
| `scripts/eval/eval_edition.py` | 34 satır | elle incele |
| `scripts/train/train_onset.py`, `scripts/eval/eval_hmm.py`, `scripts/eval/diagnose_pitch.py` | 2 satır (yol/import) | arşiv taşımalarının yansıması |
| `gtab/__init__.py` | 2 satır | elle incele |
| `CLAUDE.md`, `CONTRIBUTING.md`, `README.md` | 42 / 59 / 85 satır | **İş 3 ile birlikte** yeniden yazılır (ikisi de eski; birleştirme yerine tek elden güncelleme daha temiz) |
| Checkpoint taşımaları | 14 dosya → `checkpoints/archive/` | `d1_baseline.json` ve `fingerprint.py` yalnız `tabcrnn_deep2.pt`'a bakıyor (kökte kalıyor) → **etkilenmiyor** (doğrulandı) |
| Script arşivlemeleri | `diagnose_strings`, `eval_viterbi`, `eval_gaps_tab`, `sweep_threshold`, `train_crnn`, `train_pitch`, `train_tab`, `train_domain`, `overfit_gaps`, `build_synthtab`, `check_gaps_labels`, `diagnose_onset_features` → `archive/` | **`diagnose_strings.py` İş 2'de gerekiyor** (`A-03`) → arşivleme İş 2'den sonra |
| `ornek.musicxml` (53.526 satır) | katman-5'te yeni | örnek çıktı; repoda tutulup tutulmayacağı kullanıcı kararı |

**Katman-5'ten gelen yeni kod (çakışmasız, doğrudan eklenir):**
`gtab/render/{musicxml,score}.py` · `gtab/pipeline.py` · `gtab/decoding/{rhythm,beat_dbn}.py` ·
`gtab/data/rhythm_labels.py` · `gtab/models/{beat_net,rhythm_net}.py` ·
`gtab/evaluation/rhythm_metrics.py` · `scripts/app/transcribe.py` ·
`scripts/data/{build_rhythm,build_beat_feats}.py` · `scripts/train/{train_rhythm_q,train_beat}.py` ·
`scripts/eval/{eval_rhythm,eval_score}.py` · `docs/SOZLUK.md` · `checkpoints/{beat_net,rhythm_q}.pt`

**Mevcut daldan korunacak (katman-5'te yok, kaybedilmemeli):**
`gtab/evaluation/{protocol,selection,baselines,fingerprint,technique_eval}.py` ·
`gtab/decoding/playability.py` · `gtab/data/contour.py` ·
`scripts/eval/{check_protocol,check_inference_regression,eval_playability,inventory_technique,probe_gaps_f0}.py` ·
`scripts/data/build_contour.py` · `docs/baselines/*` · **`tests/` (8 dosya, 170 test)** ·
`docs/devlog/README15-18` + `archive/`

### 1c. Adım 3 — Tek giriş noktası (`E-01`)

`gtab/pipeline.py` omurga seçilir; `scripts/transcribe.py`'ın üç özgün değeri taşınır:
1. `--domain gaps_val|val` CLI bayrağı + **çıktıda alan yazımı** (katman-5'te `domain` yalnız fonksiyon argümanı; iki alanın sayıları karıştırılmamalı — bu kullanıcıya görünür olmalı).
2. `TranscribeError` ile açık nedenli hata: bozuk/boş/desteklenmeyen ses dosyasında **sahte çıktı üretmeyip durmak**.
3. **Ses modu + yalnız parmak** kombinasyonu: konum ses zincirinden gelir, EditionNet'e `fixed` verilir, modelin serbest tercihi yalnız bilgi olarak sayılır. Katman-5'te bu kombinasyon yok (`--mode edisyon` konumu da değiştiriyor).

ASCII TAB ikinci çıktı formatı olarak korunur.

**Önkoşul.** `E-02`: `NoteEvent`'e `finger: Optional[int] = None` (+ `hand_position`, + ritim için
`beat`). Varsayılan `None` → mevcut davranış birebir korunur.

### 1d. Birleştirmenin başarı ölçütleri

| Ölçüt | Doğrulama |
|---|---|
| Hiçbir devlog içeriği kaybolmadı | `docs/devlog/` 1..20 tek dosya/konu; `git log --follow` |
| Tüm testler geçiyor | `python -m unittest discover -s tests -t .` → 170+ test, hata yok |
| Protokol denetimi geçiyor | `python -m scripts.eval.check_protocol` → 17/17 |
| Eski çıkarım yolu kırılmadı | `python -m scripts.eval.check_inference_regression` → "REGRESYON YOK" |
| Ritim önbellekleri kuruldu | `build_rhythm` + `build_beat_feats` (`E-10`; 808 MB disk) |
| Ritim sayıları yeniden üretildi | `eval_rhythm --qnet rhythm_q.pt --beatnet beat_net.pt --user-tempo` → gaps_val IOI 0,731 (±0,002) |
| MusicXML yazıcısı doğrulandı | `eval_score` → gidiş-dönüş ≥ 0,999, ölçü doluluğu %100 |
| Tek komut çalışıyor | tek giriş noktası hem `.musicxml` hem ASCII TAB üretiyor |

**Risk.** En büyük risk `edition.py` çakışması: yanlış çözüm hem parmak modelini hem 20 testi
etkiler. Öneri: bu dosyayı **en son** ve elle çöz; sonrasında `eval_edition` ile gaps_val
tel/parmak/ürün değerlerinin **birebir** (0,766 / 0,675 / 0,382) geldiği doğrulanır — bu üçlü
P2'de referans olarak sabitlendi ve denetimde de aynı çıktı.

---

## İş 2 — Üç ucuz ölçüm (KRİTİK, eğitim yok)

Üçü de yalnız ölçüm; model/checkpoint/önbellek değişmez. **İş 1'den önce de yapılabilir** ve
`diagnose_strings.py` arşive gitmeden önce yapılmalı (`A-03`).

### 2a. Tel başına doğruluk — `A-01`, `A-05`

**Hipotez.** "A telinde ~0,60 doğruluk" iddiası iki taban eski bir modelden geliyor; harmonik
istifleme genel oracle tel doğruluğunu 0,797 → 0,878'e çıkardı, A teli yeniden ölçülmedi.

```bash
python -m scripts.eval.diagnose_strings --ckpt tabcrnn_deep2.pt --splits val val_comp
# ikinci koşu, kalibre eşikle (GuitarSet alanı):
python -m scripts.eval.diagnose_strings --ckpt tabcrnn_deep2.pt --splits val val_comp --threshold 0.7
```

**Çıkacak bilgi.** Bölüm B: tel karışıklık matrisi, tel başına doğruluk, perde bölgesi kırılımı
(`A-05`), aday sayısı / süre / eşzamanlı nota kırılımı. Bölüm D: güven (hatalar kararsız mı
sistematik mi).

**Karar kuralı (ölçümden önce yazılıyor).**
- A teli ≥ 0,85 → sorun kapanmış; `CLAUDE.md:117` ve `ISSUE_REGISTER A-01` **ÇÖZÜLDÜ** işaretlenir.
- 0,70-0,85 → kısmen; tel tarafı iyileştirmesi Katman 6'ya ertelenir.
- < 0,70 → açık sorun; adaylar sırayla: (a) 36 bin/oktav önbellek (daha ince frekans çözünürlüğü), (b) hex öğretmen damıtması. İkisi de `CLAUDE.md:117-118`'de kayıtlı; **hiçbiri bu ölçüm yapılmadan başlatılmaz.**
- Her durumda `CLAUDE.md:117`'deki kaynaksız "0,66" sayısı ölçülen değerle değiştirilir.

**Önemli.** README10 ile kıyaslanabilirlik için ilk koşu eşik 0,8 (script varsayılanı) olmalı;
karışıklık matrisi **oracle notalarda** çıkıyor (gerçek perde + sınırlar verilir), yani bu bir
tel seçimi ölçümüdür, tab F1 değil — `assert_comparable` grubu farklı.

### 2b. "Perde hiç aktif değil" kategorisinin güncel oranı — `B-01`

**Hipotez.** Kategori son kez deep2 **öncesi** ölçüldü (%40,2 gaps_val). deep2 GAPS nota F1'i
0,720 → 0,779 çıkardı; kategori de düşmüş olmalı.

```bash
python -m scripts.eval.diagnose_pitch --ckpt tabcrnn_deep2.pt \
    --load-cal checkpoints/tabcrnn_deep2.cal.json --splits gaps_val val_comp
```

**Karar kuralı.** Oran < %25 → sorun kapanmaya yakın, kapasite yolu (3b) gereksiz.
%25-35 → açık; `B-04` sonucuna göre veri mi kapasite mi kararı. > %35 → deep2 bu kategoriyi
iyileştirmemiş, kayıp başka yerden gelmiş → kapasite tekrar ölçülmeli.

**Yan ürün.** Bölüm D (fragman kırılımı) `C-01` için gerekli girdi; bölüm E tekrar kaçma (`B-02`).

### 2c. Eğitim kapsamı ölçümü — `B-04` (YENİ, yazılması gerekiyor)

**Neden kritik.** Görev tanımındaki iddia ("kaçırılan notaların ~%36'sı **eğitimde hiç görülmeyen
perdelerden**") projede ölçülmüş olan şeyle **aynı değil**. Ölçülen kategori
(`diagnose_pitch.py:135` → `"perde hic aktif degil"`) modelin **çıkış duyarlılığı** hakkındadır:
notanın zaman aralığı boyunca o perdenin olasılığı hiç eşiği geçmemiş (`col.max() <= thr`).
Eğitim verisi kapsamıyla ilgisi yok. "Eğitimde görülmeyen perde/fret" ölçümü **hiç yapılmadı**.

Bu ayrım SynthTab araştırma sorusunun (İş 6) gerekçesini belirliyor:
- Kaçan notalar gerçekten eğitimde az/hiç görülmemiş perde-fret hücrelerinden geliyorsa → **veri kapsamı** hipotezi güçlenir → sentetik ön-eğitim mantıklı.
- Kaçanlar eğitimde bol bol görülmüş hücrelerden geliyorsa → sorun kapsam değil **model duyarlılığı/kapasite** → SynthTab'in katkısı beklenmemeli (ve kapasite ölçümünün 3b'yi gereksiz bulması tutarlı olur).

**Önerilen ölçüm (eğitim yok, dakikalar).** Yeni script `scripts/eval/coverage_gap.py`:
1. Eğitim kapsamı histogramı: `gaps_fit` (perde) + GuitarSet `train`/`train_comp` (perde × tel × fret) → her hücre için eğitim nota sayısı.
2. Kaçan nota listesi: `diagnose_pitch`'in bölüm A çıktısından (ya da aynı mantıkla) `gaps_val` kaçanları.
3. Kesişim: kaçan notaların, eğitimde **0 / 1-10 / 11-100 / 100+** kez görülmüş hücrelere düşme oranı; aynı dağılım **bulunan** notalar için de çıkarılır (taban).
4. Protokol: `gaps_training_files()` ile dosya listesi, ölçüm `gaps_val`'de, `gaps_test` kapalı.

**Karar kuralı.** Kaçanların eğitimde-az-görülmüş hücrelere düşme oranı, bulunan notalardakinden
**belirgin yüksekse** (ör. ≥ 1,5 kat) → veri kapsamı hipotezi desteklenir → İş 6 Adım 1 anlamlı.
Aksi hâlde İş 6 (b) seçeneğine (tezde yeniden çerçeveleme) gidilir.

**Not.** GAPS'te tel etiketi yalnız `gaps_tab`'da (165 kayıt) olduğu için fret düzeyi kapsam
GuitarSet üzerinden, perde düzeyi kapsam GAPS üzerinden ölçülebilir; ikisi ayrı raporlanmalı
(`assert_comparable` mantığı: farklı birim).

---

## İş 3 — Vitrin dokümanları ve künye (KRİTİK, yarım gün)

`G-05` + `G-06`. Maliyeti en düşük, tez görünürlüğü en yüksek iş.

**3a. Kök `README.md`.**
- Sonuç tablosunun son satırları: 3.10 / 3.11 / 3.12 / 3.13 / Katman 5 eklenir; "güncel en iyi" `tabcrnn_deep2.pt` olur.
- **Yeni kolon: "split cinsi"** — 0,777 ve 0,924'ün yanında "validation (bağımsız test değil)", 0,779'un yanında "bağımsız test (icracı-ayrık)". Bu, `A-04`'ün tezdeki en büyük yanlış okuma riskini kapatır.
- Yol haritası kutuları güncellenir (3.10-3.12 kapandı, 3.13 P1-P3 bitti, Katman 4 **beklemede** gerekçesiyle, Katman 5 ritim+MusicXML kapandı).
- Bozuk kurulum satırı düzeltilir: `.venv\Scriptsctivate` → `.venv\Scripts\activate`.
- Test komutu eklenir (`.venv` ile koşulması notuyla — `E-09`).

**3b. `CONTRIBUTING.md`.** Taban `rep_off` → `deep2`; checkpoint tablosu güncel; `tests/` bölümü
(170 test) eklenir; "Açık işler" bölümü `ISSUE_REGISTER.md`'ye yönlendirir.

**3c. `docs/SOZLUK.md` §5** (birleştirmeden sonra). §5b "Güncel taban `tabcrnn_rep_off.pt`" →
`tabcrnn_deep2.pt` ve sayıları; §5a checkpoint tablosuna deep/deep2/edition_seq/rhythm_q/beat_net
satırları; §5c çözümleme ayarları deep2 kalibrasyonundan.

**3d. `checkpoints/PROVENANCE.md` (yeni).** Checkpoint başına: dosya · komut · init checkpoint ·
eğitim split'leri · seed · seçim split'i + ölçütü · kabul kararı · devlog referansı. Bilgi
devlog'larda dağınık hâlde **zaten var** (bu denetimde `MODEL_DATASET_INVENTORY.md §4`'te canlı
üç model için derlendi); yeni ölçüm gerekmez. Tez eki olarak doğrudan kullanılabilir.

**3e. `requirements.txt`** başlığı: Python 3.10 → 3.11 (ortam ölçüldü).

**Başarı ölçütü.** `README.md` ve `CONTRIBUTING.md`'deki her sayı `docs/baselines/d1_baseline.json`
ile uyuşuyor (`tests/test_baselines.py` o dosyayı zaten sabitliyor); her manşet sayının yanında
split cinsi var.

---

## İş 4 — P4: akor içi parmak ve açıklık kısıtı (YÜKSEK)

`D-01` + `D-02` + `C-03`. P3 bu işi ölçümle hazırladı ve kabul ölçütünü önceden yazdı.

**Neden şimdi.** Sunum slayt 11'in "fiziksel kuralları ihlal etmeyen öneri" iddiası şu an
karşılıksız: model, örtüşen basılı çifti olan akorların **%8,52'sinde** fiziksel olarak imkânsız
parmaklandırma üretiyor; editör (partisyon) aynı kurallarda **%0,00** ihlal veriyor — yani kural
kümesi fazla katı değil, model yetersiz.

**Kritik tasarım notu (P3'ün ölçtüğü).** İki kısıt **birlikte** eklenmeli. `README18 §5.5`
gösteriyor ki tel teklik kısıtı tek başına, açıklık kısıtı olmadan, saçma konumlar üretiyor:
`275_Lw1wc` 123,582 sn'de model E4'ü (MIDI 64) 5. telde açık çalmak yerine **0. telin 24.
perdesine** atamış, çünkü 69 zaten 5. teli kullanıyordu. Açıklık 22 yarım ses. Yani **önce
açıklık cezası, sonra parmak kısıtı** ya da ikisi aynı çözümde.

**Yapılacak.**
1. `resolve_chords` mantığının parmağa taşınması: zamanda **örtüşen** notalarda bir parmak yalnız bir perdede (aynı perdede çok tel = barre serbest). `C-03`: örtüşme koşulu kısıtın da parçası olmalı (yalnız onset yakınlığı değil).
2. Yumuşak açıklık cezası; üst sınır referansı editör dağılımı (maks 3, ≥4 oranı %0).
3. `D-03`: barre belirsizlik kuralı gevşetilsin (kısmi barre geçerli sayılsın) — ölçüm tarafı.
4. `D-04` + `E-02`: `hand_position` ve `finger` `NoteEvent`'e taşınsın (İş 1c ile aynı değişiklik).

**Kabul ölçütü (P3'ün önerisi, `README18 §8.5`).**

| Ölçüt | Hedef |
|---|---|
| A) çakışma, gerçek kesişim | 22 → **0** |
| B) açıklık ≥4 yarım ses oranı | editöre yaklaşsın (**≤ %1**; şu an %3,59) |
| `gaps_val` tel doğruluğu | **düşmesin** (0,766) |
| `gaps_val` parmak doğruluğu | **düşmesin** (0,675) |
| `gaps_val` ürün doğruluğu | **düşmesin** (0,382) |
| Tüm testler | geçsin (170+) |
| Çıkarım regresyonu | farksız |

**Ek raporlama kuralı (`D-05`).** Sonuçlar **iki paydada** birlikte verilmeli: gerçek kesişim
(2.963 akor; PAYDA-3 = 223-232) **ve** kendi kapsamı (9.582 akor; PAYDA-3 = 1.260). Yalnız
kesişime bakmak iyimser: modelin en kötü şekilleri editörün etiketlemediği bölgede yoğunlaşıyor
(kendi kapsamında maks açıklık 22, kesişimde 6).

**Doğrulama.**
```bash
python -m scripts.eval.eval_playability
python -m scripts.eval.eval_playability --verify-span 12
python -m scripts.eval.eval_edition                  # gaps_val regresyonu
python -m unittest discover -s tests -t .
python -m scripts.eval.check_inference_regression
```

---

## İş 5 — MuseScore 4 doğrulaması ve MusicXML gereksinimleri (YÜKSEK)

> **Dönüştürücü yazılmayacak — zaten var.** `gtab/render/musicxml.py` (katman-5) MusicXML 4.0
> score-partwise üretiyor ve gidiş-dönüş testiyle doğrulanmış (202 partisyon, 159.796 nota →
> **birebir 1,0000**). Bu iş **doğrulama + eksik alan analizi**.

### 5a. JSON → MusicXML sorusunun cevabı

**Projede JSON ara biçimi YOK.** Kullanıcının sorusundaki "JSON veri yapısı" mevcut değil;
`write_musicxml` doğrudan `measures` + paralel numpy dizileri alıyor:

```
write_musicxml(out, measures, pitch, string, fret, sig, beat_q, bpm=..., finger=..., title=...)
```

Yani dönüşüm için gereken alanlar şunlar ve **hepsi bugün üretiliyor**:

| MusicXML'in ihtiyacı | Kaynak | Durum |
|---|---|---|
| nota başlangıç/bitiş | `segment()` → saniye | ✓ |
| MIDI perde | `segment()` | ✓ |
| tel + fret | ses modu: `decode_lattice` · edisyon: `EditionNet` | ✓ |
| parmak numarası (`<fingering>`) | `EditionNet` (edisyon modu) | ✓ |
| akor gruplaması | `order_notes` (30 ms) + `build_measures` (aynı konumdaki notalar akor) | ✓ (`C-03` sınırlamasıyla) |
| tempo (`<metronome>`) | kullanıcı BPM | ✓ (yarı otomatik) |
| ölçü imi (`<time>`) | kullanıcı girdisi | ✓ (tek im) |
| ölçü çizgileri | `build_measures` (1q konumları + ölçü başı indeksi) | ✓ (ölçü içi konum 0,611) |
| nota değerleri | `build_measures` greedy ayırma (tam…128'lik, 1-2 nokta, üçleme) | ✓ |
| akort (`<staff-tuning>`) | parametre (scordatura destekli) | ✓ |
| divisions | vuruş başına 24 | ✓ |

**Eksik olan alanlar (bilerek kapsam dışı, katman-5 README16 "Bilinen sınırlar"):**
- **Ses ayrımı (voicing):** tek ritim sesi yazılıyor; klasik partisyondaki bas ↓ / melodi ↑ sap yönleri ve farklı süreler yok. Süre = sonraki başlangıca kadar (TAB'larda yaygın gösterim). Ayrı bir problem.
- `<tuplet>` köşeli parantezi yazılmıyor (yalnız `<time-modification>`); garip süreler greedy ayrılıyor (ör. 20 tick = noktalı onaltılık + 32'lik üçleme) — **doğru ama okunabilirliği iyileştirilebilir**.
- Tek ölçü imi; ölçü imi değişen parçalar tek kullanıcı imiyle temsil edilemiyor (oracle ölçümde bu parçalarda 0,05-0,85).
- Anakruz ölçüsü susla doluyor.
- 1/24 vuruş ızgarası dışındaki notalar yazılmıyor (%0,6 — 903 nota; 1q da üretemiyor).
- **Teknik işaretleri yok** (bend/slide/legato) — Katman 4 beklemede (`G-04`).

### 5b. MuseScore 4 kontrol listesi (kullanıcı çalıştırır)

Birleştirme sonrası:
```bash
python -m scripts.app.transcribe <kayit.wav> --bpm 80 --meter 3/4 --mode edisyon --out test.musicxml
```
MuseScore 4'te açıp **görsel** olarak kontrol:

| Kontrol | Beklenen |
|---|---|
| Dosya hatasız açılıyor | P1 nota portresi + P2 TAB portresi görünüyor |
| Nota portresi | 8vb sol anahtarı, çalan perde doğru oktavda |
| TAB portresi | 6 çizgi, `<staff-tuning>` standart akort, fret numaraları |
| Parmak numaraları | `<fingering>` nota üstünde (edisyon modunda) |
| Ölçü çizgileri | ölçü imi kadar süre, eksik/fazla yok (ölçüldü: doluluk %100) |
| Bağlar (tie) | ölçü çizgisini aşan süreler bağla bölünmüş |
| Üçlemeler | `<time-modification>` doğru oran (parantez yok — bilinen sınır) |
| Tempo | `<metronome>` kullanıcı BPM'i |

**KRİTİK AYRIM (tezde mutlaka yazılmalı).** MusicXML'in MuseScore'da düzgün açılması
**notaların doğru tahmin edildiğini kanıtlamaz.** İki ayrı sayı var:
- **Biçim geçerliliği:** gidiş-dönüş 1,0000 · dosya geçerliliği %100 · ölçü doluluğu %100 → **ölçüldü, tam**.
- **Müzikal doğruluk:** ölçü içi konum (gaps_test, tek ölçü imli) **0,611** · nota F1 0,779 · ölçü başı F 0,582 → **referansa dayalı, ölçüldü**.

İlk grup "yazıcı doğru çalışıyor" der; ikinci grup "sistem müziği ne kadar doğru çıkarıyor" der.
Görsel inceleme yalnız birinci grubu doğrular.

### 5c. Sırada ne var (katman-5'in kendi açık maddesi)

Katman-5 README16'nın son maddesi açık: *"Görselleştirme: MuseScore'da TAB görünümü (kullanıcı) +
ürün için AlphaTab web görüntüleyici"*. AlphaTab için muhtemelen bir JSON/JS katmanı gerekir —
**bugün yok** (bkz. 5a). Bu, sunum slayt 8 modül 8 ("Application: TAB + audio + senkronize
fretboard") kalemi; slayt 13'te **"Başarırsa ekle"** kolonunda, çekirdek hedef değil.

---

## İş 6 — SynthTab araştırma sorusu: karar (YÜKSEK)

`G-03` + `B-04`. Bu, tezin akademik özgünlük iddiasının merkezinde.

### 6a. Sunumdaki araştırma sorusu (bulundu)

**Slayt 6, "Önerilen araştırma sorusu":**

> *"Büyük ölçekli sentetik nylon-string ön-eğitimi ve gerçek klasik gitar verisiyle domain
> adaptation, polifonik audio-to-TAB genellemesini ne ölçüde iyileştirir?"*

**Slayt 10, "Ana deney":**

> *"GuitarSet-only vs SynthTab→GuitarSet vs SynthTab→GuitarSet→GAPS; tamamen görülmemiş gerçek
> klasik gitar kayıtlarında karşılaştırma."*

Bu deney tablosu `README9.md:43-52`'de kurulmuş; **3.8d satırları hiç doldurulmadı.**

### 6b. Neden atlandı ve neden bu yeterli gerekçe değil

README13 (Katman 3.12) kapasite farkını **0,004** ölçtü ve *"SynthTab (3b) gerekmedi"* sonucuna
vardı (`CLAUDE.md:109`). Bu karar **mühendislik olarak doğru** (derin model yetti, veri eklemeye
gerek kalmadı) ama **araştırma sorusunu cevaplamıyor**:

| Soru | Ne ölçer | Cevaplandı mı |
|---|---|---|
| "Model yeterince büyük mü?" | kapasite: eğitim ≈ doğrulama farkı (0,004) | **Evet** |
| "Sentetik ön-eğitim genellemeyi artırır mı?" | SynthTab→GuitarSet→GAPS vs GuitarSet-only | **Hayır** |

İkisi farklı sorulardır. Kapasite doygunluğu, sentetik ön-eğitimin **yararsız olduğunu** göstermez;
yalnız "mevcut modelin kapasitesi mevcut veriye göre darboğaz değil" der.

### 6c. İki yol (kullanıcı kararı)

**Yol A — soruyu cevapla (önerilen sıra).**

| Adım | İş | Maliyet | Kapı |
|---|---|---|---|
| **0** | `B-04` kapsam ölçümü (İş 2c) | saatler, eğitim yok | hipotez desteklenmiyorsa **dur**, Yol B'ye geç |
| 1 | SynthTab naylon alt kümesini yeniden indir (`luthier_*`, ≤30 saat, float16 CQT) | indirme + disk; `build_synthtab` kodu hazır | **kullanıcı onayı gerekli — bu denetim veri indirmedi** |
| 2 | `tabcrnn_syn.pt`: SynthTab ön-eğitimi → GuitarSet ince ayarı | bir eğitim koşusu | — |
| 3 | `tabcrnn_syn_gaps.pt`: → GAPS ince ayarı (deep2 ile aynı tarif) | bir eğitim koşusu | — |
| 4 | 4 satırlık tabloyu aynı iki sette doldur | değerlendirme | `gaps_test` yalnız kapanışta bir kez |

**Kabul kuralı (README9:70-71'de zaten yazılı, ölçümden önce konulmuş):**
*"GAPS test'te ≥ +0,03 nota F1 **ve** GuitarSet tab F1'de en fazla −0,01 kayıp → kabul. Aksi
hâlde GuitarSet-only model korunur."*

**Kontrollü deney tasarımı (kritik).** Karşılaştırmanın anlamlı olması için:
- **Aynı mimari** (deep2 = derin CNN + perde-onset kafası), aynı hiperparametreler, aynı seed.
- **Aynı GAPS ince ayar tarifi** (`--pitch-weight 2` vb.) — tek değişken SynthTab ön-eğitiminin varlığı.
- **Aynı kalibrasyon prosedürü** (alan içi, `gaps_val`'de seçim) — ön-eğitimli modele yeni kalibrasyon yapılır, deep2'nin `.cal.json`'ı kullanılmaz.
- **Taban = deep2** (0,779), `tabcrnn_comp.pt` (0,667) değil. README9'un tablosu `comp`'u taban alıyordu; o artık çok eski.
- Negatif sonuç da sonuçtur: "sentetik naylon ön-eğitimi, bu model ve veri ölçeğinde GAPS genellemesini artırmadı" **yayımlanabilir bir bulgudur** ve `CLAUDE.md:125-126`'daki ön bulguyla (sentetik naylon model için kolay, tab F1 0,73) tutarlı olur.

**Yol B — soruyu tezde yeniden çerçevele.** Araştırma sorusu şuna daraltılır: *"Gerçek klasik
gitar verisiyle (GAPS) domain adaptation + çalınabilirlik kısıtları, polifonik audio-to-TAB
genellemesini ne ölçüde iyileştirir?"* — bu soru **cevaplanmış durumda** (GAPS nota F1
0,255 → 0,779; ayrıca çalınabilirlik ölçümü ve edisyon modeli). Sentetik ön-eğitim "future work"
kolonuna taşınır ve **gerekçesi kapasite ölçümüyle belgelenir** (0,004 + aşırı öğrenme testi).

**Denetimin değerlendirmesi.** Yol B dürüst ve savunulabilir; Yol A tezi belirgin biçimde
güçlendirir. **Yol A'nın Adım 0'ı her hâlde yapılmalı** — ucuz, eğitim gerektirmiyor ve hangi
yolu seçerseniz onun gerekçesini sağlıyor.

---

## İş 7 — Akor fragmanı ve <100 ms tekrar deneyi (ORTA)

`C-01` + `B-02`. Bu ikisi **aynı kökün iki yüzü**: bir vadiyi "tekrar" sayarsan fragmanı da nota
sayarsın. Eğitimsiz yol GAPS için tükendi (`C-02`, ölçülmüş).

### 7a. Düşük maliyetli başlangıç deneyi (ölçüm, eğitim yok)

**Hipotez.** Fragman ile gerçek tekrarın ayrımı **polifoni bağlamına** bağlıdır: akor içinde
(2+ eşzamanlı nota) fragman olasılığı farklı; tek sesli pasajda vadi daha çok gerçek tekrardır.

**Ölçüm.** `diagnose_pitch` bölüm D çıktısı (fragman / kayık başlangıç / uzama / farklı perde)
**polifoni seviyesine göre** kırılır (1 / 2 / 3 / 4+ eşzamanlı nota). Bu kırılım şu an üretilmiyor
ama altyapı var (bölüm A'da süre/register/polifoni kırılımı mevcut).

**Karar.** Fragman oranı polifoni seviyesiyle belirgin değişiyorsa → çözümleme kuralı polifoniye
koşullandırılabilir (eğitimsiz, ucuz). Değişmiyorsa → eğitim gerekir (7b).

### 7b. Eğitim gerektiren yol (yalnız 7a negatifse)

Aday: **tekrar sayısı kafası** — bir zaman aralığında aynı perdenin kaç vuruşu olduğunu tahmin
eden ek çıkış. Bugüne kadar denenmedi (denenen her şey vuruş/vadi **tespitine** odaklıydı).
Gerekçe: 23 ms kare çözünürlüğünde 60-100 ms IOI = 3-4 kare; "kaç vuruş" sorusu "nerede vuruş"
sorusundan daha az çözünürlük gerektirir.

**Kabul ölçütü (önceden).** gaps_val <100 ms tekrar kaçma %55 → **≤ %35** ve GAPS nota F1 düşmesin
(≥ 0,779 − 0,01) ve GuitarSet tab F1 düşmesin (≥ 0,777 − 0,01).

### 7c. Ritim katmanı neyi onarır, neyi onaramaz (ayrım)

Kullanıcının sorusu: ritim katmanı hangi hataları düzeltir?

| Hata | Ritim katmanı onarır mı | Gerekçe |
|---|---|---|
| Nota değeri yanlış (süre/ızgara) | **EVET** | 1q kuantalayıcı oracle IOI 0,933 (kural 0,821); kuantalama "bulunan notalarda neredeyse çözüldü" |
| Arpej bölünmesi (akor birlikte) | **EVET** | akor birlikte 0,949 → 0,989 |
| Ölçü çizgisi kayması | **KISMEN** | ölçü başı F 0,582; yarı otomatik mod metrik düzeyi çözüyor (r = 1 %96) ama faz hatası kalıyor |
| **Kaçan nota (<100 ms tekrar, fragman)** | **HAYIR** | Kuantalayıcı **var olmayan notayı üretmez.** Girdisi deep2'nin bulduğu notalar; kapsama 0,810 ile sınırlı. Ölçülmüş: "gerçek vuruş + deep2 notası" satırı kapsama %81 |
| Hayalet nota | **HAYIR** | aynı gerekçe: kuantalayıcı notayı silmez, konumlandırır |
| Rubato / yerel tempo | **EVET (kısmen)** | DBN rubato'yu izliyor; kalan kayıp yerel vuruş konumu (gerçek vuruşla 0,927 vs tahminle 0,695) |

**Sonuç:** `B-02` ve `C-01` ritimden **bağımsız** çözülmeli. Ritim katmanı nota bulmanın
üzerine kurulu; altındaki kaybı telafi edemez.

---

## İş 8 — Test boşlukları (ORTA, 1 gün)

| İş | Sorun | Maliyet |
|---|---|---|
| `test_edition.py` kapısına `find_spec("mir_eval")` | `E-09` — hatalı ortamda test **hata** veriyor, atlanmıyor | bir satır |
| `gaps_score.align()` birim testi (sentetik MusicXML: tekrar işaretleri + kapo + fingering) | `G-12` — tüm parmak/ritim etiketlerinin kaynağı testsiz | yarım gün |
| Ritim/render duman testleri (birleştirme sonrası) | `E-05` — katman-5 kodu test ağında yok | yarım gün |
| Parmak izini `gaps_val`'e genişlet | `G-09` — şu an 2 GuitarSet kaydı; GAPS regresyonunu yakalamaz | dakikalar |
| Önbelleğe `schema_version` + `config_hash` | `G-11` — sessiz uyumsuzluk riski | yarım gün |
| İkinci seed koşusu (bir model) | `G-10` — "±0,01 gürültü" varsayımı ölçülmemiş | bir eğitim koşusu |

**Doğrulama:** `python -m unittest discover -s tests -t .` (`.venv` ile) · `check_protocol` ·
`check_inference_regression --split gaps_val --files 1`.

---

## İş 9 — Ek akustik datasetler (EN SON, opsiyonel araştırma)

Sunum slayt 8'de listeli, **hiçbiri entegre edilmedi**: GOAT, IDMT-SMT-GUITAR, Guitar-TECHS,
AG-PT-set, DadaGP, "kendi test setimiz".

**Bu aşamada veri indirme veya yeni dataset entegrasyonu yapılmayacak** (kullanıcı talimatı).
Değerlendirme, ileride sıraya girerse:

| Kaynak | Ne kazandırır | Neden en sonda |
|---|---|---|
| **Kendi klasik gitar test setimiz** | `A-04` (bağımsız test eksikliği) **ve** slayt 12 robustness testini birlikte kapatır — en yüksek değerli kalem | kayıt + etiketleme emeği; etiket kalitesi kontrolü gerekir |
| Guitar-TECHS | farklı kayıt perspektifleri → cihaz/oda robustness | lisans + format doğrulaması |
| GOAT, AG-PT-set | teknik etiketi (`G-04`'ün gerçek çözümü) | Katman 4 beklemede; teknik hedefi slayt 13'te "başarırsa ekle" |
| IDMT-SMT-GUITAR | cross-dataset test | elektro/akustik ağırlıklı, klasik gitara uzak |
| DadaGP | sembolik TAB istatistikleri, sentetik üretim | ses yok; edisyon modeli için ön-eğitim verisi olabilir |

**Not.** "Kendi test setimiz" kaleminin değeri diğerlerinden belirgin yüksek: projenin en büyük
yapısal ölçüm açığını (GuitarSet'te bağımsız test yok) kapatan tek yol ve ek lisans sorunu yok.
Tez takvimi izin verirse İş 9'dan öne alınması düşünülebilir.

---

## Özet: ilk üç iş

1. **İş 0 + İş 1a** — `transcribe.py`'ı yedekle, katman-5 devlog adlarını çöz. Bilgi kaybı riskini kapatır; ikisi birkaç saat.
2. **İş 2** — üç ucuz ölçüm (tel başına doğruluk, "perde hiç aktif değil" güncel oranı, eğitim kapsamı). Eğitim yok; `A-01`, `B-01`, `B-04` kapanır ve İş 4/6/7'nin yönünü belirler. `diagnose_strings.py` arşive gitmeden önce yapılmalı.
3. **İş 3** — vitrin dokümanları + checkpoint künyesi. En düşük maliyet, en yüksek tez görünürlüğü; `A-04`'ün yanlış okunma riskini kapatır.

Sonra İş 1 (tam birleştirme) → İş 4 (P4) → İş 5 (MuseScore) → İş 6 (araştırma sorusu kararı).
