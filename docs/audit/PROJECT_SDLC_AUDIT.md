# SDLC Denetimi — Klasik Gitar Audio-to-TAB Sistemi

> Denetim tarihi: 14 Ekim 2026 · Denetlenen durum: dal `Armagan-katman-4`, HEAD `546c7bd`
> (+ salt okunur karşılaştırma: `origin/katman-5-ritim` `3f1f47d`)
> Kapsam: yalnız inceleme ve ölçüm. Hiçbir kod, checkpoint, önbellek veya dal değiştirilmedi.

## 0. Yöntem ve kanıt tabanı

Bu denetim üç kaynağı karşılaştırır: (a) 21 devlog dosyası (`docs/devlog/README*.md`, 3.820 satır),
(b) 11.853 satır Python (`gtab/`, `scripts/`, `tests/`), (c) denetim sırasında fiilen çalıştırılan
doğrulama koşuları. Çalıştırılan ve **doğrulanan** sonuçlar:

| Koşu | Sonuç |
|---|---|
| `python -m unittest discover -s tests -t .` (.venv) | **170/170 geçti** (8,6 sn) |
| `python -m scripts.eval.check_protocol` | **17/17 geçti** |
| `python -m py_compile scripts/transcribe.py` | derleniyor (takipsiz dosya sözdizimsel olarak sağlam) |

Ölçülmeyen her sayı devlog'daki kaydıyla ve satır referansıyla verilir; denetim bu sayıları
yeniden üretmedi (eğitim/değerlendirme koşusu yapılmadı — talimat gereği).

---

## 1. Gereksinim analizi

**Mevcut uygulama.** Gereksinimler tek bir kaynaktan geliyor:
`inputs/Klasik_Gitar_AI_Proje_Sunumu.pptx` (15 slayt). Slayt 2 girdi/çıktı/kapsam-dışını,
slayt 9 sekiz modüllü mimariyi, slayt 12 katman bazlı ölçüm planını, slayt 13 çekirdek
hedef / "başarırsa ekle" / future work ayrımını tanımlıyor. `CLAUDE.md` her oturumda bu
belgenin okunmasını kural yapıyor (satır 6-8).

**Güçlü yönler.**
- Kapsam-dışı açıkça yazılı (gerçek zamanlı akış, kaynak ayırma, alternatif akortlar) — scope creep'e karşı gerçek bir fren.
- "Minimum güçlü başarı" tanımlı (slayt 13: polifonik nota transkripsiyonu + anlamlı tel/perde TAB + objektif benchmark) ve **ölçümle sağlandığı gösterilmiş** (GAPS nota F1 0,779 / GuitarSet tab F1 0,777).
- Slayt 3'ün "Parmak ≠ ses" ayrımı gereksinim düzeyinde yapılmış ve kodda da korunmuş (iki mod, karıştırılmıyor).

**Eksikler.**
- Gereksinimler izlenebilir bir listeye (ID'li, durumlu) dönüştürülmemiş; slayt → katman eşlemesi devlog metinlerine dağılmış.
- Slayt 6'daki **ana araştırma sorusu** ("Büyük ölçekli sentetik nylon-string ön-eğitimi ve gerçek klasik gitar verisiyle domain adaptation, polifonik audio-to-TAB genellemesini ne ölçüde iyileştirir?") hiçbir devlog'da açık gereksinim kalemi olarak takip edilmiyor ve **bugün cevapsız** (bkz. ISSUE_REGISTER `G-03`, ROADMAP_NEXT §6).
- Slayt 12'deki iki ölçüm kalemi hiç uygulanmamış: "uzman değerlendirmesi" (parmaklandırma) ve "robustness testi: mic vs USB mic vs telefon/oda".
- Katman numaralandırması ile sunum fazları arasında bire bir eşleme yok; `README9.md:41` tanımsız bir "Katman 7"e atıf yapıyor (`README16.md:50` bu çelişkiyi kendisi tespit etmiş).

**Riskler.** Tez savunmasında "hangi gereksinimi karşıladık" sorusuna 21 devlog okumadan cevap
verilememesi. Ana araştırma sorusunun cevapsız kalması (akademik özgünlük iddiası buna dayanıyor).

**Öneri.** Tek sayfalık gereksinim izleme matrisi: slayt kalemi → katman → ölçüm → durum → kanıt
dosyası. `docs/baselines/d1_baseline.json` zaten ölçüm tarafının tek kaynağı; matris ona bağlanır.

**Öncelik: YÜKSEK** (tez çıktısı için; teknik risk düşük).

---

## 2. Kapsam ve kabul kriterleri

**Mevcut uygulama.** Projenin en güçlü süreç özelliği burada. `CONTRIBUTING.md` çalışma kuralı 3:
*"'Ciddi iyileşme' ölçütü devlog'da **önceden** yazılır; sağlanmadan bir sonraki kademeye geçilmez."*
Bu kural fiilen uygulanmış ve ölçüt sağlanmadığında **kabul edilmemiş** örnekleri var:

| Karar | Ölçüt | Sonuç | Kaynak |
|---|---|---|---|
| `tabcrnn_gtab.pt` (GAPS partisyon TAB'ı ses kaybı olarak) | seçim ölçütü | **RET** — hiçbir epoch ep 0'ı geçmedi, dosya = deep2 | `CLAUDE.md:44-47`, README14 |
| `tabcrnn_aug.pt` (CQT perde kaydırma) | tab F1 | **RET** — −0,06 | README8:77 |
| Katman 3.11 Adım 4 | GAPS nota F1 | ölçüt sağlanmadı (−0,013), yine de taban kabul (tab F1 en iyi) | README12 §8f |
| Katman 5 Adım 1b (vuruş takibi) | vuruş F ≥ 0,75 | **sağlanmadı (0,631)** → yarı otomatik moda geçiş, kullanıcı kararı | katman-5 README15 |
| Katman 3.13 Adım 2a (log-lineer edisyon) | parmak > taban | **RET** (parmak < taban) → 2b dizi modeli | `CLAUDE.md:36-39` |

**Güçlü yönler.** Negatif sonuçların kaydedilmesi ve kabul edilmesi. `CLAUDE.md:119-123` ayrı bir
"Denenmiş, işe yaramayan (tekrarlama)" listesi tutuyor — 7 kalem. Bu, araştırma projelerinde en
sık kaybedilen bilgi türü.

**Eksikler.**
- Ölçüt sağlanmadığı hâlde devam edilen iki durumda (3.11 Adım 4, Katman 5 1b) gerekçe "kullanıcı kararı" olarak yazılmış; meşru, ama **ölçüt revizyonu** olarak belgelenmemiş — tezde "eşiği sonradan mı gevşettiniz?" sorusuna açık.
- Kabul kriterleri çoğunlukla tek sayı; güven aralığı yok. Tek seed, "±0,01 gürültü" varsayımı (`CLAUDE.md:57`) kural hâline gelmiş ama ölçülmemiş.

**Riskler.** Tek koşuya dayanan kabul kararları. `README17.md §6` bunu kendisi tespit etmiş:
"kabul kararı tek koşuya dayanıyordu" → P2'de gaps_val için birebir yeniden üretim yapıldı,
**gaps_test sayıları hâlâ tek koşuluk**.

**Öneri.** Ölçüt revizyonu için açık kayıt biçimi ("ölçüt X konuldu, ölçüm Y, karar Z, gerekçe").
Zaten yapılanın adını koymak yeterli; yeni süreç gerekmez.

**Öncelik: ORTA.**

---

## 3. Sistem tasarımı

**Mevcut uygulama.** Katmanlı, geriye dönük uyumlu tasarım: her katman bir öncekini kırmadan
ekleniyor (`CLAUDE.md:14-16`). Teknik karşılığı: modelde **sıcak başlatma**
(`gtab/models/nets.py:206 warm_start_state`) ve kodda **varsayılanların eski davranışı birebir
koruması** (ör. `--no-peak-search` = 3.10 davranışı; `eval_rhythm` bayraksız çıktısı Adım 0 ile birebir).

**Güçlü yönler.**
- Artık bloklar sıfır-gamma ile başlatılıyor → derin modele geçişte **birebir aynı başlangıç** (`CLAUDE.md:103-104`). "Kapasite mi veri mi" sorusunu temiz ayırmayı sağlayan tasarım kararı.
- Ortak nota şeması (`gtab/core/note_event.py:35 NoteEvent`) tüm katmanların paylaştığı tek veri sözleşmesi.
- Modüler sınır net: `gtab/` yalnız import edilir, `scripts/` çalıştırılır, script'ler birbirinden import etmez (`CONTRIBUTING.md` kural 4).

**Eksikler.**
- `NoteEvent` şeması **parmak alanı taşımıyor** (satır 35-50'de `string`, `fret`, `technique` var). Parmaklandırma çıktısı şemaya giremiyor → her tüketici kendi yan kanalını kuruyor (`scripts/transcribe.py` ayrı liste, katman-5 `pipeline.py` ayrı dizi). "Tek veri sözleşmesi" iddiası bu noktada kırık.
- Ritim/ölçü bilgisi de şemada yok (katman-5 `build_measures` çıktısı ayrı sözlük).
- `scripts/eval/` altındaki script'lerin bir kısmı artık ürün yolu değil, tanılama arşivi; ayrım dosya düzeyinde görünmüyor (katman-5 bunu `archive/` ile çözüyor — §12).

**Riskler.** Şema dışı yan kanallar çoğaldıkça entegrasyon maliyeti artar; P6 (entegrasyon) ve
Katman 5 Adım 2 aynı alana iki farklı çözüm getirmiş durumda (`E-01`).

**Öneri.** `NoteEvent`'e `finger: Optional[int]` ve `hand_position: Optional[float]` eklenmesi
(P4/P6'nın zaten planladığı iş, README18 §8.4). Varsayılan `None` → mevcut davranış birebir korunur;
`render_ascii_tab` ve MusicXML yazıcısı aynı kaynaktan okur.

**Öncelik: YÜKSEK** (iki dalın birleştirilmesinin ön koşulu).

---

## 4. Modüler mimari

**Mevcut uygulama.** 35 modül `gtab/` altında 6 pakete bölünmüş (core, data, models, decoding,
evaluation + config/paths/utils); 28 çalıştırılabilir script üç grupta (data/train/eval). Veri
akışı: `SYSTEM_ARCHITECTURE.md` Diyagram A.

**Güçlü yönler.** Bağımlılık yönü tek taraflı ve denetlenmiş: `gtab/decoding/playability.py`
yalnız `transitions.LONG_GAP_SEC` sabitini okuyor, `edition.py` / `edition_net.py` onu import
etmiyor → ürün çıkarım yolu ölçüm modülünden bağımsız (README18 §3). Ölçüm eklemenin modeli
etkilemediğini garanti eden doğru tasarım.

**Eksikler.**
- **Ürün giriş noktası iki kez yazılmış** (`E-01`): `scripts/transcribe.py` (221 satır, takipsiz, yalnız bu makinede) ve `origin/katman-5-ritim`'de `gtab/pipeline.py` (81) + `scripts/app/transcribe.py` (50). İkisi aynı zinciri kuruyor; katman-5 sürümü ritim + MusicXML da içeriyor, yerel sürüm ASCII TAB + parmak tablosu veriyor ve kalibrasyon alanını (`--domain`) kullanıcıya açıyor.
- `gtab/evaluation/` 9 modülle iki farklı amaca hizmet ediyor: ölçüm çekirdeği (`metrics`, `pitch_eval`) ve süreç kapıları (`protocol`, `selection`, `baselines`, `fingerprint`). İkincisi aslında bir "deney yönetimi" paketi; adlandırma bunu yansıtmıyor.
- Kullanılmayan/arşivlik kod mevcut dalda ayrışmamış: `gtab/data/synthtab.py` (150 satır) ham veri silindiği için bugün çalışmaz; `gtab/decoding/viterbi.py`'deki el yapımı Viterbi (`transition_cost`, `viterbi_assign`, `viterbi_transcribe`) ölçümle çürütülmüş ama yolda.

**Riskler.** İki giriş noktasının birleştirilmesi ertelenirse ikisi ayrı bakım isteyecek ve
"ürün çıktısı hangisi?" sorusu tezde cevapsız kalacak.

**Öneri.** ROADMAP_NEXT §1 İş 1: katman-5'in `gtab/pipeline.py`'ı tek omurga seçilir, yerel
`transcribe.py`'ın iki özgün değeri (alan seçimi + parmak farkı bilgisi) ona taşınır.

**Öncelik: KRİTİK.**

---

## 5. Veri tasarımı

**Mevcut uygulama.** Ham veri → önbellek (`.npz`) → PyTorch Dataset üç katmanı. Önbellek git'e
girmez (`.gitignore: /data/`), script'lerle yeniden üretilir. Şema: `SYSTEM_ARCHITECTURE.md`
Diyagram B; envanter: `MODEL_DATASET_INVENTORY.md`.

**Güçlü yönler.**
- Üç farklı etiket formatı (JAMS / MIDI+MusicXML / JAMS-sentetik) tek nota şemasına indirgenmiş — slayt 8'in "veri mühendisliği ihtiyacı" kalemi karşılanmış.
- GAPS partisyon hizalaması **bağımsız kanıtla** doğrulanmış: ölçü başı tahminleri, MIDI'den bağımsız syncpoint'lerle medyan **22,9 ms** uyumlu (katman-5 README15 Adım 0a). mir_eval vuruş penceresi ±70 ms olduğundan etiket güvenilir.
- Etiket kalitesi sayılarla raporlanmış: perde roll uyumu %98,93, "parmak 0 ⇔ boş tel" tutarlılığı %99,4, eşleşme medyanı 16 ms (`README16.md §4`).

**Eksikler.**
- `data/cache/` 825 MB, 8 dizin; **katman-5'in ihtiyaç duyduğu iki önbellek bu makinede yok**: `data/cache/rhythm/` ve `data/cache/beat_feats/` (808 MB). Katman-5 kodunu çalıştırmak önce `build_rhythm` (~3 dk) + `build_beat_feats` gerektirir (`E-10`).
- SynthTab ham verisi (`data/raw/synthtab/SynthTab_Dev.zip`, README9:22'de "indirildi ve doğrulandı") **artık diskte yok**; `data/raw/` yalnız `gaps_hf` (11 GB) içeriyor. Ana araştırma sorusunun deneyi veri olmadan kurulamaz (`G-03`).
- Önbellek şeması belgelenmemiş: hangi `.npz` hangi anahtarları taşıyor bilgisi yalnız okuyucu kodundan çıkarılabiliyor.

**Riskler.** Önbellek yeniden üretilebilirliği script'lere bağlı ama ürettiği **şemanın sürümü yok**;
`config.py` değişirse eski önbellekle yeni kod sessizce uyumsuz olur (`CONTRIBUTING.md` bu riski
"config.py değiştirmek tüm modelleri geçersiz kılar" diye not ediyor, mekanik koruma yok).

**Öneri.** (a) Önbellek dizinlerinin anahtar şemasını tek tabloda belgele; (b) yeni üretilen
`.npz`'lere `schema_version` + `config_hash` yaz (eski önbellek okunmaya devam eder).

**Öncelik: ORTA** (veri kaybı riski yok, yeniden üretim maliyeti var).

---

## 6. Model geliştirme

**Mevcut uygulama.** Tek gövde (CNN → BiLSTM) üzerine katman katman kafa eklenerek büyütülmüş:
`PitchCNN` → `TabCNN` → `TabCRNN` → `TabCRNNOnset` (+ offset, + perde-onset) → `--deep 2` artık
bloklar. Yanında iki sembolik model: `EditionNet` (tel/perde/parmak) ve katman-5'te `RhythmNet`
(kuantalama) + `BeatNet` (vuruş). Envanter: `MODEL_DATASET_INVENTORY.md`.

**Güçlü yönler.**
- **Kapasite ve veri sınırı ayrı ayrı ölçülmüş.** `scripts/train/overfit_gaps.py`: 8 kayıt, 200 epoch, dropout kapalı → GuitarSet 0,995 (kurulum doğru), GAPS onset 0,40 → 0,95 → etiket tavanı yok, darboğaz kapasite (`CLAUDE.md:97-100`). "Neden derin modele geçtik" sorusunun savunulabilir cevabı.
- Her ara karar ölçümle gerekçelenmiş. "Teli vuruş anından seç" hipotezi ölçümle **çürütülmüş** (README10 §C).
- Model türü checkpoint'ten tanınıyor (`gtab/models/inference.py:25 load_model`) → eski checkpoint'ler yüklenebilir kalıyor.

**Eksikler.**
- Slayt 10'un "CNN → CRNN → attention/Transformer" aday listesinden **attention/Transformer hiç denenmedi**. Eksiklik değil, bilinçli kapsam kararı olarak kaydedilmeli (kapasite farkı 0,004 ölçüldü).
- Slayt 10'un **ana deneyi hiç yapılmadı**: "GuitarSet-only vs SynthTab→GuitarSet vs SynthTab→GuitarSet→GAPS". README9:49-52 tabloyu kurdu, 3.8d satırları boş kaldı; README13 kapasite farkı 0,004 olduğu için 3b'yi (SynthTab) gereksiz saydı. **Dikkat: kapasite ölçümü ile araştırma sorusu aynı soru değil** — biri "model yeterince büyük mü", diğeri "sentetik ön-eğitim genellemeyi artırır mı". İkincisi cevapsız (`G-03`).
- `pitchcnn.pt`, `tabcnn.pt` ve 9 ara checkpoint ürün yolunda değil; hangisinin canlı olduğu yalnız `CLAUDE.md` metninden anlaşılıyor.

**Riskler.** Tezin özgünlük iddiası (slayt 6) sentetik ön-eğitim + domain adaptation deneyine
dayanıyor; o deney yok. Mevcut hâlde iddia "klasik gitarda ölçülmüş domain adaptation +
çalınabilirlik" olarak daraltılabilir ama bu, slayttaki araştırma sorusundan farklıdır.

**Öneri.** ROADMAP_NEXT §6: araştırma sorusunu ya (a) kontrollü SynthTab deneyiyle cevapla
(veri indirme + ~2 eğitim koşusu), ya da (b) tezde açıkça yeniden çerçeveleyip kapasite ölçümünü
"neden yapılmadı"nın kanıtı olarak sun. Karar kullanıcının.

**Öncelik: YÜKSEK** (akademik çıktı için kritik, teknik olarak opsiyonel).

---

## 7. Eğitim ve deney yönetimi

**Mevcut uygulama.** Eğitimleri kullanıcı çalıştırıyor; kod + komut + karar kuralı önceden
yazılıyor (`CLAUDE.md:20-24`). Her deney bir devlog bölümü. Katman 4 Faz 0'da bu süreç **koda
bağlandı**: `gtab/evaluation/selection.py:36 EpochSelector` (epoch seçimi, early stopping,
epoch 0 = başlangıç modeli de değerlendirilir, `kept_initial()`, `overfitting(window=3)`).

**Güçlü yönler.**
- `EpochSelector` kurucusu seçim split'ini protokolden doğruluyor → `gaps_test` ile kurulamaz (`ProtocolError`). Süreç kuralının mekanik zorlayıcısı var.
- `kept_initial()` gerçek bir olaydan türetilmiş: `tabcrnn_gtab.pt` denemesinde hiçbir epoch ep 0'ı geçmedi ve dosya deep2 olarak kaldı — kural bunu adlandırdı (README15 §5).
- Kalibrasyon maliyeti ölçülmüş ve yönetilmiş: tam kalibrasyon ~1 saat, `--save-cal`/`--load-cal` ile saniyeler (`CLAUDE.md:114`).

**Eksikler.**
- Deney kayıtları **serbest metin**; makine-okunur deney günlüğü yok. "Hangi checkpoint hangi komutla, hangi seed'le, hangi veriyle üretildi" bilgisi devlog paragraflarından çıkarılıyor; `tabcrnn_deep2.pt` için komut `CLAUDE.md:103-110` içinde dağınık.
- Tek seed. ±0,01 "gürültü" varsayımı (`CLAUDE.md:57`) hiç ölçülmemiş (ikinci seed koşusu yok). Buna karşılık **yeniden-koşu** oynaması ölçülmüş: GPU'da ±0,002 (`d1_baseline.json` toleransı).
- `EpochSelector` ve protokol kapıları **yalnız yeni kodda** zorunlu; mevcut `train_onset`, `train_edition_seq`, `eval_pitch`, `eval_hmm` bilinçli olarak değiştirilmedi (README15 "Açık riskler" 4).

**Riskler.** Eski eğitim yollarının protokol ihlali ilkesel olarak mümkün. Fiilen bir ihlal tespit
edildi ve kayda geçti: `gaps_test` bir kez karar kuralında kullanıldı (README14 Adım 2 kabul kuralı
"gaps_test tel ≥ naif + 0,05"); ağırlıklar test'e uydurulmadı ama kabul/ret kararı test'e bakılarak
verildi (README15 "Neden bu faz" 2 → `G-07`).

**Öneri.** Checkpoint başına tek satırlık üretim künyesi (`checkpoints/PROVENANCE.md`): dosya, komut,
init checkpoint, veri split'leri, seed, seçim split'i + ölçütü, kabul kararı, devlog referansı.
Üretim maliyeti düşük, tez eki olarak doğrudan kullanılabilir.

**Öncelik: YÜKSEK.**

---

## 8. Test stratejisi

**Mevcut uygulama.** Katman 4 Faz 0'a kadar (14 katman) otomatik test **yoktu**; regresyon elle
"birebir aynı" kontrolleriyle yapılıyordu. Faz 0 + P2 + P3 ile stdlib `unittest` üzerine
**170 test** kuruldu (8 dosya). Denetimde çalıştırıldı: **170/170 geçti**.

| Dosya | Test | Kapsam |
|---|---|---|
| `tests/test_playability.py` | 41 | örtüşme, çakışma, açıklık, barre, pozisyon kayması |
| `tests/test_contour.py` | 29 | F0 kontur temizliği (Katman 4 Faz 2A, beklemede) |
| `tests/test_protocol.py` | 25 | izin kapıları, sızıntı kontrolleri, karantina, yarı bölmeleri |
| `tests/test_technique_eval.py` | 21 | nota eşleştirme, macro-F1, boş/hayalet sınıf, confusion |
| `tests/test_edition.py` | 20 | akor gruplaması, parmak adayları, checkpoint davranışı, gaps_val yeniden üretimi |
| `tests/test_baselines.py` | 12 | D1 tablo bütünlüğü, tolerans, grup karıştırma koruması |
| `tests/test_fingerprint.py` | 11 | çıkarım parmak izi istatistikleri |
| `tests/test_selection.py` | 11 | seçim kapısı, early stopping, aşırı öğrenme işareti |

**Güçlü yönler.**
- Harici bağımlılık yok (stdlib `unittest`); testlerin çoğu veri seti/GPU/checkpoint gerektirmiyor → her makinede saniyeler içinde koşuyor.
- İki regresyon katmanı: (a) **D1 referans tablosu** (`docs/baselines/d1_baseline.json` + `baselines.py`) metrik karıştırmayı `assert_comparable` ile engelliyor; (b) **çıkarım parmak izi** (`fingerprint.py`) kare düzeyinde karşılaştırdığı için metrikten hassas ve saniyeler sürüyor.
- `test_edition.py` yalnız kod değil **yayımlanmış metrikleri** doğruluyor: gaps_val tel 0,766 / parmak 0,675 / ürün 0,382 birebir yeniden üretildi (README17 §6; denetimde tekrar görüldü).

**Eksikler.**
- **Kritik yollar test dışı**: `gtab/data/gaps_score.py` (MusicXML hizalama, 255 satır), `scripts/data/build_gaps_tab.py`, `scripts/eval/eval_edition.py` bütün olarak, log-lineer `fit`, uçtan uca ses→parmak zinciri (README17 §8).
- `scripts/transcribe.py` (ürün giriş noktası adayı) **hiç test edilmemiş** ve bu makinede uçtan uca hiç çalıştırılmamış — derlendiği doğrulandı, davranışı doğrulanmadı.
- `tests/test_edition.py` kapı koşulu eksik: `find_spec("torch")` + checkpoint kontrol ediliyor ama zincir `eval_edition → pitch_eval → mir_eval` olduğu için **mir_eval yoksa test atlanmıyor, hata veriyor**. Denetimde sistem Python'ıyla (torch var, mir_eval yok) tetiklendi: `ModuleNotFoundError: No module named 'mir_eval'` (`tests/test_edition.py:242`). `.venv` ile 170/170 geçiyor. README15'in "testler yalnız numpy kullanıyor" ifadesi bu dosya için geçerli değil (`E-09`).
- Katman-5 dalında `tests/` **hiç yok** (dal Faz 0'dan önce, `72295e6`'dan ayrıldı) → birleştirmede ritim/MusicXML kodu test ağına girmemiş olarak gelir (`E-05`).

**Öneri.** (a) `test_edition.py` kapısına `find_spec("mir_eval")` ekle (tek satır); (b) birleştirme
sonrası ritim/render için duman testleri; (c) `gaps_score.align()` için sentetik MusicXML ile birim
test (etiket üretiminin tek doğrulaması şu an istatistiksel).

**Öncelik: YÜKSEK** (a, b) · ORTA (c).

---

## 9. Doğrulama ve değerlendirme

**Mevcut uygulama.** Katman bazlı ölçüm (slayt 12 ile eşlenmiş): nota P/R/F1 (mir_eval, 50/100 ms),
kare perde P/R/F1, tab F1, polifoni kırılımı (1/2/3/4+), oracle tel doğruluğu, tekrar kaçma,
çalınabilirlik metrikleri (A/B/C/D/E), ritim metrikleri (vuruş F, CMLt, AMLt, IOI, ölçü içi konum).

**Güçlü yönler.**
- **Metrik karıştırma aktif olarak engelleniyor.** `gtab/evaluation/baselines.py:87 assert_comparable` farklı gruptaki sayıların kıyaslanmasını hata yapıyor; 0,815 (oracle nota + edisyon modu) ile 0,655 (uçtan uca) aynı tabloda karşılaştırılamıyor. Bu tür projelerde en sık yapılan hatanın mekanik önlemi.
- Koşulların açıkça taşınması: `d1_baseline.json` her satıra "birim", "koşullu/oracle", "split cinsi" yazıyor. `README16.md §5` üç parmaklandırma sonucunu (%63,6 / %43,4 / %33,1) koşullarıyla ayrı ayrı veriyor.
- **Oracle ile uçtan ucanın ayrılması** her katmanda yapılmış; ritimde kayıp ayrışması örnek niteliğinde: kuantalayıcı −0,18 → deep2 notaları −0,02 → vuruş takibi −0,21/−0,24 (katman-5 README15 D).
- `report_label(split)` zorunluluğu: her sayının yanında "validation (bağımsız test değil)" / "bağımsız test (icracı-ayrık)" / "keşif" etiketi.

**Eksikler.**
- **GuitarSet'te bağımsız test yok** ve bu yapısal: 6 oyuncudan 5'i eğitimde, oyuncu 05 hem seçim hem kalibrasyon hem raporlama setinde. Manşet sayılar (tab F1 0,777, nota F1 0,924) **validation**'dır. Faz 0 bunu tespit edip etiketledi ama çözemez; çözüm yeniden eğitim (`A-04`).
- Tel başına doğruluk **güncel tabanda hiç ölçülmemiş**: tek kayıt README10:30 (A teli 0,60), model `tabcrnn_gaps.pt` (iki taban öncesi), eşik 0,8, GuitarSet val, oracle notalar. `CLAUDE.md:117`'deki "A teli 0,66" sayısının **hiçbir devlog'da kaynağı yok** (`A-01`).
- "Uzman değerlendirmesi" (slayt 12, parmaklandırma) ve cihaz/oda robustness testi hiç yapılmadı.
- `gaps_test` sayıları tek koşuluk ve yeniden üretilmemiş (protokol gereği bilinçli).

**Riskler.** Tez savunmasında "0,777 test sonucu mu?" sorusu. Cevap hazır ve dürüst (validation;
Faz 0 bunu belgeliyor) ama manşet sayıların yanında her yerde taşınması gerekiyor; kök `README.md`
bugün bu etiketi taşımıyor.

**Öneri.** Kök `README.md` sonuç tablosuna "split cinsi" kolonu — tek satırlık değişiklik, tezdeki
en büyük yanlış okuma riskini kapatır.

**Öncelik: YÜKSEK.**

---

## 10. Sürüm kontrolü

**Mevcut uygulama.** Katman başına dal (`katman-<no>-<konu>`), `main`'e birleştirme kullanıcı
kararı. 9 uzak dal, 3 yerel. Çalışma ağacı temiz; tek takipsiz dosya `scripts/transcribe.py`.
Stash yok, ek worktree yok.

**Güçlü yönler.** Katman geçmişi dallarda korunmuş; hiçbir dal silinmemiş → her katmanın çıktısı
geri alınabilir. Commit mesajları katman numarasıyla etiketli.

**Eksikler ve riskler.**
1. **`main` 9 commit'te donmuş** (`9fe0b53 Merge remote placeholder repo history`); tüm iş dallarda. Üretim/teslim sürümü yok.
2. **İki aktif dal paralel ilerliyor ve çakışıyor.** Ortak ata `72295e6`; `Armagan-katman-4` 5 commit ileri (Faz 0 + parmaklandırma P1-P3), `katman-5-ritim` 2 commit ileri (temizlik + ritim/MusicXML). 104 dosya farkı.
3. **Devlog numarası çakışması (kritik).** İki dal `docs/devlog/README15.md` ve `README16.md` adlarını **farklı içerikle** kullanmış:
   - `Armagan-katman-4`: README15 = Katman 4 Faz 0 (ölçüm protokolü), README16 = P1, README17 = P2, README18 = P3.
   - `katman-5-ritim`: README15 = Katman 5 ritim/kuantalama, README16 = Katman 5 Adım 2 (MusicXML).

   Düz birleştirmede bu dosyalar çatışır ve **bir tarafın devlog'u kaybolma riski taşır**.
4. Checkpoint'ler git'te tutuluyor (`.pt`); katman-5 bunları `checkpoints/archive/` altına **taşıdı** (git mv). Birleştirmede yol değişikliği + `d1_baseline`/`fingerprint` referansları birlikte düşünülmeli.
5. `scripts/transcribe.py` takipsiz → **yalnız bu makinede var**, yedeği yok.

**Öneri (sırayla).** (a) `scripts/transcribe.py`'ı kaybetmemek için yedekle/commit'le (kullanıcı
kararı); (b) devlog numaralandırmasını birleştirme öncesi çöz (öneri: katman-5'in iki dosyasını
`README19.md`/`README20.md` yapmak — ikisi de 11 Ekim, kronoloji bozulmaz); (c) birleştirmeyi
`main`'e değil, üçüncü bir entegrasyon dalına yap. Ayrıntı: ROADMAP_NEXT §5.

**Öncelik: KRİTİK.**

---

## 11. Dokümantasyon

**Mevcut uygulama.** Dört katman: kök `README.md` (vitrin), `CONTRIBUTING.md` (dosya rehberi +
çalışma kuralları), `CLAUDE.md` (oturum senaryosu + güncel durum), `docs/devlog/README*.md`
(21 dosya, tarihsel kayıt). Katman-5'te ek olarak `docs/SOZLUK.md` (383 satır başvuru + sözlük).

**Güçlü yönler.**
- Devlog kalitesi yüksek: her dosya "neden / ne yapıldı / testler / çalıştırma / karar kuralı / sonuçlar / açık riskler" iskeletini izliyor. Negatif sonuçlar ve **kendi hataları** kayda geçmiş (README18 §6: beş ayrı ölçüm hatası ve düzeltmesi; README16 arşiv notu: sayı hatası tespit edilip doğrusu ayrıca ölçülmüş).
- Arşivleme disiplinli: taşınan devlog kopyaları `sha256` + `cmp` ile birebir doğrulanmış (README16 §8).
- `CLAUDE.md`'deki "Denenmiş, işe yaramayan" listesi tekrar çalışmayı önlüyor.

**Eksikler (ölçülen tutarsızlıklar).**

| Belge | Yazdığı | Gerçek | Kanıt |
|---|---|---|---|
| kök `README.md` sonuç tablosu | son satır Katman 3.9, tab F1 0,739 | güncel taban deep2, 0,777 | `CLAUDE.md:30-34` |
| kök `README.md` yol haritası | 3.10 / 3.13 açık `[ ]`, 3.8 açık | 3.10 / 3.11 / 3.12 kapandı, 3.13 P1-P3 bitti | README13 Karar, README18 |
| kök `README.md` "güncel en iyi" | `tabcrnn_comp.pt` | `tabcrnn_deep2.pt` | `CLAUDE.md:30` |
| kök `README.md` kurulum | `.venv\Scriptsctivate` (bozuk satır) | `.venv\Scripts\activate` | `README.md` Kurulum |
| `CONTRIBUTING.md` tablo | taban `tabcrnn_rep_off.pt`, tab F1 0,760 | deep2, 0,777 | `CLAUDE.md:30` |
| katman-5 `docs/SOZLUK.md` §5b | "Güncel taban `tabcrnn_rep_off.pt`" | deep2 | `CLAUDE.md:30` |
| `README9.md:41` | "Katman 7'ye girdi" | Katman 7 tanımsız (yol haritası 6'da bitiyor) | `README16.md:50` |

README15 bu durumu bilinçli karar olarak kaydetmiş: *"Kök README.md ve geçmiş devlog'lar bilinçli
olarak değiştirilmedi... Yol haritası/devlog listesi güncellemesi Katman 4 kapanışına bırakıldı."*
Yani teknik borç değil, **ertelenmiş iş** — ancak Katman 4 beklemeye alındığı için erteleme
süresiz hâle geldi.

**Riskler.** Projeye bakan üçüncü kişinin (danışman, jüri) ilk okuduğu dosya kök `README.md` ve
o dosya iki taban eski; tezdeki sayılarla repo sayıları uyuşmaz görünür.

**Öneri.** Kök `README.md` + `CONTRIBUTING.md` + `SOZLUK.md §5`'in tek seferde güncellenmesi;
kaynak olarak `docs/baselines/d1_baseline.json` kullanılması (tek doğruluk kaynağı zaten orada).

**Öncelik: YÜKSEK** (düşük maliyet, yüksek görünürlük).

---

## 12. Entegrasyon

**Mevcut uygulama.** Modüller arası entegrasyon sağlam (veri akışı Diyagram A'da kesintisiz).
Ürün düzeyi entegrasyonu ise iki dalda **birbirinden habersiz** iki kez çözülmüş.

| Entegrasyon noktası | `Armagan-katman-4` | `origin/katman-5-ritim` |
|---|---|---|
| WAV → TAB giriş noktası | `scripts/transcribe.py` (takipsiz, 221 satır) | `gtab/pipeline.py` + `scripts/app/transcribe.py` |
| Çıktı biçimi | ASCII TAB + nota tablosu (+ parmak) | **MusicXML** (nota portresi + 6 telli TAB portresi) |
| Ritim / ölçü | yok (saniye cinsinden) | vuruş + ölçü başı + kuantalama + ölçü çizgisi |
| Parmaklandırma | `--fingering`, konum **sabit** (`fixed`), yalnız parmak eklenir | `--mode edisyon`, konumu da EditionNet seçer |
| Kalibrasyon alanı | `--domain gaps_val\|val`, çıktıda yazılır | `domain="gaps_val"` (fonksiyon argümanı, CLI'da yok) |
| Ölçüm altyapısı (protokol/test) | var (170 test, 17 kontrol) | **yok** |
| Çalınabilirlik ölçümü | var (`playability.py`, 41 test) | yok |

**Güçlü yönler.** Her iki sürüm de mevcut bileşenleri **yeniden yazmadan bağlıyor**; ikisi de
"ses modu / edisyon modu karıştırılmaz" kuralına uyuyor (kullanıcı kararı, `CLAUDE.md:51-53`).
Katman-5 tarafında MusicXML yazıcısı **gidiş-dönüş testiyle** doğrulanmış: 202 partisyon,
159.796 nota → birebir **1,0000**, ölçü doluluğu bozuk 0.

**Eksikler.**
- `NoteEvent` parmak/ritim taşımadığı için her iki entegrasyon da şema dışı yan kanal kullanıyor (`E-02`, `E-03`).
- `scripts/transcribe.py` bu makinede uçtan uca hiç çalıştırılmamış (ses dosyası olarak yalnız GAPS ham kayıtları var; `inputs/` altında test kaydı yok).
- MuseScore 4 ile görsel doğrulama **yapılmamış**: katman-5 README16 son maddesi açık (`[ ] Görselleştirme: MuseScore'da TAB görünümü (kullanıcı)`).

**Riskler.** `E-01` birleştirmede çakışmanın merkezi. Ayrıca katman-5 `gtab/decoding/edition.py`'ı
212 satır değiştirmiş (log-lineer modeli `archive/gtab/decoding/edition_loglinear.py`'a taşıyarak);
mevcut dalda `tests/test_edition.py` içindeki `TestLogLinearDecode` (2 test) o kodu import ediyor
→ birleştirmede **test kırılır** (`E-05`).

**Öneri.** ROADMAP_NEXT §5'te üç adımlı entegrasyon planı (önce devlog adları, sonra kod, sonra
testler) ve çakışma listesi verildi. Otomatik merge yapılmadı (talimat gereği).

**Öncelik: KRİTİK.**

---

## 13. Ürün çıktısı

**Mevcut uygulama.** Üç çıktı biçimi: ASCII TAB (`gtab/decoding/decode.py:157 render_ascii_tab`),
nota tablosu (`scripts/transcribe.py:note_table`), MusicXML (katman-5
`gtab/render/musicxml.py:write_musicxml`).

**Güçlü yönler.**
- MusicXML çıktısı **standart ve doğrulanmış**: score-partwise 4.0, GAPS düzeni (P1 nota portresi 8vb sol anahtarı + `<fingering>`, P2 TAB portresi 6 çizgi + `<staff-tuning>` + `<string>`/`<fret>`), divisions = vuruş başına 24. Gidiş-dönüş birebir 1,0000.
- Uçtan uca dosya geçerliliği %100 (geri okuma + tek tip ölçü; gaps_val 15 + gaps_test 23 kayıt).
- Çıktı MuseScore / Guitar Pro / Sibelius / AlphaTab tarafından açılabilir biçimde; yazıcı seçimi gerekçeli (GAPS partisyonları da MusicXML olduğu için aynı okuyucuyla doğrulanabiliyor).

**Eksikler.**
- **Ölçü içi konum doğruluğu 0,611** (gaps_test, tek ölçü imli): bulunan notaların ~%61'i ölçüde doğru yerde. Nota değerleri daha iyi (IOI 0,695); kalan hatanın bir kısmı ölçü başı fazı (ölçü başı F 0,582) — yani ölçü çizgilerinin kayması.
- Ritim **yarı otomatik**: kullanıcı yaklaşık BPM + ölçü imi giriyor. Tam otomatik hedefler sağlanmadı (vuruş F 0,631 < 0,75; uçtan uca IOI 0,695 < 0,833).
- Bilerek kapsam dışı: tek ritim sesi (ayrı sesler/sap yönü yazılmıyor), `<tuplet>` parantezi yok, tek ölçü imi, anakruz susla doluyor.
- MuseScore 4'te **görsel doğrulama yapılmamış**. Not: MusicXML'in MuseScore'da açılması notaların doğru tahmin edildiğini **kanıtlamaz**; biçim geçerliliği (ölçüldü: %100) ile müzikal doğruluk (ölçüldü: 0,611) ayrı sayılardır.
- Slayt 8'in "Application: TAB + audio + senkronize fretboard" kalemi yok (AlphaTab web görüntüleyici katman-5'te "sırada" işaretli).

**Öneri.** MuseScore 4 doğrulaması için hazır kontrol listesi ve gereken alanlar:
`ROADMAP_NEXT.md §4` (dönüştürücü yazılmadı — zaten var).

**Öncelik: ORTA** (biçim hazır; iyileştirme ritim tarafında).

---

## 14. Bakım ve hata yönetimi

**Mevcut uygulama.** Hata yönetimi büyük ölçüde **devlog'daki "açık riskler" bölümleri** üzerinden
yürüyor. Her faz kendi sınırlamalarını listeliyor (README18 §7: 8 sınırlama; README15 "Açık riskler":
5 madde). Kodda `ProtocolError` ve `TranscribeError` gibi açık nedenli hata türleri var;
`transcribe.py` bozuk ses dosyasında **sahte çıktı üretmek yerine duruyor**.

**Güçlü yönler.** Hata türlerinin kullanıcıya açık nedenle gösterilmesi; ölçüm hatalarının
devlog'a kendi başlığıyla yazılması (README18 §6) — bulunan hatayı saklamayan bir kayıt kültürü.

**Eksikler.**
- Sorunlar **tek bir kayıtta toplanmamış**; 21 devlog'a dağılmış. Bu denetimin ürettiği `ISSUE_REGISTER.md` ilk toplu envanter.
- Kapanmış sorunların "kapandı" işareti yok; eski devlog'da açık görünen bir sorun sonraki fazda çözülmüş olabiliyor (ör. README10'daki tel hatası analizi 3.9'da kısmen kapandı, README11'de yeniden açıldı).
- `395_w41wc` karantinası yalnız yeni kodda etkili; eski script'ler `gaps_files`'ı doğrudan çağırdığı için mevcut `tabcrnn_deep2.pt` bu kaydı perde denetimi olarak görmüş olabilir (README15 "Açık riskler" 3 → `G-08`).

**Öneri.** `ISSUE_REGISTER.md`'nin canlı tutulması (her faz kapanışında durum güncellemesi).
Tek dosya, 38 kalem; bakımı düşük maliyetli.

**Öncelik: ORTA.**

---

## 15. Tekrarlanabilirlik

**Mevcut uygulama.** Sabit seed (`gtab/utils.py:7 set_seed`, varsayılan 1), sabit bağımlılık
sürümleri (`requirements.txt`, 9 Ekim 2026'da çalışan ortama kilitlenmiş, `numpy<2` zorunluluğu
gerekçeli), önbellek script'lerle yeniden üretilebilir, checkpoint'ler git'te.

**Güçlü yönler.**
- **Çıkarım parmak izi** (`fingerprint.py` + `inference_fingerprint.json`): `tabcrnn_deep2.pt` ile `val`'in ilk 2 kaydında kafa başına istatistik + tab argmax histogramı; cihaz varsayılanı CPU (deterministik; GPU ±0,002 oynuyor). Faz 0'da iki koşu birebir geldi.
- Toleranslar ölçümle konulmuş: yeniden-koşu 0,002 (GPU iç oynaması), yeniden-eğitim 0,01 (seed).
- `d1_baseline.json` + 12 test: tablo değiştirilirse test düşer.
- Yayımlanmış metriklerin yeniden üretimi fiilen kanıtlanmış: gaps_val tel/parmak/ürün birebir (README17 §6); denetimde tekrar doğrulandı.

**Eksikler.**
- **Önbellek bu makinede eksik**: katman-5 için `rhythm/` ve `beat_feats/` (808 MB) yok; SynthTab ham verisi silinmiş. Katman-5 ölçümleri bugün olduğu gibi tekrarlanamaz (önce iki build adımı gerekir).
- GuitarSet ham verisi `data/raw/` altında değil (`GUITARSET_DATA_HOME` ile dışarıda) → önbellek yeniden üretimi ortam değişkenine bağlı; belgelenmiş ama kontrol edilmiyor.
- Python sürüm uyuşmazlığı: `requirements.txt` "Python 3.10" diyor, `.venv` ve sistem Python'ı **3.11**. Çalışıyor, ama belge yanlış.
- Önbellek dosyalarında şema/config sürümü yok (§5).
- Tek seed → "±0,01 gürültü" varsayımı doğrulanmamış.

**Riskler.** "Sonuçları yeniden üretin" talebinde en büyük engel eksik önbellek + silinmiş SynthTab
verisi. Checkpoint'ler git'te olduğu için **değerlendirme** tekrarlanabilir; **eğitim** tekrarlanamaz
(eğitim verisi önbelleği yeniden üretim gerektirir).

**Öneri.** (a) Yeniden üretim sırasını tek dosyada sırala (hangi komut, ne kadar süre, kaç GB);
(b) `requirements.txt` başlığını 3.11 olarak düzelt; (c) yalnız bir model için ikinci seed koşusu —
gürültü varsayımını ölçmek adına (tez için değerli; maliyeti bir eğitim koşusu).

**Öncelik: ORTA.**

---

## SDLC yaklaşımı değerlendirmesi

Proje **hibrit**: çekirdeği araştırma odaklı geliştirme (research-driven development), üzerine
iteratif/artımlı yazılım disiplini giydirilmiş. Agile'a tam uyum yok ve bu bir eksiklik değil.

**Araştırma odaklı geliştirmeye benzeyen yönler (baskın karakter).**
- Birim "sprint" değil **hipotez**: ölç → hipotez kur → tek değişken değiştir → yeniden ölç → kabul/ret.
- Negatif sonuç birinci sınıf çıktı (`CLAUDE.md:119-123`, 7 kalem).
- Kapsam ölçüm sonucuna göre değişiyor: kapasite ölçümü SynthTab adımını gereksiz kıldı; teknik etiketi bulunmayınca Katman 4 beklemeye alınıp parmaklandırmaya dönüldü (`README16.md §1`).

**İteratif/artımlı yazılım yönleri.**
- Katman modeli = sürüm artışı; her katman öncekini kırmadan ekleniyor, geriye dönük uyumluluk "birebir aynı sonuç" testiyle zorlanıyor (`CONTRIBUTING.md` kural 5).
- Dal başına katman; devlog = sürüm notu.

**Agile'dan sapan yönler (bilinçli).**
- Sabit süreli iterasyon yok; faz uzunluğu ölçüm sonucunun belirlediği kadar.
- "Çalışan yazılım" her iterasyonda teslim edilmiyor: ürün giriş noktası (WAV→TAB) 14 katman sonra, Katman 5'te ortaya çıktı. Araştırma projesinde makul, ama **entegrasyonun sona bırakılmasının bedeli** şimdi iki rakip giriş noktası olarak görünüyor (`E-01`).
- Kullanıcı = tek paydaş + tek operatör (uzun eğitimleri kendisi koşuyor) → geri bildirim döngüsü hızlı ama tek kişiye bağlı.

**V-model / test-first benzerliği.** Yalnız Katman 4 Faz 0'da: ölçüm protokolü, kabul kriterleri ve
test altyapısı **uygulamadan önce** kuruldu (170 testin 80'i o fazda, hiç model eğitilmeden). Bu tek
faz projenin en olgun parçası.

**Genel olgunluk değerlendirmesi.**

| Alan | Olgunluk | Gerekçe |
|---|---|---|
| Ölçüm ve deney disiplini | **Yüksek** | Önceden yazılı kabul ölçütü, negatif sonuç kaydı, oracle/uçtan uca ayrımı, metrik karıştırma koruması |
| Veri protokolü ve sızıntı kontrolü | **Yüksek** | İzin kapıları koda bağlı, 17 denetim, karantina, kanal düzeyi sızıntı kontrolü |
| Dokümantasyon (devlog) | **Yüksek** | 21 dosya, tutarlı iskelet, kendi hatalarını kaydediyor |
| Test altyapısı | **Orta** | 170 test var ama yalnız son 4 fazda; kritik etiket üretim yolu testsiz |
| Dokümantasyon (vitrin) | **Düşük** | Kök README iki taban eski |
| Sürüm / dal yönetimi | **Düşük** | İki dal paralel ve devlog adları çakışıyor; `main` donmuş |
| Entegrasyon | **Düşük** | İki rakip giriş noktası; şema parmak/ritim taşımıyor |
| Tekrarlanabilirlik | **Orta** | Değerlendirme tekrarlanabilir; eğitim için eksik önbellek + silinmiş veri |

**Tek cümlelik özet.** Projenin bilimsel süreci (ölç → hipotez → kabul ölçütü → karar) olgun ve
savunulabilir; zayıf halka yazılım konfigürasyon yönetimi — iki dalın paralel ilerlemesi, çakışan
devlog adları, iki kez yazılmış ürün giriş noktası ve iki taban geride kalan vitrin dokümanı.
