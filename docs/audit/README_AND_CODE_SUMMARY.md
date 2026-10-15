# README ve Kod Özeti — Pratik Çalışma Notu

> 14 Ekim 2026 · `Armagan-katman-4` @ `546c7bd` · 21 devlog + 2 katman-5 devlog'u incelendi
> Hiçbir kod, checkpoint, dataset, devlog veya dal değiştirilmedi. Eğitim/test çalıştırılmadı.
> `[B]` = yalnız `origin/katman-5-ritim` dalında.

---

# 1. README ÖZETLERİ

**README.md — Orijinal yol haritası (Katman 0)**
Yapılan: 7 katmanlık plan (Katman 0-6) ve tasarım ilkesi yazıldı; `config.py`, `instrument.py`, `note_event.py`, veri edinme sabitlendi.
Sorun: Yok (plan belgesi).
Durum: Tamamlandı.

**README1 — Katman 1**
Yapılan: Ses → log-CQT (`features.py`), nota → kare etiketleri (`labels.py`), solo filtresi + oyuncuya göre train/val + `.npz` önbellek.
Sorun: Yok.
Durum: Tamamlandı.

**README2 — Katman 2**
Yapılan: PitchCNN eğitildi, `decode.py` ile kareler → nota → naif tab → ASCII TAB zinciri kuruldu.
Sorun: Planda CREPE yazıyordu, PitchCNN yapıldı (plan sapması).
Durum: Belirsiz — bilinçli bir karar gibi görünüyor ama hiçbir yerde gerekçelendirilmemiş.

**README3 — Katman 3**
Yapılan: Tel/perde etiketleri (`tab_labels.py`) ve TabCNN eklendi; iki ayrı metrik tanımlandı (pitch F1 / tab F1). Katman 2 kare F1 **0,853**.
Sorun: tab F1 (~0,59) << pitch F1 (0,85) → kayıp tel atamasında.
Durum: Kısmen — 3.5-3.12'de 0,777'ye çıktı, tel tarafı hâlâ açık (bkz. S-01).

**README4 — Katman 3 konsolidasyon**
Yapılan: `class_weights` bug'ı düzeltildi (mean-normalizasyon modeli sessizliğe çökertiyordu), `mode_filter`/`smooth_tab` ve sessizlik-marjı taraması eklendi.
Sorun: Yok.
Durum: Tamamlandı.

**README5 — Katman 3.5**
Yapılan: CNN + 2 katman BiLSTM (`TabCRNN`) eklendi; tab F1 0,586 → **0,634**.
Sorun: Tel sıçraması azaldı ama sürüyor.
Durum: Kısmen — 3.9'da harmonik istifleme ile büyük ölçüde iyileşti (oracle tel 0,850).

**README6 — Katman 3.6**
Yapılan: El yapımı "en az el hareketi" maliyetiyle Viterbi tel ataması kodlandı ve ölçüldü.
Sorun: Sentetikte işe yaradı (0,600 → 0,684), **gerçek veride katkı yok**.
Durum: Tamamlandı (RET kararı verildi, tekrarlanmayacaklar listesinde).

**README7 — Katman 3.6b**
Yapılan: Veriden öğrenilen log-lineer geçiş modeli (`transitions.npz`); oracle tel naif 0,206 → **0,662**.
Sorun: Uçtan uca CRNN'e katkı **0** (`w_tr=0` seçiliyor); darboğaz akustik tel ayrımı (oracle'da %74).
Durum: Tamamlandı (RET; `transitions.npz` kodda duruyor ama katkısız).

**README8 — Katman 3.7**
Yapılan: GuitarSet comp kayıtları (180) eklendi → 0,667; CQT perde kaydırmalı çoğaltma denendi.
Sorun: Çoğaltma **−0,06 zarar verdi** (tel kimliğini bozuyor); comp kazancı gürültü sınırında (+0,013).
Durum: Tamamlandı (çoğaltma RET; "GuitarSet-mic'te azalan getiri" kararı).

**README9 — Katman 3.8 (Phase 3-4)**
Yapılan: Klasik gitar planı; GAPS indirildi + önbelleğe alındı; alan farkı ölçüldü (GuitarSet modeli GAPS'te nota F1 **0,255**); alt adımlar 3.8a-3.8e tanımlandı.
Sorun: **3.8d (SynthTab nylon ön-eğitimi) hiç yapılmadı**; 12 icracı hem train hem test'te çıktı.
Durum: Sızıntı `--disjoint` ile çözüldü (ÇÖZÜLDÜ); **3.8d AÇIK** (bkz. S-15).

**README10 — Katman 3.9**
Yapılan: Hata analizi (`diagnose_strings`) → onset kafası → harmonik istifleme; tab F1 **0,739**, oracle tel **0,850**, akorlu nota F1 0,46 → 0,72.
Sorun: **A teli 0,60** (çoğu D'ye kayıyor), B→G, e→B; tel hatalarının %98-99'u komşu tel; hatalar kararsızlık kaynaklı (marj 0,09).
Durum: **AÇIK** — harmonik istifleme sonrası yalnız e teli yeniden ölçüldü (0,82→0,92); A teli bir daha hiç ölçülmedi (bkz. S-01).

**README11 — Katman 3.10**
Yapılan: Polifonide perde; kaçan/hayalet kırılımı, çözümleme kuralları (histerezis, refrakter, yedek, tepe, yeniden vuruş), offset kafası; tab F1 **0,760**.
Sorun: val_comp çıkış ölçütü (0,780/0,830) **sağlanmadı**; kaçanların %36-45'inde "perde hiç aktif değil"; <100 ms tekrarların %75'i kaçıyor.
Durum: **AÇIK** (S-04, S-05, S-07).

**README12 — Katman 3.11**
Yapılan: Hızlı tekrarlar için yerel tepe çözümlemesi, noisy-OR, vadi eğitimi; tab F1 **0,769** (en iyi).
Sorun: Kabul kuralı sağlanmadı (GAPS nota −0,013); **vadi kanıtı precision'ı çökertiyor — fragman vadisi = gerçek tekrar vadisi** → eğitimsiz yol GAPS için tükendi.
Durum: **AÇIK** (S-05, S-06).

**README13 — Katman 3.12**
Yapılan: Kapasite mi veri mi ölçüldü (`overfit_gaps.py`: GAPS onset 0,40 → 0,95 → etiket tavanı yok, darboğaz kapasite) → derin CNN + perde-onset kafası → **`tabcrnn_deep2.pt`**: GAPS nota F1 0,720 → **0,779**, tab F1 **0,777**.
Sorun: Kapasite farkı 0,004 olduğu için "Adım 3b (SynthTab) gerekmedi" denildi; <100 ms tekrar değişmedi (~%76).
Durum: Katman kapandı; **SynthTab AÇIK** (kapasite ölçümü araştırma sorusunu cevaplamıyor — S-15).

**README14 — Katman 3.13**
Yapılan: GAPS partisyon TAB'ından tel + parmak etiketi üretildi (165 kayıt); "klasik edisyon önerisi" iki yönlü dizi modeli (`edition_seq.pt`) kabul edildi — gaps_test tel **0,815**, parmak **0,636**.
Sorun: Partisyon TAB'ı ses kaybı olarak genellenmedi (`tabcrnn_gtab.pt` RET); GAPS'te ses tabanlı oracle tel **0,610 < naif 0,702**; **`gaps_test` bir kez karar kuralında kullanıldı**.
Durum: Edisyon modeli ÇÖZÜLDÜ; S-02 ve S-16 **AÇIK**.

**README15 — Katman 4 Faz 0**
Yapılan: Veri protokolü koda bağlandı (izin kapıları, karantina `395_w41wc`), teknik ölçütleri, 80 otomatik test, D1 referans tablosu, çıkarım parmak izi, seçim/early-stopping kuralları.
Sorun: GuitarSet'te bağımsız test yok (6 oyuncudan 5'i eğitimde); protokol kapıları yalnız yeni kodda zorunlu; parmak izi 2 kayıtla sınırlı.
Durum: Faz tamamlandı; **üç açık risk kendi kaydında duruyor** (S-11, S-17, S-18).

**README16 — Katman 3.13 / P1**
Yapılan: Teknik tanıma beklemeye alındı (Faz 1: **teknik etiketi yok** — GAPS 300 dosyada slide 0, glissando 0, harmonic 0); parmaklandırma incelendi, P2-P6 yol haritası çıkarıldı.
Sorun: Entegrasyon yok, `NoteEvent` parmak taşımıyor, fiziksel çalınabilirlik denetlenmiyor, testsizlik.
Durum: Testsizlik P2'de ÇÖZÜLDÜ; S-09, S-13, S-14 **AÇIK**.

**README17 — P2**
Yapılan: `tests/test_edition.py` (20 test) eklendi; `edition_seq.pt` doğrulandı; `gaps_val` metrikleri **birebir** yeniden üretildi (tel 0,766 / parmak 0,675 / ürün 0,382).
Sorun: `gaps_score.py` (MusicXML hizalama) ve `build_gaps_tab.py` hâlâ testsiz; uçtan uca zincir test dışı.
Durum: **AÇIK** (S-19).

**README18 — P3**
Yapılan: Çalınabilirlik ölçümü (yalnız sayım): model örtüşen akorların **%8,52'sinde** fiziksel çakışma üretiyor, naif %13,44, **editör %0,00**; model kendi kapsamında maks açıklık **22 yarım ses**.
Sorun: Akor içi aynı parmak yasağı yok; el açıklığı kısıtı yok; barre belirsizlik kuralı fazla temkinli; PAYDA-3 küçük (223 akor).
Durum: **AÇIK — P4'ün işi** (S-12, S-13, S-14).

**archive/README16 — Katman 4 Faz 1**
Yapılan: 230 kayıt / 386.296 nota tarandı; teknik etiket envanteri çıkarıldı.
Sorun: **Teknik etiketi yok** → teknik ancak sözde etiketle yapılabilir, klasik gitarda sayısal doğruluk iddia edilemez.
Durum: ERTELENDİ (bilinçli, gerekçeli).

**archive/README17 — Katman 4 Faz 2A**
Yapılan: Sürekli F0 altyapısı (`contour.py`), kontur temizliği, 29 test, 18 MB önbellek.
Sorun: Arşivde bilinen bir sayı hatası var (satır 189); tarihsel kayıt olduğu için düzeltilmedi.
Durum: ERTELENDİ (çıktılar korundu).

**[B] README15 — Katman 5 (Adım 0a-1b)**
Yapılan: Gerçek ritim etiketleri üretildi (syncpoint uyumu 22,9 ms); eğitimsiz taban yetmedi; bağlamlı kuantalayıcı `rhythm_q.pt` **kabul** (oracle IOI **0,933**); öğrenilen vuruş `beat_net.pt` eğitildi.
Sorun: Vuruş takibi eşiği **sağlanmadı** (F 0,631 < 0,75; uçtan uca IOI 0,695 < 0,833) → kullanıcı kararıyla **yarı otomatik mod** (kullanıcı BPM + ölçü imi girer).
Durum: Kuantalama ÇÖZÜLDÜ; vuruş takibi **AÇIK** (S-20).

**[B] README16 — Katman 5 Adım 2**
Yapılan: Ölçü modeli + MusicXML yazıcısı (`render/score.py`, `render/musicxml.py`) + uçtan uca `pipeline.py` + CLI; gidiş-dönüş **birebir 1,0000** (202 partisyon, 159.796 nota), dosya geçerliliği %100.
Sorun: Ölçü içi konum **0,611** (bulunan notaların ~%61'i ölçüde doğru yerde); ölçü başı F 0,582; MuseScore görsel kontrolü yapılmadı; AlphaTab yok.
Durum: Yazıcı ÇÖZÜLDÜ; S-20, S-21 **AÇIK**.

---

# 2. ÖNEMLİ KODLAR

## Model, inference, eğitim

| Dosya | Ne işe yarıyor? | Sorun var mı? |
|---|---|---|
| `gtab/models/nets.py` | Tüm mimariler: HarmonicStack, CNN gövdesi, ResBlock, PitchCNN, TabCNN, TabCRNN, TabCRNNOnset. | Yok. |
| `gtab/models/inference.py` | Checkpoint'ten model türünü tanır, uzun kayıtları parça parça işler, dört kafayı birlikte döndürür. | Yok (15 modül bunu kullanıyor). |
| `gtab/models/losses.py` | Tel olasılıklarından perde/perde-onset olasılığı (noisy-OR) ve offset hedefleri üretir. | Yok. |
| `gtab/models/edition_net.py` | Sembolik parmaklandırma modeli (BiLSTM); nota dizisinden tel + perde + parmak tahmin eder, ses görmez. | **Var:** satır 114 parmakları akor düzeyinde değil nota başına argmax ile seçiyor → fiziksel çakışma (S-12). |
| `scripts/train/train_onset.py` | Canlı tabanı üreten eğitim script'i (onset/offset/perde-onset kafaları, harmonik istifleme, derin CNN, hedefe özel ağırlıklar). | **Var:** protokol kapılarını (`assert_may_train`, `EpochSelector`) kullanmıyor — bilinçli, davranışı kırmamak için. |
| `scripts/train/train_edition_seq.py` | `edition_seq.pt`'ı üretir; yalnız `gaps_fit`, 30 epoch, seçim `gaps_val`'de. | Aynı: protokol kapıları kullanılmıyor. |
| `scripts/train/overfit_gaps.py` | Tanılama: 8 kayıtta dropout kapalı aşırı öğrenme → "etiket tavanı mı kapasite mi" sorusunu ayırır. | Yok (3.12 kararının kanıtı). |

## Çözümleme (decoder)

| Dosya | Ne işe yarıyor? | Sorun var mı? |
|---|---|---|
| `gtab/decoding/viterbi.py` (441 satır) | Çözümlemenin çekirdeği: perde matrisi, yerel tepe seçimi, enerji yükselişi ve `segment()` ile onset tabanlı nota bulma. | **Var:** çürütülmüş el yapımı Viterbi kodu (satır 84-203) hâlâ yolda; dosya canlı fonksiyonlarla karışık olduğu için bölünemiyor. |
| `gtab/decoding/decode.py` | Kareleri notaya çevirir, zamansal düzeltme yapar ve ASCII TAB render eder. | **Var:** `render_ascii_tab` parmak göstermiyor (S-10). |
| `gtab/decoding/transitions.py` | Öğrenilen tel geçiş modelini yükler, kafes kurar ve Viterbi ile nota başına tel/perde atar. | **Var:** katkısı 0 ölçüldü (`w_transition=0` seçiliyor) ama her çağrıda çalışıyor — gereksiz hesap. |
| `gtab/decoding/edition.py` | Notaları akorlara gruplar (onset ≤ 30 ms) ve reddedilen log-lineer edisyon modelini barındırır. | **Var:** gruplama yalnız onset'e bakıyor, zaman örtüşmesini gözetmiyor (S-08). |
| `gtab/decoding/playability.py` | Akor şekillerinin fiziksel çalınabilirliğini ölçer (çakışma, açıklık, ters sıra, barre, pozisyon). | Yok — yalnız sayıyor, kısıt uygulamıyor (tasarım kararı). |
| `gtab/decoding/rhythm.py` `[B]` | Eğitimsiz ritim tabanı: zarflar, vuruş takibi (dp/plp), ölçü başı kuralı, kural tabanlı kuantalayıcı. | Yok (1q'nun karşılaştırma tabanı). |
| `gtab/decoding/beat_dbn.py` `[B]` | numpy bar-pointer DBN; aktivasyonlardan vuruş ve ölçü başı çözer. | Yok. |

## Veri ve etiket

| Dosya | Ne işe yarıyor? | Sorun var mı? |
|---|---|---|
| `gtab/data/features.py` | Ses dosyasını 22050 Hz mono'ya indirir ve log-CQT'ye (192 bin) çevirir. | Yok. |
| `gtab/data/tab_labels.py` | Tel/perde hedeflerini, onset maskelerini, kısa nota ve vadi/fragman ağırlıklarını üretir. | Küçük: veri katmanı `decoding/viterbi`'yi import ediyor (katman ihlali, döngü yok). |
| `gtab/data/torch_dataset.py` | Dört PyTorch Dataset (pencere, sekans, GAPS perde sekansı) ve sınıf ağırlıkları. | **Var:** `class_weights` fret sınıfları üzerinden hesaplanıyor; **tel ekseninde denge yok** (S-01'in olası nedeni). |
| `gtab/data/gaps_score.py` (255 satır) | GAPS MusicXML partisyonunu açar, tekrarları çözer ve syncpoint + kapo ile sese hizalar. | **Var:** tüm parmak ve ritim etiketlerinin kaynağı, **birim testi yok** (S-19). |
| `gtab/data/gaps.py` | GAPS metadata/MIDI okur ve icracı-ayrık bölme üretir. | **Var:** `gaps_files()` karantinayı düşürmüyor; onu yapan `protocol.gaps_training_files()` (S-18). |
| `gtab/data/synthtab.py` | SynthTab zip'lerini açmadan okur ve tel notalarını tab hedefine çevirir. | **Var:** ham veri silinmiş, bugün çalışmaz (S-15). |
| `gtab/data/rhythm_labels.py` `[B]` | GAPS partisyonundan ve GuitarSet JAMS'ten gerçek vuruş/ölçü başı etiketleri üretir. | Yok. |
| `gtab/core/note_event.py` | Tüm katmanların paylaştığı nota şeması (`NoteEvent`, `Transcription`). | **Var:** **parmak ve ritim alanı yok** → her tüketici yan kanal kullanıyor (S-09). |

## Ölçüm ve protokol

| Dosya | Ne işe yarıyor? | Sorun var mı? |
|---|---|---|
| `gtab/evaluation/pitch_eval.py` (410) | Değerlendirme çekirdeği: alan içi kalibrasyon, çözümleme kuralı araması, nota/kare F1. | **Var:** tam kalibrasyon ~1 saat; `--load-cal` zorunlu pratik. |
| `gtab/evaluation/protocol.py` | Split rollerini ve izin kapılarını tanımlar; sızıntı kontrolleri ve karantinayı uygular. | Yok (17/17 denetim geçiyor). |
| `gtab/evaluation/selection.py` | Epoch seçimi, early stopping, epoch 0'ın da yarışması, aşırı öğrenme uyarısı. | Yok. |
| `gtab/evaluation/baselines.py` | D1 referans tablosunu okur ve farklı metrik gruplarının karşılaştırılmasını engeller. | Yok. |
| `gtab/evaluation/fingerprint.py` | Çıkarım parmak izi üretir/karşılaştırır (metrikten hızlı regresyon kontrolü). | **Var:** yalnız 2 GuitarSet kaydı; GAPS yolundaki regresyonu yakalamaz. |
| `scripts/eval/diagnose_strings.py` | Tel karışıklık matrisi ve kayıp ayrıştırması (S-01'in tek ölçüm aracı). | **Var:** katman-5'te `archive/`'a taşınmış — birleştirmeden önce kullanılmalı. |
| `scripts/eval/diagnose_pitch.py` | Kaçan/hayalet notaları nedenine göre kırar (bölüm A-F). | Yok. |
| `scripts/eval/eval_pitch.py` / `eval_hmm.py` | Ana perde ölçümü / ana GuitarSet tab F1 ölçümü. | Yok. |
| `scripts/eval/eval_edition.py` | Parmaklandırma ölçümü (tel, parmak, ürün, uçtan uca). | **Var:** `main()` `gaps_test` okuduğu için bütün olarak test dışı. |
| `scripts/eval/eval_playability.py` | Üç yöntem × üç payda çalınabilirlik raporu. | Yok. |

## TAB / MusicXML çıktısı ve CLI

| Dosya | Ne işe yarıyor? | Sorun var mı? |
|---|---|---|
| `gtab/render/score.py` `[B]` | Kuantalanmış konumları ölçülere, nota değerlerine, bağlara ve üçlemelere çevirir. | Küçük: `<tuplet>` parantezi yazılmıyor; garip süreler greedy ayrılıyor. |
| `gtab/render/musicxml.py` `[B]` | score-partwise MusicXML 4.0 yazar (P1 nota portresi + P2 TAB portresi, parmak, metronom, akort). | Yok (gidiş-dönüş 1,0000). |
| `gtab/pipeline.py` `[B]` | Uçtan uca zincir: CQT → nota → tel/perde → vuruş → kuantalama → ölçü. | **Var:** `--domain` CLI'ya açılmamış; `TranscribeError` benzeri açık hata yönetimi yok. |
| `scripts/app/transcribe.py` `[B]` | Komut satırı: ses + BPM + ölçü imi → `.musicxml`. | Aynı. |
| `scripts/transcribe.py` | Komut satırı: ses → nota tablosu + ASCII TAB (+ `--fingering` ile parmak, konum sabit). | **Var:** takipsiz (yedeksiz), testi yok, uçtan uca hiç çalıştırılmamış; `pipeline.py` ile işlev tekrarı (S-09'un entegrasyon tarafı). |

---

# 3. AÇIK SORUNLAR (TEK LİSTE)

Sonraki README'lerde çözülenler bu listeye alınmadı (ör. el yapımı Viterbi, CQT çoğaltma, P2 testsizliği).

| No | Kaynak README | Sorun | Olası neden | İlk önerilen çözüm |
|---|---|---|---|---|
| **S-01** | README10 (:30) | **A teli ~0,60**; `CLAUDE.md:117`'deki "0,66" kaynaksız; güncel tabanda tel başına doğruluk **hiç ölçülmedi** | Komşu tel karışıklığı; hatalar kararsızlık kaynaklı (marj 0,09) → model inharmonisiteyi yeterince görmüyor. Ek şüphe: eğitim verisinde tel dengesi ölçülmemiş | **Önce ölç:** `diagnose_strings --ckpt tabcrnn_deep2.pt --splits val val_comp`. Sonuca göre: tel-dengeli örnekleme → hex öğretmen damıtması → 36 bin/oktav |
| **S-02** | README14 | GAPS'te ses tabanlı oracle tel **0,610 < naif 0,702** (GuitarSet'te 0,878) | Alan farkı (naylon tını) + GAPS'teki "tel" editörün TAB konumu, icracının çaldığı tel olmak zorunda değil | Kısa vadede yok; ürün edisyon modunu kullanıyor. S-01 ile birlikte ele alınmalı |
| **S-03** | README3 | tab F1 (0,777) ile nota F1 (0,924) arasındaki fark = tel atama kalitesi | Uçtan uca F1 ≈ nota bulma × tel seçimi çarpımı gibi davranıyor | S-01/S-02 çözülürse düzelir; ayrı iş değil |
| **S-04** | README11 | Kaçanların **%36-45'inde "perde hiç aktif değil"** (modelin olasılığı nota boyunca eşiği hiç geçmiyor) | Kapasite (`overfit_gaps` kanıtı: GAPS onset 0,40 → 0,95, etiket tavanı yok). **Not: bu "eğitimde o perde yok" demek DEĞİL** | **Önce ölç** (son ölçüm deep2 öncesi): `diagnose_pitch --ckpt tabcrnn_deep2.pt --load-cal … --splits gaps_val val_comp`. Ayrıca eğitim kapsamı histogramı çıkarılmalı |
| **S-05** | README11, README12 | GAPS'te **<100 ms aynı perde tekrarlarının ~%55-76'sı kaçıyor** (val_comp'ta %72,5 → %22,5 düzeldi) | 60-100 ms IOI = 3-4 kare; gömülü tekrar spektrumda küçük sıçrama bırakıyor; vadi fragman vadisinden ayırt edilemiyor | Eğitimsiz yol tükendi. Denenmemiş aday: **"tekrar sayısı" kafası** (bir aralıkta kaç vuruş var) — "nerede vuruş"tan daha az zaman çözünürlüğü ister |
| **S-06** | README12 | **Akor fragmanları** — tek nota ikiye bölünüyor, akorun bir kısmı tespit ediliyor | S-05 ile aynı kök: "fragman vadisi = gerçek tekrar vadisi"; birini düzeltmek diğerini bozuyor | Düşük maliyetli ölçüm: fragman oranını **polifoni seviyesine göre** (1/2/3/4+) kır. Değişiyorsa çözümleme kuralı polifoniye koşullandırılabilir (eğitimsiz) |
| **S-07** | README11 | Akorlu kayıtlarda perde bulunamayan hücre %23,1 (solo %10,2); val_comp çıkış ölçütü sağlanmadı | Polifonide darboğaz **tel değil perde**; harmonik örtüşme | S-04/S-06 ile birlikte; `eval_pitch` polifoni kırılımı izlenmeli |
| **S-08** | README18 (K7) | 30 ms akor gruplaması **zamanda örtüşmeyen** notaları aynı akora koyuyor | `order_notes` yalnız onset yakınlığına bakıyor | Ölçüm tarafı P3'te düzeltildi; **kısıt tarafı P4'te** düzeltilmeli (kısıt yalnız örtüşen notalara) |
| **S-09** | README16 (§6.2) | `NoteEvent` **parmak ve ritim alanı taşımıyor**; iki ayrı WAV→TAB giriş noktası var (`scripts/transcribe.py` vs `pipeline.py`) | Parmaklandırma ve ritim şemadan sonra eklendi; iki dal paralel ilerledi | `finger` / `hand_position` / `beat` alanları eklenir (varsayılan `None` → davranış birebir korunur); `pipeline.py` tek omurga seçilir |
| **S-10** | README16 | `render_ascii_tab` parmak göstermiyor | Şema parmak taşımıyor | S-09 sonrası kolay |
| **S-11** | README15 | **GuitarSet'te bağımsız test yok**; 0,777 ve 0,924 **validation**'dır, kök README bunu söylemiyor | 6 oyuncudan 5'i eğitimde, oyuncu 05 hem seçim hem rapor setinde | Kök `README.md` sonuç tablosuna "split cinsi" kolonu (tek satırlık değişiklik, en büyük yanlış okuma riskini kapatır) |
| **S-12** | README18 | Model örtüşen akorların **%8,52'sinde fiziksel olarak imkânsız** parmaklandırma üretiyor (editör %0,00) | `edition_net.py:114` parmakları nota başına argmax ile seçiyor; `resolve_chords`'un parmak karşılığı yok | **P4:** `resolve_chords` mantığının parmağa taşınması. Hedef: çakışma 22 → 0, `gaps_val` parmak 0,675 düşmesin |
| **S-13** | README18 | El açıklığı kısıtı yok → model kendi kapsamında **maks 22 yarım ses** (editör maks 3) | Tel teklik kısıtı + açıklık kısıtının yokluğu birleşimi (model E4'ü 0. telin 24. perdesine atıyor) | **S-12 ile birlikte:** yumuşak açıklık cezası; üst sınır referansı editör dağılımı (≥4 oranı %0) |
| **S-14** | README18 | Barre belirsizlik kuralı fazla temkinli (editörün 15 barre adayının 12'si "belirsiz"); el pozisyonu çıktı olarak üretilmiyor | Kural kısmi barre'yi geçersiz sayıyor; `hand_position` yalnız ölçümde | Kural gevşetilir; `hand_position` S-09 ile `NoteEvent`'e taşınır |
| **S-15** | README9 (3.8d), README13 | **SynthTab nylon ön-eğitimi hiç yapılmadı** — sunumun ana araştırma sorusu cevapsız; ham veri de diskte yok | README13 kapasite farkını 0,004 ölçüp "gerekmedi" dedi; ama **kapasite sorusu ≠ araştırma sorusu** | Önce S-04'ün kapsam ölçümü (ucuz kapı). Hipotez desteklenirse: veri yeniden indirilir, 4 satırlık tablo (kabul kuralı README9:70-71'de hazır) |
| **S-16** | README14 / README15 | `gaps_test` **bir kez karar kuralında kullanıldı** (Adım 2 kabul kuralı "gaps_test tel ≥ naif + 0,05") | Protokol kapıları o sırada yoktu | Düzeltme yok (geri alınamaz); tezde dipnot. Tekrarı Faz 0'da koda bağlandı |
| **S-17** | README15 | Protokol kapıları **yalnız yeni kodda** zorunlu; `train_onset`, `eval_pitch`, `eval_hmm`, `train_edition_seq` eski yolu kullanıyor | Bilinçli: davranışı kırmamak için dokunulmadı | Dokunmamak; her **yeni** eğitim/ölçüm yolu kapıları çağırmalı |
| **S-18** | README15 | `395_w41wc` karantinası eski script'lerde etkisiz → `deep2` o kaydı perde denetimi olarak görmüş olabilir | `gaps_files()` doğrudan çağrılıyor, `gaps_training_files()` değil | Geçmiş değiştirilmiyor; yeni eğitimler kaydı görmeyecek (düşük etki: kaydın TAB etiketi zaten yok) |
| **S-19** | README17 (§8) | `gaps_score.align()` (255 satır) — **tüm parmak ve ritim etiketlerinin kaynağı, birim testi yok**; `build_gaps_tab.py` ve `eval_edition.py` de testsiz | Testler yalnız son 4 fazda eklendi | Sentetik MusicXML (tekrar işareti + kapo + fingering) ile `align()` birim testi |
| **S-20** | [B] README15, README16 | Vuruş takibi eşiği sağlanmadı (F **0,631** < 0,75); ölçü içi konum **0,611**; ölçü başı F 0,582 | İfadeli klasik müzikte vuruş takibi literatürde de zor; kalan kayıp yerel vuruş konumu (gerçek vuruşla IOI 0,927) | Yarı otomatik mod kullanılıyor (kullanıcı BPM + ölçü). Yedek: "tap" modu (kullanıcı vuruşa dokunur) |
| **S-21** | [B] README16 | MuseScore 4 **görsel doğrulaması yapılmadı**; teknik işaretleri (bend/slide/legato) ve AlphaTab görüntüleyici yok | Katman 5 Adım 2 kapandı, görselleştirme sıradaydı; Katman 4 beklemede | Birleştirme + önbellek kurulumu sonrası `.musicxml` MuseScore'da açılıp 9 maddelik kontrol listesi uygulanır |
| **S-22** | [B] README15/16 vs README15-18 | **Katman-5 ve mevcut dal `README15.md`/`README16.md` adlarını farklı içerikle kullanıyor**; katman-5'te `tests/` ve protokol kapıları yok; `rhythm/` + `beat_feats/` önbellekleri bu makinede yok (808 MB) | İki dal 11 Ekim'de paralel ilerledi | Birleştirmeden önce katman-5 devlog'ları `README19/20`'ye adlandırılır; `edition.py` çakışması elle çözülür (2 test kırılır); önbellekler kurulur |
| **S-23** | — (kod incelemesi) | `tests/test_edition.py` kapısı `mir_eval`'i kapsamıyor → o paketin olmadığı ortamda test **atlanmıyor, hata veriyor** | Kapı `torch` + checkpoint kontrol ediyor, import zinciri `eval_edition → pitch_eval → mir_eval` | Kapıya `find_spec("mir_eval")` eklenir (tek satır) |
| **S-24** | — (kod incelemesi) | Kök `README.md` ve `CONTRIBUTING.md` **iki taban geride** (taban `tabcrnn_comp`/`rep_off` gösteriyor, gerçek `deep2`); `requirements.txt` "Python 3.10" diyor, ortam 3.11; kurulum satırı bozuk | README15 güncellemeyi "Katman 4 kapanışına" bıraktı, Katman 4 beklemeye alındı | Tek seferde güncelleme; kaynak `docs/baselines/d1_baseline.json` |
| **S-25** | — (kod incelemesi) | Tek seed; "±0,01 gürültü" varsayımı hiç ölçülmemiş (sınır kararların dayanağı) | İkinci seed koşusu hiç yapılmadı | Bir model için `--seed 2` koşusu; tolerans doğrulanır ya da düzeltilir |

**Toplam: 25 açık sorun.**

---

# 4. MODEL EĞİTİMİNE HAZIRLIK

## Altı soruya kısa cevap

**1. Hangi modelle devam edeceğiz?**
**`TabCRNNOnset` (derin, `--deep 2`)** — ses modeli. Parmaklandırma için **`EditionNet`**, ritim için (birleştirme sonrası) **`RhythmNet` + `BeatNet`**. Mimari değişikliği gerekmez: kapasite farkı 0,004 ölçüldü, Transformer denemesi gerekçesiz olur.

**2. Hangi checkpoint kullanılacak?**

| Amaç | Checkpoint | Durum |
|---|---|---|
| Ses modeli tabanı | `checkpoints/tabcrnn_deep2.pt` + `tabcrnn_deep2.cal.json` | **Canlı taban** — yeni eğitimlerin `init`'i, üzerine **yazılmaz** |
| Parmaklandırma | `checkpoints/edition_seq.pt` | Canlı; bu denetimde yüklenip `gaps_val` metrikleri birebir üretildi |
| Tel geçişi | `checkpoints/transitions.npz` | Canlı ama katkısı 0 |
| Ritim | `checkpoints/rhythm_q.pt`, `beat_net.pt` | **Yalnız katman-5 dalında** — birleştirme gerekiyor |

**3. Hangi datasetlerle train yapılacak?**

| Split | İçerik | Rol |
|---|---|---|
| `train` + `train_comp` | GuitarSet oyuncu 00-04 (150 + 150) | eğitim — tel/perde denetimi |
| `gaps_fit` | GAPS 202 kayıt (karantina sonrası), 149 icracı | eğitim — perde denetimi, klasik gitar tınısı |
| `gaps_tab` (142 train kaydı) | partisyon tel + parmak | yalnız **sembolik** eğitim (`edition_seq`); ses kaybı olarak **kullanılmaz** (README14'te RET) |

Önbellekler diskte hazır (825 MB). SynthTab verisi **yok**.

**4. Validation nasıl yapılacak?**

| Split | Kullanım |
|---|---|
| `val` (GuitarSet 05 solo, 30) | **ana tab F1 ölçümü**; eşik/epoch seçimi |
| `val_comp` (GuitarSet 05 akorlu, 30) | polifoni ölçümü |
| `gaps_val` (28 kayıt, 22 icracı) | **GAPS tarafı seçim + kalibrasyon** |
| `gaps_test` | **KAPALI** — yalnız faz kapanışında, bir kez, rapor için |

Seçim `EpochSelector(selection_split)` üzerinden; `gaps_test` verilirse `ProtocolError`. Kalibrasyon **alan içi**: `val` → GuitarSet ayarı, `gaps_val` → GAPS ayarı; iki alanın sayıları birbirinin yerine kullanılamaz.

**5. Eğitim ve değerlendirme komutları hazır mı?**
**Evet, mevcut dalda hazır.** Ritim komutları için birleştirme gerekiyor.

```bash
# regresyon kapisi (hepsi .venv ile; dogrulandi)
.venv/Scripts/python -m unittest discover -s tests -t .          # 170/170, ~9 sn
.venv/Scripts/python -m scripts.eval.check_protocol              # 17/17, ~3 sn
.venv/Scripts/python -m scripts.eval.check_inference_regression  # parmak izi, CPU, ~10 sn

# olcum (egitim yok)
python -m scripts.eval.diagnose_strings --ckpt tabcrnn_deep2.pt --splits val val_comp
python -m scripts.eval.diagnose_pitch --ckpt tabcrnn_deep2.pt \
    --load-cal checkpoints/tabcrnn_deep2.cal.json --splits gaps_val val_comp
python -m scripts.eval.eval_pitch --ckpt tabcrnn_deep2.pt --load-cal checkpoints/tabcrnn_deep2.cal.json
python -m scripts.eval.eval_hmm --model crnn --ckpt tabcrnn_deep2.pt
python -m scripts.eval.eval_edition
python -m scripts.eval.eval_playability

# egitim (YENI AD zorunlu; init = deep2)
python -m scripts.train.train_onset --ckpt <yeni_ad>.pt   # + hedefe ozel bayraklar
python -m scripts.train.train_edition_seq                 # ~2 dk
```

**6. Eğitime başlamadan önce eksik bir şey var mı?**
**Evet, dört kalem:**

| Eksik | Neden önemli | Maliyet |
|---|---|---|
| **S-01/S-04 ölçümleri yapılmadı** | Hangi eğitim hipotezinin hedefi vurduğu bilinmiyor; "önce ölç" kuralı | saatler, eğitim yok |
| **`scripts/transcribe.py` takipsiz, yedeksiz** | Bir checkout/clean hatası 221 satırı siler | dakikalar |
| **`diagnose_strings.py` katman-5'te arşive taşınıyor** | S-01 ölçümü birleştirmeden **önce** yapılmalı | sıralama kararı |
| **Katman-5 önbellekleri yok** (`rhythm/`, `beat_feats/` 808 MB) | Ritim eğitimi/ölçümü çalışmaz | `build_rhythm` ~3 dk + `build_beat_feats` 808 MB |

Ayrıca S-23 (tek satırlık test kapısı) ve S-24 (vitrin dokümanları) eğitimi engellemez ama birlikte yapılması ucuz.

## Eğitim → validation → sonuç karşılaştırması senaryosu

```
1. TABAN KAYDI
   d1_baseline.json'dan ilgili satir okunur (ya da olculur).
   Ornek hedef S-01 ise: val oracle tel 0.878, A teli = OLCULECEK.

2. KABUL OLCUTU ONCEDEN YAZILIR  (devlog/README<N>.md)
   "val oracle tel >= 0.878 + 0.02 VE A teli >= +0.05 VE
    GuitarSet tab F1 >= 0.777 - 0.01 VE GAPS nota F1 >= 0.779 - 0.01"

3. TEK KONTROLLU DEGISIKLIK
   train_onset --ckpt yeni_ad.pt   (init = deep2, varsayilan = eski davranis birebir)
   Egitim split'leri: train, train_comp, gaps_fit   |   Secim split'i: val / gaps_val
   EpochSelector: epoch 0 da yarisir; hicbiri gecemezse taban korunur.

4. KALIBRASYON
   Yeni checkpoint icin alan ici kalibrasyon (--save-cal).
   deep2'nin cal.json'i KULLANILMAZ (yoksa kalibrasyon avantaji karisir).

5. AYNI KOSULLARDA OLCUM
   eval_hmm (GuitarSet tab F1) + eval_pitch (GAPS/GuitarSet nota F1)
   + diagnose_strings (hedefe ozel metrik)
   Hepsi VALIDATION setinde. gaps_test KAPALI.

6. KARSILASTIRMA
   assert_comparable ile ayni metrik grubu icinde.
   Tolerans: ayni checkpoint + ayni kod 0.002 | yeniden egitim 0.01

7. REGRESYON
   unittest (170) + check_protocol (17) + check_inference_regression

8. KARAR
   Olcut saglandi  -> yeni taban; d1_baseline.json'a satir; CLAUDE.md guncellenir
   Saglanmadi      -> taban korunur; CLAUDE.md "denenmis, ise yaramayan"a eklenir

9. KAPANIS (yalniz faz sonunda, BIR KEZ)
   gaps_test okunur; report_label("gaps_test") ile "bagimsiz test" etiketiyle raporlanir.
```

**Değişmeyecek kurallar:** mevcut checkpoint'lerin üzerine yazılmaz (yeni ad) · `gaps_test`
geliştirmede kullanılmaz · `395_w41wc` karantinada · tek seed farkları ±0,01 gürültü sayılır ·
raporlanan her sayının yanında split cinsi yazılır.
