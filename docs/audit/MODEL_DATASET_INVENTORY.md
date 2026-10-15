# Model, Checkpoint ve Dataset Envanteri

> Denetim: 14 Ekim 2026 · `Armagan-katman-4` @ `546c7bd` (+ `origin/katman-5-ritim` @ `3f1f47d`)
> Diskten doğrulanan sayılar "ölçüldü" ile işaretli; diğerleri devlog kaydı + satır referansı.

---

## 1. Dataset envanteri

### 1.1 GuitarSet

| Alan | Değer |
|---|---|
| **Kaynak** | https://guitarset.weebly.com/ · `mirdata` ile indirilir (`scripts/data/get_guitarset.py`) |
| **Lisans / erişim** | genel; ham ses repoda yok, `GUITARSET_DATA_HOME` ortam değişkeniyle dış dizine iner |
| **Kullanım amacı** | **tel/perde denetimi** — hexafonik pickup sayesinde "hangi tel çalındı" etiketli tek veri kaynağı |
| **Ses formatı** | mikrofon + hex pickup; yükleyici 22050 Hz mono'ya indirir |
| **Etiket türleri** | nota onset/offset + perde (JAMS), tel başına kanal → tel/perde; `pitch_contour` (hex, F0); `beat_position` + `tempo` (metronomla çalınmış) |
| **Teknik etiketi** | **YOK** (JAMS'te teknik namespace'i yok — Faz 1 ölçümü) |
| **Ölçek** | 360 kayıt × ~30 sn, 6 icracı (00-05) |
| **Train / val ayrımı** | oyuncu **00-04 eğitim** (train 150 solo + train_comp 150 akorlu), oyuncu **05 validation** (val 30 solo + val_comp 30 akorlu) — ölçüldü: dizin dosya sayıları 150/150/30/30 |
| **Bağımsız test** | **YOK** — bkz. `ISSUE_REGISTER A-04`. Oyuncu 05 hem seçim hem kalibrasyon hem raporlama setinde |
| **Keşif bölmeleri** | `val_half_a` (14 kayıt) / `val_half_b` (16), parça-ayrık — `may_select = may_report = False` (ölçüldü: `check_protocol`) |
| **Kapsanan teller/perdeler** | 6 tel, standart akort; fret 0-24 sınıf uzayı (`n_tab_classes` = 26 sınıf/tel: fret 0-24 + sessizlik) |
| **Dengesizlik** | sessizlik sınıfı baskın → `class_weights(scheme="inv"/"sqrt", cap)` ile dengeleniyor (`torch_dataset.py:246`) |
| **Sızıntı riski** | kayıt ve icracı kesişimi **0** (ölçüldü: `check_protocol` "GuitarSet kayit/icraci kesisimi 0") |
| **İlgili modeller** | `pitchcnn`, `tabcnn`, `tabcrnn*` zincirinin tamamı, `beat_net` (katman-5) |
| **Önbellek** | `data/cache/{train,val,train_comp,val_comp}` + `contour_guitarset` (361 dosya, 18 MB) — ölçüldü |

### 1.2 GAPS v1.1

| Alan | Değer |
|---|---|
| **Kaynak** | https://huggingface.co/datasets/xavriley/GAPS (MIT) · `scripts/data/get_gaps.py` |
| **Kullanım amacı** | **gerçek klasik gitar (naylon tel) domain adaptation + ana bağımsız test** |
| **Ses formatı** | 48 kHz stereo → 22050 Hz mono (doğrulandı: `001_mvswc` 227,8 sn MIDI / 228,3 sn ses) |
| **Etiket türleri** | hizalı MIDI (perde + zaman), MusicXML partisyon (**TAB konumu + sol el `<fingering>`**), syncpoint'ler (ölçü düzeyi), `.match` dosyaları, metadata CSV (icracı, YouTube kanalı, split) |
| **Teknik etiketi** | pratikte **YOK** — 300 dosya tarandı: slide 0, glissando 0, harmonic 0, trill 0, slur 4 dosya, tremolo 1 dosya; yalnız staccato 48 / accent 93 / dynamics 226 |
| **Ölçek** | ~14 saat, 200+ icracı; resmi bölünme train 270 / test 30 / atanmamış 101 |
| **Önbellekte** | `gaps_train` **231** kayıt, `gaps_test` **30** kayıt (ölçüldü) |
| **Neden 270 → 231** | `build_gaps --disjoint` test icracılarının train kayıtlarını hiç önbelleğe almıyor: **39 kayıt dışlandı** (ölçüldü: "yasakli 39 GAPS kaydi onbellekte 0") |
| **Train/val ayrımı** | `gaps_fit` 203 kayıt / 149 icracı (karantina sonrası **202**), `gaps_val` 28 kayıt / 22 icracı; ortak icracı **0** (ölçüldü) |
| **Test** | `gaps_test` 30 kayıt / 27 icracı — **bağımsız test (icracı-ayrık)**, yalnız faz kapanışında bir kez okunur |
| **TAB etiketleri** | `data/cache/gaps_tab/` **165 kayıt** = 142 gaps_train + 23 gaps_test (ölçüldü); 166. dosya `summary.csv` |
| **Nota sayısı (TAB etiketli)** | 159.647 nota, bunların **38.828'i (%24,3) parmak numaralı** (`README16.md §1`) |
| **Etiket kalitesi** | perde roll uyumu **%98,93** · "parmak 0 ⇔ boş tel" tutarlılığı **%99,4** · eşleşme medyanı **16 ms** · ölçü başı ↔ syncpoint uyumu medyan **22,9 ms** |
| **KRİTİK UYARI** | GAPS'teki "tel", **editörün partisyonda yazdığı TAB konumu**dur; icracının fiilen çaldığı tel olmak zorunda değil. Bu, tel doğruluğuna ölçülemez bir tavan koyar (`A-02`) |
| **Karantina** | `QUARANTINE_GAPS_IDS = ("395_w41wc",)` — tüm kimlik alanları boş (yalnız `yt_id = 2H53_rwralE`). Önbellekten **silinmedi**, yeni eğitim/seçim listelerinden düşüyor (ölçüldü: 231 → 230) |
| **Sızıntı kontrolleri (hepsi ölçüldü)** | (a) ad yazım varyantı: normalize ad ile 39 = 39, yeni yakalanan yok; (b) kanal düzeyi: 14 ortak kanal → 11 kişisel kanalın **tüm** train kayıtları dışlanmış, 3 çok-icracılı platform kanalı beklenen örtüşme; (c) adsız 3 kayıttan 2'si kanal adıyla temiz, 1'i karantinada |
| **Dengesizlik** | parmak etiketi notaların yalnız %24,3'ünde; editörler "belirgin olmayan" yerleri işaretliyor → **referans mutlak doğru değil**, tek doğru parmaklandırma yok |
| **İlgili modeller** | `tabcrnn_gaps` → `deep2` (perde denetimi), `edition_seq` (tel/perde/parmak), `rhythm_q` + `beat_net` (katman-5) |
| **Disk** | `data/raw/gaps_hf` **11 GB** (ölçüldü); alt dizinler: audio, midi, musicxml, syncpoints, match, metadata CSV |

### 1.3 SynthTab

| Alan | Değer |
|---|---|
| **Kaynak** | https://github.com/yongyizang/SynthTab · UR Box, **elle indirme** |
| **Kullanım amacı (planlanan)** | büyük ölçekli sentetik naylon ön-eğitimi — sunum Phase 4, slayt 10 ana deneyi |
| **Ölçek (kaynakta)** | ~6.700 saat; naylon alt kümesi 5.501 track / ~1.510 saat (slayt 7). Tamamı ~2 TB |
| **Projede planlanan kapsam** | yalnız naylon (`luthier_*`), ≤30 saat CQT float16 (`CLAUDE.md:28-29`, disk sınırı) |
| **Etiket türleri** | tel bazlı MIDI + JAMS; okuyucu zip'i açmadan okuyor (`gtab/data/synthtab.py`, doğrulanmış dosya yapısı belgeli) |
| **DURUM** | **VERİ DİSKTE YOK.** README9:22 "Dev Set indirildi ve doğrulandı (`data/raw/synthtab/SynthTab_Dev.zip`)" diyor; `data/raw/` bugün yalnız `gaps_hf` içeriyor (ölçüldü). Kod duruyor, veri yok |
| **Bilinen ön bulgu** | *"sentetik naylon (SynthTab Dev, 4 parça) model için kolay (tab F1 0,73), GAPS gerçek kayıt zor"* (`CLAUDE.md:125-126`) |
| **İlgili model** | yok — `tabcrnn_syn*.pt` hiç üretilmedi; README9:51-52 tablo satırları boş |
| **Etki** | Sunumun **ana araştırma sorusu cevapsız** (`ISSUE_REGISTER G-03`) |

### 1.4 Diğer kaynaklar (sunum slayt 8, hiç entegre edilmedi)

GOAT, IDMT-SMT-GUITAR, Guitar-TECHS, AG-PT-set, DadaGP ve "kendi test setimiz" (farklı klasik
gitar / icracı / telefon-mikrofon-oda) sunumda ek veri havuzu olarak listeli; **hiçbiri indirilmedi
veya kullanılmadı**. "Kendi test setimiz" özellikle değerli olurdu: hem bağımsız test eksikliğini
(`A-04`) hem cihaz/oda robustness ölçümünü (slayt 12) kapatırdı.

### 1.5 Önbellek envanteri (diskten ölçüldü)

| Dizin | Dosya | İçerik |
|---|---|---|
| `data/cache/train` | 150 | GuitarSet solo, oyuncu 00-04 — cqt + tab/onset/perde hedefleri |
| `data/cache/train_comp` | 150 | GuitarSet akorlu, oyuncu 00-04 |
| `data/cache/val` | 30 | GuitarSet solo, oyuncu 05 — **ana GuitarSet doğrulaması** |
| `data/cache/val_comp` | 30 | GuitarSet akorlu, oyuncu 05 — polifoni ölçümü |
| `data/cache/gaps_train` | 231 | GAPS, cqt + **perde** roll'u (tel etiketi yok) |
| `data/cache/gaps_test` | 30 | GAPS resmi test |
| `data/cache/gaps_tab` | 166 | 165 kayıt TAB/parmak etiketi + `summary.csv` |
| `data/cache/contour_guitarset` | 361 | hex F0 konturu (Katman 4 Faz 2A, beklemede), 18 MB |
| `data/cache/gaps_meta.csv` | 1 | GAPS metadata önbelleği |
| **`data/cache/rhythm`** | **YOK** | katman-5 gerektiriyor — `build_rhythm` (~3 dk) |
| **`data/cache/beat_feats`** | **YOK** | katman-5 gerektiriyor — `build_beat_feats`, 808 MB |
| **Toplam** | — | **825 MB** (ölçüldü) |

---

## 2. Model envanteri

### 2.1 Akustik modeller (ses görür)

| Model sınıfı | Dosya | Girdi | Çıktı |
|---|---|---|---|
| `PitchCNN` | `nets.py:58` | CQT penceresi (context 9) | çok-etiketli perde logit'i |
| `TabCNN` | `nets.py:72` | CQT penceresi (context 9) | 6 tel × 26 sınıf, tel başına softmax |
| `TabCRNN` | `nets.py:106` | CQT sekansı | (L × 6 × 26); CNN + 2 katman BiLSTM(128) |
| `TabCRNNOnset` | `nets.py:156` | CQT sekansı | tab + **onset** + (offset) + (**perde-onset**) kafaları |
| `HarmonicStack` | `nets.py:22` | CQT | spektrumun kaydırılmış kopyaları kanal olarak (ön hesap/yeni önbellek gerektirmez) |
| `ResBlock` | `nets.py:90` | — | `--deep 2` artık bloklar; **sıfır-gamma** → birebir aynı başlangıç |

Ortak gövde: `Conv2d(1→32)→BN→ReLU→Conv2d(32→64)→BN→ReLU→MaxPool2d(2)→Dropout(0.25)`
(`nets.py:50 _backbone`). TabCRNN'de zaman ekseni korunur, yalnız frekans havuzlanır.

**Ses sabitleri (`gtab/config.py` — değiştirmek tüm modelleri geçersiz kılar):**
22050 Hz · hop 512 (~23,2 ms kare, ~43 kare/sn) · log-CQT 192 bin · 24 bin/oktav
(yarım ses başına 2 bin, mikroton geçişleri için) · C1'den (32,7 Hz) 8 oktav.

### 2.2 Sembolik modeller (ses görmez)

| Model | Dosya | Girdi | Çıktı |
|---|---|---|---|
| `EditionNet` | `models/edition_net.py:50` | nota dizisi `(başlangıç, bitiş, perde)` + akor bayrağı; türetilmiş: perde gömmesi, akor boyu/sıra, başlangıç aralığı, süre, perde farkı, geçerli teller | **tel** (0-5), **perde** (pitch − açık tel), **parmak** (0 = boş tel, 1-4) |
| `ChordPositionModel` / `FingerModel` | `decoding/edition.py:44,91` | aynı | 2a log-lineer — **reddedildi** |
| `TransitionModel` | `decoding/transitions.py:48` | nota geçişleri | tel geçiş log-olasılığı — **katkısız ölçüldü** |
| `RhythmNet` (1q) | katman-5 `models/rhythm_net.py` | nota başına: en yakın tick'in vuruş içi yeri (one-hot 24) + artık, yerel vuruş aralığı, komşu uzaklıklar, perde, süre, ±8 vuruşluk tick histogramı | delta ∈ [−12, 12] → konum (c+delta)/24 |
| `BeatNet` | katman-5 `models/beat_net.py` | deep2 gövdesinin **donmuş** temsili (256) | kare başına (yok / vuruş / ölçü başı) softmax |

**Sert kısıtlar (`EditionNet`, P2'de testlerle sabitlendi):** enstrüman aralığı
(`core/instrument.py:34 pitch_to_positions`), **akor içi tel tekliği** (`edition_net.py:79
resolve_chords`), boş tel ⇔ parmak 0. **Yok:** el açıklığı sınırı, parmak sırası tutarlılığı,
**akor içi aynı parmak yasağı** — parmaklar nota başına argmax ile seçiliyor (`edition_net.py:114`
→ `ISSUE_REGISTER D-01`).

### 2.3 Checkpoint envanteri (`checkpoints/`, 20 dosya — diskten ölçüldü)

| Dosya | Katman | Eklenen | Durum |
|---|---|---|---|
| `pitchcnn.pt` | 2 | yalnız perde | tarihsel |
| `tabcnn.pt` | 3 | tel + perde, kare bazlı CNN | tarihsel |
| `tabcrnn.pt` | 3.5 | + BiLSTM | tarihsel |
| `tabcrnn_base.pt` | 3.7 | taban | tarihsel |
| `tabcrnn_comp.pt` | 3.7 | + GuitarSet akorlu kayıtlar | tarihsel |
| `tabcrnn_aug.pt` | 3.7 | + CQT perde kaydırma | **REDDEDİLDİ** (−0,06) |
| `tabcrnn_gaps.pt` | 3.8 | + GAPS perde ince ayarı | tarihsel (A-01'in ölçüm modeli) |
| `tabcrnn_onset.pt` | 3.9 | + onset kafası | tarihsel |
| `tabcrnn_onset_h.pt` | 3.9 | + harmonik istifleme | eski taban |
| `tabcrnn_poly.pt` | 3.10 | + keskin onset, GS perde kaybı, kısa nota ağırlığı | eski taban |
| `tabcrnn_rep_off.pt` | 3.10 | + offset kafası, tekrar onset ağırlığı, GAPS perde ×2 | eski taban |
| `tabcrnn_rep_valley.pt` + `.cal.json` | 3.11 | + vadi eğitimi | eski taban |
| `tabcrnn_deep.pt` | 3.12 | derin CNN + perde-onset kafası (15 ep) | ara |
| `tabcrnn_deep_try.pt` | 3.12 | 2 epoch deneme | ara |
| **`tabcrnn_deep2.pt` + `.cal.json`** | 3.12 | devam eğitimi (`--lr 1e-4`) | **CANLI TABAN** |
| **`edition_seq.pt`** | 3.13 2b | EditionNet BiLSTM | **CANLI** (parmaklandırma) |
| `edition.npz` | 3.13 2a | log-lineer | **REDDEDİLDİ** |
| `transitions.npz` | 3.6b | öğrenilen tel geçişi | CANLI ama katkı 0 (`w_tr = 0`) |

**Katman-5'te gelen iki yeni checkpoint:** `beat_net.pt` (2,78 MB), `rhythm_q.pt` (2,70 MB).
Aynı commit 14 eski dosyayı `checkpoints/archive/` altına taşıyor (+ `tabcrnn_gtab.pt`);
`tabcrnn_deep2.pt`, `edition_seq.pt`, `transitions.npz` kökte kalıyor.

### 2.4 Canlı tabanın sayıları (`tabcrnn_deep2.pt` + `tabcrnn_deep2.cal.json`)

Kaynak: `docs/baselines/d1_baseline.json` (12 test bu değerleri sabitliyor) + `CLAUDE.md:30-34`.

| Metrik | Değer | Split | **Split cinsi** | Birim / koşul |
|---|---|---|---|---|
| `gs_solo_tab_f1` | **0,777** | `val` | validation | kare düzeyi (tel, perde) token |
| `gs_solo_note_f1` | **0,924** | `val` | validation | nota olayı; sesten uçtan uca |
| `gs_poly_note_f1` | **0,807** | `val_comp` | validation | nota olayı; uçtan uca |
| `gaps_note_f1` | **0,779** | `gaps_test` | **bağımsız test** | nota olayı; uçtan uca |
| GAPS kare F1 | 0,701 | `gaps_test` | bağımsız test | kare düzeyi perde |
| `gs_oracle_string_acc` | 0,878 | `val` | validation | **oracle nota**, yalnız tel |
| `gaps_audio_oracle_string_acc` | **0,610** | `gaps_val` | validation | oracle nota, ses modu |
| `gaps_naive_string_acc` | 0,745 | `gaps_test` | bağımsız test | naif kural ("en düşük perde") |
| `gaps_note_coverage` | 0,817 | `gaps_test` | bağımsız test | etiketli notaların bulunma oranı |

**Parmaklandırma (`edition_seq.pt`) — üç ayrı koşul, karıştırılmaz:**

| Metrik | Değer | Koşul |
|---|---|---|
| `gaps_edition_string_acc` | **0,815** | gaps_test, tel etiketli tüm notalar (naif 0,745) |
| `gaps_edition_finger_acc` | **0,636** | gaps_test, **konum ORACLE** (partisyonun tel/perdesi verilir) — taban 0,566 |
| `gaps_edition_product_acc` | 0,434 | gaps_test, konumu model seçer; doğru = tel + perde + parmak üçü birden (taban 0,376) |
| `gaps_e2e_string_acc` | 0,655 | **uçtan uca** (deep2 notaları), tüm etiketli notalarda; bulunanlarda 0,802 |
| `gaps_e2e_full_finger_acc` | 0,331 | uçtan uca, tam parmak; bulunanlarda 0,397 |

**Ritim (katman-5, gaps_test):**

| Metrik | Değer | Hedef | Durum |
|---|---|---|---|
| Kuantalayıcı oracle IOI (`rhythm_q.pt`) | **0,933** | ≥ 0,90 | **KABUL** (kural 0,821) |
| Gerçek vuruş + deep2 notası IOI | 0,927 | — | — |
| Vuruş F (yarı otomatik, BPM %10 hatalı) | 0,631 | ≥ 0,75 | sağlanmadı |
| Ölçü başı F (yarı otomatik) | 0,582 | ≥ 0,60 | sağlanmadı (sınırda) |
| Uçtan uca IOI (yarı otomatik + 1q) | 0,695 | ≥ 0,833 | sağlanmadı |
| MusicXML gidiş-dönüş | **1,0000** | ≥ 0,999 | **KABUL** (202 partisyon, 159.796 nota) |
| Ölçü içi konum (tek ölçü imli) | **0,611** | eşik yok | yeni taban ölçütü |

**Çözümleme ayarları (`tabcrnn_deep2.cal.json`, alan içi seçildi):** iki alanda da onset kaynağı
**perde-onset kafası** (`--onset-source pitch`). Alan değerleri kalibrasyon dosyasından
`load_selection()` ile okunur; **iki alanın sayıları birbirinin yerine kullanılamaz.**

### 2.5 Gelişim çizgisi (GuitarSet solo tab F1 — hepsi validation)

```
0,586 TabCNN → 0,634 CRNN → 0,667 +comp → 0,697 +GAPS → 0,739 +onset+harmonik
→ 0,743 kurallar → 0,748 poly → 0,760 rep_off → 0,769 rep_valley → 0,777 deep2
```

GAPS nota F1: `0,255 → 0,360 → 0,441 → 0,592 → 0,668 → 0,689 → 0,720 → 0,779`

---

## 3. Eğitim / validation / test protokolü

Kaynak: `gtab/evaluation/protocol.py` (266 satır) + 25 test; `check_protocol` 17 kontrol
(denetimde **17/17 geçti**).

| split | içerik | eğitim | seçim | rapor | bağımsız test | rapor etiketi |
|---|---|---|---|---|---|---|
| `train`, `train_comp` | GuitarSet 00-04 (150 + 150) | ✓ | ✗ | ✗ | ✗ | eğitim verisi |
| `val`, `val_comp` | GuitarSet 05 (30 + 30) | ✗ | ✓ | ✓ | **✗** | **validation (bağımsız test değil)** |
| `val_half_a/b` | oyuncu 05 parça-ayrık yarılar (14 / 16) | ✗ | ✗ | ✗ | ✗ | keşif / duyarlılık |
| `gaps_fit` | gaps_train'in eğitim kısmı (203 → karantina sonrası **202**) | ✓ | ✗ | ✗ | ✗ | eğitim verisi |
| `gaps_val` | gaps_train'in doğrulama kısmı (28 kayıt, 22 icracı) | ✗ | ✓ | ✓ | ✗ | validation |
| `gaps_test` | resmi GAPS test (30 kayıt, 27 icracı) | ✗ | **✗** | ✓ | **✓** | bağımsız test (icracı-ayrık) |
| `gaps_skipped` | test icracılarıyla örtüşen 39 train kaydı | ✗ | ✗ | ✗ | ✗ | kullanılmaz |

**Koda bağlı yasaklar.**
- `assert_may_select("gaps_test")` → `ProtocolError`; `EpochSelector.__init__` otomatik çağırıyor.
- `assert_may_train` eğitim listesini doğrular: `val`, `val_comp`, `gaps_val`, `gaps_test`, `gaps_skipped`, `val_half_*` eğitimde kullanılamaz.
- Tanımsız split adı da hata verir (yazım hatasıyla sessizce kural dışına çıkılamaz).
- Raporlanan her sayının yanında `report_label(split)` zorunlu.
- `gaps_training_files()` → `drop_quarantined()` ile karantina kaydı düşer.

**Değerlendirme kuralları (değiştirilmeyecek — `CLAUDE.md:55-58`).**
- GuitarSet ana val: oyuncu 05 **solo** (`data/cache/val`); tüm tab F1'ler bununla kıyaslanır. `val_comp` = aynı oyuncunun akorlu kayıtları (polifoni ölçümü).
- GAPS: `--disjoint` (icracı-ayrık); eşik/parametre seçimi `gaps_val`'de; **`gaps_test` yalnız raporlama**.
- Tek seed farkları ±0,01 gürültü sayılır (**bu varsayım ölçülmemiş** — `G-10`).

**Toleranslar (`d1_baseline.json`).** yeniden-koşu **0,002** (aynı checkpoint + aynı kod; GPU iç
oynaması) · yeniden-eğitim **0,01** (seed). `assert_comparable` farklı metrik gruplarının
karşılaştırılmasını engelliyor.

**Bilinen protokol açıkları.**
1. GuitarSet'te bağımsız test yok (`A-04`) — yapısal.
2. `gaps_test` bir kez karar kuralında kullanıldı (README14 Adım 2 → `G-07`) — geri alınamaz, belgelendi.
3. Protokol kapıları yalnız yeni kodda zorunlu (`E-08`).
4. `395_w41wc` karantinası eski script'lerde etkisiz; `deep2` o kaydı görmüş olabilir (`G-08`).

---

## 4. Eğitim script'leri ve hiperparametreler

| Script | Katman | Ne yapıyor | Not |
|---|---|---|---|
| `train_pitch.py` | 2 | PitchCNN | tarihsel |
| `train_tab.py` | 3 | TabCNN (`--sweep` sessizlik-marjı taraması) | tarihsel |
| `train_crnn.py` | 3.5-3.7 | TabCRNN (`--splits`, `--ckpt`) | tarihsel |
| `train_domain.py` | 3.8 | GuitarSet (tab) + GAPS (perde) karışık batch | tarihsel |
| **`train_onset.py`** | 3.9-3.12 | onset/offset/perde-onset kafaları, harmonik istifleme, derin CNN | **canlı tabanı üreten script** |
| `train_edition.py` | 3.13 2a | log-lineer | reddedildi |
| **`train_edition_seq.py`** | 3.13 2b | EditionNet | **canlı parmak modelini üreten script** |
| `overfit_gaps.py` | 3.12 Adım 0 | kapasite/etiket tavanı testi (8 kayıt, 200 ep, dropout kapalı) | tanılama |
| `train_rhythm_q.py` | K5 1q | RhythmNet | katman-5 |
| `train_beat.py` | K5 1b | BeatNet + DBN ayar taraması | katman-5 |

**Canlı tabanın eğitim künyesi (devlog'lardan derlendi — `G-06` bunun eksikliği hakkında).**
- `tabcrnn_deep2.pt`: init = `tabcrnn_deep.pt`; `--lr 1e-4` devam eğitimi; `tabcrnn_deep.pt` ise `tabcrnn_rep_valley.pt`'tan `--deep 2 --pitch-onset-weight 1 --new-lr-mult 3`, 15 epoch. Seçim: "deep2 kafa ≥ 0,519 ve GS tab ≥ 0,697" → kafa 0,524, GS tab 0,716 → deep2 (`CLAUDE.md:103-110`).
- `edition_seq.pt`: yalnız `gaps_fit` (127 kayıt, 125 bin nota); 256 notalık kesitler, Adam `lr=1e-3`, batch 16, 30 epoch, seed 1, ~2 dk. Seçim `gaps_val` (tel + parmak ortalaması) → **epoch 17**. Parmak kaybı yalnız `(tel etiketli) & (parmak ≥ 1) & (perde > 0)` notalarda (`README16.md §4`).
- `rhythm_q.pt`: yalnız `gaps_fit`; kayıtların yarısı her epoch bozulur (vuruş gürültüsü σ ~ U(0, 0,08) + kayma ±0,1 + yarım/çift tempo %15/%15; nota sapması 10 ms, düşürme ≤ %20, sahte nota ≤ %10). CE, Adam 1e-3 + kosinüs, 80 epoch; seçim `gaps_val` oracle IOI. λ = 1,0 checkpoint'e yazılı.
- `beat_net.pt`: `gaps_fit` + GuitarSet train + train_comp (GAPS'te etiketli aralık dışı maskeli); 35 sn kesitler; `--stretch 0.25` tempo artırması, 30 epoch; seçim `gaps_val`; sonunda DBN ayarları (trans 3/10/30 × obs 16/32/64/128) `gaps_val`'de taranıp checkpoint'e yazılıyor → seçilen trans 10 / obs 32.

**Sınıf dengesizliği araçları.** `GuitarSetSeq/PitchSeq.class_weights` (`inv` / `sqrt`, `cap`),
`train_onset --oversample`, `--repeat-weight`, `--short-weight`, `--valley-weight`, `--frag-weight`,
`--pitch-weight`. Teknik tarafında ana koruma macro-F1'in ölçüt olması.

---

## 5. Yeniden üretim sırası (bu makinede)

Bugün çalışan: değerlendirme (checkpoint'ler + önbellek var). Çalışmayan: katman-5 ritim zinciri
ve SynthTab yolu.

```bash
# 1) Hazır — doğrulama (denetimde koşuldu)
python -m unittest discover -s tests -t .          # 170/170, ~9 sn  [.venv ile!]
python -m scripts.eval.check_protocol              # 17/17, ~3 sn
python -m scripts.eval.check_inference_regression  # parmak izi, CPU, ~10 sn

# 2) Mevcut dalda ölçüm (GPU, saatler)
python -m scripts.eval.eval_pitch --ckpt tabcrnn_deep2.pt --load-cal checkpoints/tabcrnn_deep2.cal.json
python -m scripts.eval.eval_edition
python -m scripts.eval.eval_playability

# 3) Katman-5 birleştirildikten SONRA gereken önbellekler (bugün YOK)
python -m scripts.data.build_rhythm        # ~3 dk
python -m scripts.data.build_beat_feats    # 808 MB, disk kontrolü gerekli

# 4) SynthTab yolu — veri silinmiş, yeniden indirme gerekir (şimdilik YAPILMAYACAK)
```

**Ortam.** `.venv` (Python **3.11**; `requirements.txt` başlığı 3.10 diyor — `G-05`),
torch 2.3.1+cu121, numpy 1.26.4 (`numpy<2` zorunlu), librosa 0.11.0, mir_eval 0.8.2,
mirdata 1.0.0, pandas 2.3.3, scipy 1.15.3 — ölçüldü. **Testler `.venv` ile koşulmalı**: sistem
Python'ında `mir_eval` yok ve `test_edition.py` kapısı bunu yakalamıyor (`E-09`).
