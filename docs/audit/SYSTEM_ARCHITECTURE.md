# Sistem Mimarisi ve Veri İlişkileri

> Durum: 14 Ekim 2026 · `Armagan-katman-4` @ `546c7bd` · karşılaştırma `origin/katman-5-ritim` @ `3f1f47d`
>
> Diyagramlar koddan çıkarıldı; her kutu gerçek bir dosya/fonksiyona karşılık gelir.
> **Çizgi anlamları her diyagramda aynı:**
> - **Düz çizgi (`-->`)** = bugün çalışan, mevcut dalda (`Armagan-katman-4`) var olan bağlantı
> - **Kesikli çizgi (`-.->`)** = **yalnız `katman-5-ritim` dalında** var; mevcut dalda YOK
> - **Noktalı/etiketli "PLAN"** = henüz yazılmamış, planlanan bağlantı

---

## Diyagram A — Genel sistem mimarisi

```mermaid
flowchart TD
    subgraph GIRIS["GİRİŞ"]
        WAV["Ses dosyası<br/>.wav / .mp3 / .flac<br/>22050 Hz mono"]
    end

    subgraph ONISLEM["ÖN İŞLEM — gtab/data/features.py"]
        CQT["log-CQT<br/>192 bin, 24 bin/oktav<br/>hop 512, ~43 kare/sn<br/>compute_cqt()"]
        NORM["_normalize()<br/>torch_dataset.py:18"]
    end

    subgraph MODEL["AKUSTİK MODEL — gtab/models/"]
        HS["HarmonicStack<br/>nets.py:22<br/>spektrum kaydırmalı kopyalar"]
        BODY["CNN gövdesi + 2x ResBlock<br/>nets.py:50,90"]
        LSTM["BiLSTM 2 katman, hidden 128<br/>TabCRNN.encode()<br/>nets.py:138"]
        H1["tab kafası<br/>6 tel x 26 sınıf"]
        H2["onset kafası<br/>tel başına"]
        H3["offset kafası"]
        H4["perde-onset kafası<br/>SEÇİLEN onset kaynağı"]
    end

    subgraph COZUMLEME["ÇÖZÜMLEME — gtab/decoding/"]
        CAL["Kalibrasyon<br/>tabcrnn_deep2.cal.json<br/>alan: gaps_val veya val"]
        RISE["energy_rise()<br/>viterbi.py:398<br/>CQT enerji sıçraması"]
        SEG["segment()<br/>viterbi.py:420<br/>onset tabanlı nota bulma<br/>tepe, refrakter, yedek,<br/>yeniden vuruş, histerezis"]
        LAT["build_lattice + decode_lattice<br/>transitions.py:168,209<br/>tel ataması, kafes Viterbi"]
    end

    subgraph SEMA["ORTAK ŞEMA — gtab/core/note_event.py"]
        NE["NoteEvent<br/>onset, offset, pitch,<br/>string, fret, technique<br/>PARMAK ALANI YOK"]
        TR["Transcription"]
    end

    subgraph EDISYON["EDİSYON ÖNERİSİ — gtab/models/edition_net.py"]
        ORD["order_notes()<br/>edition.py:29<br/>akor = onset ≤ 30 ms"]
        EN["EditionNet BiLSTM<br/>tel + perde + parmak"]
        RC["resolve_chords()<br/>edition_net.py:79<br/>akor içi tel tekliği"]
    end

    subgraph OLCUM["ÇALINABİLİRLİK ÖLÇÜMÜ — yalnız ölçüm, kısıt uygulamaz"]
        PL["playability.py<br/>çakışma A, açıklık B,<br/>ters sıra C, pozisyon D, barre E"]
    end

    subgraph RITIM["RİTİM — yalnız katman-5-ritim"]
        BN["BeatNet<br/>models/beat_net.py<br/>donmuş deep2 temsili"]
        DBN["beat_dbn.py<br/>bar-pointer DBN<br/>kullanıcı BPM ± %35 + ölçü imi"]
        RQ["RhythmNet 1q<br/>models/rhythm_net.py<br/>1/24 vuruş ızgarası"]
        MEAS["build_measures()<br/>render/score.py"]
    end

    subgraph CIKTI["ÇIKTI"]
        ASCII["ASCII TAB<br/>decode.py:157<br/>render_ascii_tab()"]
        TBL["Nota tablosu<br/>transcribe.py:note_table()"]
        MXML["MusicXML 4.0<br/>render/musicxml.py<br/>P1 nota + P2 TAB portresi"]
        MUSE["MuseScore 4 / Guitar Pro<br/>görsel doğrulama"]
        ALPHA["AlphaTab web görüntüleyici"]
    end

    WAV --> CQT --> NORM --> HS --> BODY --> LSTM
    LSTM --> H1
    LSTM --> H2
    LSTM --> H3
    LSTM --> H4

    CQT --> RISE
    CAL --> SEG
    H1 --> SEG
    H4 --> SEG
    H2 --> SEG
    H3 --> SEG
    RISE --> SEG

    SEG --> LAT
    H1 --> LAT
    LAT --> NE
    NE --> TR

    TR --> ORD --> EN --> RC
    RC --> NE

    RC --> PL
    TR --> PL

    TR --> ASCII
    TR --> TBL

    NORM -.-> BN
    LSTM -.-> BN
    BN -.-> DBN
    DBN -.-> RQ
    TR -.-> RQ
    RQ -.-> MEAS
    MEAS -.-> MXML
    RC -.-> MXML
    MXML -.-> MUSE
    MXML -.-> ALPHA

    PL -. "PLAN: P4 akor içi parmak kısıtı" .-> RC
    NE -. "PLAN: P6 parmak + el pozisyonu alanı" .-> ASCII

    classDef k5 stroke-dasharray: 5 5
    class BN,DBN,RQ,MEAS,MXML,MUSE,ALPHA k5
```

**Okuma notları.**
1. **İki ürün giriş noktası var ve bu diyagramda ikisi birleşmiş görünüyor** — gerçekte ayrı: `scripts/transcribe.py` (mevcut dal, takipsiz) yolun `ASCII`/`TBL` kısmını, `gtab/pipeline.py` + `scripts/app/transcribe.py` (katman-5) `MXML` kısmını sürüyor. Bkz. ISSUE_REGISTER `E-01`.
2. **Onset kaynağı seçimle belirleniyor.** `tabcrnn_deep2.cal.json` iki alanda da **perde-onset kafasını** (`H4`) seçti (`--onset-source pitch`); tel onset kafası (`H2`) yedekte kalıyor.
3. **Kafes Viterbi'nin katkısı sıfır ölçüldü** (`w_transition = 0` seçiliyor) ama yol açık; `decode_lattice` hâlâ çağrılıyor. Bkz. `F-01`.
4. **Edisyon modu ile ses modu karıştırılmaz** (kullanıcı kararı). Ses modu `LAT`'tan tel/perde alır; edisyon modu `EN`'den alır. `scripts/transcribe.py --fingering` üçüncü bir kullanım: konum ses zincirinden gelir, EditionNet'e `fixed` olarak verilir ve **yalnız parmak** alınır.
5. `PL` (çalınabilirlik) **hiçbir kısıt uygulamıyor**, yalnız sayıyor (README18 K1). Kısıt uygulaması P4'ün işi.

---

## Diyagram B — Veri ve şema ilişkileri

> **SQL veritabanı yok.** Aşağıdaki şema dosya biçimleri ve dizi/alan ilişkilerini gösterir;
> "tablo" = `.npz` içindeki dizi kümesi veya dataclass.

```mermaid
flowchart LR
    subgraph HAM["HAM VERİ — data/raw/ (git'te değil)"]
        GSAUD["GuitarSet<br/>360 x ~30 sn<br/>mic + hex pickup<br/>GUITARSET_DATA_HOME<br/>(dizin dışında)"]
        GAPSW["gaps_hf/audio/*.wav<br/>48 kHz stereo, 11 GB"]
        GAPSM["gaps_hf/midi/*.mid<br/>hizalı MIDI"]
        GAPSX["gaps_hf/musicxml/*<br/>partisyon, TAB + fingering"]
        GAPSS["gaps_hf/syncpoints/*<br/>ölçü düzeyi hizalama"]
        GAPSC["gaps_metadata_with_splits.csv<br/>icracı, kanal, split"]
        STZIP["SynthTab zip<br/>DİSKTE YOK — silinmiş"]
    end

    subgraph JAMS["ETİKET OKUYUCULAR — gtab/data/"]
        RJ["guitarset.py<br/>track_to_transcription()"]
        RG["gaps.py<br/>midi_to_transcription()<br/>gaps_files() split"]
        RS["gaps_score.py<br/>parse_part, unroll,<br/>align() syncpoint + kapo + ±80 ms"]
        RST["synthtab.py<br/>zip'i açmadan okur<br/>(bugün çalışmaz)"]
    end

    subgraph ONB["ÖNBELLEK — data/cache/*.npz (git'te değil, 825 MB)"]
        C1["train / val<br/>150 / 30 kayıt<br/>GuitarSet solo<br/>cqt, tab, onset, pitch roll"]
        C2["train_comp / val_comp<br/>150 / 30<br/>GuitarSet akorlu"]
        C3["gaps_train / gaps_test<br/>231 / 30<br/>cqt + PERDE roll'u<br/>(tel etiketi YOK)"]
        C4["gaps_tab/*.npz + summary.csv<br/>165 kayıt<br/>string, fret, finger_note<br/>partisyondan"]
        C5["contour_guitarset<br/>361 dosya, 18 MB<br/>hex F0 konturu (beklemede)"]
        C6["rhythm/*.npz<br/>BU MAKİNEDE YOK<br/>beats, downbeats, note_beat"]
        C7["beat_feats/tabcrnn_deep2/<br/>BU MAKİNEDE YOK, 808 MB<br/>donmuş deep2 temsili"]
    end

    subgraph DS["PyTorch Dataset — gtab/data/torch_dataset.py"]
        D1["GuitarSetFrames<br/>pencere, context 9"]
        D2["GuitarSetTab<br/>pencere + tab hedefi"]
        D3["GuitarSetSeq<br/>sekans, chunk 200<br/>class_weights, augment"]
        D4["PitchSeq<br/>perde etiketli sekans<br/>GAPS için"]
    end

    subgraph OLAY["NOTA OLAYI ŞEMASI — gtab/core/note_event.py"]
        NEV["NoteEvent<br/>onset: float<br/>offset: float<br/>pitch: int<br/>string: Optional int<br/>fret: Optional int<br/>technique: Technique<br/>confidence: float"]
        TRC["Transcription<br/>notes: List NoteEvent<br/>sort(), duration"]
        MISS["EKSİK ALANLAR<br/>finger · hand_position<br/>beat / measure / duration_value"]
    end

    subgraph CIK["DIŞA AKTARIM"]
        O1["ASCII TAB .txt"]
        O2[".musicxml<br/>divisions = 24/vuruş"]
        O3["JSON — YOK<br/>(ara biçim yazılmadı)"]
    end

    subgraph DEG["DEĞERLENDİRME KAYITLARI"]
        E1["docs/baselines/d1_baseline.json<br/>metrik, değer, split,<br/>split cinsi, birim, grup"]
        E2["docs/baselines/inference_fingerprint.json<br/>kafa istatistikleri + tab histogramı"]
        E3["checkpoints/*.cal.json<br/>alan başına eşik + kurallar"]
    end

    GSAUD --> RJ --> C1
    GSAUD --> RJ --> C2
    GSAUD --> C5
    GAPSW --> RG --> C3
    GAPSM --> RG
    GAPSX --> RS --> C4
    GAPSS --> RS
    GAPSM --> RS
    GAPSC --> RG
    STZIP -. "veri yok" .-> RST

    C1 --> D1
    C1 --> D2
    C1 --> D3
    C2 --> D3
    C3 --> D4
    C4 --> D4

    C4 -.-> C6
    C1 -.-> C6
    C1 -.-> C7
    C3 -.-> C7

    D3 --> NEV
    D4 --> NEV
    C4 --> NEV
    NEV --> TRC
    NEV --- MISS

    TRC --> O1
    TRC -.-> O2
    C6 -.-> O2

    TRC --> E1
    TRC --> E2
    TRC --> E3
```

**Gerçekten kullanılan alanlar (doğrulandı).**

| Dosya / şema | Alanlar | Kaynak |
|---|---|---|
| `data/cache/{train,val,...}/*.npz` | `cqt` (T, 192) float16; tel/perde hedefleri `tab`; onset / perde roll'u | `torch_dataset.py`, `tab_labels.py` |
| `data/cache/gaps_{train,test}/*.npz` | `cqt` + **perde** roll'u; tel etiketi yok (GAPS'te tel yalnız `gaps_tab` üzerinden) | `build_gaps.py`, `PitchSeq` |
| `data/cache/gaps_tab/*.npz` | nota başına `string`, `fret`, `finger_note`, sıra, akor bayrağı | `edition.py:244 label_sequence()` |
| `data/cache/gaps_tab/summary.csv` | `id, split, use, consistent, shift, notes, labeled, known_frames, roll_agree, fingered, open_finger0, dt_med_ms` | doğrulandı (dosya okundu) |
| `checkpoints/*.cal.json` | alan (`val` / `gaps_val`) → (ölçüt, perde eşiği, onset eşiği, çözümleme kuralları) | `pitch_eval.py:308 load_selection()` |
| `NoteEvent` | 8 alan; **parmak ve ritim yok** | `note_event.py:35-51` |

**Boşluklar.**
- **JSON ara biçimi yok.** Kullanıcı sorusundaki "JSON/JavaScript veri yapıları" projede mevcut değil; MusicXML doğrudan `measures` + diziler üzerinden yazılıyor (`write_musicxml(out, measures, pitch, string, fret, sig, beat_q, ...)`). AlphaTab için bir JSON katmanı gerekirse yeni iştir.
- `NoteEvent`'in parmak/ritim taşımaması nedeniyle hem `scripts/transcribe.py` hem `gtab/pipeline.py` paralel diziler kullanıyor.

---

## Diyagram C — Model ve checkpoint ilişkileri

```mermaid
flowchart TD
    subgraph VERI["EĞİTİM VERİSİ"]
        V1["GuitarSet train + train_comp<br/>oyuncu 00-04, 150+150"]
        V2["gaps_fit<br/>202 kayıt (karantina sonrası)"]
        V3["gaps_tab (142 train kaydı)<br/>partisyon tel + parmak"]
        V4["rhythm etiketleri<br/>GAPS 143 + GuitarSet 360"]
    end

    subgraph AKUSTIK["AKUSTİK ZİNCİR — hepsi aynı gövdeden türedi"]
        M0["pitchcnn.pt<br/>K2 · PitchCNN<br/>perde kare F1 0.853"]
        M1["tabcnn.pt<br/>K3 · TabCNN<br/>tab F1 ~0.586"]
        M2["tabcrnn.pt<br/>K3.5 · TabCRNN<br/>0.634"]
        M3["tabcrnn_comp.pt<br/>K3.7 · +akor verisi<br/>0.667"]
        M4["tabcrnn_gaps.pt<br/>K3.8 · +GAPS perde<br/>0.697"]
        M5["tabcrnn_onset_h.pt<br/>K3.9 · +onset +harmonik<br/>0.739 · oracle tel 0.850"]
        M6["tabcrnn_poly.pt<br/>K3.10 · 0.749"]
        M7["tabcrnn_rep_off.pt<br/>K3.10 · +offset kafası<br/>0.760"]
        M8["tabcrnn_rep_valley.pt<br/>K3.11 · +vadi eğitimi<br/>0.769"]
        M9["tabcrnn_deep.pt<br/>K3.12 · derin CNN<br/>+ perde-onset kafası"]
        M10["tabcrnn_deep2.pt<br/>GÜNCEL TABAN<br/>GS tab F1 0.777<br/>GAPS nota F1 0.779"]
        MX["tabcrnn_gtab.pt<br/>REDDEDİLDİ<br/>dosya = deep2"]
        MA["tabcrnn_aug.pt<br/>REDDEDİLDİ (-0.06)"]
    end

    subgraph SEMBOLIK["SEMBOLİK MODELLER — ses görmez"]
        S1["edition.npz<br/>2a log-lineer<br/>REDDEDİLDİ (parmak < taban)"]
        S2["edition_seq.pt<br/>2b EditionNet BiLSTM<br/>KABUL<br/>gaps_test tel 0.815<br/>parmak 0.636"]
        S3["transitions.npz<br/>K3.6b öğrenilen geçiş<br/>güçlü modellerde katkısız"]
    end

    subgraph K5M["RİTİM MODELLERİ — yalnız katman-5-ritim"]
        R1["rhythm_q.pt<br/>RhythmNet 1q<br/>oracle IOI 0.933 KABUL"]
        R2["beat_net.pt<br/>BeatNet + DBN<br/>yarı otomatik F 0.631<br/>eşik sağlanmadı"]
    end

    subgraph KAL["KALİBRASYON"]
        CAL2["tabcrnn_deep2.cal.json<br/>val + gaps_val<br/>onset kaynağı = perde kafası"]
        CALV["tabcrnn_rep_valley.cal.json<br/>eski taban"]
    end

    subgraph TUK["TÜKETİCİLER"]
        U1["eval_pitch · eval_hmm<br/>diagnose_pitch"]
        U2["eval_edition · eval_playability"]
        U3["scripts/transcribe.py<br/>takipsiz"]
        U4["gtab/pipeline.py<br/>scripts/app/transcribe.py<br/>eval_rhythm · eval_score"]
    end

    M0 --> M1 --> M2 --> M3 --> M4 --> M5 --> M6 --> M7 --> M8 --> M9 --> M10
    M3 --> MA
    M10 --> MX

    V1 --> M1
    V1 --> M3
    V2 --> M4
    V2 --> M9
    V1 --> M9
    V3 --> MX
    V3 --> S1
    V3 --> S2
    V1 --> S3

    M10 --> CAL2
    M8 --> CALV

    M10 --> U1
    CAL2 --> U1
    S2 --> U2
    M10 --> U2
    M10 --> U3
    CAL2 --> U3
    S2 --> U3
    S3 --> U3

    V4 -.-> R1
    V4 -.-> R2
    M10 -.-> R2
    M10 -.-> U4
    R1 -.-> U4
    R2 -.-> U4
    S2 -.-> U4
    CAL2 -.-> U4

    classDef red stroke-dasharray: 3 3
    class MX,MA,S1 red
```

**Checkpoint envanteri — mevcut dalda `checkpoints/` (20 dosya).**

| Dosya | Durum | Kullanan |
|---|---|---|
| `tabcrnn_deep2.pt` + `.cal.json` | **CANLI TABAN** | tüm değerlendirme, `transcribe.py`, katman-5 pipeline |
| `edition_seq.pt` | **CANLI** (parmaklandırma) | `eval_edition`, `eval_playability`, `transcribe.py --fingering` |
| `transitions.npz` | CANLI ama katkısı 0 ölçüldü | `decode_lattice` (w=0 seçiliyor) |
| `tabcrnn_rep_valley.pt` + `.cal.json` | eski taban (K3.11) | yok |
| `tabcrnn_deep.pt`, `tabcrnn_deep_try.pt` | ara (K3.12) | yok |
| `tabcrnn_rep_off.pt`, `tabcrnn_poly.pt`, `tabcrnn_onset_h.pt`, `tabcrnn_onset.pt` | eski tabanlar | yok |
| `tabcrnn_gaps.pt`, `tabcrnn_comp.pt`, `tabcrnn_base.pt`, `tabcrnn.pt`, `tabcnn.pt`, `pitchcnn.pt` | tarihsel | yok |
| `tabcrnn_aug.pt` | **reddedildi** (−0,06) | yok |
| `edition.npz` | **reddedildi** (2a) | yalnız test (`TestLogLinearDecode`) |

**Katman-5'te checkpoint düzeni değişti:** 14 eski dosya `checkpoints/archive/` altına taşındı
(`tabcrnn_gtab.pt` da oraya eklendi) ve iki yeni dosya geldi: `beat_net.pt`, `rhythm_q.pt`.
`tabcrnn_deep2.pt`, `edition_seq.pt`, `transitions.npz` kökte kaldı.

---

## Diyagram D — Geliştirme ve test döngüsü

```mermaid
flowchart TD
    START(["Yeni katman / faz başlıyor"])

    subgraph HAZ["1 — HAZIRLIK"]
        DOC["Devlog aç: docs/devlog/README N<br/>neden · kapsam · kabul ölçütü<br/>ÖLÇÜT ÖNCEDEN YAZILIR"]
        BASE["Taban ölç: d1_baseline.json<br/>mevcut checkpoint, değişiklik yok"]
    end

    subgraph HATA["2 — HATA ANALİZİ (eğitim yok)"]
        DIAG["diagnose_pitch · diagnose_strings<br/>eval_playability · inventory_*"]
        SPLITV["SADECE DOĞRULAMA SETİ<br/>val · val_comp · gaps_val"]
        HYP["Tek hipotez kur<br/>hangi kaybı hedefliyor?"]
    end

    subgraph GELIS["3 — GELİŞTİRME"]
        CODE["Kod yaz: yeni dosya / yeni bayrak<br/>VARSAYILAN = eski davranış birebir"]
        UNIT["python -m unittest discover -s tests -t .<br/>170 test"]
        REGR["check_inference_regression<br/>parmak izi, CPU, saniyeler"]
        PROTO["check_protocol<br/>17 sızıntı denetimi"]
    end

    subgraph EGIT["4 — EĞİTİM (kullanıcı çalıştırır)"]
        TRAIN["train_onset / train_edition_seq<br/>init = mevcut checkpoint<br/>YENİ AD, üzerine yazma yok"]
        SEL["EpochSelector<br/>seçim split'i protokolden doğrulanır<br/>epoch 0 da değerlendirilir<br/>kept_initial · early stopping"]
    end

    subgraph DEGER["5 — DEĞERLENDİRME"]
        CALIB["Kalibrasyon: alan içi eşik araması<br/>val → GuitarSet · gaps_val → GAPS<br/>--save-cal / --load-cal"]
        EVAL["eval_pitch · eval_hmm · eval_edition<br/>eval_playability · eval_rhythm"]
        CMP["Kabul ölçütüyle karşılaştır<br/>assert_comparable: grup koruması<br/>tolerans: koşu 0.002 · eğitim 0.01"]
    end

    DEC{"Ölçüt sağlandı mı?"}

    subgraph KAPAN["6 — KAPANIŞ"]
        TEST["gaps_test: YALNIZ BİR KEZ<br/>assert_may_report<br/>report_label ile etiketli"]
        WRITE["Devlog sonuçları + açık riskler<br/>CLAUDE.md güncel durum<br/>D1 tablosuna yeni satır"]
        GIT["Dal commit: katman-N-konu<br/>main birleştirme = kullanıcı kararı"]
    end

    RETRY["Negatif sonucu KAYDET<br/>CLAUDE.md 'denenmiş, işe yaramayan'<br/>taban korunur"]

    FORBID["YASAK<br/>gaps_test ile seçim / kalibrasyon<br/>assert_may_select → ProtocolError<br/>EpochSelector kurucusunda otomatik"]

    START --> DOC --> BASE --> DIAG
    DIAG --> SPLITV
    SPLITV --> HYP --> CODE
    CODE --> UNIT --> REGR --> PROTO --> TRAIN
    TRAIN --> SEL --> CALIB --> EVAL --> CMP --> DEC
    DEC -- "evet" --> TEST --> WRITE --> GIT
    DEC -- "hayır" --> RETRY --> HYP

    FORBID -. "engellenir" .- SEL
    FORBID -. "engellenir" .- CALIB
    FORBID -. "engellenir" .- SPLITV

    GIT --> START

    classDef danger stroke-width:3px
    class FORBID,TEST danger
```

**Döngünün koda bağlı kuralları.**

| Kural | Nerede zorlanıyor | Kaynak |
|---|---|---|
| `gaps_test` seçimde/kalibrasyonda kullanılamaz | `assert_may_select()` → `ProtocolError`; `EpochSelector.__init__` otomatik çağırıyor | `protocol.py:127`, `selection.py:39` |
| Eğitim split listesi doğrulanır | `assert_may_train()`; `val`, `val_comp`, `gaps_val`, `gaps_test`, `gaps_skipped`, `val_half_*` eğitimde kullanılamaz | `protocol.py:121` |
| Tanımsız split adı hata verir | yazım hatasıyla sessizce kural dışına çıkılamaz | `protocol.py:113 rule()` |
| Raporlanan her sayı split cinsi etiketi taşır | `report_label(split)` | `protocol.py:141` |
| Farklı metrik grupları karşılaştırılamaz | `assert_comparable()` | `baselines.py:87` |
| Başlangıç modeli de yarışır | `EpochSelector` epoch 0'ı değerlendirir, `kept_initial()` | `selection.py:72` |
| Karantina kaydı eğitim/seçim listelerinden düşer | `gaps_training_files()` → `drop_quarantined()` | `protocol.py:231, 226` |
| Yeni kod eski çıkarımı kırmamalı | `check_inference_regression` parmak izi | `fingerprint.py`, `scripts/eval/check_inference_regression.py` |

**Test verisinin konumu (açıkça).** `gaps_test` bu döngüde **yalnız 6. adımda, faz kapanışında,
bir kez** okunur. 2., 3., 4. ve 5. adımlarda protokolce erişilemez. GuitarSet'te bağımsız test
split'i **yok** — oyuncu 05 (`val`, `val_comp`) hem seçim hem raporlama setidir ve "validation"
olarak etiketlenir (`A-04`).

**Döngünün bilinen boşluğu.** Protokol kapıları **yalnız Faz 0 sonrası yazılan kodda** zorunlu;
`train_onset`, `train_edition_seq`, `eval_pitch`, `eval_hmm` bilinçli olarak değiştirilmedi
(README15 "Açık riskler" 4). Fiilen bir ihlal yaşandı ve kayda geçti: README14 Adım 2'nin kabul
kuralı `gaps_test`'e bakıyordu (`G-07`).
