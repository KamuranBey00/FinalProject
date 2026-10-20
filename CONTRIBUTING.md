# Katkıda bulunanlar için rehber

Bu belge projeye yeni katılan birinin **neyin nerede olduğunu, şimdiye kadar ne
yapıldığını ve nasıl katkı verileceğini** tek yerden görmesi içindir.
Ayrıntılı gerekçeler ve deney sonuçları katman katman `docs/devlog/` altındadır.

## Proje ne yapıyor?
Gitar ses kaydından **tablatür (TAB)** üretiyoruz: hangi anda, hangi telde, hangi
perdenin çalındığını ses dalgasından tahmin ediyoruz. Ana hedef (bkz.
`inputs/Klasik_Gitar_AI_Proje_Sunumu.pptx`) **klasik (naylon telli) gitarda polifonik
transkripsiyon**. Sistem katman katman kuruluyor; her katman bir öncekini kırmadan
eklenir ve ölçülür.

```
ses (.wav) -> log-CQT -> TabCRNN (CNN + BiLSTM, tel başına softmax + onset kafası)
           -> nota bulma (onset tabanlı) -> tel/perde ataması -> ASCII TAB
```

## Hızlı başlangıç
```bash
pip install -r requirements.txt

# Veri (data/ git'te yok; script'lerle üretilir)
python -m scripts.data.get_guitarset          # GuitarSet indirme (mirdata)
python -m scripts.data.build_guitarset        # solo kayıtlar -> data/cache/{train,val}
python -m scripts.data.build_tab_labels       # tel/perde etiketleri
python -m scripts.data.build_comp             # akorlu kayıtlar -> data/cache/{train_comp,val_comp}
python -m scripts.data.get_gaps --splits train test   # klasik gitar (GAPS, ~15 GB)
python -m scripts.data.build_gaps --disjoint

# En iyi modelle bir kayıttan TAB üret
python -m scripts.eval.eval_hmm --model crnn --ckpt tabcrnn_rep_off.pt \
    --demo data/cache/val/05_Rock1-130-A_solo.npz --threshold 0.7 --w-transition 0 \
    --onset-thr 0.2 --refractory 6 --fallback 10 --peak --reattack 0.4 --rise-keep 6 --offset-thr 0.5
```
Tüm komutlar **proje kökünden** `python -m` ile çalışır. `--ckpt` argümanına çıplak
dosya adı verilirse `checkpoints/` altında aranır.

## Şimdiye kadar yapılanlar
Ana metrik: GuitarSet oyuncu 05 solo kayıtlarında kare düzeyinde **tab F1** (tel + perde birlikte doğru).

| Katman | İş | Sonuç | Devlog |
|---|---|---|---|
| 0–1 | Ortak nota şeması, CQT, GuitarSet veri hattı | temel | README, README1 |
| 2 | Monofonik perde modeli (PitchCNN) | perde kare F1 0.853 | README2 |
| 3 | Tel/perde modeli (TabCNN) | tab F1 ~0.59 | README3–4 |
| 3.5 | CNN + BiLSTM (TabCRNN) | 0.634 | README5 |
| 3.6 | Viterbi tel ataması; nota başına tek pozisyon | 0.664 (el yapımı kural işe yaramadı) | README6–7 |
| 3.7 | GuitarSet akorlu kayıtlarla veri ×2 | 0.667 | README8 |
| 3.8 | Klasik gitar: GAPS ile perde etiketli ince ayar | 0.697; GAPS nota F1 0.26 → 0.36 | README9 |
| 3.9 | Hata analizi → onset kafası → harmonik istifleme | **0.739**; tel doğruluğu 0.850; akorlarda nota F1 0.46 → 0.72 | README10 |

**Denenmiş ama işe yaramayanlar** (tekrar denemeden önce devlog'a bakın): el yapımı
"en az el hareketi" maliyeti, güçlü modellerde öğrenilen geçiş önseli, CQT'yi perde
kaydırarak veri çoğaltma, teli yalnızca vuruş anından seçmek.

## Klasör yapısı ve dosyaların görevleri

### Kök dizin
| Dosya / klasör | Görevi |
|---|---|
| `README.md` | Proje özeti, pipeline, sonuç tablosu, komutlar |
| `CONTRIBUTING.md` | Bu rehber |
| `CLAUDE.md` | Claude Code'un her oturumda okuduğu çalışma senaryosu ve güncel durum |
| `requirements.txt` | Python bağımlılıkları |
| `.gitignore` | `data/` ve önbellekleri git dışında tutar |
| `inputs/` | Proje sunumu (ana plan) ve katman karar belgeleri |
| `docs/devlog/` | Katman katman geliştirme günlüğü (en yüksek numara = güncel katman) |
| `checkpoints/` | Aktif modeller (aşağıda); eskiler `checkpoints/archive/` |
| `archive/` | Eski ve reddedilen kod — aktif hatta kullanılmaz (bkz. `archive/README.md`) |
| `data/` | *(git'te yok)* `raw/` = ham veri setleri, `cache/` = CQT + etiket önbelleği |

### `gtab/` — kütüphane (yalnızca import edilir)
| Dosya | Görevi |
|---|---|
| `config.py` | Ses/CQT sabitleri (22050 Hz, hop 512, 192 bin, 24 bin/oktav). Değiştirmek tüm modelleri geçersiz kılar. |
| `paths.py` | Proje yolları; checkpoint adlarını çözer |
| `utils.py` | Seed, cihaz seçimi, `guard_new_checkpoint` (eğitimin mevcut checkpoint'i ezmesini engeller) |
| `core/instrument.py` | Akort ve "bu perde hangi tel/perdelerde çalınır" mantığı |
| `core/note_event.py` | Tüm katmanların paylaştığı nota şeması (`NoteEvent`, `Transcription`) |
| `data/features.py` | Ses → log-CQT |
| `data/labels.py` | Notalar → kare bazlı perde ve onset etiketleri |
| `data/tab_labels.py` | Tel/perde (tab) etiketleri ve tel başına onset hedefi |
| `data/torch_dataset.py` | Eğitim veri setleri: pencere (CNN), sekans (CRNN), perde etiketli sekans (GAPS) |
| `data/guitarset.py` | GuitarSet yardımcıları (oyuncu, solo/akor, nota dönüşümü) |
| `data/gaps.py` | GAPS okuyucu, icracı-ayrık doğrulama bölmesi |
| `data/gaps_score.py` | GAPS MusicXML TAB + parmak → tekrar açma + syncpoint + MIDI eşleme (Katman 3.13); nota başına partisyon konumu + ölçü imleri (Katman 5) |
| `data/rhythm_labels.py` | Ritim etiketleri: GAPS partisyon konumundan rubato'lu vuruş / ölçü başı + nota başına vuruş konumu; GuitarSet JAMS vuruşları (Katman 5) |
| `models/nets.py` | PitchCNN, TabCNN, TabCRNN, TabCRNNOnset, HarmonicStack; eski ağırlıklardan sıcak başlatma |
| `models/losses.py` | Tel olasılıklarından perde ve perde-onset olasılığı (perde etiketli veride kayıp için) |
| `models/inference.py` | Model yükleme (türü checkpoint'ten tanır; varsayılan aktif yığından), uzun kayıtlarda parça parça çıkarım |
| `models/stack.py` | Aktif model yığını (`checkpoints/active.json`) + bağımlılık denetimi: taban ses modeli ↔ kalibrasyon, BeatNet, BeatNet önbelleği, regresyon parmak izi |
| `models/edition_net.py` | "Klasik edisyon önerisi": nota dizisi → tel/perde + sol el parmağı (iki yönlü dizi modeli) |
| `decoding/decode.py` | Olasılık → nota → ASCII TAB; nota başına tel seçimi |
| `decoding/viterbi.py` | Perde matrisi, nota bulma (eşik tabanlı ve onset tabanlı), el yapımı Viterbi |
| `decoding/transitions.py` | Veriden öğrenilen tel geçiş modeli ve Viterbi'si |
| `models/rhythm_net.py` | Bağlamlı kuantalayıcı (Katman 5 Adım 1q): notalar + vuruşlar → vuruş içi konum (BiLSTM + vuruş başına ortak ızgara çözümlemesi) |
| `models/beat_net.py` | Öğrenilen vuruş + ölçü başı aktivasyonu (Katman 5 Adım 1b): deep2 gövde temsili → BiLSTM → (yok / vuruş / ölçü başı) |
| `decoding/edition.py` | Edisyon önerisi için nota dizisi yardımcıları (akor gruplama, etiket okuma, taban parmak kuralı) |
| `decoding/rhythm.py` | Ritim (Katman 5): başlangıç zarfları, vuruş takibi (dp / plp), ölçü başı + ölçü, kuantalama (vuruş içi bölüm seçimi) |
| `decoding/beat_dbn.py` | Bar-pointer DBN (tempo × ölçü içi konum × ölçü 1–4; tek karelik vuruş bölgesi) + tepe seçimi tabanı |
| `decoding/taps.py` | Tap modu (Katman 5 Adım 3): dokunuş dosyası okuma, dokunuşlar → vuruşlar (gecikme, fazla / kaçan dokunuş) + ölçü fazı Viterbi'si |
| `decoding/playability.py` | Çalınabilirlik metrikleri (zamanda örtüşen çiftlerde parmak çakışması, el açıklığı, ters parmak sırası, barre, pozisyon kayması) |
| `data/degrade.py` | Kayıt koşulu bozulmaları (yankı, telefon, gürültü, şarkı / konuşma karışımı; eğitim için rastgele zincir; Katman 6) |
| `data/hex.py` | GuitarSet hex kaydı → tel başına spektrum (yardımcı eğitim hedefi) |
| `data/contour.py` | Sürekli F0 konturu (Katman 4 hazırlığı): cent dönüşümü, nota–kontur hizalaması, kalite bayrakları |
| `evaluation/metrics.py` | Ortak metrikler (P/R/F1, tab F1, onset F1) |
| `evaluation/protocol.py`, `selection.py` | Değerlendirme protokolü: split kapıları (gaps_test seçimde kullanılamaz), karantina, seçim kuralları |
| `evaluation/technique_eval.py` | Teknik değerlendirme (Katman 4): nota olayı eşleme, sınıf başına P/R/F1, macro-F1 |
| `evaluation/baselines.py`, `fingerprint.py` | Referans tablo (`docs/baselines/d1_baseline.json`) ve çıkarım parmak izi |
| `evaluation/rhythm_metrics.py` | Ritim metrikleri: vuruş / ölçü başı F, CMLt / AMLt, tempo Acc1/2, konum ve IOI (nota değeri) doğruluğu |
| `evaluation/pitch_eval.py` | Perde/nota değerlendirme çekirdeği: alan-içi kalibrasyon (GuitarSet→val, GAPS→gaps_val), çözümleme kuralı araması, 50/100 ms nota F1 |
| `render/score.py` | Ölçülü TAB modeli (Katman 5 Adım 2): konumlar + ölçü başı + ölçü imi → ölçüler, bağlar, nota değerleri |
| `render/musicxml.py` | MusicXML yazıcı (GAPS düzeni: nota portresi + parmak, TAB portresi + akort) |
| `pipeline.py` | Uçtan uca transkripsiyon: `analyze` (CQT → notalar → tel/perde (ses / edisyon) → vuruş aktivasyonu) + `rhythm` (BPM ya da dokunuşlar → vuruşlar → 1q → ölçüler) |

### `scripts/` — çalıştırılan dosyalar
| Dosya | Görevi |
|---|---|
| `data/get_guitarset.py`, `data/get_gaps.py` | Veri setlerini indirir |
| `data/build_guitarset.py`, `data/build_tab_labels.py`, `data/build_comp.py` | GuitarSet solo ve akorlu kayıt önbelleği |
| `data/build_gaps.py` | GAPS önbelleği (`--disjoint` = icracı-ayrık) |
| `data/build_gaps_tab.py` | GAPS partisyon TAB + parmak etiketleri → `data/cache/gaps_tab/`; `--recover-repeats` → `gaps_tab_rec/` (içerikten ölçü sırası, 236 kayıt) |
| `data/get_voices.py`, `data/build_degraded.py` | Katman 6 Adım 2: şarkı + konuşma ses bankaları (VocalSet, LibriSpeech); bozulmuş CQT önbellekleri (`train_onset --degrade-root`) |
| `data/build_hex.py`, `data/build_contour.py` | Hex tel spektrumu ve F0 konturu önbellekleri |
| `data/build_rhythm.py` | Ritim etiketleri (GAPS + GuitarSet) → `data/cache/rhythm/` |
| `data/build_beat_feats.py` | deep2 gövde temsili önbelleği → `data/cache/beat_feats/<ckpt>/` (vuruş ağı girdisi) |
| `data/build_soundfont.py` | TAB görüntüleyicinin naylon gitar ses bankası: FreePats SF2 (CC0) → program 24 + de-click + SF3 → `app/soundfont/nylon_guitar.sf3.js` (README21) |
| `train/train_onset.py` | Ses modeli eğitimi (onset/offset/perde-onset kafaları, derin CNN; güncel taban deep2) |
| `train/train_edition_seq.py` | Edisyon önerisi dizi modeli (sembolik, ~2 dk) |
| `train/train_rhythm_q.py` | Bağlamlı kuantalayıcı (sembolik, vuruş/nota bozma artırması; ~11 dk) → `rhythm_q.pt` |
| `train/train_beat.py` | Vuruş + ölçü başı ağı (tempo artırması, DBN ayar taraması) → `beat_net.pt` |
| `eval/eval_pitch.py` | Nota ve kare F1, polifoni kırılımı, kalibrasyon (`--save-cal` / `--load-cal`); varsayılan splitler gaps_val val val_comp (gaps_test yalnız açıkça) |
| `eval/diagnose_pitch.py` | Kaçan/hayalet notaların nedeni, hızlı tekrarlar, tepe görünürlüğü |
| `eval/eval_hmm.py` | Tel ataması teşhisi + uçtan uca tab F1 (**ana GuitarSet ölçümü**) + `--demo` |
| `eval/eval_edition.py` | Edisyon önerisi: tel uyumu, parmak (ürün sayısı), uçtan uca (deep2 notaları) |
| `eval/eval_score.py` | Ölçülü TAB: MusicXML gidiş-dönüş + uçtan uca ölçü içi konum (Adım 2) + ölçü kaydırma tavanı + tap modu simülasyonu (Adım 3) |
| `app/tap_recorder.html` | Tarayıcıda dokunuş kaydedici (tap modu): ses çal, BOŞLUK ile vuruşlar → `<kayıt>.txt` |
| `app/tab_viewer.html` | `.musicxml` görüntüleyici + çalma (alphaTab; naylon gitar ses bankası `app/soundfont/`, ses düzeyi 0.5; README21) |
| `app/transcribe.py` | Ürün komutu: ses (ya da .npz) + `--bpm` ya da `--taps` + `--meter` (+ `--mode`, `--shift`) → `.musicxml` |
| `app/transcribe_ascii.py` | Ses → nota tablosu + ASCII TAB (`--fingering`; `--meter` + `--taps` / `--bpm` ile ölçü çizgileri) |
| `eval/check_stack.py` | Model yığını denetimi (`--set` ile yeni taban); eskiyen parçalar için yeniden üretme komutları |
| `eval/check_protocol.py`, `eval/check_inference_regression.py` | Veri envanteri + split kapıları; deep2 çıkarım çıktısı referansla birebir mi |
| `eval/eval_robust.py` | Katman 6 Adım 0: yapay kayıt koşullarında nota / kare / polifoni F1 (gaps_val) |
| `eval/eval_playability.py` | Çalınabilirlik (model / naif / editör; gaps_val) |
| `eval/inventory_technique.py`, `eval/probe_gaps_f0.py` | Katman 4 hazırlığı: teknik etiket envanteri, GAPS F0 yoklaması |
| `eval/eval_taps.py` | Tap modu gerçek dokunuşla ölçüm (dokunulan aralıkta; yarı otomatik + simüle karşılaştırması) |
| `eval/eval_rhythm.py` | Ritim: kuantalayıcı tavanı, vuruş takibi, ölçü başı, kuantalama zinciri (Katman 5); `--qnet` ile 1q, `--beatnet` ile 1b satırları |

Eski katmanların betikleri (Katman 2–3.8 eğitimleri, tamamlanmış tanılamalar, SynthTab hattı, reddedilen
adımlar) `archive/` altında; liste ve çalıştırma için `archive/README.md`.

### `tests/`
`python -m unittest discover -s tests -t .` (204 test; veri/model gerektirenler yoksa atlanır).

### `checkpoints/`
| Dosya | Katman | Not |
|---|---|---|
| **`tabcrnn_deep2.pt`** + `tabcrnn_deep2.cal.json` | 3.12 | **Güncel ses tabanı** ("duyulan konum"): GS tab F1 0.777, GAPS nota/kare 0.779 / 0.701; kalibrasyon seçimi (`--load-cal`) |
| **`edition_seq.pt`** | 3.13 | "Klasik edisyon önerisi": GAPS tel 0.815, parmak 0.636 (ürün tam 0.434) |
| `transitions.npz` | 3.6b | öğrenilen tel geçiş modeli (`eval_hmm`) |
| `rhythm_q.pt`, `beat_net.pt` | 5 | bağlamlı kuantalayıcı; vuruş ağı (deep2 temsiline bağlı) |
| `edition_seq_rec.pt` | 3.13 (Armağan) | aday parmak modeli (gaps_val tel / parmak 0.784 / 0.700); ürüne alınacak |
| `active.json` (isteğe bağlı) | — | aktif model yığını; yoksa `gtab/models/stack.py` varsayılanı |
| `archive/*` | 2 – 3.13 | eski tabanlar ve reddedilen modeller (pitchcnn … rep_valley, deep, gtab, edition.npz) |

## Çalışma kuralları
1. **Katman modeli:** Her yeni adım öncekini kırmadan eklenir. Mevcut bir checkpoint'in
   ya da önbelleğin üzerine yazılmaz; yeni ad verilir (`--ckpt yeni_ad.pt`).
2. **Her adımın bir devlog'u olur:** `docs/devlog/README<N>.md`. İçinde neden, ne
   yapıldı, testler, çalıştırma komutları, karar kuralı ve sonuçlar bulunur. Ana
   `README.md`'deki devlog listesi ve yol haritası güncellenir.
3. **Önce ölç, sonra çöz:** Yeni bir iyileştirmeden önce hata analiziyle hangi kaybın
   hedeflendiği gösterilir. "Ciddi iyileşme" ölçütü devlog'da önceden yazılır;
   sağlanmadan bir sonraki kademeye geçilmez.
4. **Kod yerleşimi:** Paylaşılan kod `gtab/`'a, çalıştırılan dosyalar `scripts/`'e gider.
   Script'ler birbirinden import etmez. Yollar için `gtab/paths.py` kullanılır.
5. **Değişiklik testi:** Yapısal bir değişiklikten sonra eski bir model değerlendirilip
   sonuçların **birebir aynı** çıktığı doğrulanır (ör. `eval_hmm --quick`).
6. **Git:** Her katman ayrı dalda (`katman-<no>-<konu>`). `main`'e birleştirme
   PR üzerinden yapılır. `data/` asla commit edilmez.

## Değerlendirme kuralları (sonuçların kıyaslanabilir kalması için)
- GuitarSet ana doğrulama: oyuncu 05 **solo** kayıtları (`data/cache/val`). Polifoni
  ölçümü için aynı oyuncunun akorlu kayıtları (`val_comp`).
- GAPS: icracı-ayrık bölme (`--disjoint`). Eşik ve parametreler `gaps_train` içinden
  ayrılan doğrulama parçasında seçilir; **`gaps_test` yalnızca raporlama** içindir.
- Tek seed ile ±0.01'lik farklar gürültü sayılır.

## Açık işler (sıradaki adımlar)
1. **GAPS kalibrasyonu:** GAPS eşiklerini GAPS doğrulamasında seçmek (eğitim gerektirmez).
2. **Polifoni için perde tarafı:** Akorlu kayıtlarda kalan kaybın çoğu tel değil perde
   tarafında: kaçırılan perdeler %23.5, yanlış pozitiflerin %75'i hayalet nota.
   Adaylar: offset kafası, onset koşullu kare kafası, GAPS onset kaybını güçlendirmek.
3. **Tel tarafı (gerekirse):** Daha ince frekans çözünürlüğü, hex kayıtlardan
   öğretmen–öğrenci eğitimi, SynthTab naylon ve GAPS partisyon TAB etiketleri.

## Veri kaynakları
- GuitarSet — https://guitarset.weebly.com/ (mirdata ile)
- GAPS v1.1 — https://huggingface.co/datasets/xavriley/GAPS (MIT)
- SynthTab — https://github.com/yongyizang/SynthTab (UR Box, elle indirme)
