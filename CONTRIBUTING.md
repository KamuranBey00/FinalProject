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
| `checkpoints/` | Eğitilmiş modeller (aşağıda) |
| `data/` | *(git'te yok)* `raw/` = ham veri setleri, `cache/` = CQT + etiket önbelleği |

### `gtab/` — kütüphane (yalnızca import edilir)
| Dosya | Görevi |
|---|---|
| `config.py` | Ses/CQT sabitleri (22050 Hz, hop 512, 192 bin, 24 bin/oktav). Değiştirmek tüm modelleri geçersiz kılar. |
| `paths.py` | Proje yolları; checkpoint adlarını çözer |
| `utils.py` | Seed, cihaz seçimi |
| `core/instrument.py` | Akort ve "bu perde hangi tel/perdelerde çalınır" mantığı |
| `core/note_event.py` | Tüm katmanların paylaştığı nota şeması (`NoteEvent`, `Transcription`) |
| `data/features.py` | Ses → log-CQT |
| `data/labels.py` | Notalar → kare bazlı perde ve onset etiketleri |
| `data/tab_labels.py` | Tel/perde (tab) etiketleri ve tel başına onset hedefi |
| `data/torch_dataset.py` | Eğitim veri setleri: pencere (CNN), sekans (CRNN), perde etiketli sekans (GAPS) |
| `data/guitarset.py` | GuitarSet yardımcıları (oyuncu, solo/akor, nota dönüşümü) |
| `data/gaps.py` | GAPS okuyucu, icracı-ayrık doğrulama bölmesi |
| `data/synthtab.py` | SynthTab okuyucu (zip'leri açmadan okur; doğrulanmış dosya yapısı belgeli) |
| `models/nets.py` | PitchCNN, TabCNN, TabCRNN, TabCRNNOnset, HarmonicStack; eski ağırlıklardan sıcak başlatma |
| `models/losses.py` | Tel olasılıklarından perde ve perde-onset olasılığı (perde etiketli veride kayıp için) |
| `models/inference.py` | Model yükleme (türü checkpoint'ten tanır), uzun kayıtlarda parça parça çıkarım |
| `decoding/decode.py` | Olasılık → nota → ASCII TAB; nota başına tel seçimi |
| `decoding/viterbi.py` | Perde matrisi, nota bulma (eşik tabanlı ve onset tabanlı), el yapımı Viterbi |
| `decoding/transitions.py` | Veriden öğrenilen tel geçiş modeli ve Viterbi'si |
| `evaluation/metrics.py` | Ortak metrikler (P/R/F1, tab F1, onset F1) |
| `evaluation/pitch_eval.py` | Perde/nota değerlendirme çekirdeği: alan-içi kalibrasyon (GuitarSet→val, GAPS→gaps_val), çözümleme kuralı araması, 50/100 ms nota F1 |

### `scripts/` — çalıştırılan dosyalar
| Dosya | Görevi |
|---|---|
| `data/get_guitarset.py`, `data/get_gaps.py` | Veri setlerini indirir |
| `data/build_guitarset.py`, `data/build_tab_labels.py`, `data/build_comp.py` | GuitarSet solo ve akorlu kayıt önbelleği |
| `data/build_gaps.py` | GAPS önbelleği (`--disjoint` = icracı-ayrık) |
| `data/build_synthtab.py` | SynthTab naylon önbelleği (`--inspect` ile yapı kontrolü) |
| `train/train_pitch.py` | Katman 2: perde modeli |
| `train/train_tab.py` | Katman 3: TabCNN |
| `train/train_crnn.py` | Katman 3.5–3.7: TabCRNN (`--splits`, `--ckpt`) |
| `train/train_domain.py` | Katman 3.8: GuitarSet (tab) + GAPS (perde) karışık eğitim |
| `train/train_onset.py` | Katman 3.9: onset kafası (`--harmonics default` ile harmonik istifleme) |
| `eval/sweep_threshold.py` | Katman 2 eşik taraması |
| `eval/eval_viterbi.py` | Katman 3.6: baseline / greedy / Viterbi karşılaştırması |
| `eval/eval_hmm.py` | Tel ataması teşhisi + uçtan uca tab F1 (**ana GuitarSet ölçümü**) + `--demo` |
| `eval/eval_pitch.py` | Nota ve kare F1, polifoni kırılımı (GAPS ve GuitarSet) |
| `eval/diagnose_strings.py` | Hata analizi: kayıp nereden geliyor, hangi teller karışıyor |
| `eval/diagnose_pitch.py` | Katman 3.10: kaçan/hayalet notaların nedeni, süre/register/polifoni kırılımı |
| `eval/check_gaps_labels.py` | Katman 3.10: GAPS `.match` eşleşme oranı ile model hatası ilişkisi |
| `eval/diagnose_onset_features.py` | Katman 3.10 Adım 3c-0: hızlı tekrarları hangi giriş özelliği ayırır (CQT / kısa pencere STFT; model yok) |

### `checkpoints/`
| Dosya | Katman | Not |
|---|---|---|
| `pitchcnn.pt`, `tabcnn.pt`, `tabcrnn.pt` | 2, 3, 3.5 | ilk modeller |
| `tabcrnn_base.pt`, `tabcrnn_comp.pt`, `tabcrnn_aug.pt` | 3.7 | taban / +akor / +çoğaltma (zararlı çıktı) |
| `tabcrnn_gaps.pt` | 3.8 | + GAPS ince ayarı |
| `tabcrnn_onset.pt` | 3.9 | + onset kafası |
| **`tabcrnn_onset_h.pt`** | 3.9 | Katman 3.9 tabanı: + harmonik istifleme (GS tab F1 0.743, kurallar + mix) |
| `tabcrnn_poly.pt` | 3.10 | + keskin onset, GuitarSet perde kaybı, kısa nota ağırlığı (tab F1 0.749 / 0.753) |
| **`tabcrnn_rep_off.pt`** | 3.10 | **Güncel taban:** + offset kafası, hızlı tekrar onset ağırlığı, GAPS perde ağırlığı ×2 (tab F1 0.760, GAPS nota/kare 0.689/0.675) |
| `transitions.npz` | 3.6b | öğrenilen tel geçiş modeli |

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
