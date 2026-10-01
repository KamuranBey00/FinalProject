# Katman 3.8 — Klasik gitar alan uyarlaması (Sunum: Phase 3–4)

## Neden bu katman
Katman 3.7'nin sonunda GuitarSet-mic üzerinde tel ayrımı için azalan getiri
noktasına gelindi (CRNN + comp + greedy = tab F1 0.667). Sunumun asıl hedefi
ise **klasik (naylon telli) gitar**. GuitarSet çelik telli akustik gitar;
modelimiz şimdiye kadar hiç naylon tel duymadı. Sunumdaki araştırma sorusu
tam olarak bu boşluğu hedefliyor:

> Büyük ölçekli sentetik nylon-string ön-eğitimi ve gerçek klasik gitar
> verisiyle domain adaptation, polifonik audio-to-TAB genellemesini ne ölçüde
> iyileştirir?

Katman 4 (teknikler) bu katmandan sonra gelir. Teknik tespiti, klasik gitar
sesinde doğru çalışan bir nota katmanına dayanmalı.

## Veri kaynakları (sunum slayt 6–7)
| Kaynak | Ne verir | Bizde rol | Durum |
|---|---|---|---|
| **GuitarSet** (solo + comp) | gerçek ses, tel/perde | tab denetimi, benchmark | hazır (`data/cache/{train,val,train_comp,val_comp}`) |
| **GAPS v1.1** (Hugging Face, MIT) | 300+ gerçek klasik gitar performansı, hizalı MIDI, MusicXML (nota + **TAB portresi + sol el parmak numarası**), `.match` hizalamaları | klasik gitar değerlendirmesi + perde denetimli ince ayar | indirme script'i hazır (15.3 GB) |
| **SynthTab** (naylon tını = `luthier_*`) | sentetik gitar, tel bazlı MIDI + JAMS etiketi | büyük ölçekli ön-eğitim | Dev Set indirildi ve doğrulandı (`data/raw/synthtab/SynthTab_Dev.zip`); tam set için luthier zip'leri elle indirilecek |

GAPS notları (doğrulandı):
- Ses 48 kHz stereo; yükleyici 22050 Hz mono'ya indirir (config ile aynı).
- MIDI zamanlaması sesle hizalı (örnek: `001_mvswc`, 227.8 sn MIDI / 228.3 sn ses).
- Resmi bölünme: train 270 / test 30 / atanmamış 101.
  **12 icracı hem train'de hem test'te** yer alıyor → sunumdaki *performer-disjoint*
  ilkesine uymuyor. Biz test'i resmi haliyle tutuyoruz (literatürle kıyas için)
  ve bu 12 icracının train kayıtlarını eğitimden çıkarıyoruz (`--disjoint`).
- Disk: 73 GB boş. GAPS 15.3 GB. SynthTab'in tamamı ~2 TB → **yalnızca nylon
  alt kümesinden sınırlı bir parça** (varsayılan ≤30 saat CQT, float16).

## Yol haritası (alt adımlar)
| Adım | İş | Sunumdaki karşılığı | Çıktı |
|---|---|---|---|
| 3.8a | GAPS indir + önbellek (CQT + perde roll'u) | Phase 0 (ortak şema) | `data/cache/gaps_{train,test}` |
| 3.8b | **Alan farkı ölçümü:** GuitarSet modelini GAPS test'te değerlendir | Değerlendirme planı: nota P/R/F1, onset toleransı, polifoni seviyeleri | taban sayılar |
| 3.8c | GAPS ile **perde denetimli** ince ayar (GuitarSet tab kaybı + GAPS perde kaybı, karışık batch) | Phase 4: classical domain adaptation | `tabcrnn_gaps.pt` |
| 3.8d | SynthTab nylon ön-eğitimi → GuitarSet → GAPS ince ayar | Phase 4: SynthTab pretraining | `tabcrnn_syn*.pt` |
| 3.8e | (ileri) GAPS partisyon TAB'ı + `.match` → klasik gitarda tel etiketi; `<fingering>` → parmaklandırma verisi | Slayt 10–11 (fingering) | Katman 7'ye girdi |

## Ana deney tablosu (sunum slayt 9: "Ana deney")
Her satır **aynı iki test setinde** ölçülür: GAPS test (görülmemiş gerçek
klasik gitar) ve GuitarSet 05 solo val.

| Eğitim | GAPS nota F1 | GAPS kare F1 | GuitarSet tab F1 |
|---|---|---|---|
| GuitarSet-only (`tabcrnn_comp.pt`) | 3.8b | 3.8b | 0.667 |
| GuitarSet → GAPS | 3.8c | 3.8c | 3.8c |
| SynthTab → GuitarSet | 3.8d | 3.8d | 3.8d |
| SynthTab → GuitarSet → GAPS | 3.8d | 3.8d | 3.8d |

## Metrikler (sunum slayt 12)
- **Nota seviyesi** P/R/F1: `mir_eval.transcription`, onset toleransı 50 ms, offset yok sayılır.
- **Kare seviyesi** perde P/R/F1.
- **Polifoni kırılımı:** aynı anda 1 / 2 / 3 / 4+ nota çalan karelerde ayrı ayrı.
- GuitarSet'te tab F1 (mevcut metrik). GAPS'te tab etiketi 3.8e'ye kadar yok.

## Ağırlık / model notu
Mimari değişmez (TabCRNN). GAPS'te yalnızca perde etiketi olduğu için perde
olasılığı, tel olasılıklarından türetilir:
`P(perde p) = 1 − Π_tel (1 − P_tel(fret = p − akort_tel))`.
Böylece tek bir tab modeli hem tab etiketli (GuitarSet, SynthTab) hem perde
etiketli (GAPS) veriden öğrenir.

## Karar kuralları
- 3.8b: GAPS nota F1, GuitarSet'teki değerden belirgin düşükse (beklenen)
  alan farkı kanıtlanmış olur ve 3.8c/3.8d anlamlıdır.
- 3.8c/d: GAPS test'te ≥ +0.03 nota F1 **ve** GuitarSet tab F1'de en fazla
  −0.01 kayıp → kabul. Aksi halde GuitarSet-only model korunur.
- Parametreler GAPS **train**'den ayrılan bir doğrulama parçasında seçilir;
  GAPS test yalnızca raporlamada kullanılır (sızıntı yok).

## Kod (yeni klasör yapısı; komutlar proje kökünden `python -m` ile)
| Dosya | Görev |
|---|---|
| `gtab/data/gaps.py`, `gtab/data/synthtab.py` | veri seti okuyucuları |
| `gtab/models/losses.py` | tel olasılıklarından perde olasılığı (noisy-OR) |
| `gtab/models/inference.py` | model yükleme + uzun kayıtlarda parça parça çıkarım |
| `scripts/data/get_gaps.py`, `build_gaps.py` | GAPS indirme + önbellek (`--disjoint`) |
| `scripts/data/build_synthtab.py` | SynthTab → önbellek (`--inspect`, `--timbre`, `--max-hours`) |
| `scripts/eval/eval_pitch.py` | nota/kare/polifoni metrikleri (GAPS + GuitarSet) |
| `scripts/train/train_domain.py` | karışık denetimli eğitim (tab CE + perde BCE), `--init` |

## SynthTab Dev bulguları (1 Ekim 2026)
Gerçek yapı ilk tahminden farklı çıktı; okuyucu buna göre yazıldı:
- Etiket: `jams/<parça>/string_1..6.mid` (gerçek zaman, perde) + JAMS
  `sandbox` içinde `string_index` (1 = ince E) ve `open_tuning`. JAMS'teki
  `note_tab` zamanı tick cinsinden olduğu için MIDI kullanılıyor.
- Her tel MIDI'sinde t=0'da perde-24 işaretleyici nota var → atılıyor.
- Çalınmayan tel için MIDI dosyası yok (169 parçanın 92'si) → o tel boş sayılıyor.
- Standart akortta olmayan ve 7 telli parçalar atlanıyor.
- Ses–etiket hizası: gecikme 0–2 kare (≤46 ms) ✓.
- `luthier_*` tınısı = Ample Guitar L (Alhambra Luthier) = **klasik naylon**.
  Dev Set'te yalnızca 6 luthier parçası var, 4'ü kullanılabilir (0.18 saat).
  Gerçek ön-eğitim için tam setin luthier zip'leri gerekiyor.
- **Erken sinyal:** `tabcrnn_comp.pt`, 4 sentetik naylon parçada tab F1 **0.73**
  (marj 0.65) veriyor. Sentetik naylon modelimiz için zor değil; GAPS'teki düşüş
  (tek parçada nota F1 ~0.20) daha çok gerçek kayıt koşullarından ve yoğun
  polifoniden geliyor olabilir. 3.8b'nin tam GAPS sonucu bunu netleştirecek.
  SynthTab indirmesine karar vermeden önce bu sonuç beklenmeli.

## Çalıştırma sırası
```bash
# 3.8a — GAPS (~15 GB)
python -m scripts.data.get_gaps --splits train test
python -m scripts.data.build_gaps --disjoint

# 3.8b — alan farkı (taban)
python -m scripts.eval.eval_pitch --ckpt tabcrnn_comp.pt --splits gaps_test val val_comp

# 3.8c — GAPS ile ince ayar + ölçüm
python -m scripts.train.train_domain --init tabcrnn_comp.pt --tab-splits train,train_comp --pitch-splits gaps_train --epochs 15 --lr 3e-4 --ckpt tabcrnn_gaps.pt
python -m scripts.eval.eval_pitch --ckpt tabcrnn_gaps.pt --splits gaps_test val val_comp
python -m scripts.eval.eval_hmm --model crnn --ckpt tabcrnn_gaps.pt

# 3.8d — SynthTab (Dev ile hat doğrulandı; tam set luthier zip'leri inince)
python -m scripts.data.build_synthtab --inspect --audio data/raw/synthtab/<luthier_klasoru> --labels data/raw/synthtab/all_jams_midi_V2_60000_tracks.zip
python -m scripts.data.build_synthtab --audio data/raw/synthtab/<luthier_klasoru> --labels data/raw/synthtab/all_jams_midi_V2_60000_tracks.zip --max-hours 30
python -m scripts.train.train_domain --tab-splits synthtab_train --pitch-splits "" --select guitarset --epochs 10 --ckpt tabcrnn_syn.pt
python -m scripts.train.train_domain --init tabcrnn_syn.pt --tab-splits train,train_comp --pitch-splits gaps_train --epochs 15 --lr 3e-4 --ckpt tabcrnn_syn_gaps.pt
```

## Doğrulama (1 Ekim 2026)
- GAPS örnek parça (`001_mvswc`): önbellek GuitarSet şemasıyla aynı; kareler %62 polifonik.
  Zaman/perde kayması taraması en iyi F1'i **0 kaymada** veriyor → hizalama doğru.
- **Ön alan farkı sinyali (tek parça, kesin değil):** `tabcrnn_comp.pt` ile GAPS
  nota F1 0.20 / kare F1 0.34; GuitarSet val (3 kayıt) nota F1 0.87.
- `--disjoint`: 27 test icracısı → 270 train kaydından 39'u eğitimden çıkıyor.
- `train_domain.py`: noisy-OR perde olasılığı elle hesapla aynı; perde kaybıyla
  öğrenme çalışıyor (tek parçada ezberleme testi: kare F1 0.39 → 0.64);
  `train()` iki modda (tab+perde, yalnızca tab) uçtan uca çalışıyor.
- `build_synthtab.py`: GuitarSet JAMS + sesi SynthTab benzeri bir zip'e
  konduğunda önbelleği **birebir** üretiyor (tab %100 aynı, CQT farkı 0.008 dB).
  Gerçek SynthTab yapısı Dev Set inince `--inspect` ile teyit edilecek.

## Sonuçlar — 3.8b / 3.8c (1 Ekim 2026)
Eşik GuitarSet val'de seçildi (0.7); GAPS test yalnızca raporlama.

| Model | GAPS nota F1 | GAPS kare F1 | GAPS kare F1 polifoni 1/2/3/4+ | GS val nota F1 | GS val_comp nota F1 | GS tab F1 | oracle tel doğruluğu |
|---|---|---|---|---|---|---|---|
| `tabcrnn_comp.pt` (GuitarSet-only) | 0.255 | 0.490 | 0.41 / 0.47 / 0.51 / 0.54 | 0.861 | 0.390 | 0.667 | 0.762 |
| **`tabcrnn_gaps.pt`** (+GAPS perde ince ayarı) | **0.360** | 0.429 | 0.54 / 0.46 / 0.41 / 0.39 | **0.876** | **0.463** | **0.697** | **0.797** |

Yorum:
- **Alan farkı kanıtlandı (3.8b):** GuitarSet'te nota F1 0.86, GAPS'te 0.26.
- **3.8c karar kuralı sağlandı:** GAPS nota F1 +0.105 (≥ +0.03) ve GuitarSet
  tab F1 kaybı yok, tersine +0.03 → `tabcrnn_gaps.pt` kabul.
- **Beklenmeyen kazanç:** Yalnızca *perde* etiketli gerçek klasik gitar verisi,
  GuitarSet'te *tel* doğruluğunu 0.762 → 0.797 yükseltti. Daha fazla gerçek
  ses, modelin akustik temsilini genel olarak güçlendiriyor.
- **Uyarı:** GAPS kare F1 düştü (0.49 → 0.43), en çok yoğun polifonide (4+:
  0.54 → 0.39). Muhtemel sebepler: (1) eşik GuitarSet'te seçildi, ince ayarlı
  modelin GAPS'teki kalibrasyonu farklı; (2) noisy-OR perde kaybı, akorlarda
  modeli az nota tahmin etmeye itiyor olabilir. Sonraki katmanda incelenecek.
- **Ölçüm notu:** val_comp nota F1'in düşük olması (0.39–0.46), kare F1
  yüksekken (0.73–0.77), `segment_notes`'un tekrar eden aynı perdeyi tek nota
  saymasından geliyor (akor tekrarları). Nota seviyesi kararlar için **onset
  bilgisi** gerekiyor.

## Durum
- [x] Yol haritası, veri kaynakları doğrulandı
- [x] Kod: get_gaps / build_gaps / eval_pitch / train_domain / build_synthtab (testli)
- [x] Repo hiyerarşik yapıya taşındı (gtab alt paketleri + scripts/ + checkpoints/); sonuçlar birebir aynı
- [x] SynthTab Dev doğrulandı (`synthtab_dev` önbelleği, 4 naylon parça)
- [x] 3.8a GAPS indirme + önbellek (231 train / 30 test, disjoint)
- [x] 3.8b alan farkı ölçümü
- [x] 3.8c GAPS ince ayar → `tabcrnn_gaps.pt` kabul
- [ ] 3.8d SynthTab (Dev Set ile hat doğrulaması → nylon alt küme)
- [ ] 3.8e partisyon TAB'ı → tel etiketi
