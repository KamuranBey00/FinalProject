# archive/ — eski ve reddedilen kod (10 Ekim 2026)

Aktif hat (`gtab/`, `scripts/`) yalnızca güncel ses tabanını (`tabcrnn_deep2.pt`) ve "klasik edisyon önerisini"
(`edition_seq.pt`) üretir ve ölçer. Bu klasördeki dosyalar aktif kod tarafından **import edilmez**; geçmiş
katmanları yeniden üretmek ve devlog'lardaki sayıları doğrulamak için saklanır. Dosyalar `git mv` ile taşındı,
git geçmişi korunuyor.

## İçerik
| Dosya | Katman | Neden arşivde |
|---|---|---|
| `scripts/train/train_pitch.py` | 2 | PitchCNN (monofonik perde) — yerini tel/perde modelleri aldı |
| `scripts/train/train_tab.py` | 3 | TabCNN — yerini TabCRNN aldı |
| `scripts/train/train_crnn.py` | 3.5–3.7 | TabCRNN — yerini `train_onset` aldı |
| `scripts/train/train_domain.py` | 3.8 | GuitarSet + GAPS karışık eğitim — `train_onset` içinde |
| `scripts/train/overfit_gaps.py` | 3.12 Adım 0b | aşırı öğrenme (kapasite) testi — tamamlandı |
| `scripts/train/train_edition.py` | 3.13 Adım 2a | log-lineer edisyon modeli — **reddedildi** (parmak < taban) |
| `scripts/eval/sweep_threshold.py` | 2 | PitchCNN eşik taraması |
| `scripts/eval/eval_viterbi.py` | 3.6 | el yapımı Viterbi karşılaştırması — **işe yaramadı**; `eval_hmm` aynı metriği ölçüyor |
| `scripts/eval/diagnose_strings.py` | 3.7 | tel hata analizi — tamamlandı (hataların ~%99'u komşu tel) |
| `scripts/eval/check_gaps_labels.py` | 3.10 | GAPS `.match` güvenilirliği — tamamlandı (güvenilmez, kullanılmıyor) |
| `scripts/eval/diagnose_onset_features.py` | 3.10–3.11 | kısa pencere / hop 256 / yeni özellikler — **reddedildi** (AUC ~0.89, CQT'yi geçmedi) |
| `scripts/eval/eval_gaps_tab.py` | 3.13 Adım 0–1 | partisyon TAB'ı ses kaybı olarak — **reddedildi** (genellenmedi) |
| `scripts/data/build_synthtab.py`, `gtab/data/synthtab.py` | 3.8d / 3.12 3b | SynthTab naylon hattı — hiç gerekmedi (kapasite farkı 0.004); veri ihtiyacı çıkarsa geri alınır |
| `gtab/decoding/edition_loglinear.py` | 3.13 Adım 2a | log-lineer konum + parmak modeli ve Viterbi'si (reddedildi) |

Checkpoint'ler: `checkpoints/archive/` (pitchcnn, tabcnn, tabcrnn … rep_off, rep_valley, deep, deep_try,
gtab = deep2 kopyası, edition.npz). Gelişim sayıları ana `README.md` ve devlog'larda.

## Çalıştırma (gerekirse)
Arşiv bir Python paketi; proje kökünden:
```bash
python -m archive.scripts.eval.eval_viterbi --ckpt checkpoints/archive/tabcrnn_comp.pt
python -m archive.scripts.train.train_edition --out archive_edition.npz
```
Eski checkpoint'ler çıplak adla bulunmaz: `--ckpt checkpoints/archive/<ad>.pt` verin. Arşiv bakımı yapılmaz;
eski betikler sonraki katmanlardaki arayüz değişiklikleri yüzünden çalışmayabilir (git geçmişindeki kendi
katman dalları birebir çalışır: `katman-3.x-...`).

## Aktif dosyalarda kalan eski kod (bilerek taşınmadı)
Dosya içi fonksiyonlar aktif kodla iç içe olduğu için yerinde bırakıldı:
- `gtab/models/nets.py`: PitchCNN, TabCNN, TabCRNN (`load_model` eski checkpoint türlerini yükleyebilsin diye).
- `gtab/decoding/viterbi.py`: el yapımı `viterbi_assign` (`eval_hmm` teşhis bölümünde karşılaştırma satırı).
- `gtab/evaluation/pitch_eval.py` / `viterbi.py`: seçilmeyen çözümleme kuralları (`re_valley`, offset eşiği vb.);
  kalibrasyon araması bunları aday olarak dener, çıkarılırsa eski `.cal.json` seçimleri birebir üretilemez.
