# CLAUDE.md — çalışma senaryosu

Bitirme projesi: ses → tel/perde → TAB (klasik gitar odaklı). Kullanıcıyla Türkçe konuş.
Dosyaların görevleri ve tüm geçmiş özet: `CONTRIBUTING.md`.

## Her oturumda
1. `inputs/` oku: `Klasik_Gitar_AI_Proje_Sunumu.pptx` = ana plan (yön buradan),
   `Katman*.docx` = katman karar belgeleri.
2. `docs/devlog/README*.md` oku: her katmanın ne yaptığı / sonucu / sıradaki adımı.
   En yüksek numaralı README = güncel katman.
3. Sunuma göre ilerle ama **katman modelinden şaşma**: her katman bir öncekini
   kırmadan eklenir; mevcut dosya/checkpoint'in üzerine yazma, yenisini ekle.
4. Yeni adım = yeni `docs/devlog/README<N>.md` (neden, ne yapıldı, testler,
   çalıştırma, karar kuralı, sonuçlar) + ana `README.md` devlog listesi/yol haritası.
5. Bir sonraki kademeye ancak README'deki "ciddi iyileşme" ölçütü sağlanınca geç;
   arkada boşluk bırakma. Önce ölç (hata analizi), sonra çözüm seç.

## Çalışma şekli
- Uzun eğitim/değerlendirmeyi kullanıcı kendisi çalıştırır: kodu yaz, küçük
  testlerle (duman testi, eski modelle birebir aynı sonuç kontrolü) doğrula,
  **komutları ver**. Sonuçları kullanıcı yapıştırır; yorumla, devlog'a işle,
  karar kuralına göre öner.
- Komutlar proje kökünden: `python -m scripts.<data|train|eval>.<ad>`.
  `--ckpt` çıplak ad → `checkpoints/` altında.
- Kod yapısı: kütüphane `gtab/` (core, data, models, decoding, evaluation;
  yollar `gtab/paths.py`), çalıştırılanlar `scripts/`. Script'ler birbirinden
  import etmez; paylaşılan kod `gtab/`'a gider.
- Git: her katman ayrı dal (`katman-<no>-<konu>`), `main`'e birleştirme kullanıcı
  kararı. Commit/push yalnızca kullanıcı isteyince.

## Değerlendirme kuralları (değiştirme)
- GuitarSet ana val: oyuncu 05 **solo** (`data/cache/val`) — tüm tab F1'ler bununla
  kıyaslanır; `val_comp` = aynı oyuncunun akorlu kayıtları (polifoni ölçümü).
- GAPS: `--disjoint` (icracı-ayrık); parametre/eşik seçimi gaps_train içinden ayrılan
  doğrulamada (`gtab.data.gaps.gaps_files`), **gaps_test sadece raporlama**.
- Tek seed farkları ±0.01 gürültü sayılır.

## Veri
- `data/` git'e girmez. Kaynaklar README'lerde; önbellek `scripts/data/*` ile üretilir.
- Disk ~73 GB boş: SynthTab'den yalnızca sınırlı naylon (`luthier_*`) alt küme.

## Güncel durum (7 Ekim 2026 itibarıyla)
- **Kullanıcı hedefi: solo fingerstyle** (klasik gitar; tek ve çok ses; hızlı/yavaş art arda notaların
  ayrılması; ileride ritim). Ana polifoni ölçütü GAPS; GuitarSet akorları "kabul edilebilir" sayıldı.
  Kalıcı çözüm istiyor (sürekli küçük sürümler değil).
- **Güncel taban: `checkpoints/tabcrnn_rep_off.pt`** (TabCRNNOnset + harmonik istifleme + offset kafası;
  tabcrnn_poly'den `train_onset --repeat-weight 3 --pitch-weight 2 --offset-weight 1`).
  GS solo tab F1 **0.760**, oracle tel 0.862, GAPS test nota/kare **0.689/0.675**, val_comp 0.769/0.817,
  GS val 0.915/0.871. Çözümleme (val seçimi): onset@0.2, eşik 0.7, refrakter 6, yedek 10, tepe,
  yeniden_vuruş 0.4, enerji_kabul 6, offset 0.5 (GAPS: onset@0.1, eşik 0.3, histerezis 0.5, enerji 6/8, offset yok).
- Gelişim (tab F1): 0.586 TabCNN → 0.634 CRNN → 0.667 +comp → 0.697 +GAPS → 0.739 +onset+harmonik →
  0.743 kurallar → 0.748 poly → **0.760 rep_off**. GAPS nota F1: 0.441 → 0.592 → 0.668 → 0.689.
- **Katman 3.10** (`docs/devlog/README11.md`, dal `katman-3.10-polifoni-perde`): fingerstyle (GAPS) çıkış
  ölçütü sağlandı, val_comp ölçütü (0.780/0.830) sağlanmadı → **kapanış kullanıcı kararı bekliyor.**
  Açık kalan: <100 ms aynı perde tekrarları (GAPS'te %75 kaçıyor), GAPS hızlı tekrar kaçma %38.6.
- Katman 3.10'da yapılanlar: çözümleme kuralları (histerezis, refrakter, yedek, tepe, yeniden vuruş),
  CQT enerji yükselişi kuralları (`energy_rise`, `rise_keep/split`), keskin onset + kısa nota ağırlığı,
  hızlı tekrar onset ağırlığı, offset kafası; tanılama `diagnose_pitch` (A–E), `diagnose_onset_features`.
- **Katman 3.11 sürüyor** (`docs/devlog/README12.md`; plan kullanıcıdan geldi): hızlı tekrarlar —
  yerel tepe çözümlemesi + noisy-OR + perde-onset. **Kaldığımız yer (8 Ekim):** Adım 0 (diagnose_pitch
  bölüm F, tepe görünürlüğü), Adım 1 (`pick_peaks`, `combine`, `calibrate_peaks`; `--no-peak-search` = 3.10)
  ve Adım 2 ölçüm satırları (`cqt_decay`, `hf_flux`) kodlandı + regresyon HEAD ile birebir; **kullanıcının
  ölçümü bekleniyor** (komutlar README12 §8a). Adım 3 (val_comp seçim parçası) kullanıcı kararı;
  Adım 4 eğitim yalnız <100 ms görünürlük < %60 ise. 3.10 kapanışı: kullanıcı README12'yi 3.11 olarak açtı.
  **Ölçüldü (8 Ekim):** Adım 0 <100 ms görünürlük %66.6 → çözümleme dalı (eğitim yok). Adım 2 özellikleri
  GAPS'te CQT'nin altında → alınmadı. Adım 1: GAPS nota 0.689 → **0.736** (precision) ama tekrar kaçma
  kötüleşti (yeniden_vuruş 0 + refrakter 6 + enerji 6 kapıları yerel tepe adaylarını eliyor). GS tab F1 0.760
  (nötr) → Adım 1 GAPS için kabul. **Sırada: Adım 1b** (vadi kanıtlı kapılar + ortak arama + tekrar odaklı
  aday seçimi; README12 §8b). **Adım 1b kodlandı (9 Ekim):** `re_valley`, `calibrate_repeats` (REP_GRID, `--rep-tol 0.01`),
  eval_pitch tekrar kaçma sütunu; testler + regresyon ✓ → **kullanıcı ölçümü bekleniyor** (README12 §8c).
  **1b ölçüldü (9 Ekim):** GAPS 0.733/0.676, tekrarlar değişmedi — vadi kanıtı precision'ı çökertiyor (fragman vadisi
  = gerçek tekrar vadisi) → eğitimsiz yol GAPS için tükendi; GS tab F1 0.760 (nötr) → 1b kabul; **sıradaki: Adım 4 eğitim** (kullanıcı onayı bekleniyor).
  val_comp tekrar kaçma <100 ms 72.5 → 22.5 ama GS val nota −0.014. `.venv`'e CUDA torch kuruldu (requirements.txt cu121).
- Tel tarafı (gerekirse): A teli 0.66; adaylar 36 bin/oktav önbellek, hex öğretmen
  damıtması, SynthTab naylon + GAPS partisyon TAB etiketleri (`.match`).
- **Denenmiş, işe yaramayan (tekrarlama):** el yapımı Viterbi hareket maliyeti;
  öğrenilen geçiş önseli güçlü modellerde katkısız; CQT perde kaydırmalı çoğaltma (−0.06);
  "teli vuruş anından seç" hipotezi (çürütüldü); kısa pencere girişi / hop 256 (ölçüldü: tekrar ayrımı
  artmıyor, AUC ~0.89); GAPS'te offset eşiği (etiket bitişleri partisyon hizalı, seçilmiyor).
- Ölçülmüş bulgular: tel hatalarının ~%99'u komşu tel; polifonide darboğaz tel değil perde;
  sentetik naylon (SynthTab Dev, 4 parça) model için kolay (tab F1 0.73), GAPS gerçek kayıt zor.

# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
