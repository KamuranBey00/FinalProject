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

## Güncel durum (10 Ekim 2026 itibarıyla)
- **Kullanıcı hedefi: solo fingerstyle** (klasik gitar; tek ve çok ses; hızlı/yavaş art arda notaların
  ayrılması; ileride ritim). Ana polifoni ölçütü GAPS; GuitarSet akorları "kabul edilebilir" sayıldı.
  Kalıcı çözüm istiyor (sürekli küçük sürümler değil).
- **GÜNCEL TABAN (10 Ekim): `checkpoints/tabcrnn_deep2.pt`** + `tabcrnn_deep2.cal.json` (Katman 3.12 kapandı,
  dal `katman-3.12-akustik`). GS solo tab F1 **0.777 / 0.777**, oracle tel 0.878, GAPS test nota/kare **0.779 / 0.701**,
  GS val 0.924 / 0.879, val_comp 0.807 / 0.827; gaps_fit 0.784 ≈ gaps_val 0.780. Çözümleme perde-onset kafasından
  (`--onset-source pitch`). Açıklar: GAPS <100 ms tekrarlar, "perde hiç aktif değil" kaçanlar (%36), akor fragmanları.
- **Katman 3.13 sürüyor (10 Ekim, `docs/devlog/README14.md`; kullanıcı isteği, Katman 4'ten önce):** GAPS partisyon
  TAB'ı + parmak numarası → tel etiketi (README9 3.8e). `gtab/data/gaps_score.py` (MusicXML + tekrar açma + syncpoint
  + MIDI eşleme), `scripts/data/build_gaps_tab.py` → `data/cache/gaps_tab/` (165 kayıt, %81 nota etiketli, perde
  uyumu %98.9, parmak bağlama %99.4), `scripts/eval/eval_gaps_tab.py` (GAPS oracle tel + bilinen karede tab F1).
  deep2 GAPS oracle tel **0.610 < naif 0.702** (GuitarSet 0.878) → `train_onset --gaps-tab-weight 1` →
  `tabcrnn_gtab.pt` → **kabul edilmedi** (seçim hiç ep 0'ı geçmedi → dosya = deep2; gaps_val tab F1 +0.004, GS tab
  −0.035): partisyon TAB'ı icracıya genellenmiyor, ses kaybı olarak kullanma.
  **Adım 2 (kullanıcı planı; sunum modül 4 + 7): "klasik edisyon önerisi" = sembolik konum + sol el parmağı.**
  2a log-lineer (`gtab/decoding/edition.py`, `edition.npz`) yetmedi (parmak < taban). **2b iki yönlü dizi modeli
  (`gtab/models/edition_net.py`, `train_edition_seq`, `edition_seq.pt`) KABUL:** gaps_test tel 0.815 (naif 0.745),
  parmak 0.636 (taban 0.566), ses modu GuitarSet 0.878 değişmedi. Ürün iki mod: "duyulan konum" (ses) / "edisyon
  önerisi"; **karıştırılmaz** (kullanıcı kararı; karışım koddan çıkarıldı). Ürün parmak (konum + parmak tam) 0.434 >
  aynı konumda taban 0.376; uçtan uca (deep2 notaları, gaps_test) tel 0.802 bulunanlarda / 0.655 tüm etiketlilerde,
  tam parmak 0.397 / 0.331 (nota bulma %81.7). Ölçüm: `python -m scripts.eval.eval_edition` (varsayılanlar).
  Teknikler (Katman 4) sonra. Aşağıdaki maddeler geçmiş kayıttır.
- **Eski taban: `checkpoints/tabcrnn_rep_off.pt`** (TabCRNNOnset + harmonik istifleme + offset kafası;
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
  = gerçek tekrar vadisi) → eğitimsiz yol GAPS için tükendi; GS tab F1 0.760 (nötr) → 1b kabul. **Adım 4 kodlandı (9 Ekim):** `train_onset --valley-weight 5 --frag-weight 3
  --oversample 3` → `tabcrnn_rep_valley.pt` (init rep_off); varsayılan = HEAD birebir; **kullanıcı eğitimi bekleniyor**
  (README12 §8e). Sonrasında 3.11 kapanışı ve **Katman 4'e geçiş** (kullanıcı kararı).
  **Adım 4 ölçüldü (9 Ekim, README12 §8f):** `tabcrnn_rep_valley.pt` tab F1 **0.769 / 0.771** (en iyi), GAPS 0.720/0.684,
  val_comp 0.783/0.818, GS val 0.916/0.880; tekrar kaçma 100–200 ms −10 puan, <100 ms değişmedi (~%73–78).
  Kabul kuralı sağlanmadı (GAPS nota −0.013). 3.11 kapandı; taban **`tabcrnn_rep_valley.pt`**.
- **Katman 3.12 sürüyor** (`docs/devlog/README13.md`): akustik katman. Adım 0 kapasite ölçümü kodlandı
  (`eval_pitch/diagnose_pitch --load-cal checkpoints/tabcrnn_rep_valley.cal.json`, sanal split `gaps_fit`;
  kalibrasyonsuz ~1 dk). **Ölçüldü:** GAPS eğitim ≈ doğrulama (0.723 / 0.718); aşırı öğrenme testi
  (`scripts/train/overfit_gaps.py`; ana ölçüt ONSET F1, dropout kapalı, GuitarSet kontrolü; 8 kayıt, 200 epoch, ~4 dk):
  GuitarSet 0.995 (kurulum doğru), GAPS onset 0.40 → 0.95 → etiket tavanı yok, **kapasite → sırada 3a**
  (derin CNN + perde-onset kafası + vadi; kullanıcı onayı). Kare F1'i ana ölçüt yapma (GAPS nota bitişi partisyondan).
  **3a kodlandı (9 Ekim):** `--deep 2 --pitch-onset-weight 1 --new-lr-mult 3` (artık bloklar sıfır-gamma → birebir
  aynı başlangıç; perde-onset kafası; ayrı lr; kafa başına log); kalibrasyon kafa varsa tel/kafa iki zincir (~2 saat,
  `--save-cal` ile sonra hızlı). **Kullanıcı: 2 epoch deneme → 15 epoch `tabcrnn_deep.pt`** (README13 §3a). Ana hedef
  GAPS onset F1 (model düzeyi) ≥ +0.03; kapasite tekrar (fark > 0.05 → 3b).
  **tabcrnn_deep.pt (15 ep):** GAPS onset (kafa) 0.377 → **0.509**, GS tab 0.707, seçim 0.627. Sırada devam eğitimi
  `tabcrnn_deep2.pt` (`--lr 1e-4`); seç: deep2 kafa ≥ 0.519 ve GS tab ≥ 0.697 → deep2, değilse deep; sonra seçilende
  BİR KEZ kalibrasyon (`--save-cal`), kapasite, tanılama, tab F1 (README13).
  **deep2 seçildi (9 Ekim):** kafa 0.524, GS tab 0.716 (plato). Sırada tabcrnn_deep2'de kalibrasyon → kapasite → tanılama → tab F1.
  **a) kalibrasyon (10 Ekim, `tabcrnn_deep2.cal.json`):** iki alanda da perde-onset kafası seçildi; GAPS test nota/kare
  **0.779 / 0.701** (0.720 → +0.059), val 0.924 / 0.879, val_comp 0.807 / 0.827; <100 ms tekrar kaçma ~%76 (değişmedi).
  **b–d (10 Ekim):** kapasite farkı 0.004 (3b gerekmedi), gaps_val tekrar kaçma <100 ms 73 → 55, tab F1 0.777 →
  kabul ✓ → **3.12 KAPANDI**, taban `tabcrnn_deep2.pt` (README13 Karar).
  Hız: kalibrasyon seçimi değişmediyse her zaman `--load-cal` kullan (tam kalibrasyon ~1 saat).
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
