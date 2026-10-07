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
- Tel tarafı (gerekirse): A teli 0.66; adaylar 36 bin/oktav önbellek, hex öğretmen
  damıtması, SynthTab naylon + GAPS partisyon TAB etiketleri (`.match`).
- **Denenmiş, işe yaramayan (tekrarlama):** el yapımı Viterbi hareket maliyeti;
  öğrenilen geçiş önseli güçlü modellerde katkısız; CQT perde kaydırmalı çoğaltma (−0.06);
  "teli vuruş anından seç" hipotezi (çürütüldü); kısa pencere girişi / hop 256 (ölçüldü: tekrar ayrımı
  artmıyor, AUC ~0.89); GAPS'te offset eşiği (etiket bitişleri partisyon hizalı, seçilmiyor).
- Ölçülmüş bulgular: tel hatalarının ~%99'u komşu tel; polifonide darboğaz tel değil perde;
  sentetik naylon (SynthTab Dev, 4 parça) model için kolay (tab F1 0.73), GAPS gerçek kayıt zor.
