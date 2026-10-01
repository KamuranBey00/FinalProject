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

## Güncel durum (1 Ekim 2026 itibarıyla)
- **Katman 3.9** (`docs/devlog/README10.md`), dal `katman-3.9-tel-tinisi` (GitHub'da).
- **En iyi model: `checkpoints/tabcrnn_onset_h.pt`** (TabCRNN + onset kafası +
  harmonik istifleme; tabcrnn_gaps.pt'den ince ayar). GuitarSet solo tab F1 0.739,
  oracle tel doğruluğu 0.850, GS val nota F1 0.909, val_comp nota F1 0.724.
- Gelişim: 0.586 (TabCNN) → 0.634 (CRNN) → 0.667 (+comp) → 0.697 (+GAPS) → 0.739 (+onset + harmonik).
- **Açık iş (kullanıcı onayı bekleniyor, sıradaki adım):**
  1. GAPS kalibrasyonu: `eval_pitch`'te GAPS eşiklerini GAPS doğrulamasında seç
     (şu an GuitarSet val'de seçiliyor → GAPS nota F1 0.407; aynı model eşik 0.5'te 0.588).
     README10'daki son iki ölçüt (GAPS nota F1 ≥ +0.05, hiçbir metrikte −0.01'den fazla
     kayıp yok) buna bağlı. Tutarsa tabcrnn_onset_h.pt yeni taban.
  2. Polifoni için perde tarafı: val_comp'ta perde bulunamayan hücre %23.5, hayalet nota
     yanlış pozitiflerin %75'i, tel hatası yalnızca %5. Önce diagnose_strings'e kaçan
     notaların süre/perde/polifoni kırılımı; sonra offset kafası / onset-koşullu kare
     kafası ya da GAPS onset kaybını güçlendirme.
  3. Tel tarafı (gerekirse): A teli 0.66; adaylar 36 bin/oktav önbellek, hex öğretmen
     damıtması, SynthTab naylon + GAPS partisyon TAB etiketleri (`.match`).
- **Denenmiş, işe yaramayan (tekrarlama):** el yapımı Viterbi hareket maliyeti;
  öğrenilen geçiş önseli güçlü modellerde katkısız; CQT perde kaydırmalı çoğaltma (−0.06);
  "teli vuruş anından seç" hipotezi (çürütüldü).
- Ölçülmüş bulgular: tel hatalarının ~%99'u komşu tel; polifonide darboğaz tel değil perde;
  sentetik naylon (SynthTab Dev, 4 parça) model için kolay (tab F1 0.73), GAPS gerçek kayıt zor.
