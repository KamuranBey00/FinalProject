# CLAUDE.md — çalışma senaryosu

Bitirme projesi: ses → tel/perde → TAB (klasik gitar odaklı). Kullanıcıyla Türkçe konuş.

## Her oturumda
1. `inputs/` oku: `Klasik_Gitar_AI_Proje_Sunumu.pptx` = ana plan (yön buradan),
   `Katman*.docx` = katman karar belgeleri.
2. `docs/devlog/README*.md` oku: her katmanın ne yaptığı / sonucu / sıradaki adımı.
   En yüksek numaralı README = güncel katman.
3. Sunuma göre ilerle ama **katman modelinden şaşma**: her katman bir öncekini
   kırmadan eklenir; mevcut dosya/checkpoint'in üzerine yazma, yenisini ekle.
4. Yeni adım = yeni `docs/devlog/README<N>.md` (neden, ne yapıldı, testler,
   çalıştırma, karar kuralı, sonuçlar) + ana `README.md` devlog listesi/yol haritası.

## Çalışma şekli
- Uzun eğitim/değerlendirmeyi kullanıcı kendisi çalıştırır: kodu yaz, küçük
  testlerle doğrula, **komutları ver**. Sonuçları kullanıcı yapıştırır; yorumla,
  devlog'a işle, karar kuralına göre öner.
- Komutlar proje kökünden: `python -m scripts.<data|train|eval>.<ad>`.
  `--ckpt` çıplak ad → `checkpoints/` altında.
- Kod yapısı: kütüphane `gtab/` (core, data, models, decoding, evaluation;
  yollar `gtab/paths.py`), çalıştırılanlar `scripts/`. Script'ler birbirinden
  import etmez; paylaşılan kod `gtab/`'a gider.

## Değerlendirme kuralları (değiştirme)
- GuitarSet ana val: oyuncu 05 **solo** (`data/cache/val`) — tüm tab F1'ler bununla kıyaslanır.
- GAPS: `--disjoint` (icracı-ayrık); parametre seçimi gaps_train içinden ayrılan
  doğrulamada, **gaps_test sadece raporlama**.
- Tek seed farkları ±0.01 gürültü sayılır.

## Veri
- `data/` git'e girmez. Kaynaklar README'lerde; önbellek `scripts/data/*` ile üretilir.
- Disk ~73 GB boş: SynthTab'den yalnızca sınırlı naylon (`luthier_*`) alt küme.
