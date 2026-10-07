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
- **Katman 3.9** (`docs/devlog/README10.md`), dal `katman-3.9-tel-tinisi` (GitHub'da).
- **En iyi model: `checkpoints/tabcrnn_onset_h.pt`** (TabCRNN + onset kafası +
  harmonik istifleme; tabcrnn_gaps.pt'den ince ayar). GuitarSet solo tab F1 0.739,
  oracle tel doğruluğu 0.850, GS val nota F1 0.909, val_comp nota F1 0.724.
- Gelişim: 0.586 (TabCNN) → 0.634 (CRNN) → 0.667 (+comp) → 0.697 (+GAPS) → 0.739 (+onset + harmonik).
- **Katman 3.9 KAPANDI (6 Ekim 2026)**: alan-içi kalibrasyonla (`eval_pitch` varsayılanı
  `--select-split auto`: GuitarSet→val, GAPS→gaps_val) GAPS nota F1 taban 0.441 → **0.592**,
  val_comp nota F1 0.463 → 0.724. Tüm ölçütler sağlandı; yeni taban `tabcrnn_onset_h.pt`.
- **Katman 3.10 sürüyor** (`docs/devlog/README11.md`, dal `katman-3.10-polifoni-perde`):
  polifonide perde. Ortak değerlendirme çekirdeği `gtab/evaluation/pitch_eval.py`.
  - Adım 0 (diagnose_pitch) + Adım 1 (eğitimsiz çözümleme kuralları: refrakter, histerezis,
    yedek; `--criterion mix`) **ölçüldü** (6 Ekim). Test: GAPS nota/kare 0.588/0.641
    (önce 0.545/0.621), val_comp 0.730/0.800 (nötr), GS val 0.907/0.865. GAPS'te kazanç,
    GuitarSet akorlarda nötr; Adım 1 kabul ölçütü (val_comp kare +0.03) sağlanmadı.
  - Bulgu: kurallar kuyruk kaybını ve onset kaçınca düşen notaları düzeltti ama kurtarılan
    notalar kayık başlıyor ("onset zamanı kaymış" %10→%21 / %11→%29). Akor "aynı perde"
    hayaletleri çift tetik değil, onset zamanlama hatası. GAPS hayaletlerinin %70'i yanlış perde.
  - Adım 1 kararı (7 Ekim): GS solo tab F1 0.743 (taban 0.739) → "kurallar + mix" varsayılan.
  - **Adım 2 ölçüldü (7 Ekim):** `tabcrnn_poly.pt` (init onset_h; `--onset-soft 0.3 --gs-pitch-weight 1.0
    --short-frames 5 --short-weight 2.0`). Kurallar + mix ile: GAPS 0.582/0.656, val_comp 0.744/0.817,
    GS val 0.905/0.869 (taban C: 0.588/0.641, 0.730/0.800, 0.907/0.865). Recall düzeldi ("hiç duyulmayan
    perde" −%32), **darboğaz artık precision**: hayaletler arttı (val_comp P 0.69, GAPS P 0.49).
    2a: @100ms farkı küçük (+0.006…+0.022) → hata küçük kayma değil, muhtemelen nota parçalanması;
    GAPS `.match` kalite ölçüsü olarak güvenilmez, etiket gürültüsü kanıtı yok.
    Çıkış ölçütü sağlanmadı; katman sürüyor.
  - GS solo tab F1 (tabcrnn_poly): greedy 0.748, Viterbi 0.753 → **tabcrnn_poly.pt yeni çalışma tabanı**.
  - **Adım 3 kodlandı, kullanıcının tam ölçümü bekleniyor** (komutlar README11): 3a diagnose_pitch
    bölüm D (hayalet alt kırılımı) — duman testinde akor hayaletlerinin %78'i fragman (nota sürerken
    zayıf onset notayı bölüyor, boşluk 0 kare); 3b `reattack` kuralı (nota sürerken yeni onset ancak
    tepe ≥ eşik ise yeni nota), kalibrasyonda {0.3…0.7} aranır; eval_hmm `--reattack`.
    Sonraki aday (gerekirse): onset pos_weight 3→1, offset kafası.
- Tel tarafı (gerekirse): A teli 0.66; adaylar 36 bin/oktav önbellek, hex öğretmen
  damıtması, SynthTab naylon + GAPS partisyon TAB etiketleri (`.match`).
- **Denenmiş, işe yaramayan (tekrarlama):** el yapımı Viterbi hareket maliyeti;
  öğrenilen geçiş önseli güçlü modellerde katkısız; CQT perde kaydırmalı çoğaltma (−0.06);
  "teli vuruş anından seç" hipotezi (çürütüldü).
- Ölçülmüş bulgular: tel hatalarının ~%99'u komşu tel; polifonide darboğaz tel değil perde;
  sentetik naylon (SynthTab Dev, 4 parça) model için kolay (tab F1 0.73), GAPS gerçek kayıt zor.
