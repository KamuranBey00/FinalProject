# Katman 5.2 — Ürünleştirme (kendi kayıtlarımız beklenirken)

> **Durum (17 Ekim 2026):** U1, U2, U3 tamam. Dal `katman-5.2-urun` (`katman-6-zorlastirma` 78e69f1'den). Katman 6 kapanışı
> (kendi kayıtlarımız, README18 P1 / 0c) ve Katman 4 veri bekliyor; bu paket veri istemez, ölçüm tabanını (deep2) değiştirmez.
> Plan: `.omc/plans/urunlestirme.md`.

## Neden
Katman 6 son katman, Katman 4 teknik kayıtlara bağlı. Veri gelene kadar ürün tarafında açık kalan üç iş: şarkılı kayıt seçeneği
(README18 Adım 2b kabulü), `edition_seq_rec.pt`'nin ürüne alınması (README17 Ek A: "ikinci tohum + çalınabilirlik sonra"),
AlphaTab görüntüleyici (Katman 5'in açık kalemi).

## U1 — Şarkılı kayıt seçeneği
- `gtab/data/separate.py` `cqt_from_file_separated(path, mode)`; `scripts/app/transcribe.py` ve `transcribe_ascii.py`
  `--separate {none,mix-vocals}` (varsayılan none; `.npz` girdisinde hata).
- Bayraksız çıktı **birebir** (2 GAPS kaydı × 2 araç, 4 / 4 dosya bayt bayt aynı); 215 test OK; çıkarım regresyonu yok.
- Süre (60 sn ses): GPU 2.7 sn, CPU 50 sn. (Windows'ta CPU'ya zorlamak için `CUDA_VISIBLE_DEVICES=-1`.)
- Kalite: README18 (gaps_val) temiz +0.001, şarkı 10 dB +0.084, konuşma +0.076, şarkı 0 dB +0.243.
- **Açık karar (kullanıcı):** varsayılan mı olsun? Kalite açısından güvenli (temizde kayıp yok); bedel demucs bağımlılığı + CPU'da ~50 sn/dk.
```bash
python -m scripts.app.transcribe sarki.wav --meter 4/4 --bpm 90 --separate mix-vocals --out sarki.musicxml
```

## U2 — Parmak modeli (edisyon) ürüne
- `build_gaps_tab --recover-repeats` → `data/cache/gaps_tab_rec` (önbellek yeniden üretildi), `train_edition_seq --seed 2
  --tab-dir data/cache/gaps_tab_rec --out edition_seq_rec_s2.pt`. `eval_playability --seq` eklendi (varsayılan aynı).
- Ölçüm (`eval_edition`, seçim gaps_val; gaps_test yalnız rapor):
  | model | tel gaps_val | tel gaps_test | ürün parmak (konum+parmak) gaps_val / gaps_test | kesin parmak çakışması (akor oranı, gaps_val) |
  |---|---|---|---|---|
  | `edition_seq.pt` (eski ürün) | 0.766 | 0.815 | 0.382 / 0.434 | 0.092 |
  | `edition_seq_rec.pt` (tohum 1) | 0.784 | 0.827 | 0.413 / 0.454 | 0.099 |
  | **`edition_seq_rec_s2.pt` (tohum 2)** | **0.790** | 0.831 | **0.420** / 0.446 | **0.080** |
  (naif tel 0.702; editör çakışması 0)
- Kural (tel ve parmak iki tohumda da ≥ eski + 0.01, çalınabilirlik kötüleşmez): tel / parmak iki tohumda da sağlandı; çalınabilirlik
  tohum 1'de hafif kötü (+0.007), tohum 2'de iyi. gaps_val'de tohum 2 üç ölçüde de önde → **`edition_seq_rec_s2.pt` seçildi**:
  `check_stack --set edition=edition_seq_rec_s2.pt` → `checkpoints/active.json`, "YIGIN TUTARLI". Ses modu etkilenmez (GuitarSet 0.878).

## U3 — AlphaTab görüntüleyici
- `scripts/app/tab_viewer.html` (tek dosya, çift tıkla açılır; internet gerekir): `.musicxml` seç → alphaTab 1.8.4 (jsdelivr, sabit
  sürüm); görünüm nota + TAB / yalnız TAB / yalnız nota; çal / durdur, hız, yakınlaştırma; başlıkta akort, kapo, ölçü ve nota sayısı.
- Doğrulama: GAPS 001_mvswc (`--meter 3/4 --bpm 60`; standart, `--tuning DADGBE --capo 2`, `--mode edisyon`): alphaTab 70 ölçü /
  1377 nota = `transcribe` çıktısı (birebir); akort / kapo doğru; parmak numaraları nota satırında; başsız Chrome'da çizim ✓.
- Bilinenler: alphaTab dosyayı iki iz okur ("Gitar" nota + "TAB"); parmak numarası yalnız nota izinde ("yalnız TAB" görünümünde yok);
  çalarken TAB izi susturulur (çift ses olmasın). Parmak "0" = boş tel (bizim gösterimimiz). **Elle kontrol (kullanıcı):** sesin
  çalması / imlecin ilerlemesi, hız kaydırıcısı, drop-D + kapo dosyasının sesi.

## Durum
- [x] U1 şarkılı kayıt seçeneği (varsayılan olup olmaması kullanıcı kararı)
- [x] U2 `edition_seq_rec_s2.pt` aktif yığında
- [x] U3 TAB görüntüleyici (ses elle kontrol edilecek)
- [ ] Kendi kayıtlarımız gelince: README18 P1 → Katman 6 kapanışı → Katman 4
