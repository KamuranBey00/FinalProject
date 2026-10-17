# Katman 6 — Zorlaştırma: gerçek dünya (polifoni, kayıt koşulu, akort, kendi kayıtlarımız)

> **Durum (16 Ekim 2026):** Adım 0 ölçüldü (kayıt koşulu → Adım 2; polifoni tetiklenmedi). **Adım 2 kodlandı (kayıt koşulu +
> şarkı / konuşma artırması) → kullanıcının indirme + eğitim + ölçümü bekleniyor.** Katman 4 (teknikler)
> kendi teknik verimiz hazır olunca (kullanıcı kararı). Taban: aktif yığın (`check_stack`; `tabcrnn_deep2.pt`).
> Katman 6'da çıkan sorunlar bu katmanda çözülür.

## Neden
README1'deki tanım "polifoni, distortion dayanıklılığı, farklı gitar / akort". Distortion klasik gitar için anlamsız; yerine
sunumdaki **kayıt koşulu dayanıklılığı** (slayt 12: "dataset-quality mic vs USB mic vs telefon / oda kaydı") ve
**kendi test setimiz** (slayt 14 adım 6) konur. Bugüne kadarki bütün sayılar stüdyo kalitesinde GuitarSet / GAPS kayıtlarından;
ürünü kullanacak kişi telefonla, odada kaydedecek. Önce hangi eksenin en çok düşürdüğü ölçülür, çözüm ona göre seçilir.

## Adım 0 — Ölçüm (eğitim yok)
| Eksen | Nasıl | Araç |
|---|---|---|
| **0a Kayıt koşulu** | gaps_val ham sesine yapay bozulma: oda yankısı (RT60 0.5 / 1.2 sn), telefon (150 Hz – 7 kHz bant + otomatik kazanç sıkıştırması), pembe gürültü (SNR 20 / 10 dB); taban + kalibrasyon sabit, etiketler aynı | `gtab/data/degrade.py`, `eval_robust` |
| **0b Polifoni** | eşzamanlı nota sayısına göre (1 / 2 / 3 / 4+) kare F1; gaps_val + val_comp | `eval_pitch` (mevcut polifoni kırılımı), `eval_robust` |
| **0c Kendi kayıtlarımız** | gerçek telefon / mikrofon / oda, bizim gitarımız; etiket egzersiz tablosundan + hizalama | `docs/KAYIT_PROTOKOLU.md`; araç (`build_own`) kayıtlar gelince |
| **0d Akort** | drop D (E14) kendi kayıtlarımızda; GAPS scordatura kayıtları | 0c ile |

Notlar:
- Bozulmalar uzunluğu korur, deterministik (kayıt başına tohum). "temiz" satırı eval_pitch gaps_val ile aynı olmalı.
- MP3 / düşük bit hızı sıkıştırması ffmpeg gerektirdiği için yok; telefon koşulu bant sınırı + sıkıştırmayla yaklaşık.
- Ölçüm gaps_val'de (seçim alanı); gaps_test yalnız katmanın son raporunda.

### Duman testi (2 kayıt, gaps_val; yalnız kod doğrulaması, karar için değil)
| koşul | nota F1 | fark |
|---|---|---|
| temiz | 0.835 | — |
| yankı 0.5 sn | 0.777 | −0.058 |
| yankı 1.2 sn | 0.680 | −0.155 |
| telefon | 0.802 | −0.033 |
| gürültü 20 dB | 0.796 | −0.038 |
| gürültü 10 dB | 0.715 | −0.120 |

### Çalıştırma
```bash
python -m scripts.eval.check_stack                                             # yığın tutarlı mı
python -m scripts.eval.eval_robust                                              # 0a + 0b (gaps_val, ~5 dk)
python -m scripts.eval.eval_pitch --load-cal checkpoints/tabcrnn_deep2.cal.json --splits gaps_val val val_comp   # 0b
```

### Karar kuralı (Adım 0 → sıradaki adım)
- **Kayıt koşulu:** gerçekçi koşullardan (yankı 0.5 sn, telefon, gürültü 20 dB) herhangi birinde nota F1 düşüşü > 0.03 →
  **Adım 2 (dayanıklılık eğitimi)** öncelikli. Ağır koşullar (yankı 1.2 sn, gürültü 10 dB) yalnız raporlanır.
- **Polifoni:** 3 ve 4+ seste kare F1, tek sese göre > 0.05 düşükse (gaps_val ya da val_comp) → **Adım 1 (polifoni)**.
- İkisi de sağlanırsa büyük düşüş önce. Hiçbiri yoksa → doğrudan Adım 3 (akort) + 0c gerçek dünya raporu.
- 0c geldiğinde: kendi kayıtlarımızdaki düşüş yapay bozulmanın gösterdiğiyle tutarlı mı (yapay bozulma gerçeği temsil ediyor mu)?

## Adım 2 — Kayıt koşulu + insan sesi dayanıklılığı (16 Ekim; kullanıcı kararı: şarkı / konuşma da bu adımda)

**Neden:** Adım 0'da gerçekçi üç koşulun üçü de > 0.03 düşürdü. Kullanıcı sorusu: gitarist çalarken şarkı söylerse / dışarıda
konuşma olursa model gitar sesini ayırabilir mi? Model hiç insan sesi duymadı; şarkıcının perdeli sesi hayalet nota üretir.
Seçenekler: (1) eğitim artırması (seçildi: ürüne ek model girmez, aynı eğitimde çözülür), (2) kaynak ayırma ön adımı (Demucs
6 kanal "gitar"; ağır, ayırma bozulması notaları bozabilir; sunumda ilk sürüm kapsam dışı) — 1 yetmezse ayrı alt adım (2b),
aynı ölçümle karşılaştırılır.

**Ne yapıldı:**
- `scripts/data/get_voices.py`: VocalSet (şarkı, 20 şarkıcı, zenodo 1193957, 2.1 GB, CC BY 4.0) + LibriSpeech dev-clean
  (konuşma, 40 konuşmacı, 0.34 GB, CC BY 4.0) arşivden doğrudan okunur → `data/raw/voices/{sing,speech}_{train,eval}.npy`
  (22050 Hz, sessizlik atılmış, 40 + 10 dk; **şarkıcı / konuşmacı ayrık**: eval = female8-9, male10-11 + son 8 konuşmacı);
  arşiv silinir (disk dar).
- `gtab/data/degrade.py`: `voice` (banka kesiti, gitar / ses oranı dB), ölçüm koşulları `sarki_10dB`, `sarki_0dB`,
  `konusma_10dB` (eval bankası); eğitim için `random_chain` (1–2 bozulma: yankı RT60 0.2–1.5 sn, telefon bandı 80–300 Hz /
  4–9 kHz, gürültü 5–30 dB, şarkı / konuşma −3…20 dB) + `apply_chain`.
- `scripts/data/build_degraded.py` → `data/cache/degraded/<k>/<split>/<id>.npz` (GuitarSet train + train_comp mikrofon sesi,
  gaps_train; kayıt başına 2 rastgele zincir; gaps doğrulama kayıtlarında eval bankası). Etiketler aynı.
- `GuitarSetSeq` / `PitchSeq` `alt_root, alt_p, alt_fixed`: parça `alt_p` olasılıkla bozulmuş kopyayla (doğrulamada hep kopya 0).
  `train_onset --degrade-root --degrade-p`: seçim skoruna **bozulmuş GAPS doğrulama onset F1'i** eklenir (`GAPS-onsetF1(bozuk)`).
  Bayraksız = eski eğitim birebir.
- `eval_robust`: ses koşulları (banka varsa), `--ckpt` / `--cal` (aday model).

**Testler:** sahte bankayla duman testi (build_degraded 4 kayıt; temiz parça eskisiyle birebir, bozuk kopya farklı, boyut aynı);
`train_onset --epochs 0 --degrade-root` → ep 0 GS-tabF1 0.716, GAPS kafa 0.524 (deep2 değerleri birebir) + bozuk ölçüsü seçimde;
204 test OK, çıkarım regresyonu yok.

**Disk:** C: boş alan ~4.7 GB (CLAUDE.md'deki 73 GB eski). get_voices zirvesi ~2.7 GB (arşiv silinince 0.27 GB), build_degraded
2 kopya ~1.5–2 GB. Sığıyor ama dar; mümkünse önce yer aç (≥ 10 GB önerilir) ya da `--copies 1`.

### Çalıştırma (sırayla)
```bash
python -m scripts.data.get_voices                                   # indirme + ~10 dk
python -m scripts.eval.eval_robust                                   # deep2 tabanı, 9 koşul (ses koşulları dahil) ~8 dk
python -m scripts.data.build_degraded                                # 2 kopya, ~1 saat
python -m scripts.train.train_onset --init tabcrnn_deep2.pt --epochs 10 --ckpt tabcrnn_robust.pt --lr 1e-4 --onset-soft 0.3 --gs-pitch-weight 1.0 --short-frames 5 --short-weight 2.0 --repeat-weight 3 --pitch-weight 2 --offset-weight 1 --valley-weight 5 --frag-weight 3 --oversample 3 --deep 2 --pitch-onset-weight 1 --new-lr-mult 3 --degrade-root data/cache/degraded --degrade-p 0.5
python -m scripts.eval.eval_robust --ckpt tabcrnn_robust.pt --cal tabcrnn_deep2.cal.json
python -m scripts.eval.eval_hmm --model crnn --ckpt tabcrnn_robust.pt --load-cal checkpoints/tabcrnn_deep2.cal.json
```
(Eğitim deep2 tarifiyle aynı, yalnız `--init tabcrnn_deep2.pt --epochs 10 --degrade-root` farklı. Aday önce deep2'nin
kalibrasyon kartıyla ölçülür; kabul edilirse kendi kartı `eval_pitch --save-cal` ile çıkarılır, ~1 saat.)

### Karar kuralı (Adım 2) — Adım 0'dan sonra düzeltildi
Adım 0'da gerçekçi koşulların kaybı yalnız 0.03–0.04 olduğundan "bozulmuşta +0.05" ölçütü bu koşullar için imkânsız (temizin
üstüne çıkılamaz). Yerine:
- **Gerçekçi koşullar** (yankı 0.5 sn, telefon, gürültü 20 dB): temize göre düşüş **yarıya iner ve ≤ 0.02**.
- **İnsan sesi** (`sarki_10dB`, `konusma_10dB`): deep2'ye göre **≥ +0.05** (tabanı önce ölçülür; düşüş zaten < 0.05 ise
  ölçüt "düşüş ≤ 0.02").
- **Ağır koşullar** (yankı 1.2 sn, gürültü 10 dB, `sarki_0dB`): ≥ +0.03.
- **Korunacaklar:** temiz gaps_val nota F1 kaybı ≤ 0.01; GuitarSet tab F1 (eval_hmm) kaybı ≤ 0.01 (tohum gürültüsü 0.009).
- Kabul → kendi kalibrasyonu + `check_stack --set acoustic=tabcrnn_robust.pt cal=tabcrnn_robust.cal.json` → yazdığı zincir
  (BeatNet önbelleği + yeniden eğitim, parmak izi). İnsan sesi ölçütü tutmazsa → Adım 2b (kaynak ayırma) ölçülür.

## Sonraki adımlar (Adım 0 sonucuna göre)
- **Adım 1 — Polifoni (akorlar):** hata analizi (kaçan perde / parçalanma / tel çakışması) → akor farkında çözümleme,
  gerekirse SynthTab naylon ön eğitimi; MusicXML'de iki ses (bas / melodi). Kabul: 3+ seste nota F1 ≥ +0.03, tek seste kayıp ≤ 0.01.
- **Adım 2 — Kayıt koşulu:** `degrade` ile eğitim artırması (`train_onset`, deep2'den ince ayar, yeni checkpoint).
  Kabul: bozulmuş gaps_val ≥ +0.05, temiz gaps_val ve GuitarSet kaybı ≤ 0.01. Yeni taban seçilirse `check_stack`'in yazdığı
  zincir (kalibrasyon, BeatNet, parmak izi) çalıştırılır.
- **Adım 3 — Akort / kapo:** kullanıcı girdisi (ör. 6. tel D, kapo k) → `Instrument` akortu pipeline, geçiş modeli,
  edisyon ve MusicXML'e; isteğe bağlı sesten tahmin. Kabul: scordatura kayıtlarında tel doğruluğu standart akortla ± 0.02.
- **Adım 4 — Gerçek dünya raporu:** son modelle kendi kayıtlarımızda uçtan uca (nota, tel, ritim, MusicXML) = katman kapanışı.

## Sonuçlar (16 Ekim, kullanıcı çalıştırması; gaps_val 28 kayıt)
| koşul | nota P | R | F1 | fark | kare F1 | polifoni kare F1 1 / 2 / 3 / 4+ | <100 ms kaçma |
|---|---|---|---|---|---|---|---|
| temiz | 0.773 | 0.788 | **0.780** | — | 0.688 | 0.63 / 0.71 / 0.71 / 0.73 | %54.7 |
| yankı 0.5 sn | 0.745 | 0.739 | 0.742 | **−0.038** | 0.668 | 0.60 / 0.70 / 0.69 / 0.71 | %59.5 |
| yankı 1.2 sn | 0.667 | 0.672 | 0.669 | −0.111 | 0.640 | 0.56 / 0.67 / 0.66 / 0.69 | %60.7 |
| telefon | 0.696 | 0.783 | 0.737 | **−0.043** | 0.675 | 0.61 / 0.70 / 0.70 / 0.72 | %53.0 |
| gürültü 20 dB | 0.720 | 0.780 | 0.749 | **−0.031** | 0.659 | 0.57 / 0.67 / 0.70 / 0.72 | %57.5 |
| gürültü 10 dB | 0.635 | 0.739 | 0.683 | −0.097 | 0.609 | 0.49 / 0.63 / 0.65 / 0.69 | %68.0 |

- Tutarlılık ✓: "temiz" = eval_pitch gaps_val (0.780 / 0.688, polifoni birebir).
- **Kayıt koşulu: üç gerçekçi koşulun üçü de > 0.03 düşüş → Adım 2 tetiklendi.** Telefon ve gürültü daha çok **precision**
  düşürüyor (hayalet nota: 0.773 → 0.696 / 0.720, recall aynı); yankı **recall** düşürüyor (sönümle notalar birbirine karışıyor).
- **Polifoni: tetiklenmedi.** 3 / 4+ seste kare F1 tek sesten düşük değil, yüksek (gaps_val 0.63 → 0.73; val_comp 0.72 → 0.88).
  Tek sesli karelerin düşüklüğü nota bitişi etiketinden (GAPS bitişleri partisyondan, README13) — polifoni sorunu değil.
- **Karar: sıradaki Adım 2 (kayıt koşulu dayanıklılığı).** Kullanıcı sorusu (16 Ekim): şarkı söyleme / dış sesler → aynı
  adımda vokal + konuşma koşulu olarak ele alınır (aşağıda).

## Adım 2 sonuçları — ilk deneme (17 Ekim, kullanıcı çalıştırması)

**İnsan sesi tabanı (deep2, gaps_val 28 kayıt; diğer 6 satır Adım 0 ile birebir):**
| koşul | nota P | R | F1 | fark | kare F1 | polifoni 1 / 2 / 3 / 4+ | <100 ms kaçma |
|---|---|---|---|---|---|---|---|
| şarkı 10 dB | 0.657 | 0.714 | 0.685 | **−0.096** | 0.586 | 0.57 / 0.63 / 0.59 / 0.60 | %59.7 |
| şarkı 0 dB | 0.495 | 0.525 | 0.510 | −0.271 | 0.431 | 0.43 / 0.47 / 0.42 / 0.44 | %73.6 |
| konuşma 10 dB | 0.649 | 0.738 | 0.690 | **−0.090** | 0.626 | 0.61 / 0.68 / 0.63 / 0.63 | %62.8 |

- İnsan sesi, gerçekçi kayıt koşullarının (−0.03…−0.04) 2–3 katı düşürüyor; hem P hem R düşüyor. Şarkıda 3 / 4+ sesli kareler
  tek sesten fazla düşüyor (0.71 → 0.59): şarkıcının perdeli sesi akor içine hayalet nota ekliyor. Ölçüt (≥ +0.05) geçerli.

**Eğitim (`--degrade-p 0.5`, 10 ep, deep2'den):** seçim hiç ep 0'ı geçmedi → **`tabcrnn_robust.pt` = deep2 kopyası**
(en iyi skor 0.591 = ep 0; ep 10 0.577). Bu dosyayla eval_robust / eval_hmm deep2'yi tekrar ölçer.
| | ep 0 (deep2) | ep 1 | ep 10 |
|---|---|---|---|
| GS tab F1 | 0.716 | 0.676 | 0.674 (**−0.042**) |
| GAPS onset (kafa, temiz) | 0.524 | 0.487 | 0.506 (−0.018) |
| GAPS onset (bozuk) | 0.412 | 0.404 | 0.430 (**+0.018**, hâlâ artıyor) |

- Bozuk kazanç küçük, temiz kayıp büyük → kural sağlanmadı. Temiz kaybın büyük kısmı **ilk epoch'ta** (GS tab −0.040): artırmadan
  mı, yoksa deep2'den yeniden başlayan eğitimin kendisinden mi (yeni Adam durumu, yeni katman lr ×3) ayrılamıyor →
  **önce kontrol koşusu** (aynı tarif, `--degrade-root` yok, 2 ep). Kontrol de düşerse sorun ince ayar tarifi; düşmezse artırma
  oranı (`--degrade-p 0.2`) / süre.
- **Kontrol koşusu (17 Ekim, `tabcrnn_ctrl.pt`, artırmasız 2 ep):** GS tab 0.716 → 0.718 / 0.712, GAPS kafa 0.524 → 0.526 / 0.531;
  tel kaybı 0.067 (artırmalıda 0.298). **Düşüş artırmadan geliyor**, ince ayar tarifinden değil → sıradaki: `--degrade-p 0.2`.
- **İkinci eğitim (17 Ekim, `--degrade-p 0.2`, 15 ep, `tabcrnn_robust2.pt`):** seçim yine ep 0'ı geçmedi (0.591) → dosya = deep2.
  En iyi epoch'larda GS tab 0.699–0.700 (−0.017), GAPS kafa temiz 0.510–0.511 (−0.013), bozuk 0.423–0.425 (+0.011…0.013).
  Payı 0.5 → 0.2 indirmek kaybı yarıya indirdi ama takası değiştirmedi (temiz kayıp ≈ bozuk kazanç).
- **T3 ayrıştırma (günlükten):** temiz GAPS onset'i **perde-onset kafasından** ölçülüyor (tel kaybı ona girmiyor) ve o da −0.013 düşüyor →
  kayıp yalnız tel değil → **P0 (tel kaybı maskesi) uygulanmaz** (plan kuralı). Bozulmuş kopyaların %53'ünde insan sesi var
  (1062 kopyanın 563'ü; zincirde ses iki kez havuzda, −3…20 dB) — insan sesi artık `mix-vocals` ile çözüldü, artırmanın en sert
  parçası gereksiz.
- **Üçüncü eğitim (17 Ekim, sessiz artırma: `build_degraded --no-voices --out degraded_nv`, `--degrade-p 0.2`, 15 ep,
  `checkpoints/archive/tabcrnn_robust3.pt`):** seçim ep 4 (0.598 vs ep 0 0.597 — gürültü içinde). Ölçüm (deep2 kalibrasyonuyla):
  | | deep2 | robust3 | fark |
  |---|---|---|---|
  | temiz | 0.780 | 0.779 | −0.001 ✓ |
  | yankı 0.5 sn (düşüş) | −0.038 | −0.035 | yarıya inmedi ✗ |
  | telefon (düşüş) | −0.043 | −0.030 | ≤ 0.02 değil ✗ |
  | gürültü 20 dB (düşüş) | −0.031 | −0.026 | ✗ |
  | yankı 1.2 sn / gürültü 10 dB | 0.669 / 0.683 | 0.677 / 0.702 | +0.008 / +0.019 (< +0.03) ✗ |
  | GS tab F1 (eval_hmm) | 0.777 | **0.756** | **−0.021** ✗ (oracle tel 0.878 → 0.864) |
  **Kabul edilmedi.** Sessiz artırma kazancı gerçek ama küçük (telefon +0.012, gürültü 10 dB +0.019), GuitarSet tel kaybı sınırın
  iki katı. **Kullanıcıyla anlaşılan kural gereği dayanıklılık eğitimi yolu KAPANDI**; taban deep2 kalır. Kayıt koşulu kaybı
  (~−0.03…−0.04) kendi kayıtlarımızla (0c) gerçekte bu kadar mı diye ölçülecek; gerçek kayıtta büyükse yeniden açılır
  (aday: robust3 + kendi kalibrasyonu, ya da GuitarSet'te bozulmasız tel kaybı).

## Adım 2b — Kaynak ayırma ölçümü (17 Ekim, kodlandı; plan `.omc/plans/katman6-kalan-adimlar.md` P2)
- `gtab/data/separate.py`: `separate(y, sr, mode)`, Demucs `htdemucs_6s`; `mode` = `guitar` (gitar kanalı) | `mix-vocals` (karışım −
  vokal kanalı). Çıktı uzunluğu girdiyle aynı. `eval_robust --separate {guitar,mix-vocals}` (bozulma → ayırma → CQT); bayraksız birebir.
- Kurulum: `demucs 4.1.0` (requirements.txt'te isteğe bağlı yorum); torch 2.3.1+cu121 / numpy 1.26.4 değişmedi. Süre GPU'da
  ~3 sn / kayıt-koşul (+ bir kez ~22 sn model yükleme).
- Duman (2 kayıt, karar için değil): nota F1 temiz / şarkı 10 dB — ayırmasız 0.835 / 0.738, `mix-vocals` 0.836 / 0.809,
  `guitar` 0.803 / 0.789 (temizde −0.03).
- **Çalıştırma:** `python -m scripts.eval.eval_robust --separate mix-vocals` ve `--separate guitar` (9 koşul, 28 kayıt).
- **`guitar` ölçüldü (17 Ekim, 28 kayıt; nota F1, ayırmasız deep2'ye göre):** temiz 0.749 (**−0.031**, R 0.788 → 0.740), yankı 0.5 sn
  −0.033, telefon −0.014, gürültü 20 dB −0.020; şarkı 10 dB 0.735 (+0.050), konuşma 10 dB 0.732 (+0.042), şarkı 0 dB 0.714 (**+0.204**).
  İnsan sesi varken çok güçlü, yokken gitarın bir kısmını atıyor (recall) → **koşulsuz kullanım reddedildi**.
- **`mix-vocals` ölçüldü (17 Ekim, 28 kayıt):**
  | koşul | ayırmasız | mix-vocals | fark |
  |---|---|---|---|
  | temiz | 0.780 | 0.781 | +0.001 |
  | yankı 0.5 sn / telefon / gürültü 20 dB | 0.742 / 0.737 / 0.749 | 0.743 / 0.738 / 0.750 | +0.001 |
  | yankı 1.2 sn / gürültü 10 dB | 0.669 / 0.683 | 0.670 / 0.683 | ≈ 0 |
  | şarkı 10 dB | 0.685 | **0.769** | **+0.084** |
  | konuşma 10 dB | 0.690 | **0.766** | **+0.076** |
  | şarkı 0 dB | 0.510 | **0.753** | **+0.243** |
  **Bütün ölçütler sağlandı → `mix-vocals` KABUL.** Karışımdan yalnız vokal kanalı çıkarıldığı için gitar sesi korunuyor (temizde kayıp
  yok); vokal kapısı gereksiz (planlanan kapı kodlanmadı). Kayıt / yankı / gürültü dayanıklılığı ayırmayla çözülmüyor (beklenen;
  Adım 2 eğitimi o eksen için). Ürüne alma (varsayılan mı seçenek mi, CPU süresi, gaps_test) ayrı karar.
- **Karar (koşul başına, ayırmasız deep2'ye göre):** şarkı 10 dB ve konuşma 10 dB her biri ≥ +0.05; temiz kaybı ≤ 0.01; yankı /
  telefon / gürültü 20 dB kaybı ≤ 0.01. Sağlanırsa "şarkılı kayıt" seçeneği adayı (ürün varsayılanı ayrı karar).

## Adım 3 — Akort / kapo (17 Ekim, kodlandı; plan P3)
- **Kural:** akustik çözümleme (segment / pitch_matrix / energy_rise) hep standart akortla — modelin çıkışı standart akorta bağlı.
  Kullanıcı akortu yalnız konum aşamasında: `gtab/decoding/retune.py` `remap_probs` (kullanıcı (tel, perde′) ← modelin standart
  sütunu; aralık dışı 0; kapoda perdeler kapodan sayılır, TAB / MusicXML `<capo>` geleneği).
- `gtab/core/instrument.py` `parse_tuning` (`DADGBE` → `DROP_D_6`), `with_capo`; `gtab/pipeline.py` `heard_positions` (ses modu ve
  `transcribe_ascii` artık aynı fonksiyon), `analyze / transcribe(..., instrument)`; standart dışı akortta geçiş modeli ağırlıksız,
  `w_transition` ≠ 0 → hata. `edition_net` `instrument` parametresi; `musicxml` `<capo>` + kullanıcı akortu; `transcribe`,
  `transcribe_ascii` `--tuning --capo`.
- **Kapsam dışı:** drop-D D2 / D#2 (MIDI 38–39) modelin aralığı (40–88) dışında → `eval_tuning`'de ayrı sütun; kapsamak için model
  çıkışını genişletme ayrı karar (eğitim).
- **Doğrulama:** standart akortta `analyze` (ses + edisyon) ve `transcribe_ascii` çıktıları 5 kayıtta **birebir** (değişiklik öncesi
  referans); 215 test OK (+11 `tests/test_tuning.py`); `check_inference_regression` regresyon yok; `eval_score` gidiş-dönüş 1.0000.
- **Scordatura sayımı (MusicXML staff-tuning):** gaps_fit 45 / 203 (40 drop-D), **gaps_val 3 / 28**, gaps_test 8 / 30 (7 drop-D).
  gaps_val az → sonuç gösterge; gaps_fit ek gösterge (akustik model o sesle eğitildi).
- Duman (`eval_tuning --limit 1`, gaps_val 169_nM1wc, 6. tel F): konum doğruluğu (tel + perde) ses modu 0.591 → **0.722**,
  edisyon 0.862 → **0.992**; tel doğruluğu değişmiyor (aynı perde için aynı model sütunu okunur; kazanç perde numarasında).
- **Çalıştırma:** `python -m scripts.eval.eval_tuning --count-only`; `python -m scripts.eval.eval_tuning` (gaps_val);
  `--splits gaps_fit` (gösterge); `--splits gaps_test` yalnız kapanışta.
- **Karar:** doğru akortla konum doğruluğu > standart akortla çözüm (ses ve edisyon ayrı); scordatura tel doğruluğu standart
  kayıtlarla ± 0.02 (gösterge).
- **Ölçüldü (17 Ekim, gaps_fit gösterge; 41 scordatura / kapo, 143 standart kayıt):** konum doğruluğu doğru akortla − standart akortla
  ses **+0.077** (0.534 → 0.610), edisyon **+0.084** (0.685 → 0.769) ✓. Tel doğruluğu scordatura vs standart kayıt: ses −0.012 ✓,
  edisyon −0.030 (sınırın biraz dışında; edisyon modeli yalnız standart akortla eğitildi + gaps_fit onun eğitim verisi → gösterge yanlı).
  Kapsam dışı (D2 / D#2) 990 / 31 264 nota (%3). **Adım 3 KABUL**; edisyon farkı ve 38–39 notaları açık not.
- Açık (dokunulmadı): `gaps_score.score_notes` `<capo>`'yu yok sayıyor (kapolu partisyonda perde kapo kadar düşük okunur);
  `render_ascii_tab` tel adlarını hep "E A D G B e" yazıyor (`transcribe_ascii` başlıkta akortu basıyor).

## Durum
- [x] Adım 0 araçları: `gtab/data/degrade.py`, `scripts/eval/eval_robust.py` (duman testi ✓), kayıt protokolü
- [x] Adım 0a/0b ölçüldü → Adım 2 (kayıt koşulu), polifoni tetiklenmedi
- [x] Adım 2 kodlandı (ses bankaları, bozulmuş önbellek, `--degrade-root`; duman testleri ✓)
- [x] Adım 2: get_voices, ses tabanı, build_degraded, ilk eğitim (seçilmedi = deep2)
- [x] Adım 2 eğitimi: kontrol + 2 deneme (p 0.2; sessiz artırma) → kabul edilmedi, yol kapandı (taban deep2)
- [x] İnsan sesi: `--separate mix-vocals` KABUL (ürüne alma ayrı karar)
- [ ] Kendi kayıtlarımız (0c) → kayıt koşulu kaybının gerçek değeri ← **KALDIĞIMIZ YER (veri bekleniyor)**
- [x] Adım 2b (kaynak ayırma) + Adım 3 (akort / kapo) kodlandı → **kullanıcı ölçümü bekleniyor**
- [ ] P0 (bozulmuş kopyada tel kaybı maskesi) yalnız T3 ayrıştırması "tel" derse
- [ ] P1 kendi kayıtlarımız (veri gelince)
- [ ] Kendi kayıtlarımız (kullanıcı / Armağan; `docs/KAYIT_PROTOKOLU.md`) → `build_own` + 0c / 0d
