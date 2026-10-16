# Katman 6 — Zorlaştırma: gerçek dünya (polifoni, kayıt koşulu, akort, kendi kayıtlarımız)

> **Durum (16 Ekim 2026):** Adım 0 (ölçüm, eğitim yok) kodlandı → **kullanıcının ölçümü bekleniyor.** Katman 4 (teknikler)
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

## Durum
- [x] Adım 0 araçları: `gtab/data/degrade.py`, `scripts/eval/eval_robust.py` (duman testi ✓), kayıt protokolü
- [ ] Adım 0a/0b ölçümü (kullanıcı) ← **KALDIĞIMIZ YER**
- [ ] Kendi kayıtlarımız (kullanıcı / Armağan; `docs/KAYIT_PROTOKOLU.md`) → `build_own` + 0c / 0d
