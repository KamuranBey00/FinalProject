# Katman 3.12 — Akustik katman: kapasite mi, veri mi?

> Durum: **Adım 0 kodlandı, ölçüm bekleniyor** (9 Ekim 2026).
> Taban: `checkpoints/tabcrnn_rep_valley.pt` (Katman 3.11 Adım 4; tab F1 0.769 / 0.771).
> Neden: Katman 3.11 çözümleme ve eğitim tarafını tüketti; GAPS'teki en büyük kayıplar akustik
> (kaçanların %40'ında perde hiç aktif değil, hayaletlerin ~%50'si farklı perde) ve <100 ms tekrarlar
> hiçbir müdahaleyle değişmedi. Sıradaki soru: model mi yetmiyor (kapasite), veri mi (genelleme)?

## Adım 0 — Kapasite ölçümü (eğitim yok)
Aynı model ve **aynı çözümleme** (doğrulamada seçilmiş, dosyadan) eğitim verisinde ve doğrulamada ölçülür.

Kod (kalıcı hızlandırma dahil):
- `eval_pitch --save-cal F` / `--load-cal F` (`pitch_eval.save_selection`, `load_selection`): kalibrasyon seçimi
  JSON'a yazılır / okunur; okununca **arama yok, yalnız seçilen eşik** değerlendirilir. `diagnose_pitch --load-cal`.
- Sanal split `gaps_fit` = gaps_train'in yalnız eğitimde kullanılan kayıtları (203; `gaps_files(...)[0]`).
  (`gaps_train` önbelleği doğrulama icracılarını da içerdiği için kapasite ölçümünde kullanılmaz.)
- `checkpoints/tabcrnn_rep_valley.cal.json`: 9 Ekim eval_pitch seçimleri (gaps_val, val).
- Kalibrasyon ilerleme çubuğu (10 Ekim; `pitch_eval._Progress`, stderr): zincir başına geçen / kalan dk.
  Yalnız gösterim, sonuçlar değişmez (duman testi: `--limit 1`).

Test: `--load-cal` ile gaps_test / val / val_comp sonuçları tam kalibrasyonlu koşuyla birebir aynı
(0.720 / 0.684, 0.916 / 0.880, 0.783 / 0.818; tekrar kaçma sütunları dahil); süre ~1 saat → **57 sn**.

### Çalıştırma
```bash
python -m scripts.eval.eval_pitch --ckpt tabcrnn_rep_valley.pt --load-cal checkpoints/tabcrnn_rep_valley.cal.json --splits gaps_fit gaps_val train val train_comp val_comp
# isteğe bağlı: eğitim verisinde hata türleri (perde hiç aktif değil oranı eğitimde de yüksek mi?)
python -m scripts.eval.diagnose_pitch --ckpt tabcrnn_rep_valley.pt --load-cal checkpoints/tabcrnn_rep_valley.cal.json --splits gaps_fit gaps_val --limit 40
```

### Karar kuralı (GAPS nota F1 esas; GuitarSet destekleyici)
fark = eğitim (gaps_fit) − doğrulama (gaps_val)
- fark ≥ 0.05 → **veri / genelleme sorunu** → önce 3b (SynthTab naylon ön-eğitimi).
- fark < 0.03 ve ikisi de düşük → **kapasite sorunu** → önce 3a (derin CNN + perde-onset kafası).
- arada → ikisi; sıra 3a, ardından 3b.
Diagnose (isteğe bağlı): eğitimde de "perde hiç aktif değil" yüksekse kapasite kanıtı güçlenir.

### Adım 0 — sonuçlar (9 Ekim 2026, tabcrnn_rep_valley, aynı çözümleme)
| split | nota P / R / F1 | kare F1 | tekrar kaçma <100 / 100–200 ms |
|---|---|---|---|
| **gaps_fit** (eğitim) | 0.695 / 0.753 / **0.723** | **0.693** | 73.9% / 43.6% |
| **gaps_val** | 0.697 / 0.741 / **0.718** | **0.660** | 73.0% / 52.7% |
| train (GS solo) | 0.861 / 0.949 / 0.903 | 0.928 | 44.1% (n=136) / 4.6% |
| val | 0.895 / 0.939 / 0.916 | 0.880 | 25.0% (n=4) / 10.6% |
| train_comp | 0.758 / 0.921 / 0.832 | 0.904 | 19.4% / 5.5% |
| val_comp | 0.734 / 0.839 / 0.783 | 0.818 | 27.5% / 20.9% |

Diagnose (gaps_fit 40 kayıt / gaps_val): kaçan %22.6 / %25.9; kaçanlarda "perde hiç aktif değil" %35.7 / %40.2;
hayaletlerin "farklı perde / sessizlik" %50.1 / %50.6; <100 ms tepe görünürlüğü %71 / %65.

Yorum:
- **GAPS: fark yok.** Nota F1 farkı 0.005 (< 0.03), kare 0.033. Model eğitim verisinde de doğrulamadaki kadar
  düşük; aynı hata türleri eğitimde de aynı oranda (perde hiç aktif değil ~%36, yanlış perde hayaletleri ~%50).
  → Karar kuralı: **kapasite sorunu → 3a.** Daha çok aynı tür veri (3b) bu tabloyu tek başına düzeltmez.
- **<100 ms tekrarlar eğitimde de kaçıyor** (GAPS %74, GuitarSet solo hex-pickup etiketlerinde bile %44):
  ezber bile yapılamıyor → veri miktarı değil, model (çözünürlük / kapasite) sınırı. GAPS etiket gürültüsüne
  bağlanamaz (GuitarSet etiketleri hassas).
- 100–200 ms: eğitim %43 / doğrulama %53 → hafif genelleme farkı.
- **GuitarSet akor:** train_comp 0.832 / 0.904 vs val_comp 0.783 / 0.818 → fark 0.05 / 0.09 = genelleme
  (veri) sorunu; tek doğrulama oyuncusu. Fingerstyle önceliği nedeniyle ikincil.
- GuitarSet solo: eğitim ≈ doğrulama (nota 0.903 / 0.916) → solo tarafta ne kapasite ne veri darboğazı belirgin.

Uyarı (alternatif açıklama): GAPS'te eğitim F1'inin düşük kalması **etiket tavanı** da olabilir (partisyondan
hizalanmış etiketler: model tutarsız etiketi ezberleyemez). 3.10 Adım 2a'da etiket gürültüsüne dair kanıt
bulunmamıştı; ama kapasiteyi kesinleştirmenin ucuz yolu küçük bir aşırı öğrenme testi (az kayıt, çok epoch:
ezberleyebiliyorsa kapasite, ezberleyemiyorsa etiket tavanı).

### Adım 0b — aşırı öğrenme testi (9 Ekim 2026, `scripts/train/overfit_gaps.py`)
İlk sürüm (kare F1 kararlı, dropout açık, kontrol yok) **geçersiz sayıldı** — kullanıcı düzeltmeleri:
- **Ana ölçüt onset F1** (başlangıçlar iyi hizalı). GAPS'te nota bitişi partisyon değerinden geldiği için
  kare F1 yalnız **notaların ilk %80'inde** ölçülür ve yalnız raporlanır.
- Epoch 0 = aynı kayıtlarda eğitim öncesi taban.
- **Dropout kapalı** (LSTM dahil), **BatchNorm eval** (cuDNN LSTM geri yayılımı train modu istediği için
  `train()` + Dropout/BN eval + `lstm.dropout = 0`).
- Perde ve onset kaybı ayrı yazdırılır (kayıp ~0 iken F1 takılıyorsa etiketler çelişir).
- **Kontrol grubu:** aynı kurulum + aynı perde kaybı 8 GuitarSet (train) kaydında (önbellekte aynı biçimde
  `frame`/`onset` perde rolleri var); GuitarSet onset F1 < 0.95 → sorun kurulumda.

Kurulum: rep_valley'den ince ayar, yalnız perde + perde-onset BCE, 8 kayıt, lr 3e-4, model düzeyi ölçü.
200 epoch: GuitarSet 36 sn, GAPS 198 sn.

| | onset F1 ep0 → 100 → 200 | kare F1 (ilk %80) ep0 → 200 | hızlı tekrar yakalama ep0 → 200 | onset kaybı 20 → 200 |
|---|---|---|---|---|
| **KONTROL GuitarSet** (48 parça) | 0.677 → 0.960 → **0.995** | 0.920 → 1.000 | 66.7% → 91.7% (n=12) | 0.0026 → 0.0011 |
| **GAPS** (254 parça) | 0.397 → 0.830 → **0.952** | 0.699 → 0.998 | 66.7% → 88.1% (n=168) | 0.0063 → 0.0021 |

Yorum:
- Kontrol geçti (0.995 ≥ 0.95) → kurulum doğru.
- **GAPS onset'leri de ezberleniyor** (0.40 → 0.95); eğri ve kayıp birlikte düzenli ilerliyor, "kayıp düşerken
  F1 takılması" yok → **etiket çelişkisi kanıtı yok.** GAPS daha yavaş (5× veri, daha zor başlangıç) ama aynı yolda.
- Kare (ilk %80) iki sette de ~1.0; hızlı tekrar onset'leri de ezberde %88–92 yakalanıyor (n küçük, dalgalı).
- 100 epoch'ta "arada" (0.839) çıkması süre etkisiydi; 200 epoch'ta karar eşiği (0.90) aşıldı.
- **Sonuç: GAPS etiketleri öğrenilebilir; tam veride eğitim F1'inin 0.72'de kalması kapasite (ve/veya
  eğitim süresi) → 3a.** Ezber hızının GuitarSet'ten yavaş olması, 3a'da daha uzun eğitimin de deneneceğini
  söylüyor (epoch sayısı kaydedilecek).

## Adım 3a — Derin CNN + perde-onset kafası + vadi (9 Ekim 2026, kodlandı)

**Ana hedef: GAPS onset F1 (model düzeyi) belirgin artmalı** — nota F1 kalibrasyondan etkilenir, onset F1
doğrudan modeli ölçer (eğitim logunda `GAPS-onsetF1`, gaps_val üzerinde; epoch 0 = rep_valley tabanı).

Kod:
- `nets.ResBlock` + `TabCRNN(deep=N)`: CNN gövdesinden sonra N artık blok (64 kanal, 3×3); son BatchNorm
  gamma = 0 → başlangıçta 0 ekler → **eski modelle birebir aynı çıktı**.
- `TabCRNNOnset(pitch_onset=True)`: doğrudan **perde-onset kafası** (LSTM → 49 perde, önsel bias);
  `forward_full` → (tab, tel onset, offset, perde-onset). Checkpoint'e `deep`, `pitch_onset`.
- Kayıp (`train_onset --pitch-onset-weight W`): GAPS'te doğrudan perde-onset etiketi; GuitarSet'te
  `losses.tab_pitch_onset_target` (tel onset hedefi x tab). Vadi / fragman / tekrar ağırlıkları bu kafaya da
  (`tab_pitch_weight`). Eski tel-onset yolu ve kayıpları aynen sürer.
- **Ayrı öğrenme hızı grubu** (`--new-lr-mult`): init'te olmayan katmanlar (artık bloklar, kafa) lr × k.
- **Kafa başına ayrı log:** `GS-onsetF1` (tel), `GS-perdeOnsetF1(kafa)`, `GAPS-onsetF1(tel)` (noisy-OR),
  `GAPS-onsetF1(kafa)`, kayıplar `pon` / `gpon`. Seçim skoru: eski 4 ölçü, GAPS onset'te iki yoldan iyisi
  (kafa yoksa eskisiyle birebir).
- **Çözümleme yeniden kalibre edilir, eski yol aday:** `calibrate` kafa varsa iki zinciri ayrı ayrı baştan sona
  arar — tel-onset yolu (`string`) ve kafa (`pitch`); doğrulamada iyisi seçilir (`onset_kaynagi`). Kafa yoksa
  tek zincir (birebir). `segment(pitch_onsets=...)`, `predict_heads(with_pitch_onset=True)`;
  `eval_hmm --onset-source string|pitch`; diagnose_pitch bölüm F'de `kafa` sütunu.

Testler:
- Derin + kafalı model, rep_valley ağırlıklarıyla: tab / tel onset / offset çıktıları **birebir** (eval ve train
  modunda artık blok 0 ekliyor) ✓; perde-onset hedefi (tel onset + tab) doğru ✓.
- Epoch 0 (küçük veri): eski ve yeni kurulum aynı skorlar (0.704 / 0.640 / 0.610 / 0.347) ve aynı seçim skoru ✓.
- Kafa öğreniyor (küçük veri, 4 epoch): `pon` 0.058 → 0.031, `gpon` 0.069 → 0.044; GS perde-onset F1 0 → 0.22 ✓.
- Regresyon: rep_valley `--load-cal` ile gaps_test / val / val_comp birebir ✓.
- Uçtan uca duman (kafalı checkpoint, küçük veri 3 epoch, kafa henüz eğitilmemiş): iki zincirli kalibrasyon çalışıyor
  (gaps_val tel 0.664 / kafa 0.619 → tel seçildi; `onset_kaynagi` JSON'a yazıldı), diagnose `--load-cal`, eval_hmm
  `--onset-source pitch` uçtan uca ✓ (kafa eğitilmediği için sayılar anlamsız).

### Çalıştırma (3a)
```bash
# 1) Kısa deneme (2 epoch): ep 0 satırı rep_valley ile aynı olmalı (tam val'de beklenen: GS-tabF1 0.704,
#    GS-onsetF1 0.640, GAPS-kareF1 0.630, GAPS-onsetF1(tel) 0.377), kayıplar ep1 -> ep2 düşmeli
python -m scripts.train.train_onset --init tabcrnn_rep_valley.pt --epochs 2 --ckpt tabcrnn_deep_try.pt --onset-soft 0.3 --gs-pitch-weight 1.0 --short-frames 5 --short-weight 2.0 --repeat-weight 3 --pitch-weight 2 --offset-weight 1 --valley-weight 5 --frag-weight 3 --oversample 3 --deep 2 --pitch-onset-weight 1 --new-lr-mult 3
# 2) Uzun eğitim (aynı bayraklar, 15 epoch = rep_valley ile karşılaştırılabilir süre)
python -m scripts.train.train_onset --init tabcrnn_rep_valley.pt --epochs 15 --ckpt tabcrnn_deep.pt --onset-soft 0.3 --gs-pitch-weight 1.0 --short-frames 5 --short-weight 2.0 --repeat-weight 3 --pitch-weight 2 --offset-weight 1 --valley-weight 5 --frag-weight 3 --oversample 3 --deep 2 --pitch-onset-weight 1 --new-lr-mult 3
# 3) Kalibrasyon (iki zincir: ~2 saat) + seçimi kaydet
python -m scripts.eval.eval_pitch --ckpt tabcrnn_deep.pt --splits gaps_test val val_comp --criterion mix --save-cal checkpoints/tabcrnn_deep.cal.json
# 4) Kapasite ölçümü tekrar (hızlı)
python -m scripts.eval.eval_pitch --ckpt tabcrnn_deep.pt --load-cal checkpoints/tabcrnn_deep.cal.json --splits gaps_fit gaps_val
# 5) Tanılama (hızlı) + tab F1 ([val] secim satırından; onset_kaynagi=pitch ise --onset-source pitch)
python -m scripts.eval.diagnose_pitch --ckpt tabcrnn_deep.pt --load-cal checkpoints/tabcrnn_deep.cal.json --splits gaps_val val_comp
python -m scripts.eval.eval_hmm --model crnn --ckpt tabcrnn_deep.pt --onset-source <..> --onset-thr <..> --off-ratio <..> --refractory <..> --fallback <..> [--peak] --reattack <..> --rise-keep <..> --rise-split <..> --offset-thr <..> --combine <..> [--peak-pick --prominence <..> --min-dist <..> --re-valley <..>]
```

### Kısa deneme sonucu (2 epoch, `tabcrnn_deep_try.pt`)
| ep | GS-tab | GS-onset (tel) | GS-perdeOnset (kafa) | GAPS-kare | GAPS-onset (tel) | GAPS-onset (kafa) | pon / gpon |
|---|---|---|---|---|---|---|---|
| 0 | 0.704 | 0.640 | 0.002 | 0.630 | 0.377 | 0.004 | — |
| 1 | 0.697 | 0.635 | 0.601 | 0.624 | 0.404 | 0.314 | 0.075 / 0.055 |
| 2 | 0.699 | 0.618 | 0.635 | 0.630 | **0.407** | 0.349 | 0.048 / 0.038 |
Epoch 0 rep_valley ile birebir ✓; kayıplar düşüyor ✓. GAPS onset (tel) 2 epoch'ta +0.030 (artık bloklar tel
yolunu da güçlendiriyor); kafa sıfırdan 0.35'e çıktı, hâlâ yükseliyor. İzlenecek: GS onset (tel) 0.640 → 0.618.
→ Uzun eğitime geçildi.

### Uzun eğitim sonucu (15 epoch, `tabcrnn_deep.pt`)
| ep | GS-tab | GS-onset (tel) | GS-perdeOnset (kafa) | GAPS-kare | GAPS-onset (tel) | GAPS-onset (kafa) |
|---|---|---|---|---|---|---|
| 0 | 0.704 | 0.640 | 0.002 | 0.630 | 0.377 | 0.004 |
| 5 | 0.697 | 0.629 | 0.686 | 0.634 | 0.441 | 0.457 |
| 10 | 0.704 | 0.635 | 0.674 | 0.652 | 0.453 | 0.493 |
| 15 | **0.707** | 0.635 | 0.676 | **0.656** | 0.474 | **0.509** |
Seçim skoru 0.588 (rep_valley) → **0.627**. **GAPS onset F1 (model düzeyi) 0.377 → 0.509 (+0.132)** — ana hedef
(≥ +0.03) fazlasıyla sağlandı; kafa ep 4'ten sonra tel yolunu geçti. GS tab F1 korunuyor (0.704 → 0.707),
GS onset (tel) aynı (0.640 → 0.635). Eğri düzleşiyor ama hâlâ hafif yükseliyor (son 5 epoch ~+0.003 / epoch).
Not: kafanın en iyi onset eşiği ep 5'ten beri ızgaranın ucunda (0.5; `ONSET_THRS` 0.1–0.5) → loglanan kafa F1'i
alt sınır olabilir; deep ve deep2 aynı ızgarayla ölçüldüğü için karşılaştırma tutarlı (kod değiştirilmedi).

### Devam eğitimi (deep2) — plan
Kalibrasyon, kapasite tekrarı ve tanılama devam eğitiminden sonra, **seçilen modelde bir kez** yapılır.
Kontroller (kod değişikliği gerekmedi): `--lr` bayrağı var (varsayılan 3e-4), zamanlayıcı cosine (epoch başına,
`T_max = epochs`, 0'a iner) → başlangıç lr 1/3 = `--lr 1e-4`. `--init tabcrnn_deep.pt` (+ `--deep 2
--pitch-onset-weight 1`) eksik anahtar bırakmıyor → yeni lr grubu boş, `--new-lr-mult` etkisiz.
**Fragman taraması bu kez `tabcrnn_deep.pt` ile** yapılır (`fragment_fns(init)`; tel / noisy-OR yolu, kafa değil).
```bash
python -m scripts.train.train_onset --init tabcrnn_deep.pt --epochs 15 --ckpt tabcrnn_deep2.pt --lr 1e-4 --onset-soft 0.3 --gs-pitch-weight 1.0 --short-frames 5 --short-weight 2.0 --repeat-weight 3 --pitch-weight 2 --offset-weight 1 --valley-weight 5 --frag-weight 3 --oversample 3 --deep 2 --pitch-onset-weight 1 --new-lr-mult 3
```
Karar (eğitim logundan, kalibrasyon yok): deep2'nin en iyi `GAPS-onsetF1(kafa)` ≥ **0.519** ve `GS-tabF1` ≥ **0.697**
→ `tabcrnn_deep2.pt`; aksi halde `tabcrnn_deep.pt`.

**Devam eğitimi sonucu (deep2, lr 1e-4, 15 epoch):** GAPS onset (kafa) ep 0 0.509 → en iyi **0.524** (ep 9),
GS tab 0.707 → 0.716; kayıtlı checkpoint = en iyi seçim skoru 0.636 (ep 9: GS-tab 0.716, GS-onset 0.642,
GAPS-kare 0.661, GAPS-onset tel 0.484 / kafa 0.524). ep 6'dan sonra kafa 0.51–0.52 arasında düz → plato.
**Karar: `tabcrnn_deep2.pt` seçildi** (0.524 ≥ 0.519, 0.716 ≥ 0.697). Toplam: rep_valley 0.377 → 0.524 (+0.147).

Seçilen model (`<secilen>`) için sırayla:
```bash
python -m scripts.eval.eval_pitch --ckpt <secilen>.pt --splits gaps_test val val_comp --criterion mix --save-cal checkpoints/<secilen>.cal.json
python -m scripts.eval.eval_pitch --ckpt <secilen>.pt --load-cal checkpoints/<secilen>.cal.json --splits gaps_fit gaps_val
python -m scripts.eval.diagnose_pitch --ckpt <secilen>.pt --load-cal checkpoints/<secilen>.cal.json --splits gaps_val val_comp
python -m scripts.eval.eval_hmm --model crnn --ckpt <secilen>.pt --onset-source <..> --onset-thr <..> --off-ratio <..> --refractory <..> --fallback <..> [--peak] --reattack <..> --rise-keep <..> --rise-split <..> --offset-thr <..> --combine <..> [--peak-pick --prominence <..> --min-dist <..> --re-valley <..>]
```
Kabul: GAPS nota F1 ≥ 0.750 ve GS solo tab F1 ≥ 0.750. Kapasite: gaps_fit − gaps_val > 0.05 → sıradaki 3b (SynthTab).
Kabul tutmazsa: sonuç README13'e yazılır, Katman 4'e geçilir.

### a) Kalibrasyon sonucu (`tabcrnn_deep2.cal.json`, 10 Ekim; gaps_val ~53 dk, val ~13 dk)
İki zincir: gaps_val tel-onset 0.729 | **kafa 0.734**; val tel-onset 0.894 | **kafa 0.902** → iki alanda da
**perde-onset kafası seçildi**.
Seçim: gaps_val → onset@0.4, eşik 0.4, histerezis 0.5, refrakter 3, yedek 10, yeniden_vuruş 0.5, enerji 6/8,
noisy-OR, yerel tepe yok. val → onset@0.2, eşik 0.7, histerezis 1.0, refrakter 0, yedek 10, tepe, yeniden_vuruş 0.3,
enerji_böl 8, offset 0.7, noisy-OR, yerel tepe (0.25 / 2).

| | rep_valley | **deep2** | fark |
|---|---|---|---|
| gaps_test nota / kare F1 | 0.720 / 0.684 (P 0.666 R 0.784) | **0.779 / 0.701** (P 0.746 R 0.813) | **+0.059 / +0.017** |
| gaps_test @100ms | — | 0.787 | |
| gaps_test tekrar kaçma <100 / 100–200 | 77.7 / 36.8 | 76.2 / **32.2** | −1.5 / −4.6 |
| GS val nota / kare | 0.916 / 0.880 | **0.924** / 0.879 | +0.008 / −0.001 |
| val_comp nota / kare | 0.783 / 0.818 | **0.807 / 0.827** | **+0.024** / +0.009 |
| val_comp tekrar kaçma <100 / 100–200 | — | 26.2 / 23.3 | |

**GAPS nota F1 0.779 ≥ 0.750 → kabulün GAPS ayağı sağlandı** (precision ve recall birlikte arttı; kazanç
çözümleme hilesi değil, model düzeyi onset iyileşmesinin nota düzeyine geçişi). val_comp da 3.10 hedefine
(0.780) ilk kez ulaştı. Açık kalan: GAPS <100 ms aynı perde tekrarları (~%76 kaçıyor, değişmedi).
Kabulün ikinci ayağı (GS solo tab F1 ≥ 0.750) eval_hmm ile ölçülecek.

### b) Kapasite tekrarı (`--load-cal`)
| | nota P / R / F1 | kare F1 | tekrar kaçma <100 / 100–200 ms |
|---|---|---|---|
| gaps_fit (eğitimde görülen, 203 kayıt) | 0.767 / 0.803 / **0.784** | 0.722 | 61.0 / 27.7 (n=5318/6439) |
| gaps_val (28 kayıt) | 0.773 / 0.787 / **0.780** | 0.688 | 54.7 / 40.5 (n=647/1020) |
Fark **0.004** (rep_valley: 0.723 / 0.718 → 0.005). Eşik 0.05'in çok altında → veri sınırı yok, **3b (SynthTab)
tetiklenmedi**. Model gördüğü ve görmediği kayıtta aynı → hâlâ aşırı öğrenme yok; iki taraf birlikte +0.06 yükseldi.

### c) Tanılama (gaps_val / val_comp)
- **GAPS:** kaçan %21.3, hayalet %22.7 (dengeli). Kaçanların en büyük nedeni **"perde hiç aktif değil" %36**
  (kare kafası), sonra onset kayması %27, onset kaçması %21, oktav/harmonik %16. Hayaletlerin %28'i aynı perde
  (çift tetik), %21'i fragman. Kare kaybı notanın **son %20'sinde %49** (partisyon kaynaklı nota bitişi, bkz.
  ölçüt kuralı).
- **GAPS hızlı tekrarlar (gaps_val):** kaçma <100 ms **73.0 → 54.7**, 100–200 ms **52.7 → 40.5**, ≥200 ms 24.2 → 22.6
  (rep_valley → deep2). gaps_test <100 ms değişmedi (77.7 → 76.2; n=130, küçük örnek). gaps_val eşik seçiminde
  kullanıldığı için hafif iyimser; yön tutarlı (gaps_test 100–200 ms de −4.6).
- **Tepe görünürlüğü (bölüm F, <100 ms, gaps_val):** kafa **61.2%** | noisy-OR 43.7% | max 38.2% → perde-onset
  kafası hızlı tekrarları tel yolundan belirgin daha iyi ayırıyor; kaçanlardan görünür olan %22 → kalan
  kaybın çoğu model düzeyinde.
- **val_comp:** kaçan %15.1, hayalet %23.1; hayaletlerin %65'i aynı perde (%60.5 fragman, önceki parçayla boşluk
  medyan 0) → akorlarda kalan hata fragmanlaşma. Tekrar kaçma <100 / 100–200 ms 26.2 / 23.3.

### d) GS solo tab F1 (`eval_hmm`, val seçimi, `--onset-source pitch`)
| | rep_valley | **deep2** |
|---|---|---|
| oracle tel seçimi (ses / greedy) | 0.862 (rep_off) | **0.878** |
| baseline (argmax+marj) | — | 0.767 |
| tab F1 greedy / Viterbi | 0.769 / 0.771 | **0.777 / 0.777** |
Öğrenilen geçiş önseli yine katkısız (w_tr=0). Tab F1 +0.008 (tek seed gürültüsü sınırında) ama düşmedi;
oracle tel +0.016 → artık bloklar tel ayrımını da güçlendirdi.

### Karar — Katman 3.12 KAPANDI (10 Ekim)
Kabul: GAPS nota F1 **0.779 ≥ 0.750** ✓, GS solo tab F1 **0.777 ≥ 0.750** ✓. Kapasite farkı 0.004 → 3b gerekmedi.
**Yeni taban: `checkpoints/tabcrnn_deep2.pt`** (+ `tabcrnn_deep2.cal.json`). Gelişim (tab F1): … 0.760 rep_off →
0.769 rep_valley → **0.777 deep2**; GAPS nota F1: 0.689 → 0.720 → **0.779**.
Katman 4'e taşınan açıklar: GAPS <100 ms aynı perde tekrarları (test %76 kaçıyor), "perde hiç aktif değil"
kaçanlar (%36), akorlarda fragman hayaletler.

### Karar kuralı
- **Ana:** GAPS onset F1 (model düzeyi, eğitim logu, gaps_val; iki yoldan iyisi) taban (ep 0, ~0.377) üzerine
  **≥ +0.03**. Hangi kafanın katkı verdiği `(tel)` / `(kafa)` sütunlarından okunur; belirsizse ablasyon
  (`--deep 0` ya da `--pitch-onset-weight 0`).
- Kabul (README13 §3a): GAPS nota F1 ≥ +0.03 (0.720 → ≥ 0.750), GS solo tab F1 ≥ 0.750.
- **Kapasite tekrar:** gaps_fit − gaps_val nota F1 farkı > 0.05 → veri sınırına gelindi → sıradaki **3b**
  (SynthTab naylon ön-eğitimi).

## Adım 3b — SynthTab naylon ön-eğitimi
- Veri sorunu çıkarsa ilk tercih; 3a'dan sonra da eklenebilir. Sunumdaki araştırma sorusunu da cevaplar
  (sentetik naylon → gerçek klasik gitar aktarımı).

## Durum
- [x] Adım 0 — kod (`--load-cal/--save-cal`, `gaps_fit`) + test (birebir, 57 sn)
- [x] Adım 0 — kapasite ölçümü: GAPS eğitim ≈ doğrulama (0.723 / 0.718) → kapasite → 3a; GS akor genelleme farkı
- [x] Adım 0b — aşırı öğrenme testi (düzeltilmiş: onset F1 ana ölçüt, dropout kapalı, GuitarSet kontrol):
  GuitarSet 0.995, GAPS onset 0.952 → etiketler öğrenilebilir, sorun kapasite → 3a
- [x] Adım 3a — kod + testler (artık blok, perde-onset kafası, lr grubu, ayrı loglar, iki zincirli kalibrasyon)
- [x] Adım 3a — kısa deneme (2 epoch) + uzun eğitim (15 epoch): GAPS onset (kafa) 0.377 → 0.509, GS tab 0.707
- [x] Adım 3a — devam eğitimi: deep2 kafa 0.524, GS tab 0.716 → **tabcrnn_deep2.pt seçildi**
- [x] Adım 3a — a) kalibrasyon: kafa seçildi; GAPS nota 0.720 → **0.779**, val_comp 0.783 → 0.807
- [x] Adım 3a — b) kapasite: fark 0.004 → 3b gerekmedi; c) tanılama; d) tab F1 **0.777 / 0.777**
- [x] **Katman 3.12 kapandı** — taban `tabcrnn_deep2.pt`; sırada Katman 4 (kullanıcı talimatı bekleniyor) ← **KALDIĞIMIZ YER**
- [ ] Adım 3b (SynthTab) — tetiklenmedi; ileride veri ihtiyacı çıkarsa
