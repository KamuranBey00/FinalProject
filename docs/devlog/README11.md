# Katman 3.10 — Polifonide perde: hayalet notalar ve kaçan perdeler

## Hedef
Katman 3.9 sonrası (taban `tabcrnn_onset_h.pt`) tel tarafı polifonide sorun değil
(akorlarda tel hatası %5). Kalan kayıp perde tarafında: akorlarda bulunamayan
perdeler ve hayalet notalar. Bu katman ikisini önce ölçer, sonra çözer.

Taban (alan-içi kalibrasyon, `--criterion note`, Katman 3.9 çözümlemesi):

| | GAPS nota F1 | GAPS kare F1 | GS val nota F1 | GS val kare F1 | val_comp nota F1 | val_comp kare F1 | GS solo tab F1 |
|---|---|---|---|---|---|---|---|
| tabcrnn_onset_h | 0.592 | 0.556 | 0.909 | 0.843 | 0.724 | 0.765 | 0.739 |

## Adım 0 — Ölçüm (`scripts/eval/diagnose_pitch.py`)
Eğitim yok. **Doğrulama** setlerinde (val_comp, gaps_val) yapıldı; test setine bakılmadı.
Çözümleme ve eşikler eval_pitch ile aynı (ortak çekirdek: `gtab/evaluation/pitch_eval.py`).

| | val_comp (GuitarSet akor) | gaps_val (klasik) |
|---|---|---|
| kaçan gerçek nota | %27.5 | %37.5 |
| hayalet (tahminlerin) | %27.6 | %47.2 |
| **Kaçanların nedeni** | | |
| perde hiç aktif değil (model duymuyor) | %44.9 | %38.3 |
| yerine oktav/harmonik seçilmiş | **%30.1** | %19.5 |
| perde aktif ama nota başlatılmamış (onset kaçtı) | %15.1 | **%31.5** |
| nota var, onset zamanı kaymış | %9.9 | %10.7 |
| **Kaçma oranı (grup içinde)** | | |
| kısa nota (<5 kare) | **%49.1** | %45.3 |
| register: pes <52 / orta / tiz 64+ | %24 / %26 / **%34** | **%48** / %34 / %37 |
| polifoni 1 / 2 / 3 / 4+ | %28 / %31 / %22 / %29 | %33 / %37 / %40 / %43 |
| **Hayaletlerin gerçek notalarla ilişkisi** | | |
| aynı perde (çift tetik / zamanlama) | **%75.4** | %29.3 |
| 5'li/12'li harmonik + oktav | %10.3 | %23.8 |
| komşu yarım/tam ses | %2.8 | %12.1 |
| diğer aralık | %10.6 | %34.1 |
| sessizlikte | %0.8 | %0.7 |
| **Kare kaybı notanın neresinde** | | |
| baş %20 / orta / son %20 | %24 / %30 / **%44** | %46 / %53 / **%73** |
| eşleşen notaların %70'inden azını kapsayan | %13 | %29 |

Bulgular:
1. **Akorlarda hayaletlerin %75'i aynı perdenin yeniden tetiklenmesi**: nota sürerken
   onset olasılığı düşüp yeniden yükseliyor → çözümleme ikinci bir nota başlatıyor.
2. **Kare kaybı en çok notanın sönümlenen sonunda**: nota tek bir yüksek eşikle
   sürdürülüyor (GuitarSet'te seçilen eşik 0.9) → kuyruk kesiliyor.
3. **GAPS'te kaçanların %32'si "perde aktif ama onset kaçtı"**: onset algılama klasik
   gitarda zayıf (onset F1 0.44) → nota tamamen düşüyor.
4. Modelin hiç duymadığı perdeler (%38–45), oktav/harmonik karışıklığı (%20–30), kısa
   notalar (~%47 kaçma), GAPS'te pes notalar (%48) → bunlar çözümlemeyle değil
   **eğitimle** çözülür (Adım 2).
5. GAPS'te "komşu yarım/tam ses" (%12) ve "diğer aralık" (%34) hayaletlerinin bir kısmı
   etiket gürültüsü olabilir (GAPS etiketleri partisyon hizalamasından geliyor).

## Adım 1 — Çözümleme kuralları (eğitim yok)
`segment_notes_onset` (`gtab/decoding/viterbi.py`) üç yeni parametre aldı; varsayılanlar
Katman 3.9 davranışını **birebir** korur (test edildi):

| Kural | Parametre | Hedeflediği bulgu |
|---|---|---|
| **Refrakter pencere**: aynı perdede nota sürerken kısa sürede gelen yeni onset yok sayılır | `refractory` (kare) | 1 — çift tetik hayaletleri |
| **Histerezis**: nota `eşik` ile başlar, `eşik × off_ratio` üstünde kaldıkça sürer | `off_ratio` | 2 — kuyruk kaybı |
| **Yedek kural**: hiçbir notanın kapsamadığı, `fallback` kare boyunca aktif kalan perde onset'siz de nota sayılır | `fallback` (kare) | 3 — onset kaçınca notanın düşmesi |

Birim testi: kuyruk 20 → 26 kareye uzadı, çift tetik tek notaya indi, onset'siz uzun
perde yedek kuralla geri geldi.

Seçim yine doğrulamada (`gtab/evaluation/pitch_eval.calibrate`):
1. Çözümleme yöntemi + perde eşiği (Katman 3.9 ile aynı).
2. En iyi onset eşiğiyle 18 kural kombinasyonu (off_ratio ∈ {1, 0.7, 0.5} × refractory ∈
   {0, 3, 6} × fallback ∈ {0, 10}), seçilen eşiğin ±0.1 komşuluğunda. Birinci aşamada
   kare-eşik kazansa bile denenir.
3. **Ölçüt:** `--criterion note` (nota F1, 3.9 ile aynı) ya da `--criterion mix`
   ((nota F1 + kare F1)/2). Nota F1 nota sonunu saymadığı için kuyruğu düzelten
   histerezisi ödüllendirmez; polifoni hedefinde ikisi de önemli → **mix önerilir**.
   Taban da aynı ölçütle yeniden ölçülür (`--no-dec-search`).

Duman testi (3 kayıt, kesin değil): `mix` ile val'de histerezis 0.5 + refrakter 6 seçildi;
val_comp kare F1 0.811 → 0.824. GAPS doğrulaması bu küçük örnekte kare-eşiği korudu.

### Çalıştırma
```bash
# taban (3.9 çözümlemesi) — mix ölçütüyle
python -m scripts.eval.eval_pitch --ckpt tabcrnn_onset_h.pt --splits gaps_test val val_comp --criterion mix --no-dec-search
# Adım 1 — yeni kurallar (iki ölçütle)
python -m scripts.eval.eval_pitch --ckpt tabcrnn_onset_h.pt --splits gaps_test val val_comp --criterion mix
python -m scripts.eval.eval_pitch --ckpt tabcrnn_onset_h.pt --splits gaps_test val val_comp --criterion note
# hata dağılımı nasıl değişti (doğrulama setleri)
python -m scripts.eval.diagnose_pitch --ckpt tabcrnn_onset_h.pt --splits val_comp gaps_val --criterion mix
# GuitarSet solo tab F1 (eval_pitch'in 'val' için seçtiği değerlerle)
python -m scripts.eval.eval_hmm --model crnn --ckpt tabcrnn_onset_h.pt --off-ratio <secilen> --refractory <secilen> --fallback <secilen>
```

### Adım 1 kabul ölçütü (eğitimsiz kazanç)
Aynı ölçütle ölçülen tabana göre: val_comp kare F1 ≥ +0.03, GAPS nota ve kare F1'de kayıp
yok (≥ −0.01), GS solo tab F1'de kayıp yok (≥ −0.01).

### Adım 1 sonuçları (6 Ekim 2026, tabcrnn_onset_h, test setleri)
| Yapılandırma | GAPS nota | GAPS kare | GS val nota | GS val kare | val_comp nota | val_comp kare |
|---|---|---|---|---|---|---|
| A: 3.9 çözümlemesi, ölçüt note (eski tablo) | 0.592 | 0.556 | 0.909 | 0.843 | 0.724 | 0.765 |
| B: 3.9 çözümlemesi, ölçüt **mix** (yeni taban) | 0.545 | 0.621 | 0.908 | 0.860 | **0.740** | 0.798 |
| C: **kurallar + mix** | **0.588** | **0.641** | 0.907 | **0.865** | 0.730 | **0.800** |
| D: kurallar + note | **0.599** | 0.556 | **0.911** | 0.843 | 0.718 | 0.765 |

Seçilen kurallar: GAPS → onset@0.2, histerezis 0.5, refrakter 6, yedek 10;
GuitarSet → onset@0.3, histerezis 1.0 (yok), refrakter 6, yedek 10.

Yorum:
- **Eğitimsiz en büyük kazanç ölçüt değişikliğinden geldi (A→B):** eşiği yalnızca nota
  F1'e göre seçmek kare F1'i feda ediyordu; mix ile GAPS kare 0.556 → 0.621, val_comp kare
  0.765 → 0.798, GS val kare 0.843 → 0.860, nota F1'ler korunuyor (GAPS hariç).
- **Kurallar GAPS'te işe yaradı (B→C):** nota +0.043, kare +0.020; GAPS doğrulaması
  ilk kez onset çözümlemesini kare-eşiğe tercih etti. GAPS'te 1 notalı karelerde kare F1
  biraz düştü (0.62 → 0.58), 2–4+ notalılarda arttı (0.60–0.65 → 0.65–0.66).
- **GuitarSet akorlarda kurallar nötr (B→C):** nota −0.010, kare +0.002 → Adım 1 kabul
  ölçütü (val_comp kare ≥ +0.03) **sağlanmadı**.
- diagnose_pitch (val_comp, C'nin GuitarSet ayarı): kaçan nota %27.5 → %22.7, kare kaybı
  nota sonunda %44 → %31, erken biten nota %13 → %8; ama hayaletler %27.6 → %30.8 ve
  "aynı perde" hayaletleri 1359 → 1509. "Onset zamanı kaymış" kaçanlar %10 → %21.
  → "Aynı perde" hayaletlerinin çoğu çift tetik değil, **onset zamanlama hatası**:
  nota bulunuyor ama başlangıcı >50 ms kayık olduğu için hem kaçan hem hayalet sayılıyor.
  Muhtemel kaynak eğitim tarafında: onset hedefi `dilate=2` ile iki kareye yayılıyor
  (onset tepe noktası bulanıklaşıyor). Bu, çözümlemeyle değil eğitimle düzelir.

Karar: **C (kurallar + mix) yeni çözümleme varsayılanı adayı** — GAPS'te net kazanç,
GuitarSet'te gürültü sınırında. Kesinleşmesi için GS solo tab F1 ölçümü gerekli:
`python -m scripts.eval.eval_hmm --model crnn --ckpt tabcrnn_onset_h.pt --refractory 6 --fallback 10`
(taban 0.739). Akorlardaki asıl kayıp (zamanlama, duyulmayan perdeler, oktav) Adım 2'ye.

### Adım 1 sonrası hata dağılımı (diagnose_pitch, doğrulama setleri)
| | val_comp önce | val_comp sonra | gaps_val önce | gaps_val sonra |
|---|---|---|---|---|
| kaçan gerçek nota | %27.5 | **%22.7** | %37.5 | %36.3 |
| hayalet (tahminlerin) | %27.6 | %30.8 | %47.2 | %47.2 |
| kaçan: perde hiç aktif değil | %44.9 | %36.3 | %38.3 | %38.7 |
| kaçan: oktav/harmonik seçilmiş | %30.1 | %29.0 | %19.5 | %19.4 |
| kaçan: onset kaçtı | %15.1 | %13.5 | **%31.5** | **%13.1** |
| kaçan: **onset zamanı kaymış** | %9.9 | **%21.1** | %10.7 | **%28.8** |
| hayalet: aynı perde | %75.4 | %67.2 | %29.3 | %28.8 |
| hayalet: harmonik + oktav + komşu + diğer aralık | %23.8 | %31.6 | %69.9 | %70.2 |
| kare kaybı nota sonunda (son %20) | %43.9 | **%31.3** | %73.3 | **%56.7** |
| %70'inden azı kapsanan (erken biten) nota | %13 | %8 | %29 | %16 |

("önce" = 3.9 çözümlemesi, note ölçütü; "sonra" = kurallar + mix.)

Yorum (Adım 0'daki yorumu düzeltir):
- Kurallar hedefledikleri iki kaybı gerçekten giderdi: **kuyruk kaybı** (nota sonu kaybı
  %44→%31 ve %73→%57) ve GAPS'te **onset kaçınca düşen notalar** (%31.5→%13.1, yedek kural).
- Ama kurtarılan notaların bir kısmı **yanlış başlangıç zamanıyla** geldi: "onset zamanı
  kaymış" iki sette de 2–3 katına çıktı. Bu notalar 50 ms toleransı geçemediği için hem
  kaçan hem hayalet sayılıyor → nota F1'e yansıyan net kazanç küçük kaldı.
- Adım 0'da "akor hayaletlerinin %75'i çift tetik" diye yorumlamıştık; refrakter pencere
  bunları azaltmadı. Doğru yorum: bunlar büyük ölçüde **onset zamanlama hatası**
  (nota doğru perdede ama başlangıç kayık). Kayıpların "çoğu çözümlemeden" değil;
  çözümleme kuyrukları ve düşen notaları düzeltti, **zamanlama ve duyulmayan perdeler
  modelin kendi sorunu**.
- **GAPS hayaletleri farklı:** %70'i yanlış perde (diğer aralık %34, harmonik %18, komşu
  ses %12, oktav %7). Bunların bir kısmı model hatası (harmonik/oktav), bir kısmı muhtemelen
  GAPS etiket gürültüsü (partisyon hizalamasından gelen etiketler) → eğitim öncesi ölçülmeli.

## Adım 2 — Eğitim (öneri, Adım 1 bulgularına göre güncellendi)
| Öncelik | Değişiklik | Hedeflediği kayıp (val_comp / gaps_val) |
|---|---|---|
| 1 | **Keskin onset hedefi**: `dilate=2` → tek kare tepe + komşu karelere yumuşak (ör. 0.5) hedef; çözümlemede onset'i yerel tepe noktasından başlatma | "onset zamanı kaymış" %21 / %29 ve "aynı perde" hayaletleri |
| 2 | **GuitarSet'e doğrudan perde kaybı** (noisy-OR perde BCE, GAPS'teki gibi; tab CE'ye ek) | "perde hiç aktif değil" %36 / %39, oktav/harmonik %29 / %19 |
| 3 | **Kısa nota / onset karesi ağırlığı** | kısa notaların kaçma oranı %47 / %50 |
| — | GAPS onset ağırlığını artırmak (Adım 0 önerisi) | **önceliği düştü**: yedek kural "onset kaçtı"yı zaten %31→%13'e indirdi |

Eğitimden önce kısa ve eğitimsiz iki kontrol:
- **Zamanlama doğrulaması:** nota F1'i 50 ms yanında 100 ms onset toleransıyla da ölçmek.
  Fark büyükse hata gerçekten zamanlama (öncelik 1 doğrulanır).
- **GAPS etiket kalitesi:** yanlış perde hayaletlerinin (%70) ne kadarının etiket gürültüsü
  olduğunu görmek için örnek kayıtlarda `.match` silme/ekleme oranlarına bakmak.

### Adım 1 kararı (7 Ekim 2026)
GS solo tab F1, GuitarSet'te seçilen kurallarla (onset@0.3, refrakter 6, yedek 10):
greedy **0.743** (taban 0.739) → kayıp yok. **"Kurallar + mix" varsayılan çözümleme oldu.**
Öğrenilen geçiş önseli yine katkısız (w_tr = 0).

## Adım 2 — uygulandı (kod + testler; kullanıcının koşusu bekleniyor)
### 2a. Eğitimsiz iki kontrol
1. **Zamanlama:** `eval_pitch` özet tablosunda yeni `@100ms` sütunu = aynı seçimle 100 ms onset
   toleransında nota F1. 50 ms ile farkı büyükse hata gerçekten zamanlama.
2. **GAPS etiket kalitesi:** `scripts/eval/check_gaps_labels.py`. `.match` dosyasından parça
   başına partisyon–performans eşleşme oranı; modelin parça başına nota F1/precision'ı ile
   Spearman korelasyonu. Yalnızca test dışı parçalar (52: eğitim 47 + gaps_val 5); hiçbir
   parametre seçilmez. Not: duman testindeki parçada eşleşme oranı 0.10 iken model F1 0.65 —
   `.match` her parçada etiket kalitesini yansıtmıyor olabilir (ör. partisyondaki tekrarlar);
   sonuç dikkatle yorumlanacak.

### 2b. Tek ince ayar koşusu (`train_onset`, yeni seçenekler; varsayılanlar = 3.9 eğitimi birebir)
| Seçenek | Ne yapar | Hedef (Adım 1 bulgusu) |
|---|---|---|
| `--onset-soft 0.3` | onset hedefi: tepe karesi 1, sonraki kare 0.3 (eskiden iki kare de 1) | kayık başlangıçlar %21 / %29 |
| `--gs-pitch-weight 1.0` | GuitarSet'e doğrudan noisy-OR perde BCE (`tab_pitch_target`) | duyulmayan perdeler %36 / %39, oktav/harmonik %29 / %19 |
| `--short-frames 5 --short-weight 2.0` | 5 kareden kısa notaların karelerine 2× ağırlık (tel CE + onset) | kısa notaların %47 / %50 kaçması |
| çözümleme `peak` | nota onset koşusunun tepe karesinden başlar; kalibrasyonda otomatik denenir | kayık başlangıçlar |

Init modelin harmonik ayarı checkpoint'ten otomatik okunur (`tabcrnn_onset_h.pt`'nin 6 kanalı korunur).

Testler: keskin hedef / kısa nota ağırlığı / perde hedefi / tepe başlangıcı birim testleri ✓;
yeni seçenekler kapalıyken eval sonuçları birebir aynı (gaps_test 0.530/0.627, val 0.880/0.853) ✓;
1 epoch eğitim duman testi ✓ (onset F1 artık tek kare hedefe göre ölçüldüğü için eğitim
logundaki GS-onsetF1, 3.9 koşularıyla doğrudan kıyaslanamaz — karar eval_pitch ile verilir).

### Çalıştırma (sırayla)
```bash
# 2a — kontroller (mevcut model)
python -m scripts.eval.eval_pitch --ckpt tabcrnn_onset_h.pt --splits gaps_test val val_comp --criterion mix
python -m scripts.eval.check_gaps_labels --ckpt tabcrnn_onset_h.pt

# 2b — ince ayar (yeni dosya; tabcrnn_onset_h.pt korunur)
python -m scripts.train.train_onset --init tabcrnn_onset_h.pt --epochs 15 --ckpt tabcrnn_poly.pt --onset-soft 0.3 --gs-pitch-weight 1.0 --short-frames 5 --short-weight 2.0

# 2b — ölçüm
python -m scripts.eval.eval_pitch --ckpt tabcrnn_poly.pt --splits gaps_test val val_comp --criterion mix
python -m scripts.eval.diagnose_pitch --ckpt tabcrnn_poly.pt --splits val_comp gaps_val --criterion mix
python -m scripts.eval.eval_hmm --model crnn --ckpt tabcrnn_poly.pt --refractory <val> --fallback <val> [--off-ratio <val>] [--peak]
```
(eval_hmm değerleri: eval_pitch çıktısındaki `[val] secim:` satırından.)

Karşılaştırma tabanı (C: kurallar + mix, tabcrnn_onset_h): GAPS 0.588 / 0.641, GS val 0.907 / 0.865,
val_comp 0.730 / 0.800, GS solo tab F1 0.743.

## Adım 2 sonuçları (7 Ekim 2026)
### 2a — kontroller (tabcrnn_onset_h)
**Zamanlama (`@100ms`):** gaps_test 0.588 → 0.610, GS val 0.907 → 0.916, val_comp 0.730 → 0.736.
Toleransı ikiye katlamak yalnızca +0.006 … +0.022 kazandırıyor → "onset zamanı kaymış"
kaçanların çoğu 50–100 ms'lik küçük kayma **değil**; nota daha geç/erken başlıyor ya da
**parçalanıyor** (aynı notanın ikinci parçası). Keskin onset hipotezi zayıf destek aldı.

**GAPS etiket kalitesi (`check_gaps_labels`, 52 test dışı parça):** eşleşme oranı tüm parçalarda
çok düşük (medyan 0.11, 0.04–0.21) → `.match` bu veri için güvenilir bir kalite ölçüsü değil.
Spearman(eşleşme, nota F1) +0.22, (eşleşme, precision) +0.05 → etiket gürültüsünün GAPS
hayaletlerini açıkladığına dair **kanıt yok**; hatalar büyük ölçüde modelde kabul edilir.

### 2b — `tabcrnn_poly.pt` (keskin onset + GuitarSet perde kaybı + kısa nota ağırlığı)
| (kurallar + mix) | GAPS nota | GAPS kare | GS val nota | GS val kare | val_comp nota | val_comp kare |
|---|---|---|---|---|---|---|
| tabcrnn_onset_h (taban C) | **0.588** | 0.641 | **0.907** | 0.865 | 0.730 | 0.800 |
| **tabcrnn_poly** | 0.582 | **0.656** | 0.905 | **0.869** | **0.744** | **0.817** |
| fark | −0.006 | +0.015 | −0.002 | +0.004 | +0.014 | +0.017 |

Polifoni kare F1 (2 / 3 / 4+ nota): GAPS 0.66/0.66/0.65 → 0.68/0.68/0.66; val_comp 0.70/0.80/0.85 → 0.70/0.82/0.87.
Eğitim logu (argmax tab F1, GS val): 0.658 → 0.685. GuitarSet çözümlemesinde ilk kez `tepe=True` seçildi.
**GS solo tab F1 henüz ölçülmedi** (aşağıdaki komut).

diagnose_pitch (taban → poly):
| | val_comp | gaps_val |
|---|---|---|
| kaçan gerçek nota | %22.7 → **%18.9** | %36.3 → **%33.8** |
| "perde hiç aktif değil" (adet) | 540 → **365** | 4004 → **3715** |
| oktav/harmonik (adet) | 431 → 389 | 2011 → 2170 |
| zamanı kaymış (adet) | 314 → 287 | 2977 → 2801 |
| hayalet (tahminlerin) | %30.8 → %31.3 (2246 → 2415 adet) | %47.2 → %49.1 (16191 → 18226) |
| kare kaybı nota sonunda | %31 → %27 | %57 → %57 |

Yorum:
- **Recall tarafı düzeldi:** GuitarSet'e doğrudan perde kaybı en çok "hiç duyulmayan perde"yi
  azalttı (−%32 akorlarda); kısa notaların kaçma oranı %47 → %44.
- **Darboğaz artık precision (hayaletler):** hayalet sayısı iki sette de arttı. val_comp nota P 0.69 /
  R 0.81, GAPS P 0.49 / R 0.71. Muhtemel nedenler: onset `pos_weight=3` + kısa nota ağırlığı +
  yedek kural daha çok nota başlatıyor; akorlarda hayaletlerin %67'si hâlâ "aynı perde" ve
  100 ms testi bunların küçük kayma olmadığını gösteriyor → **nota parçalanması** adayı.
- Kazanç gerçek ama küçük; çıkış ölçütü (taban C'ye göre nota +0.05 / kare +0.03) **sağlanmadı**.
  Karar bekleyen: tab F1 ≥ 0.733 ise `tabcrnn_poly.pt` Katman 3.10'un yeni çalışma tabanı olur.

## Adım 3 — plan: precision (hayaletler)
**3a. Ölçüm (eğitim yok):** diagnose_pitch'e hayalet alt kırılımı:
(i) aynı perdede eşleşmiş bir gerçek notanın İÇİNDE başlayan parça (fragman),
(ii) gerçek notadan önce/sonra başlayan ama eşleşmeyen nota (geç/erken başlangıç),
(iii) hangi kural üretti (onset / yedek kural), (iv) hayaletin onset tepe değeri dağılımı.
**3b. Eğitimsiz:** bulguya göre çözümleme — fragman birleştirme (aynı perdede kısa boşluk + zayıf
onset tepe noktası → tek nota), "yeniden vuruş" için daha yüksek onset eşiği, yedek kuralın
yalnızca onset'siz bölgelerde uzun aktivasyonla sınırlanması. Hepsi doğrulamada seçilir.
**3c. Eğitim (gerekirse):** onset `--pos-weight 3 → 1` (fazla onset → hayalet), sunumdaki
"offset" modülü (nota sonu kafası) ile parçalanmanın modellenmesi, GAPS harmonik/oktav
hayaletleri için sert negatif ağırlık.

## Katman 3.10 çıkış ölçütü ("ciddi iyileşme")
`tabcrnn_onset_h` tabanına göre (aynı kalibrasyon ve ölçütle):
- val_comp nota F1 ≥ +0.05 ve val_comp kare F1 ≥ +0.03
- GAPS nota F1 ≥ +0.05 ve GAPS kare F1 ≥ +0.03
- GuitarSet solo tab F1 ve nota F1'de −0.01'den fazla kayıp yok

## Durum
- [x] Adım 0 — ölçüm (diagnose_pitch)
- [x] Adım 1 — çözümleme kuralları + mix ölçütü: GAPS'te kazanç, GuitarSet akorlarda nötr (GS tab F1 kontrolü bekleniyor)
- [x] Adım 1 kararı: kurallar + mix varsayılan (GS tab F1 0.743)
- [x] Adım 2 — kontroller + ince ayar seçenekleri: kod + testler
- [x] Adım 2 — sonuçlar: tabcrnn_poly (val_comp +0.014/+0.017, GAPS kare +0.015); recall düzeldi, darboğaz precision
- [ ] GS solo tab F1 (tabcrnn_poly) ölçümü
- [ ] Adım 3 — hayalet (precision) analizi ve çözümü
