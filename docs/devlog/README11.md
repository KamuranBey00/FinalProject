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

## Katman 3.10 çıkış ölçütü ("ciddi iyileşme")
`tabcrnn_onset_h` tabanına göre (aynı kalibrasyon ve ölçütle):
- val_comp nota F1 ≥ +0.05 ve val_comp kare F1 ≥ +0.03
- GAPS nota F1 ≥ +0.05 ve GAPS kare F1 ≥ +0.03
- GuitarSet solo tab F1 ve nota F1'de −0.01'den fazla kayıp yok

## Durum
- [x] Adım 0 — ölçüm (diagnose_pitch)
- [x] Adım 1 — çözümleme kuralları + mix ölçütü: GAPS'te kazanç, GuitarSet akorlarda nötr (GS tab F1 kontrolü bekleniyor)
- [ ] Adım 2 — eğitim (Adım 1 sonuçlarına göre)
