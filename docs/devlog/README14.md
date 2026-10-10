# Katman 3.13 — GAPS partisyonundan tel etiketi + parmak numarası (README9 3.8e)

> Durum (10 Ekim 2026): Adım 0 etiketler ✓; Adım 1 (TAB ile ses eğitimi) kabul edilmedi; **Adım 2b "klasik edisyon
> önerisi" (konum + parmak, dizi modeli) kabul** — gaps_test tel 0.815 (naif 0.745), parmak 0.636 (taban 0.566). Taban: `checkpoints/tabcrnn_deep2.pt` (Katman 3.12; GAPS nota F1 0.779, GS tab F1 0.777).
> Teknikler (Katman 4) bundan sonra.

## Neden
GAPS (gerçek klasik gitar) şimdiye kadar yalnızca **perde** denetimi verdi; tel bilgisi hiç kullanılmadı.
Modelin tel seçimini öğrendiği tek gerçek veri GuitarSet (çelik tel, farklı repertuvar). Oysa GAPS MusicXML
partisyonları **TAB portresi** (`<string>`, `<fret>`) ve nota portresinde **sol el parmak numarası**
(`<fingering>`) içeriyor. Bu bilgi ses zamanına taşınabilirse: (1) klasik gitarda tel doğruluğu ilk kez
ölçülür, (2) naylon tel tınısı üzerinde tel denetimi mümkün olur, (3) parmaklandırma verisi oluşur.

## Veri bulguları (ölçüldü)
- GAPS dosyaları: `musicxml/` (P1 nota portresi + parmak, P2 TAB portresi), `syncpoints/` (ölçü → saniye;
  üçüncü değer = ölçü içi tick, 480/çeyrek), `midi/` (hizalı MIDI = **gerçek icra zamanları**; arpej/rubato
  yüzünden partisyon zamanından 100 ms+ sapabilir). `.match` yalnız 72 kayıtta ve güvenilmez → kullanılmadı.
- Syncpoint ölçü numaraları **tekrarları açılmış** sırada. Açılmış ölçü sayısı syncpoint'lerle 216/300 kayıtta
  tutuyor; kalanında icracı partisyonda yazmayan tekrarlar/atlamalar çalmış (D.C. işareti yok) → bu kayıtlar
  bu sürümde etiketlenmez.
- Perde kayması (kapo): 4 kayıtta +2/+3; diğerleri 0.
- Dışa aktarma hataları: 4 partisyonda boş `<step>` (perde TAB'dan türetilir), bazı ölçülerde `<duration>` yok
  (nota tipi + nokta + üçlemeden hesaplanır; o ölçüde `<backup>` ölçü başına döner).
- TAB **otomatik "en düşük perde" değil**: çok adaylı notalarda TAB = en düşük perde oranı kayıt başına
  %50–90; tamamen naif kayıt 6/165; notaların %31'i 5. perde ve üstü → gerçek pozisyon bilgisi taşıyor.

## Adım 0 — Etiket üretimi (kodlandı + çalıştırıldı)
Kod:
- `gtab/data/gaps_score.py`: MusicXML okuyucu (TAB + parmak; bağ devamı atlanır, süs notası sıfır süreli),
  tekrar açma (ileri/geri, `times`, 1./2. son), syncpoint enterpolasyonu, eşleme `align()`:
  1) partisyon → saniye (ölçü düzeyi) 2) kayıt geneli perde kayması (kapo) 3) kaba eşleme (±0.25 s) →
  yerel zaman düzeltmesi (±2 çeyrek kayan medyan) → ince eşleme (±80 ms, aynı perde, bire bir).
  **Tel = partisyon TAB'ı; fret = MIDI perdesi − standart açık tel** (kapo farkını içerir).
- `scripts/data/build_gaps_tab.py` → `data/cache/gaps_tab/<id>.npz` (mevcut önbelleğe dokunulmaz):
  `tab` (L,6; GuitarSet şeması), `known` (L; karedeki tüm notalar etiketli), `finger` (L,6; 0 = boş tel, 1–4,
  −1 bilinmiyor), nota düzeyi `notes/string/fret/finger_note`; `summary.csv`.
  Kullanılır kayıt: tekrar yapısı tutarlı **ve** notaların ≥ %50'si etiketli (`--min-labeled`).

Sonuç (10 Ekim 2026):
| | değer |
|---|---|
| kullanılan kayıt | gaps_train **142/231**, gaps_test **23/30** (tutarsız tekrar 70 + 5, az etiketli 19 + 2) |
| etiketli nota / bilinen kare | **%80.9 / %67.2** |
| TAB'dan türeyen perde roll'u = önbellek perde roll'u (bilinen karelerde) | **%98.93** |
| eşleşme zaman farkı | medyan 16 ms |
| parmak numaralı (etiketli notaların) | %33.6 |
| "parmak 0 ⇔ boş tel" tutarlılığı (parmak→TAB bağlama testi) | **%99.4** |

### Adım 0 — tel ölçümü (`scripts/eval/eval_gaps_tab.py`, deep2, eğitim yok)
Oracle tel: gerçek nota verilir, model yalnız teli seçer (greedy; GuitarSet'teki `eval_hmm` "ses / greedy" ile
aynı). Uçtan uca tab F1: yalnız bilinen kareler, çözümleme `tabcrnn_deep2.cal.json` gaps_val seçimi.

| split | kayıt | nota | oracle tel | çok adaylı | naif (en düşük perde) | hatalarda komşu tel | tab F1 (bilinen kare) |
|---|---|---|---|---|---|---|---|
| gaps_val | 15 | 13710 | **0.610** | 0.594 | **0.702** | %95.9 | 0.405 |
| GuitarSet solo val (referans) | 30 | | 0.878 | | 0.206 | | 0.777 |

Yorum: GuitarSet'te tel seçen model, klasik gitarda **"en düşük perde" kuralının bile altında** (0.610 < 0.702);
hataların %96'sı komşu tel. Model naylon tel tınısında teli ayıramıyor ve klasik repertuvarın pozisyon
alışkanlığını bilmiyor — tam da GAPS TAB denetiminin kapatabileceği açık. (Alternatif açıklama: partisyon TAB'ı
icracının çaldığı teli göstermiyor. Adım 1'in GuitarSet koruma ölçüsü bunu ayırt eder — aşağıda.)

## Adım 1 — GAPS partisyon TAB'ı ile tel denetimi (kodlandı, eğitim bekleniyor)
Kod:
- `PitchSeq(..., tab_dir=...)`: chunk'a (L,6) tel sınıfı eklenir; bilinmeyen kareler / etiketsiz kayıt = PAD.
- `train_onset --gaps-tab-weight W`: GAPS parçalarına GuitarSet ile aynı sınıf ağırlıklı tel CE'si (yalnız
  bilinen kareler). Doğrulama logunda `GAPS-tabF1(partisyon)` (gaps_val, bilinen kare, argmax); seçim skoruna
  girer. `W = 0` → eski kod birebir.
- Eğitim yalnız gaps_fit kayıtlarının etiketlerini görür; gaps_val ve gaps_test etiketleri yalnız ölçüm.

### Çalıştırma
```bash
# 0) (yapıldı) etiketler
python -m scripts.data.build_gaps_tab
# 0b) taban tel ölçümü (deep2; ~1 dk)
python -m scripts.eval.eval_gaps_tab --ckpt tabcrnn_deep2.pt --load-cal checkpoints/tabcrnn_deep2.cal.json --splits gaps_val gaps_test
# 1) eğitim: deep2 tarifinin aynısı + partisyon TAB kaybı
python -m scripts.train.train_onset --init tabcrnn_deep2.pt --epochs 10 --ckpt tabcrnn_gtab.pt --lr 1e-4 --onset-soft 0.3 --gs-pitch-weight 1.0 --short-frames 5 --short-weight 2.0 --repeat-weight 3 --pitch-weight 2 --offset-weight 1 --valley-weight 5 --frag-weight 3 --oversample 3 --deep 2 --pitch-onset-weight 1 --gaps-tab-weight 1
# 2) hızlı ölçüm (kalibrasyonsuz; deep2 çözümlemesiyle)
python -m scripts.eval.eval_gaps_tab --ckpt tabcrnn_gtab.pt --load-cal checkpoints/tabcrnn_deep2.cal.json --splits gaps_val gaps_test
python -m scripts.eval.eval_hmm --model crnn --ckpt tabcrnn_gtab.pt --threshold 0.7 --onset-source pitch --onset-thr 0.2 --off-ratio 1.0 --refractory 0 --fallback 10 --peak --reattack 0.3 --rise-keep 0 --rise-split 8 --offset-thr 0.7 --combine noisyor --peak-pick --prominence 0.25 --min-dist 2 --re-valley 0
# 3) yalnız 2) kabulse: tam kalibrasyon (~1 saat) + nota ölçümü
python -m scripts.eval.eval_pitch --ckpt tabcrnn_gtab.pt --splits gaps_test val val_comp --criterion mix --save-cal checkpoints/tabcrnn_gtab.cal.json
```

### Karar kuralı (Adım 1)
- **Ana:** GAPS oracle tel (gaps_val) 0.610 → **≥ 0.75** (naif 0.702'nin belirgin üstü).
- **Koruma — bağımsız gerçek (hex pickup):** GuitarSet oracle tel ≥ **0.868** (0.878 − 0.01) ve GS solo tab F1
  ≥ **0.767**. GAPS oracle artıp GuitarSet oracle 0.01'den fazla düşerse partisyon TAB'ı fiziksel tel kanıtıyla
  çelişiyor demektir → deep2 korunur, TAB yalnız ölçüm/parmaklandırma verisi olarak kalır.
- GAPS nota F1 (3. adımdaki kalibrasyonla) ≥ 0.769 (0.779 − 0.01).

### Adım 1 — sonuç (10 Ekim 2026, `tabcrnn_gtab.pt`, 10 epoch) — KABUL EDİLMEDİ
| ep | GS-tabF1 (argmax) | GAPS-onset (kafa) | GAPS-kare | GAPS-tabF1 (partisyon, gaps_val) | gtab kaybı (eğitim) |
|---|---|---|---|---|---|
| 0 (= deep2) | **0.716** | 0.524 | 0.661 | 0.443 | — |
| 1 | 0.659 | 0.514 | 0.638 | 0.422 | 1.079 |
| 7 | 0.681 | 0.501 | 0.650 | **0.451** | 0.777 |
| 10 | 0.680 | 0.516 | 0.649 | 0.447 | 0.800 |
- Seçim skoru hiçbir epoch'ta ep 0'ı geçmedi (0.597) → **kayıtlı `tabcrnn_gtab.pt` = deep2 ağırlıkları**; bu yüzden
  sonraki `eval_gaps_tab` / `eval_hmm` çıktıları deep2 ile birebir (0.610 / 0.629, GS oracle 0.878, tab F1 0.777).
- Eğitim kaybı %30 düştü (1.08 → 0.75) ama doğrulama icracılarında GAPS tab F1 yalnız +0.004–0.008 → partisyon
  TAB'ı **görülmemiş icracılara genellenmiyor**.
- Koruma ihlali: GS tab F1 (argmax) 0.716 → ~0.68 (−0.035); ilk epoch'ta −0.057. GAPS TAB kaybı GuitarSet'in hex
  (fiziksel) tel kanıtıyla çelişen yöne çekiyor.
- Yorum: partisyon TAB'ı icracının çaldığı teli değil, editörün/transkripsiyoncunun **önerdiği konumu** gösteriyor;
  sesten öğrenilebilir tutarlı bir sinyal değil. **deep2 taban kalır.** TAB, klasik gitar konum alışkanlığının
  (okunabilir TAB hedefi) ve parmaklandırmanın verisi olarak değerli → ses kaybı yerine **önsel** olarak denenmeli
  (öneri: GAPS gaps_fit TAB dizilerinden öğrenilen konum/geçiş önseli, çözümlemede; eğitim yok).

## Adım 2 — "Klasik edisyon önerisi": konum + sol el parmağı (sembolik; kullanıcı önerisi, 10 Ekim)
Sunumun 4. (çalınabilirlik) ve 7. (parmaklandırma) modülleri tek katmanda. Üründe iki mod:
**"duyulan konum"** (ses; GuitarSet'te doğrulanmış, değişmez) ve **"klasik edisyon önerisi"** (bu adım).
İki mod **karıştırılmaz** (kullanıcı kararı, 10 Ekim; 2a/2b'de denenen `w_em` karışımı koddan çıkarıldı).
- Girdi: nota dizisi (perde, başlangıç, bitiş; akor = başlangıçları ≤ 30 ms, akor içinde perde artan).
  Çıktı: her nota için (tel, perde, parmak). Eğitim: gaps_fit partisyonları (127 kayıt, 125 bin nota), ses yok.
- Model (`gtab/decoding/edition.py`): koşullu log-lineer = öğrenilen, açıklanabilir maliyet (sunum slayt 10).
  `ChordPositionModel` = Katman 3.6b geçiş modeli + akor içi Δtel/Δperde blokları;
  `FingerModel` = parmak + el pozisyonu (perde − parmak + 1) değişimi (sıralı / akor içi) + aynı parmakla kayma +
  barre + uzun boşluk. `decode`: (tel, perde, parmak) Viterbi.
  `transitions.npz` / `eval_hmm` değişmez (ayrı dosya `checkpoints/edition.npz`).
- Ölçülen veri gerçekleri: notaların %41'i akorda; akor içinde perde artan sırada tel %97.3 artıyor (%1.7 aynı tel
  = gruplama hatası) → kesin kısıt değil, öğrenilen özellik. Akor içi aynı parmak %2.7, bunun %97'si aynı perde (barre).
- Öğrenilen ağırlıklar (okunur): pozisyonda kal (Δh = 0) +2.7, ±1 +1.3/+1.4; akor içi aynı tel −1.5, bir üst tel
  +1.1; aynı parmakla kayma −1.0; akor içinde aynı parmak farklı perde −2.3.
- Duman testi NLL (val): konum 0.441 (uniform 1.409), parmak 0.981 (uniform 1.386).

### Çalıştırma
```bash
python -m scripts.train.train_edition                      # 2a, ~2 dk -> checkpoints/edition.npz
python -m scripts.train.train_edition_seq                  # 2b, ~2 dk -> checkpoints/edition_seq.pt
python -m scripts.eval.eval_edition --edition edition.npz --seq edition_seq.pt --ckpt tabcrnn_deep2.pt
```

### Karar kuralı (kullanıcı)
- Tel uyumu, gaps_test partisyonları: EDİSYON ≥ naif + 0.05 (naif 0.745 → **≥ 0.795**).
- Parmak doğruluğu: EDİSYON > taban "parmak = perde − pozisyon + 1" (ikisi de partisyon konumu verilerek).
- Ses modunda kayıp yok: GuitarSet "ses / greedy" **0.878** (kod yolu değişmedi; ölçümde tekrar edilir).
- (2a/2b'de) `w_em` karışımı gaps_val'de seçildi ve raporlandı; sonra kaldırıldı (aşağıda "Ek ölçümler").

### Adım 2a sonucu — log-lineer aday (`checkpoints/edition.npz`, 10 Ekim) — KABUL EDİLMEDİ
| yöntem | gaps_val | gaps_test | GuitarSet val |
|---|---|---|---|
| tel: naif (en düşük perde) | 0.702 | 0.745 | 0.206 |
| tel: GuitarSet önseli (3.6b, ses yok) | 0.489 | 0.496 | 0.662 |
| tel: **EDİSYON log-lineer (ses yok)** | 0.740 | **0.785** (+0.040; ölçüt ≥ 0.795 ✗) | 0.303 |
| tel: ses / greedy (duyulan konum) | 0.610 | 0.629 | **0.878** ✓ |
| tel: ses + edisyon (w_em 0.1, gaps_val'de) | 0.745 | 0.792 | 0.642 |
| parmak: taban "perde − pozisyon + 1" (partisyon konumu) | **0.642** | **0.566** | |
| parmak: EDİSYON (partisyon konumu) | 0.555 | 0.483 (✗ < taban) | |
| tam (tel + perde + parmak, kendi konumu) | 0.310 | 0.312 | |
(parmaklı nota: gaps_val 3851, gaps_test 4466.)
- GAPS önseli GuitarSet önselinden çok farklı (GAPS'te 0.49 vs 0.79): klasik edisyon alışkanlığı gerçekten ayrı bir
  hedef. GuitarSet'te edisyon 0.303 → "duyulan konum" ile "edisyon önerisi" ayrı modlar olmalı (beklendiği gibi).
- Ses ağırlığı arttıkça GAPS uyumu düşüyor (w_em 0.1: 0.745 → 2.0: 0.656): GAPS'te ses edisyon tercihini
  taşımıyor (Adım 1 bulgusuyla tutarlı).
- **Neden yetmedi:** log-lineer model yalnız ÖNCEKİ notayı görür (soldan sağa, Markov-1). Parmak ve pozisyon
  pasajın ileri ve geri bağlamıyla belirlenir; iki yönlü ±3 nota penceresi kullanan basit taban bu yüzden parmakta
  önde. → Kullanıcı planındaki yedek: **notalar üzerinde küçük iki yönlü dizi modeli** (Adım 2b).

### Adım 2b — iki yönlü dizi modeli (`gtab/models/edition_net.py`, `checkpoints/edition_seq.pt`) — KABUL
- Aşama 1 (konum): nota özellikleri (perde gömmesi; akor bayrağı/boyu/sırası; başlangıç aralığı; süre; perde farkı;
  geçerli teller) → 2 katman BiLSTM (128) → tel (yalnız geçerli teller; perde = pitch − açık tel). Akor içinde
  aynı tel iki notaya verilmez (olasılık sırasıyla). Aşama 2 (parmak): aynı özellikler + konum (tel, perde one-hot)
  → BiLSTM → parmak (boş tel = 0, basılı = 1–4). Ses kullanılmaz.
- Eğitim `scripts/train/train_edition_seq.py`: gaps_fit, 256 notalık kesitler, Adam 1e-3, 30 epoch (~2 dk, GPU);
  aşama 2'ye partisyon konumu (etiketsiz notada modelin tahmini). Seçim gaps_val (tel + parmak ortalaması) → ep 17.
  Parmak kaybı 10. epoch'tan sonra aşırı öğreniyor (0.09'a iniyor, gaps_val düşüyor); seçim bunu yakalıyor.

| yöntem | gaps_val | **gaps_test** | GuitarSet val |
|---|---|---|---|
| tel: naif | 0.702 | 0.745 | 0.206 |
| tel: EDİSYON log-lineer (2a) | 0.740 | 0.785 | 0.303 |
| tel: **EDİSYON dizi modeli (ses yok)** | 0.766 | **0.815** (+0.070 ✓) | 0.425 |
| tel: ses + edisyon dizi (w_em 0.25, gaps_val'de) | 0.768 | **0.822** | 0.573 |
| tel: ses / greedy (duyulan konum) | 0.610 | 0.629 | **0.878** ✓ |
| parmak: taban "perde − pozisyon + 1" | 0.642 | 0.566 | |
| parmak: **EDİSYON dizi modeli** (partisyon konumu) | 0.675 | **0.636** (+0.070 ✓) | |
| tam: tel + perde + parmak (kendi konumu) | 0.382 | 0.434 | |

**Karar — Adım 2 kabul (dizi modeli):** gaps_test tel uyumu 0.815 ≥ 0.795, parmak 0.636 > taban 0.566, ses modu
GuitarSet 0.878 değişmedi. Ürün: iki mod — "duyulan konum" (ses; GuitarSet 0.878) ve "klasik edisyon önerisi"
(dizi modeli; GAPS 0.815 tel, 0.636 parmak). Birleştirme GAPS'te +0.007 (0.822) ama GuitarSet'te 0.878 → 0.573;
ses ile edisyon farklı hedefler → **birleştirme varsayılan değil**, ses güveni eşiğiyle ileride (ürün kararı).
Notlar: parmak ölçümü yalnız partisyonda parmak numaralı notalarda (editörlerin işaretlediği, çoğunlukla
"belirgin olmayan" yerler); tek doğru parmaklandırma yok (sunum slayt 10) → değerler alt sınır niteliğinde.

### Adım 2b — ek ölçümler (kullanıcı isteği, 10 Ekim; `eval_edition` yeniden yazıldı)
1. **Modlar ayrıldı:** ses + edisyon karışımı GuitarSet'te duyulan teli 0.878 → 0.573 düşürdüğü için kaldırıldı
   (`edition.decode`, `edition_net.predict`, `eval_edition`). Bölüm 1 ve 2a sayıları önceki koşuyla birebir aynı
   (regresyon ✓).
2. **Ürün parmak doğruluğu (adil):** 0.636 partisyon konumu verilerek ölçülmüştü. Ürün sayısı = konumu yöntemin
   kendisi seçer, doğru = tel + perde + parmak; taban da AYNI konum üstünde ölçüldü.

| gaps_val / **gaps_test** (gerçek notalar, parmaklı basılı notalar) | gaps_val | **gaps_test** |
|---|---|---|
| naif konum + taban parmak (tamamen kural) | 0.305 | 0.323 |
| dizi modeli konumu + taban parmak | 0.348 | 0.376 |
| EDİSYON log-lineer (konum + parmak) | 0.310 | 0.312 |
| **EDİSYON dizi modeli (konum + parmak)** | **0.382** | **0.434** (aynı konumdaki tabandan +0.058, kuraldan +0.111) |

   Önceki "0.434 < 0.566" karşılaştırması farklı koşullardaydı (biri partisyon konumu, diğeri kendi konumu). Aynı
   koşulda model tabanı geçiyor. Mutlak değer düşük: parmaklı notalar editörün işaretlediği "zor" yerler, orada konum
   da daha sık yanlış; tam doğruluk iki hatanın birleşimi.
3. **Uçtan uca (gaps_test, bir kez):** deep2'nin sesten bulduğu notalar (çözümleme `tabcrnn_deep2.cal.json` gaps_val
   seçimi) → yöntemler → partisyon etiketli notalarla eşleme (aynı perde, ±50 ms). deep2 etiketli notaların
   **%81.7**'sini (11500 / 14082), parmaklıların %83.2'sini buldu.

| yöntem (deep2 notaları) | tel: bulunanlarda | tel: tüm etiketlilerde | tam parmak: bulunanlarda | tam parmak: tüm parmaklılarda |
|---|---|---|---|---|
| naif (en düşük perde) + taban parmak | 0.747 | 0.610 | 0.312 | 0.259 |
| **EDİSYON dizi modeli** [edisyon önerisi] | **0.802** | **0.655** | **0.397** | **0.331** |
| dizi konumu + taban parmak | 0.802 | 0.655 | 0.348 | 0.290 |
| ses / greedy [duyulan konum] + taban parmak | 0.620 | 0.507 | 0.320 | 0.266 |

   Bulunan notalarda edisyon tel uyumu gerçek notalardakine yakın (0.815 → 0.802; hayalet notalar bağlamı az bozuyor),
   naiften farkı korunuyor (+0.055). "Tüm etiketlilerde" sütunu ürünün uçtan uca sayısıdır; kaybın çoğu nota
   bulmadan (%18 kaçan nota), konum/parmak modülünden değil.

## Parmak numarası
Etiketli notaların %33.6'sında sol el parmağı var (`finger`, `finger_note`); bağlama tutarlılığı %99.4. Bu
katmanda kullanılmıyor; parmaklandırma katmanının (sunum slayt 10–11) girdisi olarak saklandı.

## Testler
- Eşleme: 300 kayıtta etiket kapsamı / zaman farkı / tekrar tutarlılığı (yukarıdaki tablo); perde roll'u uyumu
  %98.93; parmak bağlama %99.4.
- `eval_gaps_tab` duman testi (gaps_val, 40 sn).
- `train_onset` regresyonu (küçük kurulum: GuitarSet val + gaps_test, 1 epoch, deep2 tarifi): `--gaps-tab-weight 0`
  kayıplar ve ep 0 HEAD ile birebir; ep 1 ölçüleri HEAD'in ikinci koşusuyla birebir (HEAD kendi içinde ±0.002
  oynuyor = GPU belirsizliği). `--gaps-tab-weight 1` uçtan uca çalışıyor: `gtab` kaybı 1.23, log'da
  `GAPS-tabF1(partisyon)` 0.398 (ep 0).

## Durum
- [x] Adım 0 — hizalama + etiketler (`gaps_score`, `build_gaps_tab`): 165 kayıt, %81 nota etiketli
- [x] Adım 0 — tel ölçümü (deep2, gaps_val): oracle tel 0.610 < naif 0.702 → tel denetimi gerekli
- [x] Adım 1 — kod (`PitchSeq tab_dir`, `--gaps-tab-weight`) + testler
- [x] Adım 1 — eğitim `tabcrnn_gtab.pt`: genelleme yok (+0.004), GS tab −0.035 → **kabul edilmedi**, deep2 taban
- [x] Adım 2a — log-lineer edisyon modeli (`edition.npz`): tel 0.785, parmak 0.483 < taban → yetmedi
- [x] Adım 2b — iki yönlü dizi modeli (`edition_seq.pt`): gaps_test tel **0.815**, parmak **0.636** (taban 0.566),
  ses modu 0.878 → **kabul**
- [x] Adım 2b — ek ölçümler: modlar ayrıldı (karışım kaldırıldı); ürün parmak (tam) 0.434 > aynı konumda taban
  0.376; uçtan uca (deep2 notaları) tel 0.802 / 0.655, tam parmak 0.397 / 0.331 ← **KALDIĞIMIZ YER** (commit / Katman 4)
- [ ] Sonra: tekrarı tutarsız 75 kayıt (ölçü düzeyi hizalama), Katman 4 (teknikler)
