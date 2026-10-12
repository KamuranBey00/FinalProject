# Katman 5 — Ritim ve kuantalama (sunum modül 5: "Beat / downbeat → müzikal süre ve ölçü")

> Durum (11 Ekim 2026): **Adım 1q KABUL (`rhythm_q.pt`; gaps_test oracle IOI 0.933, uçtan uca 0.617). Adım 1b (öğrenilen
> vuruş + DBN) kodlandı; tam otomatik hedefin altında → kullanıcı kararı: yarı otomatik (BPM + ölçü girdisi), deneme uçtan uca
> 0.695 (hedef 0.833); kullanıcı kararı: bu hâliyle Adım 2'ye (README16), tap modu yedek.** Kullanıcı kararıyla
> Katman 4 (teknikler) sonraya bırakıldı; ritim Katman 3.13'ün (edisyon önerisi) üstüne eklenir.
> Ses tabanı değişmez: `checkpoints/tabcrnn_deep2.pt` (+ `.cal.json`), notalar onun çözümlemesinden.

## Neden
Şimdiye kadar çıktı "hangi anda hangi tel/perde" (saniye cinsinden). Okunabilir bir TAB / nota için her notanın
**müzikal konumu** (ölçü, vuruş, vuruş içi bölüm) ve **nota değeri** gerekir. Sunum bunu ayrı bir modül olarak
tanımlıyor ("ham onset sürelerini müzikal ritme dönüştürmek ayrı bir kuantalama problemi"). Klasik gitarda zor
taraf rubato: tempo sabit değil, vuruşlar icracıyla esner; akorlar arpejlenir, süsler vuruştan önce çalınır.

Problemi üç alt modüle ayırıyoruz (her biri ayrı ölçülür; sunum slayt 8 "her aşamanın kendi metriği"):
1. **Vuruş takibi** — ses / notalar → vuruş zamanları (tempo eğrisi).
2. **Ölçü başı + ölçü** — vuruşlar → ölçü başları ve ölçüdeki vuruş sayısı (2 / 3 / 4).
3. **Kuantalama** — nota başlangıçları + vuruşlar → vuruş biriminde konum `k + m/d` (d = vuruş içi bölüm:
   1, 2, 3, 4, 6, 8, 12; ikili ve üçlemeli aynı aramada) → nota değerleri.

## Adım 0a — Gerçek ritim etiketleri (kodlandı + çalıştırıldı)
**GAPS (klasik gitar, rubato'lu gerçek icra):** `align()` (Katman 3.13) her MIDI notasını partisyondaki konumuna
eşliyordu; artık bu konumu da döndürüyor (`score_x`, açılmış baştan tick; mevcut çıktılar birebir aynı, aşağıda test).
Eşleşen notalar = (partisyon tick'i, ses zamanı) çapaları → monoton zaman haritası (aynı tick'te en erken başlangıç =
arpej başı; sırayı bozan çapa atılır). Vuruş ve ölçü başı tick'leri MusicXML ölçü imlerinden (`<time>`, ilk portrede)
çıkar, haritayla saniyeye çevrilir → **icracının gerçek, rubato'lu vuruşları**. Ayrıca her notanın vuruş biriminde
partisyon konumu (`note_beat`) = kuantalamanın doğru cevabı.
- Vuruş birimi: x/4 → çeyrek, x/2 → ikilik, x/8 ve pay 3'ün katı → noktalı çeyrek (bileşik; 3/8 = ölçüde 1 vuruş),
  diğer x/8 → sekizlik. İlk ölçü kendi iminden kısaysa ya da ayrı kısa bir imle (1/4, 1/8 …) yazılmışsa anakruz:
  vuruşları ölçü sonuna hizalanır, başı ölçü başı sayılmaz.
- Vuruşlar yalnız çapaların kapsadığı aralıkta tutulur (dışına enterpolasyon yok).
- Kullanılır kayıt: tekrar yapısı tutarlı (Katman 3.13 ile aynı ölçüt), eşleşen nota ≥ %50, ≥ 8 vuruş.

**GuitarSet:** JAMS `beat_position` (metronomla çalınmış; `position == 1` = ölçü başı) + `tempo`.

Kod: `gtab/data/rhythm_labels.py`, `gtab/data/gaps_score.py` (`align` → `order`, `lens`, `score_x`;
`time_signatures`), `scripts/data/build_rhythm.py` → `data/cache/rhythm/<id>.npz` + `summary.csv`
(mevcut önbelleğe dokunulmaz).

Sonuç (10 Ekim 2026):
| | değer |
|---|---|
| GAPS kullanılır kayıt | gaps_train **143/231**, gaps_test **23/30** (tekrar tutarsız 70 + 5) → gaps_val 15, gaps_test 23 |
| vuruş / ölçü başı | 40 638 / 14 674 |
| eşleşen (konumu bilinen) nota | %80.0 |
| vuruşta başlayan nota | %46.3 (kalanı vuruş içi bölümlerde) |
| **haritadan ölçü başı vs bağımsız syncpoint'ler** | medyan **22.9 ms** (kayıt başına medyanların medyanı) |
| tempo (vuruş birimi) | medyan 83 BPM, aralık 26–222 |
| GuitarSet | 360 kayıt, 21 120 vuruş, 5 280 ölçü başı |

Syncpoint'ler (ölçü düzeyi, MIDI'den bağımsız el/video hizalaması) ile nota çapalı ölçü başları ~23 ms uyumlu →
etiket zaman haritası güvenilir (mir_eval vuruş penceresi ±70 ms).

## Ölçüm tanımları (`gtab/evaluation/rhythm_metrics.py`)
- **Vuruş F** (mir_eval, ±70 ms; ilk 5 sn kırpılır), **CMLt** (doğru metrik düzeyde süreklilik), **AMLt** (yarım /
  çift tempo ve ters faz da kabul). Ölçü başı F aynı fonksiyonla. Tempo Acc1 (±%4), Acc2 (×1/3, 1/2, 2, 3 kabul).
- **IOI doğruluğu (ana kuantalama ölçütü, "nota değerleri doğru mu?"):** partisyonda aynı konumdaki notalar bir
  başlangıç grubu; ardışık iki grup arasındaki kuantalanmış süre (vuruş biriminde) partisyondaki süreye eşit mi.
  Tahmini vuruşta kayıt başına tek ölçek r ∈ {1/4 … 4} seçilir (vuruş takibi farklı metrik düzeyi seçebilir; faz ve
  vuruş numarasından bağımsız). Oracle vuruşta r = 1 sabit.
- **Konum doğruluğu** (oracle vuruşta): kuantalanmış konum = partisyon konumu. **Akor birlikte:** partisyonda aynı
  konumdaki ≥ 2 notanın tek konuma düşme oranı (arpej bölünmesi). **Izgara kapsamı:** partisyon konumlarının
  bölüm sözlüğüyle gösterilebilen oranı.
- Tahmini vuruş / ölçü başı yalnız etiketli aralıkta (± yarım vuruş) puanlanır (GAPS etiketi eşleşen notalarla sınırlı).

## Adım 0b — Eğitimsiz taban (kodlandı)
`gtab/decoding/rhythm.py`:
- **Zarflar:** `cqt` = CQT spektral akı (ses), `model` = deep2 perde-onset kafasının perdeler üstü toplamı,
  `notes` = deep2 nota başlangıçlarından darbe dizisi; oracle `notes_gt` = gerçek notalar (sembolik tavan).
- **Vuruş izleyici:** `dp` = librosa dinamik programlama (tek tempo + `tightness` sapma payı 30 / 100 / 400),
  `plp` = baskın yerel nabız (tempo değişebilir; uçları dp ile aynı kuralla kırpılır).
- **Ölçü başı:** vuruş vurgusu = ±70 ms içinde başlayan nota sayısı + w × pes nota (kaydın alt %20 perdesi);
  ölçü ∈ {2, 3, 4} ve faz, vurgu farkını en çok artıran. w ∈ {0, 1, 3} gaps_val'de seçilir.
- **Kuantalayıcı:** akor (ilk notadan ≤ tol) tek konum; her vuruşta bölüm d,
  `Σ|u − m/d| + λ·karmaşıklık(d) + μ·çakışma` en küçük. λ, μ, tol gaps_val'de (oracle vuruş + gerçek nota) seçilir.

`scripts/eval/eval_rhythm.py`: A) kuantalayıcı tavanı + hata kırılımı, B) vuruş takibi (gaps_val'de seçim; gaps_test
ve GuitarSet val yalnız seçilenle), C) ölçü başı, D) kuantalama zinciri (oracle → uçtan uca, dört satır).

### Testler
- `align` değişikliği: 9 kayıtta tel / perde / parmak / nota dizileri `data/cache/gaps_tab` ile **birebir** ✓.
- Sentetik ritim (rubato'lu vuruşlar, sekizlik / onaltılık / üçleme / akor, ±12 ms sapma): oracle vuruşta konum,
  IOI ve akor 102/102, 95/95, 6/6 ✓; yarım tempo vuruşlarında IOI 95/95 ve r = 2 seçiliyor ✓; ölçü başı
  (pes nota ölçü başında, 4/4) ölçü 4 ve faz doğru ✓.
- `eval_rhythm --limit 2` uçtan uca çalışıyor (15 sn).

### Çalıştırma
```bash
python -m scripts.data.build_rhythm          # ~3 dk, yapıldı
python -m scripts.eval.eval_rhythm           # deep2 + cal varsayılan; birkaç dakika
```

## Karar kuralı (Adım 0 → sonraki adım)
- **Kuantalayıcı tavanı** (A, gaps_test IOI, gerçek vuruş + gerçek nota) **≥ 0.90** → kuantalayıcı yeterli.
  Altındaysa → **Adım 1q:** bağlamlı kuantalayıcı (hata kırılımına göre).
- **Vuruş takibi** (B, seçilen eğitimsiz yöntem, gaps_test) F **≥ 0.75** **ve** uçtan uca IOI (D) ≥ tavan − 0.10 →
  eğitimsiz yol yeterli → Adım 2 (ölçülü TAB çıktısı). Değilse → **Adım 1b:** öğrenilen vuruş + ölçü başı izleyici.
- **Ölçü başı** (C, gerçek vuruş + deep2 notası) F ≥ 0.80 → kural yeterli; değilse Adım 1b ölçü başını da öğrenir.

## Sonuçlar — Adım 0 (10 Ekim 2026, `eval_rhythm`, deep2 + cal; gaps_val 15, gaps_test 23, GS val 30 kayıt; ~6 dk)

### A) Kuantalayıcı tavanı (gerçek vuruş + gerçek nota)
Seçim (gaps_val, 105 aday): λ 0.2, μ 0.3, akor tol 0.03 (en kötü aday 0.085 → parametreler önemli; seçim uçta değil).

| | IOI | konum | akor birlikte | ızgara kapsamı |
|---|---|---|---|---|
| gaps_val | 0.819 (n = 10 657) | 0.882 | 0.926 | 1.000 |
| **gaps_test** | **0.821** (n = 9 848) | 0.896 | 0.949 | 0.997 |

IOI hata oranı, partisyon süresine göre (gaps_test): < 1/4 vuruş **%20.8**, 1/4–1/2 %14.1, 1/2–1 %20.3, > 1 %17.4;
hataların çoğu "kısa / uzun tahmin" (birleşme ≤ %13). Izgara kapsamı ~%100 → **sözlük yetiyor, sorun vuruş içi icra
sapması**: her vuruşu bağımsız kuantalayan kural, hızlı notalarda rubato / arpej / süs kaymasını bağlamsız çözemiyor.

### B) Vuruş takibi (ortalama F / CMLt / AMLt / tempo Acc1 / Acc2)
| gaps_val | dp t=30 | dp t=100 | dp t=400 | plp |
|---|---|---|---|---|
| cqt (ses) | 0.427 / .233 / .352 | 0.394 | 0.370 | **0.478** / .096 / .273 |
| model (perde-onset kafası) | 0.456 / .260 / **.414** | 0.435 | 0.414 | 0.477 |
| notes (deep2 notaları) | 0.427 | 0.421 | 0.395 | 0.460 |
| notes_gt (gerçek notalar, oracle) | 0.432 | 0.409 | 0.374 | 0.467 |

Seçilen (gaps_val F): **cqt + plp**. gaps_test **F 0.444**, CMLt 0.072, AMLt 0.257, Acc2 0.30; GuitarSet val (metronom,
GAPS seçimi aktarıldı) F 0.394, CMLt 0.022. Gerçek notalar zarfı da aynı yerde (0.439) → **darboğaz nota bulma değil,
izleyicinin kendisi**: sabit tempo (dp) rubato'yu izleyemiyor, plp yerel tempoyu izliyor ama metrik düzeyi / fazı
tutturamıyor (CMLt ~0.1). Tempo Acc1 %0–33.

### C) Ölçü başı (F; ölçü doğruluğu yalnız gerçek vuruşta) — pes nota ağırlığı 1 (gaps_val)
| | gerçek vuruş + gerçek nota | gerçek vuruş + deep2 notası | tahmini vuruş + deep2 notası |
|---|---|---|---|
| gaps_val | 0.647 (ölçü 0.53) | 0.597 (0.47) | 0.209 |
| gaps_test | 0.822 (0.70) | **0.789** (0.70) | 0.190 |
| GuitarSet val | 0.393 (0.90) | 0.376 (0.97) | 0.109 |
GuitarSet'te ölçü (4) doğru ama faz yanlış (solo doğaçlamada vurgu ölçü başında değil) → kural klasik gitarda kısmen
çalışıyor, genel değil.

### D) Kuantalama zinciri (IOI doğruluğu)
| | gaps_val | **gaps_test** | kapsama (test) | r = 1 (test) |
|---|---|---|---|---|
| gerçek vuruş + gerçek nota (tavan) | 0.819 | **0.821** | 1.000 | 1.00 |
| gerçek vuruş + deep2 notası | 0.778 | 0.799 | 0.810 | 1.00 |
| tahmini vuruş + gerçek nota | 0.516 | 0.581 | 1.000 | 0.17 |
| **tahmini vuruş + deep2 (uçtan uca)** | 0.544 | **0.588** | 0.810 | 0.13 |

Kayıp ayrışması (gaps_test): kuantalayıcı 1.000 → 0.821 (−0.18), deep2 notaları −0.02 (bulunan notalarda; %19 nota
kaçıyor), **vuruş takibi −0.21–0.24 (en büyük kayıp)**. Kayıtların yalnız %13–27'sinde vuruş takibi partisyonun
metrik düzeyini tutturuyor.

### Karar (Adım 0)
| ölçüt | eşik | sonuç |
|---|---|---|
| kuantalayıcı tavanı (A, gaps_test IOI) | ≥ 0.90 | **0.821 ✗** → Adım 1q |
| vuruş F (B, gaps_test) | ≥ 0.75 | **0.444 ✗** → Adım 1b |
| uçtan uca IOI ≥ tavan − 0.10 | ≥ 0.721 | **0.588 ✗** |
| ölçü başı (C, gerçek vuruş + deep2, gaps_test) | ≥ 0.80 | **0.789 ✗** (sınırda; val 0.597) → 1b ölçü başını da öğrenir |
**Eğitimsiz yol yetmiyor; iki ayrı öğrenme adımı gerekiyor.** deep2 notaları ritim için yeterli (−0.02).

## Önerilen sıradaki adımlar (kullanıcı kararı)
### Adım 1b — Öğrenilen vuruş + ölçü başı izleyici (en büyük kayıp: −0.23 IOI)
- Veri: GAPS gaps_fit ritim etiketleri (~128 kayıt, rubato) + GuitarSet train solo + comp (300 kayıt, metronom);
  seçim gaps_val, gaps_test yalnız rapor (mevcut bölmeler; ses modeli aynen).
- Girdi: deep2'nin donmuş ara temsili (BiLSTM çıkışı) + onset kafaları → küçük BiLSTM / TCN → kare başına vuruş ve
  ölçü başı olasılığı. (Alternatif: CQT'den sıfırdan; deep2 temsili nota bilgisini zaten taşıyor, eğitim ucuz.)
- Çözümleme: "bar pointer" DBN/HMM (tempo × ölçü içi konum × ölçü 2/3/4; tempo değişimine izin) — rubato'yu ve
  metrik düzeyi birlikte çözer; numpy ile yazılır (madmom Windows/py3.10'da sorunlu).
- Kabul (öneri): gaps_test vuruş F ≥ 0.75, ölçü başı F ≥ 0.60, uçtan uca IOI ≥ tavan − 0.10; GuitarSet val vuruş F
  raporlanır (metronomda ≥ 0.85 beklenir).

### Adım 1q — Bağlamlı kuantalayıcı (−0.18; sembolik, dakikalar)
- Kayıp kısa notalarda ve yönsüz (kısa / uzun) → bağlam gerekli. Edisyon modeli (`edition_net`) gibi başlangıç grupları
  üstünde iki yönlü dizi modeli: girdi = vuruşa göre sürekli konum, komşu IOI'ler, akor boyu; çıktı = vuruş içi ızgara
  noktası (12'lik + 8'lik birleşik sözlük). Eğitim gaps_fit (gerçek vuruşlar + vuruşlara gürültü eklenerek tahmini
  vuruşa dayanıklılık). Ses yok.
- Kabul (öneri): gaps_test oracle IOI ≥ 0.90 ve uçtan uca IOI'de kuraldan kötü değil.

Sıra önerisi: **1q önce** (ucuz, bağımsız ölçülür, 1b'nin kabul ölçütü "tavan − 0.10" ona bağlı), **sonra 1b**.
**Kullanıcı kararı (11 Ekim): Adım 0 sonuçları kullanıcının çalıştırmasında birebir aynı → 1q ile devam.**

## Adım 1q — Bağlamlı kuantalayıcı (kodlandı, 11 Ekim)
### Ön ölçüm (hedef temsili, gaps_fit / gaps_val, gerçek vuruş)
- Notanın sürekli konumu u ile partisyon konumu s arasındaki fark (1/24 vuruş = 1 tick): %66 sıfır, **%97 ≤ 2 tick**,
  %100 ≤ 12 tick (yarım vuruş). Partisyon konumlarının %99.4'ü 1/24 ızgarasında. → Kuralın hatası "nota uzakta" değil,
  **yanlış vuruş içi bölüm** (ör. 1/6 yerine 1/4).
- Uyarı: gerçek vuruşlar aynı notalardan (çapalar) türetiliyor → oracle girdisi fazla temiz; eğitimde vuruşlar bozulur.

### Model (`gtab/models/rhythm_net.py`)
- Notalar üzerinde 2 katmanlı BiLSTM (128). Nota başına girdi: en yakın tick c'nin vuruş içi yeri (one-hot 24) + artık,
  yerel vuruş aralığı, komşu notalara uzaklık (vuruş ve saniye), perde, süre, **±8 vuruştaki notaların vuruş içi tick
  dağılımı** (yerel histogram). Çıktı: delta ∈ [−12, 12] sınıfı → konum (c + delta) / 24.
- **Yapısal çözümleme** (`decode`): her vuruşta ortak bölüm d ∈ {1, 2, 3, 4, 6, 8, 12}; nota başına d ızgarasındaki en
  olası tick; d = argmax Σ log p − λ·karmaşıklık(d). λ eğitim sonunda gaps_val'de seçilip checkpoint'e yazılır.
- Eğitim (`scripts/train/train_rhythm_q.py`): yalnız gaps_fit; kayıtların yarısı her epoch bozulur (vuruş gürültüsü
  σ ~ U(0, 0.08) vuruş + sabit kayma ±0.1 + yarım / çift tempo %15 / %15; nota sapması 10 ms, düşürme ≤ %20, sahte nota
  ≤ %10). CE, Adam 1e-3 + kosinüs, 80 epoch; epoch seçimi gaps_val oracle IOI. Çıktı `checkpoints/rhythm_q.pt`.
- `eval_rhythm --qnet rhythm_q.pt`: A ve D'de kuralın yanında "1q" satırları (`--qnet` yoksa çıktı Adım 0 ile birebir).

### Tasarım denemeleri (gaps_val IOI, ölçek sabit; "gürültülü" = sabit tohumlu vuruş gürültüsü σ 0.05 + kayma ±0.1)
| deneme | oracle | gürültülü vuruş | not |
|---|---|---|---|
| kural (Adım 0) | 0.819 | 0.614 | |
| ağ, artırmasız, 15 ep, argmax | 0.815 | 0.546 | nota konumu 0.903 > kural 0.882 ama IOI ≈ konum² |
| + yerel tick histogramı | 0.840 | 0.534 | hatalar: üçlü konum → ikili ızgara (20→18, 8→6, 16→18, 4→6), 5 üçlemeli kayıtta yoğun |
| + yapısal çözümleme (λ 1) | 0.872 | — | yeniden eğitim yok |
| artırmasız, 40 ep | **0.928** | 0.517 | temiz vuruşa aşırı uyum (kuraldan kırılgan) |
| artırmalı (temiz oran 0.3), 40 ep | 0.881 | 0.767 | gaps_test: oracle **0.907**, deep2 0.910, uçtan uca 0.618 (kural 0.588) |
| artırmalı 0.3, 80 ep + kosinüs | 0.897 | **0.789** | |
| **artırmalı 0.5, 80 ep + kosinüs (varsayılan)** | **0.923** | 0.766 | |
Gözlem: gaps_val gaps_fit'ten daha üçlemeli (üçlü ağırlıklı kayıt 5/15'e karşı 22/128); 275_Lw1wc'de partisyondaki
akorlar hızlı tarama olarak çalınmış (etiket–icra eşlemesi kısmen gürültülü, düzeltilemez pay).

### Testler
- `--qnet` olmadan `eval_rhythm --limit 3` çıktısı değişiklik öncesiyle **birebir** ✓ (kuantalayıcı fonksiyon olarak geçiyor).
- Yarım / çift tempo dönüşümünde gürültüsüz hedeflerin %99.8'i geçerli ✓; doğru olasılıklar verilince `decode` konumları
  birebir geri veriyor ✓; eğitimdeki kural tabanı 0.819 = Adım 0 A bölümü ✓.

### Çalıştırma
```bash
python -m scripts.train.train_rhythm_q                       # ~11 dk -> checkpoints/rhythm_q.pt
python -m scripts.eval.eval_rhythm --qnet rhythm_q.pt        # ~6 dk
```

### Karar kuralı (1q)
- **Kabul:** gaps_test oracle IOI (A, 1q) **≥ 0.90** **ve** uçtan uca IOI (D, 1q) ≥ kural (0.588). Ek bilgi: gerçek vuruş +
  deep2 satırı, akor birlikte.
- Kabul → 1q varsayılan kuantalayıcı; **1b'nin "tavan − 0.10" ölçütü yeni tavanla** (1q oracle) hesaplanır → sırada 1b.
- Oracle < 0.90 ama uçtan uca iyiyse → kullanıcı kararı (ölçüt 0.90 ölçümden önce konuldu).

### Sonuç — Adım 1q (11 Ekim, kullanıcı çalıştırması; `rhythm_q.pt`, λ 1.0) — **KABUL**
Eğitim deneme ile birebir (gaps_val oracle 0.923, gürültülü vuruş 0.766).

| gaps_test (IOI) | kural | **1q** |
|---|---|---|
| gerçek vuruş + gerçek nota (tavan) | 0.821 (konum 0.896, akor 0.949) | **0.933** (konum 0.967, akor 0.989) |
| gerçek vuruş + deep2 notası | 0.799 | **0.927** |
| tahmini vuruş (cqt + plp) + deep2 (uçtan uca) | 0.588 | **0.617** |
gaps_val: 0.819 → 0.923, 0.778 → 0.890, uçtan uca 0.544 → 0.566. Kalan hata en çok < 1/4 vuruşta (%9.6; çoğu "uzun tahmin"
ve birleşme = arpej / süs). **Kabul: oracle 0.933 ≥ 0.90 ✓, uçtan uca 0.617 ≥ 0.588 ✓ → 1q varsayılan kuantalayıcı.**
Kuantalama artık bulunan notalarda neredeyse çözüldü; uçtan uca kaybın tamamı vuruş takibinde (0.927 → 0.617, metrik
düzey kayıtların yalnız %17'sinde doğru). 1b hedefi: uçtan uca ≥ **0.833** (= 0.933 − 0.10).

## Adım 1b — Öğrenilen vuruş + ölçü başı (kodlandı, 11 Ekim; deneme sonuçları — kullanıcı kararı bekleniyor)
### Ön ölçüm
- Ölçü başına vuruş: GAPS eğitim 1: %5.7 (3/8 gibi), 2: %43.8, 3: %21.4, 4: %27.2 (test 1: %13.9); GuitarSet hep 4.
- Yerel tempo (vuruş birimi) %1–%99: 25–260 BPM. Eğitim sesi: gaps_train 7.9 sa + GuitarSet train/comp/val 3 sa.

### Kod
- `gtab/decoding/beat_dbn.py`: numpy bar-pointer DBN (ölçü 1–4 ayrı çözülür, en olası seçilir; tempo 25–260 BPM, 50 log
  aralıklı tempo; tempo yalnız vuruş sınırında değişir, `trans_lambda`). **madmom'dan fark:** vuruş bölgesi tempoyla
  genişlemiyor (tek kare); `obs_lambda` yalnız vuruş başına log(obs_lambda − 1) ödülü. Sentetik test: madmom bölgesiyle yol
  yumuşak tepelerde 2–3 kat yavaş tempoya kilitleniyordu (vuruş F 0.42–0.53); tek kare bölgeyle 6 rubato'lu durumda
  (ölçü 1–4, 45–200 BPM, ±%20 tempo, %15 eksik vuruş, sahte tepeler) vuruş F 0.977, ölçü başı 0.981, ölçü 6/6.
  `peak_beats`: DBN'siz taban (tepe seçimi).
- `gtab/models/beat_net.py`: deep2 gövdesinin donmuş temsili (`encode`, 256) → Linear + 2 katmanlı BiLSTM (128) → kare başına
  (yok / vuruş / ölçü başı) softmax; hedef ±1 kare 0.5 yumuşatılmış.
- `scripts/data/build_beat_feats.py` → `data/cache/beat_feats/tabcrnn_deep2/` (469 kayıt, 10.6 sa, 808 MB; gaps_test yok).
- `scripts/train/train_beat.py`: gaps_fit + GuitarSet train + train_comp (GAPS'te etiketli aralık dışı maskeli); 35 sn kesitler;
  `--stretch` tempo artırması (kesit ×0.8–1.25 yeniden örneklenir, etiketler aynı oranla); epoch seçimi gaps_val tepe seçimi
  (vuruş F + ölçü başı F)/2; sonunda DBN ayarları (trans 3/10/30 × obs 16/32/64/128) gaps_val'de taranıp checkpoint'e yazılır.
- `eval_rhythm --beatnet beat_net.pt`: B2 bölümü (tepe seçimi / DBN: F, CMLt, AMLt, tempo, ölçü başı F, ölçü) + D'de
  "ogrenilen vurus" satırları. `--beatnet`/`--qnet` olmadan çıktı Adım 0 ile birebir.

### Testler
- Önbellek temsili + perde-onset kafası = `predict_heads` (en çok 2·10⁻⁴ fark, float16) ✓; tempo artırması s = 1'de kesitle
  birebir, s = 2'de etiket kareleri beklenen ✓; DBN sentetik testleri ✓; bayraksız `eval_rhythm --limit 3` birebir ✓.

### Deneme sonuçları (geçici checkpoint'ler; gaps_test yalnız rapor)
| deneme | gaps_val vuruş F / ölçü başı F (DBN) | gaps_test vuruş F / CMLt / AMLt / ölçü başı F | uçtan uca IOI (1q) val / **test** |
|---|---|---|---|
| Adım 0 (cqt + plp, kural ölçü başı) | 0.478 / 0.21 | 0.444 / 0.072 / 0.257 / 0.190 | 0.566 / **0.617** |
| BeatNet 10 ep (DBN trans 30, obs 32) | 0.549 / 0.410 | 0.443 / 0.096 / 0.310 / 0.418 | 0.670 / **0.633** |
| BeatNet 30 ep + tempo artırması (trans 10, obs 32) | 0.596 / 0.517 | 0.501 / 0.187 / 0.375 / 0.444 | 0.620 / **0.572** |
- 60 ep (artırmasız): gaps_val ölçütleri ~10. epoch'tan sonra düşüyor (ep 40: 0.470 / 0.412, kayıp 0.17) → ezber. 10. epoch'ta
  eğitim ≈ doğrulama (gaps_fit 0.570 / gaps_val 0.547) → bilgi / genelleme sınırı, kapasite değil.
- GuitarSet solo'da ağ vuruşu neredeyse görmüyor (eğitim kayıtlarında bile tepe seçimi F 0.29): doğaçlama solo sesinde (eşlik
  kaydı yok) vuruş ipucu zayıf. Ana hedef GAPS.
- **Vuruş F'si uçtan uca IOI'ye taşınmıyor:** artırmalı model tüm vuruş ölçütlerinde daha iyi ama IOI'de daha kötü. IOI kayıt
  başına TEK ölçek seçiyor; izleyici parçanın bir bölümünde metrik düzey değiştirirse (ör. iki kat tempo) o bölümün nota
  değerleri tümüyle yanlış sayılıyor. → Seçim ölçütü uçtan uca IOI (gaps_val) olmalı.
- **Hedeflerin çok altında:** vuruş F ≥ 0.75 (en iyi 0.50), uçtan uca ≥ 0.833 (en iyi 0.633). İfadeli klasik müzikte vuruş
  takibi literatürde de zor (ör. madmom ifadeli piyano icrasında F ~0.5–0.65); eşikler ölçümden önce konuldu.

### Yarı otomatik mod (kullanıcı kararı, 11 Ekim): kullanıcı BPM + ölçü verir
Ürün akışı: kullanıcı yaklaşık BPM'i (vuruş birimine göre) ve ölçüyü (ör. 3/4) girer → DBN yalnız o ölçüde ve BPM ± aralıkta
çözer; içerideki rubato'yu sesten (BeatNet) izler. Ölçü gerekli: BPM tek başına metrik düzey açısından belirsiz (2/2'de
çeyrek mi ikilik mi) ve ölçü başları ölçü olmadan kurulamaz.
- Ön ölçüm (rubato): yerel tempo / parça medyanı p5–p95 = 0.63–1.38; vuruş aralıklarının ±%20 içinde %72–78'i, ±%50 içinde
  %92'si → aralık sabitlenmez, gaps_val'de seçilir.
- Değerlendirme (`eval_rhythm --beatnet ... --user-tempo`, B3 bölümü): kullanıcı girdisi taklidi = gerçek medyan tempo ×
  (1 ± %10, işaret kayıt sırasıyla) + gerçek ölçü; ayrıca %0 hata satırı. Aralık (±%20/35/50/75) × tempo geçiş cezası
  (3/10/30) **gaps_val uçtan uca IOI** ile seçilir (vuruş F'si IOI'ye taşınmadığı için doğrudan hedef). D'de "yari otomatik"
  satırları.

Denemeler (aynı iki geçici BeatNet; seçim gaps_val; gaps_test yalnız rapor; BPM hatası %10):
| | seçilen aralık / trans | gaps_val IOI (1q) | gaps_test vuruş F / CMLt / ölçü başı F | gaps_test uçtan uca IOI (1q) | r = 1 |
|---|---|---|---|---|---|
| tam otomatik (en iyi deneme) | — | 0.670 | 0.443–0.501 / 0.10–0.19 / 0.42–0.44 | 0.572–0.633 | %13–43 |
| yarı otomatik, BeatNet 10 ep | ±%35 / 30 | 0.738 | 0.574 / 0.541 / 0.498 | 0.654 | %87 |
| **yarı otomatik, BeatNet 30 ep + tempo artırması** | ±%35 / 10 | 0.731 | **0.631 / 0.581 / 0.582** | **0.695** | %96 |
BPM hatası %0 ile %10 arasında fark yok (gaps_test F 0.625 / 0.631) → kaba tahmin yeterli. gaps_val IOI'de iki ağ berabere
(±0.01); vuruş / ölçü başı F'de artırmalı ağ önde (0.690 / 0.558 vs 0.658 / 0.517) → **varsayılan: `train_beat`
`--stretch 0.25`, 30 epoch.** GuitarSet val (solo) zayıf kalıyor (F 0.39–0.46).

**Değerlendirme:** yarı otomatik mod metrik düzeyi çözüyor (r = 1 %96) ve uçtan ucu belirgin artırıyor (0.617 → 0.695),
ama eşiklerin altında: vuruş F 0.63 < 0.75, ölçü başı 0.58 < 0.60 (sınırda), uçtan uca 0.695 < 0.833. Kalan kayıp yerel
vuruş konumu hataları (aktivasyon kalitesi): gerçek vuruşla 0.927.

### Çalıştırma (1b)
```bash
python -m scripts.data.build_beat_feats        # yapıldı (808 MB); yeni ses modelinde yeniden
python -m scripts.train.train_beat             # ~25 dk -> checkpoints/beat_net.pt (DBN ayarları içinde)
python -m scripts.eval.eval_rhythm --qnet rhythm_q.pt --beatnet beat_net.pt --user-tempo    # ~12 dk
```

### Karar kuralı (1b, yarı otomatik — gaps_test, BPM hatası %10)
- Kabul: vuruş F ≥ 0.75, ölçü başı F ≥ 0.60, uçtan uca IOI (1q) ≥ 0.833. Denemelere göre beklenen ~0.63 / 0.58 / 0.70 →
  büyük olasılıkla sağlanmaz; o durumda yön kullanıcı kararı (ör. vuruşa dokunarak girdi = "tap" modu, ya da yarı otomatik
  + kullanıcı düzeltmesiyle Adım 2'ye geçiş).

### Sonuç — Adım 1b (11 Ekim, kullanıcı çalıştırması; `beat_net.pt`, DBN trans 10 / obs 32) — denemeyle birebir
| gaps_test | vuruş F | CMLt | ölçü başı F | uçtan uca IOI (1q) | r = 1 |
|---|---|---|---|---|---|
| tam otomatik (BeatNet + DBN) | 0.501 | 0.187 | 0.444 | 0.572 | %43 |
| **yarı otomatik** (BPM %10 hatalı + ölçü; ±%35, trans 10) | **0.631** | **0.581** | **0.582** | **0.695** | %96 |
| eşik | 0.75 | | 0.60 | 0.833 | |
gaps_val yarı otomatik: F 0.690, ölçü başı 0.558, IOI 0.731. **Eşikler sağlanmadı.**
**Kullanıcı kararı (11 Ekim):** ritim bu hâliyle (yarı otomatik) kullanılır → Adım 2'ye (ölçülü TAB çıktısı) geçilir; "tap"
modu (kullanıcı vuruşa dokunur) ileride, Adım 2 yetmezse iyileştirme yedeği. Adım 2: [README16](README16.md).

## Durum
- [x] Adım 0a — ritim etiketleri (`build_rhythm`): GAPS 166 kayıt, ölçü başı syncpoint uyumu 23 ms; GuitarSet 360
- [x] Adım 0b — eğitimsiz taban (vuruş dp/plp, ölçü başı kuralı, kuantalayıcı) + `eval_rhythm` + testler
- [x] Adım 0 — ölçüm: tavan 0.821, vuruş F 0.444, uçtan uca IOI 0.588 → eğitimsiz yol yetmiyor (kullanıcı çalıştırması birebir)
- [x] Adım 1q — bağlamlı kuantalayıcı: **KABUL** (gaps_test oracle 0.933, uçtan uca 0.617)
- [x] Adım 1b — tam otomatik denemeler: vuruş F 0.50, ölçü başı 0.44, uçtan uca 0.57–0.63 (hedef 0.833)
- [x] Adım 1b yarı otomatik (kullanıcı BPM + ölçü; kullanıcı kararı) kodlandı; deneme: F 0.63, ölçü başı 0.58, uçtan uca 0.695
— kullanıcı çalıştırması birebir (F 0.631, ölçü başı 0.582, IOI 0.695); eşik yok → kullanıcı kararı: bu hâliyle Adım 2'ye
- [x] Adım 2 — ölçülü TAB çıktısı (MusicXML) → [README16](README16.md): **KAPANDI** (ölçü içi konum 0.611)
- **Katman 5 çekirdeği tamamlandı (11 Ekim):** ritim (1q + yarı otomatik 1b) + MusicXML çıktısı. Açık: AlphaTab web
  görüntüleyici (render; teknik sembolleri Katman 4'e bağlı)
- [ ] (yedek) tap modu — kullanıcı vuruşa dokunur; Adım 2 yetmezse
