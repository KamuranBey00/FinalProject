# Katman 5 Adım 2 — Ölçülü TAB çıktısı (MusicXML)

> Durum (11 Ekim 2026): **KAPANDI — yazıcı birebir, uçtan uca dosyalar %100 geçerli; ölçü içi konum gaps_test 0.611 (tek imli).**
> Önceki adım: [README15](README15.md) (ritim: 1q kuantalayıcı kabul, 1b yarı otomatik vuruş takibi kullanıcı kararıyla
> mevcut hâliyle kullanılıyor; "tap" modu ileride iyileştirme yedeği). Ses modeli, kuantalayıcı ve vuruş ağı değişmez.

## Neden
Sunumun hedef zinciri: Audio → Pitch/Onset → Polyphony → String/Fret → **Rhythm → TAB** → Fingering → Visualization.
Şimdiye kadar çıktı saniye cinsinden notalar + ASCII TAB'dı. Okunabilir bir partisyon için ölçü çizgileri, nota değerleri,
ölçü imi ve tempo gerekir; Katman 5'in ritim adımları (1q, 1b) bunları üretiyor, bu adım onları standart bir dosyaya döker.
MusicXML seçildi: MuseScore, Guitar Pro, Sibelius ve AlphaTab (web görselleştirme, Katman 5 "render" hedefi) açar; GAPS
partisyonları da MusicXML olduğu için yazıcı, projedeki GAPS okuyucusuyla birebir doğrulanabilir.

## Ne yapıldı
- **`gtab/render/score.py`** — ölçü modeli. 1q konumları (1/24 vuruş ızgarası) + bir ölçü başının vuruş indeksi + ölçü
  imi → ölçüler. Tek ritim sesi (gitar TAB'larında yaygın): aynı konumdaki notalar akor, süresi bir sonraki başlangıca kadar,
  son akor ölçü sonuna kadar; ölçü çizgisini aşan süre bağla bölünür; her parça yazılabilir değerlere ayrılır (tam … 128'lik,
  1–2 nokta, üçleme; açgözlü). İlk notadan önceki boşluk sus. Ölçü imi → (ölçüdeki vuruş, vuruş uzunluğu) README15'teki
  kuralla (`beat_ticks_of`: x/4 çeyrek, x/2 ikilik, x/8 + pay 3'ün katı noktalı çeyrek).
- **`gtab/render/musicxml.py`** — score-partwise MusicXML 4.0, GAPS düzeni: P1 nota portresi (8vb sol anahtarı, çalan perde,
  sol el parmağı `<fingering>`; metronom), P2 TAB portresi (6 çizgi, `<staff-tuning>`, `<string>` 1 = ince mi, `<fret>`).
  divisions = vuruş başına 24 → süreler doğrudan tick. Akort parametre (scordatura).
- **`gtab/pipeline.py`** — uçtan uca: CQT → deep2 notaları (GAPS çözümleme seçimi) → tel/perde (**ses** modu: deep2 tel
  olasılıkları + öğrenilen geçiş modeli; **edisyon** modu: EditionNet konum + parmak; karıştırılmaz) → BeatNet + DBN
  (kullanıcı BPM ± %35, verilen ölçü; README15 B3 seçimi) → 1q konumları → ölçüler.
- **`scripts/app/transcribe.py`** — komut satırı: `ses dosyası | .npz` + `--bpm` + `--meter` (+ `--mode`) → `.musicxml`.
  (Yeni `scripts/app/` klasörü: ürün komutları; data / train / eval dışında.)
- **`scripts/eval/eval_score.py`** — ölçüm: A) gidiş-dönüş, B) uçtan uca (aşağıda).

## Ölçüm tanımları (`eval_score`)
- **A) Gidiş-dönüş (yazıcı doğruluğu):** tek ölçü imli, anakruzsuz GAPS partisyonlarının TAB notaları (konum = partisyon) →
  `build_measures` → `write_musicxml` (partisyonun akortuyla) → `gtab.data.gaps_score.score_notes` → (ölçü, tick, perde,
  tel, perde no, parmak) birebir oranı + ölçü doluluğu (her ölçünün süre toplamı = ölçü imi).
- **B) Uçtan uca (GAPS, yarı otomatik, ses modu):** kullanıcı girdisi taklidi eval_rhythm B3 ile aynı (gerçek medyan tempo
  × (1 ± %10), işaret kayıt sırasıyla) + partisyonun **en sık** ölçü imi (kullanıcı ana ölçüyü verir). deep2 notaları gerçek
  notalara eşlenir (aynı perde, ± 50 ms).
  - **Ölçü içi konum doğruluğu** (yeni, ana ölçüt): notanın ölçü başından uzaklığı (vuruş) partisyondakiyle aynı mı —
    ölçü başı fazı + metrik düzey + kuantalama birlikte. Tümü ve **yalnız tek ölçü imli** parçalar (ölçü imi değişen
    parçaları tek kullanıcı imi temsil edemez).
  - IOI doğruluğu (eval_rhythm D "yarı otomatik + deep2 1q" ile aynı tanım; tutarlılık kontrolü), ölçü başı F.
  - Geri okuma: okuyucunun bulduğu nota / konumlu nota; ölçü uzunlukları tek tip mi.

## Testler
- **Gidiş-dönüş (tam, 11 Ekim):** 202 partisyon, 159 796 nota → **birebir 1.0000**, ölçü doluluğu bozuk **0**; 1/24 vuruş
  ızgarası dışındaki 903 nota (%0.6) yazılmaz (1q da üretemez). İlk denemede %98.3: kalan farkın tamamı scordatura
  partisyonlardı (ör. 5. tel G) → yazıcıya partisyonun akortu verilince %100 (yazıcı hatası değil).
- **Ölçü içi konum ölçütü (oracle):** gerçek konum + gerçek ölçü başlarıyla tek ölçü imli gaps_val parçalarında 1.000 (5/5);
  ölçü imi değişen parçalarda 0.05–0.85 → bu yüzden "tek imli" sütunu ayrı.
- **Komut satırı duman testi:** 027_Zpswc (gaps_test, 4/4, 86 BPM), ses ve edisyon modu: 1613 nota, 131 ölçü, kayıt başına
  5–12 sn; çıktı GAPS okuyucusuyla geri okunuyor (tüm notalar, tek tip ölçü, edisyonda 1613 parmak).
- `eval_score --limit 2` çalışıyor.

## Çalıştırma
```bash
python -m scripts.eval.eval_score                                   # ~10 dk (A + B: gaps_val 15, gaps_test 23)
python -m scripts.app.transcribe kayit.wav --bpm 80 --meter 3/4                 # ses modu
python -m scripts.app.transcribe kayit.wav --bpm 66 --meter 6/8 --mode edisyon  # edisyon önerisi + parmak
```
Çıktıyı MuseScore (ücretsiz) ile açıp görsel kontrol: nota + TAB portresi, ölçüler, bağlar, üçlemeler.

## Karar kuralı
- **Yazıcı:** gidiş-dönüş ≥ 0.999 ve ölçü doluluğu %100 → ✓ (ölçüldü).
- **Uçtan uca dosya geçerliliği:** geri okuma %100 ve tek tip ölçü %100 (gaps_val + gaps_test).
- **Tutarlılık:** B'deki IOI, eval_rhythm D "yarı otomatik + deep2 (1q)" ile ± 0.02 (gaps_test 0.695; ölçü imi kaynağı farklı
  olduğundan küçük fark olabilir).
- **Ölçü içi konum (tek imli)** raporlanır = ritim kalitesinin kullanıcıya görünen ölçüsü; eşik yok (vuruş takibi 1b'de
  kullanıcı kararıyla mevcut hâliyle) → tap modu / ileride vuruş iyileştirmesinin kabul ölçütü bu sayı olur.
- Sağlanırsa → Adım 2 kapanır; sıradaki (kullanıcı kararı): görselleştirme (AlphaTab web sayfası, sunum adım 7), tap modu
  ya da Katman 4 (teknikler).

## Bilinen sınırlar (bilerek kapsam dışı)
- Tek ritim sesi: klasik partisyondaki ayrı sesler (bas ↓ / melodi ↑ sap yönü, farklı süreler) yazılmaz; süre = sonraki
  başlangıca kadar (TAB'larda yaygın gösterim). Ses ayrımı ayrı bir problem.
- Üçlemelerde köşeli parantez (`<tuplet>`) yazılmıyor, yalnız `<time-modification>`; garip süreler açgözlü ayrılıyor
  (ör. 20 tick = noktalı onaltılık + 32'lik üçleme) — doğru ama okunabilirliği iyileştirilebilir.
- Tek ölçü imi ve standart akort (modeller standart akortla eğitildi; yazıcı akort alabiliyor). Anakruz ölçüsü susla dolar.

## Sonuçlar (11 Ekim, kullanıcı çalıştırması `eval_score`)
A) Gidiş-dönüş: 202 partisyon, 159 796 nota → **birebir 1.0000**, ölçü doluluğu bozuk **0 / 14 398**.

B) Uçtan uca (yarı otomatik, ses modu):
| | kayıt | ölçü içi konum (tümü) | **tek imli** | n | IOI (1q) | ölçü başı F | geri okuma | ölçü tek tip |
|---|---|---|---|---|---|---|---|---|
| gaps_val | 15 | 0.570 | 0.560 | 5 998 | 0.731 | 0.558 | 1.000 | 1.000 |
| **gaps_test** | 23 | 0.592 | **0.611** | 8 371 | **0.695** | 0.582 | 1.000 | 1.000 |
- Dosya geçerliliği %100 ✓; IOI ve ölçü başı F eval_rhythm B3 / D ile **birebir** (0.731 / 0.695; 0.558 / 0.582) ✓.
- Bulunan notaların ~%60'ı ölçüde doğru yerde. Nota değerleri (IOI 0.695) bundan yüksek → hatanın bir kısmı ölçü başı fazı
  (ölçü çizgisi kayması; ölçü başı F 0.58): nota değerleri doğru olsa da ölçü çizgileri yanlış yerde kalıyor.
- **Karar: Adım 2 KAPANDI** (yazıcı + dosya geçerliliği + tutarlılık ölçütleri sağlandı). Ölçü içi konum (tek imli) **0.611**
  = ritim iyileştirmelerinin (tap modu, ölçü başı) yeni taban ölçütü.

## Durum
- [x] Ölçü modeli + MusicXML yazıcı + gidiş-dönüş testi (birebir 1.0000, doluluk %100)
- [x] Uçtan uca hattı (`gtab/pipeline.py`) + komut satırı (`scripts/app/transcribe.py`) + duman testi
- [x] `eval_score`: gidiş-dönüş 1.0000, uçtan uca geçerlilik %100, ölçü içi konum 0.611 → **KAPANDI**
- [ ] Görselleştirme: MuseScore'da TAB görünümü (kullanıcı) + ürün için AlphaTab web görüntüleyici ← **SIRADA (kullanıcı kararı)**
