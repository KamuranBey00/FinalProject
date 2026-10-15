# Katman 5 Adım 3 — Kullanıcı girdisiyle ritim düzeltme: ölçü çizgisi kaydırma + tap modu

> Durum (13 Ekim 2026): **kodlandı + ölçüldü (simülasyon). Tap modu 1b'nin kabul eşiklerini gerçekçi dokunuş gürültüsüyle
> geçiyor (gaps_test vuruş F 0.976, ölçü başı F 0.889, uçtan uca IOI 0.877, ölçü içi konum 0.864). Gerçek insan dokunuşuyla
> doğrulama bekleniyor.** Önceki: [README16](README16.md) (ölçülü TAB, ölçü içi konum 0.611), [README15](README15.md) (ritim).
> Ses modeli, 1q, BeatNet değişmez.

## Neden
Adım 2'de yarı otomatik ritimle (kullanıcı BPM + ölçü) notaların ~%61'i ölçüde doğru yerdeydi; nota değerleri (IOI) 0.695.
Kullanıcı kararı (13 Ekim): önce ölçü çizgisi kaymasına ucuz bir yol, hemen ardından README15'te yedekte tutulan tap modu.

## 1) Ölçü çizgisi kaydırma (`--shift k`) — önce ölçüldü
`eval_score` B'ye "faz düzeltmeli" sütunu: kullanıcı ölçü çizgilerini parça boyunca k vuruş kaydırsa, en iyi k ile (oracle =
düğmenin tavanı) ölçü içi konum (tek imli parçalar):

| | yarı otomatik | en iyi kaydırma ile | yanlış fazlı kayıt |
|---|---|---|---|
| gaps_val | 0.560 | 0.576 | 1 / 9 |
| gaps_test | 0.611 | 0.616 | 3 / 17 |
**Sonuç: kaydırma neredeyse hiç kazandırmıyor (+0.005).** Hatalar parça boyunca sabit bir kayma değil; izleyici parça
ortasında vuruş kaçırınca / eklediğinde faz yerel kayıyor. `--shift` ucuz seçenek olarak eklendi (`transcribe(..., shift=k)`,
`scripts/app/transcribe.py --shift k`), asıl çözüm doğru vuruşlar → tap modu.

## 2) Tap modu
Kullanıcı kaydı dinlerken **ilk ölçü başından başlayarak her vuruşa dokunur** (ürün arayüzü: bir tuş / ekrana dokunma).
`gtab/decoding/taps.py`:
1. **Gecikme:** dokunuşlar δ ∈ [−150, +150] ms kaydırılır; δ = BeatNet vuruş aktivasyonunun dokunuşlardaki toplamını en
   büyüten (insan dokunuşu sabit gecikme taşır).
2. **Fazla dokunuş:** yerel vuruş aralığının yarısından yakın olan atılır.
3. **Kaçan dokunuş:** yerel aralığın 1.5 katından uzun boşluğa yalnız **ara noktada aktivasyon ≥ 0.05** ise vuruş eklenir.
   Körlemesine doldurma, kusursuz dokunuşta bile fermatalarda 5190 vuruşa 23 vuruş ekliyordu (vuruş aralığı yerel medyanın
   3.9 katına çıkabiliyor); tutulan akorda yeni nota yok → aktivasyon yok → eklenmez.
4. **Ölçü fazı Viterbi'si (`meter_phase`):** durum = vuruşun ölçüdeki yeri; geçiş +1 (normal) | +2 (kaçan vuruş → araya
   eklenir) | +0 (fazla vuruş → silinir), ε = 0.003; gözlem = vuruştaki ölçü başı payı p_db / (p_beat + p_db); ilk dokunuş
   ölçü başı. Böylece tek bir sayım hatası sonraki tüm ölçüleri kaydırmaz, vuruş listesi kendini düzeltir.
- Hat ikiye ayrıldı (`gtab/pipeline.py`): `analyze` (ses: notalar, tel/perde, BeatNet aktivasyonu; bir kez) + `rhythm`
  (vuruş kaynağı: `taps` ya da `bpm`; 1q; ölçüler). `transcribe` = ikisi; yarı otomatik çıktısı birebir aynı (regresyon ✓).
- Komut satırı: `--taps dosya` (saniyeler; satır / boşluk / virgül), `--bpm` gerekmez (metronom işareti dokunuşlardan).

### Ölçüm (`eval_score` C, simülasyon)
Kullanıcı taklidi: gerçek vuruşlar (ilk ölçü başından) + sabit gecikme + Gauss sapma, %5 kaçan (ilk dokunuş hep var),
%2 fazla (rastgele yerde). Koşullar: ideal (0 ms), **iyi (σ 20 ms, +50 ms)**, kaba (σ 40 ms, +80 ms). Ayar seçimi gaps_val
"iyi" + "kaba" (ölçü içi konum):

| gaps_val, "iyi" | vuruş F | konum | tek imli | IOI | ölçü başı F |
|---|---|---|---|---|---|
| ölçü başı = her M dokunuş (sayım) | 0.981 | 0.340 | 0.213 | 0.831 | 0.352 |
| + ölçü fazı Viterbi'si (ε 0.01, eşik 0.2) | 0.974 | 0.716 | 0.764 | 0.819 | 0.775 |
| aktivasyon tepesine yapıştırma (± 70 ms) | 0.892 | 0.699 | 0.736 | 0.818 | 0.719 |
| **seçilen: eşik 0.05, ε 0.003** | 0.976 | **0.753** | **0.826** | 0.829 | **0.817** |
Yapıştırma her koşulda zararlı → koddan çıkarıldı. (Bu tablo önbellekli hızlı taramadan; aşağıdaki tam ölçümle ±0.01.)

### Sonuçlar (13 Ekim, `eval_score` tam çalıştırma)
| | koşul | vuruş F | ölçü içi konum | tek imli | IOI (1q) | ölçü başı F |
|---|---|---|---|---|---|---|
| gaps_val | yarı otomatik (BPM %10 + ölçü) | 0.690 | 0.570 | 0.560 | 0.731 | 0.558 |
| gaps_val | tap ideal | 0.991 | 0.800 | 0.873 | 0.883 | 0.846 |
| gaps_val | **tap iyi** | 0.972 | 0.740 | 0.813 | 0.826 | 0.796 |
| gaps_val | tap kaba | 0.885 | 0.694 | 0.739 | 0.735 | 0.720 |
| gaps_test | yarı otomatik | 0.631 | 0.592 | 0.611 | 0.695 | 0.582 |
| gaps_test | tap ideal | 0.996 | 0.921 | 0.922 | 0.921 | 0.950 |
| gaps_test | **tap iyi** | **0.976** | **0.864** | **0.860** | **0.877** | **0.889** |
| gaps_test | tap kaba | 0.875 | 0.806 | 0.815 | 0.823 | 0.785 |
- 1b'nin kabul eşikleri (README15: vuruş F ≥ 0.75, ölçü başı F ≥ 0.60, uçtan uca IOI ≥ 0.833) **tap "iyi" ile sağlanıyor**;
  "kaba"da IOI 0.823 (sınırda). Ölçü içi konum 0.611 → **0.864**.
- Tavan: kusursuz dokunuşla IOI 0.921 ≈ gerçek vuruş + deep2 (0.927) → kalan kayıp dokunuş gürültüsünden.
- gaps_val ölçü içi konum testten düşük (ölçü imi değişen / anakruzlu parçalar; tek imli sütun daha yakın).

## Testler
- `analyze` + `rhythm` ayrımı: `eval_score --limit 3` B çıktısı önceki çalıştırmayla birebir ✓; tam B satırları Adım 2 ile birebir ✓.
- Komut satırı: 027_Zpswc, "iyi" taklit dokunuş dosyası (482 dokunuş) → 499 vuruş (gerçek 503), 126 ölçü, metronom 87;
  `--shift 1` ve girdisiz çağrı (hata mesajı) ✓.

## Çalıştırma
```bash
python -m scripts.eval.eval_score                                         # ~8 dk: A, B (+ faz sütunu), C (tap)
python -m scripts.app.transcribe kayit.wav --taps dokunuslar.txt --meter 3/4   # tap modu
python -m scripts.app.transcribe kayit.wav --bpm 80 --meter 3/4 --shift 1      # yarı otomatik + ölçü kaydırma
```

## Karar kuralı
- Tap modu (simülasyon, gaps_test, "iyi"): vuruş F ≥ 0.75, ölçü başı F ≥ 0.60, uçtan uca IOI ≥ 0.833 → ✓ **sağlandı**.
- Kalan koşul: **gerçek insan dokunuşuyla doğrulama** (simülasyonun gürültü modeli varsayım). Öneri: birkaç GAPS test kaydını
  dinleyip dokunuşları kaydetmek (basit bir dokunuş kayıt betiği / arayüz) → aynı ölçüm.
- Ürün önerisi: tap modu varsayılan (doğruluk), yarı otomatik (BPM + ölçü) hızlı seçenek.

## 3) Gerçek dokunuşla doğrulama (araçlar hazır, 13 Ekim)
- **`scripts/app/tap_recorder.html`** — tarayıcıda açılan dokunuş kaydedici: ses dosyası seç, çal, her vuruşta BOŞLUK (ya da
  kutuya dokun); son dokunuşu sil / sıfırla; "İndir" → `<kayıt>.txt` (saniyeler, `audio.currentTime`).
- **`scripts/eval/eval_taps.py --dir <klasör>`** — her `<id>.txt` için yalnız dokunulan aralıkta: gerçek dokunuş |
  aynı aralıkta yarı otomatik | aynı aralıkta simüle "iyi"; vuruş F, ölçü içi konum, IOI, ölçü başı F, tahmini gecikme,
  dokunuş / gerçek vuruş sayısı.
- **Gecikme araması uyarlamalı:** max_lag = min(300 ms, 0.4 × medyan dokunuş aralığı) (önce sabit 150 ms). Duman testinde
  200 ms gecikmeli (Bluetooth benzeri) taklit dokunuşla vuruş F 0.706 → 0.976; gaps_val seçim sonuçları birebir aynı
  ("iyi" 0.753, "kaba" 0.692). Gecikme tahmini doğru (200 → 209 ms, 60 → 70 ms).
- Önerilen test kayıtları (gaps_test; tek ölçü imli, çeyrek vuruş, orta tempo, kısa; ses `data/raw/gaps_hf/audio/`):
  `358_441wc` Quiet Waters 3/4 ~83 BPM (2 dk), `031_vpswc` Gebeth 4/4 ~79 (3 dk), `303_5N1wc` Córdoba 3/4 ~89 (2.4 dk).
- Karar kuralı (gerçek dokunuş, kayıt ortalaması): vuruş F ≥ 0.75, ölçü başı F ≥ 0.60, IOI ≥ 0.833 ve simüle "iyi"den
  en çok 0.05 düşük → simülasyon geçerli, tap modu kabul.

**İlk gerçek deneme (13 Ekim, 358_441wc):** vuruş F 0.213, ölçü içi konum 0.076, IOI 0.489 (yarı otomatik 0.720 / 0.838 /
0.833). Teşhis (dokunuş dosyası vs gerçek vuruşlar): (1) ilk dokunuş müzikten önce (1.05 sn; ilk vuruş 2.03 sn) → sistem onu
ölçü başı sayıyor; (2) dokunulan nabız parçanın vuruşu değil: dokunuş aralığı ~1.26 sn (~48 BPM), vuruş 0.73 sn (~83 BPM;
oran 1.73, hiçbir metrik düzey değil; ilk 20 sn aralıklar 0.85–2.8 sn) → 132 vuruşa 73 dokunuş, ±70 ms içinde %29.
Kullanıcı kendi sabit nabzını tuttu; araç "her vuruş = ölçüdeki her sayım" beklentisini yeterince anlatmıyordu.
Düzeltmeler: kaydediciye ölçü sayacı ("1 / 3, 2 / 3 …") + açık talimat; hatta ilk notadan 150 ms önceki dokunuşlar atılır
(sayım / deneme vuruşu). Simüle dokunuş sonuçları değişmedi. Tekrar deneme bekleniyor.

```bash
# tarayıcıda scripts/app/tap_recorder.html aç -> dokun -> indirilen <id>.txt dosyalarını data/taps/ altına koy
python -m scripts.eval.eval_taps --dir data/taps
```

**İkinci gerçek deneme (13 Ekim, 4 kayıt: 031, 201, 303, 358; sayaçlı kaydedici; kullanıcı müzisyen değil):**
| ortalama | vuruş F | ölçü içi konum | IOI | ölçü başı F |
|---|---|---|---|---|
| tap modu (gerçek dokunuş) | 0.196 | 0.122 | 0.488 | 0.078 |
| **dokunuş temposu → yarı otomatik** (BPM = 60 / medyan dokunuş aralığı) | 0.517 | 0.381 | **0.702** | 0.418 |
| yarı otomatik, BPM bilinen (gerçek × 1.1) | 0.539 | 0.591 | 0.741 | 0.517 |
| simüle "iyi" dokunuş | 0.963 | 0.865 | 0.857 | 0.899 |
Teşhis (dokunuş vs gerçek vuruş): dokunuş sayısı artık vuruş sayısına yakın (93/78, 64/79, 85/78, 99/97) ama dokunuşlar vuruşa
kilitli değil: en iyi sabit gecikmeyle bile ±70 ms içinde %23–28 (simülasyon "iyi": ~%100), vuruş içindeki konum neredeyse
düzgün dağılmış (%7–%91), notalara da kilitli değil (%23–46); dokunuş / vuruş aralığı oranı 0.80–1.33. Yani müzisyen olmayan
dokunuş yalnız **kaba tempo** taşıyor, vuruş konumu taşımıyor; simülasyonun gürültü modeli (± 20–40 ms, sabit gecikme) bu
kullanıcı için geçerli değil.

**Karar (13 Ekim):** tap modu ("her vuruşa dokun") müzisyen düzeyi dokunuş ister → **ileri seçenek** olarak kalır (yalnız
simülasyonla doğrulandı), varsayılan değil; daha fazla yatırım yok. Ürün için müzisyen olmayan kullanıcıya **"tap tempo"**:
birkaç dokunuştan BPM → yarı otomatik mod (IOI 0.702 ≈ BPM bilinen 0.741). Varsayılan akış: ölçü imi + (BPM ya da tap tempo).
Bu karar için 4 kayıt yeterli; ek dokunuş dosyası gerekmiyor.

## Durum
- [x] Ölçü kaydırma: ölçüldü (+0.005, ucuz seçenek olarak `--shift`)
- [x] Tap modu: `gtab/decoding/taps.py`, `pipeline.analyze/rhythm`, `--taps`, `eval_score` C → eşikler simülasyonda sağlandı
- [x] Gerçek dokunuş araçları: `tap_recorder.html`, `eval_taps`, uyarlamalı gecikme araması
- [x] Gerçek dokunuşla doğrulama: müzisyen olmayan kullanıcıda tap modu başarısız (IOI 0.488), tap tempo → yarı otomatik 0.702
  → tap modu ileri seçenek, ürün varsayılanı ölçü + BPM / tap tempo. **Katman 5 Adım 3 KAPANDI.**
