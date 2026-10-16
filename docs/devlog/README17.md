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

---

## Ek A — Armağan dalı birleşimi (`origin/Armagan-katman-4`, 15–16 Ekim)

Armağan'ın dalı 3.13'ten (`72295e6`) ayrılmıştı: Katman 4 Faz 0–2A (ölçüm protokolü, teknik etiket envanteri, sürekli F0
altyapısı), parmaklandırma P1–P3 (inceleme, test altyapısı, çalınabilirlik metrikleri) ve bir bakım turu. Önce bizim Adım 3
işimiz yeni dala pushlandı (`katman-5.1-tap-modu` @ `c16dbc6`), sonra yalnız değişen / eklenen dosyalar alındı. Armağan'ın
devlogları bizim numaralarla çakıştığı için alınmadı; özü bu bölümde.

### Eski hatalar ve çözümleri (Armağan)
| Sorun (önce) | Çözüm (şimdi) |
|---|---|
| Ürün tel seçimi test edilmemiş ayarla: `w_transition` 1.0 (tab F1 0.751); bütün ölçümler 0 ile (0.777) | ASCII aracı varsayılanı 0; **`gtab/pipeline.py` ses modu da 0** (birleşimde düzeltildi) |
| Tekrar işaretleri sesteki ölçü sayısını tutturmayan 75 GAPS kaydı atılıyordu (icracı yazmayan tekrar çalmış) | `gaps_score`: ölçü sırası çalınan içerikten (Dice benzerliği + Viterbi); `align(recover=False)` eskisiyle birebir; `build_gaps_tab --recover-repeats` → `data/cache/gaps_tab_rec` (165 → **236** kayıt, eski etiketlerle %99.9 uyum) |
| Eğitim, `--ckpt` unutulunca kullanılan modelin (`edition_seq.pt` dahil) üstüne yazıyordu | `utils.guard_new_checkpoint`; `train_onset` / `train_edition_seq` `--overwrite` olmadan durur; `--seed` |
| ASCII TAB: `712` yapışık perdeler, aynı kutuya düşen nota siliniyordu (995 notadan 959) | kutu = en uzun perde + 1, çakışan nota sonraki kutuya; 995/995 |
| Ana tab F1 ölçümü 15 bayrakla | `eval_hmm --load-cal checkpoints/tabcrnn_deep2.cal.json` |
| `test_edition` mir_eval yoksa çöküyordu | `HAVE_MIR_EVAL` kapısı |
| "±0.01 tohum gürültüsü" hiç ölçülmemişti | ölçüldü: tab F1 0.009, A teli 0.021 |
| Çalınabilirlik hiç ölçülmüyordu (sunum slayt 12) | `gtab/decoding/playability.py` + `eval_playability` (zamanda örtüşen çiftlerde parmak çakışması, açıklık, ters sıra, barre, pozisyon kayması; model / naif / editör) |
| Gerçek tel etiketi yok (hex) | `gtab/data/hex.py`, `build_hex` (GuitarSet hex → tel başına spektrum; `train_onset --hex-weight`): tel kararını değiştirmedi |

Yeni test / koruma altyapısı: `tests/` (**204 test**, `python -m unittest discover -s tests -t .`), `check_protocol` (veri
envanteri + split kapıları, 17 kontrol), `check_inference_regression` (deep2 çıkarım parmak izi, `docs/baselines/`).
Katman 4 hazırlığı (eğitim yok): `gtab/evaluation/{protocol,technique_eval,selection,baselines,fingerprint}.py`,
`gtab/data/contour.py` + `build_contour` (F0 konturu), `inventory_technique`, `probe_gaps_f0`. Faz 1 bulgusu: **GuitarSet ve
GAPS'te sol el tekniği etiketi yok** → kapsam A: vibrato + hammer-on + pull-off (kuraldan sözde etiket), B: slide ek etiket
ister, C: bend ve armonik kapsam dışı. P3 (çalınabilirlik, gaps_val kesişim 2 963 akor): kesin parmak çakışması model %8.5,
naif %13.4, editör %0; ≥ 4 yarım ses açıklık model %3.6, naif %27.4.

Aday parmak modeli `checkpoints/edition_seq_rec.pt` (kurtarılmış 182 kayıtla): gaps_val tel / parmak 0.784 / 0.700 (ürün
`edition_seq.pt` 0.766 / 0.675). **Ürüne alınacak (kullanıcı kararı), ama hemen değil**; önce ikinci tohum + çalınabilirlik.

### Birleşim (bizim tarafımız)
- İki dalda da değişen 3 dosya üç yönlü birleşti: `eval_hmm`, `train_onset` temiz (bizim `--init tabcrnn_deep2.pt` korundu);
  `gaps_score.align` dönüşü iki tarafın alanlarını birlikte veriyor (Katman 5: `order`, `lens`, `score_x`; Armağan:
  `order_source`, `bar_quality`). Eski sürümle 25 GAPS kaydında tüm alanlar **birebir** → ritim etiketleri etkilenmedi.
- `tests/test_edition.py`: arşivlenen 2a log-lineer sınıfları `archive.gtab.decoding.edition_loglinear`'dan.
- ASCII TAB aracı `scripts/transcribe.py` → **`scripts/app/transcribe_ascii.py`** (yapı: `scripts/<data|train|eval|app>`).
  Katman 5 ritmi eklendi: `--meter` + `--taps` (tap modu) / `--bpm` (yarı otomatik) → TAB'a ölçü çizgileri (`|`);
  `render_ascii_tab(bars=None)` varsayılanı eskisiyle birebir. Dokunuş dosyası okuma ortak: `gtab.decoding.taps.load_taps`.
- Alınmayanlar: Armağan devlogları, `docs/audit/`, `.vscode`, `.patch`.

## Ek B — Yapı düzenlemeleri (16 Ekim; Katman 6 öncesi)

- **Model yığını + bağımlılık denetimi (genel sistem):** `gtab/models/stack.py` + `checkpoints/active.json` (yoksa
  varsayılan). Ürün ve ölçüm varsayılanları (`pipeline.load_models`, `load_model`, `transcribe_ascii`, `eval_rhythm`,
  `eval_edition`, `diagnose_pitch`, `build_beat_feats`, `train_beat`, `check_inference_regression`) tek yerden okur.
  Taban ses modeline bağlı olanlar: kalibrasyon (`<taban>.cal.json`), BeatNet (checkpoint `feats` alanı), BeatNet önbelleği,
  regresyon parmak izi. `python -m scripts.eval.check_stack` denetler ve eskiyen her parça için yeniden üretme komutunu
  yazar; yeni taban: `--set acoustic=<yeni>.pt cal=<yeni>.cal.json`. `load_models` uyumsuz kalibrasyon / BeatNet ile
  çalışmayı reddeder (sessiz bozulma yok). Sembolik modeller (rhythm_q, edition, transitions) ses modelinden bağımsız.
- `eval_pitch --splits` varsayılanı `gaps_val val val_comp` (önce `gaps_test` açıyordu); gaps_test yalnız raporda açıkça.
- `requirements.txt`: Python 3.10 (.venv 3.10.11; Armağan notu 3.11 diyordu).
- Doğrulama: 204 test OK, `check_protocol` 17/17, regresyon yok, `check_stack` tutarlı.

### Açık notlar (kapanmadı, unutulmasın)
- **A teli:** solo A teli tel doğruluğu `tabcrnn_gaps.pt` (README10) 0.600 → **deep2 0.641** (oracle notalar, GuitarSet;
  Armağan ölçümü). Kök neden çalış alışkanlığı (eğitim icracıları bu sesleri %60 D telinde çalıyor). Ağırlık / hex denemeleri
  etkisiz. **İleride dönülecek; güncel değer 0.641 üzerinden.**
- **Tremolo (<100 ms tekrarlar):** Katman 4'te, teknik veri setleri ve teknikler hazır olunca.
- **`edition_seq_rec.pt`:** ürüne alınacak, acelesi yok.
- **Katman 6'da çıkacak sorunlar Katman 6'da çözülecek** (ör. taban değişirse BeatNet yeniden eğitimi; `check_stack`
  bunu gösterir).
