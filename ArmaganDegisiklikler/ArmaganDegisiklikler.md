# Armağan Değişiklikleri — GitHub sürümünün üzerine yapılanlar

> **Karşılaştırma tabanı:** GitHub `origin/Armagan-katman-4` @ `546c7bd`
> ("feat(eval): add overlap-aware guitar fingering playability metrics").
> **Çalışma tarihleri:** 14–16 Ekim 2026. **Commit/push yapılmadı.**
> Bu dosya iki ara devlog'un (README19 bakım turu + README20 hex deneyi) tek ve son hâlidir.
> Her konu önce **sade dille**, sonra **teknik** olarak; her yerde **Önce → Şimdi**.
> Sayılar: GuitarSet = çelik telli akustik (validation), GAPS = naylon klasik. GuitarSet tab F1
> **klasik gitar TAB doğruluğu değildir**; klasik gitarda tel/perde etiketi olmadığı için ölçülemez.

Dosyaların yeri:

| Yol | Ne |
|---|---|
| `FinalProject/ArmaganDegisiklikler/ArmaganDegisiklikler.md` | bu dosya (repoda) |
| `FinalProject/ArmaganDegisiklikler/degisiklikler.patch` | GitHub sürümüne göre satır satır fark (`git diff 546c7bd`), yeni dosyalar dahil (repoda) |
| Değişen/eklenen kodlar | **kendi yerlerinde** (`gtab/`, `scripts/`, `tests/` …), §1'deki listeye bakın (repoda) |
| `Graduation Project/ArmaganDegisiklikler/` | aynı 18 kod dosyasının, klasör yapısı korunmuş **inceleme kopyası** (repo dışında, push'a girmez) |

---

## 0. Tek bakışta: Önce → Şimdi

### Model ve ürün sonuçları

| Ne | Önce (GitHub) | Şimdi | Not |
|---|---|---|---|
| Ana ses modeli | `tabcrnn_deep2.pt` | **aynı** (dokunulmadı) | 7 eğitim denemesinin hiçbiri geçemedi (§2.8–2.10) |
| GuitarSet tab F1 (`val`) | 0.777 | 0.777 | değişmedi |
| GuitarSet nota F1 solo / akor | 0.924 / 0.807 | aynı | |
| Klasik gitar nota F1 (`gaps_val`) | 0.780 | aynı | |
| **TAB programının kullanıcıya verdiği** tab F1 | **0.751** (test edilmemiş ayar) | **0.777** | §2.1 — yalnız varsayılan ayar |
| **Parmak/tel önerisi modeli** tel / parmak (`gaps_val`, eski 15 kayıt) | 0.766 / 0.675 | **0.784 / 0.700** (aday) | §2.3 — ürüne alınmadı, karar bekliyor |
| Solo A teli tel doğruluğu | hiç ölçülmemişti ("0.60" iki model eskiydi) | **0.641** ölçüldü | artırılamadı, kök neden bulundu (§2.8) |

### Veri, kod, test, ortam

| Ne | Önce | Şimdi |
|---|---|---|
| Tel/parmak etiketli GAPS kaydı | 165 / 261 | **236 / 261** (+71, ayrı klasör) |
| Otomatik test | 170 | **204** |
| Ana tab F1 ölçüm komutu | 15 ayar elle | **`--load-cal` tek satır** |
| Eğitim yanlışlıkla eski modelin üstüne yazabiliyor | evet (`edition_seq.pt` dahil) | **hayır**, durup uyarıyor |
| ASCII TAB | `712` gibi yapışık perdeler; hızlı notalar siliniyor (örnek kayıt 36/995) | ayrık; **995/995** |
| "±0.01 tohum gürültüsü" varsayımı | hiç ölçülmemişti | **ölçüldü: 0.009** |
| VS Code | genel Python, IntelliCode hatası | `.venv`, hata yok |
| C: boş alan | 7.6 GB | **23 GB** |

---

## 1. Dosya dosya: neler eklendi, neler değişti

### Değişen dosyalar (GitHub'da vardı)

| Dosya | Önce | Şimdi |
|---|---|---|
| `gtab/data/gaps_score.py` | Tekrar işaretleri sesteki ölçü sayısını tutturamazsa kayıt tamamen atılıyordu. Testi yoktu. | `bar_similarity`, `order_from_content`, `content_order`: ölçü sırası **çalınan içerikten** bulunuyor. `align(recover=False)` varsayılanı eskisiyle birebir; `score_notes(order=None)`. |
| `gtab/data/torch_dataset.py` | `GuitarSetSeq` yalnız mikrofon + etiket veriyordu. | İsteğe bağlı `hex_dir`: tel başına hex spektrumu hedefi (verilmezse eski 6'lı öğe aynen). |
| `gtab/decoding/decode.py` | `render_ascii_tab`: 2 karakterlik kutu, dolu kutunun üstüne yazıyordu. | Kutu = en uzun perde + 1; dolu kutuda nota sonraki boş kutuya kayıyor. Tek basamaklı parçada çıktı byte düzeyinde aynı. |
| `gtab/models/nets.py` | `forward_full(x)` | `forward_full(x, return_features=False)` (yardımcı kafa için; varsayılan çağrı birebir aynı). |
| `gtab/utils.py` | `set_seed`, `get_device` | + `guard_new_checkpoint(path, overwrite)` |
| `scripts/data/build_gaps_tab.py` | Yalnız `gaps_tab`'a yazıyordu. | + `--recover-repeats`, `--out-dir` (varsayılan `gaps_tab_rec`; `gaps_tab`'ın üstüne yazmayı reddediyor), karantinadaki kayda hiç etiket üretmiyor. |
| `scripts/eval/eval_hmm.py` | Ayar kartını okuyamıyordu (15 bayrak elle). | + `--load-cal` (`val` alanı); eski bayraklar hâlâ çalışıyor. |
| `scripts/train/train_onset.py` | `--ckpt` unutulursa `tabcrnn_onset.pt` eziliyordu; `--seed` yoktu. | + `--overwrite` koruması, `--seed`, `--string-weights`, `--prior-debias`, `--prior-cap`, `--hex-weight`, `--save-epochs`. Hepsinin varsayılanı eski eğitimle birebir. |
| `scripts/train/train_edition_seq.py` | Varsayılan çıktı = kullanılan `edition_seq.pt` (eziliyordu); etiket klasörü sabit. | + `--overwrite` koruması, `--tab-dir`. |
| `tests/test_edition.py` | `mir_eval` yoksa testler çöküyordu. | `HAVE_MIR_EVAL` kapısı → atlanıyor. |
| `requirements.txt` | "Python 3.10" | "Python 3.11" (gerçek ortam 3.11.9) |
| `.vscode/settings.json` | Genel Python (`ms-python.python:system`) | `.venv` yorumlayıcısı + terminalde otomatik aktivasyon |
| `scripts/transcribe.py` | *(GitHub'da yok; makinede takipsiz vardı)* `--w-transition` varsayılanı 1.0 | Varsayılan **0.0** (§2.1). Dosya ilk kez sürüme girecek. |

### Yeni dosyalar

| Dosya | Ne |
|---|---|
| `gtab/data/hex.py` | Hex WAV → tel başına log-CQT (6 × 96, ortak dB referansı), hizalama ölçüsü |
| `scripts/data/build_hex.py` | `data/cache/hex/` önbelleği + hizalama kontrolü |
| `tests/test_ascii_tab.py` | 7 test — TAB görünümü |
| `tests/test_gaps_score.py` | 19 test — partisyon ayrıştırma, tekrar açma, eşleme, içerikten sıra |
| `tests/test_hex.py` | 8 test — hex spektrumu, veri yükleyici, checkpoint uyumu |

Proje dışı: `Graduation Project/.vscode/settings.json` (çalışma alanı kökü = üst klasör), iki IntelliCode
eklentisi kaldırıldı. Git'e girmeyen üretilmiş veri: `data/cache/gaps_tab_rec/` (236 kayıt),
`data/cache/hex/` (214 MB). Aday model: `checkpoints/deneme_readme19/edition_seq_rec.pt`.

---

## 2. Konu konu

### 2.1 Ürün, ölçülen ayarla çalışmıyordu

**Sade dil.** Program notayı bulduktan sonra teli seçerken "gitaristler genelde şöyle çalar" alışkanlık
bilgisini kullanabiliyor. Testlerde bu bilgi işe yaramadığı için kapalı ölçülmüştü (%77.7), ama TAB
programı onu **açık** kullanıyordu. Yağmur lastiğiyle test edip yaz lastiğiyle teslim etmek gibi.

**Teknik.** `eval_hmm` taraması her tabanda `w_transition = 0` seçiyor (README11:178, README13:244);
`eval_gaps_tab`, `eval_edition` da greedy (= 0) ölçüyor. `transcribe.py` varsayılanı 1.0 idi.

| `w_tr` (deep2, GuitarSet `val`, eşik 0.7) | P | R | tab F1 |
|---|---|---|---|
| **0.0** (ölçülen) | 0.753 | 0.802 | **0.777** |
| 1.0 (eski ürün) | 0.729 | 0.776 | 0.751 |

Klasik kayıt `014_vSswc`: notalar aynı (995), **215 notanın teli** değişti; ses modeli ile parmak
modelinin konum anlaşmazlığı 570 → 446. **Önce → Şimdi:** varsayılan 1.0 → 0.0. Not: Katman 5 dalındaki
`gtab/pipeline.py:64` aynı değeri (1.0) sabit kodluyor; birleşince orada da düzeltilmeli.

### 2.2 Klasik gitar kayıtlarının %37'si boşa gidiyordu → 71 kayıt kurtarıldı

**Sade dil.** Nota kâğıdındaki "başa dön" işaretlerini açınca çıkan ölçü sayısı sesle tutmayınca kayıt
atılıyordu (75 kayıt). Sebep işaretlerin yanlış okunması değil: icracılar kâğıtta **yazmayan** tekrarlar
çalmış (ör. `112_mf1wc`: 64 ölçü, işaret yok, parça **iki kez** çalınmış = 128 ölçü; barok danslarda
gelenek). Çözüm: sesteki her ölçünün kâğıttaki hangi ölçü olduğunu çalınan notalardan bulmak.

**Teknik.** Teşhis (75 kayıt): 55 işaretli + 16 işaretsiz kayıtta syncpoint > açılan ölçü; 4'ünde tersi;
D.C./D.S./Coda **0**. Yöntem: syncpoint aralıklarındaki MIDI perdeleri ↔ partisyon ölçüleri çoklu küme
Dice benzerliği → Viterbi (sonraki ölçü bedava, sıçrama `jump = 1.0`) → kapo −5…+5.

| Doğrulama (tekrar yapısı **doğru** 186 kayıt, cevap biliniyor) | Sonuç |
|---|---|
| Varsayılan `align` == diskteki `gaps_tab` | **165/165 birebir** |
| İçerikten sıra == işaretlerden sıra (`jump` 0.5 / 1.0 / 2.0) | %97.2 / **%97.4** / %96.7 |
| Üretilen etiketler eski doğru etiketlerle tel / perde / parmak | **%99.91 / %99.91 / %99.76** (133.261 nota) |

| | Önce `gaps_tab` | Şimdi `gaps_tab_rec` |
|---|---|---|
| gaps_train / gaps_test kullanılabilir | 142 / 23 | **208 / 28** |
| Etiketli nota · perde roll uyumu · \|dt\| medyan | %80.9 · %98.93 · 16.4 ms | %80.8 · %98.83 · 15.7 ms |

Kurtarılamayan 3 kayıt (etiketli < %50). Karantinadaki `395_w41wc` etiketlenmez. Ortak 165 kayıt iki
klasörde dizi dizi aynı; `gaps_tab` ve onu kullanan her şey dokunulmadan duruyor.

### 2.3 Kurtarılan verilerle parmak modeli

**Sade dil.** 71 kurtarılan kayıt ilk kez kullanıldı: parmak/tel önerisi modeli 127 yerine 182 kayıtla
eğitildi ve **hem eski hem yeni sınavda** daha iyi çıktı.

| Model | Eski `gaps_val` (15) tel / parmak | Kurtarılmış `gaps_val` (26) tel / parmak |
|---|---|---|
| `edition_seq.pt` (ürün, önce) | 0.766 / 0.675 | 0.772 / 0.644 |
| `edition_seq_rec.pt` (aday) | **0.784 / 0.700** | **0.787 / 0.673** |

Epoch seçimi kurtarılmış `gaps_val`'de → hafif iyimser. Ürüne almadan önce: çalınabilirlik ölçümü,
ikinci tohum, `test_edition.py` referans kararları. `gaps_test` açılmadı.

### 2.4 Ana ölçüm tek satır (`eval_hmm --load-cal`)

**Önce:**
`python -m scripts.eval.eval_hmm --model crnn --ckpt tabcrnn_deep2.pt --threshold 0.7 --onset-thr 0.2 --off-ratio 1.0 --refractory 0 --fallback 10 --peak --reattack 0.3 --rise-keep 0.0 --rise-split 8.0 --offset-thr 0.7 --combine noisyor --peak-pick --prominence 0.25 --min-dist 2 --re-valley 0.0 --onset-source pitch`

**Şimdi:**
`python -m scripts.eval.eval_hmm --model crnn --ckpt tabcrnn_deep2.pt --load-cal tabcrnn_deep2.cal.json`

İki komutun çıktısı `diff` ile karşılaştırıldı; tek fark başa eklenen bilgi satırı.

### 2.5 Eğitim güvenliği ve tekrarlanabilirlik

**Sade dil.** Yeni modelin adını yazmayı unutunca eski bir modelin, parmak modelinde doğrudan
**kullanılan** modelin üstüne yazılıyordu. Artık program duruyor.
**Teknik.** `guard_new_checkpoint` epoch 0 kaydından önce. Doğrulandı: bayraksız `train_onset`,
`--ckpt tabcrnn_deep2.pt` ve bayraksız `train_edition_seq` üçü de duruyor. `--seed` (varsayılan 1 = eski).

### 2.6 Testler: 170 → 204

| Yeni / değişen | Kapsam |
|---|---|
| `test_gaps_score.py` (19) | Tekrar açma (basit, işaretsiz, 1./2. son), tick/akor/bağ/süs notası, eksik `<duration>`, tel-perde-parmak, TAB'dan perde, parmağın nota portresinden taşınması, syncpoint enterpolasyonu, bire bir eşleme, içerikten sıra (işaretsiz tüm parça tekrarı, bölüm tekrarı, tek yanlış notada sıçramama) |
| `test_ascii_tab.py` (7) | Tek basamaklı parçada eski çıktıyla byte eşitliği, `-7-12` ayrımı, çakışan notaların korunması, 200 rastgele notada kayıp 0 |
| `test_hex.py` (8) | Spektrum şekli/aralığı, ortak dB referansı, uyum ölçüsü, 6 kanal zorunluluğu, veri yükleyici, `forward_full` varsayılanının birebirliği |
| `test_edition.py` | `mir_eval` kapısı (engellenerek denendi: 10 atlandı, 0 hata) |

### 2.7 ASCII TAB görünümü

**Önce:** `---712----` (7 ve 12 yapışık); aynı 0.15 sn kutusuna düşen ikinci nota birinciyi siliyordu
(`014_vSswc`: 995 notadan 959 görünüyordu). **Şimdi:** `--7-12-`; çakışan nota sonraki boş kutuya
kayıyor; **995/995**, yapışık sayı 0. Model ve ölçümler etkilenmez (yalnız ekrana yazdırma).

### 2.8 A teli: ölçüm, kök neden, başarısız denemeler

**Sade dil.** "A teli %60" iki model önceki eski modele aitti. Güncel modelde ilk kez ölçüldü: **0.641**.
Asıl bulgu: deneme sınavındaki gitarist bu sesleri A telinde çalıyor, eğitimdeki 5 gitarist ise aynı
sesleri çoğunlukla (%60) D telinde. Model teli büyük ölçüde bu **alışkanlıktan** seçiyor.

| Tel doğruluğu (oracle notalar, GuitarSet) | `tabcrnn_gaps` (README10) | deep2 |
|---|---|---|
| Solo toplam | 0.797 | 0.878 |
| **Solo A teli** (n=145) | 0.600 | **0.641** |
| Akor A teli (n=702) | 0.942 | 0.929 |

| deep2 A teli doğruluğu, sesin eğitimde A'da çalınma sıklığına göre | |
|---|---|
| A yaygın (> %65, n=43) | **0.95** |
| arada (n=37) | 0.68 |
| A nadir (< %35, n=65) | **0.42** |

Eğitim verisinde dolu kare payı: solo `train` A %8.2, akor `train_comp` A %19.2. Hataların 52'nin 51'i
D teline gidiyor. `CLAUDE.md`'deki "A teli 0.66" hiçbir ölçümle uyuşmuyor.

**Denemeler** (hepsi deep2'den, deep2 tarifi, 5 tur, `--lr 1e-4`, karar **son turda**, ölçüt: solo A ≥ 0.69):

| Deneme | Solo A | Tab F1 | Sonuç |
|---|---|---|---|
| kontrol (ağırlık yok), tohum 1 / 2 | 0.628 / 0.607 | 0.777 / 0.768 | — |
| A ×2 | 0.648 | — | gürültü |
| A ×3 | 0.662 | 0.777 | kontrolden +0.035, ölçüt yok |
| "nadir tel" ağırlığı (mutlak / göreli pay) | 0.641 / 0.641 | — | etkisiz |
| hex yardımcı kaybı, tohum 1 / 2 (§2.9) | 0.628 / 0.648 | 0.771 / 0.765 | etkisiz |

Ağırlık yolu ve hex, alışkanlığı kırmadı. **Önce → Şimdi:** model aynı, ama sorunun doğası belli. Tezde:
bu ölçü kısmen **çalış üslubunu** ölçüyor; klasik ürün tel/konumu zaten "edisyon önerisi" modundan veriyor.

### 2.9 Hexafonik yardımcı eğitim

**Sade dil.** GuitarSet her teli ayrı kanaldan kaydetmiş. Model, mikrofon sesinden "hangi telde ne kadar
ses var"ı da tahmin etmeyi öğrensin diye bir yardımcı görev eklendi. Model bunu öğrendi (kayıp 0.065 →
0.018), ama tel **kararı** değişmedi: sorun duymak değil, duyduğunu karara yansıtmak.

**Teknik.** Zenodo 3371780 `audio_hex-pickup_debleeded` (3.6 GB → 5.5 GB WAV, 25 dk) → `build_hex` →
`data/cache/hex` (6 × 96 bant, ortak dB referansı, 214 MB). Hizalama: tek telli karelerde en güçlü kanal =
etiket teli, solo %91–93 (±kaydırma testi: hizalı); akorda %53–55, çünkü karelerin %58'inde etiketsiz
teller hâlâ çınlıyor. Yardımcı kafa model **dışında** (checkpoint uyumlu), `--hex-weight 5`. Sonuç §2.8
tablosunda. Ham hex verisi sonradan silindi (önbellek duruyor; gerekirse ~25 dk yeniden indirilir).

### 2.10 Ölçüm hatası, düzeltmesi ve overfit kontrolü

**Sade dil.** Bir ara "ince ayar tab F1'i 0.777'den 0.802'ye çıkardı" diye raporladım. Kalibrasyon ve
overfit kontrolü sırasında sayılar çelişince hatayı buldum: sonuç tablosunda yanlış sütunu (recall)
okumuşum. Gerçekte iyileşme yoktu.

**Teknik.** `grep "greedy (nota" | awk '{print $5}'` → satır `greedy (nota basi)  P  R  F1`, 5. alan =
**R** (deep2'de de 0.802). Doğru sütunla bütün denemeler yeniden ölçüldü (§2.8 tablosu). Kalibrasyon
durduruldu. Overfit kontrolü (sabit çözümleme, deep2 `val` kartı):

| Parça | deep2 | aday tohum 1 | aday tohum 2 |
|---|---|---|---|
| `train` (eğitim) | 0.917 | 0.918 | 0.919 |
| `val` (deneme) | 0.777 | 0.777 | 0.768 |
| Çapraz eşik (yarı A'da seç → B'de ölç ve tersi) | **0.775** | 0.773 | 0.765 |

Ezber yok (eğitimde de kazanç +0.001), iyileşme de yok. **Tohum gürültüsü ilk kez ölçüldü:** tab F1
0.009, A teli 0.021 → "±0.01" varsayımı tab F1 için doğru.

### 2.11 Ortam ve disk

VS Code: `.venv` yorumlayıcısı, terminal `FinalProject`'te açılıyor, IntelliCode (Microsoft'un artık
desteklemediği eklenti) kaldırıldı. Disk 7.6 → 23 GB: pip önbelleği (~9 GB), TEMP (~4.7), npm (~1.8),
CrashDumps, hex ham verisi (5.5), başarısız deneme checkpoint'leri. Dokunulmayanlar (kullanıcı kararı):
İndirilenler 20 GB, Geri Dönüşüm 1.8 GB, Windows Update 10 GB.

---

## 3. Doğrulama özeti (son durum)

| Kontrol | Sonuç |
|---|---|
| `python -m unittest discover -s tests -t .` | **204/204 OK** |
| `python -m scripts.eval.check_protocol` | 17/17 |
| `python -m scripts.eval.check_inference_regression` | REGRESYON YOK (deep2 çıktısı birebir) |
| Ana metrikler (deep2) | tab F1 0.777, oracle tel 0.878, nota F1 0.924 / 0.807, `gaps_val` 0.780 — önceki ile aynı |

## 4. Yapılmayanlar ve bekleyen kararlar

| # | Konu |
|---|---|
| K1 | Commit (bu klasör + değişen dosyalar). `scripts/transcribe.py` ilk kez sürüme girecek. |
| K2 | `docs/audit/` (14 Ekim denetimi) git'e girsin mi |
| K3 | `eval_pitch` / `eval_gaps_tab` varsayılanı final setini (`gaps_test`) açıyor; değiştirilsin mi. O zamana kadar `--splits` her zaman yazılır. |
| — | `edition_seq_rec.pt` ürüne alınsın mı (§2.3) |
| Askıda | A teli sıradaki adım (öğretmen-öğrenci / SynthTab), Katman 5 bağımlılıkları (`beat_net` ↔ deep2, arşivlenen eski adlar), doğruluk sorunları (<100 ms tekrar, akor fragmanları, SynthTab deneyi) |

## 5. Uygulama ve çalıştırma

Değişiklikler projede kendi yerlerinde; bu dal çekildiğinde hepsi gelir. Yalnız `546c7bd`'deki başka bir
klona uygulamak için: `git apply ArmaganDegisiklikler/degisiklikler.patch` (temiz uygulandığı doğrulandı).

```powershell
python -m unittest discover -s tests -t .                                   # 204 test
python -m scripts.eval.eval_hmm --model crnn --ckpt tabcrnn_deep2.pt --load-cal tabcrnn_deep2.cal.json
python -m scripts.data.build_gaps_tab --recover-repeats                      # ~1 dk -> data/cache/gaps_tab_rec
python -m scripts.train.train_edition_seq --out <YENI>.pt --tab-dir data/cache/gaps_tab_rec
python -m scripts.data.build_hex                                             # hex ham verisi gerekir
python -m scripts.eval.diagnose_strings --ckpt tabcrnn_deep2.pt --splits val val_comp --threshold 0.8
python -m scripts.transcribe --audio data/raw/gaps_hf/audio/014_vSswc.wav --fingering
```
