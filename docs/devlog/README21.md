# Not — TAB görüntüleyici sesi: piyano → naylon gitar, tık ve kırpılma giderildi

> **Durum (20 Ekim 2026):** küçük bir ürün düzeltmesi, yeni katman değil. Model, çözümleme, checkpoint ve önbellek
> değişmedi; yalnız **çalma sesi** değişti. Dal `Armagan-katman-5.2-ses` (`katman-5.2-urun` üzerine).
> GuitarSet = çelik telli akustik, GAPS = naylon klasik; aşağıdaki ses ölçümlerinin hepsi GAPS kayıtlarında.

## Sorun
`scripts/app/tab_viewer.html` çalarken ses kulak tırmalıyordu: naylon gitar gibi değil, tiz ve cızırtılı.

## Kök nedenler (ölçüldü)
1. **Piyano çalıyordu.** `gtab/render/musicxml.py` enstrüman yazmıyordu → alphaTab program 0 (Acoustic Grand Piano)
   seçiyor ve bunu ilk vuruşa otomasyon olarak sabitliyor (görüntüleyiciden `playbackInfo.program` ile değişmiyor).
2. **Ses bankası çok küçük.** CDN'deki `sonivox.sf2` 128 enstrümanı 1.3 MB'ta taşıyor; naylon gitar = 3 adet
   20–60 ms'lik örnek, döngüyle uzatılıyor (22 kHz) → "vızıltılı" tını.
3. **Kırpılma.** Bütün notalar forte (velocity 95), ses düzeyi 1.0 → akorlarda tepe 1.0'ı aşıyor (cızırtı).
4. **Tık (yeni bankada).** FreePats örneklerinin 13/48'inde tel parmaktan kurtulduğu an 2–6 örnek (0.05–0.14 ms)
   genişliğinde, normal adımın 100–430 katı bir sıçrama var (yakın mikrofon + gürültü işleme). Reddedilen hipotezler:
   örnek başı kesik (fade-in yalnız 58 → 48 tık/dk), exclusive class (bankada yok), ses çalma limiti / voice stealing
   (alphaTab gerektikçe ses ekliyor), sönüm ayarı yanlış bölgede (global zone doğru, ~1.5 sn).

## Ne yapıldı
| Dosya | Değişiklik |
|---|---|
| `gtab/render/musicxml.py` | Her iz için `<score-instrument>` + `<midi-instrument><midi-program>25</midi-program>` (1 tabanlı = GM 24 naylon). Notalar birebir aynı. |
| `scripts/data/build_soundfont.py` | FreePats "Spanish classical guitar" SF2 (CC0) → preset 0 → 24, de-click (sıçrama 2 ms'de yumuşatılır, eşik 10×), 3 ms fade-in, SF3 (Ogg Vorbis 0.8; 19.8 → 2.4 MB), base64 `.js` |
| `scripts/app/soundfont/nylon_guitar.sf3.js` | Üretilen ses bankası (3.2 MB). `file://` ile de yüklenir (`<script src>`; `fetch` yerel dosyada engelli). |
| `scripts/app/tab_viewer.html` | Yerel banka `blob:` adresiyle yüklenir (yoksa sonivox'a düşer); ses düzeyi 0.5 (`playerReady`'de); her iz program 24'e zorlanır (enstrüman yazılmadan üretilmiş eski dosyalar bankada 24 olmadığı için sessiz kalırdı). |

Uygulama tuzakları (kayıt için): Windows'ta libsndfile tek seferde uzun Vorbis yazarken çöküyor (bloklarla yazılıyor);
alphaTab RIFF okuyucusu tek uzunluklu bloğun dolgu baytını atlamıyor (`smpl` çift uzunluğa tamamlanıyor, yoksa banka
sessiz); oynatıcı hazır olmadan çağrılan `api.loadSoundFont` kayboluyor. Ogg seri numarası rastgele olduğu için
SF3 her üretimde başka byte'lar verir; çözülmüş ses birebir aynı (48/48 örnek).

## Testler
- `python -m unittest discover -s tests -t .` → 215 test OK.
- **TAB değişmedi:** 300 GAPS kaydı × 2 mod (ses / edisyon) eski (HEAD) ve yeni yazıcıyla üretildi, `<part>` içerikleri
  karşılaştırıldı: **600/600 birebir aynı** (295 515 nota; tel/perde 295 515; parmak 295 515 edisyon modunda). Negatif
  kontrol: tek perde +1 → "farklı". Tempo sabit `--bpm 90 --meter 4/4` (yalnız ölçü çizgilerini etkiler; iki tarafta aynı).
- `scripts/app/tap_recorder.html` değişmedi (git farkı 0 satır).
- Tarayıcı (Playwright, yerel HTTP sunucu): yeni ve eski `.musicxml` → "Hazır.", tüm MIDI kanalları program 24, ses düzeyi
  0.5, çalma ilerliyor, konsolda hata yok (yalnız favicon 404); `soundfont/` yoksa sonivox'a düşüyor. `file://` çift tıklama
  test aracında engelli olduğu için elle denenmeli.

## Ses ölçümü (300 GAPS kaydı, tam uzunluk)
Her kaydın TAB'ı görüntüleyicinin çaldığı gibi (TAB izi susturulmuş) alphaTab 1.8.4'ün kendi dışa aktarıcısıyla
WAV'a çevrildi ve gerçek kayıtla karşılaştırıldı. Hiçbir ayar bu kayıtlara bakılarak seçilmedi; de-click eşikleri yalnız
bankanın 48 örneğinde seçildi. Ses düzeyi kuralı ölçümden önce sabitlendi: **gaps_train'de parçaların ≥ %95'inde
kırpılmasız en yüksek düzey** → 0.5 (0.45: %99.6, 0.5: %97.4, 0.55: %91.8, 0.6: %75.8).

**gaps_test (30 kayıt, yalnız raporlama), medyan:**

| | gerçek kayıt | önce (sonivox, piyano, 1.0) | naylon, de-click yok, 0.6 | **son (naylon + de-click, 0.5)** |
|---|---|---|---|---|
| spektral merkez | 332 Hz | 652 Hz | 312 Hz | **312 Hz** |
| 2 kHz üstü enerji | %0.30 | %1.86 | %0.40 | **%0.40** |
| tık / dk | 0 | 0 | 66 | **0** |
| tık olan parça | — | %0 | %100 | **%0** |
| kırpılan parça | — | %87 | %17 | **%0** |
| atak keskinliği | 4.48 | 6.10 | 6.50 | 6.51 |

Tüm 300 kayıtta: kırpılan parça %89.7 → **%2.0**, tık olan parça **%0**, spektral merkez 656 → 316 Hz (gerçek 358).
Tık ölçüsü: 6 kHz üstü bantta yerel düzeyin 12 katı ve > 0.01 sıçrama, 5 ms içinde tek olay (gerçek kayıtlarda medyan 0).

## Sınırlar
- Ölçüler sayısal; kulakla dinleme kullanıcıda. Sentez kaydın aynısı olmaz (gövde, oda, parmak farkı yok).
- **Kalan fark: atak** gerçekten sert (6.5 / 4.5) ve 2 kHz üstü enerji biraz fazla — bütün notalar aynı şiddette
  (forte) çalıyor. Nota şiddetini sesten taşımak ayrı bir iş.
- FreePats örnekleri 2008'de tek mikrofonla kaydedilmiş, gürültü filtreli (bankanın kendi notu).

## Çalıştırma
```
python -m scripts.data.build_soundfont path/to/SpanishClassicalGuitar-20190618.sf2
```
Kaynak: https://freepats.zenvoid.org/Guitar/acoustic-guitar.html (`SpanishClassicalGuitar-SF2-20190618.7z`).

## Sıradaki
- Kulakla dinleme onayı; gerekirse nota şiddeti (velocity) — model enerji bilgisini MusicXML dinamiğine taşımak.
