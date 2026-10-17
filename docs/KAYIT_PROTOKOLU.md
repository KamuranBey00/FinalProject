# Kendi klasik gitar veri setimiz — kayıt protokolü

**Amaç:** GuitarSet ve GAPS'te olmayan iki şeyi ölçmek.
- **Gerçek dünya (Katman 6):** telefon / mikrofon, gerçek oda, bizim gitarlarımız.
- **Teknikler (Katman 4, sonra):** legato, slide, vibrato, armonik.

Ne çalındığını önceden bildiğimiz için etiketleri elle yazmamız gerekmiyor:
- Her egzersizin **notası, teli ve perdesi** aşağıda yazılı.
- Zamanlar, metronoma göre çalındığı için yaklaşık biliniyor. Kesin zamanları sonra bir hizalama aracı sesten bulacak.

Bu set **eğitim için değil, ölçmek için**. Küçük ama etiketi kesin doğru olmalı.

**Kim çalmalı:** Egzersizler başlangıç düzeyinde, ama klasik gitar çalabilen biri gerekir. Ne kadar çok farklı kişi çalarsa
o kadar iyi; her kişi ayrı bir "icracı" sayılır, en az 2 kişi öneriyorum. Sette bulunan gitarist sayısı ve gitar sayısı
not edilsin.

---

## 1. Hazırlık (her oturumda)

| Ne | Nasıl |
|---|---|
| Gitar | Naylon telli klasik gitar. Akort cihazı ya da uygulamasıyla **standart akort** (E A D G B E), A = 440 Hz |
| Oda | Normal bir oda (yatak odası / salon). Klima, fan ve TV kapalı. İstenirse bazı egzersizler ikinci kez yankılı bir yerde (banyo, koridor) |
| Kayıt cihazı | **Telefon** (ses kaydedici, en yüksek kalite, WAV ya da en az 256 kbps). Varsa **aynı anda** bir de bilgisayar / USB mikrofonla kayıt: iki cihaz = cihaz dayanıklılığı ölçümü |
| Konum | Cihaz gitarın ses deliğine ~50 cm, masada sabit (elde tutma). Tüm oturum boyunca aynı yerde |
| Metronom | **Kulaklıkla** dinlenir; sesi kayda girmemeli. Varsayılan **60 BPM**, egzersizde başka yazıyorsa o |
| Başlangıç | Her kaydın başında **2 sn sessizlik**, sonra metronomun 4 tıkı boyunca bekle, 5. tıkta başla. Sonda 2 sn sessizlik |
| Hata | Yanlış bastıysan kaydı silip baştan al (etiket "doğru çalınmış" varsayıyor) |

**Dosya adı:** `<kişi>_<cihaz>_<oda>_<egzersiz>_<tekrar>.wav`, örneğin `armagan_telefon_salon_E04_1.wav`.
Dosyalar `data/raw/own/` klasörüne konur (git'e girmez). Ayrıca `data/raw/own/meta.csv` dosyasına satır başına
`dosya, kisi, gitar, cihaz, oda, egzersiz, bpm, akort, not` yazılır.

**Gösterim:** Tel 1 = en ince (e), tel 6 = en kalın (E). `t2p5` = 2. tel 5. perde, `t3p0` = 3. tel boş.

---

## 2. Egzersizler

Her egzersiz **2 kez** çalınır. Süreler 60 BPM'e göre yaklaşıktır.

### A. Tek ses ve tel ayrımı (Katman 6: tel / perde, A teli)
| Kod | Ne çal | Tempo / ritim | ~Süre |
|---|---|---|---|
| **E01** | Her tel için sırayla perde 0-1-2-3-4-5, sonra 5-4-3-2-1-0. Tel 6'dan başla, tel 1'de bitir | 60, her tıkta 1 nota | 2 dk |
| **E02** | **Aynı nota farklı tellerde.** E4: t1p0, t2p5, t3p9, t4p14 · B3: t2p0, t3p4, t4p9, t5p14 · G3: t3p0, t4p5, t5p10, t6p15 · D3: t4p0, t5p5, t6p10. Her konum 2 kez | 60, her tıkta 1 nota | 1 dk |
| **E03** | Do majör gam, 1. pozisyon: C3 (t5p3) → G4 (t1p3) çık, sonra geri in | 60, her tıkta 1 nota | 1 dk |

### B. Çok ses (Katman 6: polifoni)
| Kod | Ne çal | Tempo / ritim | ~Süre |
|---|---|---|---|
| **E04** | **Arpej** p-i-m-a (bas, sonra tel 3, 2, 1) şu akorlarla, her akor 2 ölçü: C, Am, Dm, E, G, sonra tekrar C | 60, tık başına 2 nota | 1.5 dk |
| **E05** | **Akorlar aynı anda** (parmaklar birlikte çeker): 2 ses (t6p0 + t1p0), 3 ses (C: t5p3 t3p0 t2p1), 4 ses (Am: t5p0 t4p2 t3p2 t2p1), 6 ses (E: hepsi). Her biri 4 kez, aralarda 1 tık sus | 60 | 1 dk |
| **E06** | **Bas + melodi:** bas t6p0 her tıkta, üstte t1 perde 0-1-3-5-3-1-0 ikilik notalarla | 60 | 1 dk |

### C. Hızlı tekrar / tremolo (sonra: Katman 4)
| Kod | Ne çal | Tempo / ritim | ~Süre |
|---|---|---|---|
| **E07** | **Tremolo** p-a-m-i: bas t5p0, üç tekrar t1p7 (B4). 60, 80, 100, 120 BPM'de ayrı ayrı (her tık = 1 grup p-a-m-i) | 60 / 80 / 100 / 120 | 1.5 dk |
| **E08** | Aynı nota i-m sırayla: t2p5, 4 nota / tık | 60, 90 | 1 dk |

### D. Teknikler (Katman 4; şimdi kaydedilir, sonra kullanılır)
| Kod | Ne çal | Tempo / ritim | ~Süre |
|---|---|---|---|
| **E09** | **Legato çıkış (hammer-on):** her telde 0→2 ve 2→4 (ilk nota çekilir, ikincisi yalnız sol elle vurulur) | 60, tık başına 1 çift | 1 dk |
| **E10** | **Legato iniş (pull-off):** her telde 2→0 ve 4→2 | 60 | 1 dk |
| **E11** | **Slide (glissando):** t3 2→5, t2 5→7, t1 7→10 (ikinci nota çekilmez, parmak kayar) | 60, 2 tıkta 1 slide | 1 dk |
| **E12** | **Vibrato çiftleri:** t2p7 4 tık **vibratosuz**, sonra aynı nota 4 tık **vibratolu**. t3p5, t1p12 ile tekrarla | 60 | 1.5 dk |
| **E13** | **Doğal armonik:** 12., 7. ve 5. perde üstünde, her telde 1'er kez (parmak teli değmeden üstte durur) | 60, 2 tıkta 1 | 1.5 dk |

### E. Akort ve gerçek müzik (Katman 6)
| Kod | Ne çal | Tempo / ritim | ~Süre |
|---|---|---|---|
| **E14** | **6. teli D'ye indir (drop D)**, sonra E01'in yalnız 6. ve 5. tel kısmı + E04 (D akoruyla) | 60 | 1.5 dk |
| **E15** | **Serbest parça:** çalabildiğin 1–2 kısa klasik parça (1–2 dk). Parçanın adını ve bestecisini meta.csv'ye yaz; partisyonu GAPS'te varsa etiketi oradan çıkarırız. Metronomsuz, doğal tempoda | serbest | 2–4 dk |

| **E16** | **Çalarken ses:** E04 arpejini ya da E15 parçasını çalarken (a) şarkı söyle / mırıldan, (b) biri yanında konuşsun, (c) TV / müzik arka planda kısık sesle. Her biri 1 kez | 60 / serbest | 3 dk |

**Toplam:** 2 tekrarla kişi ve cihaz başına yaklaşık 35–40 dakika. Tek oturumda bitmesi gerekmiyor.

---

## 3. Öncelik (vakit azsa)
1. **E01, E02, E04, E05, E15, E16**: Katman 6'nın ihtiyacı (tel, polifoni, gerçek müzik, çalarken insan sesi).
2. **E07, E08, E14**: tremolo ve akort.
3. **E09–E13**: teknikler (Katman 4'e kadar bekleyebilir).

## 4. Kayıttan sonra
Dosyaları `data/raw/own/` altına koy ve `meta.csv`'yi doldur. Etiket dosyalarını egzersiz tablosundan ve hizalama
aracından ben üreteceğim (`scripts/data/build_own.py`, Katman 6 Adım 0c). Sonra ölçüm `eval_robust` ve `eval_pitch`
ile yapılır.
