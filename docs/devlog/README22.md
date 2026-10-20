# Not — Farklı gitar sesleriyle test: test altyapısı + kusur haritası (Adım 1)

> **Durum (20 Ekim 2026):** yalnız ölçüm, ürün değişikliği yok. Model, çözümleme, checkpoint ve önbellek
> değişmedi. Dal `Armagan-katman-5.2-ses`. Kod `scripts/eval/gv/`, sonuçlar `docs/results/gv/`, paylaşılabilir ses
> `eval_data/`, geri kalanı `python -m scripts.data.fetch_gv_data` ile `data/raw/gv/` altına (git'e girmez).
> GuitarSet = çelik telli akustik, GAPS = naylon klasik; seçim gaps_train'den ayrılan bölümde yapılır,
> gaps_test yalnız raporlama içindir. Tek seed farkı ±0.01 gürültü sayılır.

## Soru
Model GAPS'in tek gitarı dışında başka klasik gitar seslerinde de çalışıyor mu? Ses modunda ve edisyon modunda
çıkan tab gerçek tab'larla ne kadar uyuşuyor? Hata nerede?

## Veri
| Kaynak | Ne | Repoda mı |
|---|---|---|
| Wikimedia Commons, 13 kayıt | gerçek klasik gitar performansları | **evet**, `eval_data/commons/` (CC BY-SA / kamu malı, atıf `SOURCES.tsv`) |
| Ichiyanagi SF3 | örnek kütüphanesi | **evet**, `eval_data/ichiyanagi/` (CC BY-NC 4.0, lisans dosyasıyla) |
| FreePats (görüntüleyici sesi) | örnek kütüphanesi | evet, zaten `scripts/app/soundfont/` (CC0) |
| YouTube, 13 kayıt | gerçek performanslar | hayır, telifli; betik indirir (id listesi betikte) |
| classtab.org, 3968 tab + 2338 MIDI | tab'lar | hayır, betik indirir |
| MG Soft Nylon, RJS LoFi Nylon, MDM, Custom SF2 | örnek kütüphaneleri | hayır, yeniden dağıtım izni yok/belirsiz; betik nereden alınacağını yazar |

## Testler
| Test | Betik | Ne ölçer |
|---|---|---|
| A tek nota | `test_a_single_notes` | her örnek dosyada doğru nota bulundu mu, fazladan nota / oktav hatası |
| B yeniden sentez | `test_b_resynth` | 28 gaps_test parçası 7 gitar sesiyle yeniden çalınır, aynı referansa göre F1 + tel |
| C kontrol | `test_c_sanity` | referanssız kayıtta onset/kroma tutarlılığı, MusicXML'de tel çakışması |
| E/F gerçek kayıt | `test_e_real_vs_tab` | Commons + YouTube kayıtları ↔ classtab tab'ı (README20 yöntemi: LCS hizalama) |
| tel çakışması | `clash_source`, `string_analysis` | aynı tele iki nota nereden geliyor, akor kısıtı ne değiştirir |

## Sonuçlar
**Test B (28 gaps_test, nota F1, onset ±50 ms):** gerçek kayıt 0.783; MG 0.839, Custom 0.837, Ichiyanagi 0.834,
RJS 0.814, MDM 0.766, FreePats 0.751. Model GAPS gitarına aşırı uymamış; başka gitar seslerinde de benzer çalışıyor.
(Sentez kaydın aynısı değil, oda/gövde yok; bu sayılar klasik gitar doğruluğu olarak sunulmaz.)

**Test E+F (23 kayıt):** tab notalarının %91'i bulunuyor. Tel uyumu: **ses modu %56**, edisyon %89 (temiz 19 kayıtta %94).

**Test A:** doğru nota %95–100 bulunuyor. Uzun sönümde fazladan notalar var (±12, ±19, ±24 yarım ses = üst sesler;
%68'i 1 sn'den sonra, 40 tekrar algılama).

**Kusurlar:**
1. **Aynı tele aynı anda iki nota.** 300 GAPS'ten 299'unda var (2439 kopya nota, 5457 farklı perde). Kaynak: <30 ms
   arayla duyulan 329 çift (tel ataması: ses modu her notayı ayrı seçiyor, akor kısıtı yok), 30–250 ms 460 çift
   (ritim kuantalama aynı ızgaraya sıkıştırıyor; bpm 90 sabitle ölçüldü, abartılmış olabilir).
2. **Fazladan üst ses notaları** (hayalet + oktav). 8 parçada bulunan notaların oktav kaymış oranı: gerçek kayıt %3.2,
   custom %4.0, ichiyanagi %4.6, freepats %5.3, RJS %5.7, MG %6.4, MDM %8.2. Parlak tınıda daha sık; ayrı bir
   "oktav kusuru" değil, aynı ailenin parçası.
3. **Ses modu tel %56.** Yanlış tellerin %98'i komşu tel, %91'inde doğru tel modelin 2. tercihi. Tek başına akor kısıtı
   tel doğruluğunu değiştirmiyor (0.619 → 0.620) ama çakışmayı 284 → 7'ye indiriyor.

## Ölçüm hataları (bulundu, düzeltildi)
- **classtab ayrıştırıcısı:** okunamayan tab 170 → **15**. `E-|`/`:` ayırıcıları, etiketsiz bloklar, çok versiyonlu
  dosyalar (yalnız ilk versiyon; Lágrima tab_found 0.43 → 0.91–0.96), tab üstündeki ritim satırının tel sanılması.
  Doğrulama tab'ın kendi MIDI'siyle: yeni okunanlar %97.8 uyumlu, eski blokların %99'u (63080/63696) aynı,
  kötüleşen yok, Test E/F parçaları birebir aynı. Karşılaştırma iki sürümü ayrı modül olarak yükler
  (`compare_parsers`; `reload` ortak durum paylaştırıyordu).
- **MDM 0.45 → 0.766:** iki sebep. Örneklerin başında 0–57 ms sessizlik vardı (notalar geç → ±50 ms dışında);
  `samplers.trim_lead` ile kırpıldı. pYIN 9/37 örneği +12 okumuştu; MDM'de kütüphanenin kendi SFZ eşlemesi
  kullanılıyor (`rerun_mdm`). Test A MDM oktav hatası %30 → %10.8. Eski "MDM'de büyük oktav sorunu" sonucu yanlıştı.

## Sınırlar
- Sentezlenen sesler gerçek kayıt değil: tek şiddet, oda/gövde yok.
- E/F'de tab'da zaman yok; performansçının tekrarları LCS ile hizalanıyor, tel uyumu yalnız eşleşen notalarda.
- YouTube videoları kaldırılabilir; betik o kaydı "FAILED" diye yazar.

## Çalıştırma
```
python -m scripts.data.fetch_gv_data                     # classtab + YouTube + kütüphane listesi
python -m scripts.eval.gv.validate_parser data/raw/gv/classtab
python -m scripts.eval.gv.test_b_resynth docs/results/gv/test_b_trimmed.csv
python -m scripts.eval.gv.test_e_real_vs_tab docs/results/gv/pairs_commons2.tsv out.csv
python -m scripts.eval.gv.test_e_real_vs_tab docs/results/gv/pairs_youtube.tsv out.csv
python -m scripts.eval.gv.string_analysis
```
Son sonuç dosyaları: `test_a.csv`, `test_b_trimmed.csv`, `test_e_commons_v2.csv`, `test_f_youtube_v2.csv`
(`test_b_untrimmed.csv`, `*_commons.csv`, `test_f_youtube.csv` = düzeltme öncesi, karşılaştırma için).

## Sıradaki (kullanıcı onaylı sıra)
2 akor tel çakışması + ritim sıkışması → 3 fazladan üst ses notaları kuralı → 4 ses modu tel %56 (önce el pozisyonu
takibi `w_transition` ölçümü, sonra ayrı bir **hibrit** `--mode`; ses ve edisyon çıktıları birebir aynı kalır)
→ 5 gerekirse tını çeşitlendirmeli eğitim.
