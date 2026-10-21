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

## Adım 2 — TAB'da aynı tele iki nota (21 Ekim 2026)
**Teşhis** (`clash_rhythm`, GAPS partisyonuna göre; kullanıcı girdisi taklidi eval_score B ile aynı). Ses modu
gaps_test 641 çakışma: %68 fazladan nota (192 aynı perde kopyası, 155 aynı anda hayalet), %20 ritim sıkışması,
yalnız %12 gerçek akorda yanlış tel. İlk öneri "aynı konumdaki aynı perdeyi tek yaz" **reddedildi** (kullanıcı:
gerçek tekrarları da siler) — ölçüldü: aynı konuma düşen aynı perde çiftlerinin 34'ü gerçek tekrar, 63'ü sahte.

**A. İki kanıtlı yeniden vuruş kapısı** (`viterbi.segment_notes_onset` `reattack_hi`, `reattack_rise`; 0 = kapalı).
`repeat_gate`: aynı perde çiftlerinde gerçek tekrar ve sahte bölünme sayıca eşit (762 / 754) ve aynı IOI'de; sahteler
yeniden vuruş kapısından (tepe ≥ 0.5) geçiyor ama eşiğin hemen üstünde ve enerjisiz (medyan tepe 0.57, yükseliş 1.9 dB;
gerçek 0.73, 7.9 dB). README12'deki vadi kanıtı bu ikisini ayıramamıştı. Seçim gaps_val'de, aktif seçimin ölçütüyle
(`mix` = (nota F1 + kare F1) / 2; kayıtlı 0.734 birebir üretildi) + hızlı tekrar kaçma en çok +0.01
(`calibrate_reattack`). Tek uygun aday **tepe ≥ 0.6 ya da yükseliş ≥ 2 dB** (kullanıcı seçimi; daha güçlü 0.7 / 4 dB
nota F1 +0.015 verip gaps_val'de tekrarların ~%6'sını kaçırıyordu). gaps_test: nota F1 0.779 → 0.786 (P 0.746 → 0.762,
R 0.814 → 0.813), kare F1 0.701 aynı, hızlı tekrar kaçma 0.476 → 0.478; aynı perde kopyası çakışması 192 → 149.
Yalnız `gaps_val` seçimi değişti (`checkpoints/tabcrnn_deep2.cal.json`); GuitarSet (`val`) kuralları aynı.

**B. Ses modunda akor kısıtı** (`transitions.resolve_chord_strings`, `pipeline.heard_positions`): aynı anda (30 ms,
`order_notes`) başlayan notalar ayrı tellere, emisyon log-olasılıkları toplamı en yüksek atamayla; çakışmasız gruplar
birebir. Tel doğruluğu (`chord_effect`) gaps_val 0.6283 → 0.6287, gaps_test 0.6198 → 0.6216 (nötr); <30 ms çakışmalar
gaps_test 212 → 8. Edisyon modu değişmedi.

**C. Ritim sıkışması** (`rhythm_net.separate_strings`, `pipeline.rhythm`): farklı perdeli iki nota aynı konuma ve aynı
tele düşerse, ritim ağının o konumu daha az olası bulduğu nota komşu bir konuma alınır (önceki geri, sonraki ileri;
sıra bozulmaz), decode'un ölçütüyle (log-olasılık − lam × ızgara karmaşıklığı). "Hep sonrakini kaydır" gaps_val'de
konumu 0.001–0.003 düşürüyordu → reddedildi. `eval_score` (kapalı → açık):

| | gaps_val | gaps_test (rapor) |
|---|---|---|
| ölçü içi konum / tek imli (yarı otomatik) | 0.570 / 0.560 → 0.570 / 0.560 | 0.593 / 0.611 → 0.590 / 0.608 |
| IOI | 0.733 → 0.732 | 0.693 → 0.690 |
| tap ideal konum / IOI | 0.801 / 0.885 → 0.801 / 0.886 | 0.921 / 0.920 → 0.919 / 0.921 |
| tap kaba konum / IOI | 0.694 / 0.733 → 0.696 / 0.744 | 0.805 / 0.819 → 0.803 / 0.823 |
| ölçü başı F, vuruş F, geri okuma, gidiş-dönüş | aynı | aynı |

gaps_test'te yarı otomatik konum −0.003 kalıyor (deterministik, gürültü değil); seçim setinde fark yok.

**Toplam (ses modu TAB çakışması):** gaps_val 465 → **119**, gaps_test 641 → **153**; edisyon 438 → 164. Kalan:
146 aynı perde kopyası (A'nın kapısından geçecek kadar kanıtlı; dokunulmadı), 6 aynı perdeli gerçek tekrar
sıkışması, 1 akor. Testler 228 (yeni: `test_reattack_gate`, `test_chord_strings`, `test_separate_strings`);
çıkarım parmak izi aynı.

**D. Aynı perdeli çiftler de ritim kuralıyla** (kullanıcı kararı: birleştirmek gerçek tekrarı siler, kaydırmak hiçbir
notayı silmez; gerçekse doğru konuma gider, sahteyse kısa fazladan nota olarak görünür). Üç varyant ölçüldü (gidilen
konumun ağdaki olasılığına eşik yok / %10 / %30); `eval_score` gerçek tekrarları sayar (eşleşen), sahteleri saymaz:

| | taban (C) | eşiksiz | %10 | %30 |
|---|---|---|---|---|
| gaps_val konum / IOI (yarı otomatik) | 0.570 / 0.732 | 0.569 / 0.734 | 0.569 / 0.735 | 0.569 / 0.734 |
| gaps_val tap ideal / iyi / kaba konum | 0.801 / 0.740 / 0.696 | 0.802 / 0.742 / 0.697 | 0.802 / 0.742 / 0.698 | 0.802 / 0.742 / 0.698 |
| gaps_test konum / IOI (rapor) | 0.590 / 0.690 | 0.589 / 0.688 | 0.590 / 0.689 | 0.590 / 0.690 |
| gaps_test TAB çakışması ses / edisyon | 153 / 164 | **0 / 0** | 56 / 56 | 94 / 99 |

Seçim (kullanıcı): **eşiksiz** — seçim setinde hiçbir ölçüt düşmüyor, çakışma sıfır. Eşik parametreleri seçimden sonra
koddan çıkarıldı (log'lar `docs/results/gv/*_2d_*.log`).

**Adım 2 sonu (ses modu TAB'da aynı tele iki nota):** gaps_val 465 → 0, gaps_test 641 → 0; edisyon 438 → 0
(gaps_val edisyonda 1). Nota F1 gaps_test 0.779 → 0.786; ritim konumu 0.593 → 0.589, IOI 0.693 → 0.688 (gaps_test;
gaps_val'de 0.570 → 0.569, IOI 0.733 → 0.734).

## Sıradaki (kullanıcı onaylı sıra)
2 akor tel çakışması + ritim sıkışması → 3 fazladan üst ses notaları kuralı → 4 ses modu tel %56 (önce el pozisyonu
takibi `w_transition` ölçümü, sonra ayrı bir **hibrit** `--mode`; ses ve edisyon çıktıları birebir aynı kalır)
→ 5 gerekirse tını çeşitlendirmeli eğitim.
