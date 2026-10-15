# Sorun ve Teknik Borç Envanteri

> Denetim: 14 Ekim 2026 · `Armagan-katman-4` @ `546c7bd` · `origin/katman-5-ritim` @ `3f1f47d`
>
> **Durum kategorileri:** AÇIK · ÇÖZÜLDÜ · KISMEN ÇÖZÜLDÜ · DOĞRULAMA GEREKİYOR · ERTELENDİ
>
> Her kalem, ilk görüldüğü devlog ve ilgili dosya/fonksiyonla ilişkilendirildi. Devlog'da yazılı olup
> sonradan çözülen sorunlar **ÇÖZÜLDÜ** işaretlendi; yalnız eski README'de yazdığı için açık sayılan
> kalem yok.

## Özet tablosu

| Grup | Toplam | AÇIK | KISMEN | DOĞRULAMA GEREKİYOR | ÇÖZÜLDÜ | ERTELENDİ |
|---|---|---|---|---|---|---|
| A — Tel/perde doğruluğu | 5 | 3 | 0 | 2 | 0 | 0 |
| B — Kaçırılan notalar | 5 | 2 | 2 | 1 | 0 | 0 |
| C — Akor fragmanları | 4 | 3 | 1 | 0 | 0 | 0 |
| D — Fiziksel çalınabilirlik | 5 | 5 | 0 | 0 | 0 | 0 |
| E — Entegrasyon | 10 | 8 | 1 | 0 | 1 | 0 |
| F — Viterbi | 2 | 0 | 0 | 0 | 1 | 1 |
| G — Performans / kod kalitesi / süreç | 12 | 9 | 1 | 1 | 0 | 1 |
| **TOPLAM** | **43** | **30** | **5** | **4** | **2** | **2** |

**Açık + kısmen + doğrulama gerekiyor = 39 kalem fiilen açık iş.** Kritik 10 kalem:
`E-01`, `E-04`, `A-01`, `D-01`, `G-05`, `A-04`, `B-04`, `G-03`, `B-02`, `A-02`.

---

## A — Tel / perde doğruluğu

### A-01 · "A telinde ~0,60 doğruluk" iddiasının kaynağı ve geçerliliği
- **İlk görülme:** `docs/devlog/README10.md:30` (Katman 3.9 Adım 0, hata analizi)
- **Dosya/fonksiyon:** `scripts/eval/diagnose_strings.py` bölüm B (oracle notalarda tel karışıklık matrisi)
- **Açıklama:** README10:30 aynen şöyle diyor: *"Solo'da en zayıf: **A teli 0.60** (çoğu D'ye kayıyor), B→G, e→B."* Ölçümün koşulları izlendi ve şunlar tespit edildi:
  - **Model:** `tabcrnn_gaps.pt` (Katman 3.8c) — bugünkü tabandan **iki taban geride** (deep2 değil).
  - **Dataset/split:** GuitarSet oyuncu 05 **solo** = `data/cache/val`. Protokol etiketi: **validation, bağımsız test DEĞİL** (`protocol.py`).
  - **Metrik:** oracle notalarda **tel doğruluğu** (gerçek perde + sınırlar verilir, yalnız tel seçilir) — tab F1 veya nota F1 değil.
  - **Eşik:** `--threshold 0.8` (script varsayılanı; README10 §A "greedy, eşik 0.8").
  - **Aynı koşuda genel tel doğruluğu:** solo 0,797 / polifonik 0,903.
- **Kullanıcıya/modele etkisi:** A telindeki notalar sistematik olarak D teline kayıyor → TAB'da el pozisyonu yanlış görünüyor; perde doğru olduğu için nota F1 etkilenmiyor, **tab F1 ve ürün parmaklandırması** etkileniyor.
- **Önceki çözüm girişimi:** Katman 3.9 Adım 2 (harmonik istifleme) — genel oracle tel doğruluğu 0,797 → **0,850** (README10:127-128). Aynı satırda yalnız **e teli** için önce/sonra verilmiş (0,82 → 0,92). **A teli yeniden ölçülmemiş.**
- **Mevcut durum:** **DOĞRULAMA GEREKİYOR.** Güncel tabanda (`tabcrnn_deep2.pt`, genel oracle tel 0,878) tel başına doğruluk **hiç ölçülmedi**.
- **Kanıt / uyarı:** `CLAUDE.md:117` *"Tel tarafı (gerekirse): A teli 0.66"* diyor. Bu 0,66 sayısının **hiçbir devlog'da kaynağı bulunamadı** (tüm `docs/` ve kök belgeler "0.66"/"0,66" için tarandı; bulunan tüm eşleşmeler tab F1 ya da polifoni kare F1 bağlamında, tel başına doğruluk değil). İki sayı da (0,60 ve 0,66) güncel tabanı temsil etmiyor.
- **Olası kök neden:** Komşu tel karışıklığı. Ölçülmüş: tel hatalarının **%98-99'u komşu tel** (aynı perde, 5 fret öte; G-B arası 4). Hataların %53'ünde model kola doğru daha yüksek perdeyi seçiyor. Güven analizi (README10 §D): solo'da yanlış seçimlerin medyan marjı 0,09 (doğrularda 0,25), yalnız %12'si emin → **kararsızlık**, yani model komşu telleri ayıracak tını farkını (inharmonisite, harmonik yapı) yeterince görmüyor. Sistematik önyargı (akor kalıbı) değil.
- **Önerilen çözüm:** Önce ölç (çözüm seçmeden). `diagnose_strings` güncel tabanla koşulur, tel karışıklık matrisi ve tel başına doğruluk çıkarılır. Sonra karar: (a) ölçüm 0,60'a yakınsa daha ince frekans çözünürlüğü (36 bin/oktav) veya hex öğretmen damıtması — ikisi de `CLAUDE.md:117-118`'de aday olarak kayıtlı; (b) 0,85+ ise sorun kapanmış sayılır ve `CLAUDE.md` düzeltilir.
- **Öncelik: KRİTİK** (ölçüm maliyeti düşük, karar etkisi yüksek).
- **Doğrulama yöntemi:**
  ```bash
  python -m scripts.eval.diagnose_strings --ckpt tabcrnn_deep2.pt --splits val val_comp
  ```
  Beklenen çıktı: bölüm B tel karışıklık matrisi + tel başına doğruluk. **Not:** script'in varsayılan eşiği 0,8; README10 ile kıyaslanabilirlik için aynı eşik kullanılmalı, ayrıca kalibre eşikle (0,7 / GuitarSet) ikinci koşu alınmalı. Dikkat: `katman-5-ritim` bu script'i `archive/scripts/eval/` altına taşıdı (`A-03`).

### A-02 · GAPS'te ses tabanlı tel ataması naif tabandan kötü
- **İlk görülme:** `docs/devlog/README14.md` (Katman 3.13), `CLAUDE.md:30-33`
- **Dosya:** `scripts/eval/eval_gaps_tab.py`, `gtab/decoding/transitions.py` (kafes ataması)
- **Açıklama:** `tabcrnn_deep2.pt`'ın GAPS'te oracle tel doğruluğu **0,610**, "en düşük perde" naif kuralı **0,702**; aynı model GuitarSet'te 0,878. Yani klasik gitarda ses tabanlı tel tahmini naif kuralı **geçemiyor**.
- **Etki:** GAPS (klasik gitar = ana hedef) için "duyulan konum" modu güvenilmez; ürün pratikte edisyon modeline (`edition_seq.pt`, tel 0,815) yönleniyor.
- **Önceki girişim:** `train_onset --gaps-tab-weight 1` → `tabcrnn_gtab.pt` → **RET** (hiçbir epoch ep 0'ı geçmedi; gaps_val tab F1 +0,004 ama GS tab −0,035). Sonuç kaydı: *"partisyon TAB'ı icracıya genellenmiyor, ses kaybı olarak kullanma"* (`CLAUDE.md:44-47`).
- **Mevcut durum:** **AÇIK** (bilinen, nedeni ölçülmüş, çözümsüz).
- **Kanıt:** `docs/baselines/d1_baseline.json` → `gaps_audio_oracle_string_acc` 0,610 (gaps_val), `gaps_naive_string_acc` 0,745, `gs_oracle_string_acc` 0,878.
- **Olası kök neden:** İki ayrı neden karışıyor: (a) alan farkı — model çelik telde eğitildi, naylonun tel tınısı farklı; (b) **etiket farkı** — GAPS'teki "tel", editörün partisyonda yazdığı TAB konumu; icracının fiilen çaldığı tel olmak zorunda değil. İkinci neden ölçülemez bir tavan koyuyor.
- **Önerilen çözüm:** Kısa vadede yok; ürün edisyon modunu kullanıyor ve bu kullanıcı kararıyla belgelendi. Uzun vadede `A-01`'in çözümüyle (ince çözünürlük / hex damıtma) birlikte ele alınmalı.
- **Öncelik: ORTA** (ürün etkisi azaltılmış durumda).
- **Doğrulama:** `python -m scripts.eval.eval_gaps_tab` (gaps_val; `gaps_test` kapanışa saklı).

### A-03 · `diagnose_strings.py` katman-5'te arşive taşındı
- **İlk görülme:** bu denetim · commit `37cbf39` ("Katman 3.13 temizlik")
- **Dosya:** `scripts/eval/diagnose_strings.py` → `archive/scripts/eval/diagnose_strings.py`
- **Açıklama:** `A-01`'in doğrulama aracı katman-5 dalında arşive alındı. Birleştirme sırasında bu dosya arşive giderse, tel tanılaması "arşiv kodu" sayılıp kullanılmaz hâle gelir.
- **Etki:** `A-01` kapatılmadan arşive gönderilmesi, açık bir sorunun ölçüm aracını kaybetmek anlamına gelir.
- **Mevcut durum:** **AÇIK** (yalnız katman-5 tarafında).
- **Önerilen çözüm:** `A-01` ölçümü **birleştirmeden önce** mevcut dalda koşulsun; sonuç ne olursa olsun devlog'a yazılsın. Sonra arşivleme kararı verilir.
- **Öncelik: YÜKSEK** (sıralama kararı; maliyeti sıfır).
- **Doğrulama:** `git ls-tree -r --name-only origin/katman-5-ritim | grep diagnose_strings`

### A-04 · GuitarSet'te bağımsız test seti yok
- **İlk görülme:** `docs/devlog/README15.md` "Neden bu faz" 1 (Katman 4 Faz 0, kendi tespiti)
- **Dosya:** `gtab/evaluation/protocol.py` (split rolleri), `tests/test_protocol.py`
- **Açıklama:** GuitarSet'te 6 oyuncudan 5'i eğitimde, oyuncu 05 **aynı anda** (a) `train_onset` seçim skorunun bileşeni, (b) eşik/çözümleme kalibrasyonunun yapıldığı set, (c) manşet sayıların (tab F1 0,777, nota F1 0,924) raporlandığı set. Bu sayılar **validation**'dır.
- **Etki:** Tez savunmasında "bu test sonucu mu?" sorusu. GuitarSet tarafında bağımsız genelleme kanıtı yok; bağımsız kanıt yalnız `gaps_test` (icracı-ayrık, 30 kayıt, 27 icracı).
- **Önceki girişim:** Faz 0 bunu tespit etti, `report_label()` ile etiketlemeyi zorunlu kıldı, `val_half_a/b` bölmelerini **keşif** olarak işaretledi (`may_select = may_report = False`) — yani "bir yarıda seç, diğerinde test et" yolu protokolce kapalı.
- **Mevcut durum:** **AÇIK** (yapısal; Faz 0 "bu faz onu yaratamaz" diyor).
- **Kanıt:** `check_protocol` çıktısı: *"Oyuncu 05 kesif yarilari parca-ayrik: yari A 14 / yari B 16 → KESIF; bagimsiz test DEGIL"* (denetimde koşuldu, geçti).
- **Kök neden:** Veri bölmesi kararı Katman 1'de verildi ve 14 katman boyunca tüm checkpoint'ler ona göre eğitildi.
- **Önerilen çözüm:** İki seçenek — (a) bir oyuncuyu **her şeyden** çıkararak yeniden eğitim (maliyetli: tüm zincir yeniden); (b) **tezde dürüst etiketleme** + GAPS'i ana bağımsız test olarak sunma. (b) bugün fiilen uygulanıyor; eksik olan kök `README.md`'de etiketin görünmemesi (`G-05`).
- **Öncelik: YÜKSEK** (b seçeneğinin dokümantasyonu), DÜŞÜK (a seçeneği).
- **Doğrulama:** `python -m scripts.eval.check_protocol`

### A-05 · Tel hatalarının perde bölgesine göre dağılımı güncellenmedi
- **İlk görülme:** `docs/devlog/README10.md:32` ("Perde bölgesine göre: 5–9 en iyi (0.83 / 0.94); 1–4 ve 10–14 zayıf")
- **Dosya:** `scripts/eval/diagnose_strings.py` bölüm B
- **Açıklama:** Perde bölgesi kırılımı `tabcrnn_gaps.pt` ile ölçülmüş, güncel tabanla yenilenmemiş. 1-4 perde bölgesi zayıflığı klasik gitarda kritik (repertuvarın büyük kısmı ilk pozisyonda).
- **Etki:** `A-01`'in çözüm seçiminde yön verecek bilgi eksik.
- **Mevcut durum:** **DOĞRULAMA GEREKİYOR** (`A-01` ile aynı koşuda çıkar).
- **Öncelik: ORTA** (A-01 ile birlikte, ek maliyet yok).

---

## B — Kaçırılan notalar

### B-01 · Kaçanların ~%36-45'inde "perde hiç aktif değil"
- **İlk görülme:** `docs/devlog/README11.md:23` (Katman 3.10)
- **Dosya/fonksiyon:** `scripts/eval/diagnose_pitch.py:134-135` (bölüm A kategorilendirmesi)
- **Açıklama ve TANIM (önemli):** Kategori adı yanıltıcı olabilir. Kodda tanım şu: gerçek notanın zaman aralığı `[a:b)` boyunca, modelin o perde için ürettiği olasılık **hiçbir karede eşiği geçmiyor** (`col.max() <= thr` → `"perde hic aktif degil"`). Yani bu, **modelin çıkış duyarlılığına** dair bir ifadedir — "model bu notayı hiç duymuyor". **Eğitim verisinde o perdenin bulunup bulunmadığıyla ilgisi yoktur.** (Bkz. `B-04`.)
- **Ölçülen değerler (zaman içinde):**

  | Taban | gaps_val | val_comp | Kaynak |
  |---|---|---|---|
  | 3.9 sonu | %44,9 | %38,3 | README11:23 |
  | `--gs-pitch-weight` sonrası | %36,3 | %38,7 | README11:135 |
  | GAPS perde ağırlığı ×2 sonrası | adet 4180 → 3134 (−%25) | — | README11:515 |
  | deep2 öncesi (3.12 Adım 0) | %40,2 (gaps_val) / %35,7 (gaps_fit) | — | README13:48 |
- **Etki:** Kaçan notaların en büyük tek kategorisi; çözümlemeyle (eşik/kural) düzeltilemez çünkü sinyal modelin çıkışında yok (README11:421 bunu açıkça yazıyor).
- **Önceki girişimler:** (a) GuitarSet'e doğrudan noisy-OR perde kaybı (`--gs-pitch-weight`) → %44,9 → %36,3; (b) GAPS perde ağırlığı ×2 (`--pitch-weight 2`) → adet −%25; (c) derin CNN + perde-onset kafası (deep2) → GAPS nota F1 0,720 → **0,779**.
- **Mevcut durum:** **KISMEN ÇÖZÜLDÜ.** Üç müdahale kategoriyi küçülttü ama kapatmadı; `CLAUDE.md:34` güncel açıkları sayarken bunu hâlâ listeliyor.
- **Kök neden (ölçülmüş):** Kapasite sınırı. `overfit_gaps.py` aşırı öğrenme testi: 8 kayıt, 200 epoch, dropout kapalı → GAPS onset 0,40 → **0,95**, GuitarSet kontrolü 0,995 → **etiket tavanı yok, darboğaz kapasite** (`CLAUDE.md:97-100`). Derin modele geçiş bu kanıta dayanıyordu ve işe yaradı (+0,059 nota F1).
- **Önerilen çözüm:** deep2 sonrası kategori yeniden ölçülmeli (son ölçüm deep2 ÖNCESİ). Ölçüm düşmüşse sorun kapanmaya yakın; düşmemişse sırada ya kapasite tekrar (3b) ya veri (SynthTab, `G-03`).
- **Öncelik: YÜKSEK.**
- **Doğrulama:**
  ```bash
  python -m scripts.eval.diagnose_pitch --ckpt tabcrnn_deep2.pt --load-cal checkpoints/tabcrnn_deep2.cal.json --splits gaps_val val_comp
  ```
  Bölüm A'daki "perde hiç aktif değil" oranı okunur ve yukarıdaki tabloya satır eklenir.

### B-02 · <100 ms aynı perde tekrarları GAPS'te kaçıyor
- **İlk görülme:** `docs/devlog/README11.md` (Katman 3.10), ayrıntı README12 (Katman 3.11)
- **Dosya:** `gtab/decoding/viterbi.py:243 pick_peaks`, `:269 segment_notes_onset`; `gtab/data/tab_labels.py:131 repeat_onset_weights`, `:158 valley_frames`
- **Açıklama:** Önceki nota sönmeden gelen aynı perde vuruşu (klasik tremolo: p-a-m-i, 60-100 ms IOI) spektrumda yalnız küçük bir sıçrama bırakıyor; model/çözümleme onu yeni nota saymıyor.
- **Ölçülen gelişim:**

  | Adım | GAPS <100 ms kaçma | val_comp <100 ms kaçma | Kaynak |
  |---|---|---|---|
  | 3.10 sonu | %75,0 (n=647) | %72,5 (n=80) | SOZLUK §5b |
  | 3.11 Adım 1 (yerel tepe) | kötüleşti (kapılar adayları eliyor) | — | `CLAUDE.md:85-87` |
  | 3.11 Adım 1b (vadi kanıtı) | değişmedi | — | `CLAUDE.md:89-91` |
  | 3.11 Adım 4 (vadi eğitimi) | değişmedi (~%73-78) | **%72,5 → %22,5** | README12 §8f |
  | 3.12 deep2 kalibrasyonu | ~%76 (değişmedi) · gaps_val 73 → **55** | — | `CLAUDE.md:107,112` |
- **Etki:** Tremolo pasajlarında notaların çoğu tek nota olarak birleşiyor → TAB'da ritim ve nota sayısı yanlış. Klasik gitar repertuvarında yaygın bir teknik.
- **Önceki girişimler:** 4 ayrı yol denendi. Kritik negatif bulgu: *"vadi kanıtı precision'ı çökertiyor (fragman vadisi = gerçek tekrar vadisi)"* → **eğitimsiz yol GAPS için tükendi** (`CLAUDE.md:89-91`). Ayrıca kısa pencere / hop 256 ölçüldü ve ayrım artırmadı (AUC ~0,89).
- **Mevcut durum:** **KISMEN ÇÖZÜLDÜ** — val_comp'ta büyük kazanç (%72,5 → %22,5), **GAPS'te açık** (~%55-76).
- **Kanıt:** `CLAUDE.md:34` *"Açıklar: GAPS <100 ms tekrarlar"*; görünürlük ölçümü %66,6 (3.11 Adım 0) → sorun yalnız çözümlemede değil, girişte de kısmen.
- **Kök neden:** Aynı perdenin iki vuruşu arasındaki "vadi", fragman (tek notanın yanlış bölünmesi) vadisinden ayırt edilemiyor. Bu bir **bilgi** sınırı: 23 ms kare çözünürlüğünde 60-100 ms IOI = 3-4 kare.
- **Önerilen çözüm:** İki aday — (a) ritim katmanıyla birlikte ele almak (katman-5 1q kuantalayıcı bağlam görüyor; ama kuantalayıcı **bulunmayan** notayı üretmez → bkz. ROADMAP_NEXT §3); (b) ayrı bir "tekrar sayısı" kafası (bir aralıkta kaç vuruş var) — bugüne kadar denenmedi.
- **Öncelik: YÜKSEK** (ürün kalitesinde doğrudan görünür).
- **Doğrulama:** `eval_pitch` tekrar kaçma sütunu (`gaps_val`); `diagnose_pitch` bölüm E.

### B-03 · Nota kapsamı %81,7 — uçtan uca her şeyin tavanı
- **İlk görülme:** `docs/devlog/README16.md §5` (P1), `d1_baseline.json` → `gaps_note_coverage` 0,817
- **Dosya:** `scripts/eval/eval_edition.py:205-240` (uçtan uca eşleme)
- **Açıklama:** `tabcrnn_deep2.pt`, gaps_test'te partisyonda etiketli notaların **%81,7'sini** buluyor. Geri kalan %18,3 hiç bulunmadığı için tel/perde/parmak/ritim zincirinin tamamı o notalar için üretilemiyor.
- **Etki:** Uçtan uca sayıların tavanı. Örnek: parmak doğruluğu bulunanlarda 0,397, tüm etiketli notalarda 0,331 — fark tamamen kapsamdan. Ritimde de aynı: "gerçek vuruş + deep2 notası" satırı kapsama 0,810.
- **Mevcut durum:** **AÇIK** (ölçülmüş, bilinen tavan).
- **Kök neden:** `B-01` + `B-02` + `C-01`'in toplamı. Ayrı bir sorun değil, bileşke gösterge.
- **Önerilen çözüm:** B-01/B-02/C-01 kapandıkça düzelir. Bağımsız iş yok; **ama tezde her uçtan uca sayının yanında kapsamın yazılması** gerekiyor (bugün `eval_edition` bunu yapıyor, `d1_baseline` bağlam satırı olarak tutuyor).
- **Öncelik: ORTA** (raporlama disiplini).
- **Doğrulama:** `python -m scripts.eval.eval_edition` (varsayılanlar; `gaps_test` kapanışta).

### B-04 · "Kaçanların %36'sı eğitimde hiç görülmeyen perdelerden" iddiası — KAVRAM HATASI
- **İlk görülme:** bu denetim (görev tanımındaki iddia)
- **Dosya:** `scripts/eval/diagnose_pitch.py:134-135`
- **Açıklama:** Projede ölçülmüş kategori **"perde hiç aktif değil"**dir ve tanımı `B-01`'de verildi: *modelin çıkış olasılığı o nota boyunca eşiği hiç geçmiyor.* Bu, **eğitim verisi kapsamı** hakkında bir şey söylemez. "Eğitimde hiç görülmeyen perde/fret" ise tamamen **farklı, henüz ölçülmemiş** bir istatistiktir (eğitim split'lerindeki perde/fret histogramı ile kaçan notaların perde/fret dağılımının kesişimi).
- **Etki:** Bu iki ifadenin karıştırılması, SynthTab deneyinin gerekçesini yanlış kurar. Mevcut kanıt "model duyarlılığı" yönünü destekliyor (ve kapasite ölçümü bunu doğruladı); "eğitim kapsamı" yönü **hiç test edilmedi**.
- **Mevcut durum:** **AÇIK — ölçülmedi.** İddia bugünkü hâliyle kaynaksız.
- **Kanıt:** `grep -rn "perde hiç aktif değil" docs/` tüm eşleşmeler model çıkışı bağlamında; `grep -rni "egitimde|görülmemiş"` ile eğitim kapsamı ölçümüne dair **hiçbir kayıt bulunamadı**.
- **Olası kök neden:** `diagnose_pitch`'in Türkçe kategori adı ("model duymuyor") ile "eğitimde görmedi" arasındaki kavramsal kayma.
- **Önerilen çözüm:** Ucuz ve bağımsız bir ölçüm yazılabilir (eğitim yok): eğitim split'lerinin (perde × tel/fret) kapsam histogramı → kaçan GAPS notalarının aynı uzaydaki dağılımı → "eğitimde hiç/az görülmüş hücrelerden gelen kaçan oranı". Bu ölçüm, SynthTab araştırma sorusunun **ön koşulu** olur: sayı yüksekse sentetik veri hipotezi güçlenir, düşükse hipotez daha ölçüm yapılmadan zayıflar. Ayrıntı: ROADMAP_NEXT §6 Adım 0.
- **Öncelik: KRİTİK** (tezin ana araştırma sorusunun gerekçesi buna bağlı; maliyet: bir ölçüm script'i, eğitim yok).
- **Doğrulama yöntemi:** yeni script `scripts/eval/coverage_gap.py` (yazılmadı) — `gaps_fit` + GuitarSet train etiket histogramı vs `diagnose_pitch` kaçan listesi; `assert_may_select` kapısından geçmeli, `gaps_val`'de koşmalı.

### B-05 · Onset/offset zamanlama hataları ayrı kategori olarak izlenmiyor
- **İlk görülme:** `docs/devlog/README11.md:12-13` (diagnose_pitch kategorileri)
- **Dosya:** `scripts/eval/diagnose_pitch.py:133` (`"nota var, onset zamani kaymis"`)
- **Açıklama:** Kategori var ve ölçülüyor ama zaman içindeki seyri devlog'larda tablo hâlinde izlenmemiş. GAPS'te offset eşiği denendi ve **seçilmedi**: *"etiket bitişleri partisyon hizalı"* (`CLAUDE.md:122`) — yani GAPS'te offset ölçmek etiket gürültüsüne karşı anlamsız.
- **Etki:** Düşük. Onset kayması nota F1'i 50 ms toleransla etkiliyor; ritim katmanı için önemli (ölçü içi konum 0,611).
- **Mevcut durum:** **DOĞRULAMA GEREKİYOR** (kategori ölçülüyor, seyri raporlanmıyor).
- **Öncelik: DÜŞÜK.**
- **Doğrulama:** `diagnose_pitch` bölüm A ve C (kare kaybı notanın neresinde).

---

## C — Akor fragmanları ve polifoni

### C-01 · Akor fragmanları açık
- **İlk görülme:** `docs/devlog/README11.md` (Katman 3.10 Adım 3a, hayalet alt kırılımı)
- **Dosya:** `scripts/eval/diagnose_pitch.py` bölüm D (fragman tanımı: aynı perdede zaten bulunmuş gerçek notanın içinde ikinci parça); `gtab/data/tab_labels.py:180 fragment_frames`
- **Açıklama:** Tek gerçek nota modelin çıkışında ikiye bölünüyor (fragman) veya akorun yalnız bir kısmı tespit ediliyor. `CLAUDE.md:34` güncel açıkları sayarken "akor fragmanları"nı listeliyor.
- **Etki:** Yanlış pozitif üretir (hayalet nota) ve akor eksik görünür → TAB'da olmayan nota veya eksik akor.
- **Önceki girişimler:** Fragman ağırlığı eğitimde denendi (`train_onset --frag-weight 3`, 3.11 Adım 4) — tab F1 0,769 ile en iyi sonucu verdi ama GAPS nota F1 −0,013 (kabul kuralı sağlanmadı, taban yine kabul edildi).
- **Mevcut durum:** **AÇIK.**
- **Kanıt:** `CLAUDE.md:34`; README11:421 *"hayaletlerin %53'ü farklı perde"*.
- **Kök neden:** Fragman vadisi ile gerçek tekrar vadisi ayırt edilemiyor (`B-02` ile aynı kök — bu ikisi **aynı madalyonun iki yüzü**: bir vadiyi tekrar sayarsan fragmanı da nota sayarsın).
- **Önerilen çözüm:** ROADMAP_NEXT §3'te düşük maliyetli başlangıç deneyi tasarlandı: fragman/tekrar ayrımını **polifoni bağlamıyla** koşullandırmak (akor içinde fragman olasılığı farklı). Önce ölçüm: fragman oranının polifoni seviyesine göre kırılımı.
- **Öncelik: YÜKSEK.**
- **Doğrulama:** `diagnose_pitch` bölüm D (fragman / kayık başlangıç / uzama / farklı perde kırılımı), polifoni kırılımıyla.

### C-02 · Eğitimsiz çözümleme yolu GAPS için tükendi
- **İlk görülme:** `docs/devlog/README12.md §8c-8d` (Katman 3.11 Adım 1b sonucu)
- **Dosya:** `gtab/decoding/viterbi.py` (kural kümesi), `gtab/evaluation/pitch_eval.py:280 calibrate_repeats`
- **Açıklama:** Vadi kanıtlı kapılar + ortak arama + tekrar odaklı aday seçimi kodlandı ve ölçüldü: GAPS 0,733/0,676, **tekrarlar değişmedi** → *"vadi kanıtı precision'ı çökertiyor"*. Kural tabanlı (eğitimsiz) iyileştirme GAPS'te sınırına ulaştı.
- **Etki:** `B-02` ve `C-01` için ucuz çözüm kalmadı; her ileri adım eğitim gerektiriyor.
- **Mevcut durum:** **AÇIK** (bilgi niteliğinde; "bu yolu tekrar denemeyin" kaydı).
- **Kanıt:** `CLAUDE.md:89-91`; README12 §8d.
- **Öncelik: BİLGİ** (yeni iş değil; plan yaparken dikkate alınmalı).

### C-03 · 30 ms akor gruplaması zamanda örtüşmeyen notaları akor sayıyor
- **İlk görülme:** `docs/devlog/README18.md` K7 ve §6.4 (P3 son doğrulaması)
- **Dosya:** `gtab/decoding/edition.py:29 order_notes` (`CHORD_TOL`, 30 ms)
- **Açıklama:** `order_notes` akorları yalnız **başlangıç** yakınlığına göre grupluyor; başlangıcı yakın iki nota aynı anda basılı olmak zorunda değil. Ölçüldü: bu çiftler basılı çiftlerin **%0,27-2,27'si**; etkisi editörde en net — grup tanımında maksimum açıklık 5, örtüşen tanımda **3** (yani "5 yarım ses açıklık" diye görünen akor aslında ardışık çalınan iki notaydı).
- **Etki:** Çalınabilirlik metriklerinde yanlış ihlal; akor içi tel teklik kısıtının gereksiz tetiklenmesi.
- **Önceki girişim:** P3'te **ölçüm tarafında** çözüldü: A/B/C/E yalnız zamanda örtüşen çiftlerde hesaplanıyor; eski tanım karşılaştırma için `group_span()` / `group_finger_conflicts()` olarak korundu; 6 yeni test.
- **Mevcut durum:** **KISMEN ÇÖZÜLDÜ** — ölçüm düzeltildi, **`order_notes` gruplama kodu değişmedi** (bilinçli; model davranışını kırmamak için). Yani `EditionNet`'in gördüğü akor tanımı hâlâ yalnız onset'e dayanıyor.
- **Kanıt:** README18 §5.4 tablosu; `tests/test_playability.py` (41 test).
- **Önerilen çözüm:** P4'te değerlendirilsin: tel teklik ve parmak kısıtı yalnız **örtüşen** notalara uygulanırsa `D-02`'deki saçma konumlar da azalır (bkz. README18 §5.5 — tel teklik kısıtı perde 24 gibi konumlar üretiyor).
- **Öncelik: YÜKSEK** (P4'ün parçası).
- **Doğrulama:** `python -m scripts.eval.eval_playability` örtüşme tanılaması bölümü.

### C-04 · 6'dan fazla notalı gruplarda `resolve_chords` argmax'a düşüyor
- **İlk görülme:** `docs/devlog/README18.md §5.5` "Ek bulgu"
- **Dosya:** `gtab/models/edition_net.py:79 resolve_chords`
- **Açıklama:** 12.842 grubun 3'ü 6'dan fazla nota içeriyor (30 ms içine sıkışan arpejler); 6 tel varken 7+ notaya ayrı tel verilemiyor → argmax'a düşüyor. Sonuç: **3 grupta aynı tel iki notaya verildi (%0,02)**.
- **Etki:** Ölçülmüş ve **zararsız bulundu**: bu çiftlerin hiçbiri zamanda örtüşmüyor, yani fiziksel olarak imkânsız bir durum üretmiyor.
- **Mevcut durum:** **AÇIK ama düşük etkili** (ölçülmüş, belgelenmiş).
- **Öncelik: DÜŞÜK.**
- **Doğrulama:** `eval_playability` çıktısında aynı-tel çifti sayımı.

---

## D — Fiziksel çalınabilirlik

### D-01 · Akor içi aynı parmak yasağı yok → fiziksel olarak imkânsız akorlar
- **İlk görülme:** `docs/devlog/README16.md §6.3` (P1), ölçümü `README18.md §5` (P3)
- **Dosya/satır:** `gtab/models/edition_net.py:114` — parmaklar akor düzeyinde değil **nota başına argmax** ile seçiliyor; tellerde var olan çakışma çözümünün (`resolve_chords`, satır 79) parmak karşılığı yok.
- **Açıklama:** Aynı parmak, zamanda örtüşen iki notada **farklı perdelere** atanabiliyor — fiziksel olarak imkânsız.
- **Ölçülen etki (gaps_val, gerçek kesişim 2.963 akor):**

  | Yöntem | çakışan çift | çakışmalı akor oranı (PAYDA-3) |
  |---|---|---|
  | model (EditionNet) | **22** | **%8,52** |
  | naif (en düşük perde + taban parmak) | 25 | %13,44 |
  | **editör (partisyon)** | **0** | **%0,00** |
- **Etki:** Ürün, çalınamayan parmaklandırma öneriyor. Sunum slayt 11'in "fiziksel kuralları ihlal etmeyen öneri" iddiası karşılıksız.
- **Önceki girişim:** Yok — P3 yalnız ölçüm fazıydı (K1: "hüküm değil, sayım").
- **Mevcut durum:** **AÇIK — P4'ün ana hedefi.**
- **Kanıt:** README18 §5.1 ana tablo, §5.3 madde 2.
- **Kök neden:** Parmak kafası nota başına bağımsız karar veriyor; akor düzeyi kısıt çözümü yalnız **tel** için yazılmış.
- **Önerilen çözüm:** `resolve_chords` mantığının parmağa taşınması: örtüşen notalarda bir parmak yalnız bir perdede (aynı perdede çok tel = barre serbest). **Kabul ölçütü (P3'ün önerisi):** kesişimde A = 22 → **0** ve `gaps_val` parmak doğruluğu (0,675) **düşmemeli**.
- **Öncelik: KRİTİK** (P4'ün tanımı).
- **Doğrulama:** `python -m scripts.eval.eval_playability` + `python -m scripts.eval.eval_edition` (parmak doğruluğu regresyonu) + `tests/test_playability.py`.

### D-02 · El açıklığı kısıtı yok → 22 yarım sese kadar açıklık
- **İlk görülme:** `README16.md §6.3` (P1), ölçümü `README18.md §5.2, §5.5`
- **Dosya:** `gtab/decoding/playability.py:153 hand_span` (ölçüyor), kısıt tarafı yok
- **Açıklama:** Model kendi kapsamında **maksimum 22 yarım ses** açıklık üretiyor; editörde maksimum **3**, ≥4 yarım ses oranı editörde **%0,00**, modelde %3,59, naifte %27,42.
- **Doğrulama yapıldı:** Dört aşırı örnek ham notalarla denetlendi, **hepsi gerçek model çıktısı** (değerlendirme hatası değil). Örnek (`275_Lw1wc`, 123,582 sn): model E4'ü (MIDI 64) 5. telde açık çalmak yerine **0. telin 24. perdesine** atamış; 69 zaten 5. teli kullandığı için **akor içi tel teklik kısıtı** 64'ü en pes tele itmiş.
- **Etki:** Tek bir saçma konum tüm akoru çalınamaz yapıyor.
- **Mevcut durum:** **AÇIK.**
- **Kök neden (ölçülmüş):** *"tel teklik kısıtı + açıklık kısıtının yokluğu"* birleşimi (README18 §5.5). Yani kısıt eklemenin sırası önemli: tel teklik tek başına zarar veriyor.
- **Önerilen çözüm:** Yumuşak açıklık cezası; üst sınır referansı **editör dağılımı** (maks 3, ≥4 oranı %0). P3'ün P4 kabul önerisi: B ≥4 oranı ≤ %1.
- **Öncelik: KRİTİK** (D-01 ile birlikte çözülmeli; ayrı çözülürse biri diğerini bozar).
- **Doğrulama:** `python -m scripts.eval.eval_playability --verify-span 12`

### D-03 · Barre belirsizlik kuralı fazla temkinli
- **İlk görülme:** `docs/devlog/README18.md §5.3` madde 7, §7.3
- **Dosya:** `gtab/decoding/playability.py:219 barre_detail`
- **Açıklama:** Editörün 15 barre adayının **12'si "belirsiz"** çıkıyor (kural: teller bitişik değil → belirsiz). Kısmi barre klasik icrada yaygın.
- **Etki:** Ölçüm barre'yi olduğundan sorunlu gösteriyor; kural kalibrasyonu eksik.
- **Mevcut durum:** **AÇIK** (P3 kendi önerisinde "kural gevşetilmeli" diyor).
- **Kanıt:** README18 §5.1 E satırı: editör 3 temiz / 12 belirsiz.
- **Öncelik: ORTA.**
- **Doğrulama:** `eval_playability` E bölümü; editörde belirsiz oranı düşmeli, çakışma 0 kalmalı.

### D-04 · El pozisyonu çıktı olarak üretilmiyor
- **İlk görülme:** `README16.md §4` (P1: "El pozisyonu çıktı olarak üretilmiyor")
- **Dosya:** `gtab/decoding/playability.py:183 hand_position` (hesaplıyor, ölçüm için); `gtab/core/note_event.py` (alan yok)
- **Açıklama:** `hand_position` (medyan `perde − parmak + 1`) ölçüm modülünde hazır ama ürün çıktısına girmiyor; `NoteEvent`'te alan yok. Reddedilen log-lineer modelin iç özelliğiydi (`edition.py:16,118`).
- **Etki:** Sunum slayt 11'in "önerilen sol-el parmağı **ve el pozisyonu**" çıktısı eksik.
- **Mevcut durum:** **AÇIK** (P4/P6 planında: README18 §8.4 "`NoteEvent`'e taşınması P6").
- **Öncelik: ORTA** (`E-02` ile aynı değişiklikte çözülür).

### D-05 · PAYDA-3 küçük → çalınabilirlik oranlarının güven aralığı geniş
- **İlk görülme:** `docs/devlog/README18.md §7.1`
- **Açıklama:** Gerçek kesişimde yalnız **223-232 akor** örtüşen basılı çift içeriyor (GAPS repertuvarı ağırlıkla tek sesli). %8,52 gibi oranlar 223 akor üzerinden.
- **Ek sınırlama (§7.5):** Editörün parmağı akorların yalnız **%26,6'sında** tam; modelin en kötü şekilleri editörün etiketlemediği bölgede yoğunlaşıyor → **kesişim sayıları modelin iyimser tarafı** (kendi kapsamında açıklık maksimumu 22, kesişimde 6).
- **Etki:** P4'ün "A = 0" hedefi 223 akor üzerinde ölçülecek; istatistiksel güç düşük.
- **Mevcut durum:** **AÇIK** (yapısal veri sınırlaması).
- **Önerilen çözüm:** P4 kabul ölçütü iki paydada birlikte raporlanmalı (kesişim + kendi kapsamı); yalnız kesişime bakmak iyimser sonuç verir.
- **Öncelik: ORTA** (raporlama disiplini).

---

## E — Entegrasyon

### E-01 · İKİ RAKİP WAV→TAB GİRİŞ NOKTASI
- **İlk görülme:** bu denetim
- **Dosyalar:**
  - `scripts/transcribe.py` — 221 satır, **takipsiz** (git'te yok), yalnız bu makinede
  - `origin/katman-5-ritim`: `gtab/pipeline.py` (81) + `scripts/app/transcribe.py` (50)
- **Açıklama:** İki sürüm aynı zinciri (CQT → deep2 → çözümleme → tel/perde) bağımsız olarak kuruyor. Farklar:

  | | `scripts/transcribe.py` | `gtab/pipeline.py` |
  |---|---|---|
  | Çıktı | ASCII TAB + nota tablosu | MusicXML (ölçülü) |
  | Ritim | yok | BeatNet + DBN + 1q kuantalayıcı |
  | Parmak | `--fingering`, konum **sabit** (`fixed`); modelin serbest tercihi yalnız **bilgi** olarak sayılıyor | `--mode edisyon`, konumu da EditionNet seçer |
  | Alan seçimi | `--domain gaps_val` / `val`, çıktıda yazılıyor | `domain="gaps_val"` (CLI'da yok) |
  | Hata yönetimi | `TranscribeError`: bozuk sesde **sahte çıktı üretmiyor, duruyor** | yok |
  | Test | yok | yok |
- **Etki:** "Ürün çıktısı hangisi?" sorusu cevapsız. İkisi ayrı bakım gerektirir; birleştirmede kod çakışması.
- **Mevcut durum:** **AÇIK — KRİTİK.**
- **Kanıt:** `git status` (takipsiz dosya); `git ls-tree origin/katman-5-ritim`.
- **Önerilen çözüm:** `gtab/pipeline.py` tek omurga; `transcribe.py`'ın üç özgün değeri ona taşınır: (1) `--domain` CLI bayrağı + çıktıda alan yazımı, (2) `TranscribeError` ile açık nedenli hata, (3) `--fingering` modunda konumu sabit tutup yalnız parmak alma seçeneği (ses modunun parmakla birleştirilmesi — katman-5'te bu kombinasyon yok). ASCII TAB çıktısı ikinci format olarak korunur. Ayrıntı: ROADMAP_NEXT §1 İş 1.
- **Öncelik: KRİTİK.**
- **Doğrulama:** Birleştirme sonrası tek komut hem `.musicxml` hem ASCII TAB üretmeli; `check_inference_regression` farksız geçmeli.

### E-02 · `NoteEvent` parmak alanı taşımıyor
- **İlk görülme:** `docs/devlog/README16.md §6.2` (P1)
- **Dosya/satır:** `gtab/core/note_event.py:35-51`
- **Açıklama:** Şemada `string`, `fret`, `technique`, `technique_params`, `confidence` var; **`finger` yok**. Ritim/ölçü alanları da yok.
- **Etki:** Parmaklandırma ve ritim şemaya giremiyor → her tüketici paralel dizi kullanıyor (`E-01`'deki iki sürüm bunu iki farklı şekilde yapıyor). "Tüm katmanların paylaştığı tek kaynak" iddiası kırık.
- **Mevcut durum:** **AÇIK** (P6 planında).
- **Önerilen çözüm:** `finger: Optional[int] = None` + `hand_position: Optional[float] = None` (+ ritim için `beat: Optional[float]`). Varsayılan `None` → **mevcut davranış birebir korunur**, `check_inference_regression` farksız kalır.
- **Öncelik: YÜKSEK** (`E-01`'in ön koşulu).
- **Doğrulama:** `python -m unittest discover -s tests -t .` + `check_inference_regression`.

### E-03 · `render_ascii_tab` parmak göstermiyor
- **İlk görülme:** `README16.md §6.2`
- **Dosya:** `gtab/decoding/decode.py:157 render_ascii_tab`
- **Açıklama:** ASCII TAB yalnız tel/perde çiziyor. `scripts/transcribe.py` bunu ayrı bir nota tablosuyla telafi ediyor (`note_table`), TAB'ın kendisinde parmak yok.
- **Etki:** Düşük (MusicXML tarafında `<fingering>` zaten yazılıyor).
- **Mevcut durum:** **AÇIK.**
- **Öncelik: DÜŞÜK** (`E-02` sonrası kolay).

### E-04 · DEVLOG NUMARASI ÇAKIŞMASI (iki dal aynı dosya adını farklı içerikle kullanıyor)
- **İlk görülme:** bu denetim
- **Dosyalar:** `docs/devlog/README15.md`, `docs/devlog/README16.md`
- **Açıklama:**

  | Dosya | `Armagan-katman-4` | `origin/katman-5-ritim` |
  |---|---|---|
  | `README15.md` | Katman 4 Faz 0 — ölçüm ve test protokolü | Katman 5 — ritim ve kuantalama |
  | `README16.md` | Katman 3.13 P1 — parmaklandırma incelemesi | Katman 5 Adım 2 — ölçülü TAB (MusicXML) |
  | `README17.md` | P2 — test altyapısı | (yok) |
  | `README18.md` | P3 — çalınabilirlik | (yok) |
- **Etki:** Düz `git merge` bu iki dosyada içerik çatışması üretir; dikkatsiz çözümde **bir tarafın devlog'u kaybolur**. Devlog projenin tek tarihsel kaydı olduğu için bu geri alınamaz bilgi kaybı demektir.
- **Mevcut durum:** **AÇIK — KRİTİK.**
- **Kanıt:** `git diff --stat HEAD origin/katman-5-ritim` → `docs/devlog/README15.md | 615 +-`, `README16.md | 243 +-`.
- **Kök neden:** İki dal ortak atadan (`72295e6`) ayrılıp paralel ilerledi; ikisi de "en yüksek numaralı README = güncel katman" kuralını (`CLAUDE.md:11`) bağımsız uyguladı.
- **Önerilen çözüm:** Birleştirmeden ÖNCE katman-5'in iki devlog'unu yeniden adlandır: `README15.md` → `README19.md`, `README16.md` → `README20.md` (ikisi de 11 Ekim tarihli, kronoloji bozulmaz; P1-P3 zaten 15-18 aralığını kullanıyor). İç çapraz referanslar (`README16.md`'nin "[README15](README15.md)" atfı) da güncellenmeli.
- **Öncelik: KRİTİK — birleştirmenin ilk adımı.**
- **Doğrulama:** Birleştirme sonrası `docs/devlog/` içinde 1..20 arası her numara **tek** dosya ve tek konu; `grep -rn "README1[5-9]\|README20" docs/` ile çapraz referanslar tutarlı.

### E-05 · Katman-5 dalı ölçüm/test altyapısını içermiyor
- **İlk görülme:** bu denetim
- **Açıklama:** Katman-5 `72295e6`'dan ayrıldığı için Faz 0 ve sonrası **hiç** yok: `gtab/evaluation/{protocol,selection,baselines,fingerprint,technique_eval}.py`, `gtab/decoding/playability.py`, `gtab/data/contour.py`, `scripts/eval/{check_protocol,check_inference_regression,eval_playability,inventory_technique,probe_gaps_f0}.py`, `docs/baselines/*`, **ve `tests/` dizininin tamamı**.
- **Etki:** Ritim ve MusicXML kodu 170 testlik ağa hiç girmemiş; protokol kapılarını kullanmıyor. Buna karşılık katman-5 kodu protokolü **elle** uyguluyor (gaps_fit eğitim, gaps_val seçim, gaps_test yalnız rapor) — devlog'larda tutarlı görünüyor ama mekanik zorlayıcı yok.
- **Ters yön çakışması:** Katman-5 `gtab/decoding/edition.py`'ı 212 satır değiştirdi (log-lineer modeli `archive/gtab/decoding/edition_loglinear.py`'a taşıdı). Mevcut daldaki `tests/test_edition.py::TestLogLinearDecode` (2 test) `edition.load_edition`/`decode`'u import ediyor → **birleştirmede bu iki test kırılır.**
- **Mevcut durum:** **AÇIK.**
- **Önerilen çözüm:** Birleştirme yönü **katman-5 → mevcut dal** olsun (ölçüm altyapısı korunur), `edition.py` çakışması elle çözülsün: log-lineer kod arşive gidiyorsa `test_edition.py` import yolu güncellenir ya da o iki test arşiv yoluna taşınır.
- **Öncelik: KRİTİK** (birleştirme planının parçası).
- **Doğrulama:** Birleştirme sonrası `python -m unittest discover -s tests -t .` → 170+ test, hata yok; `check_protocol` 17/17.

### E-06 · Kullanılmayan checkpoint'ler kökte duruyor (mevcut dalda)
- **İlk görülme:** katman-5 commit `37cbf39` bunu çözüyor
- **Açıklama:** Mevcut dalda `checkpoints/` 20 dosya; yalnız 3'ü canlı (`tabcrnn_deep2.pt` + `.cal.json`, `edition_seq.pt`, `transitions.npz`). Katman-5 14 dosyayı `checkpoints/archive/` altına taşıdı.
- **Etki:** Düşük (karışıklık). Ancak taşıma işlemi birleştirmede yol değişikliği üretiyor; `fingerprint.py` ve `d1_baseline.json` referansları `tabcrnn_deep2.pt`'a bakıyor (kökte kaldığı için etkilenmiyor — doğrulandı).
- **Mevcut durum:** **ÇÖZÜLDÜ** (katman-5'te; birleştirmeyle gelir).
- **Öncelik: DÜŞÜK.**

### E-07 · MusicXML / MuseScore: mevcut dalda yok, katman-5'te hazır
- **İlk görülme:** bu denetim
- **Dosyalar:** `gtab/render/musicxml.py` (128), `gtab/render/score.py` (94), `scripts/eval/eval_score.py` (163) — **yalnız katman-5**
- **Açıklama:** Kullanıcının "MusicXML dönüştürücü yazma, gereksinimleri çıkar" talimatı için önemli: **dönüştürücü zaten yazılmış ve doğrulanmış.** Gidiş-dönüş: 202 partisyon, 159.796 nota → **birebir 1,0000**; ölçü doluluğu bozuk 0/14.398. Uçtan uca dosya geçerliliği %100.
- **Etki:** Mevcut dalda çalışan kullanıcı MusicXML üretemiyor.
- **Mevcut durum:** **AÇIK** (mevcut dal için); katman-5'te ÇÖZÜLDÜ.
- **Öncelik: KRİTİK** (birleştirmenin ana kazancı). Ayrıntı: ROADMAP_NEXT §4.

### E-08 · Protokol kapıları yalnız yeni kodda zorunlu
- **İlk görülme:** `docs/devlog/README15.md` "Açık riskler" 4 (kendi tespiti)
- **Dosyalar:** `scripts/train/train_onset.py`, `scripts/train/train_edition_seq.py`, `scripts/eval/eval_pitch.py`, `scripts/eval/eval_hmm.py` — bilinçli olarak değiştirilmedi
- **Açıklama:** `assert_may_train` / `assert_may_select` / `EpochSelector` yalnız Faz 0 sonrası yazılan kodda çağrılıyor. Eski yollar `gaps_files()`'ı doğrudan kullanıyor.
- **Etki:** Protokol ihlali ilkesel olarak mümkün; fiilen bir kez yaşandı (`G-07`).
- **Mevcut durum:** **KISMEN ÇÖZÜLDÜ** (yeni kod korunuyor, eski kod serbest).
- **Önerilen çözüm:** Eski script'lere kapı eklemek **davranışı değiştirme riski taşıyor** (`gaps_training_files()` karantinayı düşürüyor → farklı eğitim listesi → farklı model). Bu yüzden önerilen: dokunmamak, ama her yeni eğitim/ölçüm yolunun kapıları çağırması kuralını sürdürmek (zaten README15'te yazılı).
- **Öncelik: ORTA** (bilinçli kabul edilmiş risk).

### E-09 · `test_edition.py` kapı koşulu `mir_eval`'i kapsamıyor
- **İlk görülme:** bu denetim (fiilen tetiklendi)
- **Dosya/satır:** `tests/test_edition.py:242` — `from scripts.eval.eval_edition import pooled`
- **Açıklama:** Kapı `importlib.util.find_spec("torch")` + checkpoint varlığını kontrol ediyor; ama import zinciri `eval_edition → gtab/evaluation/pitch_eval.py:18 → mir_eval`. `mir_eval` kurulu değilse test **atlanmıyor, hata veriyor**.
- **Kanıt (denetimde üretildi):** sistem Python'ı (torch 2.13.0+cpu var, mir_eval yok) ile `ModuleNotFoundError: No module named 'mir_eval'` → `FAILED (errors=1)`. `.venv` ile **170/170 OK**.
- **Etki:** Düşük ama yanıltıcı: `README15.md` "testler yalnız numpy kullanıyor — torch olmadan da geçiyor" diyor; bu ifade `test_edition.py` için geçerli değil. Yanlış yorumlayıcıyla koşulan test gerçek bir hata sanılabilir.
- **Mevcut durum:** **AÇIK.**
- **Önerilen çözüm:** Kapıya `find_spec("mir_eval")` ekle (tek satır). Ayrıca `CONTRIBUTING.md`'ye test komutunun `.venv` ile koşulması notu.
- **Öncelik: YÜKSEK** (maliyet: bir satır).
- **Doğrulama:** `mir_eval` olmayan bir yorumlayıcıyla suite koşulduğunda hata değil **skip** gelmeli.

### E-10 · Katman-5'in önbellekleri bu makinede yok
- **İlk görülme:** bu denetim
- **Açıklama:** Katman-5 kodu `data/cache/rhythm/` ve `data/cache/beat_feats/tabcrnn_deep2/` (808 MB) bekliyor; yerel `data/cache/` 8 dizin içeriyor ve ikisi de yok.
- **Etki:** Birleştirme sonrası `eval_rhythm` / `eval_score` / `scripts/app/transcribe.py` doğrudan çalışmaz. Önce: `python -m scripts.data.build_rhythm` (~3 dk) + `python -m scripts.data.build_beat_feats` (808 MB).
- **Mevcut durum:** **AÇIK.**
- **Kanıt:** `ls data/cache/` → contour_guitarset, gaps_meta.csv, gaps_tab, gaps_test, gaps_train, train, train_comp, val, val_comp.
- **Öncelik: YÜKSEK** (birleştirme sonrası ilk adım; disk kontrolü gerekir).

---

## F — Viterbi

### F-01 · Öğrenilen tel geçiş modeli güçlü modellerde katkısız
- **İlk görülme:** `docs/devlog/README6.md` (el yapımı), `README7.md` (öğrenilen)
- **Dosyalar:** `gtab/decoding/viterbi.py:84 transition_cost`, `:113 viterbi_assign`, `:188 viterbi_transcribe` (el yapımı); `gtab/decoding/transitions.py` (öğrenilen); `checkpoints/transitions.npz`
- **Mevcut pipeline'daki görevi:** `build_lattice` + `decode_lattice` hâlâ çağrılıyor (`scripts/transcribe.py:transcribe_cqt`, katman-5 `pipeline.py`) ama **`w_transition = 0` seçildiğinde greedy ile aynı sonucu veriyor** — ve güncel modellerde seçim 0 çıkıyor.
- **Kullanılan geçiş modeli:** veriden öğrenilen log-lineer model (`TransitionModel.fit`, L-BFGS benzeri gradyan inişi, L2 düzenlileştirme). Öncesinde el yapımı maliyet (perde mesafesi + tel değişimi + boşluk) denendi.
- **Hesaplama maliyeti:** Nota başına aday sayısı × 6 tel; kafes çözümü nota sayısında doğrusal. Pratikte ihmal edilebilir (ölçüm kaydı yok, ama çözümleme süresi içinde fark edilmiyor).
- **Hata örüntüleri:** README7 ölçümü — CRNN greedy 0,634 → öğrenilen önsel 0,662 (zayıf modelde kazanç), ama güçlü modellerde `w_tr = 0` seçiliyor. Sentetik (el pozisyonu modellenmiş) veride el yapımı Viterbi 0,600 → 0,684; **gerçek veride katkı yok** (README6:44). Yorum: gitarist gerçekte "en az el hareketi" ilkesine uymuyor; model kazandıkça önselin taşıdığı bilgi gereksizleşiyor.
- **Mevcut durum:** **ÇÖZÜLDÜ / kapatıldı** — iki kez ölçüldü, iki kez katkısız bulundu, `CLAUDE.md:119-120` "tekrarlama" listesinde.
- **Daha iyi alternatif mümkün mü:** Teorik olarak evet ama ölçüm aksini söylüyor: tel hatalarının %99'u komşu tel ve hatalar **kararsızlık** kaynaklı (marj 0,09) — yani sorun geçiş modelinin zayıflığı değil, **emisyon** (akustik) bilgisinin yetersizliği. Geçiş modelini iyileştirmek yanlış yere yatırım olur. Buna karşılık `EditionNet` (BiLSTM, sembolik) tel 0,815 ile naif 0,745'i geçiyor — yani dizi bilgisi **sembolik tarafta** işe yarıyor, akustik tarafta yarıyor demiyoruz.
- **Önerilen çözüm:** Şimdilik değiştirme (kullanıcı talimatı). Opsiyonel iyileştirme olarak kaydedildi.
- **Öncelik: ERTELENDİ.**

### F-02 · Ölü Viterbi kodu temizlenmedi
- **Dosya:** `gtab/decoding/viterbi.py:84-203` (el yapımı Viterbi), `scripts/eval/eval_viterbi.py`
- **Açıklama:** Çürütülmüş el yapımı Viterbi kodu yolda. Katman-5 `eval_viterbi.py`'ı arşive taşıdı ama `gtab/decoding/viterbi.py` içindeki fonksiyonlar kaldı (aynı dosya `segment` ve `pick_peaks` gibi **canlı** fonksiyonları da içerdiği için bölünemiyor).
- **Mevcut durum:** **ERTELENDİ** (bilinçli; `CONTRIBUTING.md` "ölü kodu haber ver, silme" kuralı).
- **Öncelik: DÜŞÜK.**

---

## G — Performans, kod kalitesi ve süreç

### G-01 · Kalibrasyon maliyeti ~1 saat
- **Dosya:** `gtab/evaluation/pitch_eval.py:314 calibrate`
- **Açıklama:** Tam kalibrasyon (eşik × onset × kural ızgarası, iki alan) ~1 saat; `--save-cal` / `--load-cal` ile saniyelere düşüyor. `CLAUDE.md:114`: *"kalibrasyon seçimi değişmediyse her zaman `--load-cal` kullan"*.
- **Etki:** Her yeni model için tek seferlik 1 saat; iterasyon hızını sınırlıyor.
- **Mevcut durum:** **KISMEN ÇÖZÜLDÜ** (önbellekleme mevcut).
- **Öncelik: DÜŞÜK** (kabul edilmiş maliyet).

### G-02 · `build_beat_feats` 808 MB önbellek üretiyor
- **Dosya:** `scripts/data/build_beat_feats.py` (katman-5)
- **Açıklama:** 469 kayıt, 10,6 saat, float16 donmuş temsil. **Ses modeli değişirse yeniden üretilmesi gerekiyor** (katman-5 README15: "yeni ses modelinde yeniden").
- **Etki:** Disk + yeniden üretim süresi; deep2 değişirse ritim zinciri de yeniden kurulmalı.
- **Mevcut durum:** **AÇIK** (tasarım sonucu).
- **Öncelik: DÜŞÜK** (bilgi; planlamada dikkate alınmalı).

### G-03 · ANA ARAŞTIRMA SORUSU CEVAPSIZ + SynthTab verisi silinmiş
- **İlk görülme:** sunum slayt 6 (araştırma sorusu) ve slayt 10 (ana deney); `docs/devlog/README9.md:40,49-52`
- **Açıklama:** Sunumun önerilen araştırma sorusu: *"Büyük ölçekli sentetik nylon-string ön-eğitimi ve gerçek klasik gitar verisiyle domain adaptation, polifonik audio-to-TAB genellemesini ne ölçüde iyileştirir?"* Ana deney tablosu (slayt 10, README9:43-52): **GuitarSet-only vs SynthTab→GuitarSet vs SynthTab→GuitarSet→GAPS**. Tablodaki 3.8d satırları **hiç doldurulmadı**.
- **Neden atlandı:** README13 (3.12) kapasite farkını 0,004 ölçtü ve *"SynthTab (3b) gerekmedi"* sonucuna vardı (`CLAUDE.md:109`). **Bu iki soru aynı değil:** kapasite ölçümü "model yeterince büyük mü" sorusunu cevaplıyor; araştırma sorusu "sentetik ön-eğitim genellemeyi artırır mı" sorusunu soruyor. İkincisi test edilmedi.
- **Veri durumu:** `data/raw/` yalnız `gaps_hf` (11 GB) içeriyor. README9:22'nin "indirildi ve doğrulandı" dediği `SynthTab_Dev.zip` **diskte yok**. `gtab/data/synthtab.py` (150 satır) ve `scripts/data/build_synthtab.py` kod olarak duruyor ama veri olmadan çalışmaz.
- **Bilinen ön bulgu:** *"sentetik naylon (SynthTab Dev, 4 parça) model için kolay (tab F1 0,73), GAPS gerçek kayıt zor"* (`CLAUDE.md:125-126`) — yani sentetik veri gerçek kayıttaki zorluğu temsil etmiyor. Bu, deneyin sonucunu tahmin etmez ama hipotezi zayıflatan bir sinyal.
- **Etki:** Tezin akademik özgünlük iddiası (slayt 6) deneysel desteksiz.
- **Mevcut durum:** **AÇIK** (ERTELENDİ olarak da okunabilir; karar kullanıcının).
- **Önerilen çözüm:** ROADMAP_NEXT §6'da iki yol: (a) `B-04`'ün ucuz kapsam ölçümüyle başla, sonuç hipotezi destekliyorsa kontrollü deneyi kur; (b) tezde soruyu yeniden çerçeveleyip kapasite ölçümünü gerekçe olarak sun.
- **Öncelik: KRİTİK** (tez çıktısı).
- **Doğrulama:** Deney yapılırsa: aynı iki test setinde (gaps_test + GuitarSet val) 4 satırlık tablo; kabul kuralı README9:70-71'de hazır (*"GAPS test'te ≥ +0,03 nota F1 ve GuitarSet tab F1'de en fazla −0,01"*).

### G-04 · Teknik etiketi yok → Katman 4 beklemede
- **İlk görülme:** `docs/devlog/README15.md` (Faz 0 ön ölçümü), ayrıntı `README16.md §1`
- **Açıklama:** 300 GAPS MusicXML dosyası tarandı: `slide` 0, `glissando` 0, `harmonic` 0, `trill-mark` 0, `slur` 4 dosya, `tremolo` 1 dosya; yalnız artikülasyon/dinamik var (`staccato` 48, `accent` 93, `dynamics` 226). GuitarSet JAMS'te teknik namespace'i yok.
- **Etki:** Teknik tespiti ancak **kuraldan türetilmiş sözde etiketlerle** yapılabilir; klasik gitarda sayısal doğruluk iddia edilemez.
- **Mevcut durum:** **ERTELENDİ** (bilinçli, gerekçeli). Faz 0-2A çıktıları korundu ve arşivlendi: `gtab/evaluation/technique_eval.py`, `gtab/data/contour.py`, `data/cache/contour_guitarset/` (18 MB), 29 + 21 test.
- **Önerilen çözüm:** Karar verildi ve doğru: emek veri-zengin hedefe (parmaklandırma: 165 kayıt, 38.828 parmaklı nota) yöneltildi. Teknik için Faz 5'te küçük bir **elle etiketli küme** zorunlu tutuldu.
- **Öncelik: ERTELENDİ** (karar isabetli).

### G-05 · Vitrin dokümanları iki taban geride
- **İlk görülme:** bu denetim · bilinçli erteleme kaydı `README15.md` "Eklenen dosyalar"
- **Dosyalar:** `README.md` (kök), `CONTRIBUTING.md`, katman-5 `docs/SOZLUK.md §5`
- **Tespit edilen tutarsızlıklar:**

  | Belge | Yazdığı | Gerçek |
  |---|---|---|
  | kök `README.md` sonuç tablosu | son satır 3.9, tab F1 0,739 | deep2, 0,777 |
  | kök `README.md` "güncel en iyi" | `tabcrnn_comp.pt` | `tabcrnn_deep2.pt` |
  | kök `README.md` yol haritası | 3.10 / 3.13 açık, 3.8 açık | 3.10-3.12 kapandı, 3.13 P1-P3 bitti |
  | kök `README.md` kurulum | `.venv\Scriptsctivate` (bozuk) | `.venv\Scripts\activate` |
  | `CONTRIBUTING.md` | taban `tabcrnn_rep_off.pt`, 0,760 | deep2, 0,777 |
  | `SOZLUK.md §5b` | "Güncel taban `tabcrnn_rep_off.pt`" | deep2 |
  | `README9.md:41` | "Katman 7'ye girdi" | Katman 7 tanımsız |
  | `requirements.txt` başlığı | "Python 3.10" | ortam 3.11 |
  | kök `README.md` / `CONTRIBUTING.md` | sonuçların yanında **split cinsi yok** | 0,777 ve 0,924 **validation**'dır |
- **Etki:** Danışman/jüri ilk `README.md`'yi okur; dosya iki taban eski ve manşet sayıların validation olduğunu söylemiyor (`A-04`).
- **Mevcut durum:** **AÇIK** (bilinçli erteleme, ama Katman 4 beklemeye alındığı için süresiz hâle geldi).
- **Önerilen çözüm:** Tek seferde güncelleme; kaynak `docs/baselines/d1_baseline.json`. Sonuç tablosuna "split cinsi" kolonu eklenir.
- **Öncelik: KRİTİK** (düşük maliyet, yüksek görünürlük).
- **Doğrulama:** `tests/test_baselines.py` D1 değerlerini zaten sabitliyor; README sayıları o dosyayla elle karşılaştırılır.

### G-06 · Checkpoint üretim künyesi yok
- **İlk görülme:** bu denetim
- **Açıklama:** "Hangi checkpoint hangi komutla, hangi init'ten, hangi veriyle, hangi seed'le üretildi ve hangi ölçütle kabul edildi" bilgisi devlog paragraflarına dağılmış. `tabcrnn_deep2.pt` için komut `CLAUDE.md:103-110` içinde parça parça.
- **Etki:** Tekrarlanabilirlik ve tez eki için zorluk; yeni bir eğitimi "aynı koşullarda" kurmak için 3-4 devlog okumak gerekiyor.
- **Mevcut durum:** **AÇIK.**
- **Önerilen çözüm:** `checkpoints/PROVENANCE.md` — checkpoint başına bir satır/blok. Mevcut bilgiden derlenebilir, yeni ölçüm gerekmez.
- **Öncelik: YÜKSEK** (tez eki değeri yüksek, maliyet düşük).

### G-07 · `gaps_test` bir kez karar kuralında kullanıldı
- **İlk görülme:** `docs/devlog/README15.md` "Neden bu faz" 2 (kendi tespiti)
- **Dosya:** README14 Adım 2 kabul kuralı
- **Açıklama:** Kabul kuralı *"gaps_test tel ≥ naif + 0,05"* biçimindeydi. Ağırlıklar test'e uydurulmadı (seçim `gaps_val`'de), ama **kabul/ret kararı test setine bakılarak** verildi.
- **Etki:** `edition_seq.pt`'ın kabul kararı tam bağımsız değil. Raporlanan gaps_test sayıları (tel 0,815 / parmak 0,636) bu yüzden hafifçe iyimser okunmalı.
- **Mevcut durum:** **AÇIK — geri alınamaz, belgelendi.** Faz 0 bunu tespit etti ve tekrarını protokolce engelledi (`assert_may_select("gaps_test")` → `ProtocolError`).
- **Önerilen çözüm:** Düzeltme yok (geçmiş değiştirilemez). Tezde bu sayının yanında dipnot: "kabul kararı test setine bakılarak verildi; seçim gaps_val'de yapıldı".
- **Öncelik: ORTA** (dürüst raporlama).

### G-08 · `395_w41wc` karantinası eski script'lerde etkisiz
- **İlk görülme:** `docs/devlog/README15.md` "Açık riskler" 3
- **Dosya:** `gtab/evaluation/protocol.py:221 is_quarantined`, `:231 gaps_training_files`
- **Açıklama:** Kimliği doğrulanamayan kayıt (tüm `performer_name`, `channel_name`, `ss_artist`, `video_title` alanları boş; elde yalnız `yt_id`) karantinaya alındı ama **önbellekten silinmedi** (mevcut sonuçların yeniden üretilebilirliği için). Eski script'ler `gaps_files()`'ı doğrudan çağırdığı için **mevcut `tabcrnn_deep2.pt` bu kaydı perde denetimi olarak görmüş olabilir** (`gaps_fit` tarafında).
- **Etki:** Düşük. Kaydın TAB etiketi zaten yok (`summary.csv`: `use=False`, notaların %39'u etiketli) → edisyon modelini etkilemiyor; etkisi yalnız perde denetimiydi.
- **Mevcut durum:** **AÇIK — kabul edilmiş, belgelenmiş risk.**
- **Kanıt:** `check_protocol` çıktısı: *"karantina: ['395_w41wc'] | onbellekte duruyor (silinmedi) | listelerden cikarilan: ['395_w41wc']"* (denetimde koşuldu).
- **Öncelik: DÜŞÜK** (geçmiş değiştirilmiyor; yeni eğitimler kaydı görmeyecek).

### G-09 · Çıkarım parmak izi yalnız 2 GuitarSet kaydıyla sınırlı
- **İlk görülme:** `docs/devlog/README15.md` "Açık riskler" 5
- **Dosya:** `docs/baselines/inference_fingerprint.json`, `gtab/evaluation/fingerprint.py`
- **Açıklama:** Referans `tabcrnn_deep2.pt` ile `val`'in ilk 2 kaydında (`05_BN1-129-Eb_solo`, `05_BN1-147-Gb_solo`). **GAPS yolundaki bir regresyonu yakalamaz.**
- **Mevcut durum:** **AÇIK** (Faz 0'ın kendi önerisi: `--split gaps_val --files 1` ile genişletilmesi).
- **Öncelik: ORTA** (katman-5 birleştirmesi öncesi genişletilmesi değerli — ritim kodu GAPS yolunu kullanıyor).
- **Doğrulama:** `python -m scripts.eval.check_inference_regression --split gaps_val --files 1`

### G-10 · Tek seed — "±0,01 gürültü" varsayımı ölçülmedi
- **İlk görülme:** `CLAUDE.md:57` (kural), `docs/baselines/d1_baseline.json` (tolerans)
- **Açıklama:** "Tek seed farkları ±0,01 gürültü sayılır" kuralı tüm kabul kararlarının altında yatıyor ama **ikinci seed koşusu hiç yapılmadı**. Buna karşılık yeniden-koşu oynaması ölçüldü (GPU ±0,002).
- **Etki:** ±0,013 gibi sınır durumlardaki kararların (3.11 Adım 4) istatistiksel dayanağı zayıf.
- **Mevcut durum:** **DOĞRULAMA GEREKİYOR.**
- **Önerilen çözüm:** Yalnız bir model için ikinci seed (`--seed 2`) koşusu; fark ölçülür ve tolerans ya doğrulanır ya düzeltilir. Maliyet: bir eğitim koşusu.
- **Öncelik: ORTA** (tez için değerli).

### G-11 · Önbellek şeması sürümsüz
- **İlk görülme:** bu denetim
- **Açıklama:** `.npz` dosyalarında `schema_version` / `config_hash` yok. `gtab/config.py` değişirse eski önbellekle yeni kod sessizce uyumsuz olur. `CONTRIBUTING.md` riski not ediyor (*"config.py değiştirmek tüm modelleri geçersiz kılar"*) ama mekanik koruma yok.
- **Etki:** Sessiz yanlış sonuç riski (düşük olasılık, yüksek etki).
- **Mevcut durum:** **AÇIK.**
- **Öncelik: ORTA.**

### G-12 · Kritik etiket üretim yolu testsiz
- **İlk görülme:** `docs/devlog/README17.md §8` (P2 kendi tespiti)
- **Dosyalar:** `gtab/data/gaps_score.py` (255 satır), `scripts/data/build_gaps_tab.py`, `scripts/eval/eval_edition.py` (bütün olarak), log-lineer `fit`
- **Açıklama:** Tüm parmaklandırma ve ritim etiketlerinin kaynağı `gaps_score.align()` (MusicXML + tekrar açma + syncpoint + MIDI eşleme). **Hiç birim testi yok**; tek doğrulaması istatistiksel (perde roll uyumu %98,93, parmak tutarlılığı %99,4, syncpoint uyumu 22,9 ms).
- **Etki:** Bu fonksiyondaki bir hata tüm etiket zincirini (parmaklandırma + ritim) sessizce bozar; istatistikler yakalayamayabilir.
- **Mevcut durum:** **AÇIK.**
- **Not:** Katman-5 `gaps_score.py`'ı 20 satır değiştirdi (`align` → `order`, `lens`, `score_x` döndürüyor) ve değişikliği **9 kayıtta `gaps_tab` ile birebir** doğruladı — iyi pratik, ama birim test değil.
- **Önerilen çözüm:** Sentetik MusicXML (tekrar işaretleri + kapo + fingering içeren küçük bir partisyon) ile `align()` birim testi.
- **Öncelik: ORTA.**
