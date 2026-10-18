# Not — Geldiğimiz nokta + YouTube / dış platformlardan ara doğrulama

> **Durum (18 Ekim 2026):** küçük bir not, yeni katman değil. Taban değişmedi (`tabcrnn_deep2.pt`, aktif yığın
> `checkpoints/active.json`). Ölçüm kuralları (GuitarSet oyuncu 05 solo, GAPS `--disjoint`) aynen geçerli.

## Geldiğimiz nokta
- Katman 3.x akustik + tel katmanları kapandı; taban `tabcrnn_deep2.pt` (GS solo tab F1 0.777, GAPS test nota 0.779).
- Katman 3.13: "duyulan konum" (ses) / "edisyon önerisi" (`edition_seq_rec_s2.pt`, README19 U2) iki ayrı mod.
- Katman 5 çekirdeği: ritim + ölçülü TAB (MusicXML), yarı otomatik (ölçü imi + BPM / tap tempo); görüntüleyici
  `scripts/app/tab_viewer.html` (README19 U3).
- Katman 6 sürüyor (README18): kayıt koşulu, `--separate mix-vocals`, akort / kapo kabul; **kendi kayıtlarımız bekleniyor**.
- Katman 4 (teknikler) teknik veri hazır olunca.

## Karar: ara sıra dış kaynaktan doğrulama
Kendi kayıtlarımız yetersizken YouTube vb. platformlarda **icracının çalarken senkron tab gösterdiği** videolar bulunup
model çıktısı o tabla karşılaştırılacak. Amaç **eğitim değil, gerçek dünya kontrolü** (hata türlerini görmek):
- Eğitim verisi olarak kullanılmaz: web tabı icracının konumu değil, zamanlama yok (README14: partisyon TAB'ı icracıya
  genellenmedi); ayrıca indirme / telif nedeniyle veri seti paylaşılamaz.
- Ana ölçütlerin yerine geçmez; sonuçlar "gözlem" olarak bu not altına (ya da Katman 6 README'sine) eklenir.
- Uygun video: naylon telli klasik gitar, gerçek icracı (Guitar Pro sentezi değil), okunur tab + imleç, tabı icracı yazmış.

### Yöntem
1. `watch` eklentisi (yt-dlp + ffmpeg) ile indir, kareleri çıkar; ses → `ffmpeg -ac 1 -ar 22050`.
2. `python -m scripts.app.transcribe_ascii --audio <wav> --limit 0 --output <tab.txt>` (varsayılan `gaps_val` ayarı).
3. Tab şeridi kırpılarak birkaç saniyede bir kare okunur, ölçü ölçü elle yazılır.
4. Nota sırası hizalaması (LCS, perde): precision / recall / F1, eşleşenlerde aynı tel, kaçan / fazla nota türleri.
   Zamanlama karşılaştırılmaz (tabda ms zamanı yok).

## İlk deneme (18 Ekim): "The Last of Us – Main Theme" (Simple Guitar Tabs, youtu.be/oCaknYTMrLY)
Naylon telli, gerçek icracı, açık telli arpej ağırlıklı. Part 1, ölçü 1–10 (1–33 sn, tabda 122 nota):

| ölçü | değer |
|---|---|
| perde recall | 0.910 |
| perde precision | 0.771 |
| perde F1 | 0.835 |
| eşleşende aynı tel | 0.559 |

- **Fazla nota:** bas E2 ölçü içinde yeniden algılanıyor (11; tabda ölçü başına 1 — uzun tınlayan basın sönümü);
  aynı perde iki telde birden (≈10).
- **Tel:** model açık tel yerine basılı konum seçiyor (B0 → G4 ×12, e3 → B8 ×11, G0 → D5 ×8, e0 → B5 ×7). Tab icracıya ait
  → gerçek hata; GAPS oracle tel 0.610 ile aynı yön.
- Sınırlar: tab elle okundu (birkaç okuma hatası olabilir), tek parça.

## Sıradaki
- Aynı ses "edisyon önerisi" modunda: açık tel tercihi tel uyumunu artırıyor mu?
- Bulunan iki hata türü (bas yeniden algılama, açık tel tercihi) kendi kayıtlarımızla / GAPS'te ölçülüp doğrulanırsa
  Katman 6 adımı olarak ele alınır (önce ölç, sonra çözüm).
- Yeni uygun video bulundukça aynı yöntemle bu tabloya satır eklenir.
