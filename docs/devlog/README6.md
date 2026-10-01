# Katman 3.6 — Viterbi tel ataması (A seçeneği)

## Teşhis
pitch F1 ~0.85 (perde doğru) ama tab F1 ~0.58 (tel yanlış). Kayıp **telde**.
Karışık mikrofon sesinde tel kimliği fiziksel olarak zayıf bir sinyal. Çözüm:
onu sadece akustikten zorlamak yerine **gitaristin kısıtıyla** birleştirmek.

## Yöntem
```
skor = w_emission · Σ log P_model(tel, fret)   −   w_transition · Σ el_hareketi
```
- **emisyon** = modelin öğrendiği olasılık (veri zenginleştikçe güçlenir)
- **geçiş** = fret mesafesi + tel atlama; açık tel indirimli; uzun boşlukta ucuzlar
- **Viterbi** = toplam skoru maksimize eden *çalınabilir* yolu global olarak bulur

Retrain **gerekmez** — mevcut model çıktıları üstünde çalışır.

## Yeni dosyalar
- `gtab/viterbi.py` — perde marjinalizasyonu, nota segmentasyonu, maliyetler, Viterbi DP
- `eval_viterbi.py` — gerçek val setinde ölçüm + parametre taraması + `--demo`

Mevcut dosyaların hiçbiri değişmedi.

## Çalıştırma
```bash
python eval_viterbi.py                 # CNN modeli, tam tarama
python eval_viterbi.py --model crnn    # CRNN modeli
python eval_viterbi.py --quick         # küçük grid (hızlı ön bakış)

# bulunan parametrelerle tab üret
python eval_viterbi.py --demo data/cache/val/05_Rock1-130-A_solo.npz \
    --threshold 0.5 --w-transition 0.6
```

## Dürüstlük garantisi
`w_transition = 0` → Viterbi **birebir greedy'ye** indirgenir (test edildi).
Tarama w_transition=0'ı seçerse script açıkça *"çalınabilirlik önseli katkı
vermedi"* yazar. Kendimizi kandırmayan kurulum.

Script üç yöntemi **aynı kare-seviye tab F1** metriğiyle karşılaştırır:
baseline (argmax+marj) · greedy (nota başına en iyi) · Viterbi.

## Test durumu
- Sentetik (el pozisyonu modellenmiş) veride: baseline 0.600 → **Viterbi 0.684**
  (+0.084), precision 0.447 → **0.651**. Tarama w_transition=0 seçmedi.
- Dejenere durum: w_transition=0 = greedy ✓
- Sınır durumları: sessiz giriş, tek nota, boş liste ✓
- Farklı enstrümanlar: bas (4 tel), 7-telli ✓
- **Gerçek kazanç gerçek val setinde ölçülmeli** — sentetik gürültü gerçek
  modelin sistematik hata desenini temsil etmez.

## İleriye dönük (veri zenginleştiğinde)
`w_emission` / `w_transition` oranı tam bunun için var: model güçlendikçe
emisyonu artır, kuralları geri çek. Her veri genişlemesinden sonra
`eval_viterbi.py`'yi yeniden çalıştırıp ağırlıkları yeniden ayarla.
