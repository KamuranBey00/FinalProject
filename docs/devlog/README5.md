# Katman 3.5 — CRNN (CNN + BiLSTM zamansal kafa)

## Neden
Kare-bazlı TabCNN'in tavanı ~0.586 tab F1'di; düşük precision **yanlış-tel
karışıklığından** (orta-nota tel sıçraması) geliyordu — marjın çözemediği bir sorun.
BiLSTM zamanı modelleyip bunu kaynağında azaltır ve **nota sınırlarını stabilize eder**
(Katman 4'te F0 yörüngesini nota-başına okumak için şart).

## Değişen / yeni dosyalar
- `gtab/torch_dataset.py` — `GuitarSetSeq` eklendi (sabit uzunlukta chunk, pad + -100 maske)
- `gtab/model.py` — `TabCRNN` eklendi (CNN gövdesi + 2 katman BiLSTM)
- `train_crnn.py` — YENİ: sekans eğitimi, vektörize metrik, tam-sekans çıkarım, marj taraması

CNN dosyaları (TabCNN, GuitarSetTab, train_tab) DEĞİŞMEDİ — CRNN paralel duruyor.

## Çalıştırma
```bash
python train_crnn.py --epochs 30
python train_crnn.py --sweep
python train_crnn.py --demo data/cache/val/05_Rock1-130-A_solo.npz --margin 0.1 --smooth 5 --prob-smooth 3
```

## Notlar
- Girdi artık SEKANS: track sabit `CHUNK=200` karelik parçalara bölünür; son parça
  sıfırla doldurulup etiketi -100 ile maskelenir (`ignore_index`, kayba girmez).
- Çıkarım tüm track'i TEK sekans olarak işler (tek forward) — CNN'in kare-kare
  döngüsünden hızlı.
- Marj / smoothing / metrik altyapısı CNN ile ortak (train_tab'den alınır).
- Sentetik doğrulama: mimari öğreniyor ve kapasitesi var; gerçek sayılar GPU + tam veride.
