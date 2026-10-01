# Katman 3.7 — Veri genişletme (GuitarSet comp + CQT çoğaltma)

## Neden
Katman 3.6b teşhisinin sonucu: tel atamasında kalan kayıp **akustik**.
Gerçek notalar verildiğinde bile CRNN teli %74 doğru seçiyor; öğrenilen
müzikal önsel CRNN'e ek bilgi getirmiyor (BiLSTM bu bağlamı sesten zaten
öğrenmiş). Bir sonraki kaldıraç, modelin her telin tınısını daha çok ve daha
çeşitli örnekte duyması.

## Ne yapıldı
1. **GuitarSet `comp` kayıtları eklendi** (180 kayıt; diskte hazırdı ama hiç kullanılmamıştı).
   Aynı gitaristler, aynı gitar, aynı mikrofon → eğitim sesi ~2 katına çıkıyor.
   Kayıtların ~%73'ünde aynı anda ≥2 tel çalıyor (akor); tel başına softmax
   şeması bunu model değişmeden destekliyor. Bu, Katman 6 (polifoni) için de
   ilk adım.
2. **CQT uzayında, etiketi koruyan çoğaltma** (yalnızca eğitimde, ham ses gerekmez):
   - perde kaydırma ±2 yarım ses = ±4 bin; etiket **aynı telde** fret±k
     (tel kimliği korunur, model her teli daha çok perdede duyar).
     Fret aralık dışına taşarsa kaydırma uygulanmaz.
   - kazanç ±6 dB ve küçük Gauss gürültüsü → kayıt seviyesine ve
     mikrofona dayanıklılık (sunumdaki "telefon/oda" robustness hedefi).

## Karşılaştırılabilirlik
- `data/cache/train` ve `data/cache/val` **değişmedi**.
- Oyuncu 00–04 comp → `data/cache/train_comp/`; oyuncu 05 comp → `data/cache/val_comp/`.
- Model seçimi ve ana metrik **her zaman** 05 solo val üzerinde yapılır; eski
  sayılarla (CRNN greedy 0.664) birebir kıyaslanabilir. `val_comp` yalnızca
  polifonik ek rapor için.
- Yeni modeller ayrı dosyalara kaydedilir (`--ckpt`); `tabcrnn.pt` korunur.

## Değişen / yeni dosyalar
- `build_comp.py` — YENİ: comp → CQT + roll + tab (yarıda kalırsa kaldığı yerden devam eder)
- `gtab/torch_dataset.py` — `GuitarSetSeq`: split listesi + `augment` (varsayılan davranış aynı)
- `train_crnn.py` — `--splits`, `--augment`, `--ckpt`, `--val-split`; `--sweep` marj aralığı 0.95'e genişletildi
- `eval_viterbi.py`, `eval_hmm.py` — `--ckpt` (varsayılan eski dosyalar)

## Testler
- Perde kaydırma: kaydırılan CQT'deki temel frekans bini yeni etiketin perdesine denk düşüyor (50 chunk) ✓
- Fret sınırları korunuyor, sessiz ve dolgu kareleri değişmiyor ✓
- Comp hattı tek kayıtta uçtan uca çalışıyor (~10 sn/kayıt) ✓
- Çoğaltmalı eğitim, Windows'ta DataLoader worker'larıyla çalışıyor (GPU ~0.8 GB) ✓

## Çalıştırma
```bash
# 1) comp önbelleği (~180 × 10 sn ≈ 30 dk)
python build_comp.py

# 2) deneyler -- her biri ayrı checkpoint
python train_crnn.py --epochs 30 --ckpt tabcrnn_base.pt                                    # (önerilen) aynı kodla taban
python train_crnn.py --epochs 30 --splits train,train_comp --ckpt tabcrnn_comp.pt          # + comp
python train_crnn.py --epochs 30 --splits train,train_comp --augment --ckpt tabcrnn_aug.pt # + comp + çoğaltma

# 3) ölçüm (3.6b ile aynı metrik; greedy / öğrenilen Viterbi dahil)
python eval_hmm.py --model crnn --ckpt tabcrnn_base.pt
python eval_hmm.py --model crnn --ckpt tabcrnn_comp.pt
python eval_hmm.py --model crnn --ckpt tabcrnn_aug.pt

# (ek) polifonik val raporu
python train_crnn.py --sweep --ckpt tabcrnn_aug.pt --val-split val_comp
```

## Karar kuralı
- Hedef: 05 solo val'de tab F1 > **0.664** (CRNN + greedy) ve oracle tel
  doğruluğu > **0.738**.
- Tek seed ile ±0.01'lik farklar gürültü sayılır.
- Kazanç varsa: en iyi checkpoint Katman 3 çıktısı olur, ardından Katman 4'e geçilir.
- Kazanç yoksa: veri miktarı darboğaz değil demektir. Sıradaki aday SynthTab
  nylon ön-eğitimi (sunumdaki Phase 4) ya da E seçeneği (mevcut haliyle Katman 4).

## Sonuçlar (1 Ekim 2026, tek seed, 05 solo val)

| Model | eğitim tab F1 (argmax) | oracle tel doğruluğu | uçtan uca en iyi tab F1 | yöntem |
|---|---|---|---|---|
| tabcrnn.pt (eski) | — | 0.738 | 0.664 | greedy |
| tabcrnn_base.pt (aynı kod, taban) | 0.522 | 0.727 | 0.654 | greedy |
| **tabcrnn_comp.pt (+comp)** | 0.541 | **0.762** | **0.667** | greedy |
| tabcrnn_aug.pt (+comp +çoğaltma) | 0.453 | 0.697 | 0.607 | öğrenilen Viterbi |

Yorum:
- **comp verisi işe yaradı ama az:** tel doğruluğu +0.035 (taban 0.727 → 0.762),
  ancak uçtan uca yalnızca +0.013. Eski ve yeni taban arasındaki fark (0.664 ve
  0.654) seed gürültüsünün ~±0.01 olduğunu gösteriyor; bu yüzden uçtan uca
  kazanç gürültü sınırında.
- **Çoğaltma zarar verdi** (−0.06). Olası sebep: perde kaydırma, tel kimliğini
  taşıyan tını ile perde konumu arasındaki fiziksel ilişkiyi bozuyor (gövde
  rezonansı ve inharmonisite kaymıyor). Model 30 epoch'ta da yakınsamadı (loss 0.38'e karşı 0.18).
- Öğrenilen önsel yine yalnızca zayıf modele (aug) yardım etti.
- Uçtan uca F1 kabaca iki faktörün çarpımı gibi davranıyor: nota/perde bulma
  (~0.82) × tel seçimi (~0.76). Tek bir faktörü birkaç puan iyileştirmek
  toplamı az etkiliyor.

**Karar:** GuitarSet-mic üzerinde tel ayrımı için azalan getiri noktasına
gelindi. Katman 3 çıktısı olarak `tabcrnn_comp.pt` + nota başına tek
pozisyon (greedy, eşik 0.7) dondurulur.
