# Katman 3 — Konsolidasyon + denetim düzeltmeleri

## Değişen dosyalar
- `gtab/torch_dataset.py` — `class_weights` yeniden yazıldı (bug düzeltmesi + `scheme`/`cap`)
- `gtab/decode.py` — `mode_filter` + `smooth_tab` (zamansal düzeltme / hayalet-nota temizliği)
- `build_tab_labels.py` — Windows dosya-tutucu düzeltmesi (yaz-önce-kapat)
- `train_tab.py` — cosine LR, seed, vektörize metrik, batch'li çıkarım, sessizlik-marjı (+sweep), smoothing

## Yeni komutlar
```bash
# 1) MEVCUT modelde marj taraması (retrain YOK — en ucuz F1 kazancı)
python train_tab.py --sweep

# 2) Daha iyi taban model (cosine + daha çok epoch; eğri hâlâ yükseliyordu)
python train_tab.py --epochs 40
python train_tab.py --sweep

# 3) Zamansal düzeltmeli, seçilen marjlı çıkarım
python train_tab.py --demo data/cache/val/05_Rock1-130-A_solo.npz --margin 0.10 --smooth 5 --prob-smooth 3

# (deney) precision lehine daha yumuşak ağırlık
python train_tab.py --epochs 40 --weight-scheme sqrt
```

## Sessizlik-marjı (yeni ana kaldıraç)
Tab modelinde precision/recall'ı retrain'siz ayarlar (Katman 2 eşik taramasının karşılığı):
- `margin > 0` → daha çok sessiz → **precision ↑**
- `margin < 0` → daha çok aktif → **recall ↑**
Tek modelle tüm P/R eğrisi gezilebilir; `--sweep` en iyi tab F1'i verini bulur.

## Varsayılan ağırlık
`inv` / `cap 20` (senin çalışan kurulumun). `sqrt` daha yumuşak/precision lehine bir deney.
Not: `class_weights`'te mean-normalizasyon YOK — kullanılmayan fret sınıfları ölçeği
bozup modeli sessizliğe çökertiyordu; kaldırıldı.
