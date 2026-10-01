# Katman 3.6b — Öğrenilen geçiş modeli (A seçeneğinin veriye dayalı hali)

## Katman 3.6 sonucu (neden bu adım)
El yapımı hareket maliyetiyle Viterbi, gerçek val setinde katkı vermedi:

| Model | baseline | greedy | Viterbi (el yapımı) | seçilen w_tr |
|---|---|---|---|---|
| CNN  | 0.523 | 0.502 | 0.511 | 0.1 |
| CRNN | 0.634 | **0.661** | 0.661 | 0.0 |

Not: Bu baseline'lar Katman 3.5 belgesindeki değerlerle (CNN 0.586 / CRNN 0.580)
uyuşmuyor; checkpoint'ler `--sweep` ile yeniden doğrulanmalı.

## Teşhis (sadece etiketlerle, ses yok)
Val'in gerçek notaları verildiğinde, yalnızca "hangi tel?" sorusunda:

| Yöntem | tel doğruluğu |
|---|---|
| naif (en düşük perde) | 0.206 |
| el yapımı maliyet (Katman 3.6) | 0.337 |
| **öğrenilen önsel** | **0.662** |

El yapımı kural, gitaristin gerçek yolunu kötü açıklıyor. Öğrenilen model
ise sesi hiç duymadan teli %66 oranında doğru buluyor ve oyuncular arasında
genelliyor (val NLL 0.407 < train 0.441; uniform 1.505).

## Yöntem
`P(pozisyon_t | pozisyon_{t-1}, perde_t)`, aday pozisyonlar üstünde bir softmax
(koşullu log-lineer model). Özellikler yorumlanabilir: Δfret, Δtel, açık tel,
kutu içinde kalma, pozisyon bölgesi, mutlak tel, uzun boşluk.
Ağırlıklar yalnızca train oyuncularından öğrenilir (oyuncu 05 hariç).
Emisyon da adaylar üstünde normalize edilir; böylece iki terim aynı ölçekte
olur ve `w_transition = 0` birebir greedy'ye eşit olur (test edildi).

## Yeni dosyalar
- `gtab/transitions.py` — TransitionModel (öğrenme/kayıt), kafes, Viterbi
- `eval_hmm.py` — teşhis + uçtan uca tab F1 + `--demo`
- `transitions.npz` — öğrenilen ağırlıklar (ilk çalıştırmada üretilir)

Mevcut dosyaların hiçbiri değişmedi.

## Testler
- Viterbi = kaba kuvvet arama (30 rastgele dizi × 4 ağırlık) ✓
- `w_tr = 0` = `eval_viterbi.decode_greedy` ✓
- Boş dizi ve tek nota ✓

## Çalıştırma
```bash
python eval_hmm.py --model crnn
python eval_hmm.py --model cnn
python eval_hmm.py --model crnn --demo data/cache/val/05_Rock1-130-A_solo.npz --threshold 0.8 --w-transition 1.0
```

## Gerçek val sonuçları (1 Ekim 2026)
Teşhis (oracle notalar, tel doğruluğu):

| | CNN | CRNN |
|---|---|---|
| öğrenilen önsel (ses yok) | 0.662 | 0.662 |
| ses / greedy | 0.592 | **0.738** |
| ses + öğrenilen önsel | **0.688** (w_tr=1.0) | 0.725 (w_tr=0.25) |

Uçtan uca tab F1:

| | baseline | greedy | öğrenilen Viterbi |
|---|---|---|---|
| CNN  | 0.523 | 0.502 | **0.544** (+0.021) |
| CRNN | 0.634 | **0.664** | 0.664 (w_tr=0, katkı yok) |

Yorum: Önsel, zamansal bağlamı olmayan CNN'e eksik olan sıralı bilgiyi veriyor
(+0.10 tel doğruluğu). CRNN'in BiLSTM'i ise bu bilgiyi sesten zaten öğrenmiş;
önsel ona ek bilgi taşımıyor, hatta hafifçe zarar veriyor. En iyi sistem:
**CRNN + nota başına tek pozisyon (greedy, eşik 0.9) = 0.664**.
Darboğaz artık akustik tel ayrımı (oracle notalarda CRNN %74). Bir sonraki
kaldıraç: daha fazla akustik veri.

## Karar kuralı
- Öğrenilen Viterbi, greedy (0.661) ve baseline'ı belirgin biçimde geçerse:
  Katman 3.6 çıktısı bu olur, ardından Katman 4'e geçilir.
- Geçmezse veri genişletilir. Önce GuitarSet'in kullanılmayan 180 `comp`
  kaydı akustik model için eklenir; önseli büyütmek için de sembolik tab
  (SynthTab/DadaGP) kullanılır.
