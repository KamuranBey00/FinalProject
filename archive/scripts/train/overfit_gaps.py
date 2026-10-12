"""
Katman 3.12 Adım 0b — Aşırı öğrenme testi: kapasite mi, etiket tavanı mı?

Adım 0'da GAPS eğitim ≈ doğrulama (nota F1 0.723 / 0.718) çıktı: model gördüğü kayıtlarda da düşük.
İki açıklama: (a) kapasite yetmiyor, (b) etiketler çelişkili, hiçbir model ezberleyemez.
Test: birkaç kayıtta çok epoch ince ayar, aynı kayıtlarda ölçüm.

Ölçüler (model düzeyi, çözümleme yok):
  - ANA: onset F1 (kare-hassas, en iyi eşik). GAPS'te başlangıçlar iyi hizalı.
  - kare F1 yalnız notaların İLK %80'inde (GAPS'te nota bitişi partisyon değerinden gelir, telin gerçek
    çınlamasını göstermez -> tam kare F1 tavana takılır; yalnız raporlanır).
  - hızlı tekrar onset'lerinin ±1 karede yakalanması; perde ve onset kaybı ayrı ayrı (kayıp ~0 iken F1
    takılıyorsa etiketler çelişiyor: eşikten bağımsız ikinci kanıt).
  - epoch 0 = aynı kayıtlarda eğitim öncesi taban.
Kurulum: dropout KAPALI (LSTM dahil), BatchNorm eval modunda (küçük veride kararlı).
Kontrol grubu: aynı kurulum + aynı perde kaybı GuitarSet kayıtlarında (hassas hex etiketleri). GuitarSet
~0.95 onset F1'e çıkmazsa sorun kurulumdadır.

Çalıştırma:
    python -m scripts.train.overfit_gaps --init tabcrnn_rep_valley.pt --files 8 --epochs 100
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from gtab.core.instrument import STANDARD_6
from gtab.data.gaps import gaps_files
from gtab.data.tab_labels import valley_frames
from gtab.data.torch_dataset import PitchSeq
from gtab.evaluation.pitch_eval import roll_to_notes
from gtab.models.inference import load_model
from gtab.models.losses import pitch_index, pitch_probs, pitch_onset_probs
from gtab.paths import CACHE_DIR
from gtab.utils import set_seed, get_device

INSTR = STANDARD_6
ONSET_THRS = (0.1, 0.2, 0.3, 0.5, 0.7)


def _f1(tp, fp, fn):
    return 2 * tp / max(2 * tp + fp + fn, 1)


def _head80(g, o):
    """(L,P) etiket + keskin onset -> notaların ilk %80'i (bool)."""
    m = np.zeros_like(g)
    for a, b, p in roll_to_notes(g, o, 0):
        m[a:a + max(1, int(np.ceil(0.8 * (b - a)))), p] = True
    return m


def evaluate(model, dl, idx, device):
    """-> dict(onset F1 ve eşiği, ilk %80 kare F1, tam kare F1, hızlı tekrar yakalama)"""
    model.eval()
    on = {h: np.zeros(3) for h in ONSET_THRS}; fast = {h: [0, 0] for h in ONSET_THRS}
    fr = np.zeros(3); fr80 = np.zeros(3)
    with torch.no_grad():
        for x, f, o, _, L in dl:
            tab, ol = model.forward_both(x.to(device))
            pf = pitch_probs(tab, idx).cpu().numpy(); po = pitch_onset_probs(tab, ol, idx).cpu().numpy()
            f, o = f.numpy() > 0.5, o.numpy() >= 0.999
            for b in range(len(L)):
                n = int(L[b]); g, ob, p = f[b, :n], o[b, :n], pf[b, :n] > 0.5
                fr += ((p & g).sum(), (p & ~g).sum(), (~p & g).sum())
                h80 = _head80(g, ob)
                fr80 += ((p & h80).sum(), (p & ~g).sum(), (~p & h80).sum())
                _, fo = valley_frames(g.astype(np.int64), ob)
                for h in ONSET_THRS:
                    q = po[b, :n] > h
                    on[h] += ((q & ob).sum(), (q & ~ob).sum(), (~q & ob).sum())
                    qd = q.copy(); qd[1:] |= q[:-1]; qd[:-1] |= q[1:]          # ±1 kare
                    fast[h][0] += int((qd & fo).sum()); fast[h][1] += int(fo.sum())
    h = max(ONSET_THRS, key=lambda h: _f1(*on[h]))
    return dict(on=_f1(*on[h]), h=h, fr80=_f1(*fr80), fr=_f1(*fr), fast=fast[h][0] / max(fast[h][1], 1),
                n_fast=fast[h][1])


def _train_no_dropout(model):
    """cuDNN LSTM geri yayılımı train modu ister: train(), ama Dropout + BatchNorm eval ve LSTM dropout 0."""
    model.train()
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.eval()
    model.lstm.dropout = 0.0


def run(name, files, init, epochs, lr, batch_size, eval_every, device, idx):
    set_seed(1)
    model, ck, _ = load_model("crnn", device, init)
    ds = PitchSeq(files, 200, onsets=True, onset_soft=ck.get("onset_soft"))
    dl, ev = DataLoader(ds, batch_size=batch_size, shuffle=True), DataLoader(ds, batch_size=batch_size)
    print(f"\n=== {name}: {len(files)} kayit ({len(ds)} parca) | {init} | lr {lr} | dropout kapali, BN eval ===")
    bce = torch.nn.BCELoss(reduction="none")
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def line(ep, lp=None, lo=None):
        r = evaluate(model, ev, idx, device)
        loss = f" | kayip perde {lp:.4f} onset {lo:.4f}" if lp is not None else " | (egitim oncesi taban)        "
        print(f"ep {ep:3d}{loss} | ONSET F1 {r['on']:.3f}@{r['h']} | kare F1 ilk%80 {r['fr80']:.3f} "
              f"(tam {r['fr']:.3f}) | hizli tekrar {r['fast']:.1%} (n={r['n_fast']})", flush=True)
        return r

    r0 = line(0)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        _train_no_dropout(model)                       # dropout kapalı, BN istatistikleri sabit
        sp = so = 0.0; n = 0
        for x, f, o, ow, L in dl:
            x, f, o, ow = x.to(device), f.to(device), o.to(device), ow.to(device)
            tab, ol = model.forward_both(x)
            mask = (torch.arange(f.shape[1], device=device)[None, :] < L.to(device)[:, None]).float()
            pp = pitch_probs(tab, idx).clamp(1e-6, 1 - 1e-6)
            po = pitch_onset_probs(tab, ol, idx).clamp(1e-6, 1 - 1e-6)
            lp = (bce(pp, f).mean(-1) * mask).sum() / mask.sum()
            lo = ((bce(po, o) * ow).mean(-1) * mask).sum() / mask.sum()
            opt.zero_grad(); (lp + lo).backward(); opt.step()
            sp += lp.item(); so += lo.item(); n += 1
        if ep % eval_every == 0 or ep == epochs:
            r = line(ep, sp / n, so / n)
    print(f"({time.time() - t0:.0f} sn)")
    return r0, r


def main(init, n_files, epochs, lr, batch_size, eval_every):
    device = get_device()
    idx = pitch_index(INSTR).to(device)
    gaps = gaps_files("gaps_train")[0][:n_files]
    gs = sorted(glob.glob(os.path.join(CACHE_DIR, "train", "*.npz")))[:n_files]
    res = {name: run(name, files, init, epochs, lr, batch_size, eval_every, device, idx)
           for name, files in (("KONTROL GuitarSet (train)", gs), ("GAPS (gaps_fit)", gaps))}
    (g0, g1), (a0, a1) = res["KONTROL GuitarSet (train)"], res["GAPS (gaps_fit)"]
    print("\n" + "=" * 92)
    print(f"{'':28s}{'onset F1 0 -> son':>22s}{'kare ilk%80 0 -> son':>24s}{'hizli tekrar 0 -> son':>24s}")
    for name, (r0, r1) in res.items():
        print(f"{name:28s}{r0['on']:>13.3f} -> {r1['on']:.3f}{r0['fr80']:>15.3f} -> {r1['fr80']:.3f}"
              f"{r0['fast']:>15.1%} -> {r1['fast']:.1%}")
    if g1["on"] < 0.95:
        verdict = "KONTROL ezberlemedi (GuitarSet onset < 0.95) -> sorun KURULUMDA; GAPS sonucu yorumlanmaz"
    elif a1["on"] >= 0.90:
        verdict = "GAPS de ezberleniyor -> etiketler ogrenilebilir; tam veride dusuk kalmasi KAPASITE -> 3a"
    elif a1["on"] < 0.80:
        verdict = "GuitarSet ezberleniyor, GAPS onset'leri takiliyor -> GAPS ONSET ETIKETLERI celisiyor (etiket tavani)"
    else:
        verdict = "ARADA -> belirsiz (kayip egrilerine birlikte bakalim)"
    print(f"\nSONUC: {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="tabcrnn_rep_valley.pt")
    ap.add_argument("--files", type=int, default=8, help="her veri setinden ezberlenecek kayit sayisi")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=10)
    a = ap.parse_args()
    main(a.init, a.files, a.epochs, a.lr, a.batch_size, a.eval_every)
