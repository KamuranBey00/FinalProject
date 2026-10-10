"""
Katman 3.13 Adım 2b — "Klasik edisyon önerisi" dizi modeli (EditionNet) eğitimi. Sembolik, ses yok, dakikalar.

Veri: build_gaps_tab etiketleri, YALNIZ gaps_fit; seçim gaps_val'de (tel uyumu + parmak doğruluğu ortalaması).
Kayıp: tel CE (partisyonda etiketli notalar) + parmak CE (parmak numaralı basılı notalar; aşama 2'ye konum olarak
partisyon konumu, etiketsiz notalarda modelin kendi tahmini verilir).
Çıktı: checkpoints/edition_seq.pt

Çalıştırma:
    python -m scripts.train.train_edition_seq
"""

import argparse
import os

import numpy as np
import torch

from gtab.data.gaps import GAPS_TAB_DIR, gaps_files
from gtab.decoding.edition import label_sequence
from gtab.models.edition_net import INSTR, EditionNet, note_features, position_features, predict
from gtab.paths import ckpt_path
from gtab.utils import get_device, set_seed

CHUNK = 256


def load(files):
    out = []
    for f in files:
        lf = os.path.join(GAPS_TAB_DIR, os.path.basename(f))
        if os.path.exists(lf):
            notes, chord, s, fr, g = label_sequence(lf)
            pidx, cont, valid = note_features(notes, chord)
            out.append(dict(notes=notes, chord=chord, s=s, fr=fr, g=g, pidx=pidx, cont=cont, valid=valid))
    return out


def batches(data, bs, rng):
    """Rastgele 256 notalık kesitler (her epoch tüm notalar ~bir kez)."""
    items = [(k, a) for k, d in enumerate(data) for a in range(0, len(d["notes"]), CHUNK)]
    rng.shuffle(items)
    for i in range(0, len(items), bs):
        yield [(data[k], min(a + rng.integers(0, CHUNK // 2), max(0, len(data[k]["notes"]) - CHUNK)) if a else 0)
               for k, a in items[i:i + bs]]


def collate(chunk, net, device):
    B = len(chunk); L = max(min(CHUNK, len(d["notes"]) - a) for d, a in chunk)
    pidx = np.zeros((B, L), np.int64); cont = np.zeros((B, L, 12), np.float32); valid = np.ones((B, L, 6), bool)
    ys = np.full((B, L), -100, np.int64); yg = np.full((B, L), -100, np.int64); sel = []
    for b, (d, a) in enumerate(chunk):
        n = min(CHUNK, len(d["notes"]) - a); sl = slice(a, a + n)
        pidx[b, :n], cont[b, :n], valid[b, :n] = d["pidx"][sl], d["cont"][sl], d["valid"][sl]
        s, g, fr = d["s"][sl], d["g"][sl], d["fr"][sl]
        ys[b, :n] = np.where(s >= 0, s, -100)
        yg[b, :n] = np.where((s >= 0) & (g >= 1) & (fr > 0), g, -100)
        sel.append((d, sl, n))
    T = lambda a: torch.as_tensor(a, device=device)
    pidx, cont, valid = T(pidx), T(cont), T(valid)
    lp = net.strings(pidx, cont, valid)
    # aşama 2 girdisi: partisyon konumu, etiketsiz notada modelin tahmini (gradyan yok)
    pred = lp.detach().argmax(-1).cpu().numpy()
    posf = np.zeros((B, L, 6 + 25 + 1), np.float32); fret0 = np.zeros((B, L), bool)
    for b, (d, sl, n) in enumerate(sel):
        st = np.where(d["s"][sl] >= 0, d["s"][sl], np.where(d["valid"][sl].any(1), pred[b, :n], -1))
        posf[b, :n] = position_features(st, d["notes"][sl])
        fret0[b, :n] = (st >= 0) & (d["notes"][sl][:, 2].astype(int) - np.array(INSTR.tuning)[st] == 0)
    lf = net.fingers(pidx, cont, T(posf), T(fret0))
    nll = torch.nn.NLLLoss(ignore_index=-100)
    return nll(lp.reshape(-1, 6), T(ys).reshape(-1)), nll(lf.reshape(-1, 5), T(yg).reshape(-1))


def evaluate(net, data, device):
    """-> (tel uyumu, parmak doğruluğu (partisyon konumu verilerek)) — eval_edition ile aynı tanım."""
    so = sn = fo = fn = 0
    for d in data:
        s, _, _ = predict(net, d["notes"], d["chord"], device)
        lab = d["s"] >= 0
        so += int((s[lab] == d["s"][lab]).sum()); sn += int(lab.sum())
        _, _, g = predict(net, d["notes"], d["chord"], device, fixed=d["s"])
        m = lab & (d["fr"] > 0) & (d["g"] >= 1)
        fo += int((g[m] == d["g"][m]).sum()); fn += int(m.sum())
    return so / max(sn, 1), fo / max(fn, 1)


def main(out, epochs, lr, bs, seed):
    set_seed(seed); device = get_device(); rng = np.random.default_rng(seed)
    fit_f, val_f = gaps_files("gaps_train")
    fit, val = load(fit_f), load(val_f)
    print(f"gaps_fit {len(fit)} kayit ({sum(len(d['notes']) for d in fit)} nota) | gaps_val {len(val)} kayit | {device}")
    net = EditionNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best = -1
    for ep in range(1, epochs + 1):
        net.train(); ls = lg = n = 0
        for chunk in batches(fit, bs, rng):
            l_s, l_g = collate(chunk, net, device)
            opt.zero_grad(); (l_s + l_g).backward(); opt.step()
            ls += l_s.item(); lg += l_g.item(); n += 1
        a_s, a_g = evaluate(net, val, device)
        sc = (a_s + a_g) / 2
        mark = ""
        if sc > best:
            best = sc; mark = "  <- kaydedildi"
            torch.save({"model": net.state_dict()}, ckpt_path(out))
        print(f"ep {ep:2d} | kayip tel {ls / n:.3f} parmak {lg / n:.3f} | gaps_val tel uyumu {a_s:.3f} parmak {a_g:.3f}{mark}",
              flush=True)
    print(f"En iyi gaps_val skoru {best:.3f} -> {ckpt_path(out)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="edition_seq.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    main(a.out, a.epochs, a.lr, a.batch_size, a.seed)
