"""
Katman 3.9 / Adım 1 — Onset kafalı CRNN eğitimi (Onsets & Frames ilkesi).

Neden (README10, Adım 0): polifonide kaybın çoğu tel değil PERDE tarafında —
gerçek notaların %23'ü bulunamıyor, yanlış pozitiflerin %81'i hayalet nota,
tekrar eden notalar tek nota sayılıyor. Onset kafası "nota burada başlıyor"
bilgisini ayrıca öğrenir; çözümlemede nota yalnızca onset ile başlatılır.

Model: TabCRNNOnset = TabCRNN (gövde + tab kafası AYNI) + tel başına onset kafası.
--init ile TabCRNN checkpoint'inden (ör. tabcrnn_gaps.pt) başlanır; onset kafası sıfırdan.

Kayıplar:
  tab verisi (GuitarSet): tel CE + tel-onset BCE
  perde verisi (GAPS):    perde BCE (noisy-OR) + perde-onset BCE (noisy-OR, onset × tel olasılığı)

Model seçimi (doğrulama, test'e bakılmaz):
  skor = ortalama( GuitarSet val tab F1, GuitarSet val tel-onset F1,
                   GAPS-val kare perde F1, GAPS-val perde-onset F1 )
  Başlangıç (init) modeli de aday: hiçbir epoch geçemezse init ağırlıkları korunur.

Adım 2 (--harmonics): girişe harmonik istifleme (HarmonicStack) eklenir; init'in
giriş katmanı h=1 kanalına taşınır, diğer kanallar 0 -> başlangıçta init ile aynı çıktı.

Çalıştırma:
    # Adım 1: onset kafası
    python -m scripts.train.train_onset --init tabcrnn_gaps.pt --epochs 15 --ckpt tabcrnn_onset.pt
    # Adım 2: + harmonik istifleme (Adım 1'in sonucundan başlar)
    python -m scripts.train.train_onset --init tabcrnn_onset.pt --harmonics default --epochs 15 --ckpt tabcrnn_onset_h.pt
    # Katman 3.10 Adım 2: keskin onset + GuitarSet perde kaybı + kısa nota ağırlığı
    python -m scripts.train.train_onset --init tabcrnn_onset_h.pt --epochs 15 --ckpt tabcrnn_poly.pt         --onset-soft 0.3 --gs-pitch-weight 1.0 --short-frames 5 --short-weight 2.0
"""

import argparse

import torch
from torch.utils.data import DataLoader

from gtab.core.instrument import STANDARD_6
from gtab.data.gaps import gaps_files
from gtab.data.tab_labels import n_tab_classes
from gtab.data.torch_dataset import GuitarSetSeq, PitchSeq
from gtab.evaluation.metrics import prf, PAD
from gtab.models.losses import pitch_index, pitch_probs, pitch_onset_probs, tab_pitch_target
from gtab.models.nets import TabCRNNOnset, DEFAULT_HARMONICS, warm_start_state
from gtab.paths import CACHE_DIR, ckpt_path
from gtab.utils import set_seed, get_device

INSTR = STANDARD_6
CHUNK = 200


# ----------------------------------------------------------------- değerlendirme
def _f(tp, fp, fn):
    return prf(tp, fp, fn)[2]


ONSET_THRS = (0.1, 0.2, 0.3, 0.4, 0.5)


def _acc(t, key, pr, g):
    t[key][0] += int((pr & g).sum()); t[key][1] += int((pr & ~g).sum()); t[key][2] += int((~pr & g).sum())


def evaluate(model, gs_val, gp_val, idx, device, thr=0.5):
    """
    -> (dict skorlar, en iyi onset eşikleri)
    tab/kare F1 sabit eşikte; onset F1 seyrek bir dedektör olduğu için ONSET_THRS
    üstünde en iyisi alınır (seçilen eşik de raporlanır -> çözümlemede kullanılır).
    """
    model.eval()
    t = dict(tab=[0, 0, 0], gf=[0, 0, 0])
    t.update({("on", h): [0, 0, 0] for h in ONSET_THRS})
    t.update({("go", h): [0, 0, 0] for h in ONSET_THRS})
    with torch.no_grad():
        for x, y, on, _, L in gs_val:
            tab, ol = model.forward_both(x.to(device))
            pred = tab.argmax(-1).cpu(); po = torch.sigmoid(ol).cpu()
            valid = y[:, :, 0] != PAD
            p, g = pred[valid], y[valid]
            ap, ag = p > 0, g > 0
            tp = int(((p == g) & ap).sum())
            t["tab"][0] += tp; t["tab"][1] += int(ap.sum()) - tp; t["tab"][2] += int(ag.sum()) - tp
            po, go = po[valid], on[valid] > 0.5
            for h in ONSET_THRS:
                _acc(t, ("on", h), po > h, go)
        if gp_val is not None:
            for x, f, on, L in gp_val:
                tab, ol = model.forward_both(x.to(device))
                pf = pitch_probs(tab, idx).cpu() > thr
                pon = pitch_onset_probs(tab, ol, idx).cpu()
                mask = torch.arange(f.shape[1])[None, :] < L[:, None]
                _acc(t, "gf", pf[mask], f[mask] > 0.5)
                for h in ONSET_THRS:
                    _acc(t, ("go", h), pon[mask] > h, on[mask] > 0.5)
    out, best_thr = {"tab": _f(*t["tab"])}, {}
    keys = ("on", "gf", "go") if gp_val is not None else ("on",)
    for k in keys:
        if k == "gf":
            out[k] = _f(*t[k]); continue
        h = max(ONSET_THRS, key=lambda h: _f(*t[(k, h)]))
        out[k], best_thr[k] = _f(*t[(k, h)]), h
    return out, best_thr


# ----------------------------------------------------------------- eğitim
def train(init, tab_splits, pitch_splits, epochs=15, lr=3e-4, batch_size=16,
          onset_weight=1.0, pitch_weight=1.0, ckpt="tabcrnn_onset.pt", seed=1, pos_weight=3.0,
          harmonics=None, onset_soft=None, gs_pitch_weight=0.0, short_frames=0, short_weight=1.0):
    set_seed(seed)
    device = get_device()
    ckpt = ckpt_path(ckpt)
    ncls = n_tab_classes(INSTR)
    idx = pitch_index(INSTR).to(device)

    init_ck = torch.load(ckpt_path(init), map_location=device) if init else None
    if harmonics is None and init_ck is not None and init_ck.get("harmonics"):
        harmonics = tuple(init_ck["harmonics"])           # init'in giriş yapısını koru
    print(f"Katman 3.10 Adim 2 secenekleri: onset_soft={onset_soft} | GuitarSet perde kaybi agirligi="
          f"{gs_pitch_weight} | kisa nota: <{short_frames} kare x{short_weight}")
    tab_ds = GuitarSetSeq(CACHE_DIR, tab_splits, CHUNK, onsets=True, onset_soft=onset_soft,
                          short_frames=short_frames, short_weight=short_weight)
    tab_dl = DataLoader(tab_ds, batch_size=batch_size, shuffle=True)
    gs_val = DataLoader(GuitarSetSeq(CACHE_DIR, "val", CHUNK, onsets=True, onset_soft=onset_soft),
                        batch_size=batch_size)
    p_tr = gp_val = None
    if pitch_splits:
        tr_f, va_f = [], []
        for sp in pitch_splits:
            a, b = gaps_files(sp); tr_f += a; va_f += b
        p_tr = DataLoader(PitchSeq(tr_f, CHUNK, onsets=True, onset_soft=onset_soft),
                          batch_size=batch_size, shuffle=True)
        gp_val = DataLoader(PitchSeq(va_f, CHUNK, onsets=True, onset_soft=onset_soft), batch_size=batch_size)
        print(f"perde: {len(tr_f)} train / {len(va_f)} val kaydi (icraciya gore ayrik)")

    model = TabCRNNOnset(INSTR.num_strings, ncls, harmonics=harmonics).to(device)
    print(f"model: TabCRNNOnset | harmonikler: {harmonics or 'yok (tek kanal CQT)'}")
    if init:
        missing = warm_start_state(init_ck["model"], model)
        assert all(k.startswith("onset_head") for k in missing), missing
        print(f"baslangic: {init}" + (" (onset kafasi sifirdan)" if missing else "")
              + (" | giris katmani harmoniklere genisletildi (h=1 = eski agirlik)"
                 if harmonics and not init_ck.get("harmonics") else ""))

    w = tab_ds.class_weights(ncls, scheme="inv", cap=20.0).to(device)
    ce = torch.nn.CrossEntropyLoss(weight=w, ignore_index=PAD, reduction="none")
    bce = torch.nn.BCELoss(reduction="none")
    # seyrek onset pozitiflerini dengele (karelerin ~%5'i)
    bce_logit = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(pos_weight, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    state = {}

    def save():
        torch.save({"model": model.state_dict(), "n_classes": ncls, "onset": True,
                    "onset_thr": state.get("thr"), "harmonics": list(harmonics) if harmonics else None,
                    "onset_soft": onset_soft}, ckpt)

    def score(ep, logs):
        m, bt = evaluate(model, gs_val, gp_val, idx, device)
        state["cur_thr"] = bt.get("go", bt.get("on"))
        print(f"ep {ep:2d} | " + " ".join(f"{k} {v:.3f}" for k, v in logs.items()) + " | " +
              " ".join(f"{n} {m[k]:.3f}" + (f"@{bt[k]}" if k in bt else "")
                       for k, n in (("tab", "GS-tabF1"), ("on", "GS-onsetF1"),
                                    ("gf", "GAPS-kareF1"), ("go", "GAPS-onsetF1")) if k in m))
        return sum(m.values()) / len(m)

    best = score(0, {}); state["thr"] = state["cur_thr"]; save()
    for ep in range(1, epochs + 1):
        model.train()
        sums = dict(tab=0.0, on=0.0, gsperde=0.0, perde=0.0, ponset=0.0); n = 0
        p_iter = iter(p_tr) if p_tr is not None else None
        for x, y, on, wt, _ in tab_dl:
            x, y, on, wt = x.to(device), y.to(device), on.to(device), wt.to(device)
            tab, ol = model.forward_both(x)
            B, L, S, C = tab.shape
            # tel CE: sınıf ağırlığı x kısa nota ağırlığı (wt=1 -> eski ağırlıklı ortalama, birebir)
            yv = y.reshape(-1); cw = w[yv.clamp(min=0)] * (yv != PAD)
            l_tab = (ce(tab.reshape(-1, C), yv) * wt.reshape(-1)).sum() / (cw * wt.reshape(-1)).sum()
            valid = (y[:, :, 0] != PAD).unsqueeze(-1).float()
            l_on = (bce_logit(ol, on) * valid * wt).sum() / (valid.sum() * S)
            loss = l_tab + onset_weight * l_on
            sums["tab"] += l_tab.item(); sums["on"] += l_on.item()
            if gs_pitch_weight > 0:                    # GuitarSet'e doğrudan perde kaybı
                pt = tab_pitch_target(y, idx)
                pp_gs = pitch_probs(tab, idx).clamp(1e-6, 1 - 1e-6)
                fm = valid.squeeze(-1)
                l_gp = (bce(pp_gs, pt).mean(-1) * fm).sum() / fm.sum()
                loss = loss + gs_pitch_weight * l_gp
                sums["gsperde"] += l_gp.item()
            if p_iter is not None:
                try:
                    xg, fg, og, Lg = next(p_iter)
                except StopIteration:
                    p_iter = iter(p_tr); xg, fg, og, Lg = next(p_iter)
                xg, fg, og = xg.to(device), fg.to(device), og.to(device)
                tg, olg = model.forward_both(xg)
                mask = (torch.arange(fg.shape[1], device=device)[None, :] < Lg.to(device)[:, None]).float()
                pp = pitch_probs(tg, idx).clamp(1e-6, 1 - 1e-6)
                po = pitch_onset_probs(tg, olg, idx).clamp(1e-6, 1 - 1e-6)
                l_p = (bce(pp, fg).mean(-1) * mask).sum() / mask.sum()
                l_po = (bce(po, og).mean(-1) * mask).sum() / mask.sum()
                loss = loss + pitch_weight * (l_p + onset_weight * l_po)
                sums["perde"] += l_p.item(); sums["ponset"] += l_po.item()
            opt.zero_grad(); loss.backward(); opt.step(); n += 1
        sched.step()
        s = score(ep, {k: v / max(n, 1) for k, v in sums.items() if v})
        if s > best:
            best = s; state["thr"] = state["cur_thr"]; save()
    print(f"En iyi secim skoru: {best:.3f} -> {ckpt} | dogrulamada secilen onset esigi: {state['thr']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="tabcrnn_gaps.pt")
    ap.add_argument("--tab-splits", default="train,train_comp")
    ap.add_argument("--pitch-splits", default="gaps_train")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--onset-weight", type=float, default=1.0)
    ap.add_argument("--pitch-weight", type=float, default=1.0)
    ap.add_argument("--ckpt", default="tabcrnn_onset.pt")
    ap.add_argument("--pos-weight", type=float, default=3.0, help="onset BCE pozitif agirligi")
    ap.add_argument("--harmonics", default="", help="ornek '0.5,1,2,3,4,5' ya da 'default'; bos = init'teki ayar")
    # Katman 3.10 Adim 2 (varsayilanlar = Katman 3.9 egitimi, birebir)
    ap.add_argument("--onset-soft", type=float, default=None,
                    help="keskin onset hedefi: tepe karesi 1, sonraki kare bu deger (or. 0.3); bos = eski dilate=2")
    ap.add_argument("--gs-pitch-weight", type=float, default=0.0, help="GuitarSet'e dogrudan perde (noisy-OR) BCE agirligi")
    ap.add_argument("--short-frames", type=int, default=0, help="bu kareden kisa notalar agirlikli (0 = kapali)")
    ap.add_argument("--short-weight", type=float, default=1.0, help="kisa nota agirligi")
    a = ap.parse_args()
    sp = lambda s: [x.strip() for x in s.split(",") if x.strip()]
    train(a.init or None, sp(a.tab_splits), sp(a.pitch_splits), a.epochs, a.lr, a.batch_size,
          a.onset_weight, a.pitch_weight, a.ckpt, pos_weight=a.pos_weight,
          harmonics=(DEFAULT_HARMONICS if a.harmonics == "default"
                     else tuple(float(h) if "." in h else int(h) for h in a.harmonics.split(",")) if a.harmonics else None),
          onset_soft=a.onset_soft, gs_pitch_weight=a.gs_pitch_weight,
          short_frames=a.short_frames, short_weight=a.short_weight)
