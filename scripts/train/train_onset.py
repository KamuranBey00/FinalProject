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
    # Katman 3.10 Adim 3c: hizli tekrar onset agirligi + GAPS perde agirligi (tabcrnn_poly'den)
    python -m scripts.train.train_onset --init tabcrnn_poly.pt --epochs 15 --ckpt tabcrnn_rep.pt \
        --onset-soft 0.3 --gs-pitch-weight 1.0 --short-frames 5 --short-weight 2.0 --repeat-weight 3 --pitch-weight 2
    # Katman 3.11 Adim 4: vadi hedefi + fragman negatifleri + hizli tekrar ornekleme (tabcrnn_rep_off'tan)
    python -m scripts.train.train_onset --init tabcrnn_rep_off.pt --epochs 15 --ckpt tabcrnn_rep_valley.pt \
        --onset-soft 0.3 --gs-pitch-weight 1.0 --short-frames 5 --short-weight 2.0 --repeat-weight 3 --pitch-weight 2 \
        --offset-weight 1 --valley-weight 5 --frag-weight 3 --oversample 3
"""

import argparse

import numpy as np

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from gtab.core.instrument import STANDARD_6
from gtab.data.gaps import GAPS_TAB_DIR, gaps_files
from gtab.data.tab_labels import n_tab_classes, fragment_frames
from gtab.decoding.viterbi import pitch_onset_matrix
from gtab.data.torch_dataset import GuitarSetSeq, PitchSeq
from gtab.evaluation.metrics import prf, PAD
from gtab.models.losses import pitch_index, pitch_probs, pitch_onset_probs, tab_pitch_target, \
    tab_pitch_onset_target, tab_pitch_weight, \
    string_offset_target, pitch_offset_target
from gtab.models.inference import load_model, predict_heads
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
    has_pon = getattr(model, "pitch_onset_head", None) is not None
    t = dict(tab=[0, 0, 0], gf=[0, 0, 0], gtab=[0, 0, 0])
    t.update({(k, h): [0, 0, 0] for k in ("on", "go", "pon", "gpo") for h in ONSET_THRS})
    with torch.no_grad():
        for x, y, on, _, _, L in gs_val:
            tab, ol, _, pl = model.forward_full(x.to(device))
            if has_pon:                                # Katman 3.12: perde-onset kafası (GuitarSet)
                valid_ = (y[:, :, 0] != PAD)
                pt_ = tab_pitch_onset_target(y, on, idx.cpu())[valid_] > 0.5
                pp_ = torch.sigmoid(pl).cpu()[valid_]
                for h in ONSET_THRS:
                    _acc(t, ("pon", h), pp_ > h, pt_)
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
            for x, f, on, _, L, *yt in gp_val:
                tab, ol, _, pl = model.forward_full(x.to(device))
                if yt:                                 # Katman 3.13: partisyon TAB'ı (bilinen kareler)
                    vt = yt[0][:, :, 0] != PAD
                    p, g = tab.argmax(-1).cpu()[vt], yt[0][vt]
                    ap, ag = p > 0, g > 0; tp = int(((p == g) & ap).sum())
                    t["gtab"][0] += tp; t["gtab"][1] += int(ap.sum()) - tp; t["gtab"][2] += int(ag.sum()) - tp
                pf = pitch_probs(tab, idx).cpu() > thr
                pon = pitch_onset_probs(tab, ol, idx).cpu()
                mask = torch.arange(f.shape[1])[None, :] < L[:, None]
                _acc(t, "gf", pf[mask], f[mask] > 0.5)
                for h in ONSET_THRS:
                    _acc(t, ("go", h), pon[mask] > h, on[mask] > 0.5)
                if has_pon:                            # Katman 3.12: perde-onset kafası (GAPS)
                    pp_ = torch.sigmoid(pl).cpu()[mask]
                    for h in ONSET_THRS:
                        _acc(t, ("gpo", h), pp_ > h, on[mask] > 0.5)
    out, best_thr = {"tab": _f(*t["tab"])}, {}
    if sum(t["gtab"]):
        out["gtab"] = _f(*t["gtab"])
    keys = ("on", "gf", "go") if gp_val is not None else ("on",)
    if has_pon:
        keys += ("pon", "gpo") if gp_val is not None else ("pon",)
    for k in keys:
        if k == "gf":
            out[k] = _f(*t[k]); continue
        h = max(ONSET_THRS, key=lambda h: _f(*t[(k, h)]))
        out[k], best_thr[k] = _f(*t[(k, h)]), h
    return out, best_thr


# ----------------------------------------------------------------- Katman 3.11 Adım 4
def fragment_fns(init, device):
    """init modelin eğitim verisinde sahte bölünme ürettiği yerler (bir kez, eğitim başında)."""
    m, _, _ = load_model("crnn", device, init)
    def gs(path, cqt, tab, hard):                 # tel düzeyi: tel onset eğrisi
        _, ons, _ = predict_heads(m, np.asarray(cqt, np.float32), device)
        return fragment_frames(ons, tab, hard)
    def gaps(path, cqt, fr, on):                  # perde düzeyi: noisy-OR perde-onset eğrisi
        probs, ons, _ = predict_heads(m, np.asarray(cqt, np.float32), device)
        return fragment_frames(pitch_onset_matrix(probs, ons, INSTR, "noisyor"), fr, on)
    return gs, gaps


def make_loader(ds, batch_size, oversample):
    """oversample > 1: hızlı tekrar içeren parçalar bu kat daha sık örneklenir (epoch boyu aynı)."""
    if oversample <= 1:
        return DataLoader(ds, batch_size=batch_size, shuffle=True)
    w = [oversample if f else 1.0 for f in ds.fast]
    print(f"  ornekleme: {sum(ds.fast)}/{len(ds)} parca hizli tekrar iceriyor (x{oversample})")
    return DataLoader(ds, batch_size=batch_size, sampler=WeightedRandomSampler(w, len(ds), replacement=True))


# ----------------------------------------------------------------- eğitim
def train(init, tab_splits, pitch_splits, epochs=15, lr=3e-4, batch_size=16,
          onset_weight=1.0, pitch_weight=1.0, ckpt="tabcrnn_onset.pt", seed=1, pos_weight=3.0,
          harmonics=None, onset_soft=None, gs_pitch_weight=0.0, short_frames=0, short_weight=1.0,
          repeat_weight=1.0, repeat_gap=3, offset_weight=0.0, valley_weight=1.0, frag_weight=1.0, oversample=1.0, deep=0,
          pitch_onset_weight=0.0, new_lr_mult=1.0, gaps_tab_weight=0.0):
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
    print(f"Katman 3.10 Adim 3c: tekrar onset agirligi x{repeat_weight} (onceki nota <= {repeat_gap} kare once)"
          f" | GAPS perde agirligi {pitch_weight} | offset kafasi agirligi {offset_weight}")
    print(f"Katman 3.11 Adim 4: vadi agirligi x{valley_weight} | fragman agirligi x{frag_weight} | "
          f"hizli tekrar parcasi ornekleme x{oversample}")
    gs_frag = gaps_frag = None
    if frag_weight != 1.0:
        print(f"  fragman taramasi: {init} egitim verisinde (bir kez)")
        gs_frag, gaps_frag = fragment_fns(init, device)
    tab_ds = GuitarSetSeq(CACHE_DIR, tab_splits, CHUNK, onsets=True, onset_soft=onset_soft,
                          short_frames=short_frames, short_weight=short_weight,
                          repeat_weight=repeat_weight, repeat_gap=repeat_gap,
                          valley_weight=valley_weight, frag_weight=frag_weight, frag_fn=gs_frag)
    tab_dl = make_loader(tab_ds, batch_size, oversample)
    gs_val = DataLoader(GuitarSetSeq(CACHE_DIR, "val", CHUNK, onsets=True, onset_soft=onset_soft),
                        batch_size=batch_size)
    p_tr = gp_val = None
    if pitch_splits:
        tr_f, va_f = [], []
        for sp in pitch_splits:
            a, b = gaps_files(sp); tr_f += a; va_f += b
        td = GAPS_TAB_DIR if gaps_tab_weight > 0 else None    # Katman 3.13: partisyon TAB'ı
        p_tr = make_loader(PitchSeq(tr_f, CHUNK, onsets=True, onset_soft=onset_soft,
                                    repeat_weight=repeat_weight, repeat_gap=repeat_gap,
                                    valley_weight=valley_weight, frag_weight=frag_weight, frag_fn=gaps_frag,
                                    tab_dir=td),
                           batch_size, oversample)
        gp_val = DataLoader(PitchSeq(va_f, CHUNK, onsets=True, onset_soft=onset_soft, tab_dir=td),
                            batch_size=batch_size)
        print(f"perde: {len(tr_f)} train / {len(va_f)} val kaydi (icraciya gore ayrik)"
              + (f" | partisyon TAB kaybi x{gaps_tab_weight} ({td})" if td else ""))

    use_off = offset_weight > 0 or bool(init_ck and init_ck.get("offset"))
    deep = max(deep, int(init_ck.get("deep") or 0) if init_ck else 0)
    use_pon = pitch_onset_weight > 0 or bool(init_ck and init_ck.get("pitch_onset"))
    model = TabCRNNOnset(INSTR.num_strings, ncls, harmonics=harmonics, offset=use_off,
                         deep=deep, pitch_onset=use_pon).to(device)
    print(f"model: TabCRNNOnset | harmonikler: {harmonics or 'yok (tek kanal CQT)'} | "
          f"Katman 3.12: artik blok {deep} | perde-onset kafasi {use_pon} (kayip x{pitch_onset_weight}) | "
          f"yeni katman lr x{new_lr_mult}")
    missing = []
    if init:
        missing = warm_start_state(init_ck["model"], model)
        assert all(k.startswith(("onset_head", "offset_head", "res.", "pitch_onset_head")) for k in missing), missing
        print(f"baslangic: {init}" + (f" (sifirdan: {sorted({k.split('.')[0] for k in missing})})" if missing else "")
              + (" | giris katmani harmoniklere genisletildi (h=1 = eski agirlik)"
                 if harmonics and not init_ck.get("harmonics") else ""))

    w = tab_ds.class_weights(ncls, scheme="inv", cap=20.0).to(device)
    ce = torch.nn.CrossEntropyLoss(weight=w, ignore_index=PAD, reduction="none")
    bce = torch.nn.BCELoss(reduction="none")
    # seyrek onset pozitiflerini dengele (karelerin ~%5'i)
    bce_logit = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(pos_weight, device=device))
    # Katman 3.12: init'te olmayan (sıfırdan) katmanlar ayrı grupta, lr x new_lr_mult
    new_p = [p for n_, p in model.named_parameters() if n_ in set(missing)]
    old_p = [p for n_, p in model.named_parameters() if n_ not in set(missing)]
    opt = torch.optim.Adam([{"params": old_p, "lr": lr}] +
                           ([{"params": new_p, "lr": lr * new_lr_mult}] if new_p else []))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    state = {}

    def save():
        torch.save({"model": model.state_dict(), "n_classes": ncls, "onset": True, "offset": use_off,
                    "deep": deep, "pitch_onset": use_pon,
                    "onset_thr": state.get("thr"), "harmonics": list(harmonics) if harmonics else None,
                    "onset_soft": onset_soft}, ckpt)

    def score(ep, logs):
        m, bt = evaluate(model, gs_val, gp_val, idx, device)
        state["cur_thr"] = bt.get("go", bt.get("on"))
        print(f"ep {ep:2d} | " + " ".join(f"{k} {v:.3f}" for k, v in logs.items()) + " | " +
              " ".join(f"{n} {m[k]:.3f}" + (f"@{bt[k]}" if k in bt else "")
                       for k, n in (("tab", "GS-tabF1"), ("on", "GS-onsetF1"), ("pon", "GS-perdeOnsetF1(kafa)"),
                                    ("gf", "GAPS-kareF1"), ("go", "GAPS-onsetF1(tel)"),
                                    ("gpo", "GAPS-onsetF1(kafa)"), ("gtab", "GAPS-tabF1(partisyon)")) if k in m))
        # seçim skoru: eski 4 ölçü; GAPS onset'te iki yoldan iyisi (kafa yoksa eskisiyle birebir)
        sel = {k: m[k] for k in ("tab", "on", "gf", "go") if k in m}
        if "gpo" in m:
            sel["go"] = max(m["go"], m["gpo"])
        if "gtab" in m:                                # Katman 3.13: GAPS tel ölçüsü de seçime girer
            sel["gtab"] = m["gtab"]
        return sum(sel.values()) / len(sel)

    best = score(0, {}); state["thr"] = state["cur_thr"]; save()
    for ep in range(1, epochs + 1):
        model.train()
        sums = dict(tab=0.0, on=0.0, off=0.0, gsperde=0.0, perde=0.0, ponset=0.0, poffset=0.0); n = 0
        p_iter = iter(p_tr) if p_tr is not None else None
        for x, y, on, wt, ow, _ in tab_dl:
            x, y, on, wt, ow = x.to(device), y.to(device), on.to(device), wt.to(device), ow.to(device)
            tab, ol, offl, pl = model.forward_full(x)
            B, L, S, C = tab.shape
            # tel CE: sınıf ağırlığı x kısa nota ağırlığı (wt=1 -> eski ağırlıklı ortalama, birebir)
            yv = y.reshape(-1); cw = w[yv.clamp(min=0)] * (yv != PAD)
            l_tab = (ce(tab.reshape(-1, C), yv) * wt.reshape(-1)).sum() / (cw * wt.reshape(-1)).sum()
            valid = (y[:, :, 0] != PAD).unsqueeze(-1).float()
            l_on = (bce_logit(ol, on) * valid * ow).sum() / (valid.sum() * S)   # ow = kısa x tekrar (varsayılan = wt)
            loss = l_tab + onset_weight * l_on
            sums["tab"] += l_tab.item(); sums["on"] += l_on.item()
            if offset_weight > 0:                      # Adım 3c: tel offset'i (son kare chunk sınırı -> maske)
                ot_, om_ = string_offset_target(y, on, onset_soft)
                vm = valid * om_.unsqueeze(-1)
                l_off = (bce_logit(offl, ot_) * vm).sum() / (vm.sum() * S).clamp(min=1)
                loss = loss + offset_weight * l_off
                sums["off"] += l_off.item()
            if pitch_onset_weight > 0:                 # Katman 3.12: perde-onset kafası (GuitarSet)
                pot_ = tab_pitch_onset_target(y, on, idx)
                wp_ = tab_pitch_weight(y, ow, idx)
                l_pon = (bce_logit(pl, pot_) * valid * wp_).sum() / (valid.sum() * pot_.shape[-1])
                loss = loss + pitch_onset_weight * l_pon
                sums["pon"] = sums.get("pon", 0.0) + l_pon.item()
            if gs_pitch_weight > 0:                    # GuitarSet'e doğrudan perde kaybı
                pt = tab_pitch_target(y, idx)
                pp_gs = pitch_probs(tab, idx).clamp(1e-6, 1 - 1e-6)
                fm = valid.squeeze(-1)
                l_gp = (bce(pp_gs, pt).mean(-1) * fm).sum() / fm.sum()
                loss = loss + gs_pitch_weight * l_gp
                sums["gsperde"] += l_gp.item()
            if p_iter is not None:
                try:
                    bg = next(p_iter)
                except StopIteration:
                    p_iter = iter(p_tr); bg = next(p_iter)
                xg, fg, og, owg, Lg = bg[:5]
                xg, fg, og, owg = xg.to(device), fg.to(device), og.to(device), owg.to(device)
                tg, olg, offg, plg = model.forward_full(xg)
                mask = (torch.arange(fg.shape[1], device=device)[None, :] < Lg.to(device)[:, None]).float()
                pp = pitch_probs(tg, idx).clamp(1e-6, 1 - 1e-6)
                po = pitch_onset_probs(tg, olg, idx).clamp(1e-6, 1 - 1e-6)
                l_p = (bce(pp, fg).mean(-1) * mask).sum() / mask.sum()
                l_po = ((bce(po, og) * owg).mean(-1) * mask).sum() / mask.sum()
                loss = loss + pitch_weight * (l_p + onset_weight * l_po)
                sums["perde"] += l_p.item(); sums["ponset"] += l_po.item()
                if gaps_tab_weight > 0:                # Katman 3.13: partisyon TAB'ı, yalnız bilinen kareler
                    yg = bg[5].to(device).reshape(-1); cwg = (w[yg.clamp(min=0)] * (yg != PAD)).sum()
                    if cwg > 0:
                        l_gt = ce(tg.reshape(-1, C), yg).sum() / cwg
                        loss = loss + gaps_tab_weight * l_gt
                        sums["gtab"] = sums.get("gtab", 0.0) + l_gt.item()
                if pitch_onset_weight > 0:             # Katman 3.12: perde-onset kafası (GAPS, doğrudan)
                    l_gpon = ((bce_logit(plg, og) * owg).mean(-1) * mask).sum() / mask.sum()
                    loss = loss + pitch_weight * pitch_onset_weight * l_gpon
                    sums["gpon"] = sums.get("gpon", 0.0) + l_gpon.item()
                if offset_weight > 0:                  # Adım 3c: perde offset'i (noisy-OR)
                    pot, pom = pitch_offset_target(fg, og, onset_soft)
                    pof = pitch_onset_probs(tg, offg, idx).clamp(1e-6, 1 - 1e-6)
                    mk = mask * pom
                    l_pof = (bce(pof, pot).mean(-1) * mk).sum() / mk.sum().clamp(min=1)
                    loss = loss + pitch_weight * offset_weight * l_pof
                    sums["poffset"] += l_pof.item()
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
    # Katman 3.10 Adim 3c (varsayilan 1 = Adim 2 egitimi, birebir)
    ap.add_argument("--repeat-weight", type=float, default=1.0, help="hizli ayni perde tekrar onset'lerinin kayip agirligi")
    ap.add_argument("--repeat-gap", type=int, default=3, help="onceki nota bu kadar kare icinde bittiyse tekrar sayilir")
    ap.add_argument("--offset-weight", type=float, default=0.0, help="offset kafasi kayip agirligi (0 = kafa yok)")
    # Katman 3.11 Adim 4 (varsayilan 1 = onceki egitim, birebir)
    ap.add_argument("--valley-weight", type=float, default=1.0, help="hizli tekrar vadisine 'vurus yok' agirligi")
    ap.add_argument("--frag-weight", type=float, default=1.0, help="init modelin sahte bolunme yerlerine 'vurus yok' agirligi")
    ap.add_argument("--oversample", type=float, default=1.0, help="hizli tekrar iceren parcalari bu kat sik ornekle")
    # Katman 3.12 Adim 3a (varsayilan = onceki egitim, birebir)
    ap.add_argument("--deep", type=int, default=0, help="CNN govdesine sifirla baslatilan artik blok sayisi")
    ap.add_argument("--pitch-onset-weight", type=float, default=0.0, help="dogrudan perde-onset kafasi kayip agirligi (0 = yok)")
    ap.add_argument("--new-lr-mult", type=float, default=1.0, help="init'te olmayan yeni katmanlarin lr carpani")
    ap.add_argument("--gaps-tab-weight", type=float, default=0.0,
                    help="Katman 3.13: GAPS partisyon TAB'i tel kaybi agirligi (0 = kapali; build_gaps_tab gerekir)")
    a = ap.parse_args()
    sp = lambda s: [x.strip() for x in s.split(",") if x.strip()]
    train(a.init or None, sp(a.tab_splits), sp(a.pitch_splits), a.epochs, a.lr, a.batch_size,
          a.onset_weight, a.pitch_weight, a.ckpt, pos_weight=a.pos_weight,
          harmonics=(DEFAULT_HARMONICS if a.harmonics == "default"
                     else tuple(float(h) if "." in h else int(h) for h in a.harmonics.split(",")) if a.harmonics else None),
          onset_soft=a.onset_soft, gs_pitch_weight=a.gs_pitch_weight,
          short_frames=a.short_frames, short_weight=a.short_weight,
          repeat_weight=a.repeat_weight, repeat_gap=a.repeat_gap, offset_weight=a.offset_weight,
          valley_weight=a.valley_weight, frag_weight=a.frag_weight, oversample=a.oversample,
          deep=a.deep, pitch_onset_weight=a.pitch_onset_weight, new_lr_mult=a.new_lr_mult,
          gaps_tab_weight=a.gaps_tab_weight)
