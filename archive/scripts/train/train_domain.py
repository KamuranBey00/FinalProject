"""
Katman 3.8c/d — Alan uyarlaması: karışık denetimli CRNN eğitimi.

İki tür veri, TEK model (TabCRNN, mimari değişmez):
  - TAB etiketli  (GuitarSet, SynthTab): tel başına cross-entropy (train_crnn ile aynı)
  - PERDE etiketli (GAPS, klasik gitar):  tel olasılıklarından türetilen perde
        P(p) = 1 - Π_tel (1 - P_tel(fret = p - akort_tel))
    üstünde BCE. Model teli kendi seçer; sadece "hangi perde çaldı" denetlenir.

Batch'ler sırayla: bir tab batch'i, bir perde batch'i. Epoch = tab loader uzunluğu.

Model seçimi (--select):
  gaps      -> GAPS doğrulama parçasında kare perde F1 (GAPS TEST'E HİÇ BAKILMAZ)
  guitarset -> GuitarSet 05 solo val tab F1 (train_crnn ile aynı)
GAPS doğrulama parçası: gaps_train içinden icracıya göre ayrılır (her 8 icracıdan
biri, deterministik) -> doğrulama da performer-disjoint.

Checkpoint adları checkpoints/ altına kaydedilir/oradan okunur.

Çalıştırma (README9):
    # 3.8c: GuitarSet modelinden GAPS'e ince ayar
    python -m scripts.train.train_domain --init tabcrnn_comp.pt --tab-splits train,train_comp \
        --pitch-splits gaps_train --epochs 15 --lr 3e-4 --ckpt tabcrnn_gaps.pt
    # 3.8d: SynthTab ön-eğitim (sadece tab)
    python -m scripts.train.train_domain --tab-splits synthtab_train --pitch-splits "" \
        --select guitarset --epochs 10 --ckpt tabcrnn_syn.pt
"""

import argparse

import torch
from torch.utils.data import DataLoader

from gtab.core.instrument import STANDARD_6
from gtab.data.torch_dataset import GuitarSetSeq, PitchSeq
from gtab.data.tab_labels import n_tab_classes
from gtab.data.gaps import gaps_files
from gtab.evaluation.metrics import evaluate_tab_seq, eval_pitch_frames
from gtab.models.nets import TabCRNN
from gtab.models.losses import pitch_index, pitch_probs
from gtab.paths import CACHE_DIR, ckpt_path
from gtab.utils import set_seed, get_device

INSTR = STANDARD_6
PAD = GuitarSetSeq.PAD
CHUNK = 200


# ----------------------------------------------------------------- eğitim
def train(tab_splits, pitch_splits, init=None, epochs=15, lr=3e-4, batch_size=16,
          pitch_weight=1.0, select="gaps", ckpt="tabcrnn_domain.pt", seed=1,
          weight_scheme="inv", weight_cap=20.0):
    set_seed(seed)
    device = get_device()
    ckpt = ckpt_path(ckpt)
    ncls = n_tab_classes(INSTR)
    idx = pitch_index(INSTR).to(device)

    tab_ds = GuitarSetSeq(CACHE_DIR, tab_splits, CHUNK) if tab_splits else None
    gs_val = DataLoader(GuitarSetSeq(CACHE_DIR, "val", CHUNK), batch_size=batch_size)

    p_tr = p_va = None
    if pitch_splits:
        tr_f, va_f = [], []
        for sp in pitch_splits:
            a, b = gaps_files(sp)
            tr_f += a; va_f += b
        p_tr = DataLoader(PitchSeq(tr_f), batch_size=batch_size, shuffle=True)
        p_va = DataLoader(PitchSeq(va_f), batch_size=batch_size)
        print(f"perde: {len(tr_f)} train / {len(va_f)} val kaydi (icraciya gore ayrik)")
    if select == "gaps" and p_va is None:
        raise ValueError("--select gaps icin --pitch-splits gerekli")

    model = TabCRNN(INSTR.num_strings, ncls).to(device)
    if init:
        model.load_state_dict(torch.load(ckpt_path(init), map_location=device)["model"])
        print(f"baslangic agirliklari: {init}")

    crit_tab = None
    if tab_ds is not None:
        w = tab_ds.class_weights(ncls, scheme=weight_scheme, cap=weight_cap).to(device)
        crit_tab = torch.nn.CrossEntropyLoss(weight=w, ignore_index=PAD)
        tab_dl = DataLoader(tab_ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = torch.nn.BCELoss(reduction="none")

    def report(ep, lt, lp):
        _, (tP, tR, tF) = evaluate_tab_seq(model, gs_val, device, INSTR)
        msg = f"ep {ep:2d} | tab-loss {lt:.3f} perde-loss {lp:.3f} | GuitarSet tab F1 {tF:.3f}"
        gF = None
        if p_va is not None:
            gP, gR, gF = eval_pitch_frames(model, p_va, idx, device)
            msg += f" | GAPS-val kare F1 {gF:.3f} (P{gP:.2f}/R{gR:.2f})"
        print(msg)
        return gF if select == "gaps" else tF

    best = report(0, 0.0, 0.0)            # başlangıç noktası (init modeli) da aday
    torch.save({"model": model.state_dict(), "n_classes": ncls}, ckpt)
    for ep in range(1, epochs + 1):
        model.train()
        lt = lp = 0.0; nt = npb = 0
        p_iter = iter(p_tr) if p_tr is not None else None
        tab_iter = iter(tab_dl) if crit_tab is not None else iter(p_tr)
        for batch in tab_iter:
            opt.zero_grad(); loss = 0.0
            if crit_tab is not None:
                x, y, _ = batch
                x, y = x.to(device), y.to(device)
                z = model(x); B, L, S, C = z.shape
                l = crit_tab(z.reshape(-1, C), y.reshape(-1)); loss = loss + l
                lt += l.item(); nt += 1
                if p_iter is not None:
                    try:
                        batch = next(p_iter)
                    except StopIteration:
                        p_iter = iter(p_tr); batch = next(p_iter)
            if p_tr is not None:
                x, f, Lv = batch
                x, f = x.to(device), f.to(device)
                pp = pitch_probs(model(x), idx).clamp(1e-6, 1 - 1e-6)
                mask = (torch.arange(f.shape[1], device=device)[None, :] < Lv.to(device)[:, None])
                l = (bce(pp, f).mean(-1) * mask).sum() / mask.sum()
                loss = loss + pitch_weight * l
                lp += l.item(); npb += 1
            loss.backward(); opt.step()
        sched.step()
        score = report(ep, lt / max(nt, 1), lp / max(npb, 1))
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "n_classes": ncls}, ckpt)
    print(f"En iyi secim skoru ({select}): {best:.3f} -> {ckpt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tab-splits", default="train,train_comp")
    ap.add_argument("--pitch-splits", default="gaps_train")
    ap.add_argument("--init", default=None, help="baslangic checkpoint'i (ince ayar)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--pitch-weight", type=float, default=1.0)
    ap.add_argument("--select", default="gaps", choices=["gaps", "guitarset"])
    ap.add_argument("--ckpt", default="tabcrnn_domain.pt")
    a = ap.parse_args()
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]
    train(split(a.tab_splits), split(a.pitch_splits), a.init, a.epochs, a.lr, a.batch_size,
          a.pitch_weight, a.select, a.ckpt)
