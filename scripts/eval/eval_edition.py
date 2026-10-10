"""
Katman 3.13 Adım 2 — "Klasik edisyon önerisi" ölçümü (konum + sol el parmağı).

İki ayrı çıktı modu (karıştırılmaz): "duyulan konum" = ses (deep2, greedy) | "edisyon önerisi" = sembolik model.
1) Tel uyumu, GERÇEK notalar (MIDI perdesi + zamanları), partisyonda etiketli notalar:
   naif (en düşük perde) | GuitarSet önseli (transitions.npz) | edisyon log-lineer | edisyon dizi modeli | ses.
   GuitarSet solo val (hex = fiziksel tel): ses 0.878 olmalı (= eval_hmm), edisyon ayrıca raporlanır.
2) Parmak (parmak numaralı basılı notalar):
   a) partisyon konumu verilerek: yalnız parmak (taban "perde − pozisyon + 1" | log-lineer | dizi)
   b) ÜRÜN: konumu yöntem kendisi seçer, doğru = (tel, perde, parmak) üçü birden; taban da aynı konum üstünde
3) UÇTAN UCA (gaps_test, bir kez): deep2'nin sesten bulduğu notalar (çözümleme gaps_val seçimi, --load-cal) ->
   konum/parmak yöntemleri -> partisyon etiketli notalarla eşleme (aynı perde, ±50 ms).
   Oranlar iki paydada: bulunan etiketli notalarda | tüm etiketli notalarda (ürün sayısı = nota kaçması dahil).

Çalıştırma:
    python -m scripts.eval.eval_edition --edition edition.npz --seq edition_seq.pt --ckpt tabcrnn_deep2.pt
"""

import argparse
import glob
import os

import numpy as np
import torch

from gtab.config import FRAME_RATE
from gtab.core.instrument import STANDARD_6
from gtab.data.gaps import GAPS_TAB_DIR, gaps_files
from gtab.data.gaps_score import match_notes
from gtab.data.torch_dataset import _normalize
from gtab.decoding.edition import decode, finger_baseline, label_sequence, load_edition, order_notes
from gtab.decoding.transitions import TransitionModel, build_lattice, decode_lattice, notes_from_tab
from gtab.decoding.viterbi import energy_rise, segment
from gtab.evaluation.pitch_eval import load_selection
from gtab.models.edition_net import EditionNet, predict as seq_predict
from gtab.models.inference import load_model, predict_heads
from gtab.paths import CACHE_DIR, ckpt_path
from gtab.utils import get_device

INSTR = STANDARD_6
ONSET_TOL = 0.05                                       # uçtan uca nota eşleme (mir_eval ile aynı tolerans)


def emissions(probs, notes):
    """Not başına {(tel, perde): log P_ses} — adaylar üstünde normalize (build_lattice ile aynı)."""
    out = []
    for on, off, p in notes:
        c = INSTR.pitch_to_positions(int(p))
        if not c:                                      # gitar aralığı dışı perde
            out.append({}); continue
        a = int(round(on * FRAME_RATE)); b = max(a + 1, int(round(off * FRAME_RATE)))
        m = np.array([probs[a:b, s, f + 1].mean() if a < len(probs) else 0.0 for s, f in c])
        out.append(dict(zip(c, np.log(m + 1e-6) - np.log(m.sum() + 1e-6 * len(c)))))
    return out


def greedy(em):
    return [max(e, key=e.get) if e else None for e in em]


def naive(notes):
    return [min(c, key=lambda x: x[1]) if (c := INSTR.pitch_to_positions(int(p))) else None for _, _, p in notes]


def string_acc(pred, gold, lab):
    """Etiketli notalarda (tel, perde) uyumu."""
    ok = [tuple(p[:2]) == g for p, g, l in zip(pred, gold, lab) if l and p is not None and p[0] is not None]
    return float(np.mean(ok)) if ok else float("nan")


def gaps_files_of(split):
    return (gaps_files("gaps_train")[1] if split == "gaps_val"
            else sorted(glob.glob(os.path.join(CACHE_DIR, split, "*.npz"))))


def gaps_tracks(split, model, device, dec=None):
    """dec = (perde eşiği, onset eşiği, kurallar) verilirse deep2'nin bulduğu notalar da eklenir (uçtan uca)."""
    out = []
    for f in gaps_files_of(split):
        lf = os.path.join(GAPS_TAB_DIR, os.path.basename(f))
        if not os.path.exists(lf):
            continue
        notes, chord, s, fr, g = label_sequence(lf)
        with np.load(f) as d:
            probs, ons, offs, pons = predict_heads(model, _normalize(d["cqt"]), device, with_pitch_onset=True)
            rise = energy_rise(d["cqt"], INSTR) if dec else None
        t = dict(notes=notes, chord=chord, gold=list(zip(s, fr)), finger=g, lab=s >= 0, em=emissions(probs, notes))
        if dec:
            thr, ot, rules = dec
            segs = segment(probs, ons, thr, ot, INSTR, rise=rise, offsets=offs,
                           pitch_onsets=pons if rules.get("onset_source") == "pitch" else None, **rules)
            det = np.array([(a / FRAME_RATE, b / FRAME_RATE, p) for a, b, p in segs], float).reshape(-1, 3)
            o, dchord = order_notes(det)
            t.update(det=det[o], det_chord=dchord, det_em=emissions(probs, det[o]))
        out.append(t)
    return out


def gs_tracks(model, device):
    out = []
    for f in sorted(glob.glob(os.path.join(CACHE_DIR, "val", "*.npz"))):
        with np.load(f) as d:
            seq = notes_from_tab(d["tab"].astype(np.int64), INSTR)
            probs, _, _, _ = predict_heads(model, _normalize(d["cqt"]), device, with_pitch_onset=True)
        notes = np.array([(a, b, p) for a, b, p, _, _ in seq], float)
        o, chord = order_notes(notes)
        gold = [(seq[i][3], seq[i][4]) for i in o]
        out.append(dict(notes=notes[o], chord=chord, gold=gold, lab=np.ones(len(o), bool),
                        em=emissions(probs, notes[o])))
    return out


def pooled(tracks, fn):
    """Kayıtlar üstünde etiketli nota ağırlıklı tel uyumu."""
    num = den = 0
    for t in tracks:
        n = int(t["lab"].sum()); a = string_acc(fn(t), t["gold"], t["lab"])
        if n and not np.isnan(a):
            num += a * n; den += n
    return num / max(den, 1)


def with_baseline_finger(notes, positions):
    """Konum listesi [(tel, perde) | None] -> [(tel, perde, taban parmak) | None]."""
    fg = finger_baseline(notes, positions)
    return [None if p is None or p[0] is None else (p[0], p[1], fg[i]) for i, p in enumerate(positions)]


def main(edition, ckpt, seq, load_cal):
    pos, fing = load_edition(ckpt_path(edition))
    tm = TransitionModel.load(ckpt_path("transitions.npz"), INSTR)
    device = get_device()
    model, _, _ = load_model("crnn", device, ckpt)
    net = EditionNet().to(device)
    net.load_state_dict(torch.load(ckpt_path(seq), map_location=device)["model"])
    _, thr, ot, rules = load_selection(load_cal)["gaps_val"]
    data = {"gaps_val": gaps_tracks("gaps_val", model, device),
            "gaps_test": gaps_tracks("gaps_test", model, device, dec=(thr, ot, rules)),
            "GuitarSet val": gs_tracks(model, device)}
    print(f"Edisyon: {edition} + {seq} | ses: {ckpt} | kayit: " + ", ".join(f"{k} {len(v)}" for k, v in data.items()))

    cache = {}

    def ll(t, key="notes", chord="chord", fixed=None):
        k = ("ll", id(t), key, fixed is not None)
        if k not in cache:
            cache[k] = decode(t[key], t[chord], pos, fing, fixed=fixed)
        return cache[k]

    def sq(t, key="notes", chord="chord", fixed=None):
        k = ("sq", id(t), key, fixed is not None)
        if k not in cache:
            s, f, g = seq_predict(net, t[key], t[chord], device, fixed=fixed)
            cache[k] = [(int(a), int(b), int(c)) if a >= 0 else None for a, b, c in zip(s, f, g)]
        return cache[k]

    # ------------------------------------------------------------ 1) tel uyumu, gerçek notalar
    rows = [("naif (en dusuk perde)", lambda t: naive(t["notes"])),
            ("GuitarSet onseli (ses yok)",
             lambda t: decode_lattice(build_lattice([(a, b, int(p)) for a, b, p in t["notes"]], None, tm,
                                                    time_unit="sec"), 0.0, 1.0)),
            ("EDISYON log-lineer", ll), ("EDISYON dizi modeli  [mod: edisyon onerisi]", sq),
            ("ses / greedy         [mod: duyulan konum]", lambda t: greedy(t["em"]))]
    print("\n1) TEL UYUMU (gercek notalar; partisyonda etiketli notalar)")
    print(f"  {'yontem':<48}" + "".join(f"{k:>15}" for k in data))
    print("  " + "-" * (48 + 15 * len(data)))
    for name, fn in rows:
        print(f"  {name:<48}" + "".join(f"{pooled(v, fn):>15.3f}" for v in data.values()))

    # ------------------------------------------------------------ 2) parmak, gerçek notalar
    print("\n2) PARMAK (gercek notalar; partisyonda parmak numarali basili notalar)")
    print(f"  {'yontem':<56}{'gaps_val':>12}{'gaps_test':>12}")
    print("  " + "-" * 80)
    res = {}
    for sp in ("gaps_val", "gaps_test"):
        c = dict(n=0, base=0, ll=0, sq=0, f_naive=0, f_base_sq=0, f_sq=0, f_ll=0)
        for t in data[sp]:
            gold = [g if l else None for g, l in zip(t["gold"], t["lab"])]
            base = finger_baseline(t["notes"], gold)
            llf = ll(t, fixed=gold)
            sqf = sq(t, fixed=np.array([g[0] if g is not None else -1 for g in gold]))
            own_sq = sq(t)
            prod = {"f_naive": with_baseline_finger(t["notes"], naive(t["notes"])),
                    "f_base_sq": with_baseline_finger(t["notes"], [None if p is None else p[:2] for p in own_sq]),
                    "f_sq": own_sq, "f_ll": ll(t)}
            for i, (g, gf) in enumerate(zip(gold, t["finger"])):
                if g is None or g[1] == 0 or gf < 1:
                    continue
                c["n"] += 1; c["base"] += base[i] == gf; c["ll"] += llf[i][2] == gf
                c["sq"] += sqf[i] is not None and sqf[i][2] == gf
                for k, pr in prod.items():
                    c[k] += pr[i] is not None and tuple(pr[i]) == (g[0], g[1], gf)
        res[sp] = c
    pr = lambda name, k: print(f"  {name:<56}" + "".join(f"{res[sp][k] / max(res[sp]['n'], 1):>12.3f}" for sp in res))
    print("  a) partisyon konumu verilerek (yalniz parmak):")
    pr("     taban: parmak = perde - pozisyon + 1", "base")
    pr("     EDISYON log-lineer", "ll")
    pr("     EDISYON dizi modeli", "sq")
    print("  b) URUN: konumu yontem secer, dogru = tel + perde + parmak:")
    pr("     naif konum + taban parmak (tamamen kural)", "f_naive")
    pr("     dizi modeli konumu + taban parmak", "f_base_sq")
    pr("     EDISYON log-lineer (konum + parmak)", "f_ll")
    pr("     EDISYON dizi modeli (konum + parmak)", "f_sq")
    print(f"  {'(parmakli nota sayisi)':<56}" + "".join(f"{res[sp]['n']:>12d}" for sp in res))

    # ------------------------------------------------------------ 3) uçtan uca, deep2 notaları (gaps_test)
    print(f"\n3) UCTAN UCA (gaps_test; deep2 notalari, cozumleme gaps_val secimi: esik {thr}, onset@{ot}; "
          f"eslesme ayni perde +-{int(ONSET_TOL * 1000)} ms)")
    methods = {"naif (en dusuk perde)": lambda t: with_baseline_finger(t["det"], naive(t["det"])),
               "EDISYON dizi modeli  [edisyon onerisi]": lambda t: sq(t, "det", "det_chord"),
               "dizi konumu + taban parmak":
                   lambda t: with_baseline_finger(t["det"], [None if p is None else p[:2]
                                                             for p in sq(t, "det", "det_chord")]),
               "ses / greedy         [duyulan konum]": lambda t: with_baseline_finger(t["det"], greedy(t["det_em"]))}
    n_lab = n_found = n_fing = n_fing_found = 0
    acc = {m: dict(tel=0, fing=0) for m in methods}
    for t in data["gaps_test"]:
        gold_notes = [(on, int(p)) for on, _, p in t["notes"]]
        mi, _ = match_notes(gold_notes, [dict(pitch=int(p)) for _, _, p in t["det"]], t["det"][:, 0], ONSET_TOL)
        preds = {m: fn(t) for m, fn in methods.items()}
        for i, (g, gf, l) in enumerate(zip(t["gold"], t["finger"], t["lab"])):
            if not l:
                continue
            j = mi[i]; n_lab += 1; n_found += j >= 0
            fingered = g[1] > 0 and gf >= 1
            n_fing += fingered; n_fing_found += fingered and j >= 0
            if j < 0:
                continue
            for m, p in preds.items():
                q = p[j]
                if q is None:
                    continue
                acc[m]["tel"] += tuple(q[:2]) == tuple(g)
                acc[m]["fing"] += fingered and tuple(q) == (g[0], g[1], gf)
    print(f"  etiketli nota {n_lab}, deep2 buldu {n_found} (%{100 * n_found / max(n_lab, 1):.1f}) | "
          f"parmakli {n_fing}, bulundu {n_fing_found} (%{100 * n_fing_found / max(n_fing, 1):.1f})")
    print(f"  {'yontem':<42}{'tel: bulunanlarda':>19}{'tum etiketlilerde':>19}{'tam parmak: bulunan':>21}{'tum parmakli':>14}")
    print("  " + "-" * 115)
    for m, a in acc.items():
        print(f"  {m:<42}{a['tel'] / max(n_found, 1):>19.3f}{a['tel'] / max(n_lab, 1):>19.3f}"
              f"{a['fing'] / max(n_fing_found, 1):>21.3f}{a['fing'] / max(n_fing, 1):>14.3f}")

    print("\nKabul (README14 Adim 2): gaps_test tel uyumu EDISYON >= naif + 0.05; parmak EDISYON > taban (2b'de de);")
    print("GuitarSet 'ses / greedy' 0.878 (duyulan konum modu degismez). Ses ve edisyon karistirilmaz.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--edition", default="edition.npz")
    ap.add_argument("--seq", default="edition_seq.pt")
    ap.add_argument("--ckpt", default="tabcrnn_deep2.pt")
    ap.add_argument("--load-cal", default="checkpoints/tabcrnn_deep2.cal.json")
    a = ap.parse_args()
    main(a.edition, a.ckpt, a.seq, a.load_cal)
