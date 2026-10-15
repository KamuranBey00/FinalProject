"""
Katman 5 Adım 3 — Tap modu, GERÇEK dokunuşlarla ölçüm (simülasyonun doğrulaması; README17).

Girdi: bir klasör, içinde <id>.txt (scripts/app/tap_recorder.html ile kaydedilen dokunuş zamanları, sn). <id> = GAPS kaydı
(önbellek data/cache/gaps_test|gaps_train/<id>.npz). Ölçüm yalnız dokunulan aralıkta (ilk dokunuş − 0.5 sn .. son + 0.5 sn):
gerçek dokunuş | aynı aralıkta yarı otomatik (gerçek medyan tempo x 1.1 + ölçü) | "dokunuş temposu" = yarı otomatik, BPM
dokunuşların medyan aralığından ("tap tempo": kullanıcı BPM bilmez, dokunur) | aynı aralıkta simüle "iyi" dokunuş
(20 ms, +50 ms, %5 kaçan, %2 fazla). Ölçütler eval_score ile aynı (vuruş F, ölçü içi konum, IOI, ölçü başı F) +
tahmini sabit gecikme (± 300 ms içinde) ve dokunuş / gerçek vuruş sayısı.

Çalıştırma:
    python -m scripts.eval.eval_taps --dir dokunuslar
"""

import argparse
import glob
import os
from collections import Counter

import numpy as np

from gtab.data.gaps import GAPS_DIR
from gtab.data.gaps_score import match_notes, time_signatures
from gtab.decoding.taps import tap_lag
from gtab.evaluation.rhythm_metrics import EPS, SCALES, beat_scores, in_span, ioi_accuracy, onset_f, tempo_of
from gtab.paths import CACHE_DIR
from gtab.pipeline import analyze, load_models, rhythm
from gtab.render.score import meter_of_signature
from gtab.utils import get_device

RHY_DIR = os.path.join(CACHE_DIR, "rhythm")
SIM = (0.02, 0.05, 0.05, 0.02)                       # eval_score "iyi" koşulu


def bar_position(s, beats, downs):
    """Partisyon konumu (vuruş) -> ölçü başından uzaklık (vuruş); eval_score ile aynı."""
    k = np.searchsorted(beats, downs - 1e-6)
    j = np.searchsorted(k, np.floor(s + EPS), side="right") - 1
    return np.where(j >= 0, s - k[np.maximum(j, 0)], np.nan)


def simulate(gb, gd, rng):
    sigma, lag, miss, extra = SIM
    b = gb[gb >= gd[0] - 1e-6]
    t = b + lag + rng.normal(0, sigma, len(b))
    t = t[np.r_[True, rng.random(len(t) - 1) >= miss]]
    return np.sort(np.r_[t, rng.uniform(t[0], t[-1], rng.binomial(len(t), extra))])


def scores(r, an, g, sig, lo, hi):
    """Dokunulan aralıkta: (vuruş F, ölçü içi konum, IOI, ölçü başı F, n)."""
    gb, gd = g["beats"], g["downbeats"]
    on = g["notes"][:, 0]
    s = np.where((on >= lo) & (on <= hi), g["note_beat"], np.nan)
    det = an["notes"]
    mi, _ = match_notes([(o, int(p)) for o, _, p in g["notes"]], [dict(pitch=int(p)) for _, _, p in det], det[:, 0], 0.05)
    qm = np.where(mi >= 0, r["q"][np.maximum(mi, 0)], np.nan)
    om = np.where(mi >= 0, det[np.maximum(mi, 0), 0], np.nan)
    M, _ = meter_of_signature(sig)
    ours, ref = (qm - r["downbeat_index"]) % M, bar_position(s, gb, gd)
    v = ~np.isnan(ours) & ~np.isnan(ref)
    a, n, _ = ioi_accuracy(qm, s, om, SCALES)
    rb, rd = gb[(gb >= lo) & (gb <= hi)], gd[(gd >= lo) & (gd <= hi)]
    return (beat_scores(rb, in_span(r["beats"], rb))[0], float(np.mean(np.abs(ours[v] - ref[v]) < EPS)) if v.any() else np.nan,
            a / max(n, 1), onset_f(rd, in_span(r["downbeats"], rd)), int(v.sum()))


def main(tap_dir):
    m = load_models(get_device())
    print(f"{'kayit':<12}{'olcu':>6}{'dokunus':>9}{'gercek vurus':>14}{'gecikme':>10}  {'kosul':<16}{'vurus F':>9}{'konum':>8}"
          f"{'IOI':>8}{'olcu basi F':>13}{'n':>7}")
    tot = {}
    for tf in sorted(glob.glob(os.path.join(tap_dir, "*.txt"))):
        tid = os.path.splitext(os.path.basename(tf))[0].split(".")[0]
        src = [p for sp in ("gaps_test", "gaps_train") for p in glob.glob(os.path.join(CACHE_DIR, sp, tid + ".npz"))]
        if not src or not os.path.exists(os.path.join(RHY_DIR, tid + ".npz")):
            print(f"{tid}: onbellek / ritim etiketi yok, atlandi"); continue
        taps = np.sort(np.array(open(tf).read().replace(",", " ").split(), float))
        with np.load(src[0]) as d:
            an = analyze(d["cqt"], m, "ses")
        g = dict(np.load(os.path.join(RHY_DIR, tid + ".npz")))
        sig = Counter(time_signatures(os.path.join(GAPS_DIR, "musicxml", tid + ".xml"))).most_common(1)[0][0]
        lo, hi = taps[0] - 0.5, taps[-1] + 0.5
        gb = g["beats"]
        n_ref = int(((gb >= taps[0] - 0.3) & (gb <= taps[-1] + 0.3)).sum())
        lag = tap_lag(taps, an["act"], max_lag=0.3)
        sim_gd = g["downbeats"][g["downbeats"] >= lo]
        runs = (("gercek dokunus", rhythm(an, m, sig, taps=taps)),
                ("yari otomatik", rhythm(an, m, sig, bpm=tempo_of(gb) * 1.1)),
                ("dokunus temposu", rhythm(an, m, sig, bpm=60 / np.median(np.diff(taps)))),
                ("simule 'iyi'", rhythm(an, m, sig, taps=simulate(gb[(gb >= lo) & (gb <= hi)], sim_gd,
                                                                    np.random.default_rng(0)))))
        for k, (lab, r) in enumerate(runs):
            sc = scores(r, an, g, sig, lo, hi)
            head = (f"{tid:<12}{'%d/%d' % sig:>6}{len(taps):>9}{n_ref:>14}{1000 * -lag:>8.0f}ms" if k == 0
                    else " " * 49)
            print(f"{head}  {lab:<16}{sc[0]:>9.3f}{sc[1]:>8.3f}{sc[2]:>8.3f}{sc[3]:>13.3f}{sc[4]:>7d}")
            tot.setdefault(lab, []).append(sc)
    print("\nortalama (kayit basina):")
    for lab, v in tot.items():
        v = np.array(v, float)
        print(f"  {lab:<16}{np.nanmean(v[:, 0]):>9.3f}{np.nanmean(v[:, 1]):>8.3f}{np.nanmean(v[:, 2]):>8.3f}{np.nanmean(v[:, 3]):>13.3f}")
    print("gecikme = dokunuslarin sesin gerisinde kalma suresi (tahmin; + = gec). Karar kurali: docs/devlog/README17.md.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="<id>.txt dokunus dosyalari klasoru")
    a = ap.parse_args()
    main(a.dir)
