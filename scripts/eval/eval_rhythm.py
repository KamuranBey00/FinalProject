"""
Katman 5 Adım 0 — Ritim ölçümü (eğitim yok). Önce: python -m scripts.data.build_rhythm

A) KUANTALAYICI TAVANI (GAPS): gerçek vuruşlar + gerçek notalar -> konum / IOI doğruluğu; parametreler (lam, mu,
   akor toleransı) gaps_val'de seçilir. IOI hata kırılımı (partisyon süresine göre) + ızgara kapsamı.
B) VURUŞ TAKİBİ: zarf {cqt (ses), model (perde-onset kafası), notes (deep2 notaları)} x izleyici {dp (tightness
   30/100/400), plp}; gaps_val'de vuruş F ile seçilir. "notes_gt" (gerçek notalar) = sembolik tavan, seçilmez.
   gaps_test ve GuitarSet val (metronom) yalnız seçilen yapılandırmayla raporlanır.
C) ÖLÇÜ BAŞI: gerçek / tahmini vuruş x gerçek / deep2 notaları; pes nota ağırlığı gaps_val'de (uçtan uca satır) seçilir.
D) KUANTALAMA ZİNCİRİ (IOI doğruluğu, GAPS): gerçek vuruş + gerçek nota (tavan) | gerçek vuruş + deep2 notası |
   tahmini vuruş + gerçek nota | tahmini vuruş + deep2 notası (uçtan uca; eşleşen notalarda + kapsama).
Tahmini vuruş / ölçü başı yalnız etiketli aralıkta (± yarım vuruş) puanlanır (GAPS etiketi eşleşen notalarla sınırlı).
deep2 notaları: kalibrasyon dosyasındaki alan-içi seçim (GAPS -> gaps_val, GuitarSet -> val).
--qnet (Adım 1q): bağlamlı kuantalayıcı (RhythmNet) A ve D'de kuralın yanında raporlanır (satır sonunda "1q").
--beatnet (Adım 1b): öğrenilen vuruş + ölçü başı (BeatNet + DBN; ayarlar checkpoint'te, gaps_val'de seçilmiş) -> B2 bölümü
   (tepe seçimi / DBN) ve D'de "ogrenilen vurus" satırları.
--user-tempo (Adım 1b yarı otomatik; --beatnet ile): kullanıcı BPM + ölçü verir -> DBN bu ölçüde ve BPM ± aralıkta çözer.
   Kullanıcı girdisi taklidi: gerçek medyan tempo x (1 ± USER_ERR, işaret kayıt sırasıyla) + gerçek ölçü. Aralık ve tempo
   geçiş cezası gaps_val uçtan uca IOI ile seçilir -> B3 bölümü + D'de "yari otomatik" satırları.

Çalıştırma:
    python -m scripts.eval.eval_rhythm                 # varsayılanlar: tabcrnn_deep2.pt + tabcrnn_deep2.cal.json
    python -m scripts.eval.eval_rhythm --limit 2       # duman testi
    python -m scripts.eval.eval_rhythm --qnet rhythm_q.pt   # + Adım 1q kuantalayıcı
    python -m scripts.eval.eval_rhythm --qnet rhythm_q.pt --beatnet beat_net.pt   # + Adım 1b
    python -m scripts.eval.eval_rhythm --qnet rhythm_q.pt --beatnet beat_net.pt --user-tempo   # + yarı otomatik
"""

import argparse
import glob
import os

import numpy as np

from gtab.config import FRAME_RATE
from gtab.core.instrument import STANDARD_6
from gtab.data.gaps import gaps_files
from gtab.data.gaps_score import match_notes
from gtab.data.labels import midi_range
from gtab.data.torch_dataset import _normalize
from gtab.decoding.beat_dbn import dbn_decode, peak_beats
from gtab.decoding.rhythm import GRIDS, downbeats, envelope_cqt, envelope_notes, quantize, track_beats
from gtab.decoding.viterbi import energy_rise, segment
from gtab.evaluation.pitch_eval import load_selection, roll_to_notes
from gtab.evaluation.rhythm_metrics import (SCALES, beat_scores, chord_together, grid_coverage, in_span, ioi_detail,
                                            onset_f, position_accuracy, tempo_of, tempo_ok)
from gtab.models.beat_net import activations, encode_features, load_beat_net
from gtab.models.inference import load_model, predict_heads
from gtab.models.rhythm_net import load_rhythm_net
from gtab.models.rhythm_net import predict as predict_q
from gtab.models.stack import active
from gtab.paths import CACHE_DIR, ckpt_path
from gtab.utils import get_device

INSTR = STANDARD_6
RHY_DIR = os.path.join(CACHE_DIR, "rhythm")
ONSET_TOL = 0.05                                     # deep2 notası <-> gerçek nota eşlemesi (aynı perde)
SOURCES = ("cqt", "model", "notes")                  # seçilebilir zarflar; "notes_gt" yalnız rapor (oracle)
TRACKERS = (("dp", 30.0), ("dp", 100.0), ("dp", 400.0), ("plp", 0.0))
Q_LAM, Q_MU, Q_TOL = (0, .05, .1, .2, .3, .5, .8), (0, .1, .3, .6, 1.0), (.03, .05, .08)
Q_GRID = [dict(lam=l, mu=m, chord_tol=t) for l in Q_LAM for m in Q_MU for t in Q_TOL]
BASS_W = (0.0, 1.0, 3.0)
USER_ERR = 0.10                                      # taklit kullanıcı BPM hatası
USER_TOL, USER_TRANS = (0.2, 0.35, 0.5, 0.75), (3.0, 10.0, 30.0)


def split_files(split):
    files = (gaps_files("gaps_train")[1] if split == "gaps_val"
             else sorted(glob.glob(os.path.join(CACHE_DIR, split, "*.npz"))))
    return [f for f in files if os.path.exists(os.path.join(RHY_DIR, os.path.basename(f)))]


def load_tracks(split, model, device, sel, limit, bnet=None):
    _, thr, ot, rules = sel
    lo, _ = midi_range(INSTR)
    out = []
    for f in split_files(split)[:limit]:
        with np.load(f) as d:
            cqt, frame, onset = d["cqt"], d["frame"], d["onset"]
        with np.load(os.path.join(RHY_DIR, os.path.basename(f))) as g:
            gt = {k: g[k] for k in g.files}
        probs, ons, offs, pons = predict_heads(model, _normalize(cqt), device, with_pitch_onset=True)
        segs = segment(probs, ons, thr, ot, INSTR, rise=energy_rise(cqt, INSTR), offsets=offs,
                       pitch_onsets=pons if rules.get("onset_source") == "pitch" else None, **rules)
        det = np.array([(a / FRAME_RATE, b / FRAME_RATE, p) for a, b, p in segs], float).reshape(-1, 3)
        if "notes" not in gt:                        # GuitarSet: gerçek notalar önbellekteki roll'lardan
            gt["notes"] = np.array([(a / FRAME_RATE, b / FRAME_RATE, p) for a, b, p in roll_to_notes(frame, onset, lo)],
                                   float).reshape(-1, 3)
        gt["tempo"] = float(gt["tempo"]) if "note_beat" not in gt else tempo_of(gt["beats"])
        T = len(cqt)
        env = dict(cqt=envelope_cqt(cqt), model=(pons if pons is not None else ons).sum(1),
                   notes=envelope_notes(det[:, 0], T), notes_gt=envelope_notes(gt["notes"][:, 0], T))
        t = dict(id=os.path.basename(f)[:-4], gt=gt, det=det, env=env)
        if bnet is not None:                         # Adım 1b: kare başına (yok, vuruş, ölçü başı)
            t["act"] = activations(bnet, encode_features(model, _normalize(cqt), device), device)
        out.append(t)
    print(f"  {split}: {len(out)} kayit", flush=True)
    return out


# ----------------------------------------------------------------- A) kuantalayıcı
def quant_scores(tracks, beats_of, notes_of, qf, scales):
    """Kayıtlar üstünde toplu (konum, IOI, akor) + IOI ayrıntısı. notes_of(t) -> (q girdisi notalar (M, 3), eşleme);
    qf(notalar, vuruşlar) -> (M,) konum."""
    c = dict(pos=[0, 0], ioi=[0, 0], chord=[0, 0], cover=[0, 0], r1=[0, 0])
    ok_all, ds_all, dq_all = [], [], []
    for t in tracks:
        s = t["gt"]["note_beat"]
        beats = beats_of(t)
        nt, idx = notes_of(t)                        # idx: gerçek nota başına q dizisindeki indeks | -1
        q_all = qf(nt, beats)
        q = np.where(idx >= 0, q_all[np.maximum(idx, 0)], np.nan)
        o = np.where(idx >= 0, nt[np.maximum(idx, 0), 0], np.nan)
        for k, (a, b) in (("pos", position_accuracy(q, s)), ("chord", chord_together(q, s))):
            c[k][0] += a; c[k][1] += b
        ok, ds, dq, r = ioi_detail(q, s, o, scales)
        c["ioi"][0] += int(ok.sum()); c["ioi"][1] += len(ds)
        c["cover"][0] += int(((idx >= 0) & ~np.isnan(s)).sum()); c["cover"][1] += int((~np.isnan(s)).sum())
        c["r1"][0] += r == 1; c["r1"][1] += 1
        ok_all.append(ok); ds_all.append(ds); dq_all.append(dq)
    res = {k: a / max(b, 1) for k, (a, b) in c.items()}
    res["n_ioi"] = c["ioi"][1]
    return res, np.concatenate(ok_all), np.concatenate(ds_all), np.concatenate(dq_all)


def true_notes(t):
    return t["gt"]["notes"], np.arange(len(t["gt"]["notes"]))


def det_notes(t):
    g, d = t["gt"]["notes"], t["det"]
    mi, _ = match_notes([(on, int(p)) for on, _, p in g], [dict(pitch=int(p)) for _, _, p in d], d[:, 0], ONSET_TOL)
    return d, mi


def rule_q(p):
    return lambda nt, beats: quantize(nt[:, 0], beats, **p)


def print_breakdown(ok, ds, dq):
    print(f"    IOI hata kirilimi (partisyon suresi, vurus):  {'n':>7}{'hata':>8}{'  birlesti (dq=0)':>18}"
          f"{'kisa tahmin':>13}{'uzun tahmin':>13}")
    for lo_, hi_, lab in ((0, .25, "< 1/4"), (.25, .5, "1/4 - 1/2"), (.5, 1, "1/2 - 1"), (1, 99, "> 1")):
        m = (ds > lo_ + 1e-9) & (ds <= hi_ + 1e-9)
        e = m & ~ok
        print(f"    {lab:<44}{int(m.sum()):>7}{e.sum() / max(m.sum(), 1):>8.3f}"
              f"{(e & (np.abs(dq) < 1e-9)).sum() / max(e.sum(), 1):>18.3f}"
              f"{(e & (dq < ds) & (np.abs(dq) > 1e-9)).sum() / max(e.sum(), 1):>13.3f}"
              f"{(e & (dq > ds)).sum() / max(e.sum(), 1):>13.3f}")


def section_a(val, test, qnet=None):
    print("\n" + "=" * 100)
    print("A) KUANTALAYICI TAVANI — gercek vuruslar + gercek notalar (olcek sabit, r = 1)")
    print("=" * 100)
    gt_beats = lambda t: t["gt"]["beats"]
    rows = []
    for p in Q_GRID:
        r, *_ = quant_scores(val, gt_beats, true_notes, rule_q(p), (1,))
        rows.append((r["ioi"], r["pos"], r["chord"], p))
    rows.sort(key=lambda x: -x[0])
    print(f"  gaps_val, en iyi 5 / {len(rows)} (secim: IOI dogrulugu)")
    print(f"  {'lam':>6}{'mu':>6}{'akor tol':>10}{'IOI':>8}{'konum':>8}{'akor birlikte':>15}")
    for ioi, pos, ch, p in rows[:5]:
        print(f"  {p['lam']:>6}{p['mu']:>6}{p['chord_tol']:>10}{ioi:>8.3f}{pos:>8.3f}{ch:>15.3f}")
    p = rows[0][3]
    print(f"  (en kotu: IOI {rows[-1][0]:.3f}; secilen: {p})")
    for k, g in (("lam", Q_LAM), ("mu", Q_MU), ("chord_tol", Q_TOL)):
        if p[k] == g[-1] and len(g) > 1:
            print(f"  UYARI: {k} taramanin ucunda ({p[k]}); aralik genisletilmeli.")
    for name, tr in (("gaps_val", val), ("gaps_test", test)):
        r, ok, ds, dq = quant_scores(tr, gt_beats, true_notes, rule_q(p), (1,))
        cov = grid_coverage(np.concatenate([t["gt"]["note_beat"] for t in tr]), GRIDS)
        print(f"\n  {name}: IOI {r['ioi']:.3f} (n={r['n_ioi']}) | konum {r['pos']:.3f} | akor birlikte {r['chord']:.3f} | "
              f"izgara kapsami {cov:.3f}")
        print_breakdown(ok, ds, dq)
        if qnet:
            r, ok, ds, dq = quant_scores(tr, gt_beats, true_notes, qnet, (1,))
            print(f"  {name} 1q: IOI {r['ioi']:.3f} (n={r['n_ioi']}) | konum {r['pos']:.3f} | akor birlikte {r['chord']:.3f}")
            print_breakdown(ok, ds, dq)
    return p


# ----------------------------------------------------------------- B) vuruş takibi
def beat_table(tracks, configs):
    """config -> (ortalama F, CMLt, AMLt, Acc1, Acc2) + tahmini vuruşlar."""
    out = {}
    for src, m, tight in configs:
        sc, est = [], []
        for t in tracks:
            b = track_beats(t["env"][src], m, tight)
            est.append(b)
            ref = t["gt"]["beats"]
            f, cmlt, amlt = beat_scores(ref, in_span(b, ref))
            a1, a2 = tempo_ok(t["gt"]["tempo"], tempo_of(in_span(b, ref)))
            sc.append((f, cmlt, amlt, a1, a2))
        out[(src, m, tight)] = (np.mean(sc, 0), est)
    return out


def _cfg(c):
    return f"{c[0]:<9}{c[1]:<4}{('t=' + str(int(c[2]))) if c[1] == 'dp' else '':<7}"


def section_b(val, others):
    print("\n" + "=" * 100)
    print("B) VURUS TAKIBI — ortalama F (+-70 ms), CMLt, AMLt; tempo Acc1 (+-%4) / Acc2 (x1/3, 1/2, 2, 3)")
    print("=" * 100)
    configs = [(s, m, t) for s in SOURCES + ("notes_gt",) for m, t in TRACKERS]
    tab = beat_table(val, configs)
    print(f"  gaps_val:  {'zarf':<9}{'iz':<4}{'':<7}{'F':>7}{'CMLt':>7}{'AMLt':>7}{'Acc1':>7}{'Acc2':>7}")
    for c in configs:
        v = tab[c][0]
        print(f"             {_cfg(c)}" + "".join(f"{x:>7.3f}" for x in v) + ("   (oracle notalar)" if c[0] == "notes_gt" else ""))
    best = max((c for c in configs if c[0] != "notes_gt"), key=lambda c: tab[c][0][0])
    print(f"  SECILEN (gaps_val F): {_cfg(best).strip()}")
    est = {"gaps_val": tab[best][1]}
    for name, tr in others.items():
        t2 = beat_table(tr, [best, ("notes_gt",) + best[1:]])
        est[name] = t2[best][1]
        for c in (best, ("notes_gt",) + best[1:]):
            print(f"  {name:<10} {_cfg(c)}" + "".join(f"{x:>7.3f}" for x in t2[c][0]))
    return best, est


# ----------------------------------------------------------------- C) ölçü başı
def meter_of(beats, downs):
    k = np.searchsorted(beats, downs)
    return int(np.median(np.diff(k))) if len(k) >= 3 else 0


# ----------------------------------------------------------------- B2) öğrenilen izleyici (Adım 1b)
def section_b2(data, dbn):
    print("\n" + "=" * 100)
    print(f"B2) OGRENILEN VURUS + OLCU BASI (Adim 1b) — BeatNet; DBN ayarlari (gaps_val): {dbn}")
    print("=" * 100)
    print(f"  {'':<12}{'':<14}{'F':>7}{'CMLt':>7}{'AMLt':>7}{'Acc1':>7}{'Acc2':>7}{'olcu basi F':>13}{'olcu':>7}")
    est = {}
    for name, tr in data.items():
        for lab, dec in (("tepe secimi", lambda a: peak_beats(a) + (0,)), ("DBN", lambda a: dbn_decode(a, **dbn))):
            sc, eb = [], []
            for t in tr:
                b, d, M = dec(t["act"])
                eb.append(b)
                ref, rd = t["gt"]["beats"], t["gt"]["downbeats"]
                f, cmlt, amlt = beat_scores(ref, in_span(b, ref))
                a1, a2 = tempo_ok(t["gt"]["tempo"], tempo_of(in_span(b, ref)))
                sc.append((f, cmlt, amlt, a1, a2, onset_f(rd, in_span(d, rd)), M == meter_of(ref, rd)))
            v = np.mean(sc, 0)
            print(f"  {name:<12}{lab:<14}" + "".join(f"{x:>7.3f}" for x in v[:5]) + f"{v[5]:>13.3f}"
                  + (f"{v[6]:>7.3f}" if lab == "DBN" else ""))
        est[name] = eb
    return est


def section_c(data, est):
    print("\n" + "=" * 100)
    print("C) OLCU BASI — ortalama olcu basi F (+-70 ms); olcu dogrulugu (vurus/olcu, yalniz gercek vurusta)")
    print("=" * 100)
    rows = (("gercek vurus + gercek nota", lambda t, i, n: t["gt"]["beats"], lambda t: t["gt"]["notes"]),
            ("gercek vurus + deep2 notasi", lambda t, i, n: t["gt"]["beats"], lambda t: t["det"]),
            ("tahmini vurus + deep2 notasi", lambda t, i, n: est[n][i], lambda t: t["det"]))

    def run(name, tr, bw):
        res = []
        for lab, bf, nf in rows:
            f = m = 0
            for i, t in enumerate(tr):
                d, M = downbeats(bf(t, i, name), nf(t), bass_weight=bw)
                ref = t["gt"]["downbeats"]
                f += onset_f(ref, in_span(d, ref))
                m += M == meter_of(t["gt"]["beats"], ref)
            res.append((lab, f / max(len(tr), 1), m / max(len(tr), 1)))
        return res

    sel = max(BASS_W, key=lambda bw: run("gaps_val", data["gaps_val"], bw)[2][1])
    print(f"  pes nota agirligi (gaps_val, uctan uca satir): {sel}")
    print(f"  {'':<12}{'':<32}{'F':>8}{'olcu':>8}")
    for name, tr in data.items():
        for lab, f, m in run(name, tr, sel):
            print(f"  {name:<12}{lab:<32}{f:>8.3f}" + (f"{m:>8.3f}" if "gercek vurus" in lab else ""))
    return sel


# ----------------------------------------------------------------- B3) yarı otomatik (Adım 1b)
def user_decode(t, i, err, tol, trans, obs):
    """Taklit kullanıcı girdisi (gerçek medyan tempo x (1 ± err), işaret kayıt sırasıyla; gerçek ölçü) -> DBN."""
    ref, rd = t["gt"]["beats"], t["gt"]["downbeats"]
    bpm, M = tempo_of(ref) * (1 + err * (-1) ** i), max(1, meter_of(ref, rd))
    return dbn_decode(t["act"], meters=(M,), min_bpm=bpm / (1 + tol), max_bpm=bpm * (1 + tol), trans_lambda=trans,
                      obs_lambda=obs)


def section_b3(data, dbn, qf):
    print("\n" + "=" * 100)
    print(f"B3) YARI OTOMATIK (Adim 1b) — kullanici BPM (gercek medyan x (1 +- {USER_ERR})) + olcu; DBN o olcude, BPM +- aralik")
    print("=" * 100)
    obs = dbn.get("obs_lambda", 16)
    val = data["gaps_val"]
    print("  secim (gaps_val, uctan uca IOI = tahmini vurus + deep2):")
    print(f"  {'aralik':>8}{'trans':>7}{'vurus F':>9}{'olcu basi F':>13}{'IOI':>8}")
    res = []
    for tol in USER_TOL:
        for tr in USER_TRANS:
            est = [user_decode(t, i, USER_ERR, tol, tr, obs) for i, t in enumerate(val)]
            fb = np.mean([beat_scores(t["gt"]["beats"], in_span(e[0], t["gt"]["beats"]))[0] for t, e in zip(val, est)])
            fd = np.mean([onset_f(t["gt"]["downbeats"], in_span(e[1], t["gt"]["downbeats"])) for t, e in zip(val, est)])
            idx = {t["id"]: k for k, t in enumerate(val)}
            r, *_ = quant_scores(val, lambda t: est[idx[t["id"]]][0], det_notes, qf, SCALES)
            res.append((r["ioi"], tol, tr))
            print(f"  {tol:>8}{tr:>7}{fb:>9.3f}{fd:>13.3f}{r['ioi']:>8.3f}")
    _, tol, tr = max(res)
    print(f"  SECILEN: aralik +-{tol}, trans {tr} (obs {obs})")
    print(f"  {'':<12}{'BPM hatasi':<12}{'F':>7}{'CMLt':>7}{'AMLt':>7}{'olcu basi F':>13}")
    est_u = {}
    for name, trk in data.items():
        for err in (0.0, USER_ERR):
            est = [user_decode(t, i, err, tol, tr, obs) for i, t in enumerate(trk)]
            sc = [beat_scores(t["gt"]["beats"], in_span(e[0], t["gt"]["beats"]))
                  + (onset_f(t["gt"]["downbeats"], in_span(e[1], t["gt"]["downbeats"])),) for t, e in zip(trk, est)]
            print(f"  {name:<12}{'%' + str(int(100 * err)):<12}" + "".join(f"{x:>7.3f}" for x in np.mean(sc, 0)[:3])
                  + f"{np.mean(sc, 0)[3]:>13.3f}")
            if err == USER_ERR:
                est_u[name] = [e[0] for e in est]
    return est_u


# ----------------------------------------------------------------- D) zincir
def section_d(data, est, p, qnet=None, est_b=None, est_u=None):
    print("\n" + "=" * 100)
    print("D) KUANTALAMA ZINCIRI — IOI dogrulugu (nota degerleri), GAPS; tahmini vurusta kayit basina olcek r")
    print("=" * 100)
    print(f"  {'':<12}{'':<34}{'IOI':>8}{'n':>8}{'kapsama':>9}{'akor birlikte':>15}{'r = 1':>8}")
    for name in ("gaps_val", "gaps_test"):
        tr = data[name]
        idx = {t["id"]: i for i, t in enumerate(tr)}
        eb = lambda t: est[name][idx[t["id"]]]
        lb = lambda t: est_b[name][idx[t["id"]]]
        ub = lambda t: est_u[name][idx[t["id"]]]
        rows = (("gercek vurus + gercek nota (tavan)", lambda t: t["gt"]["beats"], true_notes, (1,)),
                ("gercek vurus + deep2 notasi", lambda t: t["gt"]["beats"], det_notes, (1,)),
                ("tahmini vurus + gercek nota", eb, true_notes, SCALES),
                ("tahmini vurus + deep2 (UCTAN UCA)", eb, det_notes, SCALES))
        if est_b:
            rows += (("ogrenilen vurus + gercek nota", lb, true_notes, SCALES),
                     ("ogrenilen vurus + deep2 (UCTAN UCA)", lb, det_notes, SCALES))
        if est_u:
            rows += (("yari otomatik vurus + gercek nota", ub, true_notes, SCALES),
                     ("yari otomatik + deep2 (UCTAN UCA)", ub, det_notes, SCALES))
        for lab, bf, nf, sc in rows:
            for tag, qf in (("", rule_q(p)),) + ((("  1q", qnet),) if qnet else ()):
                r, *_ = quant_scores(tr, bf, nf, qf, sc)
                print(f"  {name:<12}{lab:<34}{r['ioi']:>8.3f}{r['n_ioi']:>8d}{r['cover']:>9.3f}{r['chord']:>15.3f}"
                      f"{r['r1']:>8.2f}{tag}")


def main(ckpt, load_cal, limit, qnet_ckpt, beat_ckpt, user_tempo):
    device = get_device()
    model, _, _ = load_model("crnn", device, ckpt)
    cal = load_selection(load_cal)
    print(f"Model {ckpt} | cozumleme {load_cal} | cihaz {device}")
    qnet = None
    if qnet_ckpt:
        net, lam = load_rhythm_net(qnet_ckpt, device)
        qnet = lambda nt, beats: predict_q(net, nt, beats, device, lam)
        print(f"1q kuantalayici {qnet_ckpt} (lam {lam})")
    bnet = dbn = None
    if beat_ckpt:
        bnet, dbn = load_beat_net(beat_ckpt, device)
        print(f"1b vurus agi {beat_ckpt} (DBN {dbn})")
    data = {"gaps_val": load_tracks("gaps_val", model, device, cal["gaps_val"], limit, bnet),
            "gaps_test": load_tracks("gaps_test", model, device, cal["gaps_val"], limit, bnet),
            "GS val": load_tracks("val", model, device, cal["val"], limit, bnet)}
    p = section_a(data["gaps_val"], data["gaps_test"], qnet)
    best, est = section_b(data["gaps_val"], {k: v for k, v in data.items() if k != "gaps_val"})
    section_c(data, est)
    est_b = section_b2(data, dbn) if bnet is not None else None
    est_u = section_b3(data, dbn, qnet or rule_q(p)) if bnet is not None and user_tempo else None
    section_d(data, est, p, qnet, est_b, est_u)
    print("\nKarar kurali: docs/devlog/README15.md (Adim 0).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=active()["acoustic"])
    ap.add_argument("--load-cal", default=ckpt_path(active()["cal"]))
    ap.add_argument("--limit", type=int, default=None, help="split basina en cok kayit (duman testi)")
    ap.add_argument("--qnet", default=None, help="Adim 1q kuantalayici (orn. rhythm_q.pt)")
    ap.add_argument("--beatnet", default=None, help="Adim 1b vurus agi (orn. beat_net.pt)")
    ap.add_argument("--user-tempo", action="store_true", help="Adim 1b yari otomatik (kullanici BPM + olcu; --beatnet ile)")
    a = ap.parse_args()
    main(a.ckpt, a.load_cal, a.limit, a.qnet, a.beatnet, a.user_tempo)
