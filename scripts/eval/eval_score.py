"""
Katman 5 Adım 2 — Ölçülü TAB (MusicXML) ölçümü.

A) GİDİŞ-DÖNÜŞ (yazıcı doğruluğu): tek ölçü imli, anakruzsuz GAPS partisyonlarının TAB notaları (konum = partisyon) ->
   build_measures -> write_musicxml (partisyonun kendi akortuyla) -> gtab.data.gaps_score okuyucusu -> (ölçü, tick, perde,
   tel, perde no, parmak) birebir mi; her ölçünün süre toplamı = ölçü imi mi (doluluk).
B) UÇTAN UCA (GAPS, yarı otomatik): kayıt -> gtab.pipeline.transcribe (ses modu) -> MusicXML. Kullanıcı girdisi taklidi
   eval_rhythm B3 ile aynı: gerçek medyan tempo x (1 ± %10, işaret kayıt sırasıyla) + partisyonun en sık ölçü imi
   (kullanıcı ana ölçüyü verir; ölçü imi değişen parçaları tek im temsil edemez -> ölçü içi konum ayrıca tek imlilerde).
   deep2 notaları gerçek notalara eşlenir (aynı perde, ± 50 ms):
   - ölçü içi konum doğruluğu: notanın ölçü başından uzaklığı (vuruş) partisyondakiyle aynı mı (ölçü başı fazı + metrik
     düzey + kuantalama birlikte); tümü ve yalnız tek ölçü imli parçalar
C) TAP MODU (simülasyon, Adım 3, README17): gerçek vuruşlar (ilk ölçü başından) + gecikme + sapma + kaçan / fazla
   dokunuş -> gtab.decoding.taps (gecikme, temizlik, ölçü fazı Viterbi'si) -> aynı zincir; vuruş F, ölçü içi konum, IOI,
   ölçü başı F. Ayarlar (fill_thr 0.05, eps 0.003) gaps_val'de seçildi.
   - faz düzeltmeli (oracle, tek imli): kullanıcı ölçü çizgisini parça boyunca k vuruş kaydırsa (en iyi k) -> "kaydır"
     düğmesinin tavanı; "kaymis" = en iyi k != 0 olan kayıt sayısı (Adım 3, README17)
   - IOI doğruluğu (eval_rhythm D "yari otomatik + deep2 1q" ile aynı tanım), ölçü başı F
   - çıktının geri okunabilirliği (okuyucu tüm konumlu notaları ve tek tip ölçü uzunluğunu buluyor mu)

Çalıştırma:
    python -m scripts.eval.eval_score                  # ~10 dk; önce train_beat (beat_net.pt) ve train_rhythm_q
    python -m scripts.eval.eval_score --limit 2        # duman testi
"""

import argparse
import glob
import os
import tempfile
from collections import Counter

import numpy as np

from gtab.core.instrument import Instrument
from gtab.data.gaps import GAPS_DIR, gaps_files
from gtab.data.gaps_score import TPQ, match_notes, score_notes, time_signatures
from gtab.evaluation.rhythm_metrics import EPS, SCALES, beat_scores, in_span, ioi_accuracy, onset_f, tempo_of
from gtab.models.rhythm_net import TICKS
from gtab.paths import CACHE_DIR
from gtab.pipeline import analyze, load_models, rhythm
from gtab.render.musicxml import write_musicxml
from gtab.render.score import build_measures, meter_of_signature
from gtab.utils import get_device

RHY_DIR = os.path.join(CACHE_DIR, "rhythm")
ONSET_TOL = 0.05
USER_ERR = 0.10
# tap modu taklidi: (ad, sapma sigma sn, sabit gecikme sn, kaçan oranı, fazla oranı)
TAP_CONDS = (("ideal", 0.0, 0.0, 0.0, 0.0), ("iyi", 0.02, 0.05, 0.05, 0.02), ("kaba", 0.04, 0.08, 0.05, 0.02))


def xml_of(tid):
    return os.path.join(GAPS_DIR, "musicxml", f"{tid}.xml")


# ----------------------------------------------------------------- A) gidiş-dönüş
def section_a(tmp, limit):
    print("=" * 100)
    print("A) GIDIS-DONUS — partisyon -> MusicXML yazici -> GAPS okuyucusu (tek olcu imli, anakruzsuz partisyonlar)")
    print("=" * 100)
    n_sc = tot = ok = off = bad_bar = n_bar = 0
    for x in sorted(glob.glob(os.path.join(GAPS_DIR, "musicxml", "*.xml")))[:limit]:
        sigs = time_signatures(x)
        r = score_notes(x) if len(set(sigs)) == 1 else None
        if r is None:
            continue
        _, lens, notes, tuning = r
        M, beat_q = meter_of_signature(sigs[0])
        if not np.allclose(lens, M * beat_q * TPQ):
            continue
        cum = np.r_[0, np.cumsum(lens)]
        n = [d for d in notes if d["fret"] >= 0 and d["string"] >= 1]
        q = np.array([(cum[d["bar"]] + d["tick"]) / (beat_q * TPQ) for d in n])
        grid = np.abs(q * TICKS - np.round(q * TICKS)) < 1e-6
        off += int((~grid).sum())
        q[~grid] = np.nan
        meas, first = build_measures(q, 0.0, M, beat_q)
        out = os.path.join(tmp, "a.musicxml")
        write_musicxml(out, meas, np.array([d["pitch"] for d in n]), np.array([6 - d["string"] for d in n]),
                       np.array([d["fret"] for d in n]), sigs[0], beat_q, finger=np.array([d["finger"] for d in n]),
                       instrument=Instrument("partisyon", tuple(tuning[k] for k in range(1, 7))))
        n_bar += len(meas)
        bad_bar += sum(sum(p.dur for p in m) != M * TICKS for m in meas)
        _, _, back, _ = score_notes(out)
        key = lambda d, b0=0: (d["bar"] + b0, round(d["tick"]), d["pitch"], d["string"], d["fret"], d["finger"])
        ref = {}
        for d, g in zip(n, grid):
            if g:
                ref[key(d)] = ref.get(key(d), 0) + 1
        b0 = int(round(first / M))
        got = {}
        for d in back:
            got[key(d, b0)] = got.get(key(d, b0), 0) + 1
        tot += sum(ref.values()); ok += sum(min(v, got.get(k, 0)) for k, v in ref.items()); n_sc += 1
    print(f"  {n_sc} partisyon | {tot} nota | birebir {ok / max(tot, 1):.4f} | olcu dolulugu bozuk {bad_bar}/{n_bar} | "
          f"1/24 vurus izgarasi disi (yazilmadi) {off}")
    return ok / max(tot, 1), bad_bar


# ----------------------------------------------------------------- B) uçtan uca
def bar_position(s, beats, downs):
    """Partisyon konumu (vuruş) -> ölçü başından uzaklık (vuruş); ölçü başları vuruş indeksleri."""
    k = np.searchsorted(beats, downs - 1e-6)
    j = np.searchsorted(k, np.floor(s + EPS), side="right") - 1
    return np.where(j >= 0, s - k[np.maximum(j, 0)], np.nan)


def simulate_taps(gb, gd, sigma, lag, miss, extra, rng):
    """Taklit kullanıcı: ilk ölçü başından başlayarak her vuruşa dokunur (ilk dokunuş hep var) + gecikme + sapma,
    kaçan / fazla dokunuşlar."""
    b = gb[gb >= gd[0] - 1e-6]
    t = b + lag + rng.normal(0, sigma, len(b))
    t = t[np.r_[True, rng.random(len(t) - 1) >= miss]]
    return np.sort(np.r_[t, rng.uniform(t[0], t[-1], rng.binomial(len(t), extra))])


def chain_scores(r, det, qm, om, s, gb, gd, sig, single):
    """-> sayaçlar: ölçü içi konum (tümü / tek imli + faz oracle), IOI, ölçü başı F, vuruş F."""
    M, _ = meter_of_signature(sig)
    ours = (qm - r["downbeat_index"]) % M
    ref = bar_position(s, gb, gd)
    v = ~np.isnan(ours) & ~np.isnan(ref)
    hit = int((np.abs(ours[v] - ref[v]) < EPS).sum())
    c = dict(bar=(hit, int(v.sum())), ioi=ioi_accuracy(qm, s, om, SCALES)[:2])
    if single:
        hits = [int((np.abs((ours[v] - k) % M - ref[v]) < EPS).sum()) for k in range(M)]   # faz düzeltmeli (oracle)
        c.update(bar1=(hit, int(v.sum())), bar1f=(max(hits), int(v.sum())), shifted=(int(np.argmax(hits) != 0), 1))
    return c, onset_f(gd, in_span(r["downbeats"], gd)), beat_scores(gb, in_span(r["beats"], gb))[0]


def section_b(models, split, tmp, limit):
    files = (gaps_files("gaps_train")[1] if split == "gaps_val"
             else sorted(glob.glob(os.path.join(CACHE_DIR, split, "*.npz"))))
    files = [f for f in files if os.path.exists(os.path.join(RHY_DIR, os.path.basename(f)))][:limit]
    keys = ("bar", "bar1", "bar1f", "shifted", "ioi", "read", "uniform")
    acc = {cond: (dict((k, [0, 0]) for k in keys), [], []) for cond in ["yari"] + [t[0] for t in TAP_CONDS]}
    for i, f in enumerate(files):
        tid = os.path.basename(f)[:-4]
        with np.load(f) as d:
            cqt = d["cqt"]
        with np.load(os.path.join(RHY_DIR, tid + ".npz")) as g:
            gb, gd, gn, s = g["beats"], g["downbeats"], g["notes"], g["note_beat"]
        sigs = time_signatures(xml_of(tid))
        sig = Counter(sigs).most_common(1)[0][0]
        bpm = tempo_of(gb) * (1 + USER_ERR * (-1) ** i)
        an = analyze(cqt, models, "ses")
        det = an["notes"]
        mi, _ = match_notes([(on, int(p)) for on, _, p in gn], [dict(pitch=int(p)) for _, _, p in det], det[:, 0],
                            ONSET_TOL)
        om = np.where(mi >= 0, det[np.maximum(mi, 0), 0], np.nan)
        rng = np.random.default_rng(i)
        runs = [("yari", rhythm(an, models, sig, bpm))]
        for name, sigma, lag, miss, extra in TAP_CONDS:
            taps = simulate_taps(gb, gd, sigma, lag, miss, extra, rng)
            runs.append((name, rhythm(an, models, sig, taps=taps)))
        for cond, r in runs:
            qm = np.where(mi >= 0, r["q"][np.maximum(mi, 0)], np.nan)
            c, fdi, fbi = chain_scores(r, det, qm, om, s, gb, gd, sig, len(set(sigs)) == 1)
            cnt, fd, fb = acc[cond]
            for k, (x, y) in c.items():
                cnt[k][0] += x; cnt[k][1] += y
            fd.append(fdi); fb.append(fbi)
            if cond == "yari":
                out = os.path.join(tmp, "b.musicxml")
                write_musicxml(out, r["measures"], det[:, 2].astype(int), an["string"], an["fret"], sig, r["beat_q"],
                               bpm=bpm)
                _, lens, back, _ = score_notes(out)
                cnt["read"][0] += len(back); cnt["read"][1] += int((an["string"] >= 0).sum())
                cnt["uniform"][0] += int(len(set(np.round(lens, 6))) == 1); cnt["uniform"][1] += 1
    c, fd, _ = acc["yari"]
    res = {k: a / max(b, 1) for k, (a, b) in c.items()}
    print(f"  {split:<10}{len(files):>4}{res['bar']:>10.3f}{res['bar1']:>10.3f}{res['bar1f']:>10.3f}"
          f"{c['shifted'][0]:>5d}/{c['shifted'][1]:<3d}{c['bar1'][1]:>8d}{res['ioi']:>8.3f}{np.mean(fd):>13.3f}"
          f"{res['read']:>13.3f}{res['uniform']:>13.3f}")
    return len(files), acc


def section_c(tap):
    print("\n" + "=" * 100)
    print("C) TAP MODU (simulasyon) — kullanici ilk olcu basindan her vurusa dokunur; olcu basi = ilk dokunus")
    print("   kosul: (sapma sigma, sabit gecikme, kacan, fazla dokunus)")
    print("=" * 100)
    print(f"  {'':<10}{'kosul':<34}{'vurus F':>9}{'konum':>8}{'tek imli':>10}{'IOI':>8}{'olcu basi F':>13}")
    for split, (n, acc) in tap.items():
        for cond, (c, fd, fb) in acc.items():
            if cond == "yari":
                lab = "yari otomatik (BPM %10 + olcu)"
            else:
                t = next(x for x in TAP_CONDS if x[0] == cond)
                lab = f"{t[0]} ({1000 * t[1]:.0f} ms, +{1000 * t[2]:.0f} ms, %{100 * t[3]:.0f}, %{100 * t[4]:.0f})"
            res = {k: a / max(b, 1) for k, (a, b) in c.items()}
            print(f"  {split:<10}{lab:<34}{np.mean(fb):>9.3f}{res['bar']:>8.3f}{res['bar1']:>10.3f}"
                  f"{res['ioi']:>8.3f}{np.mean(fd):>13.3f}")


def main(limit):
    device = get_device()
    with tempfile.TemporaryDirectory() as tmp:
        section_a(tmp, limit)
        models = load_models(device)
        print("\n" + "=" * 100)
        print(f"B) UCTAN UCA (yari otomatik: gercek medyan tempo x (1 +- {USER_ERR}) + en sik olcu imi; ses modu; deep2 + 1q + "
              f"BeatNet/DBN)")
        print("=" * 100)
        print(f"  {'':<10}{'kayit':>4}{'konum':>10}{'tek imli':>10}{'faz duz.':>10}{'kaymis':>9}{'n':>8}{'IOI':>8}{'olcu basi F':>13}{'geri okuma':>13}"
              f"{'olcu tek tip':>13}")
        tap = {split: section_b(models, split, tmp, limit) for split in ("gaps_val", "gaps_test")}
        section_c(tap)
    print("\nKarar kurali: docs/devlog/README16.md.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="bolum basina en cok kayit / partisyon (duman testi)")
    a = ap.parse_args()
    main(a.limit)
