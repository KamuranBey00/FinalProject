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
from gtab.evaluation.rhythm_metrics import EPS, SCALES, in_span, ioi_accuracy, onset_f, tempo_of
from gtab.models.rhythm_net import TICKS
from gtab.paths import CACHE_DIR
from gtab.pipeline import load_models, transcribe
from gtab.render.musicxml import write_musicxml
from gtab.render.score import build_measures, meter_of_signature
from gtab.utils import get_device

RHY_DIR = os.path.join(CACHE_DIR, "rhythm")
ONSET_TOL = 0.05
USER_ERR = 0.10


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


def section_b(models, split, tmp, limit):
    files = (gaps_files("gaps_train")[1] if split == "gaps_val"
             else sorted(glob.glob(os.path.join(CACHE_DIR, split, "*.npz"))))
    files = [f for f in files if os.path.exists(os.path.join(RHY_DIR, os.path.basename(f)))][:limit]
    c = dict(bar=[0, 0], bar1=[0, 0], ioi=[0, 0], read=[0, 0], uniform=[0, 0])
    fd = []
    for i, f in enumerate(files):
        tid = os.path.basename(f)[:-4]
        with np.load(f) as d:
            cqt = d["cqt"]
        with np.load(os.path.join(RHY_DIR, tid + ".npz")) as g:
            gb, gd, gn, s = g["beats"], g["downbeats"], g["notes"], g["note_beat"]
        sigs = time_signatures(xml_of(tid))
        sig = Counter(sigs).most_common(1)[0][0]
        bpm = tempo_of(gb) * (1 + USER_ERR * (-1) ** i)
        r = transcribe(cqt, models, bpm, sig, "ses")
        det, q = r["notes"], r["q"]
        mi, _ = match_notes([(on, int(p)) for on, _, p in gn], [dict(pitch=int(p)) for _, _, p in det], det[:, 0],
                            ONSET_TOL)
        qm = np.where(mi >= 0, q[np.maximum(mi, 0)], np.nan)
        om = np.where(mi >= 0, det[np.maximum(mi, 0), 0], np.nan)
        M, _ = meter_of_signature(sig)
        ours = (qm - r["downbeat_index"]) % M
        ref = bar_position(s, gb, gd)
        v = ~np.isnan(ours) & ~np.isnan(ref)
        hit = int((np.abs(ours[v] - ref[v]) < EPS).sum())
        c["bar"][0] += hit; c["bar"][1] += int(v.sum())
        if len(set(sigs)) == 1:
            c["bar1"][0] += hit; c["bar1"][1] += int(v.sum())
        a, b, _ = ioi_accuracy(qm, s, om, SCALES)
        c["ioi"][0] += a; c["ioi"][1] += b
        fd.append(onset_f(gd, in_span(r["downbeats"], gd)))
        out = os.path.join(tmp, "b.musicxml")
        write_musicxml(out, r["measures"], det[:, 2].astype(int), r["string"], r["fret"], sig, r["beat_q"], bpm=bpm)
        _, lens, back, _ = score_notes(out)
        c["read"][0] += len(back); c["read"][1] += int((r["string"] >= 0).sum())
        c["uniform"][0] += int(len(set(np.round(lens, 6))) == 1); c["uniform"][1] += 1
    res = {k: a / max(b, 1) for k, (a, b) in c.items()}
    print(f"  {split:<10}{len(files):>4}{res['bar']:>10.3f}{res['bar1']:>10.3f}{c['bar1'][1]:>8d}{res['ioi']:>8.3f}{np.mean(fd):>13.3f}"
          f"{res['read']:>13.3f}{res['uniform']:>13.3f}")


def main(limit):
    device = get_device()
    with tempfile.TemporaryDirectory() as tmp:
        section_a(tmp, limit)
        models = load_models(device)
        print("\n" + "=" * 100)
        print(f"B) UCTAN UCA (yari otomatik: gercek medyan tempo x (1 +- {USER_ERR}) + en sik olcu imi; ses modu; deep2 + 1q + "
              f"BeatNet/DBN)")
        print("=" * 100)
        print(f"  {'':<10}{'kayit':>4}{'konum':>10}{'tek imli':>10}{'n':>8}{'IOI':>8}{'olcu basi F':>13}{'geri okuma':>13}"
              f"{'olcu tek tip':>13}")
        for split in ("gaps_val", "gaps_test"):
            section_b(models, split, tmp, limit)
    print("\nKarar kurali: docs/devlog/README16.md.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="bolum basina en cok kayit / partisyon (duman testi)")
    a = ap.parse_args()
    main(a.limit)
