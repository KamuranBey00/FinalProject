"""
Katman 3.10 Adım 3c-0 — Hızlı tekrarlar için GİRİŞ ÖZELLİĞİ ölçümü (model yok, eğitim yok).

Soru: gömülü hızlı aynı perde tekrarlarını (önceki nota ses kesilmeden ya da <= 3 kare önce
bitmiş) sesin kendisi ayırt edebiliyor mu, ve hangi zaman çözünürlüğünde?
Mevcut CQT (24 bin/oktav, Q~34) pencere uzunluğu ~ Q/f: E2 ~410 ms, E4 ~100 ms ->
pes/orta registerde 100 ms'lik iki vuruş tek enerji lekesi. Hop (512) bunu değiştirmez.

Her aday özellik için perde başına enerji E_p(t) ve yükseliş rise_k(t) = E_p(t) - min(E_p(t-k..t-1)).
Karşılaştırılan iki kitle (etiketlerden, modelden bağımsız):
  tekrar onset'i : aynı perde önceki 1..4 karede çalıyor + bu karede onset   (değer: max rise, t-1..t+2)
  süren nota     : nota içinde, onset'ten >= 4 kare sonra, ±3 karede o perdede onset yok  (aynı pencere)
Rapor: medyanlar, AUC, süren notada %5 / %10 yanlış alarm eşiğinde yakalanan tekrar oranı,
IOI ve register kırılımı (%5 yanlış alarmda).

Adaylar (hepsi hop 512 -> kare ızgarası ve tüm çözümleme kuralları aynı kalır):
  cqt        : önbellekteki CQT (mevcut, energy_rise ile aynı; k=3)
  cqt_k1     : aynı CQT, k=1
  cqt_fs05   : CQT filter_scale=0.5 (pencere yarı)
  stft1024   : STFT n_fft=1024 (~46 ms), harmonikler 1..4
  stft2048   : STFT n_fft=2048 (~93 ms), harmonikler 1..4
Katman 3.11 Adım 2 (önbellekteki CQT'den; yeni önbellek gerekmez):
  cqt_decay  : sönüm telafili yükseliş: E(t) - [E(t-1) + (E(t-1) - E(t-3)) / 2]  (beklenen sönüme göre sapma)
  hf_flux    : ~2 kHz üstü CQT binlerinde (pencere ~10 ms) pozitif akı ortalaması; perdeye özgü değil
               (tırnak/parmak temasının geniş bant izi), tüm perdelere aynı değer
  decay+hf   : cqt_decay + hf_flux (basit birleşim)

Çalıştırma:
    python -m scripts.eval.diagnose_onset_features --splits gaps_val val_comp --limit 12
    python -m scripts.eval.diagnose_onset_features --splits gaps_val val_comp --fast   # yalnız önbellek özellikleri
"""

import argparse
import os

import librosa
import numpy as np
import pandas as pd

from gtab.config import SAMPLE_RATE, HOP_LENGTH, CQT_FMIN_HZ, CQT_BINS_PER_OCTAVE, CQT_N_BINS, \
    GUITARSET_DATA_HOME
from gtab.core.instrument import STANDARD_6
from gtab.data.gaps import gaps_files, GAPS_DIR
from gtab.decoding.viterbi import energy_rise
from gtab.paths import split_dir

INSTR = STANDARD_6
LO, HI = INSTR.pitch_range()
GAP = 3


# ----------------------------------------------------------------- dosyalar
def split_tracks(split, limit=None):
    """-> [(npz yolu, wav yolu)]"""
    if split == "gaps_val":
        files = gaps_files("gaps_train")[1]
        meta = pd.read_csv(os.path.join(GAPS_DIR, "gaps_metadata_with_splits.csv")).set_index("id")
        wav = lambda f: os.path.join(GAPS_DIR, meta.loc[os.path.basename(f)[:-4], "audio_path"])
    else:
        import glob
        import mirdata
        files = sorted(glob.glob(os.path.join(split_dir(split), "*.npz")))
        gs = mirdata.initialize("guitarset", data_home=GUITARSET_DATA_HOME)
        tr = gs.load_tracks()
        wav = lambda f: tr[os.path.basename(f)[:-4]].audio_mic_path
    files = files[:limit] if limit else files
    return [(f, wav(f)) for f in files]


def _npz_only(split, limit=None):
    """--fast: ses yolu gerekmez (mirdata / GAPS ses dosyası okunmaz)."""
    import glob
    files = gaps_files("gaps_train")[1] if split == "gaps_val" else \
        sorted(glob.glob(os.path.join(split_dir(split), "*.npz")))
    files = files[:limit] if limit else files
    return [(f, None) for f in files]


# ----------------------------------------------------------------- özellikler
def rise(E, k):
    prev = np.full_like(E, np.inf)
    for j in range(1, k + 1):
        sh = np.full_like(E, np.inf); sh[j:] = E[:-j]
        prev = np.minimum(prev, sh)
    return np.maximum(E - np.where(np.isinf(prev), E, prev), 0.0)


def cqt_energy(c, harmonics=(0, 24, 38), bps=2, fmin_midi=24):
    T, F = c.shape
    E = np.zeros((T, HI - LO + 1), np.float32)
    for i, p in enumerate(range(LO, HI + 1)):
        b = bps * (p - fmin_midi)
        vals = [c[:, max(0, b + h - 1):min(F, b + h + 2)].max(1) for h in harmonics if b + h < F]
        E[:, i] = np.mean(vals, axis=0)
    return E


def stft_energy(y, n_fft, harmonics=(1, 2, 3, 4)):
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=n_fft, hop_length=HOP_LENGTH)), ref=np.max).T
    freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=n_fft)
    df = freqs[1]
    E = np.zeros((S.shape[0], HI - LO + 1), np.float32)
    for i, p in enumerate(range(LO, HI + 1)):
        f0 = librosa.midi_to_hz(p); vals = []
        for h in harmonics:
            b = int(round(h * f0 / df))
            if b + 1 < S.shape[1]:
                vals.append(S[:, max(0, b - 1):b + 2].max(1))
        E[:, i] = np.mean(vals, axis=0)
    return E


HF_BIN = int(round(CQT_BINS_PER_OCTAVE * np.log2(2000.0 / CQT_FMIN_HZ)))   # ~2 kHz


def decay_rise(E):
    """Katman 3.11: beklenen doğrusal (dB) sönüme göre pozitif sapma."""
    pred = np.full_like(E, np.inf)
    pred[3:] = E[2:-1] + (E[2:-1] - E[:-3]) / 2.0
    return np.maximum(E - np.where(np.isinf(pred), E, pred), 0.0)


def hf_flux(c):
    """Katman 3.11: ~2 kHz üstü binlerde kare-kare pozitif dB farkının ortalaması -> (T, P)."""
    hf = np.asarray(c, np.float32)[:, HF_BIN:]
    d = np.zeros(len(hf), np.float32); d[1:] = np.maximum(hf[1:] - hf[:-1], 0).mean(1)
    return np.repeat(d[:, None], HI - LO + 1, axis=1)


def features(npz_cqt, wav, fast=False):
    c = np.asarray(npz_cqt, np.float32)
    E_cqt = cqt_energy(c)
    dec, hf = decay_rise(E_cqt), hf_flux(c)
    out = {
        "cqt": energy_rise(npz_cqt, INSTR),
        "cqt_k1": rise(E_cqt, 1),
        "cqt_decay": dec,
        "hf_flux": hf,
        "decay+hf": dec + hf,
    }
    if fast:
        return out
    y, _ = librosa.load(wav, sr=SAMPLE_RATE, mono=True)
    c05 = librosa.amplitude_to_db(np.abs(librosa.cqt(
        y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, fmin=CQT_FMIN_HZ, bins_per_octave=CQT_BINS_PER_OCTAVE,
        n_bins=CQT_N_BINS, filter_scale=0.5)), ref=np.max).T
    out.update({
        "cqt_fs05": rise(cqt_energy(c05), 3),
        "stft1024": rise(stft_energy(y, 1024), 3),
        "stft2048": rise(stft_energy(y, 2048), 3),
    })
    return out


# ----------------------------------------------------------------- kitleler
def populations(frame, onset):
    """-> tekrar onset'leri [(t, p, ioi_ms)], süren nota kareleri [(t, p)]"""
    fr, on = frame > 0.5, onset > 0.5
    T, P = fr.shape
    reps, sus = [], []
    for p in range(P):
        ts = np.nonzero(on[:, p])[0]
        last_on = -10 ** 9
        for t in ts:
            if t > 0 and fr[max(0, t - GAP - 1):t, p].any():
                reps.append((t, p, (t - last_on) * 1000 * HOP_LENGTH / SAMPLE_RATE))
            last_on = t
        near = np.zeros(T, bool)
        for t in ts:
            near[max(0, t - 3):t + 4] = True
        since = np.full(T, 10 ** 9)
        lt = -10 ** 9
        for t in range(T):
            if on[t, p]:
                lt = t
            since[t] = t - lt
        idx = np.nonzero(fr[:, p] & ~near & (since >= 4))[0]
        sus += [(t, p) for t in idx[::4]]                        # seyrelt (komşu kareler bağımlı)
    return reps, sus


def win_max(R, t, p):
    return R[max(0, t - 1):t + 3, p].max()


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg]); r = allv.argsort().argsort() + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main(splits, limit, fast=False):
    for split in splits:
        tracks = split_tracks(split, limit) if not fast else [(f, None) for f, _ in _npz_only(split, limit)]
        pos = {}; neg = {}; meta = []
        for k, (f, wav) in enumerate(tracks, 1):
            d = np.load(f)
            feats = features(d["cqt"], wav, fast)
            reps, sus = populations(d["frame"], d["onset"])
            T = min(len(d["frame"]), *(len(v) for v in feats.values()))
            reps = [r for r in reps if r[0] < T]; sus = [s for s in sus if s[0] < T]
            for name, R in feats.items():
                pos.setdefault(name, []).extend(win_max(R, t, p) for t, p, _ in reps)
                neg.setdefault(name, []).extend(win_max(R, t, p) for t, p in sus)
            meta += [(ioi, LO + p) for _, p, ioi in reps]
            print(f"  [{split} {k}/{len(tracks)}] {os.path.basename(f)}  tekrar {len(reps)}  suren {len(sus)}", flush=True)
        ioi = np.array([m[0] for m in meta]); pitch = np.array([m[1] for m in meta])
        print(f"\n{'=' * 100}\n{split}: {len(tracks)} kayit | tekrar onset {len(ioi)} | suren nota karesi {len(neg['cqt'])}\n{'=' * 100}")
        print(f"{'ozellik':10s} {'tekrar med':>10s} {'suren med':>9s} {'AUC':>6s} {'@FA5%':>7s} {'@FA10%':>7s} | "
              f"@FA5%: {'<100ms':>7s} {'100-200':>7s} {'>=200':>6s} | {'pes<52':>7s} {'orta':>6s} {'tiz64+':>7s}")
        groups = [("<100", ioi < 100), ("100-200", (ioi >= 100) & (ioi < 200)), (">=200", ioi >= 200),
                  ("pes", pitch < 52), ("orta", (pitch >= 52) & (pitch < 64)), ("tiz", pitch >= 64)]
        for name in pos:
            a, b = np.array(pos[name]), np.array(neg[name])
            t5, t10 = np.percentile(b, 95), np.percentile(b, 90)
            g = "  ".join(f"{100 * np.mean(a[m] > t5):6.0f}%" if m.any() else "      -" for _, m in groups)
            print(f"{name:10s} {np.median(a):10.1f} {np.median(b):9.1f} {auc(a, b):6.3f} "
                  f"{100 * np.mean(a > t5):6.0f}% {100 * np.mean(a > t10):6.0f}% |       {g}")
        print("n: " + "  ".join(f"{n}={int(m.sum())}" for n, m in groups))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["gaps_val", "val_comp"])
    ap.add_argument("--limit", type=int, default=None, help="split basina en fazla bu kadar kayit")
    ap.add_argument("--fast", action="store_true", help="yalniz onbellek (CQT) ozellikleri; ses okunmaz")
    a = ap.parse_args()
    main(a.splits, a.limit, a.fast)
