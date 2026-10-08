"""
Perde/nota değerlendirme çekirdeği (eval_pitch ve diagnose_pitch ortak kullanır).

- roll_to_notes : gerçek frame/onset roll'ları -> nota listesi
- track_scores  : bir kayıt için kare/nota/polifoni sayımları
- evaluate_split: eşik ızgarasında toplu skorlar
- calibrate     : doğrulama verisinde (perde eşiği, çözümleme yöntemi) seçimi, nota F1 ile
- selection_split: alan-içi kalibrasyon kuralı (GuitarSet -> val, GAPS -> gaps_val)
- load_split    : önbellek split'i (+ sanal 'gaps_val') -> model çıktıları
"""

import glob
import os

import mir_eval
import numpy as np

from gtab.config import FRAME_RATE
from gtab.core.instrument import STANDARD_6
from gtab.data.gaps import gaps_files
from gtab.data.torch_dataset import _normalize
from gtab.decoding.viterbi import energy_rise, pitch_matrix, segment
from gtab.evaluation.metrics import prf
from gtab.models.inference import predict_probs, predict_heads
from gtab.paths import CACHE_DIR

INSTR = STANDARD_6
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ----------------------------------------------------------------- yardımcılar
def roll_to_notes(frame, onset, lo):
    """Gerçek roll'lardan nota listesi: aktif koşular, onset işaretlerinde bölünür."""
    notes = []
    T, P = frame.shape
    for p in range(P):
        t = 0
        while t < T:
            if frame[t, p] > 0:
                s = t; t += 1
                while t < T and frame[t, p] > 0 and onset[t, p] == 0:
                    t += 1
                notes.append((s, t, lo + p))
            else:
                t += 1
    return notes


def to_mir(notes):
    if not notes:
        return np.zeros((0, 2)), np.zeros(0)
    iv = np.array([[a / FRAME_RATE, max(b, a + 1) / FRAME_RATE] for a, b, _ in notes])
    hz = mir_eval.util.midi_to_hz(np.array([p for _, _, p in notes], float))
    return iv, hz


def track_scores(probs, ons, frame, onset, thr, onset_thr=None, dec=None, rise=None, offs=None):
    """
    -> dict(frame tp/fp/fn, note tp-sayıları, polifoni grupları)
    ons (T,S) verilirse (onset kafalı model) notalar onset ile başlatılır ve kare
    tahmini de bu notalardan türetilir (Onsets & Frames); yoksa eski kare-eşik yolu.
    """
    pm, lo = pitch_matrix(probs, INSTR)
    n = min(len(pm), len(frame))
    frame, onset = frame[:n] > 0, onset[:n] > 0
    est = segment(probs[:n], None if ons is None else ons[:n], thr, onset_thr, INSTR,
                  rise=None if rise is None else rise[:n], offsets=None if offs is None else offs[:n],
                  **(dec or {}))
    if ons is None or onset_thr is None:
        pred = pm[:n] > thr
    else:
        pred = np.zeros_like(frame)
        for a, b, p in est:
            pred[a:b, p - lo] = True

    tp = int((pred & frame).sum()); fp = int((pred & ~frame).sum()); fn = int((~pred & frame).sum())

    ref = roll_to_notes(frame, onset, lo)
    ri, rp = to_mir(ref); ei, ep = to_mir(est)
    if len(ref) and len(est):
        matched = len(mir_eval.transcription.match_notes(ri, rp, ei, ep, onset_tolerance=0.05,
                                                          offset_ratio=None))
        # Katman 3.10 kontrolü: 100 ms toleransla (zamanlama hatası mı, perde hatası mı?)
        matched100 = len(mir_eval.transcription.match_notes(ri, rp, ei, ep, onset_tolerance=0.10,
                                                             offset_ratio=None))
    else:
        matched = matched100 = 0

    poly = {}
    k = frame.sum(1)
    for name, mask in (("1", k == 1), ("2", k == 2), ("3", k == 3), ("4+", k >= 4)):
        if mask.any():
            pf, ff = pred[mask], frame[mask]
            poly[name] = (int((pf & ff).sum()), int((pf & ~ff).sum()), int((~pf & ff).sum()))
    return {"frame": (tp, fp, fn), "note": (matched, len(est), len(ref)),
            "note100": (matched100, len(est), len(ref)), "poly": poly}


def aggregate(results):
    F = np.zeros(3); N = np.zeros(3); N100 = np.zeros(3); poly = {}
    for r in results:
        F += r["frame"]; N += r["note"]; N100 += r["note100"]
        for g, v in r["poly"].items():
            poly[g] = poly.get(g, np.zeros(3)) + v
    def nprf(m, ne, nr):
        p, r = m / max(ne, 1), m / max(nr, 1)
        return (p, r, 2 * p * r / max(p + r, 1e-9))
    return {"frame": prf(*F), "note": nprf(*N), "note100": nprf(*N100),
            "poly": {g: prf(*v) for g, v in poly.items()}}


def load_split(split, model, ck, kind, device, limit=None):
    if split == "gaps_val":                       # sanal split: gaps_train'in icraci-ayrik dogrulamasi
        files = gaps_files("gaps_train")[1][:limit]
    else:
        files = sorted(glob.glob(os.path.join(CACHE_DIR, split, "*.npz")))[:limit]
    if not files:
        raise FileNotFoundError(f"{CACHE_DIR}/{split} bos.")
    data = []
    for f in files:
        with np.load(f) as d:
            cqt = _normalize(d["cqt"])
            if hasattr(model, "onset_head"):
                probs, ons, offs = predict_heads(model, cqt, device)
            else:
                probs, ons, offs = predict_probs(model, ck, kind, cqt, device), None, None
            # 5. öğe: perde başına CQT enerji yükselişi (Adım 3 revizyonu); 6.: tel offset'i (Adım 3c)
            data.append((probs, ons, d["frame"], d["onset"], energy_rise(d["cqt"], INSTR), offs))
    return data


def evaluate_split(data, thresholds=THRESHOLDS, onset_thr=None, dec=None):
    return {thr: aggregate([track_scores(p, o, fr, on, thr, onset_thr, dec, *rest)
                            for p, o, fr, on, *rest in data])
            for thr in thresholds}


def selection_split(split, select_split):
    """auto -> test setinin alanına göre doğrulama split'i."""
    if select_split != "auto":
        return select_split
    return "gaps_val" if split.startswith("gaps") else "val"


# Katman 3.10: onset çözümlemesinin ek kuralları için arama ızgarası
DEC_GRID = [dict(off_ratio=o, refractory=r, fallback=f)
            for o in (1.0, 0.7, 0.5) for r in (0, 3, 6) for f in (0, 10)]
NO_DEC = dict(off_ratio=1.0, refractory=0, fallback=0)          # = Katman 3.9 davranışı
# Adım 2: kurallar seçildikten sonra "onset tepe noktasından başlat" ayrıca denenir
# Adım 3b: ardından "nota sürerken yeniden vuruş" için onset tepe eşiği
REATTACK_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)
# Adım 3 revizyonu: enerji yükselişi kanıtı (dB); refrakter bu aşamada 0'a da çekilebilir
RISE_GRID = [dict(rise_keep=k, rise_split=sp, refractory=r)
             for k in (0.0, 3.0, 6.0) for sp in (0.0, 4.0, 8.0) for r in (None, 0)
             if (k, sp, r) != (0.0, 0.0, None)]
# Adım 3c: offset kafası varsa nota bitişi için offset eşiği (0 = kapalı)
OFFSET_GRID = (0.3, 0.5, 0.7)
# Katman 3.11: yerel tepe başlangıçları (pick_peaks) — vadi derinliği x en az aralık (kare)
PEAK_GRID = [dict(peak_pick=True, prominence=pr, min_dist=md)
             for pr in (0.05, 0.1, 0.15, 0.25) for md in (1, 2, 3)]


def score(r, criterion="note"):
    """Kalibrasyon ölçütü: 'note' = nota F1 (Katman 3.9); 'mix' = (nota F1 + kare F1) / 2."""
    return r["note"][2] if criterion == "note" else 0.5 * (r["note"][2] + r["frame"][2])


def _try(best_on, data, dec, criterion, thresholds=None, onset_thr=None):
    """Tek aday: (skor, eşik, onset eşiği, dec) — mevcut en iyiden iyiyse onu döndürür."""
    _, t0, o0, _ = best_on
    o = o0 if onset_thr is None else onset_thr
    sel = evaluate_split(data, thresholds=thresholds or [t0], onset_thr=o, dec=dec)
    t = max(sel, key=lambda t: score(sel[t], criterion))
    return (score(sel[t], criterion), t, o, dec) if score(sel[t], criterion) > best_on[0] else best_on


def calibrate_peaks(data, best_on, criterion="note"):
    """
    Katman 3.11 Adım 1 — 3.10 seçiminin üstüne (yalnızca doğrulama verisinde):
      a) perde birleştirme: max | noisyor (eğitimle uyumlu)
      b) yerel tepe başlangıçları: PEAK_GRID x refrakter {seçili, 0}
      c) yakın tekrarı kesen kapıları gevşetme: yeniden vuruş 0, enerji kabul 0
      d) onset eşiği {0.05, 0.1, 0.2, 0.3} ve perde eşiği ±0.1 yeniden
    Her aday ancak doğrulama skorunu artırırsa alınır -> 3.10 seçimi her zaman aday.
    """
    d0 = best_on[3]
    best_on = _try(best_on, data, dict(d0, combine="noisyor"), criterion)
    d0 = best_on[3]
    for g in PEAK_GRID:
        best_on = _try(best_on, data, dict(d0, **g), criterion)
        if d0.get("refractory", 0):
            best_on = _try(best_on, data, dict(d0, **g, refractory=0), criterion)
    d0 = best_on[3]
    if d0.get("peak_pick"):
        for k in ("reattack", "rise_keep"):
            if d0.get(k, 0):
                best_on = _try(best_on, data, dict(best_on[3], **{k: 0.0}), criterion)
        for o in (0.05, 0.1, 0.2, 0.3):
            if o != best_on[2]:
                best_on = _try(best_on, data, best_on[3], criterion, onset_thr=o)
        t0 = best_on[1]
        near = sorted({round(min(0.9, max(0.3, t0 + d)), 1) for d in (-0.1, 0.0, 0.1)})
        best_on = _try(best_on, data, best_on[3], criterion, thresholds=near)
    return best_on


def calibrate(data, on_grid, dec_grid=None, criterion="note", peak_search=True):
    """
    Doğrulama verisinde seçim (test setine hiç bakılmaz):
      1) çözümleme yöntemi (kare-eşik / onset@eşik) + perde eşiği
      2) dec_grid verildiyse: 1. aşamadaki EN İYİ ONSET eşiğiyle ek kurallar
         (histerezis, refrakter, yedek) seçilen eşiğin ±0.1 komşuluğunda aranır;
         1. aşamada kare-eşik kazanmış olsa bile (kurallarla onset onu geçebilir).
    criterion: 'note' (varsayılan, 3.9 ile aynı) | 'mix' (nota + kare F1 ortalaması;
               nota sonunu/kuyruğu da ödüllendirir).
    -> (doğrulama skoru, perde eşiği, onset eşiği | None, dec sözlüğü)
    """
    best = (-1, None, None, dict(NO_DEC))
    best_on = None                                 # en iyi onset seçeneği (sonraki aşamalar için)
    for ot in on_grid:
        sel = evaluate_split(data, onset_thr=ot)
        t = max(sel, key=lambda t: score(sel[t], criterion))
        cand = (score(sel[t], criterion), t, ot, dict(NO_DEC))
        if cand[0] > best[0]:
            best = cand
        if ot is not None and (best_on is None or cand[0] > best_on[0]):
            best_on = cand
    if dec_grid and best_on is not None:
        # 2. aşama: onset adayı kendi içinde iyileştirilir (kare-eşik kazanmış olsa bile),
        # sonunda kare-eşik ile karşılaştırılır.
        _, thr, ot, _ = best_on
        near = sorted({round(min(0.9, max(0.3, thr + d)), 1) for d in (-0.1, 0.0, 0.1)})
        for dec in dec_grid:
            sel = evaluate_split(data, thresholds=near, onset_thr=ot, dec=dec)
            t = max(sel, key=lambda t: score(sel[t], criterion))
            if score(sel[t], criterion) > best_on[0]:
                best_on = (score(sel[t], criterion), t, ot, dict(dec))
        # tepe noktasından başlatma (Adım 2)
        f0, t0, o0, d0 = best_on
        dec = dict(d0, peak=True)
        sel = evaluate_split(data, thresholds=[t0], onset_thr=o0, dec=dec)
        if score(sel[t0], criterion) > f0:
            best_on = (score(sel[t0], criterion), t0, o0, dec)
        # yeniden vuruş eşiği (Adım 3b)
        f0, t0, o0, d0 = best_on
        for ra in REATTACK_GRID:
            dec = dict(d0, reattack=ra)
            sel = evaluate_split(data, thresholds=[t0], onset_thr=o0, dec=dec)
            if score(sel[t0], criterion) > best_on[0]:
                best_on = (score(sel[t0], criterion), t0, o0, dec)
        # enerji kanıtı + refrakteri kaldırma (hızlı tekrarlar)
        f0, t0, o0, d0 = best_on
        for g in RISE_GRID:
            dec = dict(d0, rise_keep=g["rise_keep"], rise_split=g["rise_split"])
            if g["refractory"] is not None:
                dec["refractory"] = g["refractory"]
            sel = evaluate_split(data, thresholds=[t0], onset_thr=o0, dec=dec)
            if score(sel[t0], criterion) > best_on[0]:
                best_on = (score(sel[t0], criterion), t0, o0, dec)
        # offset kafası (Adım 3c): yalnızca model offset veriyorsa
        if len(data[0]) > 5 and data[0][5] is not None:
            f0, t0, o0, d0 = best_on
            for oft in OFFSET_GRID:
                dec = dict(d0, offset_threshold=oft)
                sel = evaluate_split(data, thresholds=[t0], onset_thr=o0, dec=dec)
                if score(sel[t0], criterion) > best_on[0]:
                    best_on = (score(sel[t0], criterion), t0, o0, dec)
        if peak_search:                            # Katman 3.11 Adım 1
            best_on = calibrate_peaks(data, best_on, criterion)
        if best_on[0] > best[0]:
            best = best_on
    return best
