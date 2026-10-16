"""
Katman 4 Faz 0 — Çıkarım parmak izi: "yeni modül eklendi, eski yol değişmedi mi?"

Parmak izi = kafa başına (ortalama, std) + tab argmax sınıf histogramı, kayıt başına.
Metrik koşusundan hızlı ve daha hassastır (kare düzeyinde çıktıyı karşılaştırır).
CLI: `scripts/eval/check_inference_regression.py`.

Tab kafası softmax olduğu için tüm olasılıkların ortalaması 1/C sabittir; o yüzden tab
kafasında ortalama yerine **güven** (argmax olasılığı) kullanılır.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

TOL = 1e-4                  # kafa ortalama/std farki (CPU'da cikti deterministik)
HIST_TOL = 1e-3             # argmax histogrami: farkli kare orani
SETUP_KEYS = ("ckpt", "split", "onset_head", "deep", "pitch_onset_head")
HEADS = ("tab", "onset", "offset", "pitch_onset")


def stat(a) -> Dict[str, float]:
    return {"mean": round(float(np.mean(a)), 8), "std": round(float(np.std(a)), 8)}


def tab_stat(probs) -> Dict[str, float]:
    return {"mean": round(float(probs.max(-1).mean()), 8), "std": round(float(np.std(probs)), 8)}


def track_fingerprint(probs, ons, offs, pons) -> Dict[str, object]:
    """Bir kaydın kafa çıktıları -> parmak izi sözlüğü."""
    hist = np.bincount(probs.argmax(-1).reshape(-1), minlength=int(probs.shape[-1])).tolist()
    return {"frames": int(probs.shape[0]), "tab": tab_stat(probs), "argmax_hist": hist,
            "onset": stat(ons) if ons is not None else None,
            "offset": stat(offs) if offs is not None else None,
            "pitch_onset": stat(pons) if pons is not None else None}


def _cmp_stat(name: str, ref: Optional[Dict[str, float]], cur: Optional[Dict[str, float]],
              tol: float, bad: List[str]) -> None:
    if (ref is None) != (cur is None):
        bad.append(f"{name}: kafa varligi degisti (ref={ref is not None}, simdi={cur is not None})")
        return
    if ref is None:
        return
    for k in ("mean", "std"):
        d = abs(ref[k] - cur[k])
        if d > tol:
            bad.append(f"{name}.{k}: {ref[k]:.8f} -> {cur[k]:.8f} (|delta| {d:.2e} > {tol:.0e})")


def compare(ref: Dict[str, object], cur: Dict[str, object], tol: float = TOL,
            hist_tol: float = HIST_TOL) -> List[str]:
    """İki parmak izi -> fark listesi (boş liste = regresyon yok)."""
    bad: List[str] = []
    for key in SETUP_KEYS:
        if ref.get(key) != cur.get(key):
            bad.append(f"kurulum farki {key}: {ref.get(key)} -> {cur.get(key)}")
    if set(ref["kayitlar"]) != set(cur["kayitlar"]):
        bad.append(f"kayit listesi degisti: {sorted(ref['kayitlar'])} -> {sorted(cur['kayitlar'])}")
        return bad
    for tid, r in ref["kayitlar"].items():
        c = cur["kayitlar"][tid]
        if r["frames"] != c["frames"]:
            bad.append(f"{tid}: kare sayisi {r['frames']} -> {c['frames']}")
            continue
        for head in HEADS:
            _cmp_stat(f"{tid}.{head}", r[head], c[head], tol, bad)
        bad += _cmp_hist(tid, r, c, hist_tol)
    return bad


def _cmp_hist(tid: str, ref: Dict[str, object], cur: Dict[str, object], hist_tol: float) -> List[str]:
    """Argmax histogramı: toplam sapma / (kare x tel) oranı."""
    a: Sequence[int] = ref["argmax_hist"]; b: Sequence[int] = cur["argmax_hist"]
    if len(a) != len(b):
        return [f"{tid}.argmax_hist: sinif sayisi {len(a)} -> {len(b)}"]
    total = int(ref["frames"]) * 6
    diff = sum(abs(x - y) for x, y in zip(a, b)) / max(total, 1)
    return [f"{tid}.argmax_hist: farkli kare orani {diff:.2e} > {hist_tol:.0e}"] if diff > hist_tol else []
