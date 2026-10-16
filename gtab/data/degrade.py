"""
Katman 6 — Kayıt koşulu bozulmaları (ses dalgası üstünde; 22050 Hz mono). Adım 0'da dayanıklılık ölçümü, gerekirse Adım 2'de
eğitim artırması için. Hepsi uzunluğu korur (etiket kareleri aynı kalır) ve tohumla deterministik.

  reverb  : oda yankısı — üstel sönen gürültü dürtü yanıtı (RT60 sn), kuru/ıslak karışım
  phone   : telefon mikrofonu — 150 Hz altı ve 7 kHz üstü kesilir + otomatik kazanç benzeri sıkıştırma
  noise   : pembe gürültü (oda / klima uğultusu), verilen SNR (dB)
"""

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from gtab.config import SAMPLE_RATE

CONDITIONS = {                                      # ad -> (tür, parametreler); eval_robust varsayılan koşulları
    "temiz": ("clean", {}),
    "yanki_0.5s": ("reverb", {"rt60": 0.5, "wet": 0.4}),
    "yanki_1.2s": ("reverb", {"rt60": 1.2, "wet": 0.5}),
    "telefon": ("phone", {}),
    "gurultu_20dB": ("noise", {"snr": 20.0}),
    "gurultu_10dB": ("noise", {"snr": 10.0}),
}


def reverb(y, rt60=0.8, wet=0.4, rng=None, sr=SAMPLE_RATE):
    rng = rng or np.random.default_rng(0)
    t = np.arange(int(rt60 * sr)) / sr
    ir = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt60)        # -60 dB rt60'ta
    ir /= np.sqrt((ir ** 2).sum())
    w = fftconvolve(y, ir)[:len(y)]
    out = (1 - wet) * y + wet * w * np.sqrt((y ** 2).mean() / max((w ** 2).mean(), 1e-12))
    return out.astype(np.float32)


def phone(y, lo=150.0, hi=7000.0, ratio=4.0, sr=SAMPLE_RATE):
    out = sosfilt(butter(4, [lo, hi], btype="band", fs=sr, output="sos"), y)
    env = np.sqrt(np.convolve(out ** 2, np.ones(sr // 20) / (sr // 20), mode="same")) + 1e-6
    ref = np.percentile(env, 90)
    gain = np.where(env > ref / 4, (env / (ref / 4)) ** (1 / ratio - 1), 1.0)   # eşik üstünü sıkıştır
    return (out * gain).astype(np.float32)


def pink_noise(n, rng):
    f = np.fft.rfftfreq(n)
    spec = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) / np.sqrt(np.maximum(f, f[1]))
    x = np.fft.irfft(spec, n)
    return x / x.std()


def noise(y, snr=20.0, rng=None):
    rng = rng or np.random.default_rng(0)
    p = (y ** 2).mean()
    return (y + pink_noise(len(y), rng) * np.sqrt(p / 10 ** (snr / 10))).astype(np.float32)


def degrade(y, kind, params=None, seed=0):
    """y (n,) -> bozulmuş y (n,). kind: clean | reverb | phone | noise."""
    params, rng = params or {}, np.random.default_rng(seed)
    if kind == "clean":
        return y
    if kind == "reverb":
        return reverb(y, rng=rng, **params)
    if kind == "phone":
        return phone(y, **params)
    if kind == "noise":
        return noise(y, rng=rng, **params)
    raise ValueError(kind)
