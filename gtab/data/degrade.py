"""
Katman 6 — Kayıt koşulu bozulmaları (ses dalgası üstünde; 22050 Hz mono). Adım 0'da dayanıklılık ölçümü, Adım 2'de eğitim
artırması. Hepsi uzunluğu korur (etiket kareleri aynı kalır) ve tohumla deterministik.

  reverb  : oda yankısı — üstel sönen gürültü dürtü yanıtı (RT60 sn), kuru/ıslak karışım
  phone   : telefon mikrofonu — alt / üst bant kesilir + otomatik kazanç benzeri sıkıştırma
  noise   : pembe gürültü (oda / klima uğultusu), verilen SNR (dB)
  voice   : insan sesi karışımı — şarkı (VocalSet) ya da konuşma (LibriSpeech) bankasından rastgele kesit, gitar / ses
            oranı snr (dB). Bankalar `get_voices` ile; şarkıcı / konuşmacı ayrık "train" ve "eval" parçaları (eğitim
            artırması "train", ölçüm "eval" bankasını kullanır -> eğitimde duyulan ses ölçümde yok)
"""

import os

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from gtab.config import SAMPLE_RATE
from gtab.paths import RAW_DIR

VOICE_DIR = os.path.join(RAW_DIR, "voices")
CONDITIONS = {                                      # ad -> (tür, parametreler); eval_robust varsayılan koşulları
    "temiz": ("clean", {}),
    "yanki_0.5s": ("reverb", {"rt60": 0.5, "wet": 0.4}),
    "yanki_1.2s": ("reverb", {"rt60": 1.2, "wet": 0.5}),
    "telefon": ("phone", {}),
    "gurultu_20dB": ("noise", {"snr": 20.0}),
    "gurultu_10dB": ("noise", {"snr": 10.0}),
    "sarki_10dB": ("voice", {"bank": "sing", "snr": 10.0}),
    "sarki_0dB": ("voice", {"bank": "sing", "snr": 0.0}),
    "konusma_10dB": ("voice", {"bank": "speech", "snr": 10.0}),
}
VOICE_CONDITIONS = ("sarki_10dB", "sarki_0dB", "konusma_10dB")
_BANKS = {}


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


def load_bank(bank, part):
    """data/raw/voices/<bank>_<part>.npy (22050 Hz, float16, sessizlik atılmış, kesit başına RMS 1)."""
    key = (bank, part)
    if key not in _BANKS:
        path = os.path.join(VOICE_DIR, f"{bank}_{part}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} yok; once: python -m scripts.data.get_voices")
        _BANKS[key] = np.load(path, mmap_mode="r")
    return _BANKS[key]


def voice(y, bank="sing", snr=10.0, part="eval", rng=None):
    rng = rng or np.random.default_rng(0)
    b = load_bank(bank, part)
    s = int(rng.integers(0, max(1, len(b) - len(y))))
    v = np.asarray(b[s:s + len(y)], np.float32)
    v = np.pad(v, (0, len(y) - len(v)), mode="wrap") if len(v) < len(y) else v
    p = (y ** 2).mean()
    return (y + v * np.sqrt(p / max((v ** 2).mean(), 1e-12) / 10 ** (snr / 10))).astype(np.float32)


def degrade(y, kind, params=None, seed=0, part="eval"):
    """y (n,) -> bozulmuş y (n,). kind: clean | reverb | phone | noise | voice (part: ses bankası parçası)."""
    params, rng = params or {}, np.random.default_rng(seed)
    if kind == "clean":
        return y
    if kind == "reverb":
        return reverb(y, rng=rng, **params)
    if kind == "phone":
        return phone(y, **params)
    if kind == "noise":
        return noise(y, rng=rng, **params)
    if kind == "voice":
        return voice(y, rng=rng, part=part, **params)
    raise ValueError(kind)


def random_chain(rng, voices=True):
    """Eğitim artırması: rastgele 1–2 bozulma zinciri [(tür, parametreler), ...] (ölçüm koşullarından geniş aralık)."""
    pool = ["reverb", "phone", "noise"] + (["voice", "voice"] if voices else [])
    chain = []
    for kind in rng.choice(pool, size=int(rng.integers(1, 3)), replace=False):
        if kind == "reverb":
            chain.append((kind, {"rt60": float(rng.uniform(0.2, 1.5)), "wet": float(rng.uniform(0.2, 0.6))}))
        elif kind == "phone":
            chain.append((kind, {"lo": float(rng.uniform(80, 300)), "hi": float(rng.uniform(4000, 9000)),
                                 "ratio": float(rng.uniform(2, 6))}))
        elif kind == "noise":
            chain.append((kind, {"snr": float(rng.uniform(5, 30))}))
        else:
            chain.append((kind, {"bank": str(rng.choice(["sing", "speech"])), "snr": float(rng.uniform(-3, 20))}))
    return chain


def apply_chain(y, chain, seed=0, part="train"):
    for i, (kind, params) in enumerate(chain):
        y = degrade(y, kind, params, seed=seed * 7 + i, part=part)
    return y
