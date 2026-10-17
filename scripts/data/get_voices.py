"""
Katman 6 Adım 2 — İnsan sesi bankaları (şarkı + konuşma; eğitim artırması ve ölçüm için, gtab/data/degrade.py "voice").

Kaynaklar (ikisi de CC BY 4.0):
  şarkı   : VocalSet (Wilkins vd., ISMIR 2018; zenodo 1193957, 2.1 GB) — 20 şarkıcı, egzersiz + şarkı kesitleri
  konuşma : LibriSpeech dev-clean (Panayotov vd., 2015; openslr 12, 0.34 GB) — 40 konuşmacı
Arşiv indirilir, içinden DOĞRUDAN okunur (açılmaz), 22050 Hz mono'ya çevrilir, sessizlik atılır, kesit başına RMS 1 ->
data/raw/voices/{sing,speech}_{train,eval}.npy (float16). Şarkıcı / konuşmacı ayrık: eval = son 4 şarkıcı (female8-9,
male10-11) ve son 8 konuşmacı. Arşiv sonda silinir (disk dar; --keep ile tutulur). Süre: indirme + ~10 dk.

Çalıştırma:
    python -m scripts.data.get_voices
    python -m scripts.data.get_voices --minutes 40 10 --keep
"""

import argparse
import io
import os
import re
import tarfile
import urllib.request
import zipfile

import librosa
import numpy as np
import soundfile as sf

from gtab.config import SAMPLE_RATE
from gtab.data.degrade import VOICE_DIR

SOURCES = {
    "sing": ("https://zenodo.org/records/1193957/files/VocalSet.zip?download=1", "VocalSet.zip"),
    "speech": ("https://www.openslr.org/resources/12/dev-clean.tar.gz", "dev-clean.tar.gz"),
}
EVAL_SINGERS = {"female8", "female9", "male10", "male11"}
N_EVAL_SPEAKERS = 8


def download(url, path, tries=20):
    """Kopan bağlantıda .part dosyasından devam eder (HTTP Range)."""
    if os.path.exists(path):
        return
    print(f"indiriliyor: {url}")
    part = path + ".part"
    for t in range(tries):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:              # sunucu devamı desteklemiyor -> baştan
                    have = 0
                total = have + int(r.headers.get("Content-Length", 0))
                with open(part, "ab" if have else "wb") as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b); have += len(b)
                        print(f"\r  {have / 1e6:7.0f} / {total / 1e6:.0f} MB", end="", flush=True)
            if have >= total:
                os.replace(part, path)
                print()
                return
        except (OSError, urllib.error.URLError) as e:
            print(f"\n  baglanti koptu ({e}); devam ediliyor [{t + 1}/{tries}]")
    raise RuntimeError(f"indirme tamamlanamadi: {url} (tekrar calistir, kaldigi yerden devam eder)")


def clean(y, sr):
    """-> 22050 Hz mono, sessizlik atılmış, RMS 1 (float32) | None."""
    y = librosa.resample(y.mean(1) if y.ndim > 1 else y, orig_sr=sr, target_sr=SAMPLE_RATE)
    iv = librosa.effects.split(y, top_db=30)
    y = np.concatenate([y[a:b] for a, b in iv]) if len(iv) else y[:0]
    if len(y) < SAMPLE_RATE:
        return None
    return y / max(np.sqrt((y ** 2).mean()), 1e-6)


def members(kind, path):
    """-> [(konuşmacı, ad, okuyucu)] arşivdeki ses dosyaları."""
    if kind == "sing":
        z = zipfile.ZipFile(path)
        out = []
        for n in sorted(z.namelist()):
            m = re.search(r"/((?:fe)?male\d+)/", n)
            if n.lower().endswith(".wav") and m and "__MACOSX" not in n:
                out.append((m.group(1), n, lambda n=n: z.read(n)))
        return out
    t = tarfile.open(path)
    out = []
    for mi in t.getmembers():
        if mi.name.endswith(".flac"):
            out.append((mi.name.split("/")[-3], mi.name, lambda mi=mi: t.extractfile(mi).read()))
    return out


def build(kind, path, minutes, seed=0):
    ms = members(kind, path)
    spk = sorted({s for s, _, _ in ms})
    ev = EVAL_SINGERS if kind == "sing" else set(spk[-N_EVAL_SPEAKERS:])
    rng = np.random.default_rng(seed)
    for part, mins in zip(("train", "eval"), minutes):
        pool = [m for m in ms if (m[0] in ev) == (part == "eval")]
        rng.shuffle(pool)
        need, chunks, n = int(mins * 60 * SAMPLE_RATE), [], 0
        for s, name, read in pool:
            if n >= need:
                break
            y, sr = sf.read(io.BytesIO(read()), dtype="float32")
            y = clean(y, sr)
            if y is not None:
                chunks.append(y[:need - n]); n += len(chunks[-1])
        out = os.path.join(VOICE_DIR, f"{kind}_{part}.npy")
        np.save(out, np.concatenate(chunks).astype(np.float16))
        who = sorted({s for s, _, _ in pool})
        print(f"  {out}: {n / SAMPLE_RATE / 60:.1f} dk, {len(chunks)} kesit, {len(who)} kisi")


def main(minutes, keep):
    os.makedirs(VOICE_DIR, exist_ok=True)
    for kind, (url, fname) in SOURCES.items():
        path = os.path.join(VOICE_DIR, fname)
        download(url, path)
        build(kind, path, minutes)
        if not keep:
            os.remove(path)
    print("Bitti. Lisans: CC BY 4.0 (VocalSet: Wilkins vd. 2018; LibriSpeech: Panayotov vd. 2015).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, nargs=2, default=(40, 10), help="train / eval dakika (banka basina)")
    ap.add_argument("--keep", action="store_true", help="indirilen arsivi silme")
    a = ap.parse_args()
    main(a.minutes, a.keep)
