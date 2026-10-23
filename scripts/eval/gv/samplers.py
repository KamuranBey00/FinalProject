"""Load one-shot guitar sample sets (WAV folders, SF2, SF3) into a common form and render note lists with them.

The sampler is intentionally simple and identical for every library: nearest-root sample, resampled to the target
pitch, cut at the note offset with a 150 ms release fade. Roots are measured with pYIN rather than trusted from file
names / sample headers (libraries disagree on octave numbering).
"""
import base64
import io
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 22050
RELEASE_S = 0.15


@dataclass
class Sample:
    root: float          # measured MIDI pitch
    audio: np.ndarray    # mono float, SR


TRIM = True  # start each sample 2 ms before its sound onset (what a real sampler's offset setting does)


def trim_lead(y: np.ndarray) -> np.ndarray:
    # libraries keep 0-57 ms of pre-pluck silence; untrimmed, notes sound late and miss the 50 ms onset window
    hit = np.flatnonzero(np.abs(y) > 0.02 * np.abs(y).max())
    return y[max(int(hit[0]) - int(0.002 * SR), 0):] if TRIM and len(hit) else y


def measured_root(y: np.ndarray) -> float | None:
    f0, voiced, _ = librosa.pyin(y[: 2 * SR], fmin=70, fmax=1400, sr=SR)
    f0 = f0[voiced & ~np.isnan(f0)]
    return float(np.median(librosa.hz_to_midi(f0))) if len(f0) >= 5 else None


def sfz_roots(sfz_text: str) -> dict[str, int]:
    # "<region> sample=samples\[A#2] ... .wav key=a#2" -> {file name: MIDI}; the library's own mapping beats pYIN,
    # which reads a bright sample's strong 2nd harmonic as the fundamental (9/37 MDM samples, +12)
    pc = {"c": 0, "c#": 1, "d": 2, "d#": 3, "e": 4, "f": 5, "f#": 6, "g": 7, "g#": 8, "a": 9, "a#": 10, "b": 11}
    out = {}
    for m in re.finditer(r"sample=\S*?\\(.+?\.wav)\s+key=([a-g]#?)(-?\d)", sfz_text, re.I):
        out[m.group(1).strip()] = 12 * (int(m.group(3)) + 1) + pc[m.group(2).lower()]
    return out


def from_wavs(files: list[Path], roots: dict[str, int] | None = None) -> list[Sample]:
    out = []
    for p in files:
        y, _ = librosa.load(p, sr=SR, mono=True)
        r = float(roots[p.name]) if roots and p.name in roots else measured_root(y)
        if r is not None:
            out.append(Sample(r, trim_lead(y)))
    return out


def _chunks(data: bytes, off: int, end: int, out: dict[str, bytes]) -> None:
    while off < end:
        cid, size = data[off:off + 4].decode("latin-1"), struct.unpack_from("<I", data, off + 4)[0]
        if cid == "LIST":
            _chunks(data, off + 12, off + 8 + size, out)
        else:
            out[cid] = data[off + 8:off + 8 + size]
        nxt = off + 8 + size
        # RIFF says odd chunks get a pad byte, but some writers (MuseScore SF3) omit it: pad only if that lands on an id
        pad = (size & 1) and data[nxt + 1:nxt + 5].isalpha() and not data[nxt:nxt + 4].isalpha()
        off = nxt + int(bool(pad))


def _first_preset_samples(c: dict[str, bytes]) -> set[int]:
    # preset 0 -> its instruments -> their sample ids (gen 41 = instrument, gen 53 = sampleID)
    u16 = lambda b, o: struct.unpack_from("<H", b, o)[0]
    pbag = [u16(c["pbag"], o) for o in range(0, len(c["pbag"]), 4)]
    pgen = [struct.unpack_from("<HH", c["pgen"], o) for o in range(0, len(c["pgen"]), 4)]
    inst = [u16(c["inst"], o + 20) for o in range(0, len(c["inst"]), 22)]
    ibag = [u16(c["ibag"], o) for o in range(0, len(c["ibag"]), 4)]
    igen = [struct.unpack_from("<HH", c["igen"], o) for o in range(0, len(c["igen"]), 4)]
    b0, b1 = u16(c["phdr"], 24), u16(c["phdr"], 38 + 24)
    used = set()
    for pb in range(b0, b1):
        for op, i in pgen[pbag[pb]:pbag[pb + 1]]:
            if op == 41:
                for ib in range(inst[i], inst[i + 1]):
                    used |= {amt for op2, amt in igen[ibag[ib]:ibag[ib + 1]] if op2 == 53}
    return used


def from_viewer_font(js: Path) -> list[Sample]:
    # the viewer's SoundFont is shipped as base64 inside a .js file (see tab_viewer.html)
    b64 = re.search(r'GTAB_SOUNDFONT_B64 = "([^"]+)"', js.read_text(encoding="utf-8")).group(1)
    return from_soundfont_bytes(base64.b64decode(b64))


def from_soundfont(path: Path) -> list[Sample]:
    return from_soundfont_bytes(path.read_bytes())


def from_soundfont_bytes(data: bytes) -> list[Sample]:
    c: dict[str, bytes] = {}
    _chunks(data, 12, len(data), c)
    out, sh = [], c["shdr"]
    for i in sorted(_first_preset_samples(c)):
        o = i * 46
        start, end, _, _, rate = struct.unpack_from("<5I", sh, o + 20)
        stype = struct.unpack_from("<H", sh, o + 44)[0]
        if stype & 2:  # right channel of a stereo pair: the left one carries the same note
            continue
        if stype & 0x10:
            y, rate = sf.read(io.BytesIO(c["smpl"][start:end]))
        else:
            y = np.frombuffer(c["smpl"][start * 2:end * 2], "<i2").astype(np.float64) / 32768
        y = librosa.resample(np.asarray(y, float).reshape(len(y), -1).mean(axis=1), orig_sr=rate, target_sr=SR)
        r = measured_root(y)
        if r is not None:
            out.append(Sample(r, trim_lead(y)))
    return out


def render(notes: np.ndarray, lib: list[Sample], velocity: float | np.ndarray = 0.5) -> np.ndarray:
    # velocity: one level for all notes, or one per note (training data: real players do not play every note forte)
    roots = np.array([s.root for s in lib])
    out = np.zeros(int((notes[:, 1].max() + 2) * SR)) if len(notes) else np.zeros(SR)
    vel = np.broadcast_to(np.asarray(velocity, float), (len(notes),))
    for (on, off, pitch), v in zip(notes, vel):
        s = lib[int(np.argmin(np.abs(roots - pitch)))]
        ratio = 2 ** ((pitch - s.root) / 12)  # >1 = higher: read the sample faster
        n = int((off - on + RELEASE_S) * SR)
        idx = np.arange(n) * ratio
        idx = idx[idx < len(s.audio) - 1]
        y = np.interp(idx, np.arange(len(s.audio)), s.audio)
        rel = int(RELEASE_S * SR)
        if len(y) > rel:
            y[-rel:] *= np.linspace(1, 0, rel)
        a = int(on * SR)
        out[a:a + len(y)] += v * y[: len(out) - a]
    peak = np.abs(out).max()
    return out / peak * 0.9 if peak > 0 else out
