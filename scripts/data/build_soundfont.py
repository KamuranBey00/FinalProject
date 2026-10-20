"""
Nylon guitar SoundFont for scripts/app/tab_viewer.html (README21).

Source: FreePats "Spanish classical guitar" 2019-06-18, SF2 version, CC0 1.0 (public domain):
    https://freepats.zenvoid.org/Guitar/acoustic-guitar.html  ->  SpanishClassicalGuitar-SF2-20190618.7z
Extract the archive and pass the .sf2 path.

Processing (each step measured in README21):
  1. preset renumbered 0 -> 24 (General MIDI "Acoustic Guitar (nylon)"); alphaTab loads only the programs a score asks
     for, and our MusicXML asks for 24, so an unchanged bank stays silent.
  2. de-click: 13/48 samples jump from near silence to full level within 0.1 ms when the string leaves the finger
     (close mic + noise processing) -> audible tick; that step is softened over 2 ms, the rest of the note is untouched.
  3. 3 ms fade-in for the samples whose jump is in the very first milliseconds.
  4. SF2 -> SF3 (Ogg Vorbis per sample, quality 0.8): 19.8 MB -> 2.4 MB, waveform SNR vs SF2 ~39 dB.
  5. written as base64 inside a .js file: browsers refuse fetch() of local files on file:// but allow <script src>.

Run:
    python -m scripts.data.build_soundfont path/to/SpanishClassicalGuitar-20190618.sf2
"""

import argparse
import base64
import io
import struct
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

SHDR_SIZE, PHDR_SIZE = 46, 38
VORBIS_FLAG = 0x10
OUT_JS = Path("scripts/app/soundfont/nylon_guitar.sf3.js")
JS_HEADER = """// Nylon (classical) guitar SoundFont for tab_viewer.html, embedded as base64 so the viewer also works from file://
// (browsers block fetch() of local files but allow <script src>). Built by scripts/data/build_soundfont.py.
// Source: FreePats "Spanish classical guitar" 2019-06-18 by roberto@zenvoid.org,
//   https://freepats.zenvoid.org/Guitar/acoustic-guitar.html
// License: Creative Commons CC0 1.0 (public domain), https://creativecommons.org/publicdomain/zero/1.0/
// Changes: preset 0 -> 24 (GM nylon guitar), string-release clicks softened (2 ms) + 3 ms fade-in,
//   SF2 -> SF3 (Ogg Vorbis, quality 0.8).
"""


def read_chunks(data: bytes, off: int, end: int, out: dict[str, bytes]) -> None:
    # flatten the RIFF tree: only leaf chunks are needed, their parent LIST is implied by the SF2 layout
    while off < end:
        cid, size = data[off:off + 4].decode("ascii"), struct.unpack_from("<I", data, off + 4)[0]
        if cid == "LIST":
            read_chunks(data, off + 12, off + 8 + size, out)
        else:
            out[cid] = data[off + 8:off + 8 + size]
        off += 8 + size + (size & 1)


def chunk(cid: str, body: bytes) -> bytes:
    pad = b"\0" if len(body) & 1 else b""
    return cid.encode("ascii") + struct.pack("<I", len(body)) + body + pad


def encode_ogg(pcm16: np.ndarray, rate: int, quality: float) -> bytes:
    buf, x = io.BytesIO(), pcm16.astype(np.float32) / 32768
    # compression_level 0 = best quality in libsndfile's Vorbis mapping; written in blocks because one large
    # Vorbis write overflows the stack in libsndfile on Windows
    with sf.SoundFile(buf, "w", rate, 1, format="OGG", subtype="VORBIS", compression_level=1 - quality) as f:
        for i in range(0, len(x), 4096):
            f.write(x[i:i + 4096])
    return buf.getvalue()


def declick(x: np.ndarray, rate: int, width_ms: float = 2.0, search_ms: float = 60.0,
            ratio: float = 10.0) -> np.ndarray:
    # blend a low-passed copy in only around the largest early step (Hann-shaped weight): the edge softens,
    # the pluck and the rest of the note stay as recorded; ratio 10 = smallest threshold that cleared all 48 samples
    y = x.astype(np.float64)
    head = np.abs(np.diff(y[: int(rate * search_ms / 1000)]))
    if len(head) < 2 or head.max() < ratio * (np.median(head) + 1):
        return x
    j, half = int(np.argmax(head)), max(int(rate * width_ms / 2000), 1)
    lo, hi = max(j - half, 0), min(j + half + 1, len(y))
    ctx = max(lo - 200, 0)  # filter with context so the smoothed copy has no edge effects of its own
    smooth = sosfiltfilt(butter(2, 2500, "lp", fs=rate, output="sos"), y[ctx:hi + 200])
    g = np.hanning(hi - lo)
    y[lo:hi] = y[lo:hi] * (1 - g) + smooth[lo - ctx:hi - ctx] * g
    return np.round(np.clip(y, -32768, 32767)).astype(np.int16)


def fade_in(x: np.ndarray, rate: int, ms: float) -> np.ndarray:
    n = min(int(rate * ms / 1000), len(x))
    if n == 0:
        return x
    y = x.astype(np.float64)
    y[:n] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
    return np.round(y).astype(np.int16)


def convert_samples(c: dict[str, bytes], quality: float, fade_ms: float) -> tuple[bytes, bytes]:
    # SF3: smpl holds one Ogg stream per sample; shdr start/end become byte offsets into smpl and loop points become
    # frame offsets relative to the sample start (MuseScore / FluidSynth / alphaTab convention)
    pcm = np.frombuffer(c["smpl"], dtype="<i2")
    shdr, smpl = bytearray(c["shdr"]), bytearray()
    for o in range(0, len(shdr) - SHDR_SIZE, SHDR_SIZE):  # last record is the EOS terminator
        start, end, loop_s, loop_e, rate = struct.unpack_from("<5I", shdr, o + 20)
        stype = struct.unpack_from("<H", shdr, o + 44)[0]
        if stype != 1:
            raise ValueError(f"only mono samples are supported by alphaTab, got sample type {stype}")
        ogg = encode_ogg(fade_in(declick(pcm[start:end], rate), rate, fade_ms), rate, quality)
        struct.pack_into("<4I", shdr, o + 20, len(smpl), len(smpl) + len(ogg),
                         max(loop_s - start, 0), max(loop_e - start, 0))
        struct.pack_into("<H", shdr, o + 44, stype | VORBIS_FLAG)
        smpl += ogg
    if len(smpl) & 1:
        # keep the chunk even-sized: alphaTab's RIFF reader ignores the pad byte and would misread pdta (silent bank)
        smpl += b"\0"
    return bytes(smpl), bytes(shdr)


def renumber_presets(phdr: bytes, program: int) -> bytes:
    out = bytearray(phdr)
    for o in range(0, len(out) - PHDR_SIZE, PHDR_SIZE):  # keep the EOP terminator untouched
        struct.pack_into("<H", out, o + 20, program)
    return bytes(out)


def build_sf3(src: Path, program: int, quality: float, fade_ms: float) -> bytes:
    data, c = src.read_bytes(), {}
    read_chunks(data, 12, len(data), c)
    smpl, shdr = convert_samples(c, quality, fade_ms)
    c["phdr"], c["shdr"] = renumber_presets(c["phdr"], program), shdr
    c["ifil"] = struct.pack("<HH", 3, 1)  # SF3 version marker expected by FluidSynth / MuseScore
    info = b"INFO" + b"".join(chunk(k, c[k]) for k in ("ifil", "isng", "INAM", "ICRD", "ICMT") if k in c)
    pdta = b"pdta" + b"".join(chunk(k, c[k]) for k in
                              ("phdr", "pbag", "pmod", "pgen", "inst", "ibag", "imod", "igen", "shdr"))
    return chunk("RIFF", b"sfbk" + chunk("LIST", info) + chunk("LIST", b"sdta" + chunk("smpl", smpl))
                 + chunk("LIST", pdta))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="FreePats SpanishClassicalGuitar-20190618.sf2")
    ap.add_argument("--out", type=Path, default=OUT_JS)
    ap.add_argument("--sf3", type=Path, default=None, help="also write the raw .sf3 (for MuseScore / inspection)")
    ap.add_argument("--program", type=int, default=24, help="GM program, 0-based (24 = nylon guitar)")
    ap.add_argument("--quality", type=float, default=0.8, help="Vorbis quality 0..1")
    ap.add_argument("--fade-ms", type=float, default=3.0)
    a = ap.parse_args()
    sf3 = build_sf3(a.src, a.program, a.quality, a.fade_ms)
    if a.sf3:
        a.sf3.write_bytes(sf3)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(f"{JS_HEADER}window.GTAB_SOUNDFONT_B64 = \"{base64.b64encode(sf3).decode('ascii')}\";\n",
                     encoding="ascii")
    print(f"{a.out}: {a.out.stat().st_size / 1e6:.2f} MB (sf3 {len(sf3) / 1e6:.2f} MB)")
