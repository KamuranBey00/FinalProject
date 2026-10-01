"""
SynthTab okuyucu (sentetik, tel bazlı etiketli gitar sesi).

Kaynak: https://github.com/yongyizang/SynthTab  (UR Box'tan elle indirilir)
Dev Set ile DOĞRULANMIŞ yapı (1 Ekim 2026):

  SynthTab_Dev/
    acoustic/<tını>/<parça>/<parça>/<tını>_nonoise_mono_body.flac   (+ tel başına f0 .pkl)
    electric_*/<tını>/<parça>/<parça>/<...>.flac
    jams/<parça>/
        string_1.mid ... string_6.mid     tel başına MIDI (GERÇEK zaman, perde)
        <n> - <enstrüman>.jams            note_tab (zaman TICK cinsinden) + sandbox:
                                          {'string_index': 1..6, 'open_tuning': midi}
        tempo.txt

  - string_index 1 = İNCE E (open_tuning 64) ... 6 = KALIN E (40)
    -> bizim indeks: s = num_strings - string_index  (0 = kalın E)
  - Her tel MIDI'sinde t=0'da perde 24 (C1) işaretleyici nota var -> atılır
    (açık tel perdesinin altındaki her nota atılır).
  - Etiket zamanlaması ses ile hizalı (gecikme 0–2 kare, ölçüldü).
  - Tını 'luthier_*' = Ample Guitar L (Alhambra Luthier) = KLASİK NAYLON gitar.

Zip'ler AÇILMADAN okunur (disk tasarrufu).
"""

import io
import json
import os
import zipfile
from collections import defaultdict

import numpy as np

from gtab.core.instrument import STANDARD_6
from gtab.config import FRAME_RATE


class Source:
    """Bir klasörü ya da zip'i (iç içe zip'ler dahil) tek arayüzle gezer."""

    def __init__(self, path, label=None):
        self.path = path
        self.label = label or os.path.basename(os.path.dirname(os.path.abspath(path)))
        self.zips = []
        if os.path.isdir(path):
            self.zf = None
            self.files = []
            for root, _, fs in os.walk(path):
                for f in fs:
                    full = os.path.join(root, f)
                    if f.lower().endswith(".zip"):
                        self.zips.append(Source(full))
                    else:
                        self.files.append(os.path.relpath(full, path).replace("\\", "/"))
        else:
            self.zf = zipfile.ZipFile(path)
            self.files = [n for n in self.zf.namelist() if not n.endswith("/")]

    def all_files(self):
        for f in self.files:
            yield self, f
        for z in self.zips:
            yield from z.all_files()

    def read(self, name) -> bytes:
        if self.zf is not None:
            return self.zf.read(name)
        with open(os.path.join(self.path, name), "rb") as fh:
            return fh.read()


def _parts(name):
    return name.replace("\\", "/").split("/")


def index(audio_path, label_path=None, timbre_filter=None):
    """
    -> (audio, labels)
       audio : {(tını, parça): (src, flac_yolu)}   aynı parça farklı tınılarda AYRI örnek
       labels: {parça: {"mids": {string_index: (src, yol)}, "jams": (src, yol)}}
    timbre_filter: tını adında geçmesi gereken alt dizgi (ör. "luthier" = naylon).
    """
    audio = {}
    for src, f in Source(audio_path).all_files():
        if not f.lower().endswith((".flac", ".wav")):
            continue
        p = _parts(f)
        track = p[-2] if len(p) >= 2 else os.path.splitext(p[-1])[0]
        timbre = p[-4] if len(p) >= 4 else src.label
        if timbre_filter and timbre_filter not in timbre:
            continue
        audio.setdefault((timbre, track), (src, f))

    labels = defaultdict(lambda: {"mids": {}, "jams": None})
    for src, f in Source(label_path or audio_path).all_files():
        p = _parts(f)
        if len(p) < 2:
            continue
        track, base = p[-2], p[-1]
        if base.startswith("string_") and base.endswith(".mid"):
            labels[track]["mids"][int(base[len("string_"):-4])] = (src, f)
        elif base.endswith(".jams"):
            labels[track]["jams"] = (src, f)
    return audio, dict(labels)


def read_tuning(jam: dict, num_strings=6):
    """JAMS sandbox'tan akort -> bizim sıramızla (kalın teldenince tele) tuple ya da None."""
    tun = {}
    for ann in jam.get("annotations", []):
        sb = ann.get("sandbox") or {}
        if ann.get("namespace") == "note_tab" and "string_index" in sb:
            tun[int(sb["string_index"])] = int(sb["open_tuning"])
    if len(tun) != num_strings:
        return None
    return tuple(tun[num_strings - s] for s in range(num_strings))


def string_notes(lab, instrument=STANDARD_6):
    """
    Etiket kaydı -> ({tel: [(start_sn, end_sn, midi)]}, akort) ya da (None, sebep).
    Standart olmayan akort -> atlanır (fret etiketi tanımsız olur).
    """
    import pretty_midi
    # Hiç çalınmayan tel için string_k.mid üretilmemiş olabilir (Dev Set'te parçaların
    # çoğu böyle) -> o tel boş sayılır. Akort JAMS'ten okunduğu için JAMS zorunlu.
    if lab["jams"] is None or not lab["mids"]:
        return None, "eksik etiket"
    src, f = lab["jams"]
    tuning = read_tuning(json.loads(src.read(f)), instrument.num_strings)
    if tuning != tuple(instrument.tuning):
        return None, f"akort {tuning}"
    out = {s: [] for s in range(instrument.num_strings)}
    for si, (src, f) in lab["mids"].items():
        s = instrument.num_strings - si
        pm = pretty_midi.PrettyMIDI(io.BytesIO(src.read(f)))
        out[s] = [(n.start, n.end, n.pitch) for inst in pm.instruments for n in inst.notes
                  if n.pitch >= instrument.tuning[s]]           # perde-24 işaretleyicisi elenir
    return out, tuning


def notes_to_tab(notes, n_frames, instrument=STANDARD_6, frame_rate=FRAME_RATE):
    """{tel: [(a, b, midi)]} -> (T, S) sınıf dizisi (0 = sessiz, fret + 1)."""
    tab = np.zeros((n_frames, instrument.num_strings), np.int64)
    for s, ns in notes.items():
        for a, b, m in ns:
            fret = int(m) - instrument.tuning[s]
            if 0 <= fret <= instrument.num_frets:
                tab[max(0, int(round(a * frame_rate))):min(n_frames, int(round(b * frame_rate))), s] = fret + 1
    return tab
