"""
Katman 5 Adım 2 — MusicXML yazıcı (score-partwise; MuseScore / Guitar Pro / AlphaTab açar).

GAPS partisyonlarıyla aynı düzen: P1 nota portresi (8vb sol anahtarı, çalan perde; sol el parmağı <fingering>),
P2 TAB portresi (6 çizgi, akort <staff-tuning>, <string> 1 = ince mi, <fret>). Böylece gtab.data.gaps_score okuyucusu
çıktıyı geri okuyabilir (gidiş-dönüş testi). Süreler gtab.render.score'dan: divisions = vuruş başına TICKS.
"""

import xml.etree.ElementTree as ET

from gtab.core.instrument import STANDARD_6
from gtab.models.rhythm_net import TICKS

STEPS = ("C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B")
ALTER = (0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0)
BEAT_UNIT = {1.0: ("quarter", 0), 2.0: ("half", 0), 1.5: ("quarter", 1), 0.5: ("eighth", 0), 4.0: ("whole", 0)}
DOCTYPE = ('<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
           '"http://www.musicxml.org/dtds/partwise.dtd">\n')


def _sub(el, tag, text=None, **attr):
    e = ET.SubElement(el, tag, attr)
    if text is not None:
        e.text = str(text)
    return e


def _pitch(el, midi):
    p = _sub(el, "pitch")
    _sub(p, "step", STEPS[midi % 12])
    if ALTER[midi % 12]:
        _sub(p, "alter", ALTER[midi % 12])
    _sub(p, "octave", midi // 12 - 1)


def _attributes(m, sig, beat_q, tab, instrument):
    a = _sub(m, "attributes")
    _sub(a, "divisions", int(round(TICKS / beat_q)))
    _sub(_sub(a, "key"), "fifths", 0)
    t = _sub(a, "time")
    _sub(t, "beats", sig[0]); _sub(t, "beat-type", sig[1])
    c = _sub(a, "clef")
    if tab:
        _sub(c, "sign", "TAB"); _sub(c, "line", 5)
        sd = _sub(a, "staff-details")
        _sub(sd, "staff-lines", len(instrument.tuning))
        for i, p in enumerate(instrument.tuning):          # line 1 = en pes tel
            st = _sub(sd, "staff-tuning", line=str(i + 1))
            _sub(st, "tuning-step", STEPS[p % 12])
            if ALTER[p % 12]:
                _sub(st, "tuning-alter", ALTER[p % 12])
            _sub(st, "tuning-octave", p // 12 - 1)
    else:
        _sub(c, "sign", "G"); _sub(c, "line", 2); _sub(c, "clef-octave-change", -1)


def _note(m, piece, midi=None, chord=False, technical=()):
    n = _sub(m, "note")
    if chord:
        _sub(n, "chord")
    if midi is None:
        _sub(n, "rest")
    else:
        _pitch(n, midi)
    _sub(n, "duration", piece.dur)
    ties = [typ for typ, flag in (("stop", piece.tie_stop), ("start", piece.tie_start)) if flag and midi is not None]
    for typ in ties:
        _sub(n, "tie", type=typ)
    _sub(n, "voice", 1)
    _sub(n, "type", piece.type)
    for _ in range(piece.dots):
        _sub(n, "dot")
    if piece.triplet:
        tm = _sub(n, "time-modification")
        _sub(tm, "actual-notes", 3); _sub(tm, "normal-notes", 2)
    if ties or technical:
        nt = _sub(n, "notations")
        for typ in ties:
            _sub(nt, "tied", type=typ)
        if technical:
            tc = _sub(nt, "technical")
            for tag, val in technical:
                _sub(tc, tag, val)


def write_musicxml(path, measures, pitch, string, fret, sig, beat_q, bpm=None, finger=None, title="Transkripsiyon",
                   instrument=STANDARD_6):
    """
    measures: gtab.render.score.build_measures çıktısı; pitch / string (0 = en pes, -1 = konumsuz) / fret / finger (-1 =
    yok; None = parmak yazılmaz): nota başına diziler. sig = (pay, payda), beat_q = vuruş uzunluğu (çeyrek), bpm = vuruş/dk.
    """
    S = len(instrument.tuning)
    root = ET.Element("score-partwise", version="4.0")
    _sub(_sub(root, "work"), "work-title", title)
    pl = _sub(root, "part-list")
    for pid, name in (("P1", "Gitar"), ("P2", "TAB")):
        _sub(_sub(pl, "score-part", id=pid), "part-name", name)
    for pid, tab in (("P1", False), ("P2", True)):
        part = _sub(root, "part", id=pid)
        for k, pieces in enumerate(measures):
            m = _sub(part, "measure", number=str(k + 1))
            if k == 0:
                _attributes(m, sig, beat_q, tab, instrument)
                if bpm and not tab:
                    unit, dot = BEAT_UNIT.get(float(beat_q), ("quarter", 0))
                    d = _sub(m, "direction", placement="above")
                    mt = _sub(_sub(d, "direction-type"), "metronome")
                    _sub(mt, "beat-unit", unit)
                    if dot:
                        _sub(mt, "beat-unit-dot")
                    _sub(mt, "per-minute", int(round(bpm)))
                    _sub(d, "sound", tempo=f"{bpm * beat_q:.1f}")
            for pc in pieces:
                idx = [i for i in pc.notes if not tab or string[i] >= 0]
                if not idx:
                    _note(m, pc)
                    continue
                for j, i in enumerate(sorted(idx, key=lambda i: pitch[i])):
                    if tab:
                        tech = (("string", S - int(string[i])), ("fret", int(fret[i])))
                    else:
                        tech = (("fingering", int(finger[i])),) if finger is not None and finger[i] >= 0 else ()
                    _note(m, pc, int(pitch[i]), chord=j > 0, technical=tech)
    ET.indent(root)
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n' + DOCTYPE)
        f.write(ET.tostring(root, encoding="unicode"))
        f.write("\n")
