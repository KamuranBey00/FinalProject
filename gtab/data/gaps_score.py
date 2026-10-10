"""
Katman 3.8e — GAPS partisyonundan tel/perde etiketi + sol el parmak numarası.

GAPS'te her kayıt için:
  musicxml/<id>.xml   : nota portresi (<fingering> = sol el parmağı) + TAB portresi (<string>, <fret>)
  syncpoints/<id>.json: [ölçü (tekrarlar açılmış), saniye, (ölçü içi tick, 480/çeyrek)] -> partisyon -> ses zamanı
  midi/<id>.mid       : hizalı MIDI (önbellekteki perde etiketleri buradan)

Yol: TAB notası -> (açılmış ölçü, tick) -> syncpoint enterpolasyonu -> saniye -> aynı perdeli en yakın
MIDI notası (bire bir). Eşleşmeyen MIDI notasının teli bilinmez (maske).
"""

import json
import os
import xml.etree.ElementTree as ET

import numpy as np

TPQ = 480                                      # syncpoint tick birimi (çeyrek başına)
STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
TYPE_Q = {"breve": 8, "whole": 4, "half": 2, "quarter": 1, "eighth": .5, "16th": .25, "32nd": .125, "64th": .0625}


def _duration(el, div):
    """<duration> yoksa (birkaç partisyonda dışa aktarma hatası) nota tipi + nokta + üçleme oranından."""
    d = el.findtext("duration")
    if d is not None:
        return int(d), False
    q = TYPE_Q.get(el.findtext("type"), 0) * (2 - 0.5 ** len(el.findall("dot")))
    tm = el.find("time-modification")
    if tm is not None:
        q *= int(tm.findtext("normal-notes")) / int(tm.findtext("actual-notes"))
    return q * div, True


def _midi(step, octave, alter=0):
    step = (step or "").strip()
    if step not in STEP:                       # bozuk perde (birkaç partisyonda boş <step>) -> TAB'dan
        return None
    return 12 * (int(octave) + 1) + STEP[step] + int(round(float(alter or 0)))


def _nums(s):
    return {int(x) for x in s.replace(" ", "").split(",") if x.isdigit()}


def parse_part(part):
    """
    Bir <part> -> ölçü listesi. Her ölçü: dict(len=tick, notes=[dict(tick, pitch, string, fret, finger)],
    fwd, bwd (tekrar sayısı | 0), end (son bölüm numaraları | None), transpose).
    Bağ devamı (tie stop) notaları atlanır (yeni vuruş değil); süs notaları sıfır süreli, ana notanın tick'inde.
    """
    div, tr, tuning = 1, 0, None
    meas, ending = [], None
    for m in part.findall("measure"):
        a = m.find("attributes")
        if a is not None:
            if a.find("divisions") is not None:
                div = int(a.findtext("divisions"))
            if a.find("transpose") is not None:
                tr = int(a.findtext("transpose/chromatic") or 0) + 12 * int(a.findtext("transpose/octave-change") or 0)
            st = a.findall("staff-details/staff-tuning")
            if st:
                tuning = {int(t.get("line")): _midi(t.findtext("tuning-step"), t.findtext("tuning-octave"),
                                                    t.findtext("tuning-alter")) for t in st}
        cur = dict(notes=[], fwd=False, bwd=0, end=ending, transpose=tr, tuning=tuning)
        pos = last = mx = 0
        guessed = False                                  # süre tahmin edildiyse <backup> süresi de güvenilmez
        for el in m:
            if el.tag == "note":
                dur, g = _duration(el, div)
                guessed |= g and el.find("grace") is None
                chord = el.find("chord") is not None
                on = last if chord else pos
                if not chord:
                    last = pos
                    if el.find("grace") is None:
                        pos += dur
                p = el.find("pitch")
                tie_stop = any(t.get("type") == "stop" for t in el.findall("tie"))
                if p is not None and not tie_stop:
                    t = el.find("notations/technical")
                    f = t.findtext("fingering") if t is not None else None
                    w = _midi(p.findtext("step"), p.findtext("octave"), p.findtext("alter"))
                    cur["notes"].append(dict(
                        tick=on * TPQ / div,
                        pitch=None if w is None else w + tr,
                        string=int(t.findtext("string")) if t is not None and t.find("string") is not None else -1,
                        fret=int(t.findtext("fret")) if t is not None and t.find("fret") is not None else -1,
                        finger=int(f) if f and f.strip().isdigit() else -1))
            elif el.tag == "backup":
                pos = 0 if guessed else pos - int(el.findtext("duration"))   # tahminliyse: ses ölçü başına döner
            elif el.tag == "forward":
                pos += int(el.findtext("duration"))
            elif el.tag == "barline":
                r, e = el.find("repeat"), el.find("ending")
                if r is not None:
                    if r.get("direction") == "forward":
                        cur["fwd"] = True
                    else:
                        cur["bwd"] = int(r.get("times") or 2)
                if e is not None:
                    if e.get("type") == "start":
                        ending = cur["end"] = _nums(e.get("number", ""))
                    else:                                      # stop | discontinue: bu ölçüyle biter
                        ending = None
            mx = max(mx, pos)
        cur["len"] = mx * TPQ / div
        meas.append(cur)
    return meas


def unroll(meas):
    """Tekrar işaretleri (ileri/geri, times, 1./2. son) -> çalınış sırasındaki ölçü indeksleri."""
    order, i, start, passn, cnt, last_bwd = [], 0, 0, 1, {}, -1
    while i < len(meas):
        m = meas[i]
        if passn > 1 and m["end"] is None and i > last_bwd:      # tekrar bloğundan çıkıldı
            passn, start = 1, i
        if m["fwd"]:
            start = i
        if m["end"] is not None and passn not in m["end"]:
            i += 1
            continue
        order.append(i)
        if m["bwd"]:
            c = cnt.get(i, 1)
            if c < m["bwd"]:
                cnt[i], passn, last_bwd, i = c + 1, c + 1, max(last_bwd, i), start
                continue
            if m["end"] is None:                                 # sonsuz tekrar yok: blok bitti
                passn, start = 1, i + 1
        i += 1
    return order


def score_notes(xml_path):
    """
    MusicXML -> (açılmış ölçü sırası, ölçü uzunlukları, TAB notaları listesi).
    TAB notası: dict(bar=açılmış indeks, tick, pitch (çalan), string (1 = ince mi), fret, finger).
    Parmak numarası nota portresinden (aynı ölçü + tick + çalan perde) TAB notasına taşınır.
    """
    root = ET.parse(xml_path).getroot()
    parts = [parse_part(p) for p in root.findall("part")]
    tab_i = [k for k, p in enumerate(parts) if any(n["fret"] >= 0 for m in p for n in m["notes"])]
    if not tab_i:
        return None
    tab = parts[tab_i[0]]
    fing = {}
    for k, p in enumerate(parts):
        if k != tab_i[0]:
            for mi, m in enumerate(p):
                for n in m["notes"]:
                    if n["finger"] >= 0:
                        fing[(mi, round(n["tick"]), n["pitch"])] = n["finger"]
    order = unroll(tab)
    notes = []
    for b, mi in enumerate(order):
        m = tab[mi]
        for n in m["notes"]:
            if n["fret"] >= 0 and n["string"] >= 1 and m["tuning"]:
                n = dict(n, pitch=m["tuning"][7 - n["string"]] + n["fret"])   # çalan perde TAB'dan
            if n["pitch"] is None:
                continue
            notes.append(dict(n, bar=b, finger=fing.get((mi, round(n["tick"]), n["pitch"]), n["finger"])))
    return order, np.array([tab[mi]["len"] for mi in order]), notes, tab[0]["tuning"]


def score_times(sync, lens, notes):
    """syncpoint'ler -> her TAB notasının ses zamanı (saniye; doğrusal enterpolasyon, uçlarda doğrusal uzatma)."""
    cum = np.concatenate([[0.0], np.cumsum(lens)])
    xs = np.array([cum[min(int(e[0]), len(lens))] + (e[2] if len(e) > 2 else 0.0) for e in sync])
    ts = np.array([e[1] for e in sync], float)
    o = np.argsort(xs, kind="stable"); xs, ts = xs[o], ts[o]
    x = np.array([cum[n["bar"]] + n["tick"] for n in notes])
    t = np.interp(x, xs, ts)
    if len(xs) >= 2:
        lo, hi = x < xs[0], x > xs[-1]
        t[lo] = ts[0] + (x[lo] - xs[0]) * (ts[1] - ts[0]) / max(xs[1] - xs[0], 1e-9)
        t[hi] = ts[-1] + (x[hi] - xs[-1]) * (ts[-1] - ts[-2]) / max(xs[-1] - xs[-2], 1e-9)
    return t, x


def match_notes(midi_notes, notes, times, tol=0.1):
    """
    MIDI notaları [(başlangıç sn, perde)] <-> TAB notaları: aynı perde, |Δt| <= tol, en küçük |Δt| önce, bire bir.
    -> (her MIDI notası için TAB notası indeksi | -1, eşleşen Δt'ler)
    """
    by_p = {}
    for j, n in enumerate(notes):
        by_p.setdefault(n["pitch"], []).append(j)
    cand = []
    for i, (t0, p) in enumerate(midi_notes):
        for j in by_p.get(p, ()):
            d = abs(times[j] - t0)
            if d <= tol:
                cand.append((d, i, j))
    cand.sort()
    mi, used, dts = np.full(len(midi_notes), -1), set(), []
    for d, i, j in cand:
        if mi[i] < 0 and j not in used:
            mi[i] = j; used.add(j); dts.append(times[j] - midi_notes[i][0])
    return mi, np.array(dts)


def load_sync(gaps_dir, tid):
    with open(os.path.join(gaps_dir, "syncpoints", f"{tid}.json")) as f:
        return json.load(f)


def _local_offset(x, xm, rm, win):
    """Eşleşen notaların kalıntıları (MIDI − partisyon zamanı) -> her partisyon konumu için ±win tick kayan medyan."""
    o = np.argsort(xm); xm, rm = xm[o], rm[o]
    lo, hi = np.searchsorted(xm, xm - win), np.searchsorted(xm, xm + win, side="right")
    med = np.array([np.median(rm[a:b]) for a, b in zip(lo, hi)])
    return np.interp(x, xm, med)


def align(gaps_dir, tid, instrument_tuning, tol1=0.25, tol2=0.08, win=2 * TPQ, shifts=range(-5, 6)):
    """
    Bir GAPS kaydı -> MIDI notası başına partisyon etiketi.
      1) partisyon -> açılmış ölçü/tick -> syncpoint ile saniye (ölçü düzeyi)
      2) kayıt geneli perde kayması (kapo / akort): kaba eşlemeyi en çok artıran k (MIDI − partisyon)
      3) kaba eşleme (tol1) -> yerel zaman düzeltmesi (±win tick kayan medyan; arpej/rubato) -> ince eşleme (tol2)
    Tel = partisyon TAB'ı; perde (fret) = MIDI perdesi − standart açık tel (kapo/akort farkını içerir).
    -> dict(consistent, shift, n_bars, n_sync, notes=[(başlangıç, bitiş, perde)], string (0 = pes) | -1, fret | -1,
            finger | -1, dt)
    """
    import pretty_midi
    order, lens, notes, tuning = score_notes(os.path.join(gaps_dir, "musicxml", f"{tid}.xml"))
    sync = load_sync(gaps_dir, tid)
    n_sync = max(int(e[0]) for e in sync)
    pm = pretty_midi.PrettyMIDI(os.path.join(gaps_dir, "midi", f"{tid}.mid"))
    mn = sorted((float(n.start), float(n.end), int(n.pitch)) for i in pm.instruments if not i.is_drum for n in i.notes)
    t0, x = score_times(sync, lens, notes)
    rate = lambda k: (match_notes([(s, p - k) for s, _, p in mn], notes, t0, tol1)[0] >= 0).mean()
    shift = max(shifts, key=lambda k: (rate(k), -abs(k)))
    q = [(s, p - shift) for s, _, p in mn]
    mi, _ = match_notes(q, notes, t0, tol1)
    ok = mi >= 0
    t1 = t0
    if ok.sum() >= 2:
        xm, rm = x[mi[ok]], np.array([q[i][0] for i in np.flatnonzero(ok)]) - t0[mi[ok]]
        t1 = t0 + _local_offset(x, xm, rm, win)
    mi, dts = match_notes(q, notes, t1, tol2)
    string = np.full(len(mn), -1); fret = np.full(len(mn), -1); finger = np.full(len(mn), -1)
    for i, j in enumerate(mi):
        if j < 0 or notes[j]["string"] < 1:
            continue
        s = 6 - notes[j]["string"]                       # MusicXML tel 1 = ince E -> bizim indeks 0 = pes E
        f = mn[i][2] - instrument_tuning[s]
        if 0 <= f <= 24:
            string[i], fret[i], finger[i] = s, f, notes[j]["finger"]
    return dict(consistent=n_sync in (len(order) - 1, len(order)), shift=shift, n_bars=len(order), n_sync=n_sync,
                notes=mn, string=string, fret=fret, finger=finger, dt=dts, matched=mi >= 0)
