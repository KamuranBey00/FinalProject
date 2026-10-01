"""
Çözümleme: model çıktısı -> NoteEvent'ler -> ASCII tab.

- frames_to_notes (Katman 2): perde olasılıkları -> notalar (tel bilgisi yok)
- assign_naive   (Katman 2): naif fret ataması (en düşük perde)
- tab_to_transcription (Katman 3): tel-fret sınıfları -> notalar (GERÇEK tel/fret)
- render_ascii_tab: standart tab düzeni (tiz tel üstte)
"""

import numpy as np
from typing import List

from .note_event import NoteEvent, Transcription
from .instrument import Instrument, STANDARD_6
from .labels import midi_range
from .tab_labels import tab_class_to_fret
from .config import FRAME_RATE


# ---------- Katman 2 ----------
def frames_to_notes(frame_probs, threshold=0.5, instrument=STANDARD_6,
                    frame_rate=FRAME_RATE, min_frames=2) -> Transcription:
    lo, _ = midi_range(instrument)
    active = np.asarray(frame_probs) > threshold
    T, P = active.shape
    notes: List[NoteEvent] = []
    for p in range(P):
        t = 0
        while t < T:
            if active[t, p]:
                s = t
                while t < T and active[t, p]:
                    t += 1
                if t - s >= min_frames:
                    notes.append(NoteEvent(s / frame_rate, t / frame_rate, lo + p))
            else:
                t += 1
    return Transcription(notes=notes).sort()


def assign_naive(tr: Transcription, instrument=STANDARD_6) -> Transcription:
    for n in tr.notes:
        pos = instrument.pitch_to_positions(n.pitch)
        if pos:
            n.string, n.fret = min(pos, key=lambda sf: sf[1])
    return tr


# ---------- Katman 3 ----------
def mode_filter(seq: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Kategorik sınıf dizisine kayan-pencere ÇOĞUNLUK (mode) filtresi.
    Median değil mode: sınıflar kategorik (fret 2 ile fret 8'in "ortalaması"
    anlamsız). Notanın ortasındaki tek-kare tel/fret sıçramalarını (hayalet
    nota kaynağı) bastırır. k tek sayı olmalı.
    """
    if k <= 1:
        return seq
    assert k % 2 == 1, "k tek sayi olmali (cift k'da beraberlik sessizlige onyargili kirilir)"
    half = k // 2
    T = len(seq)
    out = seq.copy()
    for t in range(T):
        a, b = max(0, t - half), min(T, t + half + 1)
        vals, counts = np.unique(seq[a:b], return_counts=True)
        out[t] = vals[counts.argmax()]
    return out


def smooth_tab(tab_pred: np.ndarray, k: int = 5) -> np.ndarray:
    """Her teli ayrı ayrı mode-filtreler."""
    tab_pred = np.asarray(tab_pred)
    if k <= 1:
        return tab_pred
    out = tab_pred.copy()
    for s in range(tab_pred.shape[1]):
        out[:, s] = mode_filter(tab_pred[:, s], k)
    return out


def tab_to_transcription(tab_pred, instrument=STANDARD_6, frame_rate=FRAME_RATE,
                         min_frames=2, smooth=1) -> Transcription:
    """
    tab_pred: (T, num_strings) int sınıf dizisi (0=sessiz, k=fret k-1).
    Ardışık aynı-fret karelerini tek nota yapar; tel/fret GERÇEKTEN atanır.
    smooth: >1 ise segmentasyondan ÖNCE mode-filtre uygular (hayalet-nota temizliği).
    """
    tab_pred = np.asarray(tab_pred)
    if smooth > 1:
        tab_pred = smooth_tab(tab_pred, smooth)
    T, S = tab_pred.shape
    notes: List[NoteEvent] = []
    for s in range(S):
        t = 0
        while t < T:
            c = tab_pred[t, s]
            if c > 0:
                start = t
                while t < T and tab_pred[t, s] == c:
                    t += 1
                if t - start >= min_frames:
                    fret = tab_class_to_fret(int(c))
                    pitch = instrument.position_to_pitch(s, fret)
                    n = NoteEvent(start / frame_rate, t / frame_rate, pitch)
                    n.string, n.fret = s, fret
                    notes.append(n)
            else:
                t += 1
    return Transcription(notes=notes).sort()


# ---------- Ortak ----------
def render_ascii_tab(tr: Transcription, instrument=STANDARD_6, col_seconds=0.15) -> str:
    S = instrument.num_strings
    names = ["E", "A", "D", "G", "B", "e"][:S]
    if not tr.notes:
        return "\n".join(f"{names[s]}|" for s in reversed(range(S)))
    total = tr.duration()
    n_cols = max(1, int(np.ceil(total / col_seconds)) + 1)
    lanes = [["-"] * n_cols for _ in range(S)]
    for note in tr.notes:
        if not note.has_position():
            continue
        col = min(int(round(note.onset / col_seconds)), n_cols - 1)
        lanes[note.string][col] = str(note.fret)
    lines = []
    for s in reversed(range(S)):
        row = "".join(f"{c:->2}" for c in lanes[s])
        lines.append(f"{names[s]}|{row}")
    return "\n".join(lines)
