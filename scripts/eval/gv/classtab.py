"""Parse classtab.org plain-text tablature and compare a model transcription with it.

classtab layout: blocks of six lines "E|...", "B|...", ... "E|..." (top line = high e string), digits at the same
column sound together; the header holds "tuning: E A D G B E" and "video:" links to performances of that piece.
Comparison follows README20: no timing in a tab and performers play repeats, so pitch sequences are aligned with an
LCS; we report how many tab notes were found in order, how many model notes matched, and string agreement.
"""
import re
from pathlib import Path

import numpy as np

STANDARD = [64, 59, 55, 50, 45, 40]  # high e .. low E, MIDI
NAMES = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8,
         "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11}
# kept for the baseline comparison (Test E/F parser): optional E/B/G/D/A label, dashes, then the first bar
LINE = re.compile(r"^\s*(?:[EBGDAebgda][#b]?)?[\s-]*[|:]")
# a string label is any note name (altered tunings: "F#", "C", "Bb") directly followed by "-", "|" or ":"
LABEL = re.compile(r"^\s*([A-Ga-g][#b]?)\s*(?=[-|:])")


# some files hold several arrangements one after another; a performer plays one. Such files list them in the header
# ("1st version - ...", "version 2 - ...") and separate them with full-width rules of stars / equals signs. In files
# without that list the same rules only separate sections of one piece, so they must not cut the tab.
VERSION_LINE = re.compile(r"^\s*(?:\d+(?:st|nd|rd|th)\s+version|version\s+\d+)\b", re.I | re.M)
NEXT_VERSION = re.compile(r"^\s*(?:\d+(?:st|nd|rd|th)\s+version|version\s+\d+)\b|^\s*[*=]{40,}\s*$", re.I)


TAB_CHARS = set("-|:0123456789hpsbrtxHPSBX/\\~*()=<>.^ ")


def _tabby(s: str, share: float) -> bool:
    # a string line is mostly dashes and bars; rhythm / prose lines also contain "|" but not this pattern
    return s.count("-") >= 5 and ("|" in s or ":" in s) and sum(ch in TAB_CHARS for ch in s) >= share * len(s)


def line_kind(line: str) -> str | None:
    """'label' ("F#-|--3--"), 'bare' ("---7--||", "  |--0--") or None."""
    m = LABEL.match(line)
    # labelled lines may carry inline remarks ("cII", "Author: ..."): no share test, as in the baseline parser
    if m and _tabby(line[m.end():].rstrip(), 0.0):
        return "label"
    s = line.strip()
    return "bare" if s[:1] in "-|:" and _tabby(s, 0.95) else None


def content_start(line: str) -> int:
    m = LABEL.match(line)
    return m.end() if m and line_kind(line) == "label" else len(line) - len(line.lstrip())


def is_block(block: list[str]) -> bool:
    kinds = {line_kind(l) for l in block}
    if kinds == {"label"}:
        return True  # labelled systems: no further check (baseline behaviour)
    if kinds != {"bare"}:
        return False
    # unlabelled: same indentation and (nearly) same length, so a stray marker line cannot shift the strings
    indents = {content_start(l) for l in block}
    lengths = [len(l.rstrip()) for l in block]
    return len(indents) == 1 and max(lengths) - min(lengths) <= 3


def label_tuning(block: list[str]) -> list[int] | None:
    # labels top->bottom = high->low string; map to the pitch nearest the standard string
    names = [LABEL.match(l).group(1) for l in block if LABEL.match(l)]
    return _nearest(names) if len(names) == 6 else None


def _nearest(names: list[str]) -> list[int] | None:
    out = []
    for name, std in zip(names, STANDARD):
        pc = NAMES.get(name.upper())
        if pc is None:
            return None
        out.append(min((std + d for d in range(-6, 7) if (std + d) % 12 == pc), key=lambda p: abs(p - std)))
    return out


def header_tuning(text: str) -> list[int] | None:
    # "tuning: D A D G B E" lists low->high; None when the header has no tuning line
    m = re.search(r"tuning:\s*([A-G][#b]?(?:\s+[A-G][#b]?){5})", text)
    return _nearest(m.group(1).split()[::-1]) if m else None


def tuning(text: str) -> list[int] | None:
    return header_tuning(text) or STANDARD


def videos(text: str) -> list[str]:
    return re.findall(r"video:\s*(https?://\S+)", text)


def events(path: Path) -> np.ndarray | None:
    """-> (N, 3) [order, pitch, string] with string 0 = low E (pipeline convention), order = global column index."""
    text = path.read_text(encoding="latin-1")
    head = header_tuning(text)
    lines, out, base = text.splitlines(), [], 0
    several = len(VERSION_LINE.findall(text)) >= 2
    i = 0
    while i <= len(lines) - 6:
        block = lines[i:i + 6]
        if is_block(block):
            tun = head or label_tuning(block) or STANDARD  # header wins; else the labels tell the tuning
            width = max(len(l) for l in block)
            for row, l in enumerate(block):  # row 0 = high e
                start = content_start(l)
                for m in re.finditer(r"(?<!\d)(\d{1,2})(?!\d)", l[start:]):
                    # absolute column: labels of different width ("F#-" / "E--") must not shift chords apart
                    out.append((base + start + m.start(), tun[row] + int(m.group(1)), 5 - row))
            base += width + 1
            i += 6
        elif several and out and NEXT_VERSION.match(lines[i]):
            break  # first arrangement only; header lines listing versions come before any tab block
        else:
            i += 1
    return np.array(sorted(out), float) if out else None


def chords(order: np.ndarray, tol: float) -> list[list[int]]:
    # a chord = notes whose onset is within tol of the chord's first note (order must be sorted)
    groups, start = [], None
    for k, t in enumerate(order):
        if start is None or t - start > tol:
            groups.append([])
            start = t
        groups[-1].append(k)
    return groups


def sequence(order: np.ndarray, pitch: np.ndarray, tol: float) -> list[int]:
    # flatten chords low->high so the LCS sees the same order for both sides
    return [i for g in chords(order, tol) for i in sorted(g, key=lambda k: pitch[k])]


def lcs_pairs(a: np.ndarray, b: np.ndarray) -> list[tuple[int, int]]:
    # classic longest-common-subsequence table, L[i][j] = LCS length of a[i:], b[j:]
    n, m = len(a), len(b)
    L = [[0] * (m + 1) for _ in range(n + 1)]
    bl = b.tolist()
    for i in range(n - 1, -1, -1):
        ai, row, nxt = int(a[i]), L[i], L[i + 1]
        for j in range(m - 1, -1, -1):
            row[j] = nxt[j + 1] + 1 if ai == bl[j] else (nxt[j] if nxt[j] >= row[j + 1] else row[j + 1])
    pairs, i, j = [], 0, 0
    while i < n and j < m:
        if a[i] == b[j]:
            pairs.append((i, j)); i += 1; j += 1
        elif L[i + 1][j] >= L[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def compare(tab: np.ndarray, notes: np.ndarray, string: np.ndarray) -> dict[str, float]:
    ti = sequence(tab[:, 0], tab[:, 1], 0.5)
    ei = sequence(notes[:, 0], notes[:, 2], 0.03)
    a, b = tab[ti, 1].astype(int), notes[ei, 2].astype(int)
    pairs = lcs_pairs(a, b)
    same = [string[ei[j]] == tab[ti[i], 2] for i, j in pairs if string[ei[j]] >= 0]
    return {"tab_notes": len(a), "model_notes": len(b), "tab_found": len(pairs) / len(a),
            "model_matched": len(pairs) / len(b), "string_agree": float(np.mean(same)) if same else float("nan")}
