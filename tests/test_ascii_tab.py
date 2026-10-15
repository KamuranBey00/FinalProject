"""
ASCII TAB rendering tests (ArmaganDegisiklikler: R-1 adjacent two-digit frets, R-2 overwritten notes).

Pure numpy; no dataset, checkpoint or GPU needed.
Run (project root): python -m unittest tests.test_ascii_tab
"""

import unittest
from typing import List

import numpy as np

from gtab.core.instrument import STANDARD_6
from gtab.core.note_event import NoteEvent, Transcription
from gtab.decoding.decode import render_ascii_tab

COL = 0.15


def _old_render(tr: Transcription, col_seconds: float = COL) -> str:
    """Pre-ArmaganDegisiklikler renderer, kept verbatim as the reference for the backward-compatible case."""
    S = STANDARD_6.num_strings
    names = ["E", "A", "D", "G", "B", "e"][:S]
    n_cols = max(1, int(np.ceil(tr.duration() / col_seconds)) + 1)
    lanes = [["-"] * n_cols for _ in range(S)]
    for note in tr.notes:
        if not note.has_position():
            continue
        col = min(int(round(note.onset / col_seconds)), n_cols - 1)
        lanes[note.string][col] = str(note.fret)
    return "\n".join(f"{names[s]}|" + "".join(f"{c:->2}" for c in lanes[s]) for s in reversed(range(S)))


def _note(onset: float, string: int, fret: int) -> NoteEvent:
    return NoteEvent(onset, onset + 0.1, STANDARD_6.tuning[string] + fret, string=string, fret=fret)


def _cells(line: str, width: int) -> List[str]:
    """Split one rendered string line into fixed-width cells (after the 'X|' prefix)."""
    body = line.split("|", 1)[1]
    return [body[i:i + width].strip("-") for i in range(0, len(body), width)]


class TestAsciiTab(unittest.TestCase):

    def test_single_digit_layout_unchanged(self):
        # no two-digit frets and no collisions -> byte-identical to the old renderer
        tr = Transcription([_note(0.0, 0, 3), _note(0.3, 1, 5), _note(0.6, 5, 0), _note(0.6, 4, 1)])
        self.assertEqual(render_ascii_tab(tr, STANDARD_6, COL), _old_render(tr))

    def test_adjacent_two_digit_frets_are_separated(self):
        # old renderer printed "-712" for 7 then 12 in neighbouring cells
        tr = Transcription([_note(0.0, 5, 7), _note(0.15, 5, 12)])
        e_line = render_ascii_tab(tr, STANDARD_6, COL).splitlines()[0]
        self.assertIn("-7-12", e_line)
        self.assertNotIn("712", e_line)

    def test_every_cell_has_a_dash_separator(self):
        tr = Transcription([_note(0.0, 4, 11), _note(0.15, 4, 11), _note(0.30, 4, 1)])
        b_line = render_ascii_tab(tr, STANDARD_6, COL).splitlines()[1]
        self.assertEqual(_cells(b_line, 3)[:3], ["11", "11", "1"])
        self.assertTrue(all(b_line.split("|")[1][i] == "-" for i in range(0, 9, 3)))

    def test_collision_keeps_both_notes(self):
        # two notes 50 ms apart on the same string fall into one 0.15 s cell
        tr = Transcription([_note(1.00, 3, 2), _note(1.05, 3, 4)])
        g_line = render_ascii_tab(tr, STANDARD_6, COL).splitlines()[2]
        frets = [c for c in _cells(g_line, 2) if c]
        self.assertEqual(frets, ["2", "4"])                  # order preserved, nothing dropped

    def test_no_note_is_lost(self):
        rng = np.random.default_rng(0)
        notes = [_note(float(t), int(s), int(f)) for t, s, f in
                 zip(np.sort(rng.uniform(0, 3, 200)), rng.integers(0, 6, 200), rng.integers(0, 20, 200))]
        tab = render_ascii_tab(Transcription(notes), STANDARD_6, COL)
        shown = sum(len([c for c in _cells(line, 3) if c]) for line in tab.splitlines())
        self.assertEqual(shown, len(notes))

    def test_shift_past_end_extends_all_lines(self):
        tr = Transcription([_note(0.0, 0, 1), _note(0.01, 0, 2), _note(0.02, 0, 3)])
        lines = render_ascii_tab(tr, STANDARD_6, COL).splitlines()
        self.assertEqual(len({len(line) for line in lines}), 1)   # all six lines same length
        self.assertEqual([c for c in _cells(lines[5], 2) if c], ["1", "2", "3"])

    def test_empty_transcription(self):
        self.assertEqual(render_ascii_tab(Transcription([]), STANDARD_6, COL).splitlines()[0], "e|")


if __name__ == "__main__":
    unittest.main()
