"""
gtab.data.gaps_score unit tests (README17 Ek A: O-8 / audit S-19, Y-2).

Every fingering and rhythm label in gaps_tab comes from this module; until now it was checked only
through aggregate statistics. These tests build tiny hand-written MusicXML scores, so no dataset is
needed.

Run (project root): python -m unittest tests.test_gaps_score
"""

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional

import numpy as np

from gtab.data.gaps_score import (TPQ, bar_similarity, match_notes, order_from_content, parse_part,
                                  score_notes, score_times, unroll)

# standard tuning written as MusicXML <staff-tuning> lines (line 1 = low E)
TUNING = [("E", 2), ("A", 2), ("D", 3), ("G", 3), ("B", 3), ("E", 4)]
STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def midi(step: str, octave: int, alter: int = 0) -> int:
    return 12 * (octave + 1) + STEP[step] + alter


def note(step: str, octave: int, dur: int, string: Optional[int] = None, fret: Optional[int] = None,
         finger: Optional[int] = None, chord: bool = False, tie_stop: bool = False, grace: bool = False,
         alter: int = 0) -> str:
    tech = ""
    if string is not None or finger is not None:
        tech = "<notations><technical>"
        if finger is not None:
            tech += f"<fingering>{finger}</fingering>"
        if string is not None:
            tech += f"<string>{string}</string><fret>{fret}</fret>"
        tech += "</technical></notations>"
    return ("<note>" + ("<grace/>" if grace else "") + ("<chord/>" if chord else "") +
            f"<pitch><step>{step}</step>" + (f"<alter>{alter}</alter>" if alter else "") +
            f"<octave>{octave}</octave></pitch>" + ("" if grace else f"<duration>{dur}</duration>") +
            ('<tie type="stop"/>' if tie_stop else "") + tech + "</note>")


def measure(body: str, attrs: str = "", left: str = "", right: str = "") -> str:
    return f"<measure>{attrs}{left}{body}{right}</measure>"


def tab_attrs(div: int = 1, capo_transpose: int = 0) -> str:
    tun = "".join(f'<staff-tuning line="{i + 1}"><tuning-step>{s}</tuning-step>'
                  f"<tuning-octave>{o}</tuning-octave></staff-tuning>" for i, (s, o) in enumerate(TUNING))
    tr = f"<transpose><chromatic>{capo_transpose}</chromatic></transpose>" if capo_transpose else ""
    return f"<attributes><divisions>{div}</divisions>{tr}<staff-details>{tun}</staff-details></attributes>"


FWD = '<barline location="left"><repeat direction="forward"/></barline>'
BWD = '<barline location="right"><repeat direction="backward"/></barline>'


def ending(num: str, kind: str) -> str:
    return f'<barline><ending number="{num}" type="{kind}"/></barline>'


def score(*parts: List[str]) -> str:
    body = "".join(f'<part id="P{i}">{"".join(ms)}</part>' for i, ms in enumerate(parts, 1))
    return f'<?xml version="1.0"?><score-partwise>{body}</score-partwise>'


def parse(xml: str):
    return parse_part(ET.fromstring(xml).find("part"))


class TestUnroll(unittest.TestCase):

    def _meas(self, n: int, fwd=(), bwd=(), ends=None):
        ends = ends or {}
        return [dict(notes=[], fwd=i in fwd, bwd=2 if i in bwd else 0, end=ends.get(i), len=TPQ) for i in range(n)]

    def test_no_repeats(self):
        self.assertEqual(unroll(self._meas(3)), [0, 1, 2])

    def test_simple_repeat(self):
        self.assertEqual(unroll(self._meas(4, fwd={1}, bwd={2})), [0, 1, 2, 1, 2, 3])

    def test_repeat_from_start_without_forward_sign(self):
        self.assertEqual(unroll(self._meas(3, bwd={1})), [0, 1, 0, 1, 2])

    def test_first_and_second_ending(self):
        # |: 0 1 |1. 2 :|2. 3 | 4
        meas = self._meas(5, fwd={0}, bwd={2}, ends={2: {1}, 3: {2}})
        self.assertEqual(unroll(meas), [0, 1, 2, 0, 1, 3, 4])

    def test_parsed_volta_score(self):
        xml = score([measure(note("C", 4, 1, 2, 1), tab_attrs(), left=FWD),
                     measure(note("D", 4, 1, 2, 3), left=ending("1", "start"), right=ending("1", "stop") + BWD),
                     measure(note("E", 4, 1, 1, 0), left=ending("2", "start"), right=ending("2", "discontinue")),
                     measure(note("F", 4, 1, 1, 1))])
        self.assertEqual(unroll(parse(xml)), [0, 1, 0, 2, 3])


class TestParsePart(unittest.TestCase):

    def test_ticks_chords_ties_and_grace(self):
        xml = score([measure(note("C", 4, 2, 2, 1) + note("E", 4, 2, 1, 0, chord=True) +
                             note("D", 4, 1, 2, 3, grace=True) + note("D", 4, 1, 2, 3) + note("D", 4, 1, tie_stop=True),
                             tab_attrs(div=2))])
        m = parse(xml)[0]
        ticks = [(n["tick"], n["pitch"]) for n in m["notes"]]
        # chord shares the onset; grace sits at the main note's tick with no duration; tie stop is skipped
        self.assertEqual(ticks, [(0, midi("C", 4)), (0, midi("E", 4)), (TPQ, midi("D", 4)), (TPQ, midi("D", 4))])
        self.assertEqual(m["len"], 2 * TPQ)                  # 4 divisions at div=2 = 2 quarters

    def test_string_fret_finger_and_tuning(self):
        xml = score([measure(note("A", 3, 1, 3, 2, finger=2), tab_attrs())])
        m = parse(xml)[0]
        n = m["notes"][0]
        self.assertEqual((n["string"], n["fret"], n["finger"]), (3, 2, 2))
        self.assertEqual(m["tuning"][1], midi("E", 2))
        self.assertEqual(m["tuning"][6], midi("E", 4))

    def test_missing_duration_is_estimated_from_type(self):
        xml = ('<score-partwise><part id="P1"><measure><attributes><divisions>4</divisions></attributes>'
               '<note><pitch><step>C</step><octave>4</octave></pitch><type>half</type><dot/></note>'
               '</measure></part></score-partwise>')
        self.assertEqual(parse(xml)[0]["len"], 3 * TPQ)      # dotted half = 3 quarters


class TestScoreNotes(unittest.TestCase):

    def _write(self, xml: str) -> str:
        f = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-8")
        f.write(xml); f.close()
        self.addCleanup(Path(f.name).unlink)
        return f.name

    def test_pitch_from_tab_and_finger_from_notation_staff(self):
        # notation staff (P1) carries the left-hand finger, TAB staff (P2) carries string/fret
        notation = [measure(note("G", 3, 1, finger=3), "<attributes><divisions>1</divisions></attributes>")]
        tab = [measure(note("G", 3, 1, string=3, fret=0), tab_attrs())]   # MusicXML string 3 = G (1 = high E)
        order, lens, notes, tuning = score_notes(self._write(score(notation, tab)))
        self.assertEqual(order, [0])
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["pitch"], midi("G", 3))     # tuning[7 - 3] + 0 = G3
        self.assertEqual(notes[0]["finger"], 3)
        np.testing.assert_allclose(lens, [TPQ])

    def test_written_pitch_is_overridden_by_tab(self):
        # a wrong written pitch must not matter: sounding pitch comes from string + fret
        xml = score([measure(note("C", 4, 1, string=5, fret=2), tab_attrs())])
        notes = score_notes(self._write(xml))[2]
        self.assertEqual(notes[0]["pitch"], midi("A", 2) + 2)  # line 7-5 = 2 -> A2, fret 2 = B2

    def test_explicit_order_overrides_repeat_signs(self):
        xml = score([measure(note("C", 4, 1, 2, 1), tab_attrs()), measure(note("D", 4, 1, 2, 3))])
        order, _, notes, _ = score_notes(self._write(xml), order=[0, 1, 0, 1])
        self.assertEqual(order, [0, 1, 0, 1])
        self.assertEqual([n["bar"] for n in notes], [0, 1, 2, 3])

    def test_no_tab_staff_returns_none(self):
        xml = score([measure(note("C", 4, 1), "<attributes><divisions>1</divisions></attributes>")])
        self.assertIsNone(score_notes(self._write(xml)))


class TestTimesAndMatching(unittest.TestCase):

    def test_syncpoint_interpolation_and_extrapolation(self):
        lens = np.array([TPQ, TPQ])
        notes = [dict(bar=0, tick=0), dict(bar=0, tick=TPQ / 2), dict(bar=1, tick=TPQ / 2)]
        sync = [[0, 1.0], [1, 2.0]]                            # bar 0 at 1 s, bar 1 at 2 s
        t, _ = score_times(sync, lens, notes)
        np.testing.assert_allclose(t, [1.0, 1.5, 2.5])          # last one extrapolated linearly

    def test_match_is_one_to_one_and_nearest_first(self):
        notes = [dict(pitch=60), dict(pitch=60), dict(pitch=62)]
        times = np.array([1.00, 1.05, 2.00])
        mi, _ = match_notes([(1.04, 60), (1.01, 60), (2.30, 62)], notes, times, tol=0.1)
        self.assertEqual(mi.tolist(), [1, 0, -1])              # 62 is 0.3 s away -> unmatched


class TestContentOrder(unittest.TestCase):
    """Y-2: performed bar order recovered from content when repeat signs disagree with the recording."""

    BARS = [[60, 64], [62, 65], [64, 67], [65, 69]]

    def test_identical_order_without_repeats(self):
        order, q = order_from_content(bar_similarity(self.BARS, self.BARS))
        self.assertEqual(order, [0, 1, 2, 3])
        self.assertAlmostEqual(q, 1.0)

    def test_unmarked_whole_piece_repeat(self):
        # the 112_mf1wc case: no repeat sign, performer plays everything twice
        order, _ = order_from_content(bar_similarity(self.BARS * 2, self.BARS))
        self.assertEqual(order, [0, 1, 2, 3, 0, 1, 2, 3])

    def test_unmarked_section_repeat(self):
        perf = [self.BARS[i] for i in (0, 1, 0, 1, 2, 3, 2, 3)]
        order, _ = order_from_content(bar_similarity(perf, self.BARS))
        self.assertEqual(order, [0, 1, 0, 1, 2, 3, 2, 3])

    def test_single_wrong_note_does_not_cause_a_jump(self):
        perf = [list(b) for b in self.BARS]
        perf[2] = [64, 66]                                     # one wrong note in bar 2
        order, _ = order_from_content(bar_similarity(perf, self.BARS))
        self.assertEqual(order, [0, 1, 2, 3])

    def test_similarity_is_multiset_dice(self):
        sim = bar_similarity([[60, 60, 64]], [[60, 64, 67], []])
        self.assertAlmostEqual(sim[0, 0], 2 * 2 / 6)
        self.assertEqual(sim[0, 1], 0.0)


if __name__ == "__main__":
    unittest.main()
