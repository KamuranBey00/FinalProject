"""
Akort / kapo tests (Katman 6 Adım 3): parse_tuning, with_capo, remap_probs, heard_positions, out-of-range count.

Pure numpy; no dataset, checkpoint or GPU needed.
Run (project root): python -m unittest tests.test_tuning
"""

import unittest

import numpy as np

from gtab.core.instrument import DROP_D_6, STANDARD_6, parse_tuning, with_capo
from gtab.decoding.retune import remap_probs
from gtab.decoding.transitions import TransitionModel
from gtab.pipeline import heard_positions
from scripts.eval.eval_tuning import out_of_model

C = STANDARD_6.num_frets + 2                       # tab classes: 0 = silent, fret f -> f + 1


def _probs(T=4, seed=0):
    p = np.random.default_rng(seed).random((T, 6, C))
    return p / p.sum(-1, keepdims=True)


class TestParseTuning(unittest.TestCase):
    def test_presets(self):
        self.assertIs(parse_tuning("DADGBE"), DROP_D_6)
        self.assertIs(parse_tuning("D2 A2 D3 G3 B3 E4"), DROP_D_6)
        self.assertIs(parse_tuning("EADGBE"), STANDARD_6)
        self.assertIs(parse_tuning("EADGBe"), STANDARD_6)

    def test_custom(self):
        self.assertEqual(parse_tuning("DADGAD").tuning, (38, 45, 50, 55, 57, 62))
        self.assertEqual(parse_tuning("Eb Ab Db Gb Bb Eb").tuning, (39, 44, 49, 54, 58, 63))
        self.assertEqual(parse_tuning("dadgbe").tuning, DROP_D_6.tuning)

    def test_bad(self):
        with self.assertRaises(ValueError):
            parse_tuning("XADGBE")


class TestCapo(unittest.TestCase):
    def test_with_capo(self):
        c = with_capo(STANDARD_6, 2)
        self.assertEqual(c.tuning, (42, 47, 52, 57, 61, 66))
        self.assertEqual(c.num_frets, 22)
        self.assertIs(with_capo(STANDARD_6, 0), STANDARD_6)
        self.assertEqual(c.pitch_range(), (42, 88))   # same top note as without capo


class TestRemap(unittest.TestCase):
    def test_identity_on_standard(self):
        p = _probs()
        self.assertIs(remap_probs(p, STANDARD_6, STANDARD_6), p)

    def test_drop_d(self):
        p = _probs()
        q = remap_probs(p, STANDARD_6, DROP_D_6)
        np.testing.assert_array_equal(q[:, 1:], p[:, 1:])          # other strings unchanged
        np.testing.assert_array_equal(q[:, 0, 0], p[:, 0, 0])      # silent class kept
        np.testing.assert_array_equal(q[:, 0, 3], p[:, 0, 1])      # drop-D fret 2 = E2 = standard open string
        np.testing.assert_array_equal(q[:, 0, 1:3], 0)             # D2 / D#2: outside the model -> 0
        np.testing.assert_array_equal(q[:, 0, C - 1], p[:, 0, C - 3])

    def test_capo(self):
        p = _probs()
        q = remap_probs(p, STANDARD_6, STANDARD_6, capo=3)
        np.testing.assert_array_equal(q[:, :, 1], p[:, :, 4])      # open string under capo 3 = standard fret 3
        np.testing.assert_array_equal(q[:, :, C - 3:], 0)          # beyond 24 − 3 frets
        np.testing.assert_array_equal(q, remap_probs(p, STANDARD_6, with_capo(STANDARD_6, 3)))


class TestHeardPositions(unittest.TestCase):
    def _onehot(self, s, f, T=4):
        p = np.full((T, 6, C), 1e-4)
        p[:, s, f + 1] = 1.0
        return p

    def test_drop_d_e2(self):
        segs = [(0, 4, 40)]                                        # E2 heard on the 6th string (standard open)
        tm = TransitionModel(STANDARD_6)
        self.assertEqual(heard_positions(segs, self._onehot(0, 0), tm), [(0, 0)])
        self.assertEqual(heard_positions(segs, self._onehot(0, 0), tm, DROP_D_6), [(0, 2)])

    def test_capo_relative_fret(self):
        # F#4 heard at standard 1st string fret 2 -> with capo 2 it is that string's open (capo) position
        self.assertEqual(heard_positions([(0, 4, 66)], self._onehot(5, 2), TransitionModel(STANDARD_6),
                                          with_capo(STANDARD_6, 2)), [(5, 0)])

    def test_transition_weight_rejected(self):
        with self.assertRaises(ValueError):
            heard_positions([(0, 4, 40)], self._onehot(0, 0), TransitionModel(STANDARD_6), DROP_D_6, w_transition=1.0)


class TestOutOfRange(unittest.TestCase):
    def test_drop_d_low_notes_counted(self):
        pitches = np.array([38, 39, 40, 52, 88, 89])
        np.testing.assert_array_equal(out_of_model(pitches), [True, True, False, False, False, True])
        self.assertEqual([p for p in (38, 39, 40) if DROP_D_6.pitch_to_positions(p)], [38, 39, 40])


if __name__ == "__main__":
    unittest.main()
