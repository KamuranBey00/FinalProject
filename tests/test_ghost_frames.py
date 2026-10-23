"""
README22 Aşama 1a — hayalet negatifleri (tab_labels.ghost_frames) testleri (sentetik; model/veri gerekmez).

Etiket: perde 45 (A2), 10–40. kareler. Tespitler: gerçek nota (eşleşir), +12 üst ses hayaleti, aynı perdenin 15 kare
sonraki parçası, ve +5'lik ilgisiz bir hayalet. Yalnız üst ses ve parça negatif olmalı; gerçek onset'e dokunulmamalı.

Çalıştırma (proje kökünden): python -m unittest discover -s tests -t .
"""

import unittest

import numpy as np

from gtab.data.tab_labels import ghost_frames

LO, T, P = 40, 60, 30


def labels():
    frame, onset = np.zeros((T, P), np.uint8), np.zeros((T, P), np.uint8)
    frame[10:40, 45 - LO] = 1
    onset[10, 45 - LO] = 1
    return frame, onset


class TestGhostFrames(unittest.TestCase):
    def setUp(self):
        est = [(10, 40, 45), (12, 30, 57), (25, 40, 45), (12, 20, 50)]
        self.m = ghost_frames(est, *labels(), LO)

    def test_ust_ses_hayaleti_negatif(self):
        self.assertTrue(self.m[11:14, 57 - LO].all())

    def test_parca_negatif(self):
        self.assertTrue(self.m[24:27, 45 - LO].all())

    def test_ilgisiz_hayalet_ve_gercek_onset_degil(self):
        self.assertFalse(self.m[:, 50 - LO].any())
        self.assertFalse(self.m[8:13, 45 - LO].any())             # gerçek onset ±2 kare korunur
        self.assertEqual(int(self.m.sum()), 6)

    def test_tespit_yoksa_bos(self):
        self.assertFalse(ghost_frames([], *labels(), LO).any())


if __name__ == "__main__":
    unittest.main()
