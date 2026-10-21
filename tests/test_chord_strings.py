"""
README22 Adım 2B — "ses" modu akor kısıtı (transitions.resolve_chord_strings) testleri (sentetik; model/veri gerekmez).

İki nota aynı anda başlıyor ve ikisinin de en olası teli 0: kısıt, olasılık toplamı en yüksek farklı-tel atamasını
seçmeli; çakışmayan gruplar ve ayrı zamanlı notalar olduğu gibi kalmalı.

Çalıştırma (proje kökünden): python -m unittest discover -s tests -t .
"""

import unittest

import numpy as np

from gtab.decoding.transitions import decode_lattice, resolve_chord_strings


def lattice():
    cands = [[(0, 5), (1, 0)], [(0, 7), (1, 2)], [(2, 3), (3, 0)]]
    E = [np.log([0.9, 0.1]), np.log([0.8, 0.2]), np.log([0.6, 0.4])]
    T = [np.zeros((1, 2))] + [np.zeros((2, 2))] * 2
    return cands, E, T


class TestChordStrings(unittest.TestCase):
    def test_cakisan_akor_ayri_tellere(self):
        lat = lattice()
        picks = decode_lattice(lat, 1.0, 0.0)
        self.assertEqual(picks[:2], [(0, 5), (0, 7)])                 # greedy: ikisi de tel 0
        out = resolve_chord_strings(lat, picks, [[0, 1], [2]])
        self.assertEqual(out[:2], [(0, 5), (1, 2)])                   # 0.9*0.2 > 0.1*0.8
        self.assertEqual(out[2], picks[2])

    def test_ayri_zamanli_notalar_degismez(self):
        lat = lattice()
        picks = decode_lattice(lat, 1.0, 0.0)
        self.assertEqual(resolve_chord_strings(lat, picks, [[0], [1], [2]]), picks)

    def test_cakismasiz_akor_degismez(self):
        lat = lattice()
        picks = decode_lattice(lat, 1.0, 0.0)
        self.assertEqual(resolve_chord_strings(lat, picks, [[0, 2], [1]]), picks)


if __name__ == "__main__":
    unittest.main()
