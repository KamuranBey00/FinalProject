"""
README22 Adım 2A — iki kanıtlı yeniden vuruş kapısı (viterbi.segment_notes_onset reattack_hi / reattack_rise) testleri
(sentetik; model/veri gerekmez).

Senaryo: tek perde 40 kare çalıyor; 0. karede güçlü onset (0.9), nota sürerken 20. karede eşiğin hemen üstünde zayıf
bir yeniden vuruş (0.55). Bugünkü kapı (reattack 0.5) bunu yeni nota sayar; iki kanıtlı kapı ancak tepe yeterince
yüksekse ya da enerji yükseldiyse sayar.

Çalıştırma (proje kökünden): python -m unittest discover -s tests -t .
"""

import unittest

import numpy as np

from gtab.decoding.viterbi import segment_notes_onset

T = 40


def scene(rise_at_reattack: float = 0.0):
    pitch = np.zeros((T, 1), np.float32)
    pitch[:T] = 0.9
    onset = np.zeros((T, 1), np.float32)
    onset[0:2] = 0.9
    onset[20:22] = 0.55
    rise = np.zeros((T, 1), np.float32)
    rise[20] = rise_at_reattack
    return pitch, onset, rise


def n_notes(rise_at_reattack: float = 0.0, **gate) -> int:
    pitch, onset, rise = scene(rise_at_reattack)
    return len(segment_notes_onset(pitch, onset, 40, threshold=0.5, onset_threshold=0.4, reattack=0.5,
                                   rise=rise, **gate))


class TestReattackGate(unittest.TestCase):
    def test_kapali_iken_bugunku_davranis(self):
        self.assertEqual(n_notes(), 2)                                    # 0.55 >= 0.5: yeni nota
        self.assertEqual(n_notes(reattack_hi=0.0, reattack_rise=0.0), 2)

    def test_zayif_ve_enerjisiz_vurus_yeni_nota_degil(self):
        self.assertEqual(n_notes(reattack_hi=0.7), 1)
        self.assertEqual(n_notes(reattack_hi=0.7, reattack_rise=3.0), 1)

    def test_enerji_kaniti_gercek_tekrari_korur(self):
        self.assertEqual(n_notes(5.0, reattack_hi=0.7, reattack_rise=3.0), 2)
        self.assertEqual(n_notes(2.0, reattack_hi=0.7, reattack_rise=3.0), 1)

    def test_yuksek_tepe_tek_basina_yeter(self):
        self.assertEqual(n_notes(reattack_hi=0.5, reattack_rise=3.0), 2)


class TestFallbackRise(unittest.TestCase):
    # Adım 3A: onset'siz (fallback) nota ancak başında enerji yükseldiyse kabul
    def fallback_notes(self, rise_at_start: float, **gate) -> int:
        pitch = np.zeros((T, 1), np.float32)
        pitch[10:30] = 0.9                                         # 20 kare aktif, hiç onset yok
        rise = np.zeros((T, 1), np.float32)
        rise[10] = rise_at_start
        return len(segment_notes_onset(pitch, np.zeros((T, 1), np.float32), 40, threshold=0.5,
                                       onset_threshold=0.4, fallback=10, rise=rise, **gate))

    def test_kapali_iken_bugunku_davranis(self):
        self.assertEqual(self.fallback_notes(0.0), 1)

    def test_enerjisiz_onsetsiz_nota_atilir(self):
        self.assertEqual(self.fallback_notes(1.0, fallback_rise=2.0), 0)

    def test_enerjili_onsetsiz_nota_kalir(self):
        self.assertEqual(self.fallback_notes(5.0, fallback_rise=2.0), 1)


if __name__ == "__main__":
    unittest.main()
