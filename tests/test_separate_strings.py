"""
README22 Adım 2C — ritim sıkışması (rhythm_net.separate_strings) testleri (sentetik; model/veri gerekmez).

İki nota aynı tick'e (24 = 1. vuruş başı) ve aynı tele düşüyor. Ağ, 24'ü ilk nota için ikinciden daha olası
buluyor (kayan = daha zayıf olan). İkinci nota için: 24 (şimdiki), 30 (onaltılık; ızgara karmaşıklığı 2), 36
(sekizlik; karmaşıklık 1). first_lp ile ilk nota zayıf yapılırsa o geri kayar (18 en olası geri konumu).

Çalıştırma (proje kökünden): python -m unittest discover -s tests -t .
"""

import unittest

import numpy as np

from gtab.models.rhythm_net import N_CLS, TICKS, WIN, separate_strings


def case(pitch2=52, string2=1, third=None, first_lp=0.0):
    notes = [(0.0, 0.4, 50), (0.1, 0.4, pitch2)] + ([third] if third else [])
    q = np.array([1.0, 1.0] + ([third[3]] if third else []))
    n = len(notes)
    cand = np.array([24 + np.arange(-WIN, WIN + 1)] * n)
    lp = np.full((n, N_CLS), -5.0)
    for c, v in ((24, -0.1), (30, -0.5), (36, -2.0)):
        lp[1, c - 24 + WIN] = v
    for c, v in ((24, first_lp), (18, -0.3)):
        lp[0, c - 24 + WIN] = v
    string = np.array([1, string2] + ([third[4]] if third else []))
    return q, cand, lp, np.array([nt[:3] for nt in notes], float), string


class TestSeparateStrings(unittest.TestCase):
    def test_sonraki_nota_agin_tercihine_kayar(self):
        q = separate_strings(*case(), lam=1.0)
        self.assertEqual(q[0], 1.0)
        self.assertEqual(q[1] * TICKS, 30)                 # -0.5 - 2 > -2.0 - 1

    def test_zayif_olan_onceki_nota_geri_kayar(self):
        q = separate_strings(*case(first_lp=-3.0), lam=1.0)
        self.assertEqual(q[0] * TICKS, 18)                 # 18: onaltılık, ilk notanın geri en olası konumu
        self.assertEqual(q[1], 1.0)

    def test_buyuk_lam_kaba_izgarayi_secer(self):
        self.assertEqual(separate_strings(*case(), lam=5.0)[1] * TICKS, 36)

    def test_ayni_perde_de_kayar_silinmez(self):
        q = separate_strings(*case(pitch2=50), lam=1.0)            # Adım 2D: aynı perdeli çift de ayrılır
        self.assertEqual((q[0], q[1] * TICKS), (1.0, 30))
        self.assertEqual(len(q), 2)

    def test_farkli_tel_degismez(self):
        self.assertEqual(list(separate_strings(*case(string2=2))), [1.0, 1.0])

    def test_sonraki_notayi_gecmez(self):
        # 0.2 sn'deki üçüncü nota 28. tick'te (başka tel): ikinci nota 24 < t <= 28 içinde kalmalı
        q = separate_strings(*case(third=(0.2, 0.4, 60, 28 / TICKS, 3)), lam=1.0)
        self.assertTrue(24 < q[1] * TICKS <= 28)


if __name__ == "__main__":
    unittest.main()
