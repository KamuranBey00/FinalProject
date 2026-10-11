"""
P3 — Çalınabilirlik ölçüm metriklerinin testleri (sentetik; veri seti / model gerektirmez).

Her metrik elle hesaplanmış akor şekilleriyle doğrulanır. Metrikler hüküm vermez, sayar:
testler de bunu sabitler (barre çakışma sayılmaz, ters sıra "imkânsız" sayılmaz,
**örtüşmeyen notalar fiziksel ihlal sayılmaz**).

Çalıştırma (proje kökünden): python -m unittest discover -s tests -t .
"""

import unittest

import numpy as np

from gtab.decoding.playability import (OPEN_FINGER, SPAN_THRESHOLDS, UNKNOWN_FINGER, ChordShape,
                                       barre_detail, evaluate_shapes, finger_conflicts,
                                       finger_order_inversions, group_finger_conflicts, group_span,
                                       hand_position, hand_span, position_shifts, position_spread,
                                       shapes_from_notes)


def shape(triples, onset=0.0, times=None):
    """
    [(tel, perde, parmak)] -> ChordShape. `times` verilmezse tüm notalar örtüşür;
    verilirse [(başlangıç, bitiş)] listesi olarak uygulanır.
    """
    s, f, g = zip(*triples) if triples else ((), (), ())
    if times is None:
        return ChordShape(onset, tuple(s), tuple(f), tuple(g))
    on, off = zip(*times)
    return ChordShape(onset, tuple(s), tuple(f), tuple(g), tuple(on), tuple(off))


class TestOverlapAwareness(unittest.TestCase):
    """Baslangici yakin ama zamanda ortusmeyen notalar fiziksel ihlal sayilmaz."""

    def test_ortusmeyen_ayni_parmak_cakisma_degil(self):
        """Aynı parmak, farklı perde — ama ikinci nota birinci bittikten sonra başlıyor."""
        sh = shape([(1, 5, 2), (2, 7, 2)], times=[(0.0, 0.10), (0.12, 0.30)])
        self.assertEqual(sh.overlap_pairs(), [])
        self.assertEqual(len(sh.nonoverlap_pairs()), 1)
        self.assertEqual(finger_conflicts(sh), (0, 0))
        self.assertIsNone(hand_span(sh))
        self.assertEqual(finger_order_inversions(sh), 0)

    def test_ortusen_ayni_parmak_cakisma(self):
        sh = shape([(1, 5, 2), (2, 7, 2)], times=[(0.0, 0.50), (0.02, 0.60)])
        self.assertEqual(sh.overlap_pairs(), [(0, 1)])
        self.assertEqual(finger_conflicts(sh), (1, 0))
        self.assertEqual(hand_span(sh), 2)

    def test_sinir_durumu_sifir_ortusme(self):
        """Biri tam diğerinin bittiği anda başlıyor -> örtüşme 0 -> aynı anda basılmıyor."""
        sh = shape([(1, 5, 2), (2, 7, 2)], times=[(0.0, 0.20), (0.20, 0.40)])
        self.assertEqual(sh.overlap(0, 1), 0.0)
        self.assertEqual(finger_conflicts(sh), (0, 0))

    def test_eski_tanim_ortusmeyi_gozetmez(self):
        """`group_*` eski tanım: örtüşmeyen çifti de sayar (üst sınır olarak korunuyor)."""
        sh = shape([(1, 5, 2), (2, 7, 2)], times=[(0.0, 0.10), (0.12, 0.30)])
        self.assertEqual(group_finger_conflicts(sh), 1)
        self.assertEqual(group_span(sh), 2)

    def test_kismi_ortusme_sadece_ortusen_cifti_sayar(self):
        """3 nota: 0-1 örtüşüyor, 2 ayrı. Yalnız (0,1) çifti değerlendirilir."""
        sh = shape([(1, 2, 1), (2, 4, 2), (3, 20, 4)],
                   times=[(0.0, 0.5), (0.1, 0.6), (1.0, 1.5)])
        self.assertEqual(sh.overlap_pairs(), [(0, 1)])
        self.assertEqual(hand_span(sh), 2)          # 20. perde ortusmedigi icin girmez
        self.assertEqual(group_span(sh), 18)        # eski tanim 18 derdi


class TestFingerConflicts(unittest.TestCase):
    def test_bos_teller_cakismaya_girmez(self):
        sh = shape([(0, 0, OPEN_FINGER), (1, 0, OPEN_FINGER), (2, 5, 1)])
        self.assertEqual(finger_conflicts(sh), (0, 0))
        self.assertEqual(sh.fretted, [2])

    def test_ayni_parmak_ayni_perde_barre(self):
        """Aynı parmak + aynı perde + iki tel = barre adayı, çakışma DEĞİL."""
        sh = shape([(1, 5, 1), (2, 5, 1)])
        self.assertEqual(finger_conflicts(sh), (0, 1))

    def test_ayni_parmak_farkli_perde_cakisma(self):
        self.assertEqual(finger_conflicts(shape([(1, 5, 2), (2, 7, 2)])), (1, 0))

    def test_uc_farkli_perde_uc_cakisan_cift(self):
        """Bir parmağın 3 ayrı perdesi, hepsi örtüşüyor -> 3 çakışan ÇİFT (eski tanım: 2)."""
        sh = shape([(1, 3, 3), (2, 5, 3), (3, 7, 3)])
        self.assertEqual(finger_conflicts(sh), (3, 0))
        self.assertEqual(group_finger_conflicts(sh), 2)

    def test_barre_ve_cakisma_birlikte(self):
        sh = shape([(0, 2, 1), (1, 2, 1), (2, 5, 3), (3, 7, 3)])
        self.assertEqual(finger_conflicts(sh), (1, 1))

    def test_bilinmeyen_parmak_sayilmaz(self):
        sh = shape([(1, 5, UNKNOWN_FINGER), (2, 7, UNKNOWN_FINGER)])
        self.assertEqual(finger_conflicts(sh), (0, 0))
        self.assertEqual(sh.n_unknown, 2)
        self.assertFalse(sh.evaluable)


class TestHandSpan(unittest.TestCase):
    def test_aciklik(self):
        self.assertEqual(hand_span(shape([(1, 2, 1), (2, 5, 4)])), 3)
        self.assertEqual(hand_span(shape([(1, 7, 1), (2, 7, 2), (3, 12, 4)])), 5)

    def test_tek_basili_nota_tanimsiz(self):
        self.assertIsNone(hand_span(shape([(1, 5, 1)])))
        self.assertIsNone(hand_span(shape([(1, 0, OPEN_FINGER), (2, 5, 1)])))

    def test_bos_teller_acikliga_girmez(self):
        self.assertEqual(hand_span(shape([(0, 0, OPEN_FINGER), (1, 5, 1), (2, 7, 3)])), 2)


class TestFingerOrder(unittest.TestCase):
    def test_ters_sira_tespit(self):
        self.assertEqual(finger_order_inversions(shape([(1, 5, 1), (2, 3, 2)])), 1)

    def test_dogru_sira_ters_degil(self):
        self.assertEqual(finger_order_inversions(shape([(1, 3, 1), (2, 5, 2)])), 0)

    def test_barre_cifti_ters_sayilmaz(self):
        self.assertEqual(finger_order_inversions(shape([(1, 5, 1), (2, 5, 1)])), 0)
        self.assertEqual(finger_order_inversions(shape([(1, 5, 1), (2, 5, 2)])), 0)

    def test_coklu_ters_cift(self):
        self.assertEqual(finger_order_inversions(shape([(0, 7, 1), (1, 5, 2), (2, 3, 3)])), 3)


class TestHandPosition(unittest.TestCase):
    def test_pozisyon_tahmini(self):
        self.assertEqual(hand_position(shape([(1, 5, 1)])), 5.0)
        self.assertEqual(hand_position(shape([(1, 5, 1), (2, 7, 3)])), 5.0)

    def test_tutarsiz_sekil_yayilim(self):
        self.assertEqual(position_spread(shape([(1, 5, 1), (2, 12, 4)])), 4)
        self.assertEqual(position_spread(shape([(1, 5, 1), (2, 7, 3)])), 0)

    def test_bos_tel_pozisyona_girmez(self):
        self.assertEqual(hand_position(shape([(0, 0, OPEN_FINGER), (1, 9, 2)])), 8.0)

    def test_basili_nota_yoksa_tanimsiz(self):
        self.assertIsNone(hand_position(shape([(0, 0, OPEN_FINGER)])))


class TestPositionShifts(unittest.TestCase):
    def test_kisa_bosluk_kayma(self):
        r = position_shifts([shape([(1, 1, 1)], onset=0.0), shape([(1, 5, 1)], onset=0.2)])
        self.assertEqual((r["gecis"], r["kayma"], r["toplam_hareket"]), (1, 1, 4.0))
        self.assertEqual((r["kisa_bosluk_gecis"], r["uzun_bosluk_gecis"]), (1, 0))

    def test_uzun_bosluk_ayri_sayilir(self):
        r = position_shifts([shape([(1, 1, 1)], onset=0.0), shape([(1, 9, 1)], onset=2.0)])
        self.assertEqual((r["gecis"], r["kisa_bosluk_gecis"], r["uzun_bosluk_gecis"]), (1, 0, 1))

    def test_ayni_pozisyon_kayma_degil(self):
        r = position_shifts([shape([(1, 5, 1)], onset=0.0), shape([(2, 7, 3)], onset=0.3)])
        self.assertEqual((r["gecis"], r["kayma"], r["toplam_hareket"]), (1, 0, 0.0))

    def test_degerlendirilemeyen_gruplar_atlanir(self):
        a = shape([(1, 5, 1)], onset=0.0)
        bad = shape([(1, 7, UNKNOWN_FINGER)], onset=0.3)
        b = shape([(1, 5, 1)], onset=0.6)
        self.assertEqual(position_shifts([a, bad, b])["gecis"], 1)


class TestBarreDetail(unittest.TestCase):
    def test_temiz_barre(self):
        d = barre_detail(shape([(1, 5, 1), (2, 5, 1)]))
        self.assertEqual(len(d), 1)
        self.assertIsNone(d[0]["belirsiz"])
        self.assertEqual((d[0]["parmak"], d[0]["perde"], d[0]["teller"]), (1, 5, (1, 2)))

    def test_bitisik_olmayan_teller_belirsiz(self):
        self.assertEqual(barre_detail(shape([(1, 5, 1), (4, 5, 1)]))[0]["belirsiz"],
                         "teller bitisik degil")

    def test_arada_dusuk_perde_belirsiz(self):
        d = barre_detail(shape([(1, 5, 1), (3, 5, 1), (2, 3, 2)]))
        self.assertEqual(d[0]["belirsiz"], "arada daha dusuk perde var")

    def test_barre_yok(self):
        self.assertEqual(barre_detail(shape([(1, 5, 1), (2, 7, 2)])), [])

    def test_ortusmeyen_barre_sayilmaz(self):
        """Aynı parmak + aynı perde ama ardışık çalınıyorsa barre değildir."""
        sh = shape([(1, 5, 1), (2, 5, 1)], times=[(0.0, 0.1), (0.2, 0.3)])
        self.assertEqual(barre_detail(sh), [])


class TestShapesFromNotes(unittest.TestCase):
    def test_akor_bayragindan_gruplama_ve_zamanlar(self):
        notes = np.array([[0.0, 0.5, 60], [0.0, 0.5, 64], [1.0, 1.5, 67]], float)
        shapes = shapes_from_notes(notes, [False, True, False], [1, 2, 3], [5, 7, 9], [1, 3, 1])
        self.assertEqual(len(shapes), 2)
        self.assertEqual(shapes[0].frets, (5, 7))
        self.assertEqual(shapes[0].onsets, (0.0, 0.0))
        self.assertEqual(shapes[0].offsets, (0.5, 0.5))
        self.assertEqual(shapes[1].onset, 1.0)
        self.assertEqual(hand_span(shapes[0]), 2)

    def test_tek_nota(self):
        shapes = shapes_from_notes(np.array([[0.0, 0.5, 60]], float), [False], [2], [5], [1])
        self.assertTrue(shapes[0].evaluable)
        self.assertIsNone(hand_span(shapes[0]))

    def test_ortusmeyen_grup_gercek_veri_bicimi(self):
        """(başlangıç, BİTİŞ, perde) sözleşmesi: ikinci nota ilki bittikten sonra."""
        notes = np.array([[99.761, 99.786, 57], [99.789, 99.889, 52]], float)
        shapes = shapes_from_notes(notes, [False, True], [3, 2], [2, 12], [1, 3])
        self.assertEqual(shapes[0].overlap_pairs(), [])
        self.assertIsNone(hand_span(shapes[0]))
        self.assertEqual(group_span(shapes[0]), 10)


class TestUnknownPosition(unittest.TestCase):
    def test_konumu_bilinmeyen_nota(self):
        sh = shape([(2, 5, 1), (-1, -1, UNKNOWN_FINGER)])
        self.assertEqual(sh.n_unknown, 1)
        self.assertFalse(sh.evaluable)
        self.assertEqual(sh.fretted, [0])

    def test_toplamada_degerlendirilemez_sayilir(self):
        rep = evaluate_shapes([shape([(2, 5, 1), (-1, -1, UNKNOWN_FINGER)]),
                               shape([(2, 5, 1), (3, 7, 3)])], "test")
        self.assertEqual((rep.n_groups, rep.n_evaluable, rep.n_unevaluable), (2, 1, 1))


class TestAggregate(unittest.TestCase):
    def setUp(self):
        self.shapes = [
            shape([(1, 5, 2), (2, 7, 2)], onset=0.0),                       # çakışma 1, açıklık 2
            shape([(1, 5, 1), (2, 5, 1)], onset=0.5),                       # barre, açıklık 0
            shape([(1, 7, UNKNOWN_FINGER)], onset=1.0),                     # değerlendirilemez
            shape([(0, 0, OPEN_FINGER)], onset=1.5),                        # yalnız boş tel
            shape([(1, 9, 1), (2, 7, 2)], onset=2.0),                       # ters sıra, açıklık 2
            shape([(1, 2, 1), (2, 9, 3)], onset=3.0,
                  times=[(3.0, 3.1), (3.2, 3.4)]),                          # örtüşmeyen çift
        ]
        self.rep = evaluate_shapes(self.shapes, "test")

    def test_paydalar(self):
        self.assertEqual(self.rep.n_groups, 6)
        self.assertEqual(self.rep.n_evaluable, 4)
        self.assertEqual(self.rep.n_unevaluable, 1)
        self.assertEqual(self.rep.n_empty, 1)
        self.assertEqual(self.rep.n_fretted_notes, 8)
        self.assertEqual(self.rep.n_overlap_groups, 3)          # örtüşmeyen grup A/B/C'ye girmez
        self.assertEqual(self.rep.n_overlap_pairs, 3)
        self.assertEqual((self.rep.n_nonoverlap_pairs, self.rep.groups_with_nonoverlap), (1, 1))

    def test_paylar(self):
        self.assertEqual((self.rep.conflicts, self.rep.groups_with_conflict), (1, 1))
        self.assertEqual((self.rep.barre_clean, self.rep.barre_ambiguous), (1, 0))
        self.assertEqual((self.rep.inversions, self.rep.groups_with_inversion), (1, 1))
        self.assertEqual(sorted(self.rep.spans), [0, 2, 2])
        self.assertEqual(sorted(self.rep.group_spans), [0, 2, 2, 7])
        self.assertEqual(self.rep.group_conflicts, 1)

    def test_oranlar(self):
        r = self.rep.rates()
        self.assertAlmostEqual(r["degerlendirilebilir_oran"], 4 / 6)
        self.assertAlmostEqual(r["cakisma_akor_orani"], 1 / 3)       # payda: örtüşen gruplar
        self.assertAlmostEqual(r["cakisma_nota_orani"], 1 / 8)
        self.assertAlmostEqual(r["ters_sira_akor_orani"], 1 / 3)
        self.assertAlmostEqual(r["ortusmeyen_cift_orani"], 1 / 4)
        self.assertAlmostEqual(r["aciklik_medyan"], 2.0)
        for t in SPAN_THRESHOLDS:
            self.assertIn(f"aciklik>={t}", r)

    def test_determinizm(self):
        a, b = evaluate_shapes(self.shapes, "t"), evaluate_shapes(self.shapes, "t")
        self.assertEqual(a.rates(), b.rates())
        self.assertEqual((a.conflicts, a.inversions, a.spans), (b.conflicts, b.inversions, b.spans))

    def test_bos_girdi(self):
        r = evaluate_shapes([], "bos")
        self.assertEqual((r.n_groups, r.n_evaluable), (0, 0))
        self.assertTrue(np.isnan(r.rates()["aciklik_medyan"]))


if __name__ == "__main__":
    unittest.main()
