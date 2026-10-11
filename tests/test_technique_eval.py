"""
Katman 4 Faz 0 — Teknik tanıma ölçütleri testleri (elle hesaplanmış sentetik örnekler).

Çalıştırma (proje kökünden): python -m unittest discover -s tests -t .
"""

import unittest

from gtab.evaluation.technique_eval import (MISSING, MODE_E2E, MODE_MATCHED, NORMAL, ONSET_TOL,
                                            SOURCE_DERIVED, SOURCE_REAL, STATE_PHANTOM,
                                            STATE_SCORED, STATE_UNDEFINED, align_labels,
                                            assert_not_mixed, confusion, evaluate, format_report,
                                            majority_baseline, match_events)

VIB, LEG = "VIBRATO", "LEGATO"
CLASSES = (NORMAL, VIB, LEG)

# Elle hesaplanan ana ornek (4 nota, hepsi eslesir):
#   NORMAL : tp 1, fp 1, fn 0, support 1 -> P .5  R 1.0 F1 .6667
#   VIBRATO: tp 1, fp 1, fn 1, support 2 -> P .5  R .5  F1 .5
#   LEGATO : tp 0, fp 0, fn 1, support 1 -> P 0   R 0   F1 0
GOLD = [(0.00, 60, NORMAL), (0.50, 62, VIB), (1.00, 64, VIB), (1.50, 65, LEG)]
PRED = [(0.01, 60, NORMAL), (0.52, 62, NORMAL), (1.00, 64, VIB), (1.49, 65, VIB)]


class TestMatching(unittest.TestCase):
    def test_tolerans_siniri_dahil(self):
        self.assertEqual(match_events([(1.0, 60, NORMAL)], [(1.0 + ONSET_TOL, 60, NORMAL)]), [(0, 0)])
        self.assertEqual(match_events([(1.0, 60, NORMAL)], [(1.0 + ONSET_TOL + 1e-3, 60, NORMAL)]), [])

    def test_perde_farkli_eslesmez(self):
        self.assertEqual(match_events([(1.0, 60, NORMAL)], [(1.0, 61, NORMAL)]), [])

    def test_bire_bir_en_yakin_once(self):
        """Tek gerçek notaya iki aday: yalnız en yakını eşleşir."""
        pairs = match_events([(1.0, 60, NORMAL)], [(1.04, 60, VIB), (1.01, 60, LEG)])
        self.assertEqual(pairs, [(0, 1)])

    def test_eslesmeyenler_sayilir(self):
        g, p, missed, phantom = align_labels(GOLD + [(5.0, 67, LEG)], PRED + [(9.0, 69, VIB)],
                                             MODE_MATCHED)
        self.assertEqual((len(g), len(p), missed, phantom), (4, 4, 1, 1))

    def test_uctan_uca_doldurma(self):
        g, p, missed, phantom = align_labels(GOLD + [(5.0, 67, LEG)], PRED + [(9.0, 69, VIB)], MODE_E2E)
        self.assertEqual((missed, phantom), (1, 1))
        self.assertEqual(len(g), 6)
        self.assertEqual(p[4], MISSING)      # bulunamayan gercek nota
        self.assertEqual(g[5], MISSING)      # hayalet tahmin

    def test_gecersiz_mod(self):
        with self.assertRaises(ValueError):
            align_labels(GOLD, PRED, "bilinmeyen")


class TestClassScores(unittest.TestCase):
    def setUp(self):
        self.rep = evaluate(GOLD, PRED, CLASSES, SOURCE_REAL, MODE_MATCHED)

    def test_per_class_degerleri(self):
        n, v, l = (self.rep.per_class[c] for c in CLASSES)
        self.assertEqual((n.tp, n.fp, n.fn, n.support), (1, 1, 0, 1))
        self.assertAlmostEqual(n.precision, 0.5); self.assertAlmostEqual(n.recall, 1.0)
        self.assertAlmostEqual(n.f1, 2 / 3, places=6)
        self.assertEqual((v.tp, v.fp, v.fn, v.support), (1, 1, 1, 2))
        self.assertAlmostEqual(v.f1, 0.5)
        self.assertEqual((l.tp, l.fp, l.fn, l.support), (0, 0, 1, 1))
        self.assertAlmostEqual(l.precision, 0.0)      # hic tahmin yok -> P = 0
        self.assertAlmostEqual(l.f1, 0.0)

    def test_toplamlar(self):
        self.assertAlmostEqual(self.rep.macro_f1, (2 / 3 + 0.5 + 0.0) / 3, places=6)
        self.assertAlmostEqual(self.rep.weighted_f1, (2 / 3 * 1 + 0.5 * 2 + 0.0 * 1) / 4, places=6)
        self.assertAlmostEqual(self.rep.macro_f1_no_normal, 0.25, places=6)
        self.assertLess(self.rep.macro_f1_no_normal, self.rep.weighted_f1)

    def test_basliklar_tasinir(self):
        self.assertEqual(self.rep.label_source, SOURCE_REAL)
        self.assertEqual(self.rep.mode, MODE_MATCHED)
        self.assertEqual(self.rep.tol, ONSET_TOL)
        self.assertIn("nota olayi", self.rep.unit)
        self.assertIn("macro-F1 (ANA)", format_report(self.rep))

    def test_uctan_uca_cezalandirir(self):
        rep = evaluate(GOLD + [(5.0, 67, LEG)], PRED + [(9.0, 69, VIB)], CLASSES,
                       SOURCE_REAL, MODE_E2E)
        self.assertEqual(rep.per_class[LEG].support, 2)
        self.assertEqual(rep.per_class[LEG].fn, 2)
        self.assertEqual(rep.per_class[VIB].fp, 2)          # 1.49 hatasi + hayalet nota
        self.assertLess(rep.macro_f1, self.rep.macro_f1)


class TestEmptyClasses(unittest.TestCase):
    def test_tanimsiz_sinif_makroya_girmez(self):
        """support 0 ve tahmin yok -> tanimsiz; macro-F1 bu siniftan etkilenmez."""
        gold = [(0.0, 60, NORMAL)]
        pred = [(0.0, 60, NORMAL)]
        rep = evaluate(gold, pred, CLASSES, SOURCE_DERIVED)
        self.assertEqual(rep.per_class[VIB].state, STATE_UNDEFINED)
        self.assertIsNone(rep.per_class[VIB].f1)
        self.assertEqual(rep.undefined_classes, (VIB, LEG))
        self.assertAlmostEqual(rep.macro_f1, 1.0)
        self.assertAlmostEqual(rep.macro_f1_strict, 1.0)

    def test_hayalet_sinif(self):
        """support 0 ama tahmin var -> hayalet: ana macro-F1'e girmez, strict'e 0 olarak girer."""
        gold = [(0.0, 60, NORMAL), (1.0, 62, NORMAL)]
        pred = [(0.0, 60, NORMAL), (1.0, 62, VIB)]
        rep = evaluate(gold, pred, CLASSES, SOURCE_DERIVED)
        self.assertEqual(rep.per_class[VIB].state, STATE_PHANTOM)
        self.assertEqual(rep.phantom_classes, (VIB,))
        self.assertAlmostEqual(rep.per_class[VIB].precision, 0.0)
        self.assertIsNone(rep.per_class[VIB].recall)
        self.assertAlmostEqual(rep.per_class[VIB].f1, 0.0)
        self.assertAlmostEqual(rep.macro_f1, rep.per_class[NORMAL].f1)
        self.assertAlmostEqual(rep.macro_f1_strict, rep.per_class[NORMAL].f1 / 2)

    def test_hic_sinif_olculmezse_nan(self):
        rep = evaluate([], [], CLASSES, SOURCE_REAL)
        self.assertNotEqual(rep.macro_f1, rep.macro_f1)      # nan
        self.assertEqual(rep.n_pairs, 0)

    def test_support_var_tahmin_yok(self):
        rep = evaluate([(0.0, 60, VIB)], [(0.0, 60, NORMAL)], CLASSES, SOURCE_REAL)
        self.assertEqual(rep.per_class[VIB].state, STATE_SCORED)
        self.assertAlmostEqual(rep.per_class[VIB].f1, 0.0)


class TestConfusion(unittest.TestCase):
    def test_sayimlar_ve_sekil(self):
        g, p, _, _ = align_labels(GOLD, PRED, MODE_MATCHED)
        m = confusion(g, p, CLASSES)
        self.assertEqual(m.shape, (len(CLASSES) + 1, len(CLASSES) + 1))
        self.assertEqual(int(m.sum()), 4)
        self.assertEqual(m[0, 0], 1)                      # NORMAL -> NORMAL
        self.assertEqual(m[1, 0], 1)                      # VIBRATO -> NORMAL
        self.assertEqual(m[1, 1], 1)                      # VIBRATO -> VIBRATO
        self.assertEqual(m[2, 1], 1)                      # LEGATO -> VIBRATO
        self.assertEqual(m[:, -1].sum(), 0)               # eslesmis modda MISSING yok

    def test_uctan_uca_missing_satir_sutun(self):
        rep = evaluate(GOLD + [(5.0, 67, LEG)], PRED + [(9.0, 69, VIB)], CLASSES, SOURCE_REAL, MODE_E2E)
        m, labels = rep.confusion, rep.confusion_labels
        self.assertEqual(labels[-1], MISSING)
        self.assertEqual(m[labels.index(LEG), -1], 1)     # kacan gercek nota
        self.assertEqual(m[-1, labels.index(VIB)], 1)     # hayalet tahmin
        self.assertEqual(int(m.sum()), 6)

    def test_bilinmeyen_etiket_hata(self):
        with self.assertRaises(ValueError):
            confusion([NORMAL], ["BEND"], CLASSES)


class TestGuards(unittest.TestCase):
    def test_etiket_kaynagi_zorunlu(self):
        with self.assertRaises(ValueError):
            evaluate(GOLD, PRED, CLASSES, "hex-pickup")

    def test_kaynaklar_karistirilamaz(self):
        assert_not_mixed([SOURCE_REAL, SOURCE_REAL])
        with self.assertRaises(ValueError):
            assert_not_mixed([SOURCE_REAL, SOURCE_DERIVED])

    def test_sinif_listesi_dogrulanir(self):
        with self.assertRaises(ValueError):
            evaluate(GOLD, PRED, (NORMAL, NORMAL), SOURCE_REAL)
        with self.assertRaises(ValueError):
            evaluate(GOLD, PRED, (NORMAL, MISSING), SOURCE_REAL)

    def test_cogunluk_tabani(self):
        """NORMAL baskınlığı: taban yüksek weighted-F1 alır ama macro-F1'i düşüktür."""
        gold = [(0.0, 60, NORMAL), (1.0, 61, NORMAL), (2.0, 62, NORMAL), (3.0, 63, VIB)]
        base = majority_baseline(gold, CLASSES, SOURCE_DERIVED)
        self.assertAlmostEqual(base.per_class[NORMAL].recall, 1.0)
        self.assertAlmostEqual(base.per_class[VIB].f1, 0.0)
        self.assertGreater(base.weighted_f1, base.macro_f1)
        self.assertAlmostEqual(base.macro_f1_no_normal, 0.0)


if __name__ == "__main__":
    unittest.main()
