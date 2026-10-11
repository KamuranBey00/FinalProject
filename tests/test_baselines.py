"""
Katman 4 Faz 0 — D1 referans tablosu testleri (regresyon koruması).

Çekirdek değerler burada ELLE yazılı: tablo ile bu test birbirini doğrular. Bir sayı
"iyileştirmek" için tabloda değiştirilirse test düşer.

Çalıştırma (proje kökünden): python -m unittest discover -s tests -t .
"""

import unittest

from gtab.evaluation.baselines import (MODE_RERUN, MODE_RETRAIN, REQUIRED_FIELDS, assert_comparable,
                                       compare, core_metrics, format_table, load_baselines, metric,
                                       metrics, tolerance)
from gtab.evaluation.protocol import is_independent_test, rule

D1_CEKIRDEK = {
    "gs_solo_note_f1": 0.924,
    "gs_poly_note_f1": 0.807,
    "gs_solo_tab_f1": 0.777,
    "gaps_note_f1": 0.779,
    "gaps_edition_string_acc": 0.815,
    "gaps_edition_finger_acc": 0.636,
}


class TestTableIntegrity(unittest.TestCase):
    def test_alanlar_ve_tekrarsizlik(self):
        table = load_baselines()
        ids = [m["id"] for m in table["metrikler"]]
        self.assertEqual(len(ids), len(set(ids)))
        for m in table["metrikler"]:
            with self.subTest(metric=m["id"]):
                for field in REQUIRED_FIELDS:
                    self.assertIn(field, m)
                self.assertIsInstance(m["deger"], float)
                self.assertTrue(0.0 <= m["deger"] <= 1.0)
                self.assertTrue(m["kosullu"], "kosullu/oracle bilgisi bos olamaz")
                self.assertTrue(m["birim"], "degerlendirme birimi bos olamaz")

    def test_cekirdek_d1_degerleri(self):
        core = {m["id"]: m["deger"] for m in core_metrics()}
        self.assertEqual(core, D1_CEKIRDEK)

    def test_split_rolu_protokolden(self):
        """Her referans değer raporlanabilir bir split'ten gelmeli ve cinsi doğru etiketlenmeli."""
        for m in metrics().values():
            with self.subTest(metric=m["id"]):
                r = rule(m["split"])
                self.assertTrue(r.may_report, "raporlanamayan split'ten referans olmaz")
                self.assertEqual(m["split_rolu"], r.report_label)
                if m["split"] in ("val", "val_comp", "gaps_val"):
                    self.assertIn("validation", m["split_rolu"])
                    self.assertFalse(is_independent_test(m["split"]))
                if m["split"] == "gaps_test":
                    self.assertTrue(is_independent_test(m["split"]))

    def test_toleranslar(self):
        self.assertAlmostEqual(tolerance(MODE_RERUN), 0.002)
        self.assertAlmostEqual(tolerance(MODE_RETRAIN), 0.01)
        self.assertLess(tolerance(MODE_RERUN), tolerance(MODE_RETRAIN))

    def test_tablo_metni(self):
        text = format_table(only_core=True)
        self.assertEqual(len(text.splitlines()), len(D1_CEKIRDEK) + 1)
        self.assertIn("validation (bagimsiz test degil)", text)
        self.assertIn("bagimsiz test", format_table())


class TestCompare(unittest.TestCase):
    def test_tolerans_siniri(self):
        tol = tolerance(MODE_RERUN)
        base = D1_CEKIRDEK["gaps_note_f1"]
        self.assertTrue(compare("gaps_note_f1", base + tol)["ok"])
        self.assertFalse(compare("gaps_note_f1", base + tol + 1e-6)["ok"])
        self.assertTrue(compare("gaps_note_f1", base - tol)["ok"])
        self.assertFalse(compare("gaps_note_f1", base - tol - 1e-6)["ok"])

    def test_delta_ve_alanlar(self):
        res = compare("gs_solo_tab_f1", 0.760)
        self.assertAlmostEqual(res["delta"], 0.760 - 0.777, places=6)
        self.assertFalse(res["ok"])
        self.assertEqual(res["split"], "val")
        self.assertIn("validation", res["split_rolu"])

    def test_yeniden_egitim_modu_daha_gevsek(self):
        self.assertFalse(compare("gs_solo_tab_f1", 0.770, MODE_RERUN)["ok"])
        self.assertTrue(compare("gs_solo_tab_f1", 0.770, MODE_RETRAIN)["ok"])

    def test_bilinmeyen_metrik(self):
        with self.assertRaises(KeyError):
            metric("tab_f1")


class TestMetricMixing(unittest.TestCase):
    def test_ayni_grup_karsilastirilabilir(self):
        """Naif tel tabanı ile edisyon modeli aynı birim + aynı koşul -> karşılaştırılabilir."""
        assert_comparable("gaps_edition_string_acc", "gaps_naive_string_acc")
        assert_comparable("gaps_edition_finger_acc", "gaps_finger_baseline_acc")

    def test_farkli_grup_hata(self):
        """0.815 (oracle nota + edisyon) ile 0.655 (uçtan uca) aynı tabloda kıyaslanamaz."""
        for a, b in (("gaps_edition_string_acc", "gaps_e2e_string_acc"),
                     ("gaps_edition_string_acc", "gaps_audio_oracle_string_acc"),
                     ("gs_solo_note_f1", "gs_solo_tab_f1"),
                     ("gaps_edition_finger_acc", "gaps_edition_product_acc")):
            with self.subTest(pair=(a, b)), self.assertRaises(ValueError):
                assert_comparable(a, b)

    def test_oracle_bilgisi_yazili(self):
        """Oracle (koşullu) sayılar ürün sayısı gibi sunulmasın: alan açıkça söylemeli."""
        self.assertIn("ORACLE NOTA", metric("gaps_edition_string_acc")["kosullu"])
        self.assertIn("ORACLE KONUM", metric("gaps_edition_finger_acc")["kosullu"])
        self.assertIn("yok", metric("gaps_e2e_string_acc")["kosullu"])


if __name__ == "__main__":
    unittest.main()
