"""
Katman 4 Faz 0 — Model seçimi / early stopping / aşırı öğrenme kuralı testleri (sentetik epoch'lar).

Çalıştırma (proje kökünden): python -m unittest discover -s tests -t .
"""

import unittest

from gtab.evaluation.protocol import ProtocolError
from gtab.evaluation.selection import EpochSelector


class TestSelectionGate(unittest.TestCase):
    def test_test_setinde_secim_yasak(self):
        with self.assertRaises(ProtocolError):
            EpochSelector("gaps_test")
        with self.assertRaises(ProtocolError):
            EpochSelector("val_half_a")
        with self.assertRaises(ProtocolError):
            EpochSelector("gaps_fit")

    def test_kalibrasyon_setleri_izinli(self):
        for split in ("val", "val_comp", "gaps_val"):
            self.assertEqual(EpochSelector(split).selection_split, split)


class TestBestEpoch(unittest.TestCase):
    def test_en_iyi_epoch_ve_checkpoint_isareti(self):
        s = EpochSelector("gaps_val")
        self.assertTrue(s.update(0, 1.0, 0.50))        # baslangic modeli de degerlendirilir
        self.assertTrue(s.update(1, 0.9, 0.55))
        self.assertFalse(s.update(2, 0.8, 0.54))       # iyilesme yok -> checkpoint yazilmaz
        self.assertEqual(s.best.epoch, 1)
        self.assertAlmostEqual(s.best.val_score, 0.55)

    def test_baslangic_modeli_korunur(self):
        """Katman 3.13 Adım 1 davranışı: hiçbir epoch ep 0'ı geçmezse taban korunur."""
        s = EpochSelector("gaps_val")
        s.update(0, 1.0, 0.60)
        for ep, sc in ((1, 0.59), (2, 0.58), (3, 0.60)):
            self.assertFalse(s.update(ep, 1.0 - ep * 0.1, sc))
        self.assertTrue(s.kept_initial())
        self.assertIn("baslangic modeli korundu", s.format_log())

    def test_min_delta_gurultuyu_yutar(self):
        s = EpochSelector("val", min_delta=0.01)
        s.update(0, 1.0, 0.70)
        self.assertFalse(s.update(1, 0.9, 0.705))      # +0.005 < min_delta -> gurultu
        self.assertTrue(s.update(2, 0.8, 0.72))


class TestEarlyStopping(unittest.TestCase):
    def test_patience_dolunca_durur(self):
        s = EpochSelector("val", patience=2)
        s.update(0, 1.0, 0.5); s.update(1, 0.9, 0.6)
        s.update(2, 0.8, 0.59)
        self.assertFalse(s.should_stop())              # 1 epoch iyilesme yok
        s.update(3, 0.7, 0.58)
        self.assertTrue(s.should_stop())               # 2 epoch
        self.assertIn("early stopping", s.format_log())

    def test_iyilesme_sayaci_sifirlanir(self):
        s = EpochSelector("val", patience=2)
        s.update(0, 1.0, 0.5); s.update(1, 0.9, 0.49); s.update(2, 0.8, 0.51)
        self.assertEqual(s.epochs_since_best, 0)
        self.assertFalse(s.should_stop())


class TestOverfitFlag(unittest.TestCase):
    def test_kayip_dusuyor_olcut_durgun(self):
        s = EpochSelector("gaps_val")
        s.update(0, 1.00, 0.60)
        s.update(1, 0.70, 0.60)
        s.update(2, 0.40, 0.59)
        s.update(3, 0.09, 0.58)
        self.assertTrue(s.overfitting())
        self.assertIn("asiri ogrenme", s.format_log())

    def test_saglikli_egitim_isaretlenmez(self):
        s = EpochSelector("gaps_val")
        for ep, (loss, score) in enumerate([(1.0, 0.50), (0.8, 0.55), (0.6, 0.60), (0.5, 0.64)]):
            s.update(ep, loss, score)
        self.assertFalse(s.overfitting())
        self.assertNotIn("asiri ogrenme", s.format_log())

    def test_yetersiz_gecmiste_isaretlenmez(self):
        s = EpochSelector("gaps_val")
        s.update(0, 1.0, 0.5); s.update(1, 0.5, 0.5)
        self.assertFalse(s.overfitting())

    def test_log_formati(self):
        s = EpochSelector("val", metric_name="macro-F1")
        s.update(0, 1.0, 0.5, val_loss=0.9)
        log = s.format_log()
        self.assertIn("secim split'i: val", log)
        self.assertIn("macro-F1", log)
        self.assertIn("dogrulama kaybi", log)
        self.assertIn("<- en iyi", log)


if __name__ == "__main__":
    unittest.main()
