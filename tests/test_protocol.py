"""
Katman 4 Faz 0 — Veri protokolü ve sızıntı kontrolü testleri (sentetik veri; veri seti gerekmez).

Çalıştırma (proje kökünden): python -m unittest discover -s tests -t .
"""

import os
import unittest

from gtab.evaluation.protocol import (EGITIM, KESIF, QUARANTINE_GAPS_IDS, RAPOR, SECIM, SPLITS,
                                      YASAK, ProtocolError, assert_may_report, assert_may_select,
                                      assert_may_train, channel_leakage_suspects,
                                      drop_quarantined, forbidden_gaps_ids, gaps_training_files,
                                      is_independent_test, is_quarantined, normalize_performer,
                                      performer_overlap, player05_halves, report_label, rule,
                                      split_intersection, unnamed_gaps_ids)


class TestSplitRules(unittest.TestCase):
    def test_roller_tutarli(self):
        """Eğitim split'i seçimde, seçim split'i eğitimde kullanılamaz."""
        for r in SPLITS.values():
            with self.subTest(split=r.name):
                self.assertFalse(r.may_train and r.may_select, "ayni split hem egitim hem secim olamaz")
                self.assertIn(r.role, (EGITIM, SECIM, RAPOR, KESIF, YASAK))
                if r.role == EGITIM:
                    self.assertTrue(r.may_train); self.assertFalse(r.may_report)
                if r.role == YASAK:
                    self.assertFalse(r.may_train or r.may_select or r.may_report)

    def test_bagimsiz_test_yalniz_gaps_test(self):
        """Tek bağımsız test seti gaps_test; GuitarSet oyuncu 05 validation'dır."""
        self.assertEqual([n for n in SPLITS if is_independent_test(n)], ["gaps_test"])
        self.assertIn("validation", report_label("val"))
        self.assertIn("bagimsiz test", report_label("gaps_test"))

    def test_bagimsiz_test_secimde_kullanilamaz(self):
        for r in SPLITS.values():
            if r.independent_test:
                self.assertFalse(r.may_select); self.assertFalse(r.may_train)
                self.assertTrue(r.may_report)

    def test_tanimsiz_split_hata(self):
        with self.assertRaises(ProtocolError):
            rule("gaps_test2")
        with self.assertRaises(ProtocolError):
            assert_may_select("bilinmeyen")

    def test_egitim_kapisi(self):
        assert_may_train(["train", "train_comp", "gaps_fit"])
        for bad in ("val", "val_comp", "gaps_val", "gaps_test", "gaps_skipped", "val_half_a"):
            with self.subTest(split=bad), self.assertRaises(ProtocolError):
                assert_may_train([bad])

    def test_secim_kapisi(self):
        assert_may_select("val"); assert_may_select("val_comp"); assert_may_select("gaps_val")
        for bad in ("gaps_test", "train", "gaps_fit", "gaps_skipped", "val_half_a", "val_half_b"):
            with self.subTest(split=bad), self.assertRaises(ProtocolError):
                assert_may_select(bad)

    def test_kesif_yarilari_raporlanamaz(self):
        """Yarıda seçip diğer yarıda 'test' diye raporlamak protokolce engelli."""
        for half in ("val_half_a", "val_half_b"):
            with self.subTest(split=half):
                with self.assertRaises(ProtocolError):
                    assert_may_report(half)
                with self.assertRaises(ProtocolError):
                    assert_may_select(half)
                self.assertFalse(is_independent_test(half))

    def test_rapor_kapisi(self):
        for ok in ("val", "val_comp", "gaps_val", "gaps_test"):
            assert_may_report(ok)
        for bad in ("train", "gaps_fit", "gaps_skipped"):
            with self.subTest(split=bad), self.assertRaises(ProtocolError):
                assert_may_report(bad)


class TestLeakageChecks(unittest.TestCase):
    def test_kayit_kesisimi(self):
        self.assertEqual(split_intersection(["a", "b"], ["c"]), set())
        self.assertEqual(split_intersection(["a", "b"], ["b", "c"]), {"b"})

    def test_icraci_ortusmesi(self):
        """Kayıt kimlikleri farklı olsa bile aynı icracı iki split'te olmamalı."""
        train = {"t1": "05", "t2": "01"}
        test_ok = {"x1": "09", "x2": "10"}
        test_bad = {"x1": "09", "x2": "05"}
        self.assertEqual(performer_overlap(train, test_ok), set())
        self.assertEqual(performer_overlap(train, test_bad), {"05"})

    def test_yasakli_gaps_kayitlari(self):
        """split=train + icracı test'te -> yasak. Atanmamış (NaN) kayıtlar hesaba girmez."""
        rows = [
            {"id": "001", "split": "train", "performer_name": "A"},
            {"id": "002", "split": "train", "performer_name": "B"},
            {"id": "003", "split": "test", "performer_name": "B"},
            {"id": "004", "split": None, "performer_name": "B"},
            {"id": "005", "split": "train", "performer_name": None},
        ]
        self.assertEqual(forbidden_gaps_ids(rows), ["002"])

    def test_yasakli_bos_liste(self):
        rows = [{"id": "001", "split": "train", "performer_name": "A"},
                {"id": "002", "split": "test", "performer_name": "Z"}]
        self.assertEqual(forbidden_gaps_ids(rows), [])

    def test_yazim_varyanti_yakalanir(self):
        """GAPS'te olculen gercek varyantlar: aksan ve buyuk/kucuk harf farki ayni kisidir."""
        rows = [
            {"id": "001", "split": "test", "performer_name": "Ana Vidović"},
            {"id": "002", "split": "train", "performer_name": "Ana Vidovic"},
            {"id": "003", "split": "test", "performer_name": 'Tengyue "TY" Zhang'},
            {"id": "004", "split": "train", "performer_name": 'Tengyue "Ty" Zhang'},
            {"id": "005", "split": "train", "performer_name": "Zhang Wei"},
        ]
        self.assertEqual(forbidden_gaps_ids(rows), ["002", "004"])

    def test_normalize_performer(self):
        self.assertEqual(normalize_performer("Ana Vidović"), normalize_performer("ana vidovic"))
        self.assertEqual(normalize_performer("Yoo Sik Ro"), normalize_performer("Ro, Yoo Sik"))
        self.assertEqual(normalize_performer(None), "")
        self.assertEqual(normalize_performer(float("nan")), "")
        self.assertNotEqual(normalize_performer("Han Eun"), normalize_performer("Xavier Jara"))

    def test_adsiz_kayitlar(self):
        rows = [{"id": "001", "split": "train", "performer_name": None},
                {"id": "002", "split": "train", "performer_name": "A"},
                {"id": "003", "split": "test", "performer_name": None}]
        self.assertEqual(unnamed_gaps_ids(rows), ["001"])

    def test_kisisel_kanal_tamamen_dislanmali(self):
        """Kanal train+test'te ve tek icracili ise o kanalin train kayitlari egitimde kalamaz."""
        rows = [{"id": "001", "split": "test", "performer_name": "A", "channel_name": "A kanali"},
                {"id": "002", "split": "train", "performer_name": "A", "channel_name": "A kanali"}]
        self.assertEqual(channel_leakage_suspects(rows), [])          # ad eslesiyor -> dislanir
        rows[1]["performer_name"] = None                              # ad bos -> dislanmaz
        self.assertEqual(channel_leakage_suspects(rows), [("A kanali", ["002"])])

    def test_platform_kanali_ihlal_sayilmaz(self):
        """Cok icracili platform kanali (Guitar Salon / Siccas): ortusme beklenir, ihlal degil."""
        rows = [{"id": "001", "split": "test", "performer_name": "A", "channel_name": "Platform"},
                {"id": "002", "split": "train", "performer_name": "B", "channel_name": "Platform"},
                {"id": "003", "split": "train", "performer_name": "C", "channel_name": "Platform"}]
        self.assertEqual(channel_leakage_suspects(rows), [])

    def test_kanal_tek_splitte_ise_atlanir(self):
        rows = [{"id": "001", "split": "train", "performer_name": None, "channel_name": "X"},
                {"id": "002", "split": "train", "performer_name": "B", "channel_name": "X"}]
        self.assertEqual(channel_leakage_suspects(rows), [])


class TestQuarantine(unittest.TestCase):
    """Icraci kimligi dogrulanamayan kayitlar egitim/secim listelerine giremez."""

    def test_liste_bos_degil_ve_bilinen_kayit(self):
        self.assertIn("395_w41wc", QUARANTINE_GAPS_IDS)

    def test_is_quarantined_yol_ve_kimlik(self):
        self.assertTrue(is_quarantined("395_w41wc"))
        self.assertTrue(is_quarantined("395_w41wc.npz"))
        self.assertTrue(is_quarantined(os.path.join("data", "cache", "gaps_train", "395_w41wc.npz")))
        self.assertFalse(is_quarantined("394_tk1wc.npz"))

    def test_drop_quarantined_sira_korunur(self):
        files = [f"data/cache/gaps_train/{i}.npz" for i in ("001_a", "395_w41wc", "002_b")]
        kept = drop_quarantined(files)
        self.assertEqual([os.path.basename(f) for f in kept], ["001_a.npz", "002_b.npz"])
        self.assertEqual(drop_quarantined([]), [])

    def test_gaps_training_files_kaynak_split_dogrular(self):
        """gaps_test'ten egitim/dogrulama bolunemez (veri okunmadan reddedilir)."""
        for bad in ("gaps_test", "gaps_val", "val"):
            with self.subTest(split=bad), self.assertRaises(ProtocolError):
                gaps_training_files(bad)


class TestPlayer05Halves(unittest.TestCase):
    IDS = [f"05_{style}{prog}-{100 + k}-C_solo" for style in ("BN", "Funk", "Jazz", "Rock", "SS")
           for prog in (1, 2, 3) for k in (0, 1)]

    def test_parca_ayrik_ve_tam(self):
        a, b = player05_halves(self.IDS)
        piece = lambda ids: {i.split("_")[1].split("-")[0] for i in ids}
        self.assertEqual(piece(a) & piece(b), set(), "ayni parca iki yarida olamaz")
        self.assertEqual(sorted(a + b), sorted(self.IDS))
        self.assertTrue(a and b)

    def test_deterministik(self):
        self.assertEqual(player05_halves(self.IDS), player05_halves(list(reversed(self.IDS))))

    def test_her_stil_iki_yarida(self):
        a, b = player05_halves(self.IDS)
        style = lambda ids: {i.split("_")[1].split("-")[0].rstrip("0123456789") for i in ids}
        self.assertEqual(style(a), style(b), "her stil iki yarida da temsil edilmeli")


if __name__ == "__main__":
    unittest.main()
