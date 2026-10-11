"""
Katman 4 Faz 0 — Çıkarım parmak izi karşılaştırma testleri (sentetik; model/veri gerekmez).

Çalıştırma (proje kökünden): python -m unittest discover -s tests -t .
"""

import copy
import unittest

import numpy as np

from gtab.evaluation.fingerprint import HIST_TOL, TOL, compare, stat, tab_stat, track_fingerprint


def make_ref() -> dict:
    return {"ckpt": "tabcrnn_deep2.pt", "split": "val", "device": "cpu", "onset_head": True,
            "deep": 2, "pitch_onset_head": True,
            "kayitlar": {"a": {"frames": 100, "tab": {"mean": 0.9, "std": 0.1},
                               "argmax_hist": [500, 100] + [0] * 24,
                               "onset": {"mean": 0.01, "std": 0.02},
                               "offset": None,
                               "pitch_onset": {"mean": 0.002, "std": 0.03}}}}


class TestStats(unittest.TestCase):
    def test_tab_stat_guven_kullanir(self):
        """Softmax satirlarinin ortalamasi 1/C sabit; parmak izi guveni (max) tasimali."""
        probs = np.full((4, 6, 26), 1 / 26, dtype=np.float32)
        probs[:, :, 3] = 0.5
        probs /= probs.sum(-1, keepdims=True)
        self.assertAlmostEqual(tab_stat(probs)["mean"], float(probs.max(-1).mean()), places=6)
        self.assertNotAlmostEqual(tab_stat(probs)["mean"], 1 / 26)

    def test_track_fingerprint_alanlari(self):
        probs = np.random.default_rng(0).random((10, 6, 26)).astype(np.float32)
        probs /= probs.sum(-1, keepdims=True)
        fp = track_fingerprint(probs, np.zeros((10, 6), np.float32), None, np.zeros((10, 49), np.float32))
        self.assertEqual(fp["frames"], 10)
        self.assertEqual(sum(fp["argmax_hist"]), 60)        # 10 kare x 6 tel
        self.assertIsNone(fp["offset"])
        self.assertEqual(stat(np.zeros(5))["mean"], 0.0)


class TestCompare(unittest.TestCase):
    def test_ayni_parmak_izi(self):
        self.assertEqual(compare(make_ref(), make_ref()), [])

    def test_kafa_ortalamasi_degisirse_yakalanir(self):
        cur = make_ref()
        cur["kayitlar"]["a"]["onset"]["mean"] += 10 * TOL
        bad = compare(make_ref(), cur)
        self.assertEqual(len(bad), 1)
        self.assertIn("a.onset.mean", bad[0])

    def test_tolerans_icindeki_oynama_gecer(self):
        cur = make_ref()
        cur["kayitlar"]["a"]["tab"]["mean"] += TOL / 2
        self.assertEqual(compare(make_ref(), cur), [])

    def test_kafa_eklenirse_yakalanir(self):
        """Katman 4 yeni kafa ekleyip eski yolu etkilerse: offset None -> deger."""
        cur = make_ref()
        cur["kayitlar"]["a"]["offset"] = {"mean": 0.5, "std": 0.1}
        bad = compare(make_ref(), cur)
        self.assertTrue(any("kafa varligi degisti" in b for b in bad))

    def test_kurulum_farki(self):
        cur = make_ref(); cur["deep"] = 0; cur["ckpt"] = "baska.pt"
        bad = compare(make_ref(), cur)
        self.assertEqual(len(bad), 2)
        self.assertTrue(all(b.startswith("kurulum farki") for b in bad))

    def test_kayit_listesi_degisirse_erken_dur(self):
        cur = make_ref(); cur["kayitlar"]["b"] = copy.deepcopy(cur["kayitlar"]["a"])
        bad = compare(make_ref(), cur)
        self.assertEqual(len(bad), 1)
        self.assertIn("kayit listesi degisti", bad[0])

    def test_kare_sayisi_degisirse(self):
        cur = make_ref(); cur["kayitlar"]["a"]["frames"] = 99
        bad = compare(make_ref(), cur)
        self.assertEqual(len(bad), 1)
        self.assertIn("kare sayisi", bad[0])

    def test_histogram_kaymasi(self):
        ref = make_ref(); cur = copy.deepcopy(ref)
        n = int(HIST_TOL * ref["kayitlar"]["a"]["frames"] * 6) + 2   # tolerans disi
        cur["kayitlar"]["a"]["argmax_hist"][0] -= n
        cur["kayitlar"]["a"]["argmax_hist"][1] += n
        bad = compare(ref, cur)
        self.assertTrue(any("argmax_hist" in b for b in bad))

    def test_histogram_tolerans_icinde(self):
        ref = make_ref(); cur = copy.deepcopy(ref)
        cur["kayitlar"]["a"]["argmax_hist"][0] -= 0            # degisiklik yok
        self.assertEqual(compare(ref, cur), [])


if __name__ == "__main__":
    unittest.main()
