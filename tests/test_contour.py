"""
Katman 4 Faz 2A — Kontur altyapısı testleri (sentetik veri; GuitarSet/GAPS gerekmez).

Çalıştırma (proje kökünden): python -m unittest discover -s tests -t .
"""

import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from gtab.core.instrument import STANDARD_6
from gtab.data.contour import (EMPTY, F_JUMP, F_RANGE, F_SEGSTART, F_UNVOICED, F_ZERO, INVALID,
                               JUMP_CENTS, RELIABLE, STEP_SEC, SUSPECT, UNKNOWN, fallback_label,
                               flag_samples, hz_to_cents, jump_report, load_jams_contours,
                               note_contours, note_status, quality_stats, segment_ids,
                               string_range_cents)
from gtab.evaluation.technique_eval import NORMAL

G3_HZ = 196.0                      # 3. tel (indeks 3) açık: MIDI 55 = 5500 cent


def vibrato_hz(n: int, base: float = G3_HZ, depth_cents: float = 50.0, rate: float = 5.0):
    """Sentetik vibrato: base etrafında ±depth_cents, `rate` Hz, 5.805 ms adım."""
    t = np.arange(n) * STEP_SEC
    cents = 1200 * np.log2(base / 440.0) + 6900 + depth_cents * np.sin(2 * np.pi * rate * t)
    return t, 440.0 * 2 ** ((cents / 100.0 - 69.0) / 12.0)


def jams_doc(strings: dict) -> dict:
    """strings: {tel: (zamanlar, hz, index, notalar)} -> minimal JAMS sözlüğü."""
    anns = []
    for s, (t, hz, ix, notes) in strings.items():
        anns.append({"namespace": "pitch_contour", "annotation_metadata": {"data_source": str(s)},
                     "data": {"time": list(map(float, t)), "duration": [0.0] * len(t),
                              "value": [{"voiced": True, "index": int(i), "frequency": float(f)}
                                        for i, f in zip(ix, hz)],
                              "confidence": [None] * len(t)}})
        anns.append({"namespace": "note_midi", "annotation_metadata": {"data_source": str(s)},
                     "data": [{"time": float(a), "duration": float(b), "value": float(c),
                               "confidence": None} for a, b, c in notes]})
    return {"annotations": anns, "file_metadata": {"title": "sentetik"}, "sandbox": {}}


def write_jams(doc: dict) -> Path:
    fh = tempfile.NamedTemporaryFile("w", suffix=".jams", delete=False, encoding="utf-8")
    json.dump(doc, fh)
    fh.close()
    return Path(fh.name)


class TestHzToCents(unittest.TestCase):
    def test_sabit_frekans(self):
        self.assertAlmostEqual(float(hz_to_cents(np.array([440.0]))[0]), 6900.0, places=6)
        self.assertAlmostEqual(float(hz_to_cents(np.array([220.0]))[0]), 5700.0, places=6)
        self.assertAlmostEqual(float(hz_to_cents(np.array([G3_HZ]))[0]), 5500.02, places=2)

    def test_sabit_dizi_sabit_cent(self):
        c = hz_to_cents(np.full(50, G3_HZ))
        self.assertEqual(len(np.unique(np.round(c, 6))), 1)

    def test_gecersiz_girdiler_nan(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")                 # log(0) uyarisi bile olmamali
            c = hz_to_cents(np.array([0.0, -5.0, np.nan, np.inf, 440.0]))
        self.assertTrue(np.isnan(c[:4]).all())
        self.assertAlmostEqual(float(c[4]), 6900.0)

    def test_tel_araligi(self):
        lo, hi = string_range_cents(0, STANDARD_6)
        self.assertAlmostEqual(lo, (40 - 0.5) * 100)
        self.assertAlmostEqual(hi, (40 + 24 + 0.5) * 100)


class TestSegments(unittest.TestCase):
    def test_zaman_boslugu_yeni_segment(self):
        t = np.array([0.0, 0.006, 0.012, 1.0, 1.006])
        seg = segment_ids(t, np.zeros(5, int))
        self.assertEqual(seg.tolist(), [0, 0, 0, 1, 1])

    def test_index_degisimi_yeni_segment(self):
        t = np.arange(5) * STEP_SEC
        seg = segment_ids(t, np.array([0, 0, 1, 1, 1]))
        self.assertEqual(seg.tolist(), [0, 0, 1, 1, 1])

    def test_bos_girdi(self):
        self.assertEqual(len(segment_ids(np.zeros(0), np.zeros(0, int))), 0)


class TestFlags(unittest.TestCase):
    def _flags(self, hz, string=3, index=None, voiced=None):
        n = len(hz)
        t = np.arange(n) * STEP_SEC
        index = np.zeros(n, int) if index is None else index
        voiced = np.ones(n, bool) if voiced is None else voiced
        seg = segment_ids(t, index)
        return flag_samples(hz_to_cents(hz), hz, voiced, string, seg, STANDARD_6)

    def test_sifir_ve_nan_isaretlenir(self):
        f = self._flags(np.array([G3_HZ, 0.0, np.nan, G3_HZ]))
        self.assertTrue(f[1] & F_ZERO)
        self.assertTrue(f[2] & 2)                           # F_NONFINITE
        self.assertFalse(f[0] & F_ZERO)

    def test_tel_araligi_disi(self):
        """5. telde (acik E4) 61.7 Hz -> fiziksel olarak imkansiz (gercek veride goruldu)."""
        f = self._flags(np.array([329.6, 61.7, 329.6]), string=5)
        self.assertTrue(f[1] & F_RANGE)
        self.assertFalse(f[0] & F_RANGE)

    def test_unvoiced_isaretlenir(self):
        f = self._flags(np.array([G3_HZ, G3_HZ]), voiced=np.array([True, False]))
        self.assertTrue(f[1] & F_UNVOICED)

    def test_oktav_sicramasi_tespit(self):
        hz = np.full(20, G3_HZ)
        hz[10] = G3_HZ * 2                                  # bir oktav yukari
        f = self._flags(hz)
        self.assertTrue(f[10] & F_JUMP, "oktav sicramasi isaretlenmedi")
        self.assertTrue(f[11] & F_JUMP, "donus de sicrama")
        self.assertEqual(int((f & F_JUMP).astype(bool).sum()), 2)

    def test_segment_sinirindaki_fark_sicrama_sayilmaz(self):
        """Takipci kesintisinden sonraki buyuk fark gercek hareket degil -> JUMP degil."""
        hz = np.r_[np.full(5, G3_HZ), np.full(5, G3_HZ * 2)]
        f = self._flags(hz, index=np.r_[np.zeros(5, int), np.ones(5, int)])
        self.assertFalse((f & F_JUMP).any())
        self.assertTrue(f[5] & F_SEGSTART)

    def test_gercek_vibrato_sicrama_sayilmaz(self):
        _, hz = vibrato_hz(200, depth_cents=80.0, rate=6.0)
        f = self._flags(hz)
        self.assertFalse((f & F_JUMP).any(), "gercek vibrato sicrama olarak isaretlendi")

    def test_sicrama_esigi_fiziksel_siniri_asiyor(self):
        """5.8 ms'de 600 cent = ~103 yarim ses/sn; gercek slide bunun cok altinda."""
        self.assertGreater(JUMP_CENTS / (STEP_SEC * 1000), 100)


class TestNoteStatus(unittest.TestCase):
    def test_durum_siralamasi(self):
        self.assertEqual(note_status(0, 0, 1.0, 0.0), EMPTY)
        self.assertEqual(note_status(5, 0, 1.0, 0.0), INVALID)
        self.assertEqual(note_status(50, 1, 1.0, 0.0), SUSPECT)
        self.assertEqual(note_status(50, 0, 0.3, 0.0), SUSPECT)
        self.assertEqual(note_status(50, 0, 1.0, 900.0), SUSPECT)
        self.assertEqual(note_status(50, 0, 1.0, 100.0), RELIABLE)

    def test_unknown_normal_olmaz(self):
        for st in (SUSPECT, INVALID, EMPTY):
            with self.subTest(status=st):
                self.assertEqual(fallback_label(st), UNKNOWN)
                self.assertNotEqual(fallback_label(st), NORMAL)
        self.assertIsNone(fallback_label(RELIABLE))

    def test_unknown_ile_normal_ayri_etiket(self):
        self.assertNotEqual(UNKNOWN, NORMAL)


class TestLoadAndAlign(unittest.TestCase):
    def setUp(self):
        n = 200
        t3, hz3 = vibrato_hz(n)                             # 3. tel: temiz vibrato
        hz3 = hz3.copy()
        t5 = t3 + 0.0                                       # 5. tel: ayni zaman, farkli perde
        hz5 = np.full(n, 329.6)
        hz5[50] = 0.0                                       # gecersiz ornek
        hz5[120] = 61.7                                     # tel araligi disi
        notes3 = [(0.0, n * STEP_SEC, 55.0)]
        notes5 = [(0.0, n * STEP_SEC, 64.0)]
        self.path = write_jams(jams_doc({3: (t3, hz3, np.zeros(n, int), notes3),
                                         5: (t5, hz5, np.zeros(n, int), notes5)}))
        self.contours, self.notes = load_jams_contours(self.path)

    def tearDown(self):
        self.path.unlink(missing_ok=True)

    def test_tel_kimligi_korunur(self):
        self.assertEqual(sorted(self.contours), [3, 5])
        self.assertEqual(self.contours[3].string, 3)
        self.assertEqual(self.contours[5].string, 5)

    def test_nota_telden_sadece_kendi_ornegini_alir(self):
        nc3 = note_contours(self.contours[3], self.notes[3], 3)[0]
        nc5 = note_contours(self.contours[5], self.notes[5], 5)[0]
        self.assertEqual(nc3.string, 3)
        self.assertEqual(nc5.string, 5)
        self.assertEqual(nc3.n_samples, 200)
        self.assertEqual(nc5.n_valid, 198)                  # 0 Hz + aralik disi dusuldu
        self.assertAlmostEqual(nc3.midi, 55.0)
        self.assertAlmostEqual(nc5.midi, 64.0)

    def test_gercek_vibrato_korunuyor(self):
        """Yukleyici hicbir ornegi duzeltmez: salinim genligi girdiyle ayni kalir."""
        c = self.contours[3].cents
        self.assertTrue(np.isfinite(c).all())
        p2p = float(np.percentile(c, 95) - np.percentile(c, 5))
        self.assertGreater(p2p, 80.0)                       # ±50 cent -> ~95 cent tepe-tepe
        self.assertEqual(note_contours(self.contours[3], self.notes[3], 3)[0].status, RELIABLE)

    def test_gecersiz_ornek_nan_ve_bayrakli(self):
        c = self.contours[5]
        self.assertTrue(np.isnan(c.cents[50]) and bool(c.flags[50] & F_ZERO))
        self.assertTrue(np.isnan(c.cents[120]) and bool(c.flags[120] & F_RANGE))
        self.assertEqual(int(c.valid.sum()), 198)

    def test_determinizm(self):
        c2, n2 = load_jams_contours(self.path)
        for s in self.contours:
            np.testing.assert_array_equal(self.contours[s].flags, c2[s].flags)
            np.testing.assert_array_equal(self.contours[s].segment, c2[s].segment)
            np.testing.assert_array_equal(np.nan_to_num(self.contours[s].cents, nan=-1.0),
                                          np.nan_to_num(c2[s].cents, nan=-1.0))
            np.testing.assert_array_equal(self.notes[s], n2[s])

    def test_kalite_istatistigi(self):
        st = quality_stats(self.contours, self.notes)
        self.assertEqual(st["ornek"], 400)
        self.assertEqual(st["sifir"], 1)
        self.assertEqual(st["aralik_disi"], 1)
        self.assertEqual(st["nota"], 2)
        self.assertEqual(st[RELIABLE] + st[SUSPECT] + st[INVALID] + st[EMPTY], 2)

    def test_jump_report_gecersizleri_saymaz(self):
        jr = jump_report(self.contours)
        self.assertEqual(jr[">=600c"], 0)
        self.assertGreater(jr["n"], 0)


if __name__ == "__main__":
    unittest.main()


class TestCoverageDefinition(unittest.TestCase):
    """Kapsama orani GECERLI ornekler uzerinden olculmeli; gecersizler oranini sismemeli."""

    def _note_with_invalid_half(self):
        n = 200
        t, hz = vibrato_hz(n, depth_cents=10.0)
        hz = hz.copy()
        hz[n // 2:] = 0.0                               # notanin ikinci yarisi gecersiz (0 Hz)
        path = write_jams(jams_doc({3: (t, hz, np.zeros(n, int), [(0.0, n * STEP_SEC, 55.0)])}))
        try:
            contours, notes = load_jams_contours(path)
            return note_contours(contours[3], notes[3], 3)[0]
        finally:
            path.unlink(missing_ok=True)

    def test_gecersiz_ornekler_kapsamayi_sismiyor(self):
        nc = self._note_with_invalid_half()
        self.assertEqual(nc.n_samples, 200)
        self.assertEqual(nc.n_valid, 100)
        self.assertAlmostEqual(nc.coverage, 0.5, places=2)

    def test_yarisi_gecersiz_nota_supheli(self):
        nc = self._note_with_invalid_half()
        self.assertEqual(nc.status, SUSPECT)
        self.assertEqual(fallback_label(nc.status), UNKNOWN)


class TestStringWithoutContour(unittest.TestCase):
    """note_midi var ama pitch_contour yok: notalar sessizce atlanmamali."""

    def setUp(self):
        n = 120
        t, hz = vibrato_hz(n, depth_cents=10.0)
        doc = jams_doc({3: (t, hz, np.zeros(n, int), [(0.0, n * STEP_SEC, 55.0)])})
        # 5. tel: yalniz note_midi anotasyonu (pitch_contour hic yok)
        doc["annotations"].append({"namespace": "note_midi",
                                   "annotation_metadata": {"data_source": "5"},
                                   "data": [{"time": 0.0, "duration": 0.5, "value": 64.0,
                                             "confidence": None}]})
        self.path = write_jams(doc)
        self.contours, self.notes = load_jams_contours(self.path)

    def tearDown(self):
        self.path.unlink(missing_ok=True)

    def test_konturu_olmayan_telin_notasi_sayilir(self):
        st = quality_stats(self.contours, self.notes)
        self.assertEqual(st["nota"], 2, "konturu olmayan telin notasi sayima girmedi")

    def test_konturu_olmayan_nota_empty(self):
        st = quality_stats(self.contours, self.notes)
        self.assertEqual(st[EMPTY], 1)
        self.assertEqual(st[RELIABLE], 1)

    def test_konturu_olmayan_nota_unknown_olur(self):
        self.assertEqual(fallback_label(EMPTY), UNKNOWN)
        self.assertNotEqual(fallback_label(EMPTY), NORMAL)
