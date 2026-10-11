"""
P2 — Sol el parmaklandırma (Katman 3.13 "klasik edisyon önerisi") testleri.

Amaç: MEVCUT davranışı sabitlemek. Hiçbir fiziksel kısıt eklenmiyor, model değiştirilmiyor;
testler bugünkü çıktıyı olduğu gibi doğruluyor (regresyon ağı).

İki grup:
  1. Sentetik testler — veri seti, checkpoint ve GPU gerektirmez; her makinede çalışır.
  2. Kapılı testler (`skipUnless`) — `checkpoints/edition_seq.pt` ve/veya `data/cache/gaps_tab/`
     varsa çalışır: checkpoint yükleme, tahmin üretme ve **gaps_val** metriklerinin yeniden
     üretimi. `gaps_test` hiç okunmaz (Faz 0 protokolü: test yalnız nihai raporlama).

Çalıştırma (proje kökünden): python -m unittest discover -s tests -t .
"""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gtab.core.instrument import STANDARD_6
from gtab.decoding.edition import (CHORD_TOL, ChordPositionModel, FingerModel, decode,
                                   finger_baseline, label_sequence, order_notes)
from gtab.paths import CACHE_DIR, ckpt_path

CKPT = Path(ckpt_path("edition_seq.pt"))
GAPS_TAB = Path(CACHE_DIR) / "gaps_tab"
HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_CKPT = HAVE_TORCH and CKPT.exists()
HAVE_GAPS_TAB = GAPS_TAB.exists() and any(GAPS_TAB.glob("*.npz"))

# README14 Adım 2b, gaps_val kolonu (geliştirme/doğrulama split'i)
REF_GAPS_VAL = {"tel": 0.766, "parmak_oracle_konum": 0.675, "urun_tam": 0.382}
REF_TOL = 0.005

E2, G4, B4, E4 = 40, 67, 71, 64          # MIDI


def net_on_cpu(trained: bool):
    """EditionNet (CPU). trained=True ise mevcut checkpoint yüklenir."""
    import torch
    from gtab.models.edition_net import EditionNet
    net = EditionNet()
    if trained:
        net.load_state_dict(torch.load(CKPT, map_location="cpu")["model"])
    return net


def predict_cpu(net, notes, chord, fixed=None):
    from gtab.models.edition_net import predict
    return predict(net, np.asarray(notes, float), np.asarray(chord, bool), "cpu", fixed=fixed)


class TestOrderNotes(unittest.TestCase):
    """Akor gruplama: 30 ms toleransı ve akor içinde perde artan sıra."""

    def test_akor_toleransi(self):
        notes = [(0.0, 0.5, 60), (0.01, 0.5, 64), (0.5, 0.5, 67)]
        o, chord = order_notes(notes)
        self.assertEqual(chord.tolist(), [False, True, False])

    def test_tolerans_disi_ayri_akor(self):
        notes = [(0.0, 0.5, 60), (CHORD_TOL + 0.01, 0.5, 64)]
        _, chord = order_notes(notes)
        self.assertEqual(chord.tolist(), [False, False])

    def test_akor_icinde_perde_artan(self):
        notes = [(0.0, 0.5, 67), (0.0, 0.5, 60), (0.0, 0.5, 64)]
        o, _ = order_notes(notes)
        self.assertEqual([int(np.asarray(notes, float)[i, 2]) for i in o], [60, 64, 67])


class TestFingerCandidates(unittest.TestCase):
    """Parmak adayları: boş tel -> yalnız 0, basılı -> yalnız 1-4."""

    def test_cands(self):
        self.assertEqual(FingerModel.cands(0), [0])
        self.assertEqual(FingerModel.cands(5), [1, 2, 3, 4])

    def test_finger_baseline_bos_tel_sifir(self):
        notes = [(0.0, 0.5, E2)]
        self.assertEqual(finger_baseline(notes, [(0, 0)]), [0])

    def test_finger_baseline_pozisyondan(self):
        """parmak = perde - pozisyon + 1; pozisyon = pencereki en düşük basılı perde."""
        notes = [(0.0, 0.3, 60), (0.3, 0.3, 62), (0.6, 0.3, 64)]
        out = finger_baseline(notes, [(2, 5), (2, 7), (2, 8)])
        self.assertEqual(out, [1, 3, 4])

    def test_finger_baseline_1_4_arasinda_kirpilir(self):
        notes = [(0.0, 0.3, 60), (0.3, 0.3, 72)]
        out = finger_baseline(notes, [(2, 1), (2, 20)])
        self.assertTrue(all(1 <= g <= 4 for g in out))


class TestLabelSequence(unittest.TestCase):
    """build_gaps_tab etiket dosyasının okunması (sentetik npz)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "sentetik.npz"
        np.savez(self.tmp,
                 notes=np.array([[0.5, 1.0, 64], [0.0, 0.4, 60], [0.5, 1.0, 67]], np.float32),
                 string=np.array([1, 2, -1], np.int8),
                 fret=np.array([9, 5, -1], np.int8),
                 finger_note=np.array([3, 1, -1], np.int8))

    def tearDown(self):
        self.tmp.unlink(missing_ok=True)

    def test_sira_ve_etiketler(self):
        notes, chord, s, f, g = label_sequence(self.tmp)
        self.assertEqual([int(p) for p in notes[:, 2]], [60, 64, 67])     # zamana göre sıralı
        self.assertEqual(s.tolist(), [2, 1, -1])
        self.assertEqual(f.tolist(), [5, 9, -1])
        self.assertEqual(g.tolist(), [1, 3, -1])
        self.assertEqual(chord.tolist(), [False, False, True])            # 64 ve 67 aynı akorda


class TestLogLinearDecode(unittest.TestCase):
    """2a log-lineer model (reddedildi ama kod duruyor): çıktı aralıkları ve `fixed` davranışı."""

    def setUp(self):
        self.pos, self.fing = ChordPositionModel(), FingerModel()       # sıfır ağırlık
        self.notes = [(0.0, 0.4, E2), (0.5, 0.4, G4), (0.5, 0.4, B4)]
        _, self.chord = order_notes(self.notes)

    def test_cikti_gecerli(self):
        out = decode(self.notes, self.chord, self.pos, self.fing)
        self.assertEqual(len(out), 3)
        for (s, f, g), (_, _, p) in zip(out, self.notes):
            self.assertIn(s, range(6))
            self.assertEqual(f, p - STANDARD_6.tuning[s])
            self.assertTrue(0 <= f <= STANDARD_6.num_frets)
            self.assertEqual(g == 0, f == 0)                            # boş tel <-> parmak 0

    def test_fixed_konum_korunur(self):
        out = decode(self.notes, self.chord, self.pos, self.fing, fixed=[(0, 0), None, None])
        self.assertEqual(out[0][:2], (0, 0))


@unittest.skipUnless(HAVE_CKPT, "torch veya checkpoints/edition_seq.pt yok")
class TestEditionNetCheckpoint(unittest.TestCase):
    """Kabul edilen dizi modeli: checkpoint yüklenir, tahmin üretir, sert kısıtlar tutar."""

    @classmethod
    def setUpClass(cls):
        cls.net = net_on_cpu(trained=True)

    def test_checkpoint_yuklenir_ve_tahmin_uretir(self):
        notes = [(0.0, 0.5, 60), (0.6, 0.5, 64), (1.2, 0.5, 67)]
        _, chord = order_notes(notes)
        s, fret, g = predict_cpu(self.net, notes, chord)
        self.assertEqual((len(s), len(fret), len(g)), (3, 3, 3))

    def test_bos_tel_parmak_sifir(self):
        """E2 yalnız açık 6. telde çalınabilir -> konum (0,0), parmak 0 olmak zorunda."""
        notes = [(0.0, 0.8, E2)]
        s, fret, g = predict_cpu(self.net, notes, np.array([False]))
        self.assertEqual((int(s[0]), int(fret[0])), (0, 0))
        self.assertEqual(int(g[0]), 0)

    def test_basili_notada_parmak_1_4(self):
        notes = [(0.0, 0.5, 60), (0.6, 0.5, 65)]
        _, chord = order_notes(notes)
        _, fret, g = predict_cpu(self.net, notes, chord)
        for f, gg in zip(fret, g):
            if f > 0:
                self.assertIn(int(gg), (1, 2, 3, 4))

    def test_akor_icinde_tel_tekligi(self):
        """Aynı akordaki notalara aynı tel verilemez (resolve_chords)."""
        notes = [(0.0, 1.0, E4), (0.0, 1.0, G4), (0.0, 1.0, B4)]
        o, chord = order_notes(notes)
        s, _, _ = predict_cpu(self.net, np.asarray(notes, float)[o], chord)
        self.assertEqual(len(set(s.tolist())), 3, f"tel tekrarı: {s.tolist()}")

    def test_konum_adaylari_gecerli(self):
        """Her nota için fret = perde - açık tel ve 0 <= fret <= 24."""
        rng = np.random.default_rng(0)
        pitches = rng.integers(40, 88, size=24)
        notes = [(0.4 * k, 0.3, int(p)) for k, p in enumerate(pitches)]
        _, chord = order_notes(notes)
        s, fret, g = predict_cpu(self.net, notes, chord)
        for si, fi, gi, (_, _, p) in zip(s, fret, g, notes):
            self.assertIn(int(si), range(6))
            self.assertEqual(int(fi), p - STANDARD_6.tuning[int(si)])
            self.assertTrue(0 <= int(fi) <= STANDARD_6.num_frets)
            self.assertEqual(int(gi) == 0, int(fi) == 0)

    def test_determinizm(self):
        notes = [(0.0, 0.5, 60), (0.6, 0.5, 64), (0.6, 0.5, 67), (1.4, 0.5, 71)]
        o, chord = order_notes(notes)
        nt = np.asarray(notes, float)[o]
        a = predict_cpu(self.net, nt, chord)
        b = predict_cpu(self.net, nt, chord)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_fixed_konum_korunur(self):
        notes = [(0.0, 0.5, 60), (0.6, 0.5, 64)]
        _, chord = order_notes(notes)
        s, fret, _ = predict_cpu(self.net, notes, chord, fixed=np.array([5, -1]))
        self.assertEqual(int(s[0]), 5)
        self.assertEqual(int(fret[0]), 60 - STANDARD_6.tuning[5])


@unittest.skipUnless(HAVE_CKPT and HAVE_GAPS_TAB, "torch, edition_seq.pt veya data/cache/gaps_tab yok")
class TestGapsValReproduction(unittest.TestCase):
    """
    README14 Adım 2b'nin **gaps_val** kolonunun yeniden üretimi (geliştirme split'i).
    `gaps_test` okunmaz. Metrik tanımları `scripts/eval/eval_edition.py`'den alınır.
    """

    @classmethod
    def setUpClass(cls):
        from gtab.evaluation.protocol import gaps_training_files
        cls.net = net_on_cpu(trained=True)
        _, val = gaps_training_files()
        cls.tracks = []
        for f in val:
            lf = GAPS_TAB / os.path.basename(f)
            if not lf.exists():
                continue
            notes, chord, s, fr, g = label_sequence(lf)
            cls.tracks.append(dict(notes=notes, chord=chord, gold=list(zip(s, fr)),
                                   finger=g, lab=s >= 0))
        cls.free = {id(t): cls._sq(cls, t) for t in cls.tracks}

    def _sq(self, t, fixed=None):
        s, f, g = predict_cpu(self.net, t["notes"], t["chord"], fixed=fixed)
        return [(int(a), int(b), int(c)) if a >= 0 else None for a, b, c in zip(s, f, g)]

    def test_kayit_ve_nota_sayisi(self):
        self.assertGreater(len(self.tracks), 0)
        print(f"\n    gaps_val: {len(self.tracks)} etiketli kayit, "
              f"{sum(len(t['notes']) for t in self.tracks)} nota")

    def test_tel_uyumu(self):
        from scripts.eval.eval_edition import pooled          # metrik tanımı birebir aynı kaynak
        acc = pooled(self.tracks, lambda t: self.free[id(t)])
        print(f"    tel uyumu (dizi modeli)      : {acc:.3f}  (README14 gaps_val {REF_GAPS_VAL['tel']})")
        self.assertAlmostEqual(acc, REF_GAPS_VAL["tel"], delta=REF_TOL)

    def test_parmak_oracle_konum_ve_urun(self):
        n = ok_f = ok_prod = 0
        for t in self.tracks:
            gold = [g if l else None for g, l in zip(t["gold"], t["lab"])]
            fixed = np.array([g[0] if g is not None else -1 for g in gold])
            sqf = self._sq(t, fixed=fixed)
            own = self.free[id(t)]
            for i, (g, gf) in enumerate(zip(gold, t["finger"])):
                if g is None or g[1] == 0 or gf < 1:
                    continue                                  # boş tel ve parmaksız notalar dışarıda
                n += 1
                ok_f += sqf[i] is not None and sqf[i][2] == gf
                ok_prod += own[i] is not None and tuple(own[i]) == (g[0], g[1], gf)
        a_f, a_p = ok_f / max(n, 1), ok_prod / max(n, 1)
        print(f"    parmak (partisyon konumu)    : {a_f:.3f}  (README14 gaps_val {REF_GAPS_VAL['parmak_oracle_konum']})")
        print(f"    urun (tel+perde+parmak)      : {a_p:.3f}  (README14 gaps_val {REF_GAPS_VAL['urun_tam']})")
        print(f"    parmakli basili nota sayisi  : {n}")
        self.assertAlmostEqual(a_f, REF_GAPS_VAL["parmak_oracle_konum"], delta=REF_TOL)
        self.assertAlmostEqual(a_p, REF_GAPS_VAL["urun_tam"], delta=REF_TOL)


if __name__ == "__main__":
    unittest.main()
