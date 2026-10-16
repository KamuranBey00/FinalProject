"""
README17 Ek A hex auxiliary target tests: per-string spectra, dataset plumbing, checkpoint compatibility.

No dataset needed (synthetic 6-channel WAV + tiny temporary cache).
Run (project root): python -m unittest tests.test_hex
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from gtab.config import SAMPLE_RATE
from gtab.data.hex import POOL, hex_string_spectra, string_energy_agreement
from gtab.data.torch_dataset import GuitarSetSeq


def _six_channel_wav(path: Path, seconds: float = 1.0, loud: int = 1) -> None:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    ch = np.zeros((len(t), 6), np.float32)
    for s, f in enumerate((82.4, 110.0, 146.8, 196.0, 246.9, 329.6)):
        ch[:, s] = (0.5 if s == loud else 0.01) * np.sin(2 * np.pi * f * t)
    sf.write(str(path), ch, SAMPLE_RATE)


class TestHexSpectra(unittest.TestCase):

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_shape_range_and_padding(self):
        p = self.dir / "a.wav"; _six_channel_wav(p)
        spec = hex_string_spectra(p, 60)                       # 1 s ~ 44 frames -> zero-padded to 60
        self.assertEqual(spec.shape, (60, 6, 192 // POOL))
        self.assertTrue(float(spec.min()) >= 0.0 and float(spec.max()) <= 1.0)
        self.assertEqual(float(spec[55:].max()), 0.0)

    def test_common_reference_keeps_string_levels(self):
        # a per-channel reference would make all six channels look equally loud
        p = self.dir / "b.wav"; _six_channel_wav(p, loud=1)
        e = hex_string_spectra(p, 40).astype(np.float32)[5:35].max(2).mean(0)   # peak bin per channel (pure tones)
        self.assertEqual(int(e.argmax()), 1)
        self.assertGreater(e[1] - np.delete(e, 1).max(), 0.3)       # 0.5 vs 0.01 amplitude = 34 dB = 0.42

    def test_agreement_metric(self):
        spec = np.zeros((4, 6, 3), np.float32)
        spec[:, 2] = 1.0
        tab = np.zeros((4, 6), np.int64); tab[:, 2] = 5          # labelled string 2 is loudest
        self.assertEqual(string_energy_agreement(spec, tab), 1.0)
        tab2 = np.zeros((4, 6), np.int64); tab2[:, 0] = 3
        self.assertEqual(string_energy_agreement(spec, tab2), 0.0)

    def test_rejects_non_hex_audio(self):
        p = self.dir / "mono.wav"
        sf.write(str(p), np.zeros(SAMPLE_RATE, np.float32), SAMPLE_RATE)
        with self.assertRaises(ValueError):
            hex_string_spectra(p, 10)


class TestDatasetPlumbing(unittest.TestCase):

    def setUp(self):
        self.cache = Path(tempfile.mkdtemp())
        (self.cache / "train").mkdir()
        T = 250
        tab = np.zeros((T, 6), np.int64); tab[10:60, 1] = 3
        np.savez(self.cache / "train" / "00_x_solo.npz", cqt=np.full((T, 192), -40, np.float32),
                 tab=tab, onset=np.zeros((T, 49), np.uint8), frame=np.zeros((T, 49), np.uint8))
        self.hex_dir = self.cache / "hex"; self.hex_dir.mkdir()
        np.savez(self.hex_dir / "00_x_solo.npz", spec=np.full((T, 6, 96), 0.5, np.float16))

    def test_without_hex_dir_items_unchanged(self):
        ds = GuitarSetSeq(str(self.cache), "train", 200, onsets=True)
        self.assertEqual(len(ds[0]), 6)                        # old 6-tuple

    def test_with_hex_dir_adds_aligned_targets(self):
        ds = GuitarSetSeq(str(self.cache), "train", 200, onsets=True, hex_dir=str(self.hex_dir))
        self.assertEqual(len(ds), 2)
        item = ds[1]                                            # second chunk: 50 real frames + padding
        self.assertEqual(len(item), 8)
        hx, has = item[6], item[7]
        self.assertTrue(has)
        self.assertEqual(tuple(hx.shape), (200, 6, 96))
        self.assertAlmostEqual(float(hx[:50].mean()), 0.5, places=3)
        self.assertEqual(float(hx[50:].abs().sum()), 0.0)

    def test_missing_hex_file_is_masked(self):
        (self.hex_dir / "00_x_solo.npz").unlink()
        ds = GuitarSetSeq(str(self.cache), "train", 200, onsets=True, hex_dir=str(self.hex_dir))
        self.assertFalse(ds[0][7])


class TestCheckpointCompatibility(unittest.TestCase):

    def test_forward_full_default_unchanged(self):
        import torch
        from gtab.models.nets import TabCRNNOnset
        torch.manual_seed(0)
        m = TabCRNNOnset(harmonics=(0.5, 1, 2), offset=True, deep=1, pitch_onset=True).eval()
        x = torch.rand(1, 1, 20, 192)
        with torch.no_grad():
            a = m.forward_full(x)
            b = m.forward_full(x, return_features=True)
        self.assertEqual(len(a), 4)
        self.assertEqual(len(b), 5)
        for u, v in zip(a, b[:4]):
            self.assertTrue(torch.equal(u, v))
        self.assertEqual(b[4].shape[-1], m.head.in_features)
        # no hex head inside the model: state_dict keys are unchanged
        self.assertFalse(any(k.startswith("hex") for k in m.state_dict()))


if __name__ == "__main__":
    unittest.main()
