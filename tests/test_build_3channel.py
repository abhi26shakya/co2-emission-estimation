"""
Unit tests for build_3channel.py's triple_folder(): the function that
pairs NO2/SO2/VIIRS single-channel tiles into the 3-channel stacks Track A
actually trains on. Bugs here (a wrong join, a silent gap-fill, a missing
clip) would corrupt every downstream Track A result without any obvious
symptom -- worth locking down with direct tests.
"""
import os
import sys
import shutil
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_3channel as b3c


class TestTripleFolder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.no2_dir = os.path.join(self.tmp, "no2")
        self.so2_dir = os.path.join(self.tmp, "so2")
        self.viirs_dir = os.path.join(self.tmp, "viirs")
        self.out_dir = os.path.join(self.tmp, "out")
        for d in (self.no2_dir, self.so2_dir, self.viirs_dir):
            os.makedirs(d)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _save(self, dir_, name, arr):
        np.save(os.path.join(dir_, name), arr.astype(np.float32))

    def test_only_common_filenames_are_paired(self):
        # "A.npy" exists in all three sources, "B.npy" only in NO2/SO2 --
        # only A should make it through, and the count returned must match.
        shape = (4, 4)
        self._save(self.no2_dir, "A.npy", np.ones(shape))
        self._save(self.so2_dir, "A.npy", np.ones(shape))
        self._save(self.viirs_dir, "A.npy", np.ones(shape))
        self._save(self.no2_dir, "B.npy", np.ones(shape))
        self._save(self.so2_dir, "B.npy", np.ones(shape))
        # no B.npy in viirs_dir -- B should be dropped

        saved = b3c.triple_folder(self.no2_dir, self.so2_dir, self.viirs_dir, self.out_dir)
        self.assertEqual(saved, 1)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, "A.npy")))
        self.assertFalse(os.path.exists(os.path.join(self.out_dir, "B.npy")))

    def test_mismatched_shapes_are_skipped(self):
        self._save(self.no2_dir, "C.npy", np.ones((4, 4)))
        self._save(self.so2_dir, "C.npy", np.ones((4, 4)))
        self._save(self.viirs_dir, "C.npy", np.ones((8, 8)))  # wrong shape

        saved = b3c.triple_folder(self.no2_dir, self.so2_dir, self.viirs_dir, self.out_dir)
        self.assertEqual(saved, 0)

    def test_gaps_filled_with_channel_own_mean(self):
        no2 = np.array([[1.0, np.nan], [3.0, 4.0]])
        self._save(self.no2_dir, "D.npy", no2)
        self._save(self.so2_dir, "D.npy", np.ones((2, 2)))
        self._save(self.viirs_dir, "D.npy", np.ones((2, 2)))

        b3c.triple_folder(self.no2_dir, self.so2_dir, self.viirs_dir, self.out_dir)
        stacked = np.load(os.path.join(self.out_dir, "D.npy"))
        # NaN should be replaced by nanmean of the other 3 values: (1+3+4)/3
        expected_fill = np.nanmean(no2)
        self.assertAlmostEqual(float(stacked[0, 0, 1]), expected_fill, places=5)
        self.assertFalse(np.any(np.isnan(stacked)))

    def test_so2_and_viirs_negative_values_clamped_to_zero(self):
        self._save(self.no2_dir, "E.npy", np.full((2, 2), -5.0))   # NO2 NOT clamped
        self._save(self.so2_dir, "E.npy", np.full((2, 2), -5.0))   # SO2 clamped
        self._save(self.viirs_dir, "E.npy", np.full((2, 2), -5.0))  # VIIRS clamped

        b3c.triple_folder(self.no2_dir, self.so2_dir, self.viirs_dir, self.out_dir)
        stacked = np.load(os.path.join(self.out_dir, "E.npy"))
        self.assertTrue(np.all(stacked[0] == -5.0))  # NO2 channel untouched
        self.assertTrue(np.all(stacked[1] == 0.0))   # SO2 channel clamped
        self.assertTrue(np.all(stacked[2] == 0.0))   # VIIRS channel clamped

    def test_output_shape_is_channels_first_stack(self):
        shape = (6, 6)
        self._save(self.no2_dir, "F.npy", np.ones(shape))
        self._save(self.so2_dir, "F.npy", np.full(shape, 2.0))
        self._save(self.viirs_dir, "F.npy", np.full(shape, 3.0))

        b3c.triple_folder(self.no2_dir, self.so2_dir, self.viirs_dir, self.out_dir)
        stacked = np.load(os.path.join(self.out_dir, "F.npy"))
        self.assertEqual(stacked.shape, (3, 6, 6))
        self.assertTrue(np.all(stacked[0] == 1.0))
        self.assertTrue(np.all(stacked[1] == 2.0))
        self.assertTrue(np.all(stacked[2] == 3.0))


if __name__ == "__main__":
    unittest.main()
