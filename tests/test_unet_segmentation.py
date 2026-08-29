"""
Unit tests for unet_segmentation.py -- the reusable pieces behind Week
21's segmentation-only U-Net (facility-level split, NaN handling,
Dice/IoU metrics, model shape). Does not run full training -- see
train_unet_segmentation.py for that.
"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unet_segmentation as useg


class TestFacilityLevelSplit(unittest.TestCase):
    def test_no_facility_appears_in_two_splits(self):
        rng = np.random.default_rng(0)
        # 20 facilities, 1-5 tiles each
        facility_ids = []
        for fid in range(20):
            for _ in range(rng.integers(1, 6)):
                facility_ids.append(fid)
        rng.shuffle(facility_ids)

        train_idx, val_idx, test_idx = useg.facility_level_split(facility_ids, seed=1)
        train_fids = set(facility_ids[i] for i in train_idx)
        val_fids = set(facility_ids[i] for i in val_idx)
        test_fids = set(facility_ids[i] for i in test_idx)

        self.assertEqual(len(train_fids & val_fids), 0)
        self.assertEqual(len(train_fids & test_fids), 0)
        self.assertEqual(len(val_fids & test_fids), 0)

    def test_every_tile_assigned_exactly_once(self):
        facility_ids = [0, 0, 1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10]
        train_idx, val_idx, test_idx = useg.facility_level_split(facility_ids, seed=2)
        all_idx = sorted(train_idx + val_idx + test_idx)
        self.assertEqual(all_idx, list(range(len(facility_ids))))

    def test_unique_negative_ids_split_independently(self):
        # negatives given unique ids (e.g. -1, -2, ...) should split
        # roughly tile-level, with no leakage concern since each is its
        # own "facility" of size 1
        facility_ids = list(range(0, 10)) * 3 + list(range(-1, -31, -1))
        train_idx, val_idx, test_idx = useg.facility_level_split(facility_ids, seed=3,
                                                                   train_frac=0.7, val_frac=0.15)
        self.assertGreater(len(train_idx), 0)
        self.assertGreater(len(val_idx), 0)
        self.assertGreater(len(test_idx), 0)

    def test_split_seed_is_reproducible(self):
        facility_ids = list(range(30))
        r1 = useg.facility_level_split(facility_ids, seed=7)
        r2 = useg.facility_level_split(facility_ids, seed=7)
        self.assertEqual(r1, r2)


class TestNanHandling(unittest.TestCase):
    def test_nan_pixels_filled_and_marked_invalid(self):
        tile = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
        filled, valid = useg.fill_nan_and_validity(tile[None, :, :])
        self.assertFalse(np.isnan(filled).any())
        self.assertEqual(filled[0, 0, 1], useg.BG_XCO2_PPM)
        self.assertTrue(valid[0, 0, 0])
        self.assertFalse(valid[0, 0, 1])

    def test_no_nan_leaves_all_valid(self):
        tile = np.ones((1, 4, 4), dtype=np.float32) * 415.0
        filled, valid = useg.fill_nan_and_validity(tile)
        self.assertTrue(valid.all())
        np.testing.assert_array_equal(filled, tile)


class TestBuildModelInput(unittest.TestCase):
    def test_shape_and_channel_content(self):
        # WEEK 23: channel 0 must be the normalized tile untouched,
        # channel 1 the valid mask as float 0/1 -- this is the exact
        # function both train_unet_segmentation.py and
        # sanity_check_real_tiles.py call, so this test is the
        # consistency guarantee for both pipelines at once.
        norm_tile = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float32)
        valid = np.array([[True, False], [True, True]])
        x = useg.build_model_input(norm_tile, valid)
        self.assertEqual(x.shape, (2, 2, 2))
        np.testing.assert_array_equal(x[0], norm_tile)
        np.testing.assert_array_equal(x[1], valid.astype(np.float32))

    def test_accepts_float_or_bool_valid_mask(self):
        norm_tile = np.zeros((3, 3), dtype=np.float32)
        valid_bool = np.array([[True, False, True]] * 3)
        valid_float = valid_bool.astype(np.float32)
        x_bool = useg.build_model_input(norm_tile, valid_bool)
        x_float = useg.build_model_input(norm_tile, valid_float)
        np.testing.assert_array_equal(x_bool, x_float)

    def test_compatible_with_unet_two_channel_input(self):
        norm_tile = np.random.default_rng(0).normal(size=(64, 64)).astype(np.float32)
        valid = np.random.default_rng(1).random((64, 64)) > 0.9
        x = useg.build_model_input(norm_tile, valid)
        model = useg.UNetSmall(in_channels=2, base_channels=4)
        out = model(torch.from_numpy(x).unsqueeze(0))
        self.assertEqual(out.shape, (1, 1, 64, 64))


class TestDiceIou(unittest.TestCase):
    def test_perfect_match_gives_one(self):
        pred = np.array([[1, 1, 0], [0, 1, 0]])
        target = np.array([[1, 1, 0], [0, 1, 0]])
        self.assertAlmostEqual(useg.dice_coefficient(pred, target), 1.0, places=4)
        self.assertAlmostEqual(useg.iou_score(pred, target), 1.0, places=4)

    def test_no_overlap_gives_near_zero(self):
        pred = np.array([[1, 0], [0, 0]])
        target = np.array([[0, 0], [0, 1]])
        self.assertLess(useg.dice_coefficient(pred, target), 0.01)
        self.assertLess(useg.iou_score(pred, target), 0.01)

    def test_invalid_pixels_excluded(self):
        # a false positive that falls entirely in an invalid (cloud)
        # region must not count against the score
        pred = np.array([[1, 1], [0, 0]])
        target = np.array([[0, 1], [0, 0]])
        valid = np.array([[0, 1], [1, 1]])  # top-left excluded
        dice_with_mask = useg.dice_coefficient(pred, target, valid=valid)
        dice_without_mask = useg.dice_coefficient(pred, target)
        self.assertGreater(dice_with_mask, dice_without_mask)

    def test_torch_tensor_inputs_work(self):
        pred = torch.tensor([[1, 0], [1, 1]])
        target = torch.tensor([[1, 0], [1, 0]])
        d = useg.dice_coefficient(pred, target)
        self.assertGreater(d, 0.0)
        self.assertLessEqual(d, 1.0)


class TestMaskedLoss(unittest.TestCase):
    def test_loss_is_finite_and_decreases_toward_perfect_prediction(self):
        torch.manual_seed(0)
        target = torch.zeros(2, 1, 8, 8)
        target[:, :, 2:5, 2:5] = 1.0
        valid = torch.ones_like(target)

        bad_logits = torch.zeros_like(target)  # predicts 0.5 everywhere
        good_logits = (target * 10.0) - 5.0    # confidently correct

        bad_loss = useg.masked_bce_dice_loss(bad_logits, target, valid)
        good_loss = useg.masked_bce_dice_loss(good_logits, target, valid)
        self.assertTrue(torch.isfinite(bad_loss))
        self.assertTrue(torch.isfinite(good_loss))
        self.assertLess(good_loss.item(), bad_loss.item())

    def test_invalid_pixels_do_not_affect_loss(self):
        # Wild logits at an INVALID pixel must not change the loss --
        # only the pixel's exclusion from the valid mask matters, not
        # whatever the network happens to predict there.
        torch.manual_seed(0)
        target = torch.zeros(1, 1, 4, 4)
        target[:, :, 0, 0] = 1.0
        valid = torch.ones_like(target)
        valid[:, :, 3, 3] = 0.0  # pixel (3,3) excluded

        logits_a = torch.zeros_like(target)
        logits_b = torch.zeros_like(target)
        logits_b[:, :, 3, 3] = 999.0  # extreme prediction at the excluded pixel only

        loss_a = useg.masked_bce_dice_loss(logits_a, target, valid)
        loss_b = useg.masked_bce_dice_loss(logits_b, target, valid)
        self.assertAlmostEqual(loss_a.item(), loss_b.item(), places=4)


class TestUNetShape(unittest.TestCase):
    def test_forward_shape_matches_input(self):
        model = useg.UNetSmall(in_channels=1, base_channels=4)
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, (2, 1, 64, 64))

    def test_device_str_is_mps_or_cpu(self):
        self.assertIn(useg.device_str(), ("mps", "cpu"))


if __name__ == "__main__":
    unittest.main()
