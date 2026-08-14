"""
Unit tests for lofo_track_a.py's facility-grouping and fold-splitting
logic -- specifically guarding against a regression of the Week 11
leakage bug (a random tile-level split let the same facility's tiles land
in both train and test across different months). facility_fold_indices()
is the one function whose correctness this entire LOFO harness depends
on: get it wrong and every recall number the harness reports is silently
inflated by leakage again, exactly like Week 11's tile-level split was.
"""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lofo_track_a as lofo


class TestMonthSuffixRegex(unittest.TestCase):
    def test_strips_year_month_suffix(self):
        self.assertEqual(lofo._MONTH_SUFFIX.sub("", "Talcher_2020_03.npy"), "Talcher")
        self.assertEqual(lofo._MONTH_SUFFIX.sub("", "MUNDRA_TPP_2019_12.npy"), "MUNDRA_TPP")

    def test_does_not_strip_facility_names_containing_digits(self):
        # a facility name that itself ends in digits close to the pattern
        # shape must not be mistaken for a _YYYY_MM.npy suffix.
        self.assertEqual(lofo._MONTH_SUFFIX.sub("", "Plant1_2020_06.npy"), "Plant1")


class TestFacilityFoldIndices(unittest.TestCase):
    def test_held_out_facility_fully_excluded_from_train(self):
        # 3 facilities, 2 tiles each -- holding out "B" must remove ALL of
        # B's tiles from train, not just some, and put exactly those in test.
        groups = ["A", "A", "B", "B", "C", "C"]
        tr_idx, te_idx = lofo.facility_fold_indices(groups, "B")
        train_groups = {groups[i] for i in tr_idx}
        test_groups = {groups[i] for i in te_idx}

        self.assertNotIn("B", train_groups)
        self.assertEqual(test_groups, {"B"})
        self.assertEqual(set(tr_idx) | set(te_idx), set(range(len(groups))))  # partition, nothing lost
        self.assertEqual(set(tr_idx) & set(te_idx), set())                    # no overlap

    def test_no_leakage_across_all_folds(self):
        # replicate the harness's own fold loop: for every facility held
        # out in turn, none of its tiles may appear in that fold's train set.
        groups = ["Anpara"] * 3 + ["Korba"] * 2 + ["Rihand"] * 4
        for held_out in sorted(set(groups)):
            tr_idx, te_idx = lofo.facility_fold_indices(groups, held_out)
            for i in tr_idx:
                self.assertNotEqual(groups[i], held_out,
                                     f"leakage: {held_out}'s own tile ended up in train")
            for i in te_idx:
                self.assertEqual(groups[i], held_out)

    def test_unknown_facility_yields_empty_test_set(self):
        groups = ["A", "B"]
        tr_idx, te_idx = lofo.facility_fold_indices(groups, "NotAFacility")
        self.assertEqual(te_idx, [])
        self.assertEqual(tr_idx, [0, 1])


if __name__ == "__main__":
    unittest.main()
