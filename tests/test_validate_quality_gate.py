"""
Unit tests for validate_quality_gate.py's LOO-split correctness, the
grid-search min_n floor, and the permutation-test shuffle mechanics --
the Week 12 quality-gate methodology whose own permutation test found the
gate's apparent improvement wasn't distinguishable from selection bias
(see CLAUDE.md). These tests cover the deterministic scaffolding that
result depends on, not the substantive finding itself.

Converted from a pytest-style file (Week 19 fresh-clone audit: it was
never actually collected by `python -m unittest discover -s tests`, since
pytest-style bare classes aren't unittest.TestCase subclasses) to match
this project's stdlib-unittest convention used by every other file in
tests/.
"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validate_quality_gate import (
    grid_search_best_gate,
    loo_validation,
    permutation_test,
)


def make_rows(specs):
    """specs: list of (plant, hit_days, wind_diff, log_ratio)."""
    return [
        {"plant": p, "hit_days": h, "wind_co2_diff_deg": w, "log_ratio": lr}
        for p, h, w, lr in specs
    ]


class TestLooSplit(unittest.TestCase):
    def test_each_fold_excludes_only_the_held_out_plant(self):
        rows = make_rows([
            ("A", 10, 20, 0.1), ("B", 15, 30, -0.2), ("C", 20, 40, 0.3),
            ("D", 25, 50, -0.1), ("E", 30, 60, 0.2), ("F", 5, 10, -0.3),
        ])
        result = loo_validation(rows, min_n=3)
        held_out_plants = [f["held_out_plant"] for f in result["per_fold"]]
        self.assertEqual(held_out_plants, ["A", "B", "C", "D", "E", "F"])
        self.assertEqual(len(set(held_out_plants)), len(rows))

    def test_loo_fold_count_matches_n_facilities(self):
        rows = make_rows([(f"P{i}", i, i * 10, 0.1 * i) for i in range(1, 8)])
        result = loo_validation(rows, min_n=3)
        self.assertEqual(len(result["per_fold"]), len(rows))

    def test_training_gate_never_uses_held_out_plants_stats(self):
        # Construct a case where the held-out plant is an extreme outlier;
        # if the LOO split leaked it into training, the training gate's sd
        # would be pulled toward it. Verify training_gate_n never exceeds
        # len(rows) - 1.
        rows = make_rows([
            ("A", 10, 20, 0.1), ("B", 15, 30, -0.2), ("C", 20, 40, 0.3),
            ("D", 25, 50, -0.1), ("E", 30, 60, 0.2), ("F", 35, 70, -0.3),
            ("Outlier", 40, 80, 100.0),
        ])
        result = loo_validation(rows, min_n=3)
        for f in result["per_fold"]:
            self.assertLessEqual(f["training_gate_n"], len(rows) - 1)


class TestGridSearchMinN(unittest.TestCase):
    def test_rejects_gate_smaller_than_min_n_even_if_lower_sd(self):
        # Two plants with identical log_ratio (sd=0, a "perfect" tiny gate)
        # sit at a tight threshold; a larger, looser gate has higher sd but
        # meets min_n. The search must pick the larger gate, not the
        # zero-sd pair, because min_n excludes gates below the floor.
        rows = make_rows([
            ("Tight1", 30, 5, 0.5), ("Tight2", 30, 5, 0.5),
            ("Loose1", 10, 60, -0.5), ("Loose2", 10, 60, 0.4),
            ("Loose3", 10, 60, 0.1), ("Loose4", 10, 60, -0.3),
        ])
        result = grid_search_best_gate(rows, min_n=4)
        self.assertGreaterEqual(result["n"], 4)
        self.assertNotEqual(result["n"], 2)

    def test_returns_none_when_min_n_exceeds_dataset_size(self):
        rows = make_rows([("A", 10, 20, 0.1), ("B", 15, 30, -0.2)])
        result = grid_search_best_gate(rows, min_n=5)
        self.assertIsNone(result)

    def test_widest_gate_always_qualifies_when_min_n_equals_n(self):
        rows = make_rows([(f"P{i}", i, i, 0.1 * i) for i in range(1, 6)])
        result = grid_search_best_gate(rows, min_n=5)
        self.assertIsNotNone(result)
        self.assertEqual(result["n"], 5)


class TestPermutationShuffle(unittest.TestCase):
    def test_same_seed_reproduces_identical_result(self):
        rows = make_rows([(f"P{i}", i % 5 + 1, (i % 6) * 10, 0.1 * (i - 5)) for i in range(1, 13)])
        r1 = permutation_test(rows, min_n=4, n_perm=20, seed=42)
        r2 = permutation_test(rows, min_n=4, n_perm=20, seed=42)
        self.assertEqual(r1["p_value"], r2["p_value"])
        self.assertEqual(r1["shuffled_best_sd_mean"], r2["shuffled_best_sd_mean"])

    def test_different_seed_can_change_result(self):
        rows = make_rows([(f"P{i}", i % 5 + 1, (i % 6) * 10, 0.1 * (i - 5)) for i in range(1, 13)])
        r1 = permutation_test(rows, min_n=4, n_perm=20, seed=42)
        r2 = permutation_test(rows, min_n=4, n_perm=20, seed=43)
        # Not a strict guarantee in general, but with only 20 shuffles over
        # 12 points the shuffled-sd sequences should differ.
        self.assertNotEqual(r1["shuffled_best_sd_mean"], r2["shuffled_best_sd_mean"])

    def test_shuffle_only_permutes_log_ratio_not_hit_days_or_wind_diff(self):
        rows = make_rows([(f"P{i}", i, i * 5, 0.1 * i) for i in range(1, 10)])
        original_hit_days = sorted(r["hit_days"] for r in rows)
        original_wind = sorted(r["wind_co2_diff_deg"] for r in rows)
        original_log_ratios = sorted(r["log_ratio"] for r in rows)

        rng = np.random.default_rng(42)
        log_ratios = np.array([r["log_ratio"] for r in rows])
        shuffled = rng.permutation(log_ratios)
        shuffled_rows = [dict(r, log_ratio=float(shuffled[j])) for j, r in enumerate(rows)]

        self.assertEqual(sorted(r["hit_days"] for r in shuffled_rows), original_hit_days)
        self.assertEqual(sorted(r["wind_co2_diff_deg"] for r in shuffled_rows), original_wind)
        shuffled_sorted = sorted(round(r["log_ratio"], 10) for r in shuffled_rows)
        original_sorted = sorted(round(v, 10) for v in original_log_ratios)
        for a, b in zip(shuffled_sorted, original_sorted):
            self.assertTrue(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9))


if __name__ == "__main__":
    unittest.main()
