"""
Unit tests for physics_gaussian.py's pure math and month-stratification
logic (the two areas this codebase's own history shows are most likely to
silently produce a wrong number: the Week 6 wind-averaging bug and the
ShriSingajiMalwa seasonal-sampling artifact were both bugs in exactly this
kind of small, easy-to-get-subtly-wrong arithmetic).

Deliberately does not test estimate_emission_rate() end-to-end (it reads
from data/<Plant>_soundings.npz and calls Earth Engine for wind data) --
that's an integration path better exercised by actually running the
pipeline, not a unit test. These tests cover the deterministic building
blocks it's built from.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import physics_gaussian as pg


class TestColumnMassEnhancement(unittest.TestCase):
    def test_zero_ppm_gives_zero_mass(self):
        self.assertEqual(pg.column_mass_enhancement(0.0), 0.0)

    def test_scales_linearly_with_ppm(self):
        # column_mass_enhancement is dppm * const -- doubling ppm must
        # exactly double the result, not just "increase" it.
        one = pg.column_mass_enhancement(1.0)
        ten = pg.column_mass_enhancement(10.0)
        self.assertAlmostEqual(ten, one * 10, places=9)

    def test_known_value(self):
        # dry_air_col_mass = P_SURF/G = 101325/9.80665 ~= 10332.27 kg/m^2
        # result = dppm * 1e-6 * dry_air_col_mass * (M_CO2/M_AIR)
        dppm = 1.0
        expected = (dppm * 1e-6 * (pg.P_SURF / pg.G) * (pg.M_CO2 / pg.M_AIR))
        self.assertAlmostEqual(pg.column_mass_enhancement(dppm), expected, places=12)


class TestImeKg(unittest.TestCase):
    def test_near_below_background_contributes_zero(self):
        # every near-plant sounding below the background mean must be
        # clipped to zero excess, not counted as negative mass.
        near = np.array([400.0, 401.0, 402.0])  # all below bg_mean=410
        ime = pg._ime_kg(near, bg_mean=410.0)
        self.assertEqual(ime, 0.0)

    def test_positive_excess_is_summed_not_averaged(self):
        near = np.array([411.0, 412.0])  # each 1 and 2 ppm above bg
        bg_mean = 410.0
        ime = pg._ime_kg(near, bg_mean)
        expected = (pg.column_mass_enhancement(1.0) + pg.column_mass_enhancement(2.0)) * pg.FOOTPRINT_AREA_M2
        self.assertAlmostEqual(ime, expected, places=6)


class TestMonthOfDay(unittest.TestCase):
    def test_extracts_month_from_yyyymmdd(self):
        self.assertEqual(pg._month_of_day(20200315), 3)
        self.assertEqual(pg._month_of_day(20191201), 12)

    def test_zero_sentinel_returns_none(self):
        self.assertIsNone(pg._month_of_day(0))


class TestMonthStratifyBg(unittest.TestCase):
    def test_no_day_data_returns_unchanged(self):
        bg = np.array([1.0, 2.0, 3.0])
        out, stratified = pg._month_stratify_bg(bg, None, {1, 2})
        np.testing.assert_array_equal(out, bg)
        self.assertFalse(stratified)

    def test_restricts_to_shared_months_when_enough_remain(self):
        # background drawn from two months (Jan, Apr); near-plant zone
        # only covers January -- this is exactly the ShriSingajiMalwa
        # shape of bug: April's background values should be dropped.
        bg = np.array([410.0] * 6 + [416.0] * 6)  # 6 Jan (low), 6 Apr (high)
        bg_day = np.array([20200101 + i for i in range(6)] +
                           [20200401 + i for i in range(6)])
        out, stratified = pg._month_stratify_bg(bg, bg_day, {1}, min_kept=5)
        self.assertTrue(stratified)
        self.assertEqual(len(out), 6)
        self.assertTrue(np.all(out == 410.0))  # only January values survive

    def test_falls_back_when_too_few_kept(self):
        # only 2 January soundings in the background -- below min_kept=5,
        # so the function must fall back to the FULL unrestricted
        # background rather than returning an unusably small population.
        bg = np.array([410.0, 411.0] + [416.0] * 6)
        bg_day = np.array([20200101, 20200102] + [20200401 + i for i in range(6)])
        out, stratified = pg._month_stratify_bg(bg, bg_day, {1}, min_kept=5)
        self.assertFalse(stratified)
        np.testing.assert_array_equal(out, bg)


class TestBgDefinitionRelStd(unittest.TestCase):
    def test_returns_zero_with_too_few_definitions(self):
        # a near-empty sounding population won't produce
        # MIN_BG_DEFINITIONS usable alternates -- must degrade to 0, not
        # raise or return NaN.
        dist = np.array([0.1, 0.1])
        xco2 = np.array([412.0, 413.0])
        near = np.array([412.0, 413.0])
        rel_std, n_defs = pg._bg_definition_rel_std(dist, xco2, near)
        self.assertEqual(rel_std, 0.0)
        self.assertLess(n_defs, pg.MIN_BG_DEFINITIONS)

    def test_month_stratification_reduces_available_definitions_or_leaves_result_finite(self):
        rng = np.random.default_rng(0)
        n = 200
        dist = rng.uniform(0.0, 1.2, n)
        xco2 = rng.normal(412.0, 1.0, n)
        near = xco2[dist < pg.NEAR]
        day = np.full(n, 20200101)  # all soundings same day/month
        near_months = {1}
        rel_std, n_defs = pg._bg_definition_rel_std(dist, xco2, near, day=day,
                                                      near_months_set=near_months)
        self.assertTrue(np.isfinite(rel_std))
        self.assertGreaterEqual(n_defs, 0)


if __name__ == "__main__":
    unittest.main()
