"""
Unit tests for plume_model.py's Gaussian plume implementation. Verifies
the physics against known analytic properties of the equation, not
against invented "expected outputs" -- these are properties any correct
implementation of the Briggs/Pasquill-Gifford plume equation must satisfy.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plume_model as pm


class TestSigmaMonotonicity(unittest.TestCase):
    def test_sigma_y_increases_with_downwind_distance(self):
        x = np.array([100.0, 1000.0, 10000.0])
        for cls in "ABCDEF":
            sy = pm._sigma_y(x, cls)
            self.assertTrue(np.all(np.diff(sy) > 0), f"sigma_y not monotonic for class {cls}")

    def test_sigma_z_increases_with_downwind_distance(self):
        x = np.array([100.0, 1000.0, 10000.0])
        for cls in "ABCDEF":
            sz = pm._sigma_z(x, cls)
            self.assertTrue(np.all(np.diff(sz) > 0), f"sigma_z not monotonic for class {cls}")

    def test_unstable_classes_disperse_faster_than_stable(self):
        # Class A (most unstable) must have larger sigma_y/sigma_z than
        # class F (most stable) at the same downwind distance -- this is
        # the defining physical property of the stability classification.
        x = np.array([5000.0])
        self.assertGreater(pm._sigma_y(x, "A")[0], pm._sigma_y(x, "F")[0])
        self.assertGreater(pm._sigma_z(x, "A")[0], pm._sigma_z(x, "F")[0])


class TestGroundLevelConcentration(unittest.TestCase):
    def test_zero_upwind_of_source(self):
        conc = pm.ground_level_concentration(1e6, 2.0, 220.0, x_m=-500.0, y_m=0.0)
        self.assertEqual(float(conc), 0.0)

    def test_zero_far_off_crosswind_axis(self):
        near_axis = pm.ground_level_concentration(1e6, 2.0, 220.0, x_m=5000.0, y_m=0.0)
        far_off_axis = pm.ground_level_concentration(1e6, 2.0, 220.0, x_m=5000.0, y_m=50000.0)
        self.assertGreater(float(near_axis), float(far_off_axis))
        self.assertAlmostEqual(float(far_off_axis), 0.0, places=10)

    def test_scales_linearly_with_emission_rate(self):
        c1 = pm.ground_level_concentration(1e6, 2.0, 220.0, x_m=3000.0, y_m=0.0)
        c2 = pm.ground_level_concentration(2e6, 2.0, 220.0, x_m=3000.0, y_m=0.0)
        self.assertAlmostEqual(float(c2) / float(c1), 2.0, places=6)

    def test_higher_wind_speed_dilutes_concentration(self):
        slow = pm.ground_level_concentration(1e6, 1.0, 220.0, x_m=3000.0, y_m=0.0)
        fast = pm.ground_level_concentration(1e6, 5.0, 220.0, x_m=3000.0, y_m=0.0)
        self.assertGreater(float(slow), float(fast))

    def test_raises_on_calm_wind(self):
        with self.assertRaises(ValueError):
            pm.ground_level_concentration(1e6, 0.0, 220.0, x_m=1000.0, y_m=0.0)

    def test_taller_stack_lowers_ground_level_peak_at_moderate_distance(self):
        # A taller stack keeps the plume centerline higher above ground,
        # so ground-level concentration close to the source should be
        # lower for a taller stack (until far enough downwind that the
        # plume has spread vertically past H).
        short_stack = pm.ground_level_concentration(1e6, 2.0, 100.0, x_m=2000.0, y_m=0.0)
        tall_stack = pm.ground_level_concentration(1e6, 2.0, 400.0, x_m=2000.0, y_m=0.0)
        self.assertGreater(float(short_stack), float(tall_stack))


class TestConcentrationAtLocations(unittest.TestCase):
    def test_point_evaluation_matches_grid_evaluation(self):
        # concentration_at_locations() and plume_grid() must agree at
        # coincident points -- they share _rotate_to_plume_frame(), but
        # this guards against the two call sites drifting apart.
        grid, ex, nx = pm.plume_grid(1e6, 2.0, wind_from_deg=200.0, extent_km=20, resolution_m=1000)
        i, j = 15, 10  # arbitrary interior grid point
        east_km, north_km = ex[j], nx[i]
        point_val = pm.concentration_at_locations(1e6, 2.0, 200.0, east_km, north_km)
        self.assertAlmostEqual(float(point_val), float(grid[i, j]), places=6)

    def test_accepts_array_of_locations(self):
        vals = pm.concentration_at_locations(1e6, 2.0, 90.0, east_km=[1.0, 2.0, 3.0], north_km=[0.0, 0.0, 0.0])
        self.assertEqual(len(vals), 3)
        self.assertTrue(np.all(np.isfinite(vals)))


class TestPlumeGrid(unittest.TestCase):
    def test_grid_shape_matches_axes(self):
        grid, ex, nx = pm.plume_grid(1e6, 2.0, wind_from_deg=270.0, extent_km=10, resolution_m=1000)
        self.assertEqual(grid.shape, (len(nx), len(ex)))

    def test_plume_points_downwind_not_upwind(self):
        # Wind FROM the west (270 deg) means the plume travels east. The
        # concentration integrated over the eastern half of the grid must
        # exceed the western half.
        grid, ex, nx = pm.plume_grid(1e6, 2.0, wind_from_deg=270.0, extent_km=20, resolution_m=500)
        east_half = grid[:, ex > 0].sum()
        west_half = grid[:, ex < 0].sum()
        self.assertGreater(east_half, west_half)

    def test_rotating_wind_direction_rotates_plume(self):
        # Wind FROM the north (0 deg) => plume travels south => mass
        # concentrated in the southern (negative-north) half of the grid.
        grid, ex, nx = pm.plume_grid(1e6, 2.0, wind_from_deg=0.0, extent_km=20, resolution_m=500)
        south_half = grid[nx < 0, :].sum()
        north_half = grid[nx > 0, :].sum()
        self.assertGreater(south_half, north_half)

    def test_grid_is_finite_and_nonnegative(self):
        grid, _, _ = pm.plume_grid(5e7, 1.5, wind_from_deg=108.0)
        self.assertTrue(np.all(np.isfinite(grid)))
        self.assertTrue(np.all(grid >= 0))


if __name__ == "__main__":
    unittest.main()
