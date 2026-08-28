"""
Unit tests for simulate_training_pairs.py's synthetic tile generator.
Verifies the three documented bug fixes (near-field clip, wind-speed
floor, tile resolution) and the Task 2 area-averaging behavior, not
invented expected outputs.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simulate_training_pairs as sim


class TestTileGeometry(unittest.TestCase):
    def test_matches_track_a_resolution(self):
        self.assertEqual(sim.SIZE_KM, 60)
        self.assertEqual(sim.PX, 64)
        self.assertAlmostEqual(sim.PX_SIZE_M, 937.5, places=1)

    def test_pixel_centers_shape_and_span(self):
        east_km, north_km = sim._pixel_centers_km()
        self.assertEqual(east_km.shape, (sim.PX, sim.PX))
        self.assertLess(east_km.max(), sim.SIZE_KM / 2.0)
        self.assertGreater(east_km.min(), -sim.SIZE_KM / 2.0)


class TestNearFieldGuard(unittest.TestCase):
    def test_upwind_pixels_are_zero(self):
        # Wind from the north (0 deg) => plume travels south => pixels
        # north of the source (positive north_km) are upwind.
        conc = sim.area_averaged_concentration_kg_m3(
            Q_t_per_year=1e7, wind_speed_ms=2.0, wind_from_deg=0.0,
            stack_height_m=220.0, stability_class="B")
        east_km, north_km = sim._pixel_centers_km()
        upwind = conc[north_km > 5.0]
        self.assertTrue(np.all(upwind == 0.0))

    def test_no_near_field_singularity(self):
        # A pixel essentially at the source must not blow up -- the
        # near-field floor (max(3*H, 300)) must bound it.
        conc = sim.area_averaged_concentration_kg_m3(
            Q_t_per_year=4.9e7, wind_speed_ms=1.2, wind_from_deg=180.0,
            stack_height_m=150.0, stability_class="A")
        self.assertTrue(np.all(np.isfinite(conc)))
        self.assertTrue(np.all(conc >= 0.0))


class TestAreaAveragingConverges(unittest.TestCase):
    def test_increasing_subgrid_density_converges(self):
        # Task 2: area-averaging should converge as the subgrid gets
        # denser, not diverge -- this distinguishes a real physics
        # effect from a discretization artifact.
        kwargs = dict(Q_t_per_year=2e7, wind_speed_ms=2.0, wind_from_deg=90.0,
                      stack_height_m=220.0, stability_class="B")
        orig_n = sim.AREA_AVG_N
        try:
            peaks = []
            for n in (3, 9, 15):
                sim.AREA_AVG_N = n
                conc = sim.area_averaged_concentration_kg_m3(**kwargs)
                peaks.append(conc.max())
            self.assertAlmostEqual(peaks[1], peaks[2], delta=peaks[2] * 0.05)
        finally:
            sim.AREA_AVG_N = orig_n


class TestMakeTile(unittest.TestCase):
    def test_positive_tile_shape_and_types(self):
        rng = np.random.default_rng(0)
        tile, mask, params = sim.make_tile(rng, positive=True)
        self.assertEqual(tile.shape, (sim.PX, sim.PX))
        self.assertEqual(mask.shape, (sim.PX, sim.PX))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertTrue(params["positive"])
        self.assertGreater(params["q_t_per_year"], 0.0)

    def test_negative_tile_has_no_plume(self):
        rng = np.random.default_rng(0)
        tile, mask, params = sim.make_tile(rng, positive=False)
        self.assertEqual(params["q_t_per_year"], 0.0)
        self.assertTrue(np.all(mask == 0))

    def test_cloud_gap_fraction_in_expected_range(self):
        rng = np.random.default_rng(1)
        for _ in range(10):
            tile, mask, params = sim.make_tile(rng, positive=True)
            self.assertGreaterEqual(params["cloud_gap_frac"], sim.CLOUD_GAP_FRAC_RANGE[0] - 0.05)
            self.assertLessEqual(params["cloud_gap_frac"], sim.CLOUD_GAP_FRAC_RANGE[1] + 0.05)
            self.assertTrue(np.any(np.isnan(tile)))

    def test_wind_speed_never_below_real_data_floor(self):
        rng = np.random.default_rng(2)
        for _ in range(20):
            _, _, params = sim.make_tile(rng, positive=True)
            self.assertGreaterEqual(params["wind_speed_ms"], 1.2)

    def test_positive_tile_has_ime_readout_field(self):
        rng = np.random.default_rng(3)
        _, _, params = sim.make_tile(rng, positive=True)
        self.assertIn("ime_readout_ppm", params)
        self.assertTrue(np.isfinite(params["ime_readout_ppm"]))


class TestImeStyleReadout(unittest.TestCase):
    def test_matches_physics_ime_geometry_constants(self):
        # Task 4: the readout MUST use physics_ime.py's own NEAR/BG_IN/
        # BG_OUT constants, not invented values -- this pins that link.
        import physics_ime
        self.assertAlmostEqual(sim.IME_NEAR_KM, physics_ime.NEAR * sim.KM_PER_DEG)
        self.assertAlmostEqual(sim.IME_BG_IN_KM, physics_ime.BG_IN * sim.KM_PER_DEG)
        self.assertAlmostEqual(sim.IME_BG_OUT_KM, physics_ime.BG_OUT * sim.KM_PER_DEG)

    def test_readout_grid_covers_full_background_annulus(self):
        # The whole point of the separate calibration grid is that it must
        # be big enough to contain BG_OUT -- if it weren't, the background
        # mean would be silently wrong.
        self.assertGreater(sim.READOUT_HALF_EXTENT_KM, sim.IME_BG_OUT_KM)

    def test_negative_tile_readout_is_near_zero(self):
        rng = np.random.default_rng(4)
        vals = [sim.ime_style_readout_ppm(rng, 0.0, 2.0, 90.0, 220.0, "B")
                for _ in range(20)]
        self.assertLess(abs(np.mean(vals)), 0.05)

    def test_positive_tile_readout_exceeds_negative_tile_readout(self):
        rng = np.random.default_rng(5)
        pos = sim.ime_style_readout_ppm(rng, 2e7, 2.0, 90.0, 220.0, "B")
        neg = sim.ime_style_readout_ppm(rng, 0.0, 2.0, 90.0, 220.0, "B")
        self.assertGreater(pos, neg)


if __name__ == "__main__":
    unittest.main()
