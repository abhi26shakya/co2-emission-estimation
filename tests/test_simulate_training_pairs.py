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


class TestMultiDayPooling(unittest.TestCase):
    def test_n_days_matches_wind_dirs_returned(self):
        rng = np.random.default_rng(6)
        pooled, wind_dirs = sim.multi_day_ime_readout_ppm(rng, 2e7, 2.0, 220.0, "B", n_days=7)
        self.assertEqual(len(wind_dirs), 7)
        self.assertTrue(np.isfinite(pooled))

    def test_wind_directions_vary_across_days(self):
        # Each day must resample its own wind direction, not reuse one.
        rng = np.random.default_rng(7)
        _, wind_dirs = sim.multi_day_ime_readout_ppm(rng, 2e7, 2.0, 220.0, "B", n_days=10)
        self.assertGreater(len(set(wind_dirs)), 1)

    def test_pooling_uses_real_hit_days_range(self):
        # Task 5: N must come from the real hit_days distribution, not an
        # arbitrary constant -- pins the documented real range.
        self.assertEqual(min(sim.HIT_DAYS_POOL), 1)
        self.assertEqual(max(sim.HIT_DAYS_POOL), 25)
        self.assertEqual(len(sim.HIT_DAYS_POOL), 30)

    def test_make_tile_shares_facility_params_but_resamples_wind_direction(self):
        # Scoping constraint: individual tiles must still be single-
        # snapshot -- shared (q, wind_speed, stack, stability) but a FRESH
        # wind_from_deg every call.
        rng = np.random.default_rng(8)
        q, wind_speed, stack_height, stability = 2e7, 2.0, 220.0, "B"
        dirs = []
        for _ in range(5):
            _, _, params = sim.make_tile(rng, positive=True, q=q, wind_speed=wind_speed,
                                          stack_height=stack_height, stability=stability)
            self.assertEqual(params["q_t_per_year"], q)
            self.assertEqual(params["wind_speed_ms"], wind_speed)
            self.assertEqual(params["stack_height_m"], stack_height)
            self.assertEqual(params["stability_class"], stability)
            dirs.append(params["wind_from_deg"])
        self.assertGreater(len(set(dirs)), 1)

    def test_make_tile_accepts_explicit_wind_from_deg(self):
        # Task 6: one SAM scan = one training tile requires the SAME
        # wind_from_deg driving both -- make_tile must accept, not just
        # sample, this value.
        rng = np.random.default_rng(9)
        _, _, params = sim.make_tile(rng, positive=True, q=2e7, wind_speed=2.0,
                                      stack_height=220.0, stability="B", wind_from_deg=123.4)
        self.assertEqual(params["wind_from_deg"], 123.4)


class TestSamGeometry(unittest.TestCase):
    def test_raw_footprint_count_matches_geometry(self):
        self.assertEqual(sim.RAW_FOOTPRINTS_PER_SAM_SCAN,
                          sim.N_SWATHS * sim.N_FRAMES_PER_SWATH * sim.FOOTPRINTS_PER_FRAME)

    def test_sam_box_covers_80km_target(self):
        self.assertGreaterEqual(sim.N_SWATHS * sim.SWATH_WIDTH_KM, 80.0)
        self.assertGreaterEqual(sim.N_FRAMES_PER_SWATH * sim.FRAME_SPACING_KM, 80.0)

    def test_sam_scan_footprints_within_box_and_retention_frac(self):
        rng = np.random.default_rng(10)
        east, north = sim._sam_scan_footprint_offsets_km(rng, retention_frac=1.0)
        self.assertEqual(east.size, sim.RAW_FOOTPRINTS_PER_SAM_SCAN)
        self.assertTrue(np.all(np.abs(east) <= sim.SAM_BOX_HALF_KM))
        self.assertTrue(np.all(np.abs(north) <= sim.SAM_BOX_HALF_KM))

    def test_retention_frac_scales_footprint_count(self):
        rng = np.random.default_rng(11)
        east_full, _ = sim._sam_scan_footprint_offsets_km(rng, retention_frac=1.0)
        east_half, _ = sim._sam_scan_footprint_offsets_km(rng, retention_frac=0.5)
        self.assertLess(east_half.size, east_full.size)

    def test_background_footprints_within_annulus(self):
        rng = np.random.default_rng(12)
        east, north = sim._background_footprint_offsets_km(rng, bg_density_km2=0.01)
        if east.size > 0:
            r = np.sqrt(east ** 2 + north ** 2)
            self.assertTrue(np.all(r >= sim.IME_BG_IN_KM))
            self.assertTrue(np.all(r <= sim.IME_BG_OUT_KM))

    def test_evaluate_footprints_positive_q_exceeds_zero_q(self):
        rng = np.random.default_rng(13)
        east = np.array([5.0, 10.0])
        north = np.array([-30.0, -30.0])  # downwind if wind blows from north (0 deg)
        pos = sim._evaluate_footprints_ppm(rng, east, north, 2e7, 2.0, 0.0, 220.0, "B")
        zero = sim._evaluate_footprints_ppm(rng, east, north, 0.0, 2.0, 0.0, 220.0, "B")
        self.assertTrue(np.all(pos >= zero - 5))  # noise-tolerant, but should not be systematically lower


class TestSamQRecovery(unittest.TestCase):
    def test_recover_q_returns_result_and_n_days_wind_dirs(self):
        rng = np.random.default_rng(14)
        result, wind_dirs, n_near, n_bg, ppm = sim.recover_q_from_sam_scans(
            rng, "test_facility", 2e7, 2.0, 220.0, "B", n_days=5,
            retention_frac=0.4, bg_density_ratio=0.045)
        self.assertEqual(len(wind_dirs), 5)
        self.assertGreater(n_near, 0)
        self.assertIsNotNone(result)
        self.assertIn("q_t_per_year", result)
        self.assertGreater(result["q_t_per_year"], 0)

    def test_validate_sam_sounding_counts_same_order_of_magnitude(self):
        # This IS the task's required validation gate: simulated sounding
        # counts must not be off by an order of magnitude from real.
        rows = sim.validate_sam_sounding_counts(seed=99, n_repeats=2)
        self.assertEqual(len(rows), len(sim.SAM_VALIDATION_FACILITIES))
        for row in rows:
            self.assertLess(row["near_ratio_sim_over_real"], 10.0)
            self.assertGreater(row["near_ratio_sim_over_real"], 0.1)


if __name__ == "__main__":
    unittest.main()
