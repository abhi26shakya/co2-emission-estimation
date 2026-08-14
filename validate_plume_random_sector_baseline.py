"""
Robustness follow-up to Phase 2 (NOVEL_METHODOLOGY_PROPOSAL.md Sec 9),
requested explicitly before the spatial-consistency finding is trusted.

Phase 2 found that soundings in the plume model's predicted downwind
sector show significantly higher XCO2 excess than soundings outside it,
for all three prototype facilities. Two confounds could produce that
result WITHOUT the plume model's wind-direction reasoning being correct:

  1. A pure distance-from-source effect: near-source soundings read
     higher regardless of true direction, and if they happen to
     disproportionately land in the sector Phase 2 tested (by chance,
     independent of wind), the test would show a spurious effect.
  2. OCO-3 swath/orbital-geometry sampling bias: this project has already
     found (diagnose_shrisingajimalwa.py) that OCO-3's narrow swath can
     produce non-uniform azimuthal or seasonal sampling around a given
     plant -- if soundings are NOT azimuthally uniform for reasons having
     nothing to do with the plume, some 90-degree-wide sectors could look
     systematically different from others by construction, before any
     physics is involved.

This script controls for BOTH at once: for each facility, it repeats
Phase 2's exact sector test (same soundings, same excess values, same
+/-45-degree half-width) but with the sector centered on many random
bearings instead of the plume-predicted one, building an empirical null
distribution of z-scores. If the true wind-predicted sector's z-score is
not meaningfully more extreme than this null distribution, the apparent
spatial consistency is likely explainable by geometry/sampling alone, not
real plume physics -- and that must be reported honestly, not hidden.

Reuses Phase 2's own sounding-loading and significance-check functions
directly (not reimplemented), so the null-distribution sectors are tested
under the exact same statistical procedure as the true result they're
being compared against.
"""
import json

import numpy as np

from validate_plume_spatial_consistency import (
    PROTOTYPE_FACILITIES, PLUME_SECTOR_HALF_WIDTH_DEG,
    load_and_project_soundings, sector_significance_check,
)

N_RANDOM_TRIALS = 2000
RNG_SEED = 0


def random_sector_z(bearing_deg, excess_ppm, sector_center_deg, half_width_deg):
    angular_diff = np.abs((bearing_deg - sector_center_deg + 180) % 360 - 180)
    in_sector = angular_diff < half_width_deg
    if in_sector.sum() < 5 or (~in_sector).sum() < 5:
        return None
    result = sector_significance_check(excess_ppm[in_sector], excess_ppm[~in_sector])
    return result["z_score"]


def main():
    plant_results = {r["plant"]: r for r in json.load(open("data/plant_results.json"))}
    rng = np.random.default_rng(RNG_SEED)
    extent_km = 30.0

    results = {}
    for name in PROTOTYPE_FACILITIES:
        pr = plant_results[name]
        wind_from_deg = (pr["wind_deg"] + 180.0) % 360.0
        plume_travel_deg = (wind_from_deg + 180.0) % 360.0

        proj = load_and_project_soundings(name, pr, extent_km)
        n = proj["n_within_extent"]
        print(f"\n=== {name} (N={n} soundings) ===")
        if n < 10:
            print("  too few soundings, skipping")
            results[name] = {"skipped": True}
            continue

        bearing_deg = (np.degrees(np.arctan2(proj["east_km"], proj["north_km"])) + 360) % 360
        excess_ppm = proj["excess_ppm"]

        true_z = random_sector_z(bearing_deg, excess_ppm, plume_travel_deg, PLUME_SECTOR_HALF_WIDTH_DEG)

        null_z = []
        random_centers = rng.uniform(0, 360, size=N_RANDOM_TRIALS)
        for center in random_centers:
            z = random_sector_z(bearing_deg, excess_ppm, center, PLUME_SECTOR_HALF_WIDTH_DEG)
            if z is not None and np.isfinite(z):
                null_z.append(z)
        null_z = np.array(null_z)

        # one-sided: our hypothesis is specifically that the wind-aligned
        # sector reads HIGHER, matching Phase 2's own one-sided z>2 framing
        empirical_p = float(np.mean(null_z >= true_z)) if len(null_z) else float("nan")
        percentile = float(100 * np.mean(null_z < true_z)) if len(null_z) else float("nan")

        print(f"  true (wind-predicted) sector z = {true_z:.2f}")
        print(f"  null distribution (n={len(null_z)} random sectors): "
              f"mean={null_z.mean():+.2f}  std={null_z.std():.2f}  "
              f"5th/95th pct=[{np.percentile(null_z,5):+.2f}, {np.percentile(null_z,95):+.2f}]")
        print(f"  true z is at the {percentile:.1f}th percentile of the null distribution "
              f"(empirical one-sided p = {empirical_p:.4f})")

        results[name] = {
            "n_soundings": n, "true_z": true_z,
            "null_mean_z": float(null_z.mean()), "null_std_z": float(null_z.std()),
            "null_5th_pct": float(np.percentile(null_z, 5)), "null_95th_pct": float(np.percentile(null_z, 95)),
            "true_z_percentile_in_null": percentile, "empirical_p_one_sided": empirical_p,
            "distinguishable_from_random_geometry": bool(empirical_p < 0.05),
        }

    conclusion = (
        f"{sum(1 for r in results.values() if r.get('distinguishable_from_random_geometry'))}/"
        f"{len([r for r in results.values() if not r.get('skipped')])} facilities show a "
        "wind-predicted-sector effect distinguishable from the random-sector null distribution "
        "at p<0.05. Where the true z-score sits well outside the null distribution's spread, the "
        "Phase-2 spatial-consistency finding survives this robustness check -- it is not fully "
        "explained by pure distance-from-source or azimuthal sampling geometry alone. Where it "
        "does not, that facility's Phase-2 result should be treated as inconclusive, not "
        "confirmed, and reported that way."
    )
    print(f"\n{conclusion}")

    json.dump({"n_random_trials": N_RANDOM_TRIALS, "sector_half_width_deg": PLUME_SECTOR_HALF_WIDTH_DEG,
               "results": results, "conclusion": conclusion},
              open("data/plume_maps/random_sector_baseline_results.json", "w"), indent=2)
    print("\n[SAVED] data/plume_maps/random_sector_baseline_results.json")


if __name__ == "__main__":
    main()
