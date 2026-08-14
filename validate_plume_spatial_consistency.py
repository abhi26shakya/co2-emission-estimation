"""
Phase 2 of NOVEL_METHODOLOGY_PROPOSAL.md: does the plume model's predicted
spatial pattern agree with WHERE the real OCO-3 soundings actually
observed elevated CO2, across every eligible processed facility (scaled
up from the original 3-facility prototype after the random-sector
robustness check showed N=3 wasn't enough statistical power to draw a
confident conclusion either way, see Sec 10)?

This is explicitly a SPATIAL CONSISTENCY check, not a pixel-accuracy
benchmark (see NOVEL_METHODOLOGY_PROPOSAL.md Sec 4.2's honesty
constraint): OCO-3's sparse per-sounding footprint (~1.6km, non-daily
revisit) cannot support a claim that the plume map is pixel-accurate.
What CAN be tested honestly: among the soundings actually collected near
each plant, do the ones showing higher XCO2 excess over background tend
to sit in the direction the plume model predicts the plume points,
more than chance would produce?

Method (deliberately simple, consistent with this project's established
small-N statistical discipline -- no new dependency, no scipy):
  1. For each eligible facility, load its real soundings (lat, lon,
     xco2), restricted to the plume grid's spatial extent (30km).
  2. Compute each sounding's excess over the SAME background definition
     physics_gaussian.py already uses (NEAR/BG_IN/BG_OUT, imported
     directly, not reimplemented) -- for consistency with the rest of
     this project's Track B methodology.
  3. Evaluate the plume model's predicted concentration at each
     sounding's exact (east_km, north_km) location via
     plume_model.concentration_at_locations() (Phase 2's point-evaluation
     addition to plume_model.py).
  4. Two independent tests, both reported honestly regardless of outcome:
     (a) Pearson correlation between predicted plume concentration and
         observed XCO2 excess across all near-plant soundings.
     (b) A directional test: split soundings into "in the plume's
         downwind sector" (within +/-45 deg of the plume axis) vs.
         "outside the sector," and compare mean excess between the two
         groups (same z-score-style significance check pattern as
         diagnose_negative_enhancement.py's significance_check(), reused
         for consistency of statistical approach across this project).
"""
import json

import numpy as np

import plume_model as pm
from physics_gaussian import NEAR, BG_IN, BG_OUT
from build_plume_maps import eligible_facilities

# Same facility list build_plume_maps.py used (every facility with a wind
# direction, a Track B estimate, and a soundings file) -- computed once
# here, at module level, so this script and validate_plume_random_sector_
# baseline.py (which imports ALL_FACILITIES from here) can't disagree on
# scope. Matches this project's existing convention of module-level data
# loading (e.g. reliability_model.py, lofo_recall_correlates.py).
ALL_FACILITIES = eligible_facilities(
    {r["plant"]: r for r in json.load(open("data/plant_results.json"))},
    {r["plant"]: r for r in json.load(open("data/emission_estimates.json"))},
)
PLUME_SECTOR_HALF_WIDTH_DEG = 45.0
KM_PER_DEG_LAT = 111.0


def load_and_project_soundings(name, plant_row, extent_km):
    d = np.load(f"data/{name}_soundings.npz")
    lat, lon, xco2 = d["lat"], d["lon"], d["xco2"]
    plat, plon = plant_row["lat"], plant_row["lon"]

    # Same background definition as physics_gaussian.py, imported not
    # reimplemented, so the "excess" values here are directly comparable
    # to the ones Track B's own IME estimate is built from.
    dist_deg = np.sqrt((lat - plat) ** 2 + (lon - plon) ** 2)
    bg_mean = xco2[(dist_deg > BG_IN) & (dist_deg < BG_OUT)].mean()

    # Equirectangular local projection -- adequate at this scale (tens of
    # km), consistent with the small-angle approximation implicit in
    # physics_gaussian.py's degree-based near/background circles.
    km_per_deg_lon = KM_PER_DEG_LAT * np.cos(np.radians(plat))
    east_km = (lon - plon) * km_per_deg_lon
    north_km = (lat - plat) * KM_PER_DEG_LAT

    within_extent = np.hypot(east_km, north_km) < extent_km
    return {
        "east_km": east_km[within_extent], "north_km": north_km[within_extent],
        "excess_ppm": (xco2 - bg_mean)[within_extent],
        "bg_mean_ppm": float(bg_mean), "n_within_extent": int(within_extent.sum()),
    }


def sector_significance_check(in_sector, out_sector):
    diff = float(in_sector.mean() - out_sector.mean())
    se = float(np.hypot(in_sector.std(ddof=1) / np.sqrt(len(in_sector)),
                          out_sector.std(ddof=1) / np.sqrt(len(out_sector))))
    z = diff / se if se > 0 else float("nan")
    return {"n_in_sector": int(len(in_sector)), "n_out_sector": int(len(out_sector)),
            "mean_excess_in_sector_ppm": float(in_sector.mean()),
            "mean_excess_out_sector_ppm": float(out_sector.mean()),
            "diff_ppm": diff, "se_ppm": se, "z_score": z,
            "in_sector_significantly_higher": bool(z > 2)}


def main():
    plant_results = {r["plant"]: r for r in json.load(open("data/plant_results.json"))}
    emission_estimates = {r["plant"]: r for r in json.load(open("data/emission_estimates.json"))}
    extent_km = 30.0  # matches plume_model.plume_grid()'s default

    results = {}
    for name in ALL_FACILITIES:
        pr, est = plant_results[name], emission_estimates[name]
        wind_from_deg = (pr["wind_deg"] + 180.0) % 360.0

        proj = load_and_project_soundings(name, pr, extent_km)
        n = proj["n_within_extent"]
        print(f"\n=== {name} (N={n} soundings within {extent_km:.0f}km) ===")
        if n < 10:
            print("  too few soundings within the plume extent for a meaningful check, skipping")
            results[name] = {"n_within_extent": n, "skipped": True}
            continue

        predicted_conc = pm.concentration_at_locations(
            est["q_t_per_year"], est["wind_speed_ms"], wind_from_deg,
            proj["east_km"], proj["north_km"])

        # (a) correlation
        r = float(np.corrcoef(predicted_conc, proj["excess_ppm"])[0, 1])
        print(f"  Pearson r(predicted plume concentration, observed XCO2 excess) = {r:+.3f}")

        # (b) directional sector test
        bearing_deg = (np.degrees(np.arctan2(proj["east_km"], proj["north_km"])) + 360) % 360
        plume_travel_deg = (wind_from_deg + 180.0) % 360.0
        angular_diff = np.abs((bearing_deg - plume_travel_deg + 180) % 360 - 180)
        in_sector = angular_diff < PLUME_SECTOR_HALF_WIDTH_DEG

        if in_sector.sum() >= 5 and (~in_sector).sum() >= 5:
            sector = sector_significance_check(proj["excess_ppm"][in_sector], proj["excess_ppm"][~in_sector])
            print(f"  sector test: in-sector n={sector['n_in_sector']} mean={sector['mean_excess_in_sector_ppm']:+.3f} ppm  "
                  f"out-of-sector n={sector['n_out_sector']} mean={sector['mean_excess_out_sector_ppm']:+.3f} ppm  "
                  f"z={sector['z_score']:.2f}")
        else:
            sector = {"skipped": True, "reason": "too few soundings in one of the two sectors"}
            print(f"  sector test skipped: in-sector n={int(in_sector.sum())}, "
                  f"out-of-sector n={int((~in_sector).sum())} -- too few for a meaningful comparison")

        results[name] = {
            "n_within_extent": n, "bg_mean_ppm": proj["bg_mean_ppm"],
            "wind_from_deg": wind_from_deg, "plume_travel_deg": plume_travel_deg,
            "pearson_r_predicted_vs_observed": r,
            "sector_test": sector,
        }

    honesty_note = (
        "This is a spatial CONSISTENCY check, not a pixel-accuracy benchmark -- OCO-3's sparse "
        "per-sounding footprint (~1.6km, non-daily revisit) cannot support a claim that the plume "
        "map is pixel-accurate. A weak or null result for a facility with already-poor wind/CO2 "
        "alignment (see plant_results.json's wind_co2_diff_deg, computed independently in "
        "process_plant.py) is an EXPECTED, consistent finding, not a failure of this validation --  "
        "the plume model inherits whatever wind-alignment quality that facility already had."
    )
    print(f"\n{honesty_note}")

    json.dump({"facility_list": ALL_FACILITIES, "extent_km": extent_km,
               "sector_half_width_deg": PLUME_SECTOR_HALF_WIDTH_DEG,
               "results": results, "note": honesty_note},
              open("data/plume_maps/spatial_consistency_results.json", "w"), indent=2)
    print("\n[SAVED] data/plume_maps/spatial_consistency_results.json")


if __name__ == "__main__":
    main()
