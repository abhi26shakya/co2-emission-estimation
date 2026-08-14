"""
Day-matched follow-up to the Phase 2 spatial-consistency finding
(NOVEL_METHODOLOGY_PROPOSAL.md Sec 11): the earlier test compared every
sounding against ONE annual/overpass-averaged wind direction for the
whole facility, regardless of which specific day that sounding was taken.
This script instead compares each sounding against the ACTUAL ERA5 wind
direction on ITS OWN overpass day (fetch_daily_wind_direction.py), the
most likely fixable cause flagged in Sec 11 before concluding the
plume-direction hypothesis itself is wrong.

Method: for each sounding within the plume extent that has a day-matched
wind-direction entry, compute whether that sounding sits within +/-45deg
of THAT DAY's downwind direction (not one fixed facility-wide direction).
Soundings whose day has no wind match are excluded from this test (not
silently defaulted to the annual mean, which would defeat the purpose).

Robustness null: instead of testing against uniformly random single
bearings (which would erase the day-to-day directional structure this
test is specifically trying to use), each trial applies the SAME random
rotation offset to every sounding's own day-specific direction --
preserving the real day-to-day wind variability pattern while testing
whether the untouched (unrotated) alignment beats an arbitrary rigid
rotation of that same pattern.
"""
import json

import numpy as np

from build_plume_maps import eligible_facilities
from validate_plume_spatial_consistency import sector_significance_check
from physics_gaussian import NEAR, BG_IN, BG_OUT

PLUME_SECTOR_HALF_WIDTH_DEG = 45.0
KM_PER_DEG_LAT = 111.0
EXTENT_KM = 30.0
N_RANDOM_TRIALS = 2000
RNG_SEED = 0


def day_int_to_key(day_int):
    return str(int(day_int)) if day_int > 0 else None


def load_day_matched_soundings(name, plant_row, daily_wind):
    d = np.load(f"data/{name}_soundings.npz")
    if "day" not in d.files:
        return None
    lat, lon, xco2, day = d["lat"], d["lon"], d["xco2"], d["day"]
    plat, plon = plant_row["lat"], plant_row["lon"]

    dist_deg = np.sqrt((lat - plat) ** 2 + (lon - plon) ** 2)
    bg_mean = xco2[(dist_deg > BG_IN) & (dist_deg < BG_OUT)].mean()

    km_per_deg_lon = KM_PER_DEG_LAT * np.cos(np.radians(plat))
    east_km = (lon - plon) * km_per_deg_lon
    north_km = (lat - plat) * KM_PER_DEG_LAT
    within_extent = np.hypot(east_km, north_km) < EXTENT_KM

    east_km, north_km, excess_ppm, day = (east_km[within_extent], north_km[within_extent],
                                            (xco2 - bg_mean)[within_extent], day[within_extent])

    # per-sounding day-matched plume travel direction ("toward", same
    # convention the daily_wind cache stores -- see module docstring for
    # why no +180 conversion is needed here, unlike build_plume_maps.py
    plume_travel_deg = np.full(len(day), np.nan)
    for i, d_int in enumerate(day):
        key = day_int_to_key(d_int)
        if key in daily_wind:
            plume_travel_deg[i] = daily_wind[key]

    has_match = np.isfinite(plume_travel_deg)
    bearing_deg = (np.degrees(np.arctan2(east_km, north_km)) + 360) % 360

    return {
        "bearing_deg": bearing_deg[has_match], "excess_ppm": excess_ppm[has_match],
        "plume_travel_deg": plume_travel_deg[has_match],
        "n_total_within_extent": int(within_extent.sum()), "n_day_matched": int(has_match.sum()),
    }


def sector_z(bearing_deg, excess_ppm, travel_deg, half_width_deg):
    angular_diff = np.abs((bearing_deg - travel_deg + 180) % 360 - 180)
    in_sector = angular_diff < half_width_deg
    if in_sector.sum() < 5 or (~in_sector).sum() < 5:
        return None
    return sector_significance_check(excess_ppm[in_sector], excess_ppm[~in_sector])["z_score"]


def main():
    plant_results = {r["plant"]: r for r in json.load(open("data/plant_results.json"))}
    emission_estimates = {r["plant"]: r for r in json.load(open("data/emission_estimates.json"))}
    facilities = eligible_facilities(plant_results, emission_estimates)
    rng = np.random.default_rng(RNG_SEED)

    results = {}
    for name in facilities:
        try:
            daily_wind = json.load(open(f"data/daily_wind/{name}_daily_wind.json"))
        except FileNotFoundError:
            print(f"\n=== {name} === no daily wind cache, skipping")
            results[name] = {"skipped": True, "reason": "no daily wind cache"}
            continue

        proj = load_day_matched_soundings(name, plant_results[name], daily_wind)
        if proj is None:
            results[name] = {"skipped": True, "reason": "no day field in soundings"}
            continue

        print(f"\n=== {name} (N={proj['n_total_within_extent']} within extent, "
              f"{proj['n_day_matched']} day-matched) ===")
        if proj["n_day_matched"] < 15:
            print("  too few day-matched soundings, skipping")
            results[name] = {"n_day_matched": proj["n_day_matched"], "skipped": True,
                              "reason": "too few day-matched soundings"}
            continue

        true_z = sector_z(proj["bearing_deg"], proj["excess_ppm"], proj["plume_travel_deg"],
                           PLUME_SECTOR_HALF_WIDTH_DEG)
        if true_z is None:
            print("  true day-matched sector too small on one side, skipping")
            results[name] = {"n_day_matched": proj["n_day_matched"], "skipped": True,
                              "reason": "day-matched sector too small"}
            continue

        null_z = []
        for offset in rng.uniform(0, 360, size=N_RANDOM_TRIALS):
            rotated_travel = (proj["plume_travel_deg"] + offset) % 360
            z = sector_z(proj["bearing_deg"], proj["excess_ppm"], rotated_travel, PLUME_SECTOR_HALF_WIDTH_DEG)
            if z is not None and np.isfinite(z):
                null_z.append(z)
        null_z = np.array(null_z)
        empirical_p = float(np.mean(null_z >= true_z)) if len(null_z) else float("nan")
        percentile = float(100 * np.mean(null_z < true_z)) if len(null_z) else float("nan")

        print(f"  true (day-matched) sector z = {true_z:.2f}")
        print(f"  rotated-null distribution (n={len(null_z)}): mean={null_z.mean():+.2f}  std={null_z.std():.2f}")
        print(f"  true z at {percentile:.1f}th percentile (empirical one-sided p = {empirical_p:.4f})")

        results[name] = {
            "n_total_within_extent": proj["n_total_within_extent"],
            "n_day_matched": proj["n_day_matched"], "true_z": true_z,
            "null_mean_z": float(null_z.mean()), "null_std_z": float(null_z.std()),
            "true_z_percentile_in_null": percentile, "empirical_p_one_sided": empirical_p,
            "distinguishable_from_rotated_null": bool(empirical_p < 0.05),
        }

    tested = [r for r in results.values() if not r.get("skipped")]
    survived = sum(1 for r in tested if r.get("distinguishable_from_rotated_null"))
    conclusion = (
        f"{survived}/{len(tested)} facilities show a day-matched wind-direction sector effect "
        "distinguishable from a rotated-null baseline at p<0.05, using each sounding's own "
        "overpass-day wind direction instead of one annual/overpass-averaged direction per "
        "facility. Compare against the non-day-matched result (1/14) in "
        "data/plume_maps/random_sector_baseline_results.json."
    )
    print(f"\n{conclusion}")

    json.dump({"n_random_trials": N_RANDOM_TRIALS, "sector_half_width_deg": PLUME_SECTOR_HALF_WIDTH_DEG,
               "results": results, "conclusion": conclusion},
              open("data/plume_maps/day_matched_results.json", "w"), indent=2)
    print("\n[SAVED] data/plume_maps/day_matched_results.json")


if __name__ == "__main__":
    main()
