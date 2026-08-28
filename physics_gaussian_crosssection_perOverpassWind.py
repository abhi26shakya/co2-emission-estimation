"""
Week 18, task 2: does per-overpass wind DIRECTION -- as opposed to the
single annual-mean direction physics_gaussian_crosssection.py used
uniformly across all 30 plants -- specifically improve the Gaussian
cross-section fit? WEEK15_LOG.txt and RESEARCH_PAPER.md's §5.2.8 both name
this as an untested limitation, not a fixed one: per-overpass direction
(data/daily_wind/<Plant>_daily_wind.json) exists for only 18/30 plants,
which is why the original run used the coarser annual mean for all 30
instead.

This script reruns ONLY those 18 plants, replacing the single per-plant
rotation angle with a PER-SOUNDING one: each near-plant sounding is
rotated into the downwind/crosswind frame using its own overpass day's
cached wind direction, not the plant's annual mean. Soundings whose day
has no cached direction are dropped, not fallback-rotated -- a fallback
would silently reintroduce the annual mean and defeat the point of the
test. Everything else (background computation, the sigma cap, MIN_DOWNWIND,
wind-SPEED matching for U_eff) is reused unchanged from
physics_gaussian_crosssection.py, so this isolates direction specifically.

Does NOT touch data/gaussian_crosssection_results.json or
data/gaussian_crosssection_vs_ime.png -- writes only
data/gaussian_crosssection_perOverpassWind_results.json.
"""
import json
import os

import numpy as np
from scipy.optimize import curve_fit

import baseline_capacity as bc
import physics_gaussian_crosssection as gx
import physics_ime as pg

OUT_JSON = "data/gaussian_crosssection_perOverpassWind_results.json"
ORIGINAL_JSON = "data/gaussian_crosssection_results.json"
DAILY_WIND_DIR = "data/daily_wind"

# the 18 plants with a cached per-overpass wind-direction series
ELIGIBLE_PLANTS = sorted(
    f[: -len("_daily_wind.json")] for f in os.listdir(DAILY_WIND_DIR)
    if f.endswith("_daily_wind.json")
)


def fit_plant_per_overpass_wind(name, plant_row, wind_speed_series, daily_direction):
    """Same pipeline as physics_gaussian_crosssection.fit_plant(), except the
    downwind/crosswind rotation uses each sounding's OWN overpass-day wind
    direction (from `daily_direction`) instead of one plant-wide annual-mean
    angle. Soundings whose day isn't in `daily_direction` are excluded before
    the downwind split, not rotated with a fallback angle."""
    npz_path = gx.pg.NPZ_PATHS.get(name, f"data/{name}_soundings.npz")
    try:
        d = np.load(npz_path)
    except FileNotFoundError:
        print(f"[{name}] no saved soundings, skipping")
        return {"plant": name, "status": "skipped", "reason": "no soundings file"}

    lat, lon, xco2 = d["lat"], d["lon"], d["xco2"]
    has_days = "day" in d.files
    if not has_days:
        print(f"[{name}] no per-sounding day field, can't match per-overpass direction, skipping")
        return {"plant": name, "status": "skipped", "reason": "no per-sounding day field"}
    day = d["day"]

    plat, plon = plant_row["lat"], plant_row["lon"]
    dist = np.sqrt((lat - plat) ** 2 + (lon - plon) ** 2)

    near_mask = dist < pg.NEAR
    bg_mask = (dist > pg.BG_IN) & (dist < pg.BG_OUT)
    if near_mask.sum() < 5 or bg_mask.sum() < 5:
        print(f"[{name}] too few near/bg soundings, skipping")
        return {"plant": name, "status": "skipped", "reason": "too few near/bg soundings"}
    bg_mean = float(xco2[bg_mask].mean())

    xco2_near = xco2[near_mask]
    day_near = day[near_mask]
    lat_near = lat[near_mask]
    lon_near = lon[near_mask]

    # per-sounding direction match: only soundings whose overpass day has a
    # cached wind direction are used at all
    has_direction = np.array([str(int(dy)) in daily_direction for dy in day_near])
    n_with_direction = int(has_direction.sum())
    if n_with_direction < gx.MIN_DOWNWIND:
        print(f"[{name}] too few soundings with a cached per-overpass wind direction "
              f"({n_with_direction} < {gx.MIN_DOWNWIND}), skipping")
        return {"plant": name, "status": "skipped",
                "reason": f"too few soundings with per-overpass wind direction "
                          f"({n_with_direction} < {gx.MIN_DOWNWIND})",
                "n_soundings_with_direction_match": n_with_direction}

    xco2_m = xco2_near[has_direction]
    day_m = day_near[has_direction]
    lat_m = lat_near[has_direction]
    lon_m = lon_near[has_direction]
    theta = np.radians(np.array([daily_direction[str(int(dy))] for dy in day_m]))

    m_per_deg_lon = gx.M_PER_DEG_LAT * np.cos(np.radians(plat))
    dx_m = (lon_m - plon) * m_per_deg_lon
    dy_m = (lat_m - plat) * gx.M_PER_DEG_LAT
    # per-sounding rotation, one theta per sounding rather than one for the plant
    x_dw = dx_m * np.sin(theta) + dy_m * np.cos(theta)
    y_cw = dx_m * np.cos(theta) - dy_m * np.sin(theta)

    downwind = x_dw > 0
    n_downwind = int(downwind.sum())
    n_days_used = int(len(np.unique(day_m[downwind]))) if n_downwind > 0 else 0

    if n_downwind < gx.MIN_DOWNWIND:
        print(f"[{name}] too few downwind soundings ({n_downwind} < {gx.MIN_DOWNWIND}), skipping")
        return {"plant": name, "status": "skipped",
                "reason": f"too few downwind soundings ({n_downwind} < {gx.MIN_DOWNWIND})",
                "n_soundings_with_direction_match": n_with_direction,
                "n_downwind_soundings": n_downwind, "n_overpass_days_used": n_days_used}

    excess_ppm = np.clip(xco2_m[downwind] - bg_mean, 0, None)
    excess_mass = pg.column_mass_enhancement(excess_ppm)
    y = y_cw[downwind]

    if int(np.count_nonzero(excess_mass > 0)) < gx.MIN_DOWNWIND:
        print(f"[{name}] too few downwind soundings above background, skipping")
        return {"plant": name, "status": "skipped",
                "reason": "too few downwind soundings above background",
                "n_soundings_with_direction_match": n_with_direction,
                "n_downwind_soundings": n_downwind, "n_overpass_days_used": n_days_used}

    A0 = float(excess_mass.max())
    sigma0 = min(float(np.std(y)) or 1000.0, gx.MAX_SIGMA_M / 2)
    try:
        popt, _ = curve_fit(gx._gaussian, y, excess_mass, p0=[A0, sigma0],
                             bounds=([0.0, 1.0], [np.inf, gx.MAX_SIGMA_M]), maxfev=5000)
    except (RuntimeError, ValueError) as e:
        print(f"[{name}] curve fit did not converge, skipping")
        return {"plant": name, "status": "skipped",
                "reason": f"curve fit did not converge ({str(e)[:60]})",
                "n_soundings_with_direction_match": n_with_direction,
                "n_downwind_soundings": n_downwind, "n_overpass_days_used": n_days_used}

    A_fit, sigma_fit = float(popt[0]), float(popt[1])
    if not (A_fit > 0 and sigma_fit > 0 and np.isfinite(A_fit) and np.isfinite(sigma_fit)):
        print(f"[{name}] fit produced invalid parameters, skipping")
        return {"plant": name, "status": "skipped", "reason": "fit produced invalid parameters",
                "n_soundings_with_direction_match": n_with_direction,
                "n_downwind_soundings": n_downwind, "n_overpass_days_used": n_days_used}
    if sigma_fit > 0.99 * gx.MAX_SIGMA_M:
        print(f"[{name}] fit degenerate (sigma pinned at cap), skipping")
        return {"plant": name, "status": "skipped",
                "reason": f"degenerate fit: sigma pinned at cap ({gx.MAX_SIGMA_M/1000:.0f}km)",
                "n_soundings_with_direction_match": n_with_direction,
                "n_downwind_soundings": n_downwind, "n_overpass_days_used": n_days_used}

    # wind SPEED matching is unchanged from the original script -- only
    # direction is under test here
    uniq_days = np.unique(day_m[downwind])
    matched = np.array(
        [wind_speed_series[int(dy)] for dy in uniq_days if int(dy) in wind_speed_series], dtype=float)
    if matched.size >= 3:
        wind_speed = float(matched.mean())
        wind_mode = "per-overpass"
    elif wind_speed_series:
        wind_speed = float(np.mean(list(wind_speed_series.values())))
        wind_mode = "annual-mean (fallback)"
    else:
        print(f"[{name}] no wind speed data available, skipping")
        return {"plant": name, "status": "skipped", "reason": "no wind speed data available",
                "n_soundings_with_direction_match": n_with_direction,
                "n_downwind_soundings": n_downwind, "n_overpass_days_used": n_days_used}

    u_eff = pg.ALPHA * wind_speed
    q_kg_s = u_eff * A_fit * sigma_fit * np.sqrt(2 * np.pi)
    q_t_yr = q_kg_s * pg.SEC_PER_YEAR / 1000
    r2 = gx._fit_r2(y, excess_mass, A_fit, sigma_fit)

    result = {
        "plant": name,
        "status": "fit",
        "wind_direction_mode": "per-overpass",
        "n_soundings_with_direction_match": n_with_direction,
        "n_downwind_soundings": n_downwind,
        "n_overpass_days_used": n_days_used,
        "bg_mean_ppm": bg_mean,
        "gaussian_amplitude_kg_m2": A_fit,
        "gaussian_sigma_m": sigma_fit,
        "fit_r2": r2,
        "wind_speed_ms": wind_speed,
        "wind_speed_mode": wind_mode,
        "u_eff_ms": u_eff,
        "q_kg_s": q_kg_s,
        "q_t_per_year": q_t_yr,
    }
    print(f"[{name}] n_direction_matched={n_with_direction} n_downwind={n_downwind} "
          f"days_used={n_days_used} sigma={sigma_fit/1000:.2f}km R2={r2:.2f} "
          f"U_eff={u_eff:.2f}m/s ({wind_mode}) -> Q = {q_t_yr:,.0f} t/yr")
    return result


def main():
    print(f"Eligible plants (cached per-overpass wind direction): {len(ELIGIBLE_PLANTS)}")
    print(f"  {ELIGIBLE_PLANTS}\n")

    plant_rows = {r["plant"]: r for r in json.load(open("data/plant_results.json"))}
    cea = json.load(open("data/cea_ground_truth_2020_21.json"))["facilities"]
    ime = {r["plant"]: r for r in json.load(open("data/emission_estimates.json"))}
    original = {r["plant"]: r for r in json.load(open(ORIGINAL_JSON))["per_plant"]}

    results = []
    for name in ELIGIBLE_PLANTS:
        row = plant_rows[name]
        wind_speed_series = gx._load_wind_series(name, row["lat"], row["lon"])
        daily_direction = json.load(open(f"{DAILY_WIND_DIR}/{name}_daily_wind.json"))
        results.append(fit_plant_per_overpass_wind(name, row, wind_speed_series, daily_direction))

    fitted = {r["plant"]: r for r in results if r["status"] == "fit"}
    skipped = [r["plant"] for r in results if r["status"] == "skipped"]
    print(f"\n=== {len(fitted)}/{len(results)} of the 18 eligible plants fit "
          f"with per-overpass wind direction, {len(skipped)} skipped ===")

    # --- 1. fit-status comparison against the original annual-mean-direction run ---
    status_comparison = []
    for name in ELIGIBLE_PLANTS:
        old_status = "fit" if original.get(name, {}).get("status") == "fit" else "skipped"
        new_status = "fit" if name in fitted else "skipped"
        status_comparison.append({
            "plant": name, "annual_mean_direction": old_status,
            "per_overpass_direction": new_status,
            "changed": old_status != new_status,
        })
    n_unlocked = sum(1 for r in status_comparison
                      if r["annual_mean_direction"] == "skipped" and r["per_overpass_direction"] == "fit")
    n_newly_skipped = sum(1 for r in status_comparison
                           if r["annual_mean_direction"] == "fit" and r["per_overpass_direction"] == "skipped")
    print(f"\n=== Fit-status comparison (N={len(ELIGIBLE_PLANTS)} eligible plants) ===")
    for r in status_comparison:
        flag = "  <-- CHANGED" if r["changed"] else ""
        print(f"  {r['plant']:20s} annual={r['annual_mean_direction']:8s} "
              f"per-overpass={r['per_overpass_direction']:8s}{flag}")
    print(f"  Unlocked by per-overpass direction (skip->fit): {n_unlocked}")
    print(f"  Newly skipped (fit->skip): {n_newly_skipped}")

    # --- 2. for plants that fit BOTH ways, does per-overpass direction move Q closer to CEA? ---
    both_fit = [name for name in ELIGIBLE_PLANTS
                if name in fitted and original.get(name, {}).get("status") == "fit" and name in cea]
    direct_comparison = []
    for name in both_fit:
        cea_val = cea[name]["abs_emissions_t_co2"]
        q_annual = original[name]["q_t_per_year"]
        q_perop = fitted[name]["q_t_per_year"]
        log_annual = float(np.log(q_annual / cea_val))
        log_perop = float(np.log(q_perop / cea_val))
        direct_comparison.append({
            "plant": name, "cea_t_per_year": cea_val,
            "q_annual_mean_direction": q_annual, "q_per_overpass_direction": q_perop,
            "abs_log_ratio_annual_mean": abs(log_annual),
            "abs_log_ratio_per_overpass": abs(log_perop),
            "moved_closer_to_cea": abs(log_perop) < abs(log_annual),
        })
    print(f"\n=== Direct comparison, fit under both methods (N={len(both_fit)}) ===")
    n_closer = sum(1 for r in direct_comparison if r["moved_closer_to_cea"])
    for r in direct_comparison:
        direction = "closer" if r["moved_closer_to_cea"] else "further"
        print(f"  {r['plant']:20s} |log ratio| annual={r['abs_log_ratio_annual_mean']:.3f} "
              f"per-overpass={r['abs_log_ratio_per_overpass']:.3f}  ({direction})")
    print(f"  Moved closer to CEA: {n_closer}/{len(both_fit)}")

    # --- 3. does this change the §5.2.8 verdict? LOO R^2 vs CEA, restricted to a
    # consistent plant set across all three (IME, annual-mean Gaussian, per-overpass
    # Gaussian) so the comparison is apples-to-apples, not different N's ---
    def loo_vs_cea(plants, q_vals, label):
        cea_vals = np.array([cea[p]["abs_emissions_t_co2"] for p in plants])
        if len(plants) < 4:
            return {"label": label, "n": len(plants), "note": "too few plants for LOO regression"}
        pred = bc.loo_linear_log_predictions(np.log(q_vals).reshape(-1, 1), np.log(cea_vals))
        return bc.evaluate(np.log(cea_vals), pred, plants, label)

    perop_plants = [p for p in fitted if p in ime and p in cea]
    perop_q = np.array([fitted[p]["q_t_per_year"] for p in perop_plants])
    ime_on_perop_set_q = np.array([ime[p]["q_t_per_year"] for p in perop_plants])
    annual_on_perop_set_plants = [p for p in perop_plants if original.get(p, {}).get("status") == "fit"]
    annual_on_perop_set_q = np.array([original[p]["q_t_per_year"] for p in annual_on_perop_set_plants])
    ime_on_annual_set_q = np.array([ime[p]["q_t_per_year"] for p in annual_on_perop_set_plants])

    loo_perop = loo_vs_cea(perop_plants, perop_q,
                            f"Gaussian, per-overpass direction (N={len(perop_plants)})")
    loo_ime_on_perop_set = loo_vs_cea(perop_plants, ime_on_perop_set_q,
                                       f"IME, same N={len(perop_plants)} plants")
    loo_annual_on_shared_set = loo_vs_cea(
        annual_on_perop_set_plants, annual_on_perop_set_q,
        f"Gaussian, annual-mean direction, restricted to N={len(annual_on_perop_set_plants)} "
        f"plants fit both ways")
    loo_ime_on_shared_set = loo_vs_cea(
        annual_on_perop_set_plants, ime_on_annual_set_q,
        f"IME, same N={len(annual_on_perop_set_plants)} plants fit both ways")

    print(f"\n=== LOO R^2 vs CEA (baseline_capacity.py methodology) ===")
    for r in (loo_perop, loo_ime_on_perop_set, loo_annual_on_shared_set, loo_ime_on_shared_set):
        if "loo_r2" in r:
            print(f"  {r['label']:65s} R^2={r['loo_r2']:.3f}  MAE(log)={r['loo_mae_log']:.3f}  "
                  f"within_2x={r['within_2x']}/{r['n']}")
        else:
            print(f"  {r['label']:65s} {r.get('note')}")

    out = {
        "method": "gaussian cross-sectional flux, per-overpass wind direction "
                  "(Week 18 test of physics_gaussian_crosssection.py's annual-mean-direction limitation)",
        "eligible_plants": ELIGIBLE_PLANTS,
        "n_eligible": len(ELIGIBLE_PLANTS),
        "n_fit": len(fitted),
        "n_skipped": len(skipped),
        "per_plant": results,
        "fit_status_comparison": {
            "rows": status_comparison,
            "n_unlocked_skip_to_fit": n_unlocked,
            "n_newly_skipped_fit_to_skip": n_newly_skipped,
        },
        "direct_comparison_fit_both_ways": {
            "n": len(both_fit),
            "rows": direct_comparison,
            "n_moved_closer_to_cea": n_closer,
        },
        "loo_vs_cea": {
            "gaussian_per_overpass_direction": loo_perop,
            "ime_same_plant_set": loo_ime_on_perop_set,
            "gaussian_annual_mean_direction_shared_subset": loo_annual_on_shared_set,
            "ime_shared_subset": loo_ime_on_shared_set,
        },
    }
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"\n[SAVED] {OUT_JSON}")


if __name__ == "__main__":
    main()
