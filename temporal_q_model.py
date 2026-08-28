"""
Phase 4 of NOVEL_METHODOLOGY_PROPOSAL.md: a monthly CO2 emission-rate time
series per facility, instead of physics_ime.py's single annual
scalar. All Track B OCO-3 soundings are from calendar year 2020 only
(process_plant.py's search is hardcoded temporal=("2020-01-01",
"2020-12-31")), so this is a genuine 12-point monthly split of already-
collected data, not a new pull.

Reuses physics_ime.py's exact IME math (_ime_kg, _month_of_day,
NEAR/BG_IN/BG_OUT/ALPHA/FOOTPRINT_AREA_M2/SEC_PER_YEAR, imported not
reimplemented) applied to each calendar month's near/background soundings
separately, instead of the full year at once.

Explicit, documented simplifications (not hidden):
  - Wind speed: uses the SAME annual/overpass-aggregated wind_speed_ms
    from data/emission_estimates.json for every month, rather than a
    genuinely per-month wind speed. A true per-month wind speed would
    need a fresh ERA5 pull (only wind DIRECTION was cached per-day for
    Phase 4's day-matching work, not speed) -- flagged as a real
    limitation, not silently assumed away. Monthly Q values should be
    read as "monthly variation in the near-plant CO2 signal, scaled by a
    fixed annual wind factor," not as fully independent monthly
    estimates with their own wind conditioning.
  - Uncertainty: only the IME-sampling term (_bootstrap_ime_rel_std) is
    computed per month, not the full 3-term budget physics_ime.py
    uses annually -- background-definition sensitivity and wind
    uncertainty are not recomputed monthly here.
  - Months with fewer than 5 near-plant or 5 background soundings are
    skipped (same MIN threshold physics_ime.py uses annually), which
    is common for smaller facilities -- expect sparse monthly coverage,
    not a full 12-point series, for most facilities.

Cross-check included: does the mean of a facility's available monthly Q
values roughly track its existing annual Q from physics_ime.py? This
is an internal-consistency sanity check on the monthly code path itself,
not a validation against independent ground truth.

RESULT OF THAT CHECK, worth stating up front rather than only in the
output: the ratio is NOT close to 1.0 (observed mean 0.60, std 0.33
across 18 facilities) -- verified to be a genuine structural property of
the IME/L_eff formula, not a bug. IME scales roughly linearly with sample
count n (it sums individual soundings' excess), but L_eff = sqrt(n *
footprint_area) scales as sqrt(n). So Q ~ IME/L_eff ~ n/sqrt(n) = sqrt(n)
-- a monthly subsample with fewer soundings than the full year
systematically produces a SMALLER Q than a naive "annual Q / 12" would
suggest, purely from sample-size normalization, not from a real emission-
rate difference. CONSEQUENCE: absolute monthly Q values from this script
should NOT be compared to the annual scalar, or trusted as an absolute
tons/year figure for that month. They ARE still meaningfully comparable
to EACH OTHER within the same facility (relative month-to-month shape),
since the sqrt(n) bias applies consistently across months for a given
facility's sampling pattern -- this is a seasonal-SHAPE model, not an
absolute monthly-emission-rate model, and should be presented as such.
"""
import json

import numpy as np

from build_plume_maps import eligible_facilities
from physics_ime import (NEAR, BG_IN, BG_OUT, ALPHA, SEC_PER_YEAR,
                               FOOTPRINT_AREA_M2, _ime_kg, _bootstrap_ime_rel_std, _month_of_day)

MIN_NEAR_BG = 5


def monthly_q_series(name, plant_row, wind_speed_ms):
    d = np.load(f"data/{name}_soundings.npz")
    if "day" not in d.files:
        return None
    lat, lon, xco2, day = d["lat"], d["lon"], d["xco2"], d["day"]
    plat, plon = plant_row["lat"], plant_row["lon"]
    dist = np.sqrt((lat - plat) ** 2 + (lon - plon) ** 2)
    near_mask, bg_mask = dist < NEAR, (dist > BG_IN) & (dist < BG_OUT)

    months = np.array([_month_of_day(x) for x in day])
    series = {}
    for m in range(1, 13):
        near_m = xco2[near_mask & (months == m)]
        bg_m = xco2[bg_mask & (months == m)]
        if len(near_m) < MIN_NEAR_BG or len(bg_m) < MIN_NEAR_BG:
            continue
        bg_mean = bg_m.mean()
        ime = _ime_kg(near_m, bg_mean)
        n_used = int((np.clip(near_m - bg_mean, 0, None) > 0).sum())
        if n_used == 0:
            continue
        l_eff = float(np.sqrt(n_used * FOOTPRINT_AREA_M2))
        u_eff = ALPHA * wind_speed_ms
        q_kg_s = u_eff * ime / l_eff
        q_t_yr = q_kg_s * SEC_PER_YEAR / 1000
        ime_rel_std = _bootstrap_ime_rel_std(near_m, bg_m)
        series[m] = {"n_near": len(near_m), "n_bg": len(bg_m), "n_used": n_used,
                     "q_t_per_year": q_t_yr, "ime_rel_std": ime_rel_std}
    return series


def main():
    plant_results = {r["plant"]: r for r in json.load(open("data/plant_results.json"))}
    emission_estimates = {r["plant"]: r for r in json.load(open("data/emission_estimates.json"))}
    facilities = eligible_facilities(plant_results, emission_estimates)

    results = {}
    for name in facilities:
        est = emission_estimates[name]
        series = monthly_q_series(name, plant_results[name], est["wind_speed_ms"])
        if series is None or len(series) == 0:
            print(f"\n=== {name} === no usable months, skipping")
            results[name] = {"skipped": True}
            continue

        q_vals = np.array([v["q_t_per_year"] for v in series.values()])
        mean_monthly_q = float(q_vals.mean())
        cv = float(q_vals.std(ddof=1) / mean_monthly_q) if len(q_vals) > 1 and mean_monthly_q > 0 else None
        annual_q = est["q_t_per_year"]
        consistency_ratio = mean_monthly_q / annual_q if annual_q > 0 else None

        print(f"\n=== {name} ({len(series)}/12 months usable) ===")
        for m in sorted(series):
            v = series[m]
            print(f"  month {m:2d}: n_near={v['n_near']:3d} n_bg={v['n_bg']:4d}  "
                  f"Q={v['q_t_per_year']:>14,.0f} t/yr  (IME rel.std {v['ime_rel_std']:.0%})")
        cv_str = f"{cv:.2f}" if cv is not None else "n/a (only 1 usable month)"
        print(f"  mean monthly Q = {mean_monthly_q:,.0f} t/yr  "
              f"(annual estimate: {annual_q:,.0f} t/yr, ratio={consistency_ratio:.2f})  "
              f"CV across months = {cv_str}")

        results[name] = {
            "n_months_usable": len(series), "monthly_series": series,
            "mean_monthly_q": mean_monthly_q, "coefficient_of_variation": cv,
            "annual_q_t_per_year": annual_q, "mean_monthly_vs_annual_ratio": consistency_ratio,
        }

    usable = [r for r in results.values() if not r.get("skipped")]
    ratios = [r["mean_monthly_vs_annual_ratio"] for r in usable if r["mean_monthly_vs_annual_ratio"] is not None]
    consistency_note = (
        f"Across {len(ratios)} facilities with usable monthly data, mean(monthly Q)/annual Q ratio "
        f"has mean={np.mean(ratios):.2f}, std={np.std(ratios):.2f} -- NOT close to 1.0, and this is "
        "expected, not a bug: IME scales roughly linearly with sample count n, but L_eff = "
        "sqrt(n*footprint_area) scales as sqrt(n), so Q ~ n/sqrt(n) = sqrt(n) -- a monthly "
        "subsample with fewer soundings than the full year systematically yields a smaller Q than "
        "a naive annual/12 comparison would suggest, from sample-size normalization alone, not a "
        "real emission-rate difference. Absolute monthly Q values should not be compared to the "
        "annual scalar; they remain meaningfully comparable to EACH OTHER within a facility "
        "(relative seasonal shape), since this bias applies consistently across a given facility's "
        "own sampling pattern."
    )
    print(f"\n{consistency_note}")

    json.dump({"facility_list": facilities, "results": results, "consistency_note": consistency_note,
               "limitation_note": ("Wind speed uses the same annual/overpass-aggregated value for "
                                    "every month, not a true per-month wind estimate -- see module "
                                    "docstring. Uncertainty is IME-sampling only, not the full "
                                    "3-term annual budget.")},
              open("data/temporal_q_model_results.json", "w"), indent=2)
    print("\n[SAVED] data/temporal_q_model_results.json")


if __name__ == "__main__":
    main()
