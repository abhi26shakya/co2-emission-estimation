"""
Week 13, experiment 1: does overpass coverage explain the failure?

Vindhyachal (17 overpass days) gives our best IME result (Q 36.7 Mt vs CEA
33.2 Mt, 11% error). Most other plants have 4-7 days and fail badly. This
tests whether day count alone is the cause: take the three plants with the
best coverage (Vindhyachal, Rihand, Sasan), randomly subsample down to
fewer overpass days, and see how the estimate degrades.

Method: for each plant and each n_days in N_DAYS_LIST, draw N_REPEATS
random subsets (without replacement) of that many overpass days, restrict
ALL soundings (near-plant and background) to those days, and rerun the
existing IME estimate (physics_gaussian.estimate_emission_rate_from_arrays)
on the subsample. Record the distribution of Q across repeats.

Reuses physics_gaussian.py's IME math directly rather than reimplementing
it -- see that module's estimate_emission_rate_from_arrays(), factored out
for exactly this purpose.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import physics_gaussian as pg

SEED = 42
N_REPEATS = 200
N_DAYS_LIST = [17, 15, 12, 10, 8, 6, 5, 4, 3]
PLANTS = ["Vindhyachal", "Rihand", "Sasan"]
STABLE_ERROR_TOL = 0.5      # |median error vs CEA| <= this counts as "stable"
STABLE_MIN_VALID_FRAC = 0.9  # >= this fraction of the 200 runs must succeed

# vindhyachal_soundings.npz is lowercase, unlike every other plant's file
pg.NPZ_PATHS["Vindhyachal"] = "data/vindhyachal_soundings.npz"


def _load_wind_series(name, lat, lon):
    """Cache the full-year ERA5 wind series per plant (data/<plant>_wind_series_cache.json)
    -- subsampling only needs to change which days' soundings are used, not
    refetch wind, and repeated Earth Engine calls are slow."""
    cache_path = f"data/{name}_wind_series_cache.json"
    if os.path.exists(cache_path):
        raw = json.load(open(cache_path))
        return {int(k): v for k, v in raw.items()}
    wind_series = pg._fetch_wind_series(lat, lon)
    json.dump(wind_series, open(cache_path, "w"))
    return wind_series


def run_plant(name, plant_row, cea_truth_t, rng):
    npz_path = pg.NPZ_PATHS.get(name, f"data/{name}_soundings.npz")
    d = np.load(npz_path)
    lat, lon, xco2, day = d["lat"], d["lon"], d["xco2"], d["day"]
    unique_days = np.unique(day)
    n_available = len(unique_days)

    wind_series = _load_wind_series(name, plant_row["lat"], plant_row["lon"])

    # full-coverage baseline (sanity check against data/emission_estimates.json)
    full = pg.estimate_emission_rate_from_arrays(name, lat, lon, xco2, day, plant_row, wind_series)
    print(f"\n=== {name}: {n_available} overpass days available, "
          f"full-coverage Q = {full['q_t_per_year']:,.0f} t/yr, CEA = {cea_truth_t:,.0f} t/yr ===")

    plant_result = {
        "plant": name,
        "n_days_available": int(n_available),
        "cea_truth_t_per_year": cea_truth_t,
        "full_coverage_q_t_per_year": full["q_t_per_year"],
        "by_n_days": [],
    }

    for n_days in N_DAYS_LIST:
        if n_days > n_available:
            continue
        qs = []
        for _ in range(N_REPEATS):
            sample_days = rng.choice(unique_days, size=n_days, replace=False)
            mask = np.isin(day, sample_days)
            r = pg.estimate_emission_rate_from_arrays(
                name, lat[mask], lon[mask], xco2[mask], day[mask], plant_row, wind_series
            )
            if r is not None:
                qs.append(r["q_t_per_year"])

        qs = np.array(qs)
        n_valid = len(qs)
        if n_valid == 0:
            print(f"  n_days={n_days:2d}: all {N_REPEATS} runs failed (too few soundings)")
            plant_result["by_n_days"].append({
                "n_days": n_days, "n_valid_runs": 0, "n_failed_runs": N_REPEATS,
                "median_q_t_per_year": None, "std_q_t_per_year": None,
                "p10_q_t_per_year": None, "p90_q_t_per_year": None,
                "error_vs_cea": None,
            })
            continue

        median_q = float(np.median(qs))
        std_q = float(np.std(qs, ddof=1)) if n_valid > 1 else 0.0
        p10, p90 = float(np.percentile(qs, 10)), float(np.percentile(qs, 90))
        error_vs_cea = median_q / cea_truth_t - 1
        print(f"  n_days={n_days:2d}: median Q={median_q:,.0f} t/yr  "
              f"spread(std)={std_q:,.0f} ({std_q/median_q:.0%})  "
              f"error vs CEA={error_vs_cea:+.0%}  "
              f"valid={n_valid}/{N_REPEATS}")
        plant_result["by_n_days"].append({
            "n_days": n_days,
            "n_valid_runs": n_valid,
            "n_failed_runs": N_REPEATS - n_valid,
            "median_q_t_per_year": median_q,
            "std_q_t_per_year": std_q,
            "p10_q_t_per_year": p10,
            "p90_q_t_per_year": p90,
            "error_vs_cea": error_vs_cea,
        })

    return plant_result


def make_plot(results, out_path="data/overpass_density_plot.png"):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5), sharey=False)
    if len(results) == 1:
        axes = [axes]
    for ax, res in zip(axes, results):
        rows = [r for r in res["by_n_days"] if r["median_q_t_per_year"] is not None]
        x = [r["n_days"] for r in rows]
        y = [r["median_q_t_per_year"] / 1e6 for r in rows]
        lo = [r["p10_q_t_per_year"] / 1e6 for r in rows]
        hi = [r["p90_q_t_per_year"] / 1e6 for r in rows]
        ax.plot(x, y, "o-", color="tab:blue", label="median Q (subsampled)")
        ax.fill_between(x, lo, hi, alpha=0.2, color="tab:blue", label="p10-p90 spread")
        ax.axhline(res["cea_truth_t_per_year"] / 1e6, color="tab:red", linestyle="--",
                    label="CEA truth")
        ax.set_title(f"{res['plant']} (n_available={res['n_days_available']})")
        ax.set_xlabel("overpass days used")
        ax.set_ylabel("Q (Mt CO2/yr)")
        ax.invert_xaxis()
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\n[SAVED] {out_path}")


def _is_stable(row):
    return (row["median_q_t_per_year"] is not None
            and abs(row["error_vs_cea"]) <= STABLE_ERROR_TOL
            and row["n_valid_runs"] >= STABLE_MIN_VALID_FRAC * N_REPEATS)


def _stable_threshold(res):
    """Smallest n_days still inside an UNBROKEN stable run down from full
    coverage (descending scan, stop at the first n_days that fails the
    stability criteria). A plain ascending "first n_days that happens to
    pass" scan is exploitable by noise -- Rihand's n_days=4 subsample beats
    the tolerance by luck even though n_days=5 and 6 fail it, so an
    ascending scan would report 4 as "stable" despite no real trend
    supporting that. Returns None if even full coverage isn't stable."""
    threshold = None
    for row in sorted(res["by_n_days"], key=lambda r: -r["n_days"]):
        if _is_stable(row):
            threshold = row["n_days"]
        else:
            break
    return threshold


if __name__ == "__main__":
    plant_rows = {r["plant"]: r for r in json.load(open("data/plant_results.json"))}
    cea = json.load(open("data/cea_ground_truth_2020_21.json"))["facilities"]
    q_correction = json.load(open("data/q_correction_model_results.json"))

    rng = np.random.default_rng(SEED)
    results = []
    for name in PLANTS:
        cea_truth_t = cea[name]["abs_emissions_t_co2"]
        results.append(run_plant(name, plant_rows[name], cea_truth_t, rng))

    out = {
        "seed": SEED,
        "n_repeats": N_REPEATS,
        "n_days_list": N_DAYS_LIST,
        "stable_error_tol": STABLE_ERROR_TOL,
        "stable_min_valid_frac": STABLE_MIN_VALID_FRAC,
        "plants": results,
    }
    out_path = "data/overpass_density_results.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\n[SAVED] {out_path}")

    make_plot(results)

    print("\n=== Threshold check ===")
    print(f"(stable = |median error vs CEA| <= {STABLE_ERROR_TOL:.0%} "
          f"and >= {STABLE_MIN_VALID_FRAC:.0%} of {N_REPEATS} runs succeed)")
    thresholds = {}
    for res in results:
        t = _stable_threshold(res)
        thresholds[res["plant"]] = t
        print(f"  {res['plant']}: smallest stable n_days = {t}")

    valid_thresholds = [t for t in thresholds.values() if t is not None]
    if valid_thresholds:
        threshold = max(valid_thresholds)
        n_clear = sum(1 for row in q_correction["feature_table"] if row["hit_days"] >= threshold)
        n_total = len(q_correction["feature_table"])
        print(f"\n  Adopted threshold (max across the 3 plants) = {threshold} overpass days")
        print(f"  {n_clear}/{n_total} of the 30 plants have hit_days >= {threshold} "
              f"(data/q_correction_model_results.json feature_table)")
        out["threshold_days"] = threshold
        out["n_plants_clearing_threshold"] = n_clear
        out["n_plants_total"] = n_total
        json.dump(out, open(out_path, "w"), indent=2)
    else:
        print("  No plant reached a stable estimate at any tested n_days -- "
              "day count alone does not explain the failure pattern.")
