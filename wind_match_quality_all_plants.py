"""
Week 13, experiment 4: does wind-matching quality explain Rihand's error?

Experiment 3's summary noted Rihand had only 4-5 of 16 overpass days with
a real per-overpass wind match (data/emission_estimates.json), unlike
experiments 1 (overpass density) and 2 (signal-to-noise) and 3
(background definition), which were all ruled out as Rihand's specific
problem. This checks wind-matching quality directly.

physics_gaussian.py's n_wind_days_matched is threshold-gated: a plant
with 1-2 real per-overpass matches (below MIN_WIND_DAYS_MATCHED=3) still
reports n_wind_days_matched=0, because the code falls back to the
annual-mean speed for the whole plant. That masks real variation across
the 15 (of 24) plants in fallback mode, so this script uses the new
n_wind_days_raw_matched / wind_days_matched_speeds fields (additive-only
change to physics_gaussian.py, verified against data/emission_estimates.json
before use) to get the real, unthresholded match count and the actual
per-day speeds those matches found.
"""
import json
import os

import numpy as np

import physics_gaussian as pg

# vindhyachal_soundings.npz is lowercase, unlike every other plant's file
pg.NPZ_PATHS["Vindhyachal"] = "data/vindhyachal_soundings.npz"


def _load_wind_series(name, lat, lon):
    cache_path = f"data/{name}_wind_series_cache.json"
    if os.path.exists(cache_path):
        raw = json.load(open(cache_path))
        return {int(k): v for k, v in raw.items()}
    wind_series = pg._fetch_wind_series(lat, lon)
    json.dump(wind_series, open(cache_path, "w"))
    return wind_series


def plant_wind_row(name, plant_row):
    npz_path = pg.NPZ_PATHS.get(name, f"data/{name}_soundings.npz")
    d = np.load(npz_path)
    if "day" not in d.files:
        print(f"[{name}] no per-sounding day field, skipping")
        return None
    lat, lon, xco2, day = d["lat"], d["lon"], d["xco2"], d["day"]
    hit_days = int(len(np.unique(day)))

    wind_series = _load_wind_series(name, plant_row["lat"], plant_row["lon"])
    r = pg.estimate_emission_rate_from_arrays(name, lat, lon, xco2, day, plant_row, wind_series)
    if r is None:
        print(f"[{name}] too few near/bg soundings, skipping")
        return None

    raw_matched = r["n_wind_days_raw_matched"]
    wind_match_rate = raw_matched / hit_days if hit_days > 0 else None

    # "where checkable": for plants that fell back to the annual-mean speed,
    # how far off is that annual mean from the real per-day speeds that WERE
    # found (even though there were too few of them to trigger per-overpass
    # mode)? Only defined when there's at least one real matched day.
    annual_mean_error_pct = None
    n_checkable = 0
    if r["wind_mode"] == "annual-mean (fallback)" and raw_matched > 0:
        speeds = np.array(r["wind_days_matched_speeds"])
        annual_mean = r["wind_speed_ms"]  # the fallback value actually used
        pct_diffs = np.abs(speeds - annual_mean) / annual_mean
        annual_mean_error_pct = float(pct_diffs.mean() * 100)
        n_checkable = int(len(speeds))

    row = {
        "plant": name,
        "hit_days": hit_days,
        "wind_mode": r["wind_mode"],
        "n_wind_days_reported_matched": r["n_wind_days_matched"],  # threshold-gated, as in emission_estimates.json
        "n_wind_days_raw_matched": raw_matched,                     # unthresholded, real count
        "wind_match_rate": wind_match_rate,
        "annual_mean_error_pct_where_checkable": annual_mean_error_pct,
        "n_checkable_fallback_days": n_checkable,
    }
    rate_str = f"{wind_match_rate:.0%}" if wind_match_rate is not None else "n/a"
    err_str = f"{annual_mean_error_pct:.0f}%" if annual_mean_error_pct is not None else "n/a"
    print(f"[{name}] mode={r['wind_mode']:24s} raw_matched={raw_matched}/{hit_days} "
          f"rate={rate_str:>5s}  annual_mean_error(checkable, n={n_checkable})={err_str}")
    return row


def main():
    plant_rows = {r["plant"]: r for r in json.load(open("data/plant_results.json"))}
    q_correction = json.load(open("data/q_correction_model_results.json"))
    feature_table = {r["plant"]: r for r in q_correction["feature_table"]}

    rows = []
    for name in sorted(plant_rows):
        row = plant_wind_row(name, plant_rows[name])
        if row is not None:
            rows.append(row)

    # --- Rihand check ---
    rihand = next(r for r in rows if r["plant"] == "Rihand")
    print(f"\n=== Rihand check ===")
    print(f"  wind_match_rate = {rihand['wind_match_rate']:.1%} "
          f"({rihand['n_wind_days_raw_matched']}/{rihand['hit_days']} days)")

    rates = [r["wind_match_rate"] for r in rows if r["wind_match_rate"] is not None]
    rank = sorted(rates).index(rihand["wind_match_rate"]) + 1
    percentile = rank / len(rates) * 100
    print(f"  Rank among {len(rates)} plants with a computable rate: "
          f"{rank}/{len(rates)} (percentile {percentile:.0f}, low=worse)")
    rihand_is_low_outlier = percentile <= 25
    print(f"  Low outlier (bottom quartile) = {rihand_is_low_outlier}")

    # --- predictor comparison: wind_match_rate vs hit_days, SNR, bg swing ---
    matched = [(r, feature_table[r["plant"]]) for r in rows
               if r["plant"] in feature_table and r["wind_match_rate"] is not None]
    rate_vals = np.array([r["wind_match_rate"] for r, _ in matched])
    abs_log_ratio = np.array([abs(f["log_ratio"]) for _, f in matched])
    r_wind_match = float(np.corrcoef(rate_vals, abs_log_ratio)[0, 1])

    r_hit_days = -0.30971260219321334
    snr_data = json.load(open("data/snr_all_plants.json"))
    r_snr = snr_data["q1_snr_vs_hit_days"]["r_snr_abs_log_ratio"]
    bg_data = json.load(open("data/bg_sensitivity_all_plants.json"))
    r_bg_swing = bg_data["predictor_comparison"]["r_swing_pct_abs_log_ratio"]

    print(f"\n=== Predictor comparison (N={len(matched)}) ===")
    print(f"  r(wind_match_rate, |log_ratio|) = {r_wind_match:+.3f}")
    print(f"  r(hit_days, |log_ratio|)        = {r_hit_days:+.3f}  (Week 12)")
    print(f"  r(SNR, |log_ratio|)              = {r_snr:+.3f}  (exp 2)")
    print(f"  r(bg swing%, |log_ratio|)        = {r_bg_swing:+.3f}  (exp 3)")
    ranked = sorted(
        [("wind_match_rate", abs(r_wind_match)), ("hit_days", abs(r_hit_days)),
         ("SNR", abs(r_snr)), ("bg swing%", abs(r_bg_swing))],
        key=lambda x: -x[1],
    )
    print(f"  -> ranked by |r| (strongest first): {[f'{n} ({v:.3f})' for n, v in ranked]}")
    is_strongest = ranked[0][0] == "wind_match_rate"
    print(f"  wind_match_rate is the strongest predictor so far = {is_strongest}")

    out = {
        "n_plants": len(rows),
        "plants": sorted(rows, key=lambda r: (r["wind_match_rate"] is None, r["wind_match_rate"] or 0.0)),
        "rihand_check": {
            "wind_match_rate": rihand["wind_match_rate"],
            "n_wind_days_raw_matched": rihand["n_wind_days_raw_matched"],
            "hit_days": rihand["hit_days"],
            "rank": rank,
            "n_total": len(rates),
            "percentile": percentile,
            "is_low_outlier_bottom_quartile": rihand_is_low_outlier,
        },
        "predictor_comparison": {
            "n_matched": len(matched),
            "r_wind_match_rate_abs_log_ratio": r_wind_match,
            "r_hit_days_abs_log_ratio": r_hit_days,
            "r_snr_abs_log_ratio": r_snr,
            "r_bg_swing_pct_abs_log_ratio": r_bg_swing,
            "ranked_by_abs_r": [{"predictor": n, "abs_r": v} for n, v in ranked],
            "wind_match_rate_is_strongest": is_strongest,
        },
    }
    out_path = "data/wind_match_quality_all_plants.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
