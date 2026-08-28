"""
Gaussian cross-sectional-flux Q estimator -- an alternate to
physics_gaussian.py's IME mass-balance method, run on the same OCO-3 XCO2
soundings. Rather than summing excess mass over a fixed-radius near-plant
disk and dividing by sqrt(area) (IME's L_eff), this rotates near-plant
soundings into a downwind(x)/crosswind(y) frame using the plant's
wind direction, keeps only the downwind soundings, and fits a Gaussian to
the crosswind excess-mass profile:

    excess_mass_per_area(y) = A * exp(-y^2 / (2*sigma^2))          [kg/m^2]

Integrating that Gaussian across the crosswind axis analytically gives the
plume's line density (kg/m) at the cross-section, and multiplying by the
effective advection speed gives flux directly:

    Q = U_eff * A * sigma * sqrt(2*pi)                              [kg/s]

This is the "cross-sectional flux" method referenced alongside IME in the
point-source XCO2/CH4 literature (e.g. Varon et al. 2018 compares both for
CH4 plumes) -- conceptually different from IME in that it uses the plume's
actual spatial (crosswind) shape and only downwind soundings, instead of a
direction-agnostic near-plant disk.

Physical constants, background-annulus zones (NEAR/BG_IN/BG_OUT), and
wind-series caching are reused unchanged from physics_gaussian.py (see
CLAUDE.md: never reimplement shared physics). ALPHA (U_eff = ALPHA *
wind_speed) is kept at the same 0.5 literature value physics_gaussian.py
uses -- the physical justification (10m wind speed vs. deeper
boundary-layer mixing) applies identically to column data regardless of
which method computes the flux.

Wind direction: plant_results.json's wind_deg field (process_plant.py's
ANNUAL mean "direction wind blows toward", degrees from north) is used for
ALL 30 plants uniformly. Per-overpass wind DIRECTION (data/daily_wind/)
exists for only 18/30 plants, so using it would silently restrict this
script to a smaller, different plant set than IME's -- the annual mean is
coarser (physics_gaussian.py's own docstring flags the same coarseness for
wind SPEED before its per-overpass upgrade) but keeps the comparison over
all 30 plants directly comparable. Flagged here, not fixed.

Other caveats (kept deliberately simple for a first cut of this method):
  - background is a single unstratified annulus mean, NOT
    physics_gaussian.py's month-stratified background (_month_stratify_bg)
  - wind speed uses per-overpass-day matching when >=3 downwind days match
    the cached ERA5 series, else the annual-series mean -- no std/
    uncertainty term is computed (physics_gaussian.py's three-term sigma is
    not replicated here)
  - the Gaussian fit pools ALL downwind near-plant soundings for a plant
    into one cross-section, not a true per-overpass or per-downwind-distance
    profile -- a coarser use of "cross-sectional flux" than the dense
    airborne-imaging literature the method is drawn from
  - skips a plant when there are too few downwind soundings, or when the
    curve fit doesn't converge to valid (positive, finite) parameters
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import baseline_capacity as bc
import physics_gaussian as pg

MIN_DOWNWIND = 8             # min downwind soundings to attempt a fit
M_PER_DEG_LAT = 111320.0
# Physical cap on the fitted crosswind sigma: a plume can't be resolved as
# wider than several times the near-plant disk (pg.NEAR=0.25deg ~27.8km
# radius) it's sampled from. Without this, curve_fit on sparse/noisy data
# sometimes drives sigma to its unbounded upper limit while A->0, keeping
# the residual flat but making A*sigma (and Q) meaningless -- observed as
# sigma in the tens of thousands of km for several plants on the first run.
MAX_SIGMA_M = 3 * pg.NEAR * M_PER_DEG_LAT
OUT_JSON = "data/gaussian_crosssection_results.json"
OUT_PNG = "data/gaussian_crosssection_vs_ime.png"

# vindhyachal_soundings.npz is lowercase, unlike every other plant's file
pg.NPZ_PATHS["Vindhyachal"] = "data/vindhyachal_soundings.npz"


def _load_wind_series(name, lat, lon):
    """Same cache-or-fetch pattern as overpass_density_experiment.py /
    wind_match_quality_all_plants.py -- avoids repeated slow Earth Engine
    calls across plants that already have a cached series."""
    cache_path = f"data/{name}_wind_series_cache.json"
    if os.path.exists(cache_path):
        raw = json.load(open(cache_path))
        return {int(k): v for k, v in raw.items()}
    wind_series = pg._fetch_wind_series(lat, lon)
    json.dump(wind_series, open(cache_path, "w"))
    return wind_series


def _gaussian(y, A, sigma):
    return A * np.exp(-(y ** 2) / (2 * sigma ** 2))


def _fit_r2(y, excess_mass, A, sigma):
    pred = _gaussian(y, A, sigma)
    ss_res = float(np.sum((excess_mass - pred) ** 2))
    ss_tot = float(np.sum((excess_mass - excess_mass.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_plant(name, plant_row, wind_series):
    npz_path = pg.NPZ_PATHS.get(name, f"data/{name}_soundings.npz")
    try:
        d = np.load(npz_path)
    except FileNotFoundError:
        print(f"[{name}] no saved soundings, skipping")
        return {"plant": name, "status": "skipped", "reason": "no soundings file"}

    lat, lon, xco2 = d["lat"], d["lon"], d["xco2"]
    has_days = "day" in d.files
    day = d["day"] if has_days else None

    plat, plon = plant_row["lat"], plant_row["lon"]
    dist = np.sqrt((lat - plat) ** 2 + (lon - plon) ** 2)

    near_mask = dist < pg.NEAR
    bg_mask = (dist > pg.BG_IN) & (dist < pg.BG_OUT)
    if near_mask.sum() < 5 or bg_mask.sum() < 5:
        print(f"[{name}] too few near/bg soundings, skipping")
        return {"plant": name, "status": "skipped", "reason": "too few near/bg soundings"}
    bg_mean = float(xco2[bg_mask].mean())

    # rotate near-plant soundings into downwind(x)/crosswind(y) meters using
    # the plant's annual-mean wind direction
    theta = np.radians(plant_row["wind_deg"])
    m_per_deg_lon = M_PER_DEG_LAT * np.cos(np.radians(plat))
    dx_m = (lon[near_mask] - plon) * m_per_deg_lon
    dy_m = (lat[near_mask] - plat) * M_PER_DEG_LAT
    x_dw = dx_m * np.sin(theta) + dy_m * np.cos(theta)
    y_cw = dx_m * np.cos(theta) - dy_m * np.sin(theta)
    xco2_near = xco2[near_mask]
    day_near = day[near_mask] if has_days else None

    downwind = x_dw > 0
    n_downwind = int(downwind.sum())
    n_days_used = int(len(np.unique(day_near[downwind]))) if has_days and n_downwind > 0 else 0

    if n_downwind < MIN_DOWNWIND:
        print(f"[{name}] too few downwind soundings ({n_downwind} < {MIN_DOWNWIND}), skipping")
        return {"plant": name, "status": "skipped",
                "reason": f"too few downwind soundings ({n_downwind} < {MIN_DOWNWIND})",
                "n_downwind_soundings": n_downwind, "n_overpass_days_used": n_days_used}

    excess_ppm = np.clip(xco2_near[downwind] - bg_mean, 0, None)
    excess_mass = pg.column_mass_enhancement(excess_ppm)  # kg/m^2
    y = y_cw[downwind]

    if int(np.count_nonzero(excess_mass > 0)) < MIN_DOWNWIND:
        print(f"[{name}] too few downwind soundings above background, skipping")
        return {"plant": name, "status": "skipped",
                "reason": "too few downwind soundings above background",
                "n_downwind_soundings": n_downwind, "n_overpass_days_used": n_days_used}

    A0 = float(excess_mass.max())
    sigma0 = float(np.std(y)) or 1000.0
    sigma0 = min(sigma0, MAX_SIGMA_M / 2)
    try:
        popt, _ = curve_fit(_gaussian, y, excess_mass, p0=[A0, sigma0],
                             bounds=([0.0, 1.0], [np.inf, MAX_SIGMA_M]), maxfev=5000)
    except (RuntimeError, ValueError) as e:
        print(f"[{name}] curve fit did not converge, skipping")
        return {"plant": name, "status": "skipped",
                "reason": f"curve fit did not converge ({str(e)[:60]})",
                "n_downwind_soundings": n_downwind, "n_overpass_days_used": n_days_used}

    A_fit, sigma_fit = float(popt[0]), float(popt[1])
    if not (A_fit > 0 and sigma_fit > 0 and np.isfinite(A_fit) and np.isfinite(sigma_fit)):
        print(f"[{name}] fit produced invalid parameters, skipping")
        return {"plant": name, "status": "skipped", "reason": "fit produced invalid parameters",
                "n_downwind_soundings": n_downwind, "n_overpass_days_used": n_days_used}
    if sigma_fit > 0.99 * MAX_SIGMA_M:
        print(f"[{name}] fit degenerate (sigma pinned at cap), skipping")
        return {"plant": name, "status": "skipped",
                "reason": f"degenerate fit: sigma pinned at cap ({MAX_SIGMA_M/1000:.0f}km)",
                "n_downwind_soundings": n_downwind, "n_overpass_days_used": n_days_used}

    # wind speed: per-overpass mean over the matched downwind days when
    # available (>=3 matches), else the full annual series mean -- same
    # fallback structure as physics_gaussian.py, without its std term
    if has_days and wind_series:
        uniq_days = np.unique(day_near[downwind])
        matched = np.array(
            [wind_series[int(dy)] for dy in uniq_days if int(dy) in wind_series], dtype=float)
    else:
        matched = np.array([])
    if matched.size >= 3:
        wind_speed = float(matched.mean())
        wind_mode = "per-overpass"
    elif wind_series:
        wind_speed = float(np.mean(list(wind_series.values())))
        wind_mode = "annual-mean (fallback)"
    else:
        print(f"[{name}] no wind data available, skipping")
        return {"plant": name, "status": "skipped", "reason": "no wind data available",
                "n_downwind_soundings": n_downwind, "n_overpass_days_used": n_days_used}

    u_eff = pg.ALPHA * wind_speed
    q_kg_s = u_eff * A_fit * sigma_fit * np.sqrt(2 * np.pi)
    q_t_yr = q_kg_s * pg.SEC_PER_YEAR / 1000
    r2 = _fit_r2(y, excess_mass, A_fit, sigma_fit)

    result = {
        "plant": name,
        "status": "fit",
        "n_downwind_soundings": n_downwind,
        "n_overpass_days_used": n_days_used,
        "bg_mean_ppm": bg_mean,
        "gaussian_amplitude_kg_m2": A_fit,
        "gaussian_sigma_m": sigma_fit,
        "fit_r2": r2,
        "wind_speed_ms": wind_speed,
        "wind_mode": wind_mode,
        "u_eff_ms": u_eff,
        "q_kg_s": q_kg_s,
        "q_t_per_year": q_t_yr,
    }
    print(f"[{name}] n_downwind={n_downwind} days_used={n_days_used} "
          f"sigma={sigma_fit/1000:.2f}km R2={r2:.2f} U_eff={u_eff:.2f}m/s "
          f"({wind_mode}) -> Q = {q_t_yr:,.0f} t/yr")
    return result


def main():
    plant_rows = {r["plant"]: r for r in json.load(open("data/plant_results.json"))}
    cea = json.load(open("data/cea_ground_truth_2020_21.json"))["facilities"]
    ime = {r["plant"]: r for r in json.load(open("data/emission_estimates.json"))}

    results = []
    for name in sorted(plant_rows):
        row = plant_rows[name]
        wind_series = _load_wind_series(name, row["lat"], row["lon"])
        results.append(fit_plant(name, row, wind_series))

    fitted = [r for r in results if r["status"] == "fit"]
    skipped = [r for r in results if r["status"] == "skipped"]
    print(f"\n=== {len(fitted)}/{len(results)} plants fit, {len(skipped)} skipped ===")

    # --- 1 & 2: IME vs Gaussian, on plants where BOTH succeeded ---
    common = [r for r in fitted if r["plant"] in ime]
    common_plants = [r["plant"] for r in common]
    gauss_q = np.array([r["q_t_per_year"] for r in common])
    ime_q = np.array([ime[p]["q_t_per_year"] for p in common_plants])
    log_r = float(np.corrcoef(np.log(ime_q), np.log(gauss_q))[0, 1]) if len(common) >= 3 else float("nan")
    ratios = {p: float(ime_q[i] / gauss_q[i]) for i, p in enumerate(common_plants)}
    print(f"\n=== IME vs Gaussian cross-section (N={len(common)} plants with both) ===")
    print(f"  r(log IME_Q, log Gaussian_Q) = {log_r:.3f}")
    for p in common_plants:
        print(f"  {p:20s} IME/Gaussian ratio = {ratios[p]:.2f}")

    # --- 3: Rihand check ---
    rihand_cea = cea["Rihand"]["abs_emissions_t_co2"]
    rihand_gauss = next((r for r in fitted if r["plant"] == "Rihand"), None)
    rihand_ime = ime.get("Rihand")
    rihand_check = {"cea_t_per_year": rihand_cea}
    if rihand_gauss:
        g_ratio = rihand_gauss["q_t_per_year"] / rihand_cea
        rihand_check["gaussian_q_t_per_year"] = rihand_gauss["q_t_per_year"]
        rihand_check["gaussian_log_ratio_vs_cea"] = float(np.log(g_ratio))
    else:
        rihand_check["gaussian_status"] = "skipped"
    if rihand_ime:
        i_ratio = rihand_ime["q_t_per_year"] / rihand_cea
        rihand_check["ime_q_t_per_year"] = rihand_ime["q_t_per_year"]
        rihand_check["ime_log_ratio_vs_cea"] = float(np.log(i_ratio))
    if "gaussian_log_ratio_vs_cea" in rihand_check and "ime_log_ratio_vs_cea" in rihand_check:
        rihand_check["gaussian_closer_to_cea"] = (
            abs(rihand_check["gaussian_log_ratio_vs_cea"]) < abs(rihand_check["ime_log_ratio_vs_cea"]))
    print(f"\n=== Rihand check ===")
    print(f"  CEA = {rihand_cea:,.0f} t/yr")
    print(f"  {json.dumps(rihand_check, indent=2)}")

    # --- 4: LOO R^2 vs CEA, same metric as baseline_capacity.py ---
    def loo_vs_cea(plants, q_vals, label):
        cea_vals = np.array([cea[p]["abs_emissions_t_co2"] for p in plants])
        if len(plants) < 4:
            return {"label": label, "n": len(plants), "note": "too few plants for LOO regression"}
        pred = bc.loo_linear_log_predictions(np.log(q_vals).reshape(-1, 1), np.log(cea_vals))
        return bc.evaluate(np.log(cea_vals), pred, plants, label)

    gauss_all_plants = [r["plant"] for r in fitted]
    gauss_all_q = np.array([r["q_t_per_year"] for r in fitted])
    ime_all_plants = list(ime.keys())
    ime_all_q = np.array([ime[p]["q_t_per_year"] for p in ime_all_plants])

    loo_gauss_full = loo_vs_cea(gauss_all_plants, gauss_all_q,
                                 f"Gaussian cross-section, all fit plants (N={len(gauss_all_plants)})")
    loo_ime_full = loo_vs_cea(ime_all_plants, ime_all_q,
                               f"IME, all plants (N={len(ime_all_plants)})")
    loo_gauss_common = loo_vs_cea(common_plants, gauss_q,
                                   f"Gaussian cross-section, common subset (N={len(common_plants)})")
    loo_ime_common = loo_vs_cea(common_plants, ime_q,
                                 f"IME, common subset (N={len(common_plants)})")

    print(f"\n=== LOO R^2 vs CEA (baseline_capacity.py methodology) ===")
    for r in (loo_gauss_full, loo_ime_full, loo_gauss_common, loo_ime_common):
        if "loo_r2" in r:
            print(f"  {r['label']:55s} R^2={r['loo_r2']:.3f}  MAE(log)={r['loo_mae_log']:.3f}  "
                  f"within_2x={r['within_2x']}/{r['n']}")
        else:
            print(f"  {r['label']:55s} {r.get('note')}")

    # --- scatter plot: IME Q vs Gaussian Q, colored by Gaussian |log_ratio| vs CEA ---
    fig, ax = plt.subplots(figsize=(7, 6))
    gauss_abs_log_ratio_vs_cea = np.array([
        abs(np.log(gauss_q[i] / cea[p]["abs_emissions_t_co2"])) for i, p in enumerate(common_plants)
    ])
    sc = ax.scatter(ime_q, gauss_q, c=gauss_abs_log_ratio_vs_cea, cmap="inferno_r",
                     s=70, edgecolors="black", linewidths=0.5, zorder=3)
    lims = [min(ime_q.min(), gauss_q.min()) * 0.5, max(ime_q.max(), gauss_q.max()) * 2]
    ax.plot(lims, lims, "k--", alpha=0.4, zorder=1, label="1:1")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("IME Q (t/yr)")
    ax.set_ylabel("Gaussian cross-section Q (t/yr)")
    ax.set_title(f"IME vs Gaussian cross-section Q (N={len(common_plants)}, r_log={log_r:.2f})")
    for i, p in enumerate(common_plants):
        ax.annotate(p, (ime_q[i], gauss_q[i]), fontsize=7, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("|log(Gaussian Q / CEA)|")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"\n[SAVED] {OUT_PNG}")

    out = {
        "method": "gaussian cross-sectional flux",
        "n_plants_total": len(results),
        "n_fit": len(fitted),
        "n_skipped": len(skipped),
        "per_plant": results,
        "ime_vs_gaussian": {
            "n_common": len(common),
            "common_plants": common_plants,
            "r_log_ime_gaussian": log_r,
            "ime_over_gaussian_ratio": ratios,
        },
        "rihand_check": rihand_check,
        "loo_vs_cea": {
            "gaussian_all_fit_plants": loo_gauss_full,
            "ime_all_plants": loo_ime_full,
            "gaussian_common_subset": loo_gauss_common,
            "ime_common_subset": loo_ime_common,
        },
    }
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"[SAVED] {OUT_JSON}")


if __name__ == "__main__":
    main()
