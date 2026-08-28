"""
This implements IME (Integrated Mass Enhancement), not a classic Gaussian
plume fit. See physics_gaussian_crosssection.py for Nassar's original
cross-section Gaussian method.

Emission-rate estimate from OCO-3 column-CO2 enhancement via the Integrated
Mass Enhancement (IME) method -- the standard way to turn a satellite CO2/CH4
plume into an emission rate (Varon et al. 2018 for CH4; the same mass-balance
logic underlies the power-plant CO2 estimates in Nassar et al. 2017 and
Reuter et al. 2019 for OCO-2/3). A classic ground-level Gaussian plume model
doesn't apply directly to column data (XCO2 is a dry-air column average, not
a surface concentration), so IME is the column-consistent analogue: it sums
the excess CO2 mass across the near-plant footprint and divides by an
effective plume length and advection speed.

    Q = U_eff * IME / L_eff

  IME   = total column-mass CO2 enhancement over near-plant soundings (kg)
  L_eff = effective plume length, sqrt(footprint area covered) (m)
  U_eff = effective advection speed (m/s); Varon et al. use U_eff = alpha*U,
          alpha ~= 0.5 to account for the wind speed being measured at 10m
          while the plume mixes through a deeper layer

Wind handling (updated from the Week 6 annual-mean-scalar version):
  the literature (e.g. wind-driven flux-uncertainty studies for comparable
  CH4 point-source work) identifies wind speed as the dominant term in
  point-source emission-rate uncertainty. This version fetches a per-day
  ERA5 wind-speed series for the year and, when the saved soundings carry a
  per-sounding date (data/<Plant>_soundings.npz's "day" field, added by
  process_plant.py), matches each near-plant sounding to its own overpass
  day's wind speed ("per-overpass" mode) instead of a single annual mean.
  Existing sounding files saved before this change have no "day" field, so
  they fall back to the full daily-speed distribution's mean and std
  ("annual-mean (fallback)" mode) -- still an improvement over a bare mean,
  since it yields an uncertainty term, but not truly per-overpass. Re-run
  process_plant.py for a plant to get "day" and unlock per-overpass mode.

Uncertainty: Q is linear in U_eff (IME and L_eff held fixed), so the wind
speed's relative std maps directly to Q's relative std from that term. A
second, independent term captures IME sampling noise (background/near-plant
population size) via bootstrap resampling. A third term (added Week 10,
after diagnose_talcher.py found the first two badly understated Talcher's
real uncertainty) captures background-annulus definition sensitivity: IME
is recomputed under several alternate, equally-reasonable background-zone
definitions (holding the near-plant zone fixed), and the relative std
across those alternates is the third term. This matters most for plants
with a thin/weak plume signal -- Talcher's IME swung 20% across background
definitions vs. 4% for a well-bracketed plant (Rihand), and its old,
narrower two-term uncertainty missed its Climate TRACE benchmark by ~2x.
The three relative stds are combined in quadrature (independent-error
assumption) into a single sigma, reported as q_t_per_year_std alongside
the point estimate q_t_per_year.

Month stratification (added after diagnose_shrisingajimalwa.py found a real
failure case): the background population -- both the main BG_IN/BG_OUT
annulus and each alternate definition in the sensitivity term -- is
restricted to only the months the near-plant zone was actually sampled in,
when per-sounding dates are available. Without this, a near-plant zone
sampled in one season compared against a background blended across several
other seasons can produce a spurious near-vs-background difference from
nothing but XCO2's real seasonal cycle (ShriSingajiMalwa: near=January
only, background also drew from April/May, ~4ppm seasonally higher on its
own, which looked like a significant negative "enhancement" until
restricted to the one shared month). Falls back to the unrestricted
background when day data isn't available or stratifying would leave too
few background soundings (results report month_stratified / n_bg_before/
after_month_filter so this is always visible, never silent).

Caveats (still a coarse, single-scalar estimate, not a fitted plume image):
  - assumes standard surface pressure (no local met pressure)
  - fixed, approximate footprint area per OCO-3 sounding (not per-overpass)
  - only "hits" whichever soundings happen to fall in the near-plant/
    background zones already used by co2_enhancement.py / process_plant.py
  - the reported sigma now covers wind, IME-sampling noise, and background-
    definition sensitivity, but still not boundary-layer-height uncertainty
    (no per-overpass PBL height data is used anywhere in this pipeline)
"""
import json
import time
import numpy as np

G = 9.80665                 # m/s^2
P_SURF = 101325.0           # Pa, standard surface pressure (no local met data)
M_CO2 = 0.04401              # kg/mol
M_AIR = 0.02897              # kg/mol
FOOTPRINT_AREA_M2 = 2.25e6  # ~1.5km x 1.5km, approx OCO-3 sounding footprint
ALPHA = 0.5                  # U_eff = ALPHA * wind speed (Varon et al. 2018 default)
SEC_PER_YEAR = 3600 * 24 * 365
N_BOOT = 500                 # bootstrap resamples for IME sampling uncertainty
MIN_WIND_DAYS_MATCHED = 3    # below this, fall back to the full-series mean/std

# same near-plant / background zones as co2_enhancement.py and process_plant.py
NEAR = 0.25
BG_IN, BG_OUT = 0.4, 0.9

# alternate background-annulus definitions for the sensitivity term (added
# Week 10, from diagnose_talcher.py) -- same 5 definitions tested there,
# holding the near-plant zone (NEAR) fixed and varying only where
# "background" is drawn from
BG_DEFINITIONS = [(0.4, 0.9), (0.3, 0.8), (0.35, 0.7), (0.4, 1.1), (0.5, 1.0)]
MIN_BG_DEFINITIONS = 3       # below this many valid alternates, term is 0

NPZ_PATHS = {}


def column_mass_enhancement(dppm):
    """ppm XCO2 excess -> kg CO2 excess per m^2 column (well-mixed dry-air column)."""
    dry_air_col_mass = P_SURF / G  # kg/m^2 of dry air in the column
    return dppm * 1e-6 * dry_air_col_mass * (M_CO2 / M_AIR)


def _ime_kg(near, bg_mean):
    excess = np.clip(near - bg_mean, 0, None)
    return float(np.sum(column_mass_enhancement(excess) * FOOTPRINT_AREA_M2))


def _bootstrap_ime_rel_std(near, bg, n_boot=N_BOOT, seed=0):
    """
    Relative std of IME under resampling of the near-plant and background
    populations -- captures sampling noise from small near/background
    counts, independent of the wind term.
    """
    rng = np.random.default_rng(seed)
    n_near, n_bg = len(near), len(bg)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        near_b = rng.choice(near, size=n_near, replace=True)
        bg_b = rng.choice(bg, size=n_bg, replace=True)
        boots[i] = _ime_kg(near_b, bg_b.mean())
    boots = boots[boots > 0]
    if len(boots) < 10:
        return 0.0
    mean = boots.mean()
    return float(boots.std(ddof=1) / mean) if mean > 0 else 0.0


def _month_of_day(day_int):
    """YYYYMMDD int -> month int (1-12), or None for the 0/missing sentinel
    used by sounding files predating process_plant.py's "day" field."""
    return int(str(int(day_int))[4:6]) if day_int > 0 else None


def _month_stratify_bg(bg_vals, bg_day, near_months_set, min_kept=5):
    """
    Restrict a background population to only the months the near-plant zone
    was actually sampled in, per diagnose_shrisingajimalwa.py's finding:
    comparing near-plant soundings from one month against a background ring
    blended across several other months' regional XCO2 baselines can
    produce a spurious near-vs-background difference that has nothing to do
    with the plant (ShriSingajiMalwa: near=January only, background also
    drew from April/May, ~4ppm seasonally higher, which by itself produced
    a significant "negative enhancement"). Falls back to the unrestricted
    background (and reports that no stratification was applied) when day
    data isn't available or restricting would leave too few background
    soundings to be usable -- same MIN_BG_DEFINITIONS-style graceful
    degradation as the rest of this module, since an empty/tiny restricted
    background is worse than an unstratified one, not an improvement.
    Returns (possibly-restricted bg_vals, stratified: bool).
    """
    if bg_day is None or near_months_set is None:
        return bg_vals, False
    bg_months = np.array([_month_of_day(x) for x in bg_day])
    keep = np.isin(bg_months, list(near_months_set))
    if int(keep.sum()) < min_kept:
        return bg_vals, False
    return bg_vals[keep], True


def _bg_definition_rel_std(dist, xco2, near, day=None, near_months_set=None):
    """
    Relative std of IME (kg) across several reasonable background-annulus
    definitions, holding the near-plant zone fixed -- see
    diagnose_talcher.py, which found this to be a much larger error source
    than wind or IME-sampling noise for plants with a thin/weak plume
    signal. Returns (rel_std, n_definitions_used).

    When day/near_months_set are supplied, each alternate background
    definition is also month-stratified against the near-plant zone's
    months (see _month_stratify_bg) -- otherwise the sensitivity term
    itself could reintroduce the same seasonal-imbalance artifact this
    stratification is meant to prevent, just spread across 5 definitions
    instead of the single main one.
    """
    imes = []
    for bg_in, bg_out in BG_DEFINITIONS:
        bg_zone_mask = (dist > bg_in) & (dist < bg_out)
        bg = xco2[bg_zone_mask]
        bg_day = day[bg_zone_mask] if day is not None else None
        bg, _ = _month_stratify_bg(bg, bg_day, near_months_set)
        if len(bg) < 5:
            continue
        imes.append(_ime_kg(near, bg.mean()))
    if len(imes) < MIN_BG_DEFINITIONS:
        return 0.0, len(imes)
    imes = np.array(imes)
    mean = imes.mean()
    rel_std = float(imes.std(ddof=1) / mean) if mean > 0 else 0.0
    return rel_std, len(imes)


def estimate_emission_rate(plant_row, wind_series):
    """
    plant_row: one entry from data/plant_results.json
    wind_series: {YYYYMMDD (int): daily-mean ERA5 10m wind speed (m/s)},
                 from _fetch_wind_series()
    returns a result dict, or None if the plant can't be estimated
    """
    name = plant_row["plant"]
    npz_path = NPZ_PATHS.get(name, f"data/{name}_soundings.npz")
    try:
        d = np.load(npz_path)
    except FileNotFoundError:
        print(f"[{name}] no saved soundings at {npz_path}, skipping")
        return None

    lat, lon, xco2 = d["lat"], d["lon"], d["xco2"]
    has_days = "day" in d.files
    day = d["day"] if has_days else None
    return estimate_emission_rate_from_arrays(name, lat, lon, xco2, day, plant_row, wind_series)


def estimate_emission_rate_from_arrays(name, lat, lon, xco2, day, plant_row, wind_series):
    """
    Same IME math as estimate_emission_rate(), factored out so callers can
    rerun the estimate on an in-memory (e.g. subsampled) set of soundings
    without touching data/<plant>_soundings.npz -- see
    overpass_density_experiment.py, which subsamples overpass days to test
    whether Q accuracy depends on day count. `day` may be None (no
    per-sounding dates), matching estimate_emission_rate()'s has_days
    fallback path.
    """
    has_days = day is not None
    plat, plon = plant_row["lat"], plant_row["lon"]
    dist = np.sqrt((lat - plat) ** 2 + (lon - plon) ** 2)

    near_mask = dist < NEAR
    bg_mask = (dist > BG_IN) & (dist < BG_OUT)
    near, bg = xco2[near_mask], xco2[bg_mask]
    near_day = day[near_mask] if has_days else None
    bg_day = day[bg_mask] if has_days else None
    if len(near) < 5 or len(bg) < 5:
        print(f"[{name}] too few soundings (near={len(near)}, bg={len(bg)}), skipping")
        return None

    # month-stratify background against near-plant months, per
    # diagnose_shrisingajimalwa.py -- see _month_stratify_bg's docstring
    near_months_set = (set(m for m in (_month_of_day(x) for x in near_day) if m is not None)
                        if has_days else None)
    n_bg_before_month_filter = len(bg)
    bg, month_stratified = _month_stratify_bg(bg, bg_day, near_months_set)

    bg_mean = bg.mean()
    excess_ppm = np.clip(near - bg_mean, 0, None)
    used = excess_ppm > 0
    n_used = int(used.sum())
    if n_used == 0:
        print(f"[{name}] no soundings above background, skipping")
        return None

    ime = _ime_kg(near, bg_mean)
    plume_area = n_used * FOOTPRINT_AREA_M2
    l_eff = float(np.sqrt(plume_area))

    # --- wind: per-overpass match when we have sounding dates, else fall
    # back to the daily-speed series' mean/std (still better than a bare
    # annual-mean scalar, since it yields an uncertainty term). Dedupe to
    # one wind value per unique overpass day first -- near_day[used] has
    # one entry per sounding, and a single overpass contributes many
    # near-plant soundings on the same day, so matching without dedup
    # both mis-names MIN_WIND_DAYS_MATCHED (a sounding count, not a day
    # count) and deflates wind_speed_std (duplicated identical values
    # pull the sample std toward 0, understating wind uncertainty). ---
    matched = np.array([])
    if has_days and wind_series:
        uniq_days = np.unique(near_day[used])
        matched = np.array(
            [wind_series[int(dy)] for dy in uniq_days if int(dy) in wind_series],
            dtype=float,
        )

    if matched.size >= MIN_WIND_DAYS_MATCHED:
        wind_speed_mean = float(matched.mean())
        wind_speed_std = float(matched.std(ddof=1))
        wind_mode = "per-overpass"
        n_wind_matched = int(matched.size)
    elif wind_series:
        vals = np.array(list(wind_series.values()), dtype=float)
        wind_speed_mean = float(vals.mean())
        wind_speed_std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        wind_mode = "annual-mean (fallback)"
        n_wind_matched = 0
    else:
        print(f"[{name}] no wind data available, skipping")
        return None

    u_eff = ALPHA * wind_speed_mean
    q_kg_s = u_eff * ime / l_eff
    q_t_yr = q_kg_s * SEC_PER_YEAR / 1000

    wind_rel_std = wind_speed_std / wind_speed_mean if wind_speed_mean > 0 else 0.0
    ime_rel_std = _bootstrap_ime_rel_std(near, bg)
    bg_rel_std, n_bg_defs = _bg_definition_rel_std(dist, xco2, near, day=day, near_months_set=near_months_set)
    q_rel_std = float(np.sqrt(wind_rel_std ** 2 + ime_rel_std ** 2 + bg_rel_std ** 2))
    q_t_yr_std = q_t_yr * q_rel_std

    result = {
        "plant": name,
        "n_soundings_used": n_used,
        "month_stratified": month_stratified,
        "n_bg_before_month_filter": n_bg_before_month_filter,
        "n_bg_after_month_filter": len(bg),
        "ime_kg": ime,
        "l_eff_m": l_eff,
        "wind_speed_ms": wind_speed_mean,
        "wind_speed_std_ms": wind_speed_std,
        "wind_mode": wind_mode,
        "n_wind_days_matched": n_wind_matched,
        # unthresholded versions of the above -- n_wind_matched/wind_mode are
        # gated by MIN_WIND_DAYS_MATCHED, so a plant with 1-2 real per-overpass
        # matches still reports n_wind_days_matched=0 in fallback mode. These
        # two keep the actual matched.size and the per-day speeds it found,
        # for callers that need to see what was discarded (e.g.
        # wind_match_quality_all_plants.py, which measures how often a real
        # per-overpass match exists at all, and how far the annual-mean
        # fallback would have been from those real values).
        "n_wind_days_raw_matched": int(matched.size),
        "wind_days_matched_speeds": matched.tolist(),
        "u_eff_ms": u_eff,
        "wind_rel_std": wind_rel_std,
        "ime_rel_std": ime_rel_std,
        "bg_rel_std": bg_rel_std,
        "n_bg_definitions": n_bg_defs,
        "q_rel_std": q_rel_std,
        "q_kg_s": q_kg_s,
        "q_t_per_year": q_t_yr,
        "q_t_per_year_std": q_t_yr_std,
        "q_t_per_year_low": q_t_yr - q_t_yr_std,
        "q_t_per_year_high": q_t_yr + q_t_yr_std,
    }
    strat_note = (f"month-stratified bg: {n_bg_before_month_filter}->{len(bg)}"
                  if month_stratified else "bg not month-stratified")
    print(f"[{name}] IME={ime:,.0f} kg  L_eff={l_eff/1000:.2f} km  "
          f"U_eff={u_eff:.2f} m/s ({wind_mode}, n={n_wind_matched})  {strat_note}  ->  "
          f"Q = {q_t_yr:,.1f} +/- {q_t_yr_std:,.1f} t/yr "
          f"(wind {wind_rel_std:.0%}, IME {ime_rel_std:.0%}, bg {bg_rel_std:.0%})")
    return result


def _fetch_wind_series(lat, lon, year=2020):
    """
    Daily ERA5 10m wind speed at (lat, lon) for `year`, via Earth Engine.
    Returns {YYYYMMDD (int): speed_m_s (float)}, one entry per day with
    valid data. Speed is computed per-day from that day's (u,v) mean, i.e.
    hypot(mean_u, mean_v) -- daily-scale vector averaging is fine here (the
    Week 6 bug was averaging across the *whole year*, which cancels out
    most of the speed as wind direction rotates through seasons; a single
    day's direction doesn't rotate enough for that to matter materially).
    """
    import ee
    ee.Initialize(project="opportune-lore-415218")
    pt = ee.Geometry.Point([lon, lat])
    coll = (ee.ImageCollection("ECMWF/ERA5/DAILY")
            .select(["u_component_of_wind_10m", "v_component_of_wind_10m"])
            .filterDate(f"{year}-01-01", f"{year}-12-31"))

    rows = coll.getRegion(pt, scale=27830).getInfo()
    header, records = rows[0], rows[1:]
    i_u = header.index("u_component_of_wind_10m")
    i_v = header.index("v_component_of_wind_10m")
    i_t = header.index("time")

    series = {}
    for row in records:
        u, v, t_ms = row[i_u], row[i_v], row[i_t]
        if u is None or v is None or t_ms is None:
            continue
        day = int(time.strftime("%Y%m%d", time.gmtime(t_ms / 1000)))
        series[day] = float(np.hypot(u, v))
    return series


if __name__ == "__main__":
    rows = json.load(open("data/plant_results.json"))

    results = []
    for row in rows:
        try:
            wind_series = _fetch_wind_series(row["lat"], row["lon"])
        except Exception as e:
            print(f"[{row['plant']}] wind fetch failed: {str(e)[:60]}, skipping")
            continue
        r = estimate_emission_rate(row, wind_series)
        if r:
            results.append(r)

    out_path = "data/emission_estimates.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\n[SAVED] {len(results)} plant emission estimates -> {out_path}")
