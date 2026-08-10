"""
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

Caveats (this is a coarse, single-scalar estimate, not a fitted plume image):
  - assumes standard surface pressure (no local met pressure)
  - wind is a single 2020 annual mean, not per-overpass
  - only "hits" whichever soundings happen to fall in the near-plant/
    background zones already used by co2_enhancement.py / process_plant.py
"""
import json
import numpy as np

G = 9.80665                 # m/s^2
P_SURF = 101325.0           # Pa, standard surface pressure (no local met data)
M_CO2 = 0.04401              # kg/mol
M_AIR = 0.02897              # kg/mol
FOOTPRINT_AREA_M2 = 2.25e6  # ~1.5km x 1.5km, approx OCO-3 sounding footprint
ALPHA = 0.5                  # U_eff = ALPHA * wind speed (Varon et al. 2018 default)

# same near-plant / background zones as co2_enhancement.py and process_plant.py
NEAR = 0.25
BG_IN, BG_OUT = 0.4, 0.9

NPZ_PATHS = {
    "Vindhyachal": "data/vindhyachal_soundings.npz",
}


def column_mass_enhancement(dppm):
    """ppm XCO2 excess -> kg CO2 excess per m^2 column (well-mixed dry-air column)."""
    dry_air_col_mass = P_SURF / G  # kg/m^2 of dry air in the column
    return dppm * 1e-6 * dry_air_col_mass * (M_CO2 / M_AIR)


def estimate_emission_rate(plant_row, wind_speed_ms):
    """
    plant_row: one entry from data/plant_results.json
    wind_speed_ms: mean wind speed at the plant (m/s), e.g. from ERA5 (see wind_check.py)
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
    plat, plon = plant_row["lat"], plant_row["lon"]
    dist = np.sqrt((lat - plat) ** 2 + (lon - plon) ** 2)

    near = xco2[dist < NEAR]
    bg = xco2[(dist > BG_IN) & (dist < BG_OUT)]
    if len(near) < 5 or len(bg) < 5:
        print(f"[{name}] too few soundings (near={len(near)}, bg={len(bg)}), skipping")
        return None

    bg_mean = bg.mean()
    excess_ppm = np.clip(near - bg_mean, 0, None)
    n_used = int((excess_ppm > 0).sum())
    if n_used == 0:
        print(f"[{name}] no soundings above background, skipping")
        return None

    ime = float(np.sum(column_mass_enhancement(excess_ppm) * FOOTPRINT_AREA_M2))  # kg
    plume_area = n_used * FOOTPRINT_AREA_M2
    l_eff = float(np.sqrt(plume_area))
    u_eff = ALPHA * wind_speed_ms

    q_kg_s = u_eff * ime / l_eff
    q_t_yr = q_kg_s * 3600 * 24 * 365 / 1000

    result = {
        "plant": name,
        "n_soundings_used": n_used,
        "ime_kg": ime,
        "l_eff_m": l_eff,
        "wind_speed_ms": wind_speed_ms,
        "u_eff_ms": u_eff,
        "q_kg_s": q_kg_s,
        "q_t_per_year": q_t_yr,
    }
    print(f"[{name}] IME={ime:,.0f} kg  L_eff={l_eff/1000:.2f} km  "
          f"U_eff={u_eff:.2f} m/s  ->  Q = {q_kg_s:.2f} kg/s  ({q_t_yr:,.0f} t/yr)")
    return result


def _fetch_wind_speed(lat, lon):
    """
    Mean 2020 ERA5 10m wind speed at (lat, lon), via Earth Engine.
    Averages the daily speed magnitude (mean of |wind|), not the magnitude of
    the daily-averaged (u,v) vector -- the latter cancels out most of the
    speed over a year as wind direction rotates, and badly understates the
    true mean speed (fine for wind_check.py's direction-only check, wrong here).
    """
    import ee
    ee.Initialize(project="opportune-lore-415218")
    pt = ee.Geometry.Point([lon, lat])
    coll = (ee.ImageCollection("ECMWF/ERA5/DAILY")
            .select(["u_component_of_wind_10m", "v_component_of_wind_10m"])
            .filterDate("2020-01-01", "2020-12-31"))

    def add_speed(img):
        u = img.select("u_component_of_wind_10m")
        v = img.select("v_component_of_wind_10m")
        return u.hypot(v).rename("speed")

    mean_speed_img = coll.map(add_speed).mean()
    vals = mean_speed_img.reduceRegion(ee.Reducer.mean(), pt, scale=27830).getInfo()
    return float(vals["speed"])


if __name__ == "__main__":
    rows = json.load(open("data/plant_results.json"))

    results = []
    for row in rows:
        try:
            speed = _fetch_wind_speed(row["lat"], row["lon"])
        except Exception as e:
            print(f"[{row['plant']}] wind fetch failed: {str(e)[:60]}, skipping")
            continue
        r = estimate_emission_rate(row, speed)
        if r:
            results.append(r)

    out_path = "data/emission_estimates.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\n[SAVED] {len(results)} plant emission estimates -> {out_path}")
