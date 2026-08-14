"""
Fetches per-day ERA5 wind DIRECTION (not just speed) for every facility in
build_plume_maps.ALL_FACILITIES -- the data needed to test each OCO-3
sounding against the actual wind direction on ITS OWN overpass day,
instead of one annual/overpass-averaged direction applied to every
sounding regardless of date.

physics_gaussian.py's _fetch_wind_series() already pulls this exact same
ERA5 collection (ECMWF/ERA5/DAILY) per day, but only keeps the SPEED
(hypot(u,v)) -- direction is computed nowhere per-day in this codebase;
process_plant.py's existing wind_deg is a single ANNUAL mean, not
per-day. This is the genuinely new data pull needed for the day-matching
fix proposed in NOVEL_METHODOLOGY_PROPOSAL.md Sec 11.

Direction convention: same as plant_results.json's existing wind_deg
field (degrees(atan2(u,v)) -- the direction wind blows TOWARD, per
process_plant.py's own "toward X deg" print), for direct consistency
with the rest of this project's wind-direction handling. Downstream
scripts must convert to the "FROM" convention plume_model.py expects,
exactly as build_plume_maps.py already does for the annual value.

Caches one JSON per facility: {"YYYYMMDD": direction_toward_deg, ...}.
Skips facilities whose cache already exists (same re-run-safe pattern as
export_new_positive_tiles.py).
"""
import json
import os
import time

import ee
import numpy as np

from build_plume_maps import eligible_facilities

ee.Initialize(project="opportune-lore-415218")

YEARS = (2019, 2020)


def fetch_daily_wind_direction(lat, lon, years):
    pt = ee.Geometry.Point([lon, lat])
    directions = {}
    for year in years:
        coll = (ee.ImageCollection("ECMWF/ERA5/DAILY")
                .select(["u_component_of_wind_10m", "v_component_of_wind_10m"])
                .filterDate(f"{year}-01-01", f"{year}-12-31"))
        rows = coll.getRegion(pt, scale=27830).getInfo()
        header, records = rows[0], rows[1:]
        i_u = header.index("u_component_of_wind_10m")
        i_v = header.index("v_component_of_wind_10m")
        i_t = header.index("time")
        for row in records:
            u, v, t_ms = row[i_u], row[i_v], row[i_t]
            if u is None or v is None or t_ms is None:
                continue
            day = time.strftime("%Y%m%d", time.gmtime(t_ms / 1000))
            direction_toward_deg = float(np.degrees(np.arctan2(u, v)) % 360)
            directions[day] = direction_toward_deg
    return directions


def main():
    os.makedirs("data/daily_wind", exist_ok=True)
    plant_results = {r["plant"]: r for r in json.load(open("data/plant_results.json"))}
    emission_estimates = {r["plant"]: r for r in json.load(open("data/emission_estimates.json"))}
    facilities = eligible_facilities(plant_results, emission_estimates)
    print(f"Fetching daily wind direction for {len(facilities)} facilities, years {YEARS}")

    for name in facilities:
        out_path = f"data/daily_wind/{name}_daily_wind.json"
        if os.path.exists(out_path):
            print(f"[{name}] cache exists, skipping")
            continue
        pr = plant_results[name]
        try:
            directions = fetch_daily_wind_direction(pr["lat"], pr["lon"], YEARS)
        except Exception as e:
            print(f"[{name}] FAILED: {str(e)[:80]}")
            continue
        json.dump(directions, open(out_path, "w"))
        print(f"[{name}] {len(directions)} days cached -> {out_path}")


if __name__ == "__main__":
    main()
