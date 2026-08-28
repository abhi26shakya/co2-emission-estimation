"""
Week 16: regional XCO2 context map -- closes the mapping-track item from
the project blueprint (same descriptive spirit as the STEL / DNN-CAMS
regional-XCO2-context papers surveyed there). Answers "what does
background CO2 look like across the study region," for the paper's
Methods/Data section -- NOT a new prediction model, no fitting, no
per-plant emission estimate. Every plant's own Q estimate still comes
from physics_gaussian.py (IME) or physics_gaussian_crosssection.py; this
script only visualizes the raw pooled XCO2 field those estimates are
drawn from.

Data: reuses the OCO-3 soundings already downloaded for all 30 plants
(data/*_soundings.npz) -- no new download, no Earth Engine call. Every
sounding from every plant is pooled into one dataset (soundings are not
plant-exclusive; a plant's "background" soundings from co2_enhancement.py
/ process_plant.py's own near/background zones are already, incidentally,
regional-context soundings for whichever OTHER plants happen to be
nearby -- pooling makes that overlap explicit and useful instead of
implicit and wasted).

Grid: 0.5-degree cells, per user's explicit spec (not scanned/tuned).
Bounding box: the 30 plants' own lat/lon extent (14.70-29.92 N,
69.55-88.10 E) plus a 1.0-degree margin on each side, rounded outward to
the nearest 0.5-degree grid line for clean edges -- 13.5 to 30.5 N,
69.0 to 88.5 E (34 x 39 = 1,326 possible cells; most will be empty/masked
away from plant clusters, which is expected for a descriptive map of
where OCO-3 actually has coverage, not an interpolated/smoothed product).

Per-cell statistic: mean XCO2 (ppm), masked as NaN (not zero -- "no
data" must not read as "low CO2") when a cell has fewer than
MIN_SOUNDINGS_PER_CELL soundings. Per explicit direction, this floor is
3 (not just >0): 1-2 soundings is not enough to trust a per-cell mean,
so those cells are masked identically to empty cells, and the threshold
itself is recorded in the output JSON's metadata
(n_soundings_min_required) so the choice is documented, not silent.
"""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GRID_RES_DEG = 0.5
MARGIN_DEG = 1.0
MIN_SOUNDINGS_PER_CELL = 3
OUT_PNG = "data/regional_xco2_map.png"
OUT_JSON = "data/regional_xco2_grid.json"


def load_pooled_soundings():
    lats, lons, xco2s = [], [], []
    files = sorted(glob.glob("data/*_soundings.npz"))
    for f in files:
        d = np.load(f)
        lats.append(d["lat"]); lons.append(d["lon"]); xco2s.append(d["xco2"])
    lat = np.concatenate(lats)
    lon = np.concatenate(lons)
    xco2 = np.concatenate(xco2s)
    print(f"Pooled {len(xco2):,} soundings from {len(files)} plant files")
    return lat, lon, xco2


def compute_bbox(plant_rows):
    lats = np.array([r["lat"] for r in plant_rows])
    lons = np.array([r["lon"] for r in plant_rows])
    lat_min = np.floor((lats.min() - MARGIN_DEG) / GRID_RES_DEG) * GRID_RES_DEG
    lat_max = np.ceil((lats.max() + MARGIN_DEG) / GRID_RES_DEG) * GRID_RES_DEG
    lon_min = np.floor((lons.min() - MARGIN_DEG) / GRID_RES_DEG) * GRID_RES_DEG
    lon_max = np.ceil((lons.max() + MARGIN_DEG) / GRID_RES_DEG) * GRID_RES_DEG
    return float(lat_min), float(lat_max), float(lon_min), float(lon_max)


def bin_grid(lat, lon, xco2, lat_min, lat_max, lon_min, lon_max):
    n_lat = int(round((lat_max - lat_min) / GRID_RES_DEG))
    n_lon = int(round((lon_max - lon_min) / GRID_RES_DEG))

    in_box = (lat >= lat_min) & (lat < lat_max) & (lon >= lon_min) & (lon < lon_max)
    lat_i = np.floor((lat[in_box] - lat_min) / GRID_RES_DEG).astype(int)
    lon_i = np.floor((lon[in_box] - lon_min) / GRID_RES_DEG).astype(int)
    xco2_in = xco2[in_box]

    sum_grid = np.zeros((n_lat, n_lon))
    count_grid = np.zeros((n_lat, n_lon), dtype=int)
    np.add.at(sum_grid, (lat_i, lon_i), xco2_in)
    np.add.at(count_grid, (lat_i, lon_i), 1)

    mean_grid = np.full((n_lat, n_lon), np.nan)
    enough = count_grid >= MIN_SOUNDINGS_PER_CELL
    mean_grid[enough] = sum_grid[enough] / count_grid[enough]

    n_soundings_used = int(xco2_in.size)
    n_soundings_outside_bbox = int((~in_box).sum())
    return mean_grid, count_grid, n_lat, n_lon, n_soundings_used, n_soundings_outside_bbox


def main():
    plant_rows = json.load(open("data/plant_results.json"))
    cea = json.load(open("data/cea_ground_truth_2020_21.json"))["facilities"]

    lat, lon, xco2 = load_pooled_soundings()
    lat_min, lat_max, lon_min, lon_max = compute_bbox(plant_rows)
    print(f"Bounding box: lat [{lat_min}, {lat_max}]  lon [{lon_min}, {lon_max}]  "
          f"(0.5-degree margin=1.0 each side, rounded to grid)")

    mean_grid, count_grid, n_lat, n_lon, n_used, n_outside = bin_grid(
        lat, lon, xco2, lat_min, lat_max, lon_min, lon_max)
    n_cells_total = n_lat * n_lon
    n_cells_with_data = int((count_grid >= MIN_SOUNDINGS_PER_CELL).sum())
    n_cells_masked_sparse = int(((count_grid > 0) & (count_grid < MIN_SOUNDINGS_PER_CELL)).sum())
    n_cells_empty = int((count_grid == 0).sum())
    print(f"Grid: {n_lat} x {n_lon} = {n_cells_total} cells  "
          f"({n_cells_with_data} with >= {MIN_SOUNDINGS_PER_CELL} soundings, "
          f"{n_cells_masked_sparse} masked as too-sparse (1-{MIN_SOUNDINGS_PER_CELL - 1}), "
          f"{n_cells_empty} empty)")
    print(f"{n_used:,} soundings used, {n_outside:,} fell outside the bounding box")

    # --- plot ---
    fig, ax = plt.subplots(figsize=(11, 8))
    lat_edges = np.arange(lat_min, lat_max + GRID_RES_DEG / 2, GRID_RES_DEG)
    lon_edges = np.arange(lon_min, lon_max + GRID_RES_DEG / 2, GRID_RES_DEG)
    masked = np.ma.masked_invalid(mean_grid)
    pc = ax.pcolormesh(lon_edges, lat_edges, masked, cmap="viridis", shading="flat")
    cbar = fig.colorbar(pc, ax=ax)
    cbar.set_label("mean XCO2 (ppm)")

    plant_lats = np.array([r["lat"] for r in plant_rows])
    plant_lons = np.array([r["lon"] for r in plant_rows])
    plant_names = [r["plant"] for r in plant_rows]
    cea_vals = np.array([cea[p]["abs_emissions_t_co2"] for p in plant_names])
    sizes = 20 + 180 * (cea_vals - cea_vals.min()) / (cea_vals.max() - cea_vals.min())
    sc = ax.scatter(plant_lons, plant_lats, s=sizes, c=cea_vals, cmap="autumn_r",
                     edgecolors="black", linewidths=0.8, zorder=3)
    cbar2 = fig.colorbar(sc, ax=ax, location="left", pad=0.08)
    cbar2.set_label("CEA absolute CO2 emissions (t/yr)")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Regional XCO2 context ({n_cells_with_data} cells, "
                 f">= {MIN_SOUNDINGS_PER_CELL} soundings/cell) + 30 plant locations")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"\n[SAVED] {OUT_PNG}")

    # --- JSON: metadata + non-masked cells only ---
    cells = []
    for i in range(n_lat):
        for j in range(n_lon):
            if count_grid[i, j] >= MIN_SOUNDINGS_PER_CELL:
                cells.append({
                    "lat_bin_center": round(lat_min + (i + 0.5) * GRID_RES_DEG, 3),
                    "lon_bin_center": round(lon_min + (j + 0.5) * GRID_RES_DEG, 3),
                    "mean_xco2_ppm": round(float(mean_grid[i, j]), 4),
                    "n_soundings": int(count_grid[i, j]),
                })

    out = {
        "grid_resolution_deg": GRID_RES_DEG,
        "margin_deg": MARGIN_DEG,
        "bounding_box": {"lat_min": lat_min, "lat_max": lat_max,
                          "lon_min": lon_min, "lon_max": lon_max},
        "n_soundings_min_required": MIN_SOUNDINGS_PER_CELL,
        "note": ("Cells with fewer than n_soundings_min_required soundings are masked "
                 "(excluded from `cells` below and shown as NaN/blank in the PNG), "
                 "identically to empty cells -- a per-cell mean from 1-2 soundings is "
                 "not trustworthy enough to report. This floor is a deliberate choice, "
                 "not a silent default."),
        "n_plants": len(plant_rows),
        "n_soundings_pooled": int(xco2.size),
        "n_soundings_used_in_bbox": n_used,
        "n_soundings_outside_bbox": n_outside,
        "n_cells_total": n_cells_total,
        "n_cells_with_data": n_cells_with_data,
        "n_cells_masked_sparse_1_to_min_minus_1": n_cells_masked_sparse,
        "n_cells_empty": n_cells_empty,
        "cells": cells,
    }
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"[SAVED] {OUT_JSON}")


if __name__ == "__main__":
    main()
