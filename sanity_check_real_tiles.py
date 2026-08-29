"""
Week 21 addendum: QUALITATIVE-ONLY sanity check of the trained
segmentation U-Net (unet_segmentation.pt) on real OCO-3 tiles for
Vindhyachal and Rihand -- the paper's best/worst IME cases, chosen for
continuity with the rest of this project (see RESEARCH_PAPER.md Sec 9).

HARD CONSTRAINT: there is no real ground-truth segmentation mask for any
real facility -- that is exactly why Week 21 trained on simulated data
in the first place. This script computes NO Dice/IoU/accuracy number
against real tiles. It only produces an inspectable figure (real tile +
predicted mask overlay) and is meant to be read qualitatively: does the
predicted mask fall over a plausible plume-shaped region, or is it
visibly nonsensical / firing on background noise. See WEEK21_LOG.txt's
qualitative addendum for the actual read of these figures.

Real tiles are built by gridding real OCO-3 soundings
(data/<plant>_soundings.npz: lat, lon, xco2, day) onto the SAME 60km/
64px grid convention used throughout this project (SIZE_KM=60, PX=64,
~937.5m/px) and the SAME km/111deg conversion used elsewhere
(physics_ime.py, simulate_training_pairs.py) -- centered on the plant,
using its single BEST-COVERED real overpass day (the day with the most
in-tile soundings), since the model was trained on single-snapshot
tiles, not multi-day composites. Grid cells with no real soundings that
day are left NaN, handled by the same fill/validity-mask logic used in
training.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from unet_segmentation import UNetSmall, device_str, fill_nan_and_validity

KM_PER_DEG = 111.0
SIZE_KM, PX = 60, 64
CHECKPOINT_PATH = "unet_segmentation.pt"
RESULTS_PATH = "data/unet_segmentation_results.json"
OUT_FIG_PATH = "data/unet_segmentation_real_tile_sanity_check.png"
MASK_THRESHOLD = 0.5

PLANTS = ["Vindhyachal", "Rihand"]
SOUNDING_FILES = {
    "Vindhyachal": "data/vindhyachal_soundings.npz",
    "Rihand": "data/Rihand_soundings.npz",
}


def grid_real_tile(lat, lon, xco2, plat, plon):
    """Bins real soundings onto the project's standard 60km/64px grid,
    averaging xco2 within each cell; empty cells are NaN."""
    east_km = (lon - plon) * KM_PER_DEG
    north_km = (lat - plat) * KM_PER_DEG
    half = SIZE_KM / 2.0
    px_size = SIZE_KM / PX

    col = np.floor((east_km + half) / px_size).astype(int)
    row = np.floor((north_km + half) / px_size).astype(int)
    in_bounds = (col >= 0) & (col < PX) & (row >= 0) & (row < PX)

    tile = np.full((PX, PX), np.nan, dtype=np.float32)
    sums = np.zeros((PX, PX), dtype=np.float64)
    counts = np.zeros((PX, PX), dtype=np.int64)
    for r, c, v in zip(row[in_bounds], col[in_bounds], xco2[in_bounds]):
        sums[r, c] += v
        counts[r, c] += 1
    has_data = counts > 0
    tile[has_data] = (sums[has_data] / counts[has_data]).astype(np.float32)
    return tile, has_data.sum()


def best_covered_day(lat, lon, day, plat, plon):
    east_km = (lon - plon) * KM_PER_DEG
    north_km = (lat - plat) * KM_PER_DEG
    in_tile = (np.abs(east_km) < SIZE_KM / 2) & (np.abs(north_km) < SIZE_KM / 2)
    days, counts = np.unique(day[in_tile], return_counts=True)
    best = days[np.argmax(counts)]
    return int(best), int(counts.max())


def main():
    pr = json.load(open("data/plant_results.json"))
    plant_latlon = {p["plant"]: (p["lat"], p["lon"]) for p in pr}
    results = json.load(open(RESULTS_PATH))
    train_mean, train_std = results["train_mean"], results["train_std"]

    model = UNetSmall(in_channels=1, base_channels=16)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()

    fig, axes = plt.subplots(len(PLANTS), 3, figsize=(9, 3 * len(PLANTS)))

    print("=== Week 21 addendum: QUALITATIVE-ONLY real-tile sanity check ===")
    print("No accuracy metric is computed -- no real ground-truth mask exists.\n")

    for row_i, name in enumerate(PLANTS):
        d = np.load(SOUNDING_FILES[name])
        lat, lon, xco2, day = d["lat"], d["lon"], d["xco2"], d["day"]
        plat, plon = plant_latlon[name]

        best_day, n_soundings = best_covered_day(lat, lon, day, plat, plon)
        day_mask = day == best_day
        tile, n_cells_with_data = grid_real_tile(
            lat[day_mask], lon[day_mask], xco2[day_mask], plat, plon)

        filled, valid = fill_nan_and_validity(tile[None, :, :])
        norm = (filled[0] - train_mean) / train_std
        x = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            prob = torch.sigmoid(model(x))[0, 0].numpy()
        pred_mask = prob > MASK_THRESHOLD
        pred_display = np.where(valid[0], pred_mask.astype(float), 0.5)
        frac_valid = float(valid[0].mean())
        frac_predicted_plume = float((pred_mask & valid[0]).mean())

        print(f"{name}: best-covered real overpass day {best_day}, "
              f"{n_soundings} soundings in-tile -> {n_cells_with_data}/{PX*PX} grid cells "
              f"with real data ({frac_valid:.1%} coverage). "
              f"Predicted plume pixels: {frac_predicted_plume:.1%} of tile.")

        axes[row_i, 0].imshow(tile, cmap="viridis")
        axes[row_i, 0].set_title(f"{name}: real XCO2 tile\n(day {best_day}, {frac_valid:.0%} coverage)")
        axes[row_i, 1].imshow(valid[0], cmap="gray", vmin=0, vmax=1)
        axes[row_i, 1].set_title("real data coverage\n(white=real sounding, black=no data)")
        axes[row_i, 2].imshow(pred_display, cmap="gray", vmin=0, vmax=1)
        axes[row_i, 2].set_title("predicted plume mask\n(gray=no data, QUALITATIVE ONLY)")
        for ax in axes[row_i]:
            ax.axis("off")

    plt.suptitle("Week 21 addendum -- QUALITATIVE sanity check only, NOT a validation\n"
                  "(no real ground-truth mask exists to score against)", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_FIG_PATH, dpi=100)
    plt.close()
    print(f"\nFigure saved to {OUT_FIG_PATH}")
    print("See WEEK21_LOG.txt for the qualitative visual description.")


if __name__ == "__main__":
    main()
