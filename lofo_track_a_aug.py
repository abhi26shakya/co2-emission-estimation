"""
Follow-up to lofo_track_a.py testing whether light data augmentation closes
any of the generalization gap it exposed (mean recall 47.2%, tile-weighted
48.7%). Motivated by two observations from that run and lofo_recall_correlates.py:

1. activity_prob_mean predicts per-facility LOFO recall almost perfectly
   (r=+0.90) -- weak/ambiguous combustion signature predicts failure.
2. The 16 newly-added facilities have only 12 tiles each (2020 only) vs. 24
   for the original top-5 (2019-2020) -- half the temporal diversity per
   facility, and LOFO test folds as small as n=12.

Augmentation can't manufacture new *signal*, so it won't fix (1). But it
can act as a cheap variance-reduction lever against (2): with no new Earth
Engine pulls, this checks whether some of the 0%-recall facilities were
underfit specifically due to limited training-set diversity per fold,
before paying for a real fix (a second data year -- see
export_new_positive_tiles_2019.py).

Identical to lofo_track_a.py in every respect (same folds, same seeds, same
epochs, same architecture) except: training tiles get a random one of
{identity, horizontal flip, vertical flip, 90-degree rotation} applied per
epoch (spatial augmentations only -- NO2/SO2/VIIRS tiles are geographically
symmetric-agnostic, a stack of raw physical quantities, so flips/rotations
are label-preserving; per-channel intensity is untouched to avoid distorting
the physical units of column density / FRP).
"""
import numpy as np, glob, os, re, random, json, time, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 30
_MONTH_SUFFIX = re.compile(r"_\d{4}_\d{2}\.npy$")


def load_folder(path, label):
    X, y, groups = [], [], []
    for f in sorted(glob.glob(f"{path}/*.npy")):
        arr = np.load(f).astype(np.float32)
        X.append(arr); y.append(label)
        groups.append(_MONTH_SUFFIX.sub("", os.path.basename(f)))
    return X, y, groups


class Detector3(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, x):
        return self.net(x)


class AugmentedTiles(Dataset):
    """Applies a random dihedral-group-4 transform (identity / hflip / vflip
    / 90-deg rotation) per __getitem__ call, re-sampled every epoch since
    DataLoader re-iterates the Dataset each epoch."""
    def __init__(self, X, y):
        self.X = X  # (N, 3, 64, 64) normalized
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        tile = self.X[idx]
        choice = random.randint(0, 3)
        if choice == 1:
            tile = np.flip(tile, axis=1)
        elif choice == 2:
            tile = np.flip(tile, axis=2)
        elif choice == 3:
            tile = np.rot90(tile, k=1, axes=(1, 2))
        return torch.tensor(np.ascontiguousarray(tile)), self.y[idx]


def train_and_eval(Xtr, ytr, Xte, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    Xtr_arr = np.stack(Xtr).astype(np.float32)
    mean = np.array([Xtr_arr[:, c].mean() for c in range(3)], dtype=np.float32)
    std = np.array([Xtr_arr[:, c].std() + 1e-12 for c in range(3)], dtype=np.float32)
    Xtr_norm = (Xtr_arr - mean[None, :, None, None]) / std[None, :, None, None]

    ytr_arr = np.array(ytr, dtype=np.int64)
    tr_dl = DataLoader(AugmentedTiles(Xtr_norm, torch.tensor(ytr_arr)),
                        batch_size=16, shuffle=True)

    model = Detector3().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    lossf = nn.CrossEntropyLoss()
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()

    Xte_arr = np.stack(Xte).astype(np.float32)
    Xte_norm = (Xte_arr - mean[None, :, None, None]) / std[None, :, None, None]

    model.eval()
    with torch.no_grad():
        xb = torch.tensor(Xte_norm, dtype=torch.float32).to(DEVICE)
        pred = model(xb).argmax(1).cpu().numpy()
    recall = float((pred == 1).mean())
    return recall


def main():
    Xp, yp, gp = load_folder("data/threech/positive", 1)
    Xh, yh, gh = load_folder("data/threech/hard_negative", 0)
    Xr, yr, gr = load_folder("data/threech/negative", 0)

    plant_groups = sorted(set(gp))
    print(f"LOFO+aug harness: {len(plant_groups)} plant facilities, "
          f"{len(Xh)} hard negatives + {len(Xr)} rural negatives held fixed in train each fold\n")

    baseline = {r["plant"]: r["recall"]
                for r in json.load(open("data/lofo_track_a_results.json"))["per_facility"]}

    results = []
    t0 = time.time()
    for i, held_out in enumerate(plant_groups):
        tr_idx = [j for j, g in enumerate(gp) if g != held_out]
        te_idx = [j for j, g in enumerate(gp) if g == held_out]
        Xtr = [Xp[j] for j in tr_idx] + Xh + Xr
        ytr = [yp[j] for j in tr_idx] + yh + yr
        Xte = [Xp[j] for j in te_idx]

        recall = train_and_eval(Xtr, ytr, Xte, seed=i)
        base = baseline.get(held_out)
        results.append({"plant": held_out, "n_tiles": len(te_idx), "recall": recall,
                         "baseline_recall": base,
                         "delta": (recall - base) if base is not None else None})
        elapsed = time.time() - t0
        delta_str = f"  delta={recall - base:+.2f}" if base is not None else ""
        print(f"[{i+1}/{len(plant_groups)}] {held_out:20s} n_tiles={len(te_idx):2d}  "
              f"recall={recall:.3f}{delta_str}  (elapsed {elapsed:.0f}s)")

    mean_recall = float(np.mean([r["recall"] for r in results]))
    total_tiles = sum(r["n_tiles"] for r in results)
    weighted_recall = sum(r["recall"] * r["n_tiles"] for r in results) / total_tiles

    print(f"\n=== LOFO+aug summary (N={len(results)} facilities) ===")
    print(f"  unweighted mean recall: {mean_recall:.3f}  (baseline: 0.472)")
    print(f"  tile-weighted mean recall: {weighted_recall:.3f}  (baseline: 0.487)")
    worst = sorted(results, key=lambda r: r["recall"])[:5]
    print(f"  worst-generalizing facilities: {[(r['plant'], round(r['recall'],2)) for r in worst]}")

    out = {
        "epochs": EPOCHS,
        "n_facilities": len(results),
        "per_facility": results,
        "mean_recall": mean_recall,
        "tile_weighted_mean_recall": weighted_recall,
        "baseline_mean_recall": 0.4722222222222222,
        "baseline_tile_weighted_mean_recall": 0.48717948717948717,
        "note": ("Same 21-fold LOFO structure as lofo_track_a.py, but training tiles get a "
                 "random dihedral-4 spatial transform (identity/hflip/vflip/90-rot) applied "
                 "per-epoch via AugmentedTiles. Tests whether the baseline LOFO recall gap "
                 "was partly a training-set-diversity artifact (16 of 20 facilities have only "
                 "12 tiles each) rather than purely a weak-signal problem."),
    }
    json.dump(out, open("data/lofo_track_a_aug_results.json", "w"), indent=2)
    print("\n[SAVED] data/lofo_track_a_aug_results.json")


if __name__ == "__main__":
    main()
