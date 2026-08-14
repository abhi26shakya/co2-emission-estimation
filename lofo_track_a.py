"""
General leave-one-facility-out (LOFO) CV harness for Track A -- roadmap
step 5 from NEXT_STEPS.md, which was previously only satisfied ad hoc by
Week 11's single random 80/20 facility-level split in train_3channel.py's
facility_level_split(). A single random split only ever tests generalization
on whichever handful of facilities happened to land in that one test set
(4 plant facilities for the "mixed" run, per WEEK11_LOG.txt / this
session's retrain). This script instead holds out each of the 20 plant
facilities in turn, training a fresh model on the rest each time, so every
facility gets exactly one turn as a genuinely unseen test case.

Mirrors train_3channel.py's "2ch_mixed" configuration (plants vs. ALL
negatives: hard_negative + rural) since that's the more generally-
applicable checkpoint per extract_activity_signal.py's own rationale.
Negatives are never held out here -- only the positive (plant) class is
cycled through, since the question this harness answers is specifically
"does the detector generalize to an unseen physical plant," not negative-
class generalization (which Week 7-10's hard-negative expansion already
targets separately).

This trains 20 fresh models (one per held-out facility) rather than
reusing the single saved checkpoint -- LOFO by definition requires that
each fold's model never sees its own held-out facility during training,
which the single "detector3_2ch_mixed_facility_split.pt" checkpoint from
train_3channel.py does not guarantee for 16 of the 20 facilities (those
were in that run's training set).
"""
import numpy as np, glob, os, re, random, json, time, torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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


def train_and_eval(Xtr, ytr, Xte, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    Xtr_arr = np.stack(Xtr).astype(np.float32)
    mean = np.array([Xtr_arr[:, c].mean() for c in range(3)], dtype=np.float32)
    std = np.array([Xtr_arr[:, c].std() + 1e-12 for c in range(3)], dtype=np.float32)
    Xtr_norm = (Xtr_arr - mean[None, :, None, None]) / std[None, :, None, None]

    ytr_arr = np.array(ytr, dtype=np.int64)
    tr_dl = DataLoader(TensorDataset(torch.tensor(Xtr_norm), torch.tensor(ytr_arr)),
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


def facility_fold_indices(groups, held_out):
    """
    Split a list of per-tile facility-group labels into (train_idx, test_idx)
    for one LOFO fold: every tile whose group != held_out goes to train,
    every tile whose group == held_out goes to test. Factored out of main()
    so it's directly unit-testable -- this exact split (by facility, not by
    tile) is what Week 11's leakage bug got wrong (a random tile-level split
    let the same facility's tiles land in both train and test), so this
    function is the one piece of this harness a regression here would be
    most costly to miss silently.
    """
    tr_idx = [j for j, g in enumerate(groups) if g != held_out]
    te_idx = [j for j, g in enumerate(groups) if g == held_out]
    return tr_idx, te_idx


def main():
    Xp, yp, gp = load_folder("data/threech/positive", 1)
    Xh, yh, gh = load_folder("data/threech/hard_negative", 0)
    Xr, yr, gr = load_folder("data/threech/negative", 0)

    plant_groups = sorted(set(gp))
    print(f"LOFO harness: {len(plant_groups)} plant facilities, "
          f"{len(Xh)} hard negatives + {len(Xr)} rural negatives held fixed in train each fold\n")

    results = []
    t0 = time.time()
    for i, held_out in enumerate(plant_groups):
        tr_idx, te_idx = facility_fold_indices(gp, held_out)
        Xtr = [Xp[j] for j in tr_idx] + Xh + Xr
        ytr = [yp[j] for j in tr_idx] + yh + yr
        Xte = [Xp[j] for j in te_idx]

        recall = train_and_eval(Xtr, ytr, Xte, seed=i)
        results.append({"plant": held_out, "n_tiles": len(te_idx), "recall": recall})
        elapsed = time.time() - t0
        print(f"[{i+1}/{len(plant_groups)}] {held_out:20s} n_tiles={len(te_idx):2d}  "
              f"recall={recall:.3f}  (elapsed {elapsed:.0f}s)")

    mean_recall = float(np.mean([r["recall"] for r in results]))
    total_tiles = sum(r["n_tiles"] for r in results)
    weighted_recall = sum(r["recall"] * r["n_tiles"] for r in results) / total_tiles

    print(f"\n=== LOFO summary (N={len(results)} facilities) ===")
    print(f"  unweighted mean recall: {mean_recall:.3f}")
    print(f"  tile-weighted mean recall: {weighted_recall:.3f}")
    worst = sorted(results, key=lambda r: r["recall"])[:5]
    print(f"  worst-generalizing facilities: {[(r['plant'], round(r['recall'],2)) for r in worst]}")

    out = {
        "epochs": EPOCHS,
        "n_facilities": len(results),
        "per_facility": results,
        "mean_recall": mean_recall,
        "tile_weighted_mean_recall": weighted_recall,
        "note": ("Each row trains a fresh model with that one facility's tiles fully "
                 "excluded from training (all other 19 plant facilities + all hard/rural "
                 "negatives included). recall = fraction of the held-out facility's own "
                 "tiles classified as 'plant'. This is the true LOFO analogue of the single "
                 "random facility-split test recall reported in train_3channel.py."),
    }
    json.dump(out, open("data/lofo_track_a_results.json", "w"), indent=2)
    print("\n[SAVED] data/lofo_track_a_results.json")


if __name__ == "__main__":
    main()
