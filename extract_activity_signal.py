"""
Extract an activity signal from Track A's already-trained 3-channel CNN
(NO2+SO2+VIIRS combustion-activity classifier) for each of Track B's 9
facilities (data/plant_results.json), per RESEARCH_PLAN.md's roadmap step 3:
"use the already-trained NO2/SO2/VIIRS CNN's activity signal (calibrated
probability or an intermediate embedding) as a covariate in a small,
physically interpretable correction model" for Track B's IME estimate.

Per RESEARCH_PLAN.md Sec 15, the CNN is reused AS-IS -- not retrained --
so this script only does inference, using the "2ch_mixed" checkpoint
(detector3_2ch_mixed.pt, trained on plants vs. all negatives: hard
negatives + rural), the more generally-applicable of the two saved
checkpoints for out-of-training-distribution facilities like these.

Normalization caveat: train_3channel.py normalizes each channel using
mean/std computed from that run's own training set, but never saves those
stats to disk -- only the model weights. To apply the same normalization
at inference time, this script reloads the original training tiles
(data/threech/positive + hard_negative + negative) and recomputes the
identical per-channel mean/std train_3channel.py would have computed for
the "2ch_mixed" run. This is deterministic (no randomness in the mean/std
computation itself, only in the train/test split), so it exactly
reproduces the training-time normalization.
"""
import json
import glob
import numpy as np
import torch
import torch.nn as nn

DEVICE = "cpu"
CHECKPOINT = "detector3_2ch_mixed.pt"

# facility short name (data/plant_results.json) -> Track A export filename
# prefix, for the 4 facilities already covered by the original 5-plant
# export (export_monthly.py/export_so2.py/export_viirs.py). The 5 facilities
# newly exported this week (export_facility_tiles.py) use their own short
# name directly as the filename prefix, so no aliasing needed for those.
TRACK_A_ALIAS = {
    "Vindhyachal": "VINDH_CHAL_STPS",
    "Sasan": "SASAN_UMPP",
    "Mundra": "MUNDRA_TPP",
    "Tirora": "TIRORA_TPP",
}


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

    def forward_with_embedding(self, x):
        feat = self.net[:-1](x)       # (N, 64), pre-final-Linear
        logits = self.net[-1](feat)   # (N, 2)
        return logits, feat


def load_folder(path):
    return [np.load(f).astype(np.float32) for f in sorted(glob.glob(f"{path}/*.npy"))]


def training_norm_stats():
    """Reproduce train_3channel.py's per-channel mean/std for the
    'mixed' run (plants + hard_negative + rural), exactly as it computes
    them (over the full stacked X before any train/test split)."""
    Xp = load_folder("data/threech/positive")
    Xh = load_folder("data/threech/hard_negative")
    Xr = load_folder("data/threech/negative")
    X = np.stack(Xp + Xh + Xr)  # (N,3,64,64)
    means, stds = [], []
    for c in range(3):
        m, s = X[:, c].mean(), X[:, c].std() + 1e-12
        means.append(m)
        stds.append(s)
    return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)


def build_tile(no2_path, so2_path, viirs_path):
    no2 = np.load(no2_path).astype(np.float32)
    so2 = np.load(so2_path).astype(np.float32)
    viirs = np.load(viirs_path).astype(np.float32)
    if no2.shape != so2.shape or no2.shape != viirs.shape:
        return None
    no2 = np.where(np.isfinite(no2), no2, np.nanmean(no2))
    so2 = np.where(np.isfinite(so2), so2, np.nanmean(so2))
    viirs = np.where(np.isfinite(viirs), viirs, np.nanmean(viirs))
    so2 = np.clip(so2, 0, None)
    viirs = np.clip(viirs, 0, None)
    return np.stack([no2, so2, viirs], axis=0)  # (3,64,64)


def monthly_paths(name, year=2020):
    if name in TRACK_A_ALIAS:
        alias = TRACK_A_ALIAS[name]
        no2_dir, so2_dir, viirs_dir = "data/monthly/positive", "data/so2/positive", "data/viirs/positive"
        prefix = alias
    else:
        no2_dir, so2_dir, viirs_dir = ("data/activity_tiles/no2", "data/activity_tiles/so2",
                                       "data/activity_tiles/viirs")
        prefix = name
    paths = []
    for m in range(1, 13):
        n = f"{no2_dir}/{prefix}_{year}_{m:02d}.npy"
        s = f"{so2_dir}/{prefix}_{year}_{m:02d}.npy"
        v = f"{viirs_dir}/{prefix}_{year}_{m:02d}.npy"
        paths.append((n, s, v))
    return paths


def main():
    import os
    model = Detector3().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    mean, std = training_norm_stats()
    print(f"Recomputed training normalization: mean={mean}, std={std}")

    rows = json.load(open("data/plant_results.json"))
    results = []
    for row in rows:
        name = row["plant"]
        probs, embeds = [], []
        for no2_p, so2_p, viirs_p in monthly_paths(name):
            if not (os.path.exists(no2_p) and os.path.exists(so2_p) and os.path.exists(viirs_p)):
                continue
            tile = build_tile(no2_p, so2_p, viirs_p)
            if tile is None:
                continue
            norm = (tile - mean[:, None, None]) / std[:, None, None]
            x = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits, feat = model.forward_with_embedding(x)
                prob = torch.softmax(logits, dim=1)[0, 1].item()
            probs.append(prob)
            embeds.append(feat[0].numpy())

        if not probs:
            print(f"[{name}] no usable monthly tiles, skipping")
            continue

        probs = np.array(probs)
        embed_mean = np.mean(np.stack(embeds), axis=0)
        result = {
            "plant": name,
            "n_months": len(probs),
            "activity_prob_mean": float(probs.mean()),
            "activity_prob_std": float(probs.std()) if len(probs) > 1 else 0.0,
            "embedding_mean": embed_mean.tolist(),
        }
        results.append(result)
        print(f"[{name}] n_months={len(probs)}  activity_prob={result['activity_prob_mean']:.3f} "
              f"+/- {result['activity_prob_std']:.3f}")

    json.dump(results, open("data/activity_signals.json", "w"), indent=2)
    print(f"\n[SAVED] {len(results)} facility activity signals -> data/activity_signals.json")


if __name__ == "__main__":
    main()
