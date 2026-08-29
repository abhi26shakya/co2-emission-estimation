"""
Week 21: trains the segmentation-only U-Net (unet_segmentation.py) on
Week 20's simulated (XCO2 tile, plume mask) pairs
(data/simulated_train/simulated_tiles.npz). Predicts the plume MASK
only -- q_t_per_year is loaded here ONLY to report per-Q-bucket test
metrics (does the model do worse on low-signal tiles), never as a
training target. See unet_segmentation.py's module docstring and
WEEK21_LOG.txt for the scope boundary this respects.

WEEK 23: input is now 2-channel (unet_segmentation.build_model_input) --
channel 0 the normalized XCO2 tile, channel 1 the explicit valid/missing
mask -- to test whether Week 22's persistent real-tile coverage-
tracking behavior came from the model relying on the NaN-fill value's
boundary discontinuity as an implicit shortcut feature. See
WEEK23_LOG.txt. sanity_check_real_tiles.py calls the SAME
build_model_input() function on real tiles -- that is the only way this
comparison is meaningful (see WEEK23_LOG.txt's explicit consistency
check).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from unet_segmentation import (
    UNetSmall, build_model_input, device_str, dice_coefficient,
    facility_level_split, fill_nan_and_validity, iou_score, masked_bce_dice_loss,
)

SEED = 42
NPZ_PATH = "data/simulated_train/simulated_tiles.npz"
META_PATH = "data/simulated_train/simulated_tiles_meta.json"
RESULTS_PATH = "data/unet_segmentation_results.json"
CHECKPOINT_PATH = "unet_segmentation.pt"
QUALITATIVE_FIG_PATH = "data/unet_segmentation_qualitative.png"

BATCH_SIZE = 16
EPOCHS = 60
LR = 1e-3
MASK_THRESHOLD = 0.5  # predicted-probability threshold for binarizing IoU/Dice
IN_CHANNELS = 2  # WEEK 23: [normalized tile, explicit valid mask]


class TileMaskDataset(Dataset):
    def __init__(self, norm_tiles, masks, valid):
        # WEEK 23: 2-channel model input, built via the SAME
        # build_model_input() sanity_check_real_tiles.py uses on real
        # tiles -- see module docstring.
        x = np.stack([build_model_input(norm_tiles[i], valid[i]) for i in range(len(norm_tiles))])
        self.x = torch.from_numpy(x)                                    # (N,2,64,64)
        self.masks = torch.from_numpy(masks.astype(np.float32)).unsqueeze(1)
        self.valid = torch.from_numpy(valid.astype(np.float32)).unsqueeze(1)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.masks[i], self.valid[i]


def load_data():
    d = np.load(NPZ_PATH)
    tiles, masks, q_values = d["tiles"], d["masks"], d["q_t_per_year"]
    meta = json.load(open(META_PATH))
    params = meta["params"]
    assert len(params) == tiles.shape[0], "tiles/meta length mismatch"

    facility_ids = []
    neg_counter = -1
    for p in params:
        if p["positive"]:
            facility_ids.append(p["facility_id"])
        else:
            facility_ids.append(neg_counter)  # each negative tile is its own independent unit
            neg_counter -= 1
    facility_ids = np.array(facility_ids)
    return tiles, masks, q_values, facility_ids


def evaluate(model, loader, device):
    model.eval()
    dices, ious = [], []
    with torch.no_grad():
        for xb, yb, vb in loader:
            xb, yb, vb = xb.to(device), yb.to(device), vb.to(device)
            probs = torch.sigmoid(model(xb))
            pred = (probs > MASK_THRESHOLD).float()
            for i in range(xb.shape[0]):
                dices.append(dice_coefficient(pred[i], yb[i], vb[i]))
                ious.append(iou_score(pred[i], yb[i], vb[i]))
    return np.array(dices), np.array(ious)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = device_str()
    print(f"Device: {device}")

    tiles, masks, q_values, facility_ids = load_data()
    print(f"Loaded {len(tiles)} tiles "
          f"({int((q_values > 0).sum())} positive, {int((q_values == 0).sum())} negative)")

    filled_tiles, valid = fill_nan_and_validity(tiles)

    train_idx, val_idx, test_idx = facility_level_split(facility_ids, seed=SEED,
                                                          train_frac=0.7, val_frac=0.15)
    n_train_fac = len(set(facility_ids[i] for i in train_idx))
    n_val_fac = len(set(facility_ids[i] for i in val_idx))
    n_test_fac = len(set(facility_ids[i] for i in test_idx))
    print(f"Facility-level split: train={len(train_idx)} tiles/{n_train_fac} facilities, "
          f"val={len(val_idx)} tiles/{n_val_fac} facilities, "
          f"test={len(test_idx)} tiles/{n_test_fac} facilities")
    assert not (set(facility_ids[train_idx]) & set(facility_ids[test_idx])), "facility leakage: train/test"
    assert not (set(facility_ids[train_idx]) & set(facility_ids[val_idx])), "facility leakage: train/val"
    assert not (set(facility_ids[val_idx]) & set(facility_ids[test_idx])), "facility leakage: val/test"

    # Normalize using TRAIN-split statistics only (no leakage into val/test).
    train_mean = filled_tiles[train_idx].mean()
    train_std = filled_tiles[train_idx].std() + 1e-6
    norm_tiles = (filled_tiles - train_mean) / train_std
    print(f"Normalization (train-split stats): mean={train_mean:.3f} std={train_std:.3f}")

    train_ds = TileMaskDataset(norm_tiles[train_idx], masks[train_idx], valid[train_idx])
    val_ds = TileMaskDataset(norm_tiles[val_idx], masks[val_idx], valid[val_idx])
    test_ds = TileMaskDataset(norm_tiles[test_idx], masks[test_idx], valid[test_idx])

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = UNetSmall(in_channels=IN_CHANNELS, base_channels=16).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: UNetSmall, {n_params:,} parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_dice = -1.0
    best_state = None
    history = []
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb, vb in train_dl:
            xb, yb, vb = xb.to(device), yb.to(device), vb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = masked_bce_dice_loss(logits, yb, vb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        val_dices, val_ious = evaluate(model, val_dl, device)
        val_dice_mean = float(val_dices.mean())
        history.append(dict(epoch=epoch, train_loss=epoch_loss / n_batches, val_dice=val_dice_mean))
        if val_dice_mean > best_val_dice:
            best_val_dice = val_dice_mean
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:3d}  train_loss {epoch_loss / n_batches:.4f}  "
                  f"val_dice {val_dice_mean:.4f}")

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Saved best checkpoint (val_dice={best_val_dice:.4f}) to {CHECKPOINT_PATH}")

    # --- Final evaluation on the held-out TEST split (facility-disjoint from train/val) ---
    test_dices, test_ious = evaluate(model, test_dl, device)
    test_q = q_values[test_idx]
    test_positive_mask = test_q > 0

    def _stats(vals):
        return dict(mean=float(vals.mean()), median=float(np.median(vals)),
                     min=float(vals.min()), max=float(vals.max()), n=int(len(vals)))

    overall_dice_stats = _stats(test_dices)
    overall_iou_stats = _stats(test_ious)
    positive_dice_stats = _stats(test_dices[test_positive_mask])
    positive_iou_stats = _stats(test_ious[test_positive_mask])
    negative_dice_stats = _stats(test_dices[~test_positive_mask]) if (~test_positive_mask).any() else None

    # Low-Q subset: bottom tercile of TEST positive tiles by true Q, per
    # the explicit instruction to report performance on weak-signal tiles
    # separately, not just the aggregate.
    pos_q = test_q[test_positive_mask]
    pos_dice = test_dices[test_positive_mask]
    pos_iou = test_ious[test_positive_mask]
    low_q_threshold = float(np.percentile(pos_q, 33.3)) if len(pos_q) > 0 else None
    low_q_mask = pos_q <= low_q_threshold if low_q_threshold is not None else np.zeros(0, dtype=bool)
    low_q_dice_stats = _stats(pos_dice[low_q_mask]) if low_q_mask.sum() > 0 else None
    low_q_iou_stats = _stats(pos_iou[low_q_mask]) if low_q_mask.sum() > 0 else None
    high_q_mask = ~low_q_mask
    high_q_dice_stats = _stats(pos_dice[high_q_mask]) if high_q_mask.sum() > 0 else None

    print("\n=== TEST SET RESULTS (facility-held-out) ===")
    print(f"Overall Dice: {overall_dice_stats}")
    print(f"Overall IoU:  {overall_iou_stats}")
    print(f"Positive-tile Dice: {positive_dice_stats}")
    print(f"Positive-tile IoU:  {positive_iou_stats}")
    if negative_dice_stats:
        print(f"Negative-tile Dice (should be near 1.0, empty mask correctly predicted "
              f"empty): {negative_dice_stats}")
    if low_q_dice_stats:
        print(f"LOW-Q positive tiles (bottom tercile, Q<={low_q_threshold:.2e} t/yr) "
              f"Dice: {low_q_dice_stats}")
        print(f"HIGH-Q positive tiles Dice: {high_q_dice_stats}")

    # --- Qualitative figure: a handful of test tiles, predicted vs true mask ---
    model.eval()
    rng = np.random.default_rng(SEED)
    example_idx = rng.choice(len(test_idx), size=min(6, len(test_idx)), replace=False)
    fig, axes = plt.subplots(len(example_idx), 3, figsize=(9, 3 * len(example_idx)))
    if len(example_idx) == 1:
        axes = axes[None, :]
    with torch.no_grad():
        for row, local_i in enumerate(example_idx):
            global_i = test_idx[local_i]
            x_np = build_model_input(norm_tiles[global_i], valid[global_i])
            x = torch.from_numpy(x_np).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
            pred_mask = prob > MASK_THRESHOLD

            # Cloud-gap (invalid) pixels are excluded from scoring -- show
            # them as mid-gray (0.5) in the mask panels so a predicted
            # blob that doesn't count against Dice isn't mistaken for a
            # scoring bug (found and fixed this session: a cloud-masked
            # false positive can make an otherwise-plain figure look
            # inconsistent with its own reported Dice).
            v = valid[global_i]
            true_display = np.where(v, masks[global_i].astype(float), 0.5)
            pred_display = np.where(v, pred_mask.astype(float), 0.5)

            axes[row, 0].imshow(tiles[global_i], cmap="viridis")
            axes[row, 0].set_title(f"input tile (Q={q_values[global_i]:.1e})")
            axes[row, 1].imshow(true_display, cmap="gray", vmin=0, vmax=1)
            axes[row, 1].set_title("true mask (gray=cloud gap)")
            axes[row, 2].imshow(pred_display, cmap="gray", vmin=0, vmax=1)
            dice_i = dice_coefficient(pred_mask, masks[global_i].astype(bool), valid[global_i])
            axes[row, 2].set_title(f"predicted mask (dice={dice_i:.2f})")
            for ax in axes[row]:
                ax.axis("off")
    plt.tight_layout()
    plt.savefig(QUALITATIVE_FIG_PATH, dpi=100)
    plt.close()
    print(f"Qualitative examples saved to {QUALITATIVE_FIG_PATH}")

    results = dict(
        seed=SEED,
        n_tiles_total=len(tiles),
        n_train_tiles=len(train_idx), n_val_tiles=len(val_idx), n_test_tiles=len(test_idx),
        n_train_facilities=n_train_fac, n_val_facilities=n_val_fac, n_test_facilities=n_test_fac,
        model_params=n_params,
        in_channels=IN_CHANNELS,
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
        best_val_dice=best_val_dice,
        train_mean=float(train_mean), train_std=float(train_std),
        mask_threshold=MASK_THRESHOLD,
        history=history,
        test_overall_dice=overall_dice_stats,
        test_overall_iou=overall_iou_stats,
        test_positive_dice=positive_dice_stats,
        test_positive_iou=positive_iou_stats,
        test_negative_dice=negative_dice_stats,
        test_low_q_dice=low_q_dice_stats,
        test_low_q_iou=low_q_iou_stats,
        test_high_q_dice=high_q_dice_stats,
        low_q_threshold_t_per_year=low_q_threshold,
    )
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
