"""
Week 21: U-Net SEGMENTATION-ONLY model for the Track B DL blueprint
(4ypblueprint.pdf, Paper 2 / Dumont Le Brazidec 2024 -- stage 1 of the
U-Net -> CNN regression architecture). Trains on the synthetic
(XCO2 tile, plume mask) pairs from Week 20's simulate_training_pairs.py
(data/simulated_train/simulated_tiles.npz).

SCOPE, explicitly bounded (see WEEK21_LOG.txt / RESEARCH_PAPER.md):
Week 20's 8-task investigation (WEEK20_LOG.txt, SIMULATOR_METHODOLOGY_NOTE.md)
concluded that the simulated Q labels (q_t_per_year in the same npz)
cannot be trusted to this project's own accuracy standard -- Q regression
as the blueprint scoped it is a documented deviation, NOT attempted here.
This module builds ONLY the segmentation stage: predicting the plume
MASK, which is built directly from the true, noise-free physics field
and was never part of the Q-calibration pipeline that failed. Do not add
a regression head to this model, and do not use q_t_per_year as a
training target anywhere in this file.

Reusable, testable pieces live here (model, facility-level split, NaN
handling, Dice/IoU metrics); the training loop itself is in
train_unet_segmentation.py, matching this project's split between
physics_ime.py (reusable math) and its analysis/training scripts.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BG_XCO2_PPM = 415.0  # matches simulate_training_pairs.py's own background constant


def device_str():
    """Machine is a MacBook Air M1, no CUDA -- use mps if available else cpu
    (CLAUDE.md's hard rule), not the cuda-then-cpu check the older
    train_detector.py/train_3channel.py scripts happen to use (which
    silently falls back to cpu on this machine, missing MPS acceleration)."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def facility_level_split(facility_ids, seed=42, train_frac=0.7, val_frac=0.15):
    """
    Splits tile INDICES by facility_id, not by individual tile -- this
    project's own hard rule (README/CLAUDE.md: "Never split train/test at
    tile level. Split by facility"), the same fix Track A applied at
    Week 11 (see train_3channel.py's facility_level_split). Multiple
    tiles share a facility_id (different simulated days/wind directions
    for the same synthetic plant, see simulate_training_pairs.py Task 5's
    facility grouping) -- a tile-level split would let the same
    facility's tiles land in both train and test.

    facility_ids: array-like, one entry per tile. Negative tiles have no
    real facility (each is fully independently sampled by
    simulate_training_pairs.py's negative-tile loop -- no shared
    parameters across negative tiles to leak), so callers should assign
    each negative tile its OWN unique id (e.g. a per-tile negative index)
    before calling this -- splitting is then effectively tile-level for
    negatives (safe, since there is nothing to leak) while genuinely
    facility-level for positives.

    Returns (train_idx, val_idx, test_idx), each a sorted list of tile
    indices into the original arrays.
    """
    facility_ids = np.asarray(facility_ids)
    unique_ids = sorted(set(facility_ids.tolist()))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_ids)

    n = len(shuffled)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    train_ids = set(shuffled[:n_train].tolist())
    val_ids = set(shuffled[n_train:n_train + n_val].tolist())
    test_ids = set(shuffled[n_train + n_val:].tolist())

    train_idx = [i for i, fid in enumerate(facility_ids) if fid in train_ids]
    val_idx = [i for i, fid in enumerate(facility_ids) if fid in val_ids]
    test_idx = [i for i, fid in enumerate(facility_ids) if fid in test_ids]
    return train_idx, val_idx, test_idx


def fill_nan_and_validity(tiles, fill_value=BG_XCO2_PPM):
    """
    simulate_training_pairs.py bakes realistic cloud-gap NaNs into each
    tile (~15-55% of pixels per tile). Conv layers can't consume NaN, and
    the ground-truth mask (built from the TRUE noise-free field) is
    defined everywhere regardless of cloud cover -- but the network has
    no evidence at a cloud-occluded pixel, so those pixels are excluded
    from the LOSS and from evaluation metrics via a validity mask, not
    silently treated as ordinary training signal. This is the simpler of
    the two documented options (mask out of the loss vs. an extra input
    channel) -- chosen because it needs no change to the model's input
    channel count and matches train_detector.py's existing NaN-fill
    convention for the INPUT itself (nanmean-style fill), just applied
    per-batch with a fixed constant here since normalization already
    centers on BG_XCO2_PPM.

    Returns (filled_tiles, valid_mask) with the same shape as tiles;
    valid_mask is True where the input pixel is real (not cloud-occluded).
    """
    valid = np.isfinite(tiles)
    filled = np.where(valid, tiles, fill_value).astype(np.float32)
    return filled, valid


def build_model_input(norm_tile, valid_mask):
    """
    WEEK 23: builds the model's 2-channel input array -- channel 0 the
    normalized, NaN-filled XCO2 tile, channel 1 the explicit valid/
    missing mask (1.0=valid, 0.0=missing) -- to test the hypothesis
    (WEEK22_LOG.txt) that Week 22's persistent coverage-pattern-tracking
    behavior comes from the model relying on the NaN-fill VALUE's sharp
    boundary discontinuity as an implicit, poorly-generalizing shortcut
    feature, rather than genuine signal. Giving the model the valid mask
    explicitly removes the need to infer it from that discontinuity.

    THIS EXACT FUNCTION is called by both train_unet_segmentation.py
    (simulated tiles) and sanity_check_real_tiles.py (real tiles) -- the
    only way to guarantee the two pipelines build this channel
    identically is to share the literal code, not to duplicate matching
    logic in two files. norm_tile: already normalized
    ((filled-train_mean)/train_std). valid_mask: bool or 0/1 array, same
    shape as norm_tile.

    Returns a (2, H, W) float32 array.
    """
    norm_tile = np.asarray(norm_tile, dtype=np.float32)
    valid_channel = np.asarray(valid_mask, dtype=np.float32)
    return np.stack([norm_tile, valid_channel], axis=0)


def dice_coefficient(pred, target, valid=None, eps=1e-6):
    """
    pred, target: bool/0-1 arrays or tensors, any shape. valid: same
    shape, True where the pixel should count (cloud-occluded pixels
    excluded). Returns a scalar Dice = 2|pred & target| / (|pred| + |target|).
    """
    pred = pred.astype(bool) if isinstance(pred, np.ndarray) else pred.bool()
    target = target.astype(bool) if isinstance(target, np.ndarray) else target.bool()
    if valid is not None:
        valid = valid.astype(bool) if isinstance(valid, np.ndarray) else valid.bool()
        pred = pred & valid
        target = target & valid
    if isinstance(pred, np.ndarray):
        intersection = np.logical_and(pred, target).sum()
        denom = pred.sum() + target.sum()
    else:
        intersection = (pred & target).sum().item()
        denom = pred.sum().item() + target.sum().item()
    return (2.0 * intersection + eps) / (denom + eps)


def iou_score(pred, target, valid=None, eps=1e-6):
    """Intersection-over-union, same conventions as dice_coefficient()."""
    pred = pred.astype(bool) if isinstance(pred, np.ndarray) else pred.bool()
    target = target.astype(bool) if isinstance(target, np.ndarray) else target.bool()
    if valid is not None:
        valid = valid.astype(bool) if isinstance(valid, np.ndarray) else valid.bool()
        pred = pred & valid
        target = target & valid
    if isinstance(pred, np.ndarray):
        intersection = np.logical_and(pred, target).sum()
        union = np.logical_or(pred, target).sum()
    else:
        intersection = (pred & target).sum().item()
        union = (pred | target).sum().item()
    return (intersection + eps) / (union + eps)


def masked_bce_dice_loss(logits, target, valid, dice_weight=1.0):
    """
    Combined BCE + soft-Dice loss, both restricted to valid (non-cloud-
    occluded) pixels. Dice is included because plume masks are sparse
    (~4% positive pixels on positive tiles, see WEEK21_LOG.txt) -- BCE
    alone under-weights the minority (plume) class on an imbalanced mask.
    logits, target, valid: (B,1,H,W) float tensors; target/valid in {0,1}.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp(min=1.0)

    probs = torch.sigmoid(logits) * valid
    target_v = target * valid
    intersection = (probs * target_v).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target_v.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + 1.0) / (denom + 1.0)
    dice_loss = 1.0 - dice.mean()

    return bce + dice_weight * dice_loss


class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    """
    Compact U-Net for 64x64 single-channel input, sized for a MacBook Air
    M1 (no CUDA) and a small (~2000-tile) dataset -- 3 encoder/decoder
    levels (64->32->16->8) rather than the classic 5-level U-Net, which
    would be oversized for both the input resolution and dataset size
    here. Outputs raw logits (B,1,64,64); apply sigmoid externally.
    """
    def __init__(self, in_channels=1, base_channels=16):
        super().__init__()
        c = base_channels
        self.enc1 = _ConvBlock(in_channels, c)
        self.enc2 = _ConvBlock(c, c * 2)
        self.enc3 = _ConvBlock(c * 2, c * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = _ConvBlock(c * 4, c * 8)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = _ConvBlock(c * 8 + c * 4, c * 4)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = _ConvBlock(c * 4 + c * 2, c * 2)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = _ConvBlock(c * 2 + c, c)

        self.out_conv = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)                 # (B, c, 64, 64)
        e2 = self.enc2(self.pool(e1))      # (B, 2c, 32, 32)
        e3 = self.enc3(self.pool(e2))      # (B, 4c, 16, 16)
        b = self.bottleneck(self.pool(e3))  # (B, 8c, 8, 8)

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))   # (B, 4c, 16, 16)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # (B, 2c, 32, 32)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (B, c, 64, 64)

        return self.out_conv(d1)  # (B, 1, 64, 64) logits
