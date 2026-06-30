"""
dl_spine.py
-----------
Paper 2 (Dumont Le Brazidec 2024) architecture: a two-stage
U-Net (plume segmentation) -> CNN (emission-rate regression) spine.

Designed for a 4 GB laptop GPU (RTX 3050):
    * 64x64 tiles, batch 16-32
    * mixed precision (amp) on by default
    * 3rd "valid-pixel mask" channel for cloud robustness (your novelty hook)

Includes a SYNTHETIC PLUME GENERATOR so you can train TODAY without waiting
on OCO/TROPOMI downloads. Swap it for real SMARTCARB / OCO-3 SAM tiles later.

Run:
    python dl_spine.py            # smoke-test: builds data, trains a few steps
Author scaffold for: Abhishek Kumar Shakya
"""

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ======================================================================
# 1. SYNTHETIC DATA  --  realistic enough to develop the whole pipeline
# ======================================================================
class SyntheticPlumeDataset(Dataset):
    """
    Generates 2-channel [XCO2, NO2] tiles with a Gaussian plume of known
    emission rate Q, a binary plume mask, and a valid-pixel mask (random
    cloud holes). Targets: the mask (for U-Net) and Q (for the regressor).

    This mimics the SMARTCARB simulated dataset used by Paper 2: clean,
    labelled, unlimited. The sim-to-real gap is handled later via transfer
    learning on real OCO-3 SAM images.
    """
    def __init__(self, n_samples=2000, size=64, cloud=True, seed=0):
        self.n = n_samples
        self.size = size
        self.cloud = cloud
        self.rng = np.random.default_rng(seed)
        # Pre-generate so __getitem__ is cheap & reproducible.
        self.samples = [self._make() for _ in range(n_samples)]

    def _make(self):
        s = self.size
        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)

        # --- random plume parameters ---
        Q = self.rng.uniform(1.0, 40.0)              # emission rate [Mt/yr]
        cx = self.rng.uniform(s * 0.3, s * 0.7)      # source x
        cy = self.rng.uniform(s * 0.3, s * 0.7)      # source y
        wind_ang = self.rng.uniform(0, 2 * math.pi)  # wind direction
        wind_spd = self.rng.uniform(2.0, 8.0)        # [m/s]
        length = self.rng.uniform(s * 0.4, s * 0.9)  # plume length in px
        width = self.rng.uniform(2.5, 6.0)           # cross-wind width in px

        # downwind axis
        dx, dy = math.cos(wind_ang), math.sin(wind_ang)
        # coordinates relative to source, rotated into along/cross wind
        rx = (xx - cx)
        ry = (yy - cy)
        along = rx * dx + ry * dy                    # distance downwind
        cross = -rx * dy + ry * dx                   # cross-wind distance

        # plume only downwind (along >= 0), widening with distance
        sigma = width * (1.0 + along / max(length, 1e-3) * 1.5)
        sigma = np.clip(sigma, 1.0, None)
        downwind_mask = (along >= 0) & (along <= length)
        # amplitude scales with Q and inversely with wind (physics!)
        amp = (Q / wind_spd) * 0.15
        plume = amp * np.exp(-(cross ** 2) / (2 * sigma ** 2)) * downwind_mask

        # --- XCO2 channel: plume + wavy background + noise ---
        bg = 410.0 + 1.5 * np.sin(xx / s * 2 * math.pi + self.rng.uniform(0, 6))
        xco2 = bg + plume + self.rng.normal(0, 0.3, (s, s)).astype(np.float32)

        # --- NO2 channel: co-emitted proxy, sharper & higher SNR ---
        no2 = plume * 2.2 + self.rng.normal(0, 0.15, (s, s)).astype(np.float32)
        no2 = np.clip(no2, 0, None)

        # --- ground-truth plume mask (where enhancement is meaningful) ---
        mask = (plume > 0.05 * max(amp, 1e-3)).astype(np.float32)

        # --- valid-pixel mask: random cloud holes (cloud-robustness hook) ---
        valid = np.ones((s, s), np.float32)
        if self.cloud and self.rng.random() < 0.7:
            n_clouds = self.rng.integers(1, 4)
            for _ in range(n_clouds):
                ccx, ccy = self.rng.uniform(0, s, 2)
                crad = self.rng.uniform(s * 0.08, s * 0.22)
                hole = ((xx - ccx) ** 2 + (yy - ccy) ** 2) < crad ** 2
                valid[hole] = 0.0
            # masked pixels are unobserved -> set channels to background-ish
            xco2 = np.where(valid > 0, xco2, 410.0)
            no2 = np.where(valid > 0, no2, 0.0)

        # standardize channels (per-tile here; use train-set stats in production)
        xco2_n = (xco2 - 410.0) / 3.0
        no2_n = no2 / 2.0

        img = np.stack([xco2_n, no2_n, valid], axis=0).astype(np.float32)  # (3,H,W)
        return {
            "image": img,
            "mask": mask[None, :, :],          # (1,H,W)
            "Q": np.float32(Q),
            "wind": np.float32(wind_spd),
        }

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        d = self.samples[i]
        return (
            torch.from_numpy(d["image"]),
            torch.from_numpy(d["mask"]),
            torch.tensor(d["Q"]),
            torch.tensor(d["wind"]),
        )


# ======================================================================
# 2. STAGE 1  --  U-Net for plume segmentation
# ======================================================================
class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.SiLU(),
        )
    def forward(self, x): return self.net(x)


class UNet(nn.Module):
    """Compact U-Net sized for 4 GB VRAM. Input (3,64,64) -> mask (1,64,64)."""
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.d2 = DoubleConv(base, base * 2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.bott = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.u3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.u2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.u1 = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        b = self.bott(self.pool(c3))
        x = self.u3(torch.cat([self.up3(b), c3], 1))
        x = self.u2(torch.cat([self.up2(x), c2], 1))
        x = self.u1(torch.cat([self.up1(x), c1], 1))
        return self.head(x)                       # logits (1,H,W)


def dice_bce_loss(logits, target, eps=1e-6):
    """Dice + BCE: standard, stable segmentation loss for thin plumes."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((2, 3))
    dice = 1 - ((2 * inter + eps) / (p.sum((2, 3)) + target.sum((2, 3)) + eps))
    return bce + dice.mean()


# ======================================================================
# 3. STAGE 2  --  CNN regressor for emission rate Q
# ======================================================================
class CNNRegressor(nn.Module):
    """
    Input: [XCO2, NO2, predicted_mask] (3,64,64) + scalar wind  ->  Q (scalar).
    MC-dropout kept active at inference gives uncertainty (your novelty).
    """
    def __init__(self, in_ch=3, p_drop=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.drop = nn.Dropout(p_drop)
        self.head = nn.Sequential(
            nn.Linear(128 + 1, 64), nn.SiLU(), nn.Dropout(p_drop),
            nn.Linear(64, 1),
        )

    def forward(self, x, wind):
        f = self.features(x).flatten(1)           # (B,128)
        f = self.drop(f)
        f = torch.cat([f, wind[:, None]], 1)      # add wind scalar
        return self.head(f).squeeze(1)            # (B,)


@torch.no_grad()
def mc_dropout_predict(model, x, wind, n=30):
    """
    Monte-Carlo dropout: keep dropout ON, run n forward passes.
    Returns (mean_Q, std_Q). std is your calibrated-ish uncertainty.
    """
    model.train()                                 # dropout active
    preds = torch.stack([model(x, wind) for _ in range(n)], 0)
    return preds.mean(0), preds.std(0)


# ======================================================================
# 4. SMOKE TEST  --  prove the whole spine runs end to end
# ======================================================================
def _smoke_test():
    print(f"Device: {DEVICE}")
    torch.manual_seed(0)

    # tiny dataset so the test is fast; scale n_samples up for real training
    train = SyntheticPlumeDataset(n_samples=64, size=64, seed=1)
    loader = DataLoader(train, batch_size=16, shuffle=True, num_workers=0)

    unet = UNet().to(DEVICE)
    reg = CNNRegressor().to(DEVICE)
    opt_u = torch.optim.AdamW(unet.parameters(), lr=3e-4, weight_decay=1e-4)
    opt_r = torch.optim.AdamW(reg.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(DEVICE, enabled=(DEVICE == "cuda"))

    print("\n--- Stage 1: train U-Net (2 quick epochs) ---")
    for ep in range(2):
        unet.train(); tot = 0
        for img, mask, Q, wind in loader:
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            opt_u.zero_grad()
            with torch.amp.autocast(DEVICE, enabled=(DEVICE == "cuda")):
                loss = dice_bce_loss(unet(img), mask)
            scaler.scale(loss).backward(); scaler.step(opt_u); scaler.update()
            tot += loss.item()
        print(f"  epoch {ep}  seg-loss {tot/len(loader):.4f}")

    print("\n--- Stage 2: train CNN regressor on predicted masks (2 epochs) ---")
    unet.eval()
    for ep in range(2):
        reg.train(); tot = 0
        for img, mask, Q, wind in loader:
            img, Q, wind = img.to(DEVICE), Q.to(DEVICE), wind.to(DEVICE)
            with torch.no_grad():
                pred_mask = torch.sigmoid(unet(img))
            reg_in = torch.cat([img[:, :2], pred_mask], 1)  # [XCO2,NO2,mask]
            opt_r.zero_grad()
            with torch.amp.autocast(DEVICE, enabled=(DEVICE == "cuda")):
                pred_Q = reg(reg_in, wind)
                loss = F.smooth_l1_loss(pred_Q, Q)          # Huber on Q
            scaler.scale(loss).backward(); scaler.step(opt_r); scaler.update()
            tot += loss.item()
        print(f"  epoch {ep}  reg-loss {tot/len(loader):.4f}")

    print("\n--- Inference with MC-dropout uncertainty ---")
    img, mask, Q, wind = next(iter(loader))
    img, wind = img.to(DEVICE), wind.to(DEVICE)
    with torch.no_grad():
        pm = torch.sigmoid(unet(img))
    reg_in = torch.cat([img[:, :2], pm], 1)
    mean_Q, std_Q = mc_dropout_predict(reg, reg_in, wind, n=30)
    for i in range(min(4, len(Q))):
        print(f"  true Q {Q[i]:5.1f}  |  pred {mean_Q[i]:5.1f} +/- {std_Q[i]:.1f} Mt/yr")
    print("\nSpine runs end-to-end. Scale n_samples to ~5000 and epochs to ~100 for real training.")


if __name__ == "__main__":
    _smoke_test()
