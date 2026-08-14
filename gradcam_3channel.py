"""
Extends gradcam.py's Grad-CAM approach (built for the old 1-channel-NO2
detector) to Detector3, the NO2+SO2+VIIRS 3-channel detector actually used
for the paper's headline LOFO results (train_3channel.py, lofo_track_a.py).
This closes the "still open" item flagged in RESEARCH_PAPER.md Sec 8
(Conclusion, item 10) and Sec 7 (Limitations) -- narrowly: a 3-channel
attention diagnostic, NOT the deprioritized cross-modal plume-fusion idea
(comparing CAM centroid vs. wind-predicted plume axis), which stays
deprioritized since NOVEL_METHODOLOGY_PROPOSAL.md Sec 11-12 found the
plume's spatial claim itself unvalidated.

Reuses gradcam.py's exact hook pattern (forward/backward hooks on the last
conv layer, gradient-weighted channel sum, ReLU, bilinear upsample) --
adapted to Detector3's flat `nn.Sequential` named `net` (train_3channel.py)
instead of gradcam.py's separate features/head split, and to a 3-channel
(not 1-channel) input.

New capability the 1-channel version couldn't offer: per-channel
attribution. Since the input tensor's 3 channels are physically distinct
(NO2, SO2, VIIRS thermal), the gradient-weighted activation map is also
broken down by how much each input channel contributed to the CAM (via
input*gradient saliency per channel, summed and normalized to shares) --
answering "is the detector actually using NO2, or SO2, or thermal, for
this particular tile" rather than only "where does it look."

Diagnostic scope, chosen deliberately: one well-generalizing facility
(Rihand, 100% LOFO recall) as a baseline, plus the two facilities
lofo_track_a.py's exhaustive LOFO evaluation found at 0% recall even
after the 2019-data-year fix (Kudgi, ShriSingajiMalwa) --
diagnose_lofo_weak_facilities.py already found their raw NO2/SO2 tile
intensity sits at or below the rural-negative noise floor; this script
asks the complementary question of whether the trained detector's
attention still localizes on the plant coordinate for these tiles or
looks elsewhere, using the facility-split checkpoint whose LOFO behavior
the paper actually reports on (detector3_2ch_mixed_facility_split.pt).
"""
import glob
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHANNEL_NAMES = ["NO2", "SO2", "VIIRS"]
CHECKPOINT = "detector3_2ch_mixed_facility_split.pt"


class Detector3(nn.Module):
    """Identical architecture to train_3channel.py's Detector3."""

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


model = Detector3().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True))
model.eval()

# net[8] is the 3rd Conv2d (32->64 filters), the same "last conv before
# global pooling" position gradcam.py hooks in its own features/head split.
activations, gradients = {}, {}
target_conv = model.net[8]


def fwd_hook(m, i, o):
    activations["v"] = o.detach()


def bwd_hook(m, gi, go):
    gradients["v"] = go[0].detach()


target_conv.register_forward_hook(fwd_hook)
target_conv.register_full_backward_hook(bwd_hook)


def normalize_ref():
    files = glob.glob("data/threech/positive/*.npy") + glob.glob("data/threech/hard_negative/*.npy")
    allX = np.stack([np.where(np.isfinite(a := np.load(f)), a, np.nanmean(a)) for f in files])
    # per-channel mean/std, matching build_3channel.py's per-channel normalization intent
    return allX.mean(axis=(0, 2, 3)), allX.std(axis=(0, 2, 3)) + 1e-12


MEAN, STD = normalize_ref()  # shape (3,)


def gradcam3(arr):
    """arr: (3,64,64) raw tile. Returns (cam, prob_plant, channel_shares)."""
    a = np.where(np.isfinite(arr), arr, np.nanmean(arr, axis=(1, 2), keepdims=True))
    x_norm = (a - MEAN[:, None, None]) / STD[:, None, None]
    x = torch.tensor(x_norm[None], dtype=torch.float32, device=DEVICE)
    x.requires_grad_(True)
    logits = model(x)
    model.zero_grad()
    logits[0, 1].backward()  # gradient wrt "plant" class

    g = gradients["v"][0]        # (C,h,w) at target_conv's output resolution
    A = activations["v"][0]
    weights = g.mean(dim=(1, 2))
    cam = F.relu((weights[:, None, None] * A).sum(0))
    cam = cam / (cam.max() + 1e-8)
    cam = F.interpolate(cam[None, None], size=arr.shape[1:], mode="bilinear")[0, 0].cpu().numpy()

    # Per-channel attribution: input*gradient saliency, summed spatially per
    # channel and normalized to shares -- answers which raw input channel
    # (NO2/SO2/VIIRS) drove the "plant" logit for this specific tile.
    input_grad = x.grad[0].detach().cpu().numpy()  # (3,64,64)
    channel_saliency = np.abs(input_grad * x_norm).sum(axis=(1, 2))
    channel_shares = (channel_saliency / (channel_saliency.sum() + 1e-12)).tolist()

    prob_plant = torch.softmax(logits, 1)[0, 1].item()
    return cam, prob_plant, channel_shares


def pick_tile(facility, prefer_year="2020"):
    fs = sorted(glob.glob(f"data/threech/positive/{facility}_*.npy"))
    preferred = [f for f in fs if prefer_year in f]
    chosen = preferred or fs
    return chosen[len(chosen) // 2] if chosen else None


FACILITIES = [
    ("Rihand", "well-generalizing (100% LOFO recall)"),
    ("Kudgi", "0% LOFO recall even after 2019-data-year fix"),
    ("ShriSingajiMalwa", "0% LOFO recall even after 2019-data-year fix"),
]

summary = {}
fig, axes = plt.subplots(2, len(FACILITIES), figsize=(4.5 * len(FACILITIES), 8))
for j, (facility, label) in enumerate(FACILITIES):
    f = pick_tile(facility)
    if f is None:
        print(f"[skip] no tiles found for {facility}")
        continue
    arr = np.load(f).astype(np.float32)  # (3,64,64)
    cam, prob, shares = gradcam3(arr)

    rgb_display = np.stack([
        np.where(np.isfinite(arr[c]), arr[c], np.nanmean(arr[c])) for c in range(3)
    ])
    disp = (rgb_display - rgb_display.min(axis=(1, 2), keepdims=True)) / (
        rgb_display.max(axis=(1, 2), keepdims=True) - rgb_display.min(axis=(1, 2), keepdims=True) + 1e-12)

    axes[0, j].imshow(np.transpose(disp, (1, 2, 0)))
    axes[0, j].set_title(f"{facility}\n({label})", fontsize=8)
    axes[0, j].axis("off")

    axes[1, j].imshow(np.transpose(disp, (1, 2, 0)))
    axes[1, j].imshow(cam, cmap="jet", alpha=0.5)
    share_str = ", ".join(f"{n}:{s:.0%}" for n, s in zip(CHANNEL_NAMES, shares))
    axes[1, j].set_title(f"Grad-CAM  P(plant)={prob:.2f}\n{share_str}", fontsize=7)
    axes[1, j].axis("off")

    summary[facility] = {
        "tile_used": f,
        "prob_plant": prob,
        "channel_shares": {n: s for n, s in zip(CHANNEL_NAMES, shares)},
        "dominant_channel": CHANNEL_NAMES[int(np.argmax(shares))],
        "note": label,
    }
    print(f"{facility:20s} P(plant)={prob:.3f}  channel_shares={share_str}")

plt.tight_layout()
plt.savefig("data/gradcam_3channel.png", dpi=120, bbox_inches="tight")
print("Saved data/gradcam_3channel.png")

json.dump({
    "checkpoint": CHECKPOINT,
    "caveat": (
        "Diagnostic scope: 1 tile per facility (the temporal-midpoint tile, "
        "preferring 2020 if available), not an exhaustive per-tile study. "
        "Complements diagnose_lofo_weak_facilities.py's raw-intensity finding "
        "(Kudgi/ShriSingajiMalwa sit at or below the rural-negative noise "
        "floor) by showing where/what the trained detector attends to on "
        "these specific tiles, not a claim about every tile from either "
        "facility."
    ),
    "facilities": summary,
}, open("data/gradcam_3channel_summary.json", "w"), indent=2)
print("Saved data/gradcam_3channel_summary.json")
