import numpy as np, glob, os, random, torch
import torch.nn as nn
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0); random.seed(0)

class Detector3(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.BatchNorm2d(16), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(64, 2))
    def forward(self, x): return self.net(x)

model = Detector3().to(DEVICE)
model.load_state_dict(torch.load("detector3_2ch_hard_only.pt", map_location=DEVICE))
model.eval()

def load_folder(path):
    items = []
    for f in sorted(glob.glob(f"{path}/*.npy")):
        arr = np.load(f).astype(np.float32)      # shape (3,64,64), already gap-filled by build_3channel.py
        items.append((os.path.basename(f).replace(".npy",""), arr))
    return items

plants = load_folder("data/threech/positive")
hards  = load_folder("data/threech/hard_negative")

# Reproduce the EXACT balanced hard-negative subset train_3channel.py trained on,
# so the per-channel normalization stats match what the model actually saw.
idx = list(range(len(hards))); random.shuffle(idx); idx = idx[:len(plants)]
hard_bal = [hards[i] for i in idx]

allX = np.stack([a for _, a in plants] + [a for _, a in hard_bal])  # (N,3,64,64)
mean = np.zeros(3, dtype=np.float32); std = np.zeros(3, dtype=np.float32)
for c in range(3):
    mean[c] = allX[:, c].mean(); std[c] = allX[:, c].std() + 1e-12

def predict(arr):
    x = ((arr - mean[:, None, None]) / std[:, None, None])[None]
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32).to(DEVICE))
        prob = torch.softmax(logits, 1)[0, 1].item()
    return prob

print("Hard negatives most often mistaken for PLANTS (3-channel NO2+SO2+VIIRS model):")
scores = {}
for name, arr in hards:
    base = name.rsplit("_", 2)[0]
    scores.setdefault(base, []).append(predict(arr))
ranked = sorted(scores.items(), key=lambda kv: -np.mean(kv[1]))
for base, ps in ranked:
    print(f"  {base:22s} avg P(plant)={np.mean(ps):.2f}  (n={len(ps)})")

print("\n================ STEEL-PLANT FOCUS (Bhilai / Jamshedpur) ================")
print(f"{'source':22s} {'Week3 (NO2)':>12s} {'Week4 (+SO2)':>13s} {'Week5 (+VIIRS)':>15s}")
week3 = {"ind_Bhilai": 0.55, "ind_Jamshedpur": 0.58}
week4 = {"ind_Bhilai": 0.59, "ind_Jamshedpur": 0.54}
for base in ["ind_Bhilai", "ind_Jamshedpur"]:
    w5 = np.mean(scores.get(base, [float("nan")]))
    print(f"{base:22s} {week3[base]:12.2f} {week4[base]:13.2f} {w5:15.2f}")

# visualize 4 worst false alarms
worst = []
for name, arr in hards:
    worst.append((predict(arr), name, arr))
worst.sort(key=lambda t: -t[0])
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
for ax, (p, name, arr) in zip(axes, worst[:4]):
    im = ax.imshow(arr[0], cmap="jet")   # NO2 channel for visualization
    ax.set_title(f"{name}\nP(plant)={p:.2f}  (WRONG)", fontsize=8)
    ax.axis("off"); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig("data/worst_false_alarms_3ch.png", dpi=120, bbox_inches="tight")
print("\nSaved data/worst_false_alarms_3ch.png")
