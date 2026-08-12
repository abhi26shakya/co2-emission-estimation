import numpy as np, glob, os, re, random, torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)
torch.manual_seed(0); np.random.seed(0); random.seed(0)

# Facility-level group id per tile: filename minus the trailing _YYYY_MM.npy
# suffix, e.g. "VINDH_CHAL_STPS_2020_01.npy" -> "VINDH_CHAL_STPS",
# "city_Delhi_2019_01.npy" -> "city_Delhi". Used to keep all of one
# physical site's monthly tiles on the same side of the train/test split --
# RESEARCH_PLAN.md flagged since Week 7 that the original random tile-level
# split lets the same site's tiles land in both train and test, a leakage
# risk this week fixes.
_MONTH_SUFFIX = re.compile(r"_\d{4}_\d{2}\.npy$")

def load_folder(path, label):
    X, y, groups = [], [], []
    for f in sorted(glob.glob(f"{path}/*.npy")):
        arr = np.load(f).astype(np.float32)      # shape (3,64,64)
        X.append(arr); y.append(label)
        groups.append(_MONTH_SUFFIX.sub("", os.path.basename(f)))
    return X, y, groups

Xp, yp, gp = load_folder("data/threech/positive",      1)   # plants
Xh, yh, gh = load_folder("data/threech/hard_negative",  0)  # cities/industry/hwy
Xr, yr, gr = load_folder("data/threech/negative",       0)  # rural

# 3-CHANNEL detector: note nn.Conv2d(3, ...) as the first layer
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

def facility_level_split(groups, labels, test_frac=0.2, seed=0):
    """
    Split tile indices by facility/site group, not by individual tile, so
    all of one physical site's monthly tiles land on the same side --
    fixes the leakage risk of a random tile-level split (the original
    split logic below, kept as tile_level_split for a direct before/after
    comparison of how much it inflated accuracy).

    Stratified by class: with only 5 positive-class facilities, a plain
    group shuffle risks a degenerate test set with zero positives (or, less
    likely, zero negatives), making recall/precision undefined. Positive
    and negative groups are shuffled and held out separately so both
    classes are guaranteed to appear in test.
    """
    rng = random.Random(seed)
    groups_by_class = {0: set(), 1: set()}
    for g, y in zip(groups, labels):
        groups_by_class[int(y)].add(g)

    test_groups = set()
    for cls, cls_groups in groups_by_class.items():
        cls_groups = sorted(cls_groups)
        rng.shuffle(cls_groups)
        n_test_groups = max(1, round(test_frac * len(cls_groups)))
        test_groups.update(cls_groups[:n_test_groups])

    train_idx = [i for i, g in enumerate(groups) if g not in test_groups]
    test_idx = [i for i, g in enumerate(groups) if g in test_groups]
    return train_idx, test_idx, sorted(test_groups)


def tile_level_split(n, test_frac=0.2, seed=0):
    """Original random-at-the-tile split -- kept only to quantify the
    leakage's effect on reported accuracy, not used for the saved model."""
    n_test = max(1, int(test_frac * n)); n_train = n - n_test
    g = torch.Generator().manual_seed(seed)
    tr, te = random_split(range(n), [n_train, n_test], generator=g)
    return list(tr), list(te)


def run(Xlist, ylist, groups, tag, epochs=30, split="facility"):
    X = np.stack(Xlist)                           # (N,3,64,64)
    y = np.array(ylist, dtype=np.int64)
    # normalize each channel separately
    for c in range(3):
        m, s = X[:,c].mean(), X[:,c].std()+1e-12
        X[:,c] = (X[:,c]-m)/s
    X = torch.tensor(X); y = torch.tensor(y)
    ds = TensorDataset(X, y)

    if split == "facility":
        train_idx, test_idx, test_groups = facility_level_split(groups, ylist)
        print(f"  split=facility-level  test facilities={test_groups}")
    else:
        train_idx, test_idx = tile_level_split(len(ds))
        print(f"  split=tile-level (leaky baseline, for comparison only)")
    tr, te = Subset(ds, train_idx), Subset(ds, test_idx)
    tr_dl = DataLoader(tr,batch_size=16,shuffle=True); te_dl=DataLoader(te,batch_size=16)

    model = Detector3().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    lossf = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        for xb,yb in tr_dl:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE)
            opt.zero_grad(); lossf(model(xb),yb).backward(); opt.step()

    model.eval(); correct=total=tp=tn=fp=fn=0
    with torch.no_grad():
        for xb,yb in te_dl:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE)
            pred=model(xb).argmax(1)
            correct+=(pred==yb).sum().item(); total+=yb.size(0)
            tp+=((pred==1)&(yb==1)).sum().item(); tn+=((pred==0)&(yb==0)).sum().item()
            fp+=((pred==1)&(yb==0)).sum().item(); fn+=((pred==0)&(yb==1)).sum().item()
    acc=100*correct/total
    print(f"\n=== {tag} ===")
    print(f"  test accuracy: {acc:.1f}%   (chance=50%)")
    print(f"  plants correct (recall): {100*tp/max(tp+fn,1):.0f}%   "
          f"negatives correct: {100*tn/max(tn+fp,1):.0f}%")
    print(f"  false alarms (neg->plant): {fp}   missed plants: {fn}")
    # facility-split checkpoints get their own filename -- extract_activity_signal.py
    # already depends on the existing detector3_2ch_mixed.pt (tile-level split);
    # overwriting it here would silently change what an already-committed
    # data/activity_signals.json was actually computed from.
    suffix = "_facility_split" if split == "facility" else ""
    torch.save(model.state_dict(), f"detector3_{tag}{suffix}.pt")
    return acc

# --- balanced plants vs HARD negatives (same fair test as Week 3/4) ---
idx=list(range(len(Xh))); random.shuffle(idx); idx=idx[:len(Xp)]
Xh_bal=[Xh[i] for i in idx]; yh_bal=[0]*len(Xh_bal); gh_bal=[gh[i] for i in idx]

print("\n--- 2ch_hard_only: tile-level (leaky) split, for comparison ---")
acc_hard_leaky = run(Xp+Xh_bal, yp+yh_bal, gp+gh_bal, "2ch_hard_only", split="tile")
print("\n--- 2ch_hard_only: facility-level split (fixed) ---")
acc_hard = run(Xp+Xh_bal, yp+yh_bal, gp+gh_bal, "2ch_hard_only", split="facility")

# --- plants vs ALL negatives ---
print("\n--- 2ch_mixed: tile-level (leaky) split, for comparison ---")
acc_mix_leaky = run(Xp+Xh+Xr, yp+yh+yr, gp+gh+gr, "2ch_mixed", split="tile")
print("\n--- 2ch_mixed: facility-level split (fixed) ---")
acc_mix = run(Xp+Xh+Xr, yp+yh+yr, gp+gh+gr, "2ch_mixed", split="facility")

print("\n================ COMPARISON ================")
print(f"  Week 3  NO2-only      hard_only (tile-level split)     : 77.1%")
print(f"  Week 4  NO2+SO2       hard_only (tile-level split)     : 79.2%")
print(f"  Week 5  NO2+SO2+VIIRS hard_only (tile-level split)     : {acc_hard_leaky:.1f}%")
print(f"  Week 10 NO2+SO2+VIIRS hard_only (facility-level split) : {acc_hard:.1f}%")
print(f"  Week 10 NO2+SO2+VIIRS mixed     (tile-level split)     : {acc_mix_leaky:.1f}%")
print(f"  Week 10 NO2+SO2+VIIRS mixed     (facility-level split) : {acc_mix:.1f}%")
