import numpy as np, glob, os, random, torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)
torch.manual_seed(0); np.random.seed(0); random.seed(0)

def load_folder(path, label):
    X, y = [], []
    for f in sorted(glob.glob(f"{path}/*.npy")):
        arr = np.load(f).astype(np.float32)      # shape (3,64,64)
        X.append(arr); y.append(label)
    return X, y

Xp, yp = load_folder("data/threech/positive",      1)   # plants
Xh, yh = load_folder("data/threech/hard_negative",  0)  # cities/industry/hwy
Xr, yr = load_folder("data/threech/negative",       0)  # rural

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

def run(Xlist, ylist, tag, epochs=30):
    X = np.stack(Xlist)                           # (N,3,64,64)
    y = np.array(ylist, dtype=np.int64)
    # normalize each channel separately
    for c in range(3):
        m, s = X[:,c].mean(), X[:,c].std()+1e-12
        X[:,c] = (X[:,c]-m)/s
    X = torch.tensor(X); y = torch.tensor(y)
    ds = TensorDataset(X, y)
    n_test = max(1,int(0.2*len(ds))); n_train=len(ds)-n_test
    tr, te = random_split(ds,[n_train,n_test],
                          generator=torch.Generator().manual_seed(0))
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
    torch.save(model.state_dict(), f"detector3_{tag}.pt")
    return acc

# --- balanced plants vs HARD negatives (same fair test as Week 3/4) ---
idx=list(range(len(Xh))); random.shuffle(idx); idx=idx[:len(Xp)]
Xh_bal=[Xh[i] for i in idx]; yh_bal=[0]*len(Xh_bal)
acc_hard = run(Xp+Xh_bal, yp+yh_bal, "2ch_hard_only")

# --- plants vs ALL negatives ---
acc_mix = run(Xp+Xh+Xr, yp+yh+yr, "2ch_mixed")

print("\n================ COMPARISON ================")
print(f"  Week 3  NO2-only      hard_only : 77.1%")
print(f"  Week 4  NO2+SO2       hard_only : 79.2%")
print(f"  Week 5  NO2+SO2+VIIRS hard_only : {acc_hard:.1f}%")
