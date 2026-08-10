import numpy as np, glob, os

def triple_folder(no2_dir, so2_dir, viirs_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    no2_files   = {os.path.basename(f): f for f in glob.glob(f"{no2_dir}/*.npy")}
    so2_files   = {os.path.basename(f): f for f in glob.glob(f"{so2_dir}/*.npy")}
    viirs_files = {os.path.basename(f): f for f in glob.glob(f"{viirs_dir}/*.npy")}
    common = sorted(set(no2_files) & set(so2_files) & set(viirs_files))
    saved = 0
    for name in common:
        no2   = np.load(no2_files[name]).astype(np.float32)
        so2   = np.load(so2_files[name]).astype(np.float32)
        viirs = np.load(viirs_files[name]).astype(np.float32)
        if no2.shape != so2.shape or no2.shape != viirs.shape:
            continue
        # fill gaps with each channel's own mean
        no2   = np.where(np.isfinite(no2), no2, np.nanmean(no2))
        so2   = np.where(np.isfinite(so2), so2, np.nanmean(so2))
        viirs = np.where(np.isfinite(viirs), viirs, np.nanmean(viirs))
        # SO2 noise and VIIRS FRP can't be negative physically -- clamp at 0
        so2   = np.clip(so2, 0, None)
        viirs = np.clip(viirs, 0, None)
        stacked = np.stack([no2, so2, viirs], axis=0)   # shape (3, 64, 64)
        np.save(f"{out_dir}/{name}", stacked)
        saved += 1
    print(f"{out_dir:32s} paired={saved}  (NO2={len(no2_files)}, SO2={len(so2_files)}, VIIRS={len(viirs_files)})")
    return saved

os.makedirs("data/threech", exist_ok=True)
p = triple_folder("data/monthly/positive",      "data/so2/positive",      "data/viirs/positive",      "data/threech/positive")
h = triple_folder("data/monthly/hard_negative", "data/so2/hard_negative", "data/viirs/hard_negative", "data/threech/hard_negative")
r = triple_folder("data/monthly/negative",      "data/so2/negative",      "data/viirs/negative",      "data/threech/negative")
print(f"\nTotal 3-channel tiles: plants={p}  hard_neg={h}  rural={r}")
print("Each tile shape = (3, 64, 64): channel 0 = NO2, channel 1 = SO2, channel 2 = VIIRS MaxFRP")
