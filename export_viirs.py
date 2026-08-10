import ee, numpy as np, pandas as pd, os, urllib.request, io
ee.Initialize(project="opportune-lore-415218")

YEARS  = [2019, 2020]
MONTHS = list(range(1, 13))

# NASA/VIIRS/002/VNP14A1: daily 1km active-fire/thermal-anomaly product.
# MaxFRP = Maximum Fire Radiative Power (MW), a continuous magnitude like
# our NO2/SO2 channels. Most tiles will be 0/NaN away from active thermal
# anomalies -- power-plant stack heat is a weak, sub-pixel signal at 1km,
# so sparse coverage here is expected, not a bug.
def get_viirs_tile(lat, lon, year, month, size_km=60, px=64):
    point  = ee.Geometry.Point(lon, lat)
    region = point.buffer(size_km * 1000 / 2).bounds()
    start  = ee.Date.fromYMD(year, month, 1)
    end    = start.advance(1, "month")
    img = (ee.ImageCollection("NASA/VIIRS/002/VNP14A1")
           .select("MaxFRP")
           .filterDate(start, end)
           .filterBounds(region).mean())
    url = img.clip(region).getDownloadURL({
        "region": region, "dimensions": f"{px}x{px}", "format": "NPY"})
    resp = urllib.request.urlopen(url).read()
    arr = np.load(io.BytesIO(resp))
    band = arr.dtype.names[0]
    return np.array(arr[band], dtype=np.float32)

def pull(items, outdir):
    os.makedirs(outdir, exist_ok=True)
    saved = skipped = 0
    for name, lat, lon in items:
        for y in YEARS:
            for m in MONTHS:
                fname = f"{outdir}/{name}_{y}_{m:02d}.npy"
                if os.path.exists(fname):
                    saved += 1; continue
                try:
                    tile = get_viirs_tile(lat, lon, y, m)
                    if np.isfinite(tile).mean() < 0.30:
                        skipped += 1
                        print(f"  skip {name} {y}-{m:02d}")
                        continue
                    np.save(fname, tile); saved += 1
                    print(f"  ok   {name} {y}-{m:02d}  mean={np.nanmean(tile):.3e}")
                except Exception as e:
                    skipped += 1
                    print(f"  FAIL {name} {y}-{m:02d}: {e}")
    print(f"  [done] saved={saved} skipped={skipped}")

# same three groups as export_so2.py
plants = pd.read_csv("data/top5_plants.csv")
plant_items = [(str(r["name"]).replace(" ","_").replace("/","_"),
                r["latitude"], r["longitude"]) for _, r in plants.iterrows()]

rural = [
    ("rural_MP_forest",22.50,80.20),("rural_Chhattisgarh",20.30,81.60),
    ("rural_Rajasthan",27.00,73.20),("rural_Odisha_hills",20.10,84.20),
    ("rural_Telangana",18.60,79.10),
]

hard = pd.read_csv("data/hard_negatives.csv")
hard_items = [(str(r["name"]), r["latitude"], r["longitude"]) for _, r in hard.iterrows()]

print("=== VIIRS positive (plants) ==="); pull(plant_items, "data/viirs/positive")
print("=== VIIRS rural ===");             pull(rural,       "data/viirs/negative")
print("=== VIIRS hard negatives ===");    pull(hard_items,  "data/viirs/hard_negative")

print("\nVIIRS export complete.")
