import ee, numpy as np, pandas as pd, os, urllib.request, io
ee.Initialize(project="opportune-lore-415218")

os.makedirs("data/monthly/positive", exist_ok=True)
os.makedirs("data/monthly/negative", exist_ok=True)

plants = pd.read_csv("data/top5_plants.csv")

# same rural points as Day 4
negatives = [
    ("rural_MP_forest",    22.50, 80.20),
    ("rural_Chhattisgarh", 20.30, 81.60),
    ("rural_Rajasthan",    27.00, 73.20),
    ("rural_Odisha_hills", 20.10, 84.20),
    ("rural_Telangana",    18.60, 79.10),
]

YEARS  = [2019, 2020]
MONTHS = list(range(1, 13))

def get_month_tile(lat, lon, year, month, size_km=60, px=64):
    point  = ee.Geometry.Point(lon, lat)
    region = point.buffer(size_km * 1000 / 2).bounds()
    start  = ee.Date.fromYMD(year, month, 1)
    end    = start.advance(1, "month")
    coll = (ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
            .select("tropospheric_NO2_column_number_density")
            .filterDate(start, end)
            .filterBounds(region))
    img = coll.mean()
    url = img.clip(region).getDownloadURL({
        "region": region, "dimensions": f"{px}x{px}", "format": "NPY"})
    resp = urllib.request.urlopen(url).read()
    arr = np.load(io.BytesIO(resp))
    band = arr.dtype.names[0]
    return np.array(arr[band], dtype=np.float32)

def pull_set(items, outdir, label):
    saved, skipped = 0, 0
    for name, lat, lon in items:
        for y in YEARS:
            for m in MONTHS:
                fname = f"{outdir}/{name}_{y}_{m:02d}.npy"
                if os.path.exists(fname):        # resume-friendly: skip done ones
                    saved += 1
                    continue
                try:
                    tile = get_month_tile(lat, lon, y, m)
                    # a tile with almost no valid pixels is useless — skip it
                    valid_frac = np.isfinite(tile).mean()
                    if valid_frac < 0.30:
                        skipped += 1
                        print(f"  skip {name} {y}-{m:02d}  (only {valid_frac*100:.0f}% valid)")
                        continue
                    np.save(fname, tile)
                    saved += 1
                    print(f"  ok   {name} {y}-{m:02d}  mean={np.nanmean(tile):.3e} valid={valid_frac*100:.0f}%")
                except Exception as e:
                    skipped += 1
                    print(f"  FAIL {name} {y}-{m:02d}: {e}")
    print(f"[{label}] saved={saved}  skipped={skipped}")

# positive: the 5 plants (name, lat, lon)
plant_items = [(str(r["name"]).replace(" ", "_").replace("/", "_"),
                r["latitude"], r["longitude"]) for _, r in plants.iterrows()]

print("=== POSITIVE (plants) ===")
pull_set(plant_items, "data/monthly/positive", "positive")

print("=== NEGATIVE (rural) ===")
pull_set(negatives, "data/monthly/negative", "negative")

# final count
pos = len(os.listdir("data/monthly/positive"))
neg = len(os.listdir("data/monthly/negative"))
print(f"\nDONE. positive tiles={pos}  negative tiles={neg}  total={pos+neg}")