import ee, numpy as np, pandas as pd, os, urllib.request, io
ee.Initialize(project="opportune-lore-415218")

YEARS  = [2019, 2020]
MONTHS = list(range(1, 13))

def get_so2_tile(lat, lon, year, month, size_km=60, px=64):
    point  = ee.Geometry.Point(lon, lat)
    region = point.buffer(size_km * 1000 / 2).bounds()
    start  = ee.Date.fromYMD(year, month, 1)
    end    = start.advance(1, "month")
    img = (ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_SO2")
           .select("SO2_column_number_density")
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
                    tile = get_so2_tile(lat, lon, y, m)
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

# same three groups as your NO2 tiles
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

print("=== SO2 positive (plants) ==="); pull(plant_items, "data/so2/positive")
print("=== SO2 rural ===");             pull(rural,       "data/so2/negative")
print("=== SO2 hard negatives ===");    pull(hard_items,  "data/so2/hard_negative")

print("\nSO2 export complete.")