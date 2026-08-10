import ee, numpy as np, pandas as pd, os, urllib.request, io
ee.Initialize(project="opportune-lore-415218")

os.makedirs("data/monthly/hard_negative", exist_ok=True)

hard = pd.read_csv("data/hard_negatives.csv")   # name, latitude, longitude

YEARS  = [2019, 2020]
MONTHS = list(range(1, 13))

def get_month_tile(lat, lon, year, month, size_km=60, px=64):
    point  = ee.Geometry.Point(lon, lat)
    region = point.buffer(size_km * 1000 / 2).bounds()
    start  = ee.Date.fromYMD(year, month, 1)
    end    = start.advance(1, "month")
    img = (ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
           .select("tropospheric_NO2_column_number_density")
           .filterDate(start, end)
           .filterBounds(region).mean())
    url = img.clip(region).getDownloadURL({
        "region": region, "dimensions": f"{px}x{px}", "format": "NPY"})
    resp = urllib.request.urlopen(url).read()
    arr = np.load(io.BytesIO(resp))
    band = arr.dtype.names[0]
    return np.array(arr[band], dtype=np.float32)

saved, skipped = 0, 0
for _, row in hard.iterrows():
    name = str(row["name"])
    for y in YEARS:
        for m in MONTHS:
            fname = f"data/monthly/hard_negative/{name}_{y}_{m:02d}.npy"
            if os.path.exists(fname):          # resume-friendly
                saved += 1
                continue
            try:
                tile = get_month_tile(row["latitude"], row["longitude"], y, m)
                valid_frac = np.isfinite(tile).mean()
                if valid_frac < 0.30:
                    skipped += 1
                    print(f"  skip {name} {y}-{m:02d} (only {valid_frac*100:.0f}% valid)")
                    continue
                np.save(fname, tile)
                saved += 1
                print(f"  ok   {name} {y}-{m:02d}  mean={np.nanmean(tile):.3e}")
            except Exception as e:
                skipped += 1
                print(f"  FAIL {name} {y}-{m:02d}: {e}")

total = len(os.listdir("data/monthly/hard_negative"))
print(f"\nDONE. hard-negative tiles saved={total}  (skipped={skipped})")