import ee, numpy as np, pandas as pd, os, urllib.request, io
ee.Initialize(project="opportune-lore-415218")

os.makedirs("data/tiles", exist_ok=True)
plants = pd.read_csv("data/top5_plants.csv")

def get_no2_tile(lat, lon, year=2020, size_km=60, px=64):
    point  = ee.Geometry.Point(lon, lat)
    region = point.buffer(size_km * 1000 / 2).bounds()
    img = (ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
           .select("tropospheric_NO2_column_number_density")
           .filterDate(f"{year}-01-01", f"{year}-12-31")
           .filterBounds(region).mean())
    url = img.clip(region).getDownloadURL({
        "region": region, "dimensions": f"{px}x{px}",
        "format": "NPY"})
    resp = urllib.request.urlopen(url).read()
    arr = np.load(io.BytesIO(resp))
    # NPY download returns a structured array; pull the single band out
    band = arr.dtype.names[0]
    return np.array(arr[band], dtype=np.float32)

for _, row in plants.iterrows():
    try:
        tile = get_no2_tile(row["latitude"], row["longitude"])
        name = str(row["name"]).replace(" ", "_").replace("/", "_")
        np.save(f"data/tiles/{name}.npy", tile)
        print(f"saved {name:30s} shape={tile.shape} mean={np.nanmean(tile):.3e}")
    except Exception as e:
        print(f"FAILED {row['name']}: {e}")