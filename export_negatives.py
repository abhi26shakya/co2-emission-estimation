import ee, numpy as np, os, urllib.request, io
ee.Initialize(project="opportune-lore-415218")

os.makedirs("data/tiles_negative", exist_ok=True)

# Rural / low-emission points: forest or sparsely populated areas,
# deliberately far from big cities and power plants.
negatives = [
    ("rural_MP_forest",    22.50, 80.20),   # Satpura forest belt, Madhya Pradesh
    ("rural_Chhattisgarh", 20.30, 81.60),   # forested central Chhattisgarh
    ("rural_Rajasthan",    27.00, 73.20),   # arid, sparse Rajasthan
    ("rural_Odisha_hills", 20.10, 84.20),   # hilly interior Odisha
    ("rural_Telangana",    18.60, 79.10),   # rural Telangana
]

def get_no2_tile(lat, lon, year=2020, size_km=60, px=64):
    point  = ee.Geometry.Point(lon, lat)
    region = point.buffer(size_km * 1000 / 2).bounds()
    img = (ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
           .select("tropospheric_NO2_column_number_density")
           .filterDate(f"{year}-01-01", f"{year}-12-31")
           .filterBounds(region).mean())
    url = img.clip(region).getDownloadURL({
        "region": region, "dimensions": f"{px}x{px}", "format": "NPY"})
    resp = urllib.request.urlopen(url).read()
    arr = np.load(io.BytesIO(resp))
    band = arr.dtype.names[0]
    return np.array(arr[band], dtype=np.float32)

for name, lat, lon in negatives:
    try:
        tile = get_no2_tile(lat, lon)
        np.save(f"data/tiles_negative/{name}.npy", tile)
        print(f"saved {name:22s} shape={tile.shape} mean={np.nanmean(tile):.3e}")
    except Exception as e:
        print(f"FAILED {name}: {e}")