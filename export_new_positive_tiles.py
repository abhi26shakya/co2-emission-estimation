"""
Export NO2/SO2/VIIRS positive-class tiles for the 16 Track B facilities added
during the facility-set expansion (Week 9 + this session) that Track A's
detector has never seen as positive-class examples -- it was only ever
trained on the original 5-plant set in data/top5_plants.csv (4 distinct
sites: Vindhyachal, Sasan, Mundra x2, Tirora). This is roadmap item 2 from
NEXT_STEPS.md ("More Track A positive-class facilities"), prep for a future
retrain with a much larger positive class.

Unlike export_facility_tiles.py (which wrote 2020-only tiles to
data/activity_tiles/ for the activity-signal cross-check, deliberately kept
separate from training data), this script writes directly into
data/monthly/positive, data/so2/positive, data/viirs/positive -- the actual
folders build_3channel.py / train_3channel.py read as the positive class.

5 of the 16 facilities (Talcher, Rihand, Sipat, ChandrapurCoal, Anpara)
already have 2020 tiles downloaded in data/activity_tiles/ from Week 10; those
are copied over instead of re-downloaded from Earth Engine. The remaining 11
are fetched fresh. All facilities get 2020 only (not 2019, matching Track B's
OCO-3/wind analysis year and export_facility_tiles.py's precedent) -- adding
2019 would double an already-large download for a second year not used
anywhere else in Track B.
"""
import ee, numpy as np, os, shutil, urllib.request, io, json
ee.Initialize(project="opportune-lore-415218")

YEAR = 2020
MONTHS = list(range(1, 13))
SIZE_KM, PX = 60, 64

COVERED_ALIASES = {"Vindhyachal", "Sasan", "Mundra", "Tirora"}

SOURCES = {
    "no2":   ("COPERNICUS/S5P/OFFL/L3_NO2", "tropospheric_NO2_column_number_density", "data/monthly/positive"),
    "so2":   ("COPERNICUS/S5P/OFFL/L3_SO2", "SO2_column_number_density", "data/so2/positive"),
    "viirs": ("NASA/VIIRS/002/VNP14A1", "MaxFRP", "data/viirs/positive"),
}


def get_tile(collection, band, lat, lon, year, month):
    point = ee.Geometry.Point(lon, lat)
    region = point.buffer(SIZE_KM * 1000 / 2).bounds()
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    img = (ee.ImageCollection(collection).select(band)
           .filterDate(start, end).filterBounds(region).mean())
    url = img.clip(region).getDownloadURL({
        "region": region, "dimensions": f"{PX}x{PX}", "format": "NPY"})
    resp = urllib.request.urlopen(url).read()
    arr = np.load(io.BytesIO(resp))
    band_name = arr.dtype.names[0]
    return np.array(arr[band_name], dtype=np.float32)


rows = json.load(open("data/plant_results.json"))
facilities = [(r["plant"], r["lat"], r["lon"]) for r in rows
              if r["plant"] not in COVERED_ALIASES]
print(f"Facilities needing positive tiles: {[f[0] for f in facilities]} (n={len(facilities)})")

for source, (collection, band, outdir) in SOURCES.items():
    os.makedirs(outdir, exist_ok=True)
    activity_dir = f"data/activity_tiles/{source}"
    copied = saved = skipped = 0
    for name, lat, lon in facilities:
        for m in MONTHS:
            fname = f"{name}_{YEAR}_{m:02d}.npy"
            dest = f"{outdir}/{fname}"
            if os.path.exists(dest):
                saved += 1
                continue
            src = f"{activity_dir}/{fname}"
            if os.path.exists(src):
                shutil.copy(src, dest)
                copied += 1
                continue
            try:
                tile = get_tile(collection, band, lat, lon, YEAR, m)
                if np.isfinite(tile).mean() < 0.30:
                    skipped += 1
                    print(f"  skip {source} {name} {YEAR}-{m:02d} (too few valid pixels)")
                    continue
                np.save(dest, tile)
                saved += 1
                print(f"  ok   {source} {name} {YEAR}-{m:02d}  mean={np.nanmean(tile):.3e}")
            except Exception as e:
                skipped += 1
                print(f"  FAIL {source} {name} {YEAR}-{m:02d}: {str(e)[:60]}")
    print(f"[{source}] copied={copied} downloaded={saved} skipped={skipped}")

print("\nExport complete.")
