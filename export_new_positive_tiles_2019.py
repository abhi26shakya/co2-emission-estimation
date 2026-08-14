"""
Follow-up to export_new_positive_tiles.py: that script deliberately exported
2020-only tiles for the 16 newly-added Track B facilities, noting a second
year "would double an already-large download for a second year not used
anywhere else in Track B." This script pays that cost specifically to close
a data-quantity asymmetry surfaced by lofo_track_a.py: the original top-5
facilities (Vindhyachal, Sasan, Mundra x2, Tirora) have 24 tiles each
(2019+2020), while the 16 newly-added ones have only 12 (2020 only) --
half the temporal diversity, and correspondingly small (n=12) LOFO test
folds. Exporting 2019 brings every facility to the same 24-tile depth
before re-running the LOFO harness.

Same source bands, tile geometry, and positive-class output directories as
export_new_positive_tiles.py -- only YEAR differs. No activity_tiles/
shortcut here (that cache only ever held 2020 tiles), so every facility/
month/source is a fresh Earth Engine pull.
"""
import ee, numpy as np, os, urllib.request, io, json
ee.Initialize(project="opportune-lore-415218")

YEAR = 2019
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
print(f"Facilities needing 2019 positive tiles: {[f[0] for f in facilities]} (n={len(facilities)})")

for source, (collection, band, outdir) in SOURCES.items():
    os.makedirs(outdir, exist_ok=True)
    saved = skipped = 0
    for name, lat, lon in facilities:
        for m in MONTHS:
            fname = f"{name}_{YEAR}_{m:02d}.npy"
            dest = f"{outdir}/{fname}"
            if os.path.exists(dest):
                saved += 1
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
    print(f"[{source}] downloaded/kept={saved} skipped={skipped}")
