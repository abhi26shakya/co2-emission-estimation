"""
Export NO2/SO2/VIIRS monthly tiles for the Track B facility set that don't
already have them from Track A's original export (export_monthly.py /
export_so2.py / export_viirs.py, which only covered the 5 plants in
data/top5_plants.csv). This is prep for extract_activity_signal.py, which
runs the already-trained Track A CNN over these tiles to get a per-facility
activity signal for each of the 9 Track B facilities (Week 9's
data/plant_results.json), per RESEARCH_PLAN.md's roadmap step 3.

Only pulls 2020 (not 2019, unlike the original Track A export) since
Track B's OCO-3/wind analysis is 2020-only -- the activity signal needs to
line up with the same year as the emission-rate estimate it will eventually
correct.

Tiles are written to data/activity_tiles/{no2,so2,viirs}/ rather than
data/monthly|so2|viirs/positive/ so this doesn't silently change what a
future Track A retrain would train on -- RESEARCH_PLAN.md Sec 15 says reuse
the existing trained CNN as-is, not retrain it.
"""
import ee, numpy as np, pandas as pd, os, urllib.request, io, json
ee.Initialize(project="opportune-lore-415218")

YEAR = 2020
MONTHS = list(range(1, 13))
SIZE_KM, PX = 60, 64

# facilities already covered by Track A's original export (data/top5_plants.csv
# names, mapped to the short aliases used in data/plant_results.json)
COVERED_ALIASES = {"Vindhyachal", "Sasan", "Mundra", "Tirora"}


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


SOURCES = {
    "no2":   ("COPERNICUS/S5P/OFFL/L3_NO2", "tropospheric_NO2_column_number_density"),
    "so2":   ("COPERNICUS/S5P/OFFL/L3_SO2", "SO2_column_number_density"),
    "viirs": ("NASA/VIIRS/002/VNP14A1", "MaxFRP"),
}

rows = json.load(open("data/plant_results.json"))
facilities = [(r["plant"], r["lat"], r["lon"]) for r in rows
              if r["plant"] not in COVERED_ALIASES]

print(f"Facilities needing new tile export: {[f[0] for f in facilities]}")

for source, (collection, band) in SOURCES.items():
    outdir = f"data/activity_tiles/{source}"
    os.makedirs(outdir, exist_ok=True)
    saved = skipped = 0
    for name, lat, lon in facilities:
        for m in MONTHS:
            fname = f"{outdir}/{name}_{YEAR}_{m:02d}.npy"
            if os.path.exists(fname):
                saved += 1
                continue
            try:
                tile = get_tile(collection, band, lat, lon, YEAR, m)
                if np.isfinite(tile).mean() < 0.30:
                    skipped += 1
                    print(f"  skip {source} {name} {YEAR}-{m:02d} (too few valid pixels)")
                    continue
                np.save(fname, tile)
                saved += 1
                print(f"  ok   {source} {name} {YEAR}-{m:02d}  mean={np.nanmean(tile):.3e}")
            except Exception as e:
                skipped += 1
                print(f"  FAIL {source} {name} {YEAR}-{m:02d}: {str(e)[:60]}")
    print(f"[{source}] saved={saved} skipped={skipped}")

print("\nExport complete.")
