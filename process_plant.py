import sys, os, json, time, signal, fcntl
import ee, numpy as np, requests, io
import earthaccess, xarray as xr
import pandas as pd

# earthaccess.download() has no built-in timeout and has been observed to
# hang indefinitely on a stalled connection (seen twice this session, on
# Kahalgaon and Mouda, both stuck in CLOSE_WAIT to urs.earthdata.nasa.gov).
# A per-granule alarm turns that hang into a normal "skip and move on" case,
# handled by the existing except block below.
class _DownloadTimeout(Exception):
    pass

def _alarm_handler(signum, frame):
    raise _DownloadTimeout("download stalled")

signal.signal(signal.SIGALRM, _alarm_handler)
GRANULE_TIMEOUT_S = 90

# ---------- plant registry ----------
# Loaded from data/candidate_plants.csv (Week 9 facility-set expansion,
# built by pick_plants.py) instead of a hardcoded 4-entry dict. The first 4
# plants processed in Weeks 6-8 (Vindhyachal, Sasan, Mundra, Tirora) keep
# their existing short aliases in that CSV, so this generalization doesn't
# change their lat/lon keys or break already-saved data/plant_results.json
# rows / data/<Plant>_soundings.npz files.
_candidates = pd.read_csv("data/candidate_plants.csv")
PLANTS = {row["name"]: (row["latitude"], row["longitude"]) for _, row in _candidates.iterrows()}

if len(sys.argv) < 2 or sys.argv[1] not in PLANTS:
    print("Usage: python process_plant.py <PlantName>")
    print("Choices:", ", ".join(PLANTS)); sys.exit()

NAME = sys.argv[1]
PLAT, PLON = PLANTS[NAME]
print(f"\n===== Processing {NAME}  ({PLAT}, {PLON}) =====")

# ---------- per-plant process lock ----------
# Two 2026-08-13 incidents (an ad hoc restart left two sequential loops
# running against the same remaining-plant list; a deliberate parallel run
# raced a leftover sequential loop) each ended up with two processes
# scanning the SAME plant concurrently against the same checkpoint file --
# only the final data/plant_results.json write was flock-protected, not the
# per-granule checkpoint. Locking the checkpoint write itself wouldn't
# really fix this: two independent scan loops each hold their own
# in-memory klat/klon/kco2/kday, so serializing their writes just makes
# them cleanly overwrite each other's progress instead of corrupting a
# file mid-write -- the actual bug is two loops running for one plant at
# all. Acquiring an exclusive, non-blocking lock on a per-plant lockfile
# at start prevents that outright: a second process for the same NAME
# fails fast with a clear error instead of silently racing the first.
_lock_path = f"data/{NAME}.lock"
_lockfile = open(_lock_path, "w")
try:
    fcntl.flock(_lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    print(f"[{NAME}] another process already holds the lock for this plant "
          f"({_lock_path}) -- refusing to start a duplicate scan.")
    sys.exit(1)

# ---------- 1. CO2: scan OCO-3 near this plant ----------
earthaccess.login(persist=True)
SEARCH = (PLON-1.0, PLAT-1.0, PLON+1.0, PLAT+1.0)   # tight box = fast
results = earthaccess.search_data(
    short_name="OCO3_L2_Lite_FP", version="11r",
    temporal=("2020-01-01", "2020-12-31"), bounding_box=SEARCH)
print(f"[CO2] granules near {NAME}: {len(results)}")

BOX = 0.5
os.makedirs("data/scan_tmp", exist_ok=True)

# ---------- checkpointing: resume the granule-download loop after a stall/
# kill instead of re-downloading everything from granule 0. Progress is
# flushed to disk after every granule so a restart only redoes the one
# granule that was in flight, not the whole (slow, network-bound) scan. ----
CKPT_PATH = f"data/{NAME}_scan_checkpoint.npz"
start_i = 0
klat, klon, kco2, kday, hits = [], [], [], [], 0
if os.path.exists(CKPT_PATH):
    try:
        ckpt = np.load(CKPT_PATH)
        if int(ckpt["n_results"]) == len(results):  # same search, safe to resume
            klat, klon, kco2, kday = list(ckpt["lat"]), list(ckpt["lon"]), list(ckpt["xco2"]), list(ckpt["day"])
            hits = int(ckpt["hits"])
            start_i = int(ckpt["next_i"])
            print(f"[CO2] resuming from checkpoint: granule {start_i}/{len(results)}, "
                  f"{len(kco2)} soundings, {hits} hit-days so far")
        else:
            print("[CO2] checkpoint doesn't match this search, starting over")
    except Exception as e:
        print(f"[CO2] checkpoint unreadable ({str(e)[:60]}), starting over")

for i in range(start_i, len(results)):
    g = results[i]
    try:
        signal.alarm(GRANULE_TIMEOUT_S)
        try:
            f = earthaccess.download([g], local_path="data/scan_tmp")[0]
        finally:
            signal.alarm(0)
        ds = xr.open_dataset(f)
        lat, lon = ds["latitude"].values, ds["longitude"].values
        xco2, qf = ds["xco2"].values, ds["xco2_quality_flag"].values
        # sounding_id encodes YYYYMMDDHHMMSSmm; integer-divide down to YYYYMMDD
        # so each sounding can later be matched to that day's ERA5 wind speed
        # (per-overpass wind conditioning, see physics_gaussian.py).
        day = (ds["sounding_id"].values // 10**8).astype(np.int64)
        ds.close(); os.remove(f)
        m = (qf==0)&np.isfinite(xco2)& \
            (lat>PLAT-BOX)&(lat<PLAT+BOX)&(lon>PLON-BOX)&(lon<PLON+BOX)
        if m.sum()>0:
            klat.extend(lat[m]); klon.extend(lon[m]); kco2.extend(xco2[m])
            kday.extend(day[m]); hits+=1
    except Exception as e:
        print(f"  skip {i}: {str(e)[:40]}")
    # write-then-rename: a savez() interrupted mid-write (the exact failure
    # mode this checkpoint exists to survive) would otherwise leave a
    # truncated .npz that the next run can't load. np.savez() silently
    # appends ".npz" to the filename if it isn't already there, so the
    # tmp name must end in ".npz" itself or the later os.replace() looks
    # for a path that was never actually written.
    tmp_ckpt = CKPT_PATH.replace(".npz", ".tmp.npz")
    np.savez(tmp_ckpt, lat=np.array(klat), lon=np.array(klon), xco2=np.array(kco2),
             day=np.array(kday, dtype=np.int64), hits=hits, next_i=i+1, n_results=len(results))
    os.replace(tmp_ckpt, CKPT_PATH)

klat, klon, kco2, kday = np.array(klat), np.array(klon), np.array(kco2), np.array(kday, dtype=np.int64)
print(f"[CO2] hit-days: {hits}, soundings: {len(kco2)}")
os.remove(CKPT_PATH)  # scan completed cleanly, no longer needed

if len(kco2) < 20:
    print(f"[!] Too few soundings for {NAME}. Recording as low-coverage.")
    enh = bg_std = near_mean = bg_mean = float("nan")
else:
    dist = np.sqrt((klat-PLAT)**2+(klon-PLON)**2)
    near = kco2[dist<0.25]; bg = kco2[(dist>0.4)&(dist<0.9)]
    near_mean, bg_mean = near.mean(), bg.mean()
    enh, bg_std = near_mean-bg_mean, bg.std()
    print(f"[CO2] enhancement = {enh:+.2f} ppm (bg std {bg_std:.2f})")
np.savez(f"data/{NAME}_soundings.npz", lat=klat, lon=klon, xco2=kco2, day=kday)

# ---------- 2. NO2: co-location ----------
ee.Initialize(project="opportune-lore-415218")
region = ee.Geometry.Rectangle([PLON-BOX, PLAT-BOX, PLON+BOX, PLAT+BOX])
no2img = (ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
          .select("tropospheric_NO2_column_number_density")
          .filterDate("2020-01-01","2020-12-31").filterBounds(region).mean().clip(region))
url = no2img.getDownloadURL({"region":region,"scale":2000,"format":"NPY"})
no2 = np.array(np.load(io.BytesIO(requests.get(url).content))
               ["tropospheric_NO2_column_number_density"].tolist(), dtype=float)
ny, nx = no2.shape
r, c = np.unravel_index(np.nanargmax(no2), no2.shape)
no2_lat = (PLAT+BOX)-(r/(ny-1))*(2*BOX)
no2_lon = (PLON-BOX)+(c/(nx-1))*(2*BOX)
no2_km = np.sqrt((no2_lat-PLAT)**2+(no2_lon-PLON)**2)*111
print(f"[NO2] peak {no2_km:.0f} km from plant")

# ---------- 3. Wind alignment ----------
try:
    pt = ee.Geometry.Point([PLON, PLAT])
    wind = (ee.ImageCollection("ECMWF/ERA5/DAILY")
            .select(["u_component_of_wind_10m","v_component_of_wind_10m"])
            .filterDate("2020-01-01","2020-12-31").mean())
    wv = wind.reduceRegion(ee.Reducer.mean(), pt, scale=27830).getInfo()
    u, v = wv["u_component_of_wind_10m"], wv["v_component_of_wind_10m"]
    wind_deg = (np.degrees(np.arctan2(u, v)))%360
    if len(kco2) >= 20:
        thr = np.percentile(kco2, 80); hi = kco2>=thr
        odeg = (np.degrees(np.arctan2(klon[hi].mean()-PLON, klat[hi].mean()-PLAT)))%360
        wdiff = abs((wind_deg-odeg+180)%360-180)
    else:
        odeg = wdiff = float("nan")
    print(f"[WIND] toward {wind_deg:.0f} deg, CO2 offset {odeg:.0f} deg, diff {wdiff:.0f} deg")
except Exception as e:
    wind_deg = odeg = wdiff = float("nan")
    print(f"[WIND] skipped: {str(e)[:40]}")

# ---------- 4. Append to results table ----------
row = {"plant":NAME,"lat":PLAT,"lon":PLON,"hit_days":hits,"soundings":int(len(kco2)),
       "co2_enhancement_ppm":round(float(enh),3) if enh==enh else None,
       "bg_std_ppm":round(float(bg_std),3) if bg_std==bg_std else None,
       "no2_peak_km":round(float(no2_km),1),
       "wind_deg":round(float(wind_deg),0) if wind_deg==wind_deg else None,
       "co2_offset_deg":round(float(odeg),0) if odeg==odeg else None,
       "wind_co2_diff_deg":round(float(wdiff),0) if wdiff==wdiff else None}

RES = "data/plant_results.json"
# flock guards this read-modify-write: when plants are processed in parallel
# (multiple process_plant.py instances), two unsynchronized writers finishing
# around the same time would otherwise silently drop one plant's result.
import fcntl
with open(RES, "a+") as lockf:
    fcntl.flock(lockf, fcntl.LOCK_EX)
    lockf.seek(0)
    contents = lockf.read()
    allres = json.loads(contents) if contents.strip() else []
    allres = [r for r in allres if r["plant"]!=NAME]  # replace if re-run
    allres.append(row)
    lockf.seek(0); lockf.truncate()
    json.dump(allres, lockf, indent=2)
print(f"\n[SAVED] {NAME} -> {RES}")
print(json.dumps(row, indent=2))