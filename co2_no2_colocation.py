import ee, numpy as np, requests, io
import matplotlib.pyplot as plt

ee.Initialize(project="opportune-lore-415218")

PLANT_LAT, PLANT_LON = 24.10, 82.67
BOX = 0.5

region = ee.Geometry.Rectangle(
    [PLANT_LON-BOX, PLANT_LAT-BOX, PLANT_LON+BOX, PLANT_LAT+BOX])

no2_img = (ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
           .select("tropospheric_NO2_column_number_density")
           .filterDate("2020-01-01", "2020-12-31")
           .filterBounds(region)
           .mean()
           .clip(region))

# --- pull NO2 as a real grid via a numpy download (forces resampling) ---
# scale in meters; ~2 km pixels gives a nice grid over a ~110 km box
url = no2_img.getDownloadURL({
    "region": region,
    "scale": 2000,            # 2 km pixels -> ~55x55 grid
    "format": "NPY"
})
resp = requests.get(url)
data = np.load(io.BytesIO(resp.content))
# structured array -> plain 2D float
no2 = np.array(data["tropospheric_NO2_column_number_density"].tolist(), dtype=float)
print("NO2 grid shape:", no2.shape, " mean:", np.nanmean(no2))

# --- load saved CO2 soundings ---
d = np.load("data/vindhyachal_soundings.npz")
clat, clon, cxco2 = d["lat"], d["lon"], d["xco2"]

# --- plot: NO2 heatmap + CO2 dots ---
fig, ax = plt.subplots(figsize=(9,8))
im = ax.imshow(no2, origin="upper", cmap="Blues",
               extent=[PLANT_LON-BOX, PLANT_LON+BOX, PLANT_LAT-BOX, PLANT_LAT+BOX],
               alpha=0.85, aspect="auto")
cb1 = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb1.set_label("NO2 column (mol/m2)")

sc = ax.scatter(clon, clat, c=cxco2, cmap="inferno", s=16, edgecolor="none")
cb2 = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.12)
cb2.set_label("XCO2 (ppm)")

ax.scatter([PLANT_LON],[PLANT_LAT], marker="*", s=500,
           edgecolor="lime", facecolor="none", linewidth=2.5, label="Vindhyachal")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title("Vindhyachal: NO2 (blue) vs CO2 (dots), 2020")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig("data/vindhyachal_co2_no2.png", dpi=130)
print("Saved data/vindhyachal_co2_no2.png")

# --- numeric co-location check ---
ny, nx = no2.shape
r, c = np.unravel_index(np.nanargmax(no2), no2.shape)
# row 0 is TOP (north) because origin='upper'
no2_peak_lat = (PLANT_LAT+BOX) - (r/(ny-1))*(2*BOX)
no2_peak_lon = (PLANT_LON-BOX) + (c/(nx-1))*(2*BOX)
print(f"\nNO2 peak: lat={no2_peak_lat:.3f}, lon={no2_peak_lon:.3f}")
print(f"Plant:    lat={PLANT_LAT:.3f}, lon={PLANT_LON:.3f}")
dist_km = np.sqrt((no2_peak_lat-PLANT_LAT)**2+(no2_peak_lon-PLANT_LON)**2)*111
print(f"NO2 peak is {dist_km:.0f} km from the plant.")