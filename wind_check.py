import ee, numpy as np
import matplotlib.pyplot as plt

ee.Initialize(project="opportune-lore-415218")

PLANT_LAT, PLANT_LON = 24.10, 82.67

# --- 1. mean 2020 wind (u,v at 10m) over the plant ---
pt = ee.Geometry.Point([PLANT_LON, PLANT_LAT])
wind = (ee.ImageCollection("ECMWF/ERA5/DAILY")
        .select(["u_component_of_wind_10m", "v_component_of_wind_10m"])
        .filterDate("2020-01-01", "2020-12-31")
        .mean())
vals = wind.reduceRegion(ee.Reducer.mean(), pt, scale=27830).getInfo()
u = vals["u_component_of_wind_10m"]   # eastward wind (m/s)
v = vals["v_component_of_wind_10m"]   # northward wind (m/s)
print(f"Mean 2020 wind: u(east)={u:.2f} m/s, v(north)={v:.2f} m/s")

# wind blows TOWARD this direction; plume goes downwind (same dir as wind vector)
wind_speed = np.hypot(u, v)
wind_to_deg = (np.degrees(np.arctan2(u, v))) % 360   # 0=N,90=E
print(f"Wind speed: {wind_speed:.2f} m/s, blowing toward {wind_to_deg:.0f} deg (0=N,90=E,180=S,270=W)")

# --- 2. where is the CO2 enhancement, relative to the plant? ---
d = np.load("data/vindhyachal_soundings.npz")
lat, lon, xco2 = d["lat"], d["lon"], d["xco2"]
# take the HIGH-CO2 soundings (top 20%) and find their average offset from plant
thresh = np.percentile(xco2, 80)
hi = xco2 >= thresh
off_lat = lat[hi].mean() - PLANT_LAT
off_lon = lon[hi].mean() - PLANT_LON
off_deg = (np.degrees(np.arctan2(off_lon, off_lat))) % 360  # direction of high-CO2 from plant
off_km  = np.hypot(off_lat, off_lon) * 111
print(f"\nHigh-CO2 (top 20%) center is {off_km:.0f} km from plant, toward {off_deg:.0f} deg")

# --- 3. do they align? ---
diff = abs((wind_to_deg - off_deg + 180) % 360 - 180)
print(f"\nWind-blows-toward: {wind_to_deg:.0f} deg")
print(f"High-CO2 offset:   {off_deg:.0f} deg")
print(f"Angular difference: {diff:.0f} deg")
if diff < 45:
    print("-> ALIGNED: high CO2 sits downwind of the plant. Plume-consistent!")
elif diff < 90:
    print("-> Partial alignment. Suggestive but not clean (mixed winds likely).")
else:
    print("-> Not aligned. Aggregate winds too mixed; needs per-day analysis (Option 2).")

# --- 4. visualize: plant, wind arrow, high-CO2 dots ---
plt.figure(figsize=(8,8))
plt.scatter(lon, lat, c=xco2, cmap="inferno", s=14, alpha=0.6)
plt.colorbar(label="XCO2 (ppm)")
plt.scatter(lon[hi], lat[hi], facecolor="none", edgecolor="red", s=40, label="top-20% CO2")
plt.scatter([PLANT_LON],[PLANT_LAT], marker="*", s=500, color="lime",
            edgecolor="black", label="Vindhyachal")
# wind arrow (scaled)
sc = 0.15/max(wind_speed,0.1)
plt.arrow(PLANT_LON, PLANT_LAT, u*sc, v*sc, width=0.006,
          color="cyan", length_includes_head=True, label="mean wind")
plt.legend(); plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.title("Vindhyachal: mean wind vs high-CO2 location")
plt.tight_layout(); plt.savefig("data/vindhyachal_wind.png", dpi=130)
print("\nSaved data/vindhyachal_wind.png")