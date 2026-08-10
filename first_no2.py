import ee, geemap
ee.Initialize(project="opportune-lore-415218")

# Singrauli / Vindhyachal area — one of India's largest coal complexes
plant_lat, plant_lon = 24.10, 82.67

point = ee.Geometry.Point(plant_lon, plant_lat)
region = point.buffer(40000).bounds()       # ~40 km box around the plant

no2 = (ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
       .select("tropospheric_NO2_column_number_density")
       .filterDate("2020-01-01", "2020-12-31")
       .filterBounds(region)
       .mean())                              # year-average NO2

Map = geemap.Map(center=[plant_lat, plant_lon], zoom=8)
Map.addLayer(no2.clip(region),
             {"min": 0, "max": 0.0002,
              "palette": ["blue", "green", "yellow", "red"]}, "NO2")
Map.save("no2_map.html")
print("Saved no2_map.html — open it in your browser")