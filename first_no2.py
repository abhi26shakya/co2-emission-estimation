import ee, urllib.request
ee.Initialize(project="opportune-lore-415218")

plant_lat, plant_lon = 24.10, 82.67
point = ee.Geometry.Point(plant_lon, plant_lat)
region = point.buffer(60000).bounds()

no2 = (ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
       .select("tropospheric_NO2_column_number_density")
       .filterDate("2020-01-01", "2020-12-31")
       .filterBounds(region)
       .mean())

viz = {"min": 0, "max": 0.00015,
       "palette": ["000080", "0000ff", "00ffff", "00ff00",
                   "ffff00", "ff8000", "ff0000"]}
url = no2.clip(region).getThumbURL({
    "region": region, "dimensions": 800, "format": "png", **viz})

urllib.request.urlretrieve(url, "no2_map.png")
print("Saved no2_map.png")
