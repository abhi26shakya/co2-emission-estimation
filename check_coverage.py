"""
Cheap OCO-3 coverage pre-check for candidate plants (Week 9, facility-set
expansion). process_plant.py's granule count is printed as a side effect of
the same search call this script makes, but process_plant.py then downloads
every granule -- expensive. This script only calls earthaccess.search_data()
(metadata only) so the full candidate list can be ranked/filtered before
committing to the slow per-plant download+process step.

Granule count near the plant isn't a perfect proxy for a usable estimate --
Tirora had 671 soundings but only 5 hit-days and still came out too low;
Mundra had 57 soundings and was skipped outright (see WEEK6-8 logs) -- so
this is a ranking/triage tool, not a hard pass/fail gate.
"""
import json
import pandas as pd
import earthaccess

earthaccess.login(persist=True)

candidates = pd.read_csv("data/candidate_plants.csv")

results = []
for _, row in candidates.iterrows():
    name, plat, plon = row["name"], row["latitude"], row["longitude"]
    bbox = (plon - 1.0, plat - 1.0, plon + 1.0, plat + 1.0)
    try:
        granules = earthaccess.search_data(
            short_name="OCO3_L2_Lite_FP", version="11r",
            temporal=("2020-01-01", "2020-12-31"), bounding_box=bbox)
        n = len(granules)
    except Exception as e:
        print(f"[{name}] search failed: {str(e)[:60]}")
        n = None
    print(f"[{name}] granules: {n}")
    results.append({"plant": name, "capacity_mw": float(row["capacity_mw"]),
                     "latitude": float(plat), "longitude": float(plon),
                     "granules": n})

results.sort(key=lambda r: (r["granules"] is not None, r["granules"]), reverse=True)
json.dump(results, open("data/candidate_coverage.json", "w"), indent=2)
print(f"\n[SAVED] {len(results)} candidates ranked by granule count -> data/candidate_coverage.json")
