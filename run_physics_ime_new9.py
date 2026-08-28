"""
One-off runner: apply physics_ime.py's IME emission-rate estimate to
just the 9 newly-added candidate facilities (Lalitpur, Akaltara, Bellary,
Dadri(Nctpp), KGudemNew, Pryagraj(Bara), Raichur, RayalSeema, Sagardighi),
merging into the existing data/emission_estimates.json rather than
recomputing (and re-fetching ERA5 wind for) all 30 candidates.
"""
import json
from physics_ime import estimate_emission_rate, _fetch_wind_series

NEW9 = [
    "Lalitpur", "Akaltara", "Bellary", "Dadri(Nctpp)", "KGudemNew",
    "Pryagraj(Bara)", "Raichur", "RayalSeema", "Sagardighi",
]

rows = {r["plant"]: r for r in json.load(open("data/plant_results.json"))}

new_results = []
for name in NEW9:
    row = rows[name]
    try:
        wind_series = _fetch_wind_series(row["lat"], row["lon"])
    except Exception as e:
        print(f"[{name}] wind fetch failed: {str(e)[:60]}, skipping")
        continue
    r = estimate_emission_rate(row, wind_series)
    if r:
        new_results.append(r)

out_path = "data/emission_estimates.json"
existing = json.load(open(out_path))
existing = [r for r in existing if r["plant"] not in NEW9]  # replace if re-run
existing.extend(new_results)
json.dump(existing, open(out_path, "w"), indent=2)
print(f"\n[SAVED] {len(new_results)} new estimates, {len(existing)} total -> {out_path}")
