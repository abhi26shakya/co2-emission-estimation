import re
import pandas as pd

df = pd.read_csv("data/powerplants.csv")
india_coal = df[(df["country"] == "IND") & (df["primary_fuel"] == "Coal")]
top5 = india_coal.sort_values("capacity_mw", ascending=False).head(5)

cols = ["name", "capacity_mw", "latitude", "longitude"]
top5[cols].to_csv("data/top5_plants.csv", index=False)
print(top5[cols].to_string(index=False))
print("\nSaved data/top5_plants.csv")

# ---------------------------------------------------------------------------
# Week 9: facility-set expansion. The dataset lists separate generating
# units at the same site under different names (e.g. "MUNDRA TPP" and
# "MUNDRA UMPP" ~0.05 deg apart) -- collapse those into one candidate per
# physical site (keep the highest-capacity unit) before taking the top N,
# so process_plant.py doesn't waste an OCO-3 scan re-processing the same
# plume from two names.
N_CANDIDATES = 30
CLUSTER_DEG = 0.05  # ~5 km; units this close are treated as one site

# The 4 plants already processed in Weeks 6-8 (data/plant_results.json,
# data/<Plant>_soundings.npz) use these short aliases, not the raw dataset
# "name" field. Keep them identical so process_plant.py can be repointed at
# data/candidate_plants.csv without changing keys for plants that already
# have saved results.
KNOWN_ALIASES = {
    "VINDH_CHAL STPS": "Vindhyachal",
    "SASAN UMPP": "Sasan",
    "MUNDRA TPP": "Mundra",
    "MUNDRA UMPP": "Mundra",
    "TIRORA TPP": "Tirora",
}

SUFFIX_RE = re.compile(
    r"\s+(STPS|TPP|TPS|UMPP|MPP|CCPP|CPP|GPP|SUPER THERMAL POWER STATION)\b.*$",
    re.IGNORECASE,
)


def short_alias(raw_name: str) -> str:
    if raw_name in KNOWN_ALIASES:
        return KNOWN_ALIASES[raw_name]
    cleaned = SUFFIX_RE.sub("", raw_name).strip()
    cleaned = re.sub(r"[_\-]+", " ", cleaned)
    return cleaned.title().replace(" ", "")


coal_sorted = india_coal.sort_values("capacity_mw", ascending=False).reset_index(drop=True)

kept_rows = []
kept_coords = []
for _, row in coal_sorted.iterrows():
    lat, lon = row["latitude"], row["longitude"]
    if any(abs(lat - klat) < CLUSTER_DEG and abs(lon - klon) < CLUSTER_DEG for klat, klon in kept_coords):
        continue  # same site as an already-kept (higher-capacity) unit
    kept_rows.append(row)
    kept_coords.append((lat, lon))
    if len(kept_rows) >= N_CANDIDATES:
        break

candidates = pd.DataFrame(kept_rows)
candidates["name"] = candidates["name"].apply(short_alias)
candidates = candidates[cols]
candidates.to_csv("data/candidate_plants.csv", index=False)
print(f"\n{candidates.to_string(index=False)}")
print(f"\nSaved data/candidate_plants.csv ({len(candidates)} deduplicated candidates)")