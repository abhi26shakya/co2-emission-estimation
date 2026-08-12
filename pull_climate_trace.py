"""
Pull Climate TRACE's independent power-sector CO2 estimates for India and
match them to Track B's 9 facilities, as an external benchmark -- per
RESEARCH_PLAN.md roadmap step 6 ("Climate TRACE as an independent
benchmark specifically for Indian coal plants") and Sec 7 ("use only as
independent benchmark, never as training label").

API discovered empirically via api.climatetrace.org/v6/swagger/openapi.json
(no official endpoint docs were guessed -- verified live with curl before
writing this script): GET /v6/assets accepts countries/sectors/gas/year
query params and returns per-asset EmissionsSummary + a Centroid
[lon, lat], which for Vindhyachal matches our own coordinates exactly
(82.6719, 24.0983), confirming lat/lon matching is reliable here.

Year caveat: Climate TRACE's asset-level API only has data from 2021
onward (year=2020 returns zero assets) -- Track B's OCO-3/wind analysis
is for 2020. This script uses 2021, the nearest available year, and this
one-year offset should be kept in mind when interpreting the comparison
(coal-plant output is fairly stable year to year for baseload plants, but
this is not a same-year comparison).

Per RESEARCH_PLAN.md Sec 10/12: this is benchmark-vs-benchmark, not
benchmark-vs-ground-truth. Climate TRACE's own accuracy on Indian plants
is itself unvalidated (it's a satellite-thermal-proxy ML pipeline, not
CEMS-measured data) -- a mismatch should be read as "two independent,
imperfect estimators disagreeing," not proof either one is wrong.
"""
import json
import urllib.request
import numpy as np

API_BASE = "https://api.climatetrace.org/v6"
YEAR = 2021          # nearest available year to Track B's 2020 analysis
MATCH_TOLERANCE_DEG = 0.05   # ~5km, same clustering radius used in pick_plants.py


def fetch_india_power_assets(year):
    url = f"{API_BASE}/assets?countries=IND&sectors=power&year={year}&gas=co2&limit=1000"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.load(resp)
    return data.get("assets", [])


def nearest_asset(lat, lon, assets):
    best, best_dist = None, None
    for a in assets:
        centroid = a.get("Centroid", {}).get("Geometry")
        if not centroid:
            continue
        a_lon, a_lat = centroid
        dist = np.hypot(lat - a_lat, lon - a_lon)
        if best_dist is None or dist < best_dist:
            best, best_dist = a, dist
    return best, best_dist


def main():
    print(f"Fetching Climate TRACE India power-sector CO2 assets for {YEAR}...")
    assets = fetch_india_power_assets(YEAR)
    print(f"  {len(assets)} assets returned")

    plant_rows = {r["plant"]: r for r in json.load(open("data/plant_results.json"))}
    q_estimates = {r["plant"]: r for r in json.load(open("data/emission_estimates.json"))}

    results = []
    for name, row in plant_rows.items():
        lat, lon = row["lat"], row["lon"]
        match, dist_deg = nearest_asset(lat, lon, assets)
        if match is None or dist_deg > MATCH_TOLERANCE_DEG:
            print(f"[{name}] no Climate TRACE match within {MATCH_TOLERANCE_DEG} deg "
                  f"(nearest dist={dist_deg:.4f} deg)" if match else f"[{name}] no match found")
            results.append({"plant": name, "climate_trace_match": None})
            continue

        ct_co2_t = match["EmissionsSummary"][0]["EmissionsQuantity"]
        entry = {
            "plant": name,
            "climate_trace_name": match["Name"],
            "climate_trace_id": match["Id"],
            "climate_trace_year": YEAR,
            "climate_trace_co2_t": ct_co2_t,
            "match_dist_deg": float(dist_deg),
        }
        our_q = q_estimates.get(name)
        if our_q:
            q = our_q["q_t_per_year"]
            entry["our_q_t_per_year"] = q
            entry["our_q_year"] = 2020
            entry["ratio_ours_over_ct"] = q / ct_co2_t if ct_co2_t else None
            lo, hi = our_q["q_t_per_year_low"], our_q["q_t_per_year_high"]
            entry["bracketed_by_our_interval"] = bool(lo <= ct_co2_t <= hi)
            print(f"[{name}] Climate TRACE ({YEAR})={ct_co2_t:,.0f} t  vs  "
                  f"ours (2020)={q:,.0f} t [{lo:,.0f}, {hi:,.0f}]  "
                  f"ratio={entry['ratio_ours_over_ct']:.2f}  "
                  f"bracketed={entry['bracketed_by_our_interval']}")
        else:
            print(f"[{name}] Climate TRACE ({YEAR})={ct_co2_t:,.0f} t  "
                  f"(no Track B estimate to compare -- plant was skipped)")
        results.append(entry)

    out = {
        "year_caveat": ("Climate TRACE data is for 2021 (nearest available year; "
                         "asset-level API has no 2020 coverage). Track B's OCO-3/wind "
                         "estimates are for 2020. This is a one-year-offset comparison, "
                         "not same-year."),
        "benchmark_caveat": ("Benchmark-vs-benchmark, not benchmark-vs-ground-truth. "
                              "Climate TRACE's own accuracy on Indian plants is unvalidated. "
                              "Per RESEARCH_PLAN.md, not used as a training label."),
        "facilities": results,
    }
    json.dump(out, open("data/climate_trace_comparison.json", "w"), indent=2)
    print("\n[SAVED] data/climate_trace_comparison.json")


if __name__ == "__main__":
    main()
