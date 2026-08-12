"""
Diagnose why Talcher's IME emission-rate estimate (physics_gaussian.py,
7.16M +/- 1.65M t/yr) undershoots its Climate TRACE benchmark (14.24M t/yr,
2021) despite having the tightest, most confident uncertainty band (+/-20%)
of any plant processed this session -- the "tight interval, but wrong" case
flagged after pull_climate_trace.py's comparison.

Reproduces the ad hoc investigation done interactively:
1. Rules out a plant-identity mismatch (Climate TRACE match distance).
2. Rules out the 2020-vs-2021 year offset as the primary cause, using
   Climate TRACE's own multi-year trend for this exact asset.
3. Shows Talcher's raw enhancement is tiny relative to background noise
   (signal-to-noise), unlike a well-bracketed plant (Rihand, used as the
   comparison case throughout).
4. Directly tests sensitivity of the IME estimate to the background-annulus
   definition -- physics_gaussian.py's docstring already states the
   reported sigma does NOT cover background-subtraction structural
   uncertainty, and this script quantifies how large that gap actually is
   for Talcher vs. a robust plant.

Conclusion (see WEEK10_LOG.txt for the write-up): Talcher's estimate isn't
a bug -- it's a real case where the IME method's implicit assumption
(background cleanly separable from signal) breaks down for a plant with a
genuinely thin OCO-3 CO2 signature, and the current uncertainty model
doesn't capture that fragility.
"""
import json
import numpy as np

NEAR = 0.25
BG_IN, BG_OUT = 0.4, 0.9
COMPARISON_PLANT = "Rihand"  # a well-bracketed plant, used as the "robust" baseline


def near_bg_stats(name, plant_row, bg_in=BG_IN, bg_out=BG_OUT, near_r=NEAR):
    d = np.load(f"data/{name}_soundings.npz")
    lat, lon, xco2 = d["lat"], d["lon"], d["xco2"]
    plat, plon = plant_row["lat"], plant_row["lon"]
    dist = np.sqrt((lat - plat) ** 2 + (lon - plon) ** 2)
    near = xco2[dist < near_r]
    bg = xco2[(dist > bg_in) & (dist < bg_out)]
    if len(bg) < 5 or len(near) < 5:
        return None
    bg_mean = bg.mean()
    excess = near - bg_mean
    used = excess > 0
    return {
        "near_n": int(len(near)), "bg_n": int(len(bg)),
        "bg_mean": float(bg_mean), "bg_std": float(bg.std()),
        "near_mean": float(near.mean()),
        "n_used": int(used.sum()),
        "frac_near_above_bg": float(used.sum() / len(near)),
        "excess_used_mean": float(excess[used].mean()) if used.any() else None,
        "excess_used_median": float(np.median(excess[used])) if used.any() else None,
        "signal_to_noise": float((near.mean() - bg_mean) / bg.std()),
    }


def bg_definition_sensitivity(name, plant_row, near_r=NEAR):
    """IME's sensitivity to the background-annulus boundaries -- a proxy
    for the structural uncertainty physics_gaussian.py's docstring says
    its reported sigma does NOT include."""
    d = np.load(f"data/{name}_soundings.npz")
    lat, lon, xco2 = d["lat"], d["lon"], d["xco2"]
    plat, plon = plant_row["lat"], plant_row["lon"]
    dist = np.sqrt((lat - plat) ** 2 + (lon - plon) ** 2)
    near = xco2[dist < near_r]
    definitions = [(0.4, 0.9), (0.3, 0.8), (0.35, 0.7), (0.4, 1.1), (0.5, 1.0)]
    ime_proxies = []
    for bg_in, bg_out in definitions:
        bg = xco2[(dist > bg_in) & (dist < bg_out)]
        if len(bg) < 5:
            continue
        excess = np.clip(near - bg.mean(), 0, None)
        ime_proxies.append(float(excess.sum()))
    if not ime_proxies:
        return None
    ime_proxies = np.array(ime_proxies)
    return {
        "definitions_tested": len(ime_proxies),
        "ime_proxy_min": float(ime_proxies.min()),
        "ime_proxy_max": float(ime_proxies.max()),
        "ime_proxy_range_pct_of_default": float(
            (ime_proxies.max() - ime_proxies.min()) / ime_proxies[0] * 100),
    }


def climate_trace_year_trend(asset_id, years=(2021, 2022, 2023, 2024)):
    import urllib.request
    trend = {}
    for y in years:
        url = (f"https://api.climatetrace.org/v6/assets?countries=IND&sectors=power"
               f"&year={y}&gas=co2&limit=1000")
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.load(resp)
        for a in data.get("assets", []):
            if a["Id"] == asset_id:
                trend[y] = a["EmissionsSummary"][0]["EmissionsQuantity"]
                break
    return trend


def main():
    plant_rows = {r["plant"]: r for r in json.load(open("data/plant_results.json"))}
    emis = {r["plant"]: r for r in json.load(open("data/emission_estimates.json"))}
    ct = json.load(open("data/climate_trace_comparison.json"))
    ct_talcher = next(f for f in ct["facilities"] if f["plant"] == "Talcher")

    print("=== 1. Plant-identity check (Climate TRACE match) ===")
    print(f"  Climate TRACE match: {ct_talcher['climate_trace_name']} "
          f"(dist={ct_talcher['match_dist_deg']:.4f} deg -- effectively exact)")

    print("\n=== 2. Year-offset check: Climate TRACE's own trend for this asset ===")
    trend = climate_trace_year_trend(ct_talcher["climate_trace_id"])
    for y, v in sorted(trend.items()):
        print(f"  {y}: {v:,.0f} t")
    if 2021 in trend and 2024 in trend:
        cagr = (trend[2024] / trend[2021]) ** (1 / 3) - 1
        extrapolated_2020 = trend[2021] / (1 + cagr)
        print(f"  CAGR 2021-2024: {cagr:.1%}")
        print(f"  Extrapolated 2020 estimate: {extrapolated_2020:,.0f} t "
              f"(ours: {emis['Talcher']['q_t_per_year']:,.0f} t -- "
              f"still a {extrapolated_2020 / emis['Talcher']['q_t_per_year']:.1f}x gap "
              f"after removing the year-offset)")

    print("\n=== 3. Signal-to-noise: Talcher vs. a well-bracketed plant ===")
    stats = {}
    for name in ["Talcher", COMPARISON_PLANT]:
        s = near_bg_stats(name, plant_rows[name])
        stats[name] = s
        print(f"  {name}: near_n={s['near_n']} bg_n={s['bg_n']} "
              f"bg_std={s['bg_std']:.3f}  signal/noise={s['signal_to_noise']:.3f}  "
              f"frac_near_above_bg={s['frac_near_above_bg']:.1%}  "
              f"excess_used_median={s['excess_used_median']:.3f}")

    print("\n=== 4. Background-annulus definition sensitivity ===")
    sensitivity = {}
    for name in ["Talcher", COMPARISON_PLANT]:
        sens = bg_definition_sensitivity(name, plant_rows[name])
        sensitivity[name] = sens
        print(f"  {name}: IME proxy range = {sens['ime_proxy_range_pct_of_default']:.0f}% "
              f"of the default-definition value across {sens['definitions_tested']} "
              f"reasonable background definitions")

    result = {
        "climate_trace_match": ct_talcher,
        "climate_trace_year_trend": trend,
        "near_bg_stats": stats,
        "bg_definition_sensitivity": sensitivity,
        "conclusion": (
            "Talcher's estimate is low primarily because its raw CO2 enhancement "
            "(near_mean - bg_mean) is tiny relative to background noise (signal/noise "
            f"~{stats['Talcher']['signal_to_noise']:.2f}, vs "
            f"~{stats[COMPARISON_PLANT]['signal_to_noise']:.2f} for {COMPARISON_PLANT}), "
            "and only ~half of near-plant soundings exceed background at all -- a "
            "near coin-flip. The IME estimate is correspondingly far more sensitive "
            "to the background-annulus definition than a robust plant's. "
            "physics_gaussian.py's reported q_rel_std does not include this "
            "background-definition structural uncertainty, so the tight (+/-20%) "
            "interval understates the real uncertainty for this specific plant. "
            "The 2020-vs-2021 year offset (Climate TRACE has no 2020 asset data) "
            "explains only a modest fraction of the gap once Climate TRACE's own "
            "multi-year growth trend for this asset is extrapolated back to 2020."
        ),
    }
    json.dump(result, open("data/talcher_diagnosis.json", "w"), indent=2)
    print("\n[SAVED] data/talcher_diagnosis.json")


if __name__ == "__main__":
    main()
