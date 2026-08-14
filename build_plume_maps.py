"""
Phase 1 (NOVEL_METHODOLOGY_PROPOSAL.md): builds plume_model.py's Gaussian
plume hotspot map for every eligible processed facility.

Originally prototyped on 3 facilities (Rihand, Talcher, Anpara) to catch
methodology issues cheaply -- it found one real bug (see module docstring
history in NOVEL_METHODOLOGY_PROPOSAL.md Sec 8) and, in the Phase 2
robustness follow-up, found the spatial-consistency claim needed more
statistical power than 3 facilities could provide (Sec 10). Scaled up
here to every facility with the inputs the plume model needs: a wind
direction (plant_results.json's wind_deg), a Track B emission estimate
(emission_estimates.json), and a saved soundings file (for the spatial
validation scripts downstream, which import ALL_FACILITIES from here).

IMPORTANT WIND-CONVENTION NOTE (caught during verification, before this
script was first run): data/plant_results.json's "wind_deg" field is
computed in process_plant.py as degrees(atan2(u, v)) on the raw ERA5 wind
vector and printed there as "toward {wind_deg} deg" -- i.e. it is the
direction the wind blows TOWARD, not the standard meteorological "blows
FROM" convention. plume_model.plume_grid() expects the standard "FROM"
convention (matching how the physics literature specifies it). This
script converts explicitly (wind_from_deg = wind_deg + 180) rather than
silently assuming -- getting this backwards would point every plume 180
degrees the wrong way without erroring, exactly the kind of bug this
project's own "verify at every stage" instruction exists to catch.

Ablation: for each facility, generates the plume under the default
assumptions (stability class B, H=220m) plus a sensitivity sweep over
stability class {A, B, C} and stack height {150, 220, 275}m, per
NOVEL_METHODOLOGY_PROPOSAL.md Sec 6's confirmed decision (documented
default + explicit sensitivity ablation, not a claimed-precise value).
"""
import json
import os

import numpy as np

import plume_model as pm

ABLATION_STABILITY_CLASSES = ["A", "B", "C"]
ABLATION_STACK_HEIGHTS_M = [150.0, 220.0, 275.0]


def eligible_facilities(plant_results, emission_estimates):
    """Every facility with everything the plume model needs: a wind
    direction, a Track B emission estimate, and a saved soundings file
    (needed by the downstream spatial-validation scripts, not this script
    itself, but computed here once so all three scripts agree on scope)."""
    names = []
    for name, est in emission_estimates.items():
        pr = plant_results.get(name)
        if pr is None or pr.get("wind_deg") is None:
            continue
        if not os.path.exists(f"data/{name}_soundings.npz"):
            continue
        names.append(name)
    return sorted(names)


def load_facility_inputs(name, plant_results, emission_estimates):
    pr = plant_results[name]
    est = emission_estimates[name]
    wind_deg_toward = pr["wind_deg"]
    if wind_deg_toward is None:
        raise ValueError(f"{name}: no wind_deg in plant_results.json, cannot build a plume")
    wind_from_deg = (wind_deg_toward + 180.0) % 360.0  # see module docstring
    return {
        "plant": name,
        "lat": pr["lat"], "lon": pr["lon"],
        "Q_t_per_year": est["q_t_per_year"],        # RAW physics estimate, see plume_model.py docstring
        "Q_t_per_year_std": est["q_t_per_year_std"],
        "wind_speed_ms": est["wind_speed_ms"],
        "wind_deg_toward": wind_deg_toward,          # as stored (process_plant.py's "toward" convention)
        "wind_from_deg": wind_from_deg,              # converted, standard meteorological convention
    }


def main():
    os.makedirs("data/plume_maps", exist_ok=True)
    plant_results = {r["plant"]: r for r in json.load(open("data/plant_results.json"))}
    emission_estimates = {r["plant"]: r for r in json.load(open("data/emission_estimates.json"))}
    facilities = eligible_facilities(plant_results, emission_estimates)
    print(f"Building plume maps for {len(facilities)} facilities: {facilities}")

    summary = {}
    for name in facilities:
        inputs = load_facility_inputs(name, plant_results, emission_estimates)
        print(f"\n=== {name} ===")
        print(f"  Q = {inputs['Q_t_per_year']:,.0f} +/- {inputs['Q_t_per_year_std']:,.0f} t/yr  "
              f"U = {inputs['wind_speed_ms']:.2f} m/s  "
              f"wind blows toward {inputs['wind_deg_toward']:.0f} deg "
              f"(from {inputs['wind_from_deg']:.0f} deg)")

        # default-assumption plume
        grid, ex, nx = pm.plume_grid(
            inputs["Q_t_per_year"], inputs["wind_speed_ms"], inputs["wind_from_deg"])
        peak_conc = float(grid.max())
        peak_idx = np.unravel_index(np.argmax(grid), grid.shape)
        peak_east_km, peak_north_km = ex[peak_idx[1]], nx[peak_idx[0]]
        peak_dist_km = float(np.hypot(peak_east_km, peak_north_km))
        print(f"  [default: class={pm.DEFAULT_STABILITY_CLASS}, H={pm.DEFAULT_STACK_HEIGHT_M:.0f}m] "
              f"peak ground-level conc = {peak_conc:.3e} kg/m^3 at {peak_dist_km:.1f} km downwind")

        np.savez(f"data/plume_maps/{name}_default.npz", grid=grid, east_km=ex, north_km=nx,
                  **{k: v for k, v in inputs.items() if k not in ("plant",)})

        # sensitivity ablation
        ablation = []
        for cls in ABLATION_STABILITY_CLASSES:
            for H in ABLATION_STACK_HEIGHTS_M:
                g, _, _ = pm.plume_grid(inputs["Q_t_per_year"], inputs["wind_speed_ms"],
                                          inputs["wind_from_deg"], H_m=H, stability_class=cls)
                ablation.append({"stability_class": cls, "stack_height_m": H,
                                  "peak_conc_kg_m3": float(g.max())})
                print(f"    class={cls} H={H:.0f}m -> peak = {float(g.max()):.3e} kg/m^3")

        summary[name] = {
            "inputs": {k: v for k, v in inputs.items() if k != "plant"},
            "default_result": {"peak_conc_kg_m3": peak_conc, "peak_downwind_dist_km": peak_dist_km,
                                "stability_class": pm.DEFAULT_STABILITY_CLASS,
                                "stack_height_m": pm.DEFAULT_STACK_HEIGHT_M},
            "ablation": ablation,
        }

    ablation_range_note = (
        "Peak ground-level concentration varies substantially across the stability-class x "
        "stack-height sensitivity grid for every prototype facility -- this range should be "
        "reported alongside any single default-assumption number in the paper, not hidden. "
        "See NOVEL_METHODOLOGY_PROPOSAL.md Sec 6 for why these specific defaults were chosen "
        "(documented literature/regulatory defaults, not fitted or invented)."
    )
    print(f"\n{ablation_range_note}")

    json.dump({"facility_list": facilities, "facilities": summary,
               "note": ablation_range_note},
              open("data/plume_maps/prototype_summary.json", "w"), indent=2)
    print("\n[SAVED] data/plume_maps/prototype_summary.json (+ per-facility .npz grids)")


if __name__ == "__main__":
    main()
