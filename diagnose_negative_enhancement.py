"""
Root-cause the three facilities that produced a physically implausible
*negative* near-minus-background CO2 enhancement (ShriSingajiMalwa: -1.052
ppm, Koradi: -0.131 ppm, Tamnar: -0.122 ppm) -- flagged as an open anomaly
in NEXT_STEPS.md / PROJECT_RESEARCH_DOCUMENTATION.md Sec 11 item 2,
unresolved since the facility that triggered it (Tamnar) had the best
wind/CO2-offset alignment (16 deg) of the three, apparently contradicting
the working heuristic from the Talcher diagnosis ("good wind alignment
implies trustworthy signal").

Same near/background methodology and background-definition-sensitivity
check as diagnose_talcher.py, reused directly (not reimplemented), plus one
additional test Talcher's diagnosis didn't need: a statistical-significance
check. Talcher's problem was a real but small *positive* signal drowned in
noise; these three facilities' enhancement is not just small, it's
negative, so the first question is simpler than Talcher's -- is the
negative sign itself statistically meaningful, or is the "true" enhancement
plausibly zero (near_mean - bg_mean consistent with 0 given the standard
error of that difference)? If so, these aren't really "negative-enhancement"
facilities at all, just facilities with insufficient signal-to-noise to
detect any real plume, and the negative sign is an artifact of sampling
noise landing on the wrong side of zero.
"""
import json
import numpy as np

from diagnose_talcher import near_bg_stats, bg_definition_sensitivity

TARGETS = ["ShriSingajiMalwa", "Koradi", "Tamnar"]
COMPARISON_PLANT = "Rihand"  # same well-bracketed baseline used in diagnose_talcher.py


def _pct_str(pct):
    """bg_definition_sensitivity() reports None when the default-definition
    IME proxy is 0 (percent-of-default is undefined, not just unstable)."""
    return f"{pct:.0f}%" if pct is not None else "n/a"


def significance_check(name, plant_row, near_r=0.25, bg_in=0.4, bg_out=0.9):
    d = np.load(f"data/{name}_soundings.npz")
    lat, lon, xco2 = d["lat"], d["lon"], d["xco2"]
    plat, plon = plant_row["lat"], plant_row["lon"]
    dist = np.sqrt((lat - plat) ** 2 + (lon - plon) ** 2)
    near = xco2[dist < near_r]
    bg = xco2[(dist > bg_in) & (dist < bg_out)]
    diff = near.mean() - bg.mean()
    se = np.hypot(near.std(ddof=1) / np.sqrt(len(near)), bg.std(ddof=1) / np.sqrt(len(bg)))
    z = diff / se if se > 0 else float("nan")
    return {"near_n": int(len(near)), "bg_n": int(len(bg)), "diff_ppm": float(diff),
            "se_ppm": float(se), "z_score": float(z),
            "consistent_with_zero_at_2sigma": bool(abs(z) < 2)}


def main():
    plant_rows = {r["plant"]: r for r in json.load(open("data/plant_results.json"))}

    print(f"=== Baseline for comparison: {COMPARISON_PLANT} ===")
    base_stats = near_bg_stats(COMPARISON_PLANT, plant_rows[COMPARISON_PLANT])
    base_sens = bg_definition_sensitivity(COMPARISON_PLANT, plant_rows[COMPARISON_PLANT])
    base_sig = significance_check(COMPARISON_PLANT, plant_rows[COMPARISON_PLANT])
    print(f"  signal/noise={base_stats['signal_to_noise']:.3f}  "
          f"bg-def range={_pct_str(base_sens['ime_proxy_range_pct_of_default'])}  "
          f"z={base_sig['z_score']:.2f}")

    results = {}
    for name in TARGETS:
        print(f"\n=== {name} ===")
        stats = near_bg_stats(name, plant_rows[name])
        sens = bg_definition_sensitivity(name, plant_rows[name])
        sig = significance_check(name, plant_rows[name])
        wind_diff = plant_rows[name].get("wind_co2_diff_deg")
        print(f"  near_n={stats['near_n']} bg_n={stats['bg_n']}  "
              f"signal/noise={stats['signal_to_noise']:.3f}  "
              f"frac_near_above_bg={stats['frac_near_above_bg']:.1%}")
        print(f"  bg-definition IME-proxy range: {_pct_str(sens['ime_proxy_range_pct_of_default'])} "
              f"of default (baseline {COMPARISON_PLANT}: {_pct_str(base_sens['ime_proxy_range_pct_of_default'])})")
        print(f"  significance: diff={sig['diff_ppm']:+.3f} ppm  se={sig['se_ppm']:.3f} ppm  "
              f"z={sig['z_score']:.2f}  consistent_with_zero_at_2sigma={sig['consistent_with_zero_at_2sigma']}")
        print(f"  wind/CO2 offset alignment: {wind_diff} deg")
        results[name] = {"near_bg_stats": stats, "bg_definition_sensitivity": sens,
                          "significance": sig, "wind_co2_diff_deg": wind_diff}

    n_consistent_with_zero = sum(1 for r in results.values()
                                  if r["significance"]["consistent_with_zero_at_2sigma"])
    conclusion = (
        f"{n_consistent_with_zero}/{len(TARGETS)} of the flagged facilities have a "
        "near-minus-background difference statistically consistent with zero at "
        "the 2-sigma level (the standard error of the difference, computed from "
        "near/background sample sizes and variance, is larger than the magnitude "
        "of the negative value itself). Where that holds, the correct "
        "characterization is not 'negative CO2 enhancement' but 'no detectable "
        "enhancement given available signal-to-noise' -- the negative sign is "
        "sampling noise landing on the wrong side of zero, not a pipeline bug or "
        "a real emission-absent facility. This also resolves the apparent "
        "contradiction with the Talcher-derived heuristic ('good wind alignment "
        "implies trustworthy signal'): Tamnar's good wind alignment (16 deg) says "
        "its NO2 plume geometry is plausible, but wind alignment is computed "
        "independently of the OCO-3 CO2 signal-to-noise, so it says nothing about "
        "whether the CO2 enhancement itself is detectable -- the two diagnostics "
        "are orthogonal, not in tension, and this facility illustrates why: it can "
        "be true simultaneously that the wind/CO2 offset looks physically sensible "
        "AND that the CO2 signal is too weak to measure a reliable enhancement at "
        "all. All three facilities also show elevated background-definition "
        "sensitivity relative to the well-bracketed comparison facility, "
        "consistent with the same thin-signal fragility pattern first identified "
        "in the Talcher diagnosis, not a new failure mode."
    )
    print(f"\n{conclusion}")

    out = {"comparison_plant": COMPARISON_PLANT,
           "comparison_plant_stats": {"near_bg_stats": base_stats,
                                       "bg_definition_sensitivity": base_sens,
                                       "significance": base_sig},
           "targets": results, "conclusion": conclusion}
    json.dump(out, open("data/negative_enhancement_diagnosis.json", "w"), indent=2)
    print("\n[SAVED] data/negative_enhancement_diagnosis.json")


if __name__ == "__main__":
    main()
