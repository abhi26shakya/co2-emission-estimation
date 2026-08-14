"""
Root-cause TalwandiSabo's negative near-minus-background CO2 enhancement
(-1.277 ppm, bg_std=2.018), flagged when its plant_results.json row was
committed (see commit "Commit TalwandiSabo (already processed) and expand
candidate list to 30"). Same shape of anomaly as the three facilities
diagnose_negative_enhancement.py / diagnose_shrisingajimalwa.py resolved
earlier this session (all statistically-noise or seasonal-sampling
artifacts, not real negative signals or pipeline bugs) -- this script
runs that same two-stage diagnosis against TalwandiSabo specifically,
reusing the exact functions from both scripts rather than reimplementing
them.

Stage 1 (significance_check, from diagnose_negative_enhancement.py): is
the negative sign itself statistically meaningful, or consistent with
zero given the standard error of the near-minus-background difference?

Stage 2 (only if Stage 1 says the negative sign IS significant, as it was
for ShriSingajiMalwa): a seasonal-sampling-imbalance check, same
same-month-only comparison diagnose_shrisingajimalwa.py used -- are
near-plant and background soundings drawn from systematically different
months, such that restricting both to their one shared month flips the
sign?
"""
import json
import numpy as np

from diagnose_talcher import near_bg_stats, bg_definition_sensitivity
from diagnose_negative_enhancement import significance_check
from diagnose_shrisingajimalwa import seasonal_breakdown, same_month_comparison, directional_breakdown

TARGET = "TalwandiSabo"
COMPARISON_PLANT = "Rihand"  # same well-bracketed baseline used throughout this session


def main():
    plant_rows = {r["plant"]: r for r in json.load(open("data/plant_results.json"))}
    row = plant_rows[TARGET]

    print(f"=== Baseline for comparison: {COMPARISON_PLANT} ===")
    base_stats = near_bg_stats(COMPARISON_PLANT, plant_rows[COMPARISON_PLANT])
    base_sens = bg_definition_sensitivity(COMPARISON_PLANT, plant_rows[COMPARISON_PLANT])
    base_sig = significance_check(COMPARISON_PLANT, plant_rows[COMPARISON_PLANT])
    print(f"  signal/noise={base_stats['signal_to_noise']:.3f}  "
          f"bg-def range={base_sens['ime_proxy_range_pct_of_default']:.0f}%  "
          f"z={base_sig['z_score']:.2f}")

    print(f"\n=== {TARGET}: Stage 1 -- statistical-significance check ===")
    stats = near_bg_stats(TARGET, row)
    sens = bg_definition_sensitivity(TARGET, row)
    sig = significance_check(TARGET, row)
    wind_diff = row.get("wind_co2_diff_deg")
    print(f"  near_n={stats['near_n']} bg_n={stats['bg_n']}  "
          f"signal/noise={stats['signal_to_noise']:.3f}  "
          f"frac_near_above_bg={stats['frac_near_above_bg']:.1%}")
    print(f"  bg-definition IME-proxy range: {sens['ime_proxy_range_pct_of_default']:.0f}% "
          f"of default (baseline {COMPARISON_PLANT}: {base_sens['ime_proxy_range_pct_of_default']:.0f}%)")
    print(f"  significance: diff={sig['diff_ppm']:+.3f} ppm  se={sig['se_ppm']:.3f} ppm  "
          f"z={sig['z_score']:.2f}  consistent_with_zero_at_2sigma={sig['consistent_with_zero_at_2sigma']}")
    print(f"  wind/CO2 offset alignment: {wind_diff} deg")

    out = {
        "target": TARGET, "comparison_plant": COMPARISON_PLANT,
        "comparison_plant_stats": {"near_bg_stats": base_stats,
                                    "bg_definition_sensitivity": base_sens,
                                    "significance": base_sig},
        "stage1_near_bg_stats": stats,
        "stage1_bg_definition_sensitivity": sens,
        "stage1_significance": sig,
        "wind_co2_diff_deg": wind_diff,
    }

    if sig["consistent_with_zero_at_2sigma"]:
        conclusion = (
            f"{TARGET}'s near-minus-background difference (z={sig['z_score']:.2f}) is "
            "statistically consistent with zero at the 2-sigma level. Same "
            "characterization as Koradi/Tamnar: not 'negative CO2 enhancement' but "
            "'no detectable enhancement given available signal-to-noise' -- the "
            "negative sign is sampling noise landing on the wrong side of zero, not "
            "a pipeline bug. Stage 2 (seasonal-sampling check) not run, since it was "
            "only needed for ShriSingajiMalwa, the one facility where the negative "
            "sign WAS statistically significant."
        )
        print(f"\n{conclusion}")
        out["stage2_seasonal_check_run"] = False
        out["conclusion"] = conclusion
    else:
        print(f"\n=== {TARGET}: Stage 2 -- seasonal-sampling-imbalance check ===")
        print("(negative sign is statistically significant, same as ShriSingajiMalwa -- "
              "checking for the same seasonal near/background mismatch)")
        season = seasonal_breakdown(TARGET, row)
        direction = directional_breakdown(TARGET, row)
        same_month = same_month_comparison(TARGET, row)
        print(f"  near_mean_month={season['near_mean_month']}  bg_mean_month={season['bg_mean_month']}  "
              f"offset={season['month_offset']}")
        print(f"  near month counts: {season['near_month_counts']}")
        print(f"  bg month counts:   {season['bg_month_counts']}")
        print(f"  directional (bg-ring) quadrant spread: {direction['quadrant_spread_ppm']}")
        if same_month["common_months"]:
            print(f"  same-month comparison: common_months={same_month['common_months']}  "
                  f"near_n={same_month['near_n']}  bg_n={same_month['bg_n']}")
            print(f"  near_mean={same_month['near_mean_xco2']:.3f}  bg_mean={same_month['bg_mean_xco2']:.3f}  "
                  f"diff={same_month['diff_ppm']:+.3f}  se={same_month['se_ppm']:.3f}  "
                  f"z={same_month['z_score']:.2f}" if same_month['z_score'] is not None else "  z=n/a")
            flips = same_month["diff_ppm"] > 0 and sig["diff_ppm"] < 0
            conclusion = (
                f"{TARGET}'s negative sign IS statistically significant (z={sig['z_score']:.2f}), "
                f"same as ShriSingajiMalwa. Seasonal breakdown: near-plant soundings averaged "
                f"month {season['near_mean_month']}, background averaged month {season['bg_mean_month']} "
                f"(offset {season['month_offset']}). Restricting to the "
                f"{len(same_month['common_months'])} shared month(s) "
                f"{'flips the sign to positive' if flips else 'does NOT flip the sign'} "
                f"(same-month diff={same_month['diff_ppm']:+.3f} ppm, z={same_month['z_score']}). "
                + ("This matches the ShriSingajiMalwa pattern: a seasonal-sampling-imbalance "
                   "artifact, not a real negative signal or pipeline bug."
                   if flips else
                   "Unlike ShriSingajiMalwa, the seasonal explanation does not resolve this case -- "
                   "the negative enhancement remains unexplained and should be treated as an open "
                   "anomaly, not assumed to be a sampling artifact.")
            )
        else:
            conclusion = (
                f"{TARGET}'s negative sign IS statistically significant (z={sig['z_score']:.2f}), "
                "but near-plant and background soundings share no common month, so the "
                "same-month comparison diagnose_shrisingajimalwa.py used cannot be applied "
                "directly here. Remains an open, unexplained anomaly."
            )
        print(f"\n{conclusion}")
        out["stage2_seasonal_check_run"] = True
        out["seasonal_breakdown"] = season
        out["directional_breakdown"] = direction
        out["same_month_comparison"] = same_month
        out["conclusion"] = conclusion

    json.dump(out, open("data/talwandisabo_negative_enhancement_diagnosis.json", "w"), indent=2)
    print("\n[SAVED] data/talwandisabo_negative_enhancement_diagnosis.json")


if __name__ == "__main__":
    main()
