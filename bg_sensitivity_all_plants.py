"""
Week 13, experiment 3: does the background-CO2 definition explain
Rihand's error?

Week 10's diagnose_talcher.py already tested this for Talcher (swing
20.2% across 5 background-annulus definitions) and Rihand (swing 4.3%),
using bg_definition_sensitivity(). This script reuses that function
UNMODIFIED across all 30 plants -- Week 13 experiments 1 (overpass
density) and 2 (signal-to-noise) both ruled out their respective causes
for Rihand's bad Q estimate; this checks background-definition
sensitivity as a third candidate.
"""
import json

import numpy as np

from diagnose_talcher import bg_definition_sensitivity

RIHAND_WEEK10_SWING_PCT = 4.2945295537750985  # from data/talcher_diagnosis.json


def main():
    plant_rows = {r["plant"]: r for r in json.load(open("data/plant_results.json"))}
    q_correction = json.load(open("data/q_correction_model_results.json"))
    feature_table = {r["plant"]: r for r in q_correction["feature_table"]}

    rows = []
    for name in sorted(plant_rows):
        sens = bg_definition_sensitivity(name, plant_rows[name])
        if sens is None:
            print(f"[{name}] too few background soundings for any definition, skipping")
            continue
        row = {"plant": name, **sens}
        rows.append(row)
        pct = row["ime_proxy_range_pct_of_default"]
        pct_str = f"{pct:.1f}%" if pct is not None else "n/a (default IME proxy is 0)"
        print(f"[{name}] swing={pct_str}  "
              f"(definitions_tested={row['definitions_tested']})")

    rihand = next(r for r in rows if r["plant"] == "Rihand")
    rihand_swing = rihand["ime_proxy_range_pct_of_default"]
    print(f"\n=== Rihand check ===")
    print(f"  Week 10 (2-plant run): 4.3%")
    print(f"  This run (30-plant):   {rihand_swing:.1f}%")
    rihand_still_small = rihand_swing is not None and rihand_swing < 20.0  # Talcher's swing, the known-bad reference
    print(f"  Still small (< Talcher's 20.2% reference) = {rihand_still_small}")

    # correlate swing% with |log_ratio| on the N=24 with both a swing and a log_ratio
    matched = [(r, feature_table[r["plant"]]) for r in rows
               if r["plant"] in feature_table and r["ime_proxy_range_pct_of_default"] is not None]
    swing_vals = np.array([r["ime_proxy_range_pct_of_default"] for r, _ in matched])
    abs_log_ratio = np.array([abs(f["log_ratio"]) for _, f in matched])
    r_swing = float(np.corrcoef(swing_vals, abs_log_ratio)[0, 1])

    r_hit_days = -0.30971260219321334   # Week 12 feature table
    snr_data = json.load(open("data/snr_all_plants.json"))
    r_snr = snr_data["q1_snr_vs_hit_days"]["r_snr_abs_log_ratio"]

    print(f"\n=== Swing% vs hit_days vs SNR as predictors of |log_ratio| (N={len(matched)}) ===")
    print(f"  r(swing%, |log_ratio|)   = {r_swing:+.3f}")
    print(f"  r(hit_days, |log_ratio|) = {r_hit_days:+.3f}  (Week 12)")
    print(f"  r(SNR, |log_ratio|)      = {r_snr:+.3f}  (Week 13 exp 2)")
    ranked = sorted(
        [("swing%", abs(r_swing)), ("hit_days", abs(r_hit_days)), ("SNR", abs(r_snr))],
        key=lambda x: -x[1],
    )
    print(f"  -> ranked by |r| (strongest first): {[f'{n} ({v:.3f})' for n, v in ranked]}")

    rihand_ruled_out = rihand_still_small
    conclusion = (
        "Background-annulus definition is RULED OUT as Rihand's specific problem, same as "
        "overpass coverage (exp 1) and signal-to-noise (exp 2): its swing stayed small "
        f"({rihand_swing:.1f}%, well below Talcher's known-bad 20.2%) under the full "
        "30-plant run, not just the original 2-plant comparison. Narrows the open question "
        "to wind-day matching quality or plume geometry/plant layout."
        if rihand_ruled_out else
        "Background-annulus definition sensitivity grew large enough under the full 30-plant "
        f"run ({rihand_swing:.1f}%) to be a plausible contributor to Rihand's error, unlike "
        "the Week 10 2-plant comparison suggested."
    )
    print(f"\n{conclusion}")

    out = {
        "n_plants": len(rows),
        "plants": sorted(rows, key=lambda r: (r["ime_proxy_range_pct_of_default"] is None,
                                               r["ime_proxy_range_pct_of_default"] or 0.0)),
        "rihand_check": {
            "week10_swing_pct": RIHAND_WEEK10_SWING_PCT,
            "this_run_swing_pct": rihand_swing,
            "still_small_vs_talcher_reference": rihand_still_small,
        },
        "predictor_comparison": {
            "n_matched": len(matched),
            "r_swing_pct_abs_log_ratio": r_swing,
            "r_hit_days_abs_log_ratio": r_hit_days,
            "r_snr_abs_log_ratio": r_snr,
            "ranked_by_abs_r": [{"predictor": n, "abs_r": v} for n, v in ranked],
        },
        "conclusion": conclusion,
    }
    out_path = "data/bg_sensitivity_all_plants.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
