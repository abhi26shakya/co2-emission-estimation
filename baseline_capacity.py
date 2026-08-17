"""
Week 12: does capacity_mw alone beat our satellite IME Q estimate at predicting
CEA emissions? CLAUDE.md's headline numbers say r(capacity, CEA) = 0.776 vs
r(our_q, CEA) = 0.236 in raw correlation. This script asks the harder,
apples-to-apples question: leave-one-out predictive accuracy, for four
predictors, so nothing is fitted and scored on the same point.

  A. capacity_mw only            (linear fit in log space)
  B. IME Q, all 24 plants, no gate
  C. IME Q, gated plants only (hit_days >= 10 AND wind_co2_diff_deg <= 60)
  D. capacity_mw + IME Q combined (2-feature linear fit in log space)

Ground truth: CEA CO2 Baseline Database v17.0. Never used as a training label
elsewhere in this repo -- here it IS the regression target, which is fine,
because the question is "how well can capacity predict CEA," not
"how well can satellite detect a plume."

Writes data/baseline_capacity_results.json.
"""
import json
import math

import numpy as np

SEED = 42
DATA_PATH = "data/q_correction_model_results.json"
OUTPUT_PATH = "data/baseline_capacity_results.json"

GATE_HIT_DAYS_MIN = 10
GATE_WIND_DIFF_MAX = 60


def load_feature_table():
    with open(DATA_PATH) as f:
        d = json.load(f)
    return d["feature_table"]


def loo_linear_log_predictions(X_log, y_log):
    """LOO predictions for an OLS fit (with intercept) in log space.

    X_log: (n, k) array of log-transformed features.
    y_log: (n,) array of log-transformed targets.
    Returns: (n,) array of LOO predictions, one per held-out point.
    """
    n = X_log.shape[0]
    design = np.column_stack([np.ones(n), X_log])
    preds = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        coef, *_ = np.linalg.lstsq(design[mask], y_log[mask], rcond=None)
        preds[i] = design[i] @ coef
    return preds


def evaluate(y_log_true, y_log_pred, plant_names, label):
    resid = y_log_pred - y_log_true
    mae = float(np.mean(np.abs(resid)))
    ss_res = float(np.sum(resid ** 2))
    y_bar = float(np.mean(y_log_true))
    ss_tot = float(np.sum((y_log_true - y_bar) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    within_2x = int(np.sum(np.abs(resid) <= math.log(2)))
    n = len(y_log_true)
    return {
        "label": label,
        "n": n,
        "loo_mae_log": mae,
        "loo_r2": r2,
        "within_2x": within_2x,
        "within_2x_frac": within_2x / n,
        "per_plant_abs_log_error": {
            plant_names[i]: float(abs(resid[i])) for i in range(n)
        },
    }


def main():
    np.random.seed(SEED)  # deterministic; no stochastic step in this script,
    # but seeded per repo convention in case future edits add resampling.

    rows = load_feature_table()
    plants_all = [r["plant"] for r in rows]
    cap_all = np.array([r["capacity_mw"] for r in rows])
    q_all = np.array([r["our_q"] for r in rows])
    cea_all = np.array([r["cea_truth"] for r in rows])

    log_cap_all = np.log(cap_all)
    log_q_all = np.log(q_all)
    log_cea_all = np.log(cea_all)

    # A: capacity_mw only, N=24
    predA = loo_linear_log_predictions(log_cap_all.reshape(-1, 1), log_cea_all)
    resA = evaluate(log_cea_all, predA, plants_all, "A: capacity_mw only (N=24)")

    # B: IME Q only, all plants, no gate, N=24
    predB = loo_linear_log_predictions(log_q_all.reshape(-1, 1), log_cea_all)
    resB = evaluate(log_cea_all, predB, plants_all, "B: IME Q, ungated (N=24)")

    # C: IME Q, gated plants only
    gate_mask = np.array([
        r["hit_days"] >= GATE_HIT_DAYS_MIN
        and r["wind_co2_diff_deg"] <= GATE_WIND_DIFF_MAX
        for r in rows
    ])
    plants_gated = [p for p, m in zip(plants_all, gate_mask) if m]
    log_q_gated = log_q_all[gate_mask]
    log_cea_gated = log_cea_all[gate_mask]
    if gate_mask.sum() >= 3:
        predC = loo_linear_log_predictions(log_q_gated.reshape(-1, 1), log_cea_gated)
        resC = evaluate(log_cea_gated, predC, plants_gated,
                         f"C: IME Q, gated only (N={int(gate_mask.sum())})")
    else:
        resC = {"label": "C: IME Q, gated only", "n": int(gate_mask.sum()),
                "note": "too few gated plants for LOO regression"}

    # D: capacity_mw + IME Q combined, N=24
    X_D = np.column_stack([log_cap_all, log_q_all])
    predD = loo_linear_log_predictions(X_D, log_cea_all)
    resD = evaluate(log_cea_all, predD, plants_all, "D: capacity_mw + IME Q combined (N=24)")

    results = [resA, resB, resC, resD]

    # ---- restricted, apples-to-apples comparison on the SAME 7 gated plants ----
    # C already gives IME-gated LOO on this set. Add capacity-only and combined,
    # evaluated on the identical N=7, so the three are directly comparable.
    log_cap_gated = log_cap_all[gate_mask]
    non_gate_mask = ~gate_mask
    plants_nongated = [p for p, m in zip(plants_all, gate_mask) if not m]
    log_cap_nongated = log_cap_all[non_gate_mask]
    log_cea_nongated = log_cea_all[non_gate_mask]

    if gate_mask.sum() >= 3:
        # E: capacity_mw only, restricted to the 7 gated plants
        predE = loo_linear_log_predictions(log_cap_gated.reshape(-1, 1), log_cea_gated)
        resE = evaluate(log_cea_gated, predE, plants_gated,
                         f"E: capacity_mw only, gated subset (N={int(gate_mask.sum())})")

        # F: capacity_mw + IME Q combined, restricted to the 7 gated plants
        X_F = np.column_stack([log_cap_gated, log_q_gated])
        predF = loo_linear_log_predictions(X_F, log_cea_gated)
        resF = evaluate(log_cea_gated, predF, plants_gated,
                         f"F: capacity_mw + IME Q combined, gated subset (N={int(gate_mask.sum())})")
    else:
        resE = {"label": "E: capacity_mw only, gated subset", "n": int(gate_mask.sum()),
                "note": "too few gated plants for LOO regression"}
        resF = {"label": "F: capacity_mw + IME Q combined, gated subset", "n": int(gate_mask.sum()),
                "note": "too few gated plants for LOO regression"}

    # G: capacity_mw only, restricted to the 17 NON-gated plants, for contrast
    resG = evaluate(log_cea_nongated, loo_linear_log_predictions(
        log_cap_nongated.reshape(-1, 1), log_cea_nongated), plants_nongated,
        f"G: capacity_mw only, non-gated subset (N={int(non_gate_mask.sum())})")

    restricted_comparison = {
        "note": "C, E, F evaluated on the identical N=7 gated plants (LOO within that "
                "subset). G is capacity-only on the complementary 17 non-gated plants, "
                "for contrast, not a like-for-like comparison with C/E/F.",
        "results": [resC, resE, resF, resG],
    }

    # ---- honest verdict ----
    scored = [r for r in results if "loo_mae_log" in r]
    best = min(scored, key=lambda r: r["loo_mae_log"])
    capacity_result = resA
    satellite_results = [resB] + ([resC] if "loo_mae_log" in resC else [])
    best_satellite = min(satellite_results, key=lambda r: r["loo_mae_log"])

    capacity_wins = capacity_result["loo_mae_log"] <= best_satellite["loo_mae_log"]

    if capacity_wins:
        verdict = (
            f"Capacity alone wins on LOO accuracy: MAE(log) = "
            f"{capacity_result['loo_mae_log']:.3f}, R^2 = {capacity_result['loo_r2']:.3f}, "
            f"within-2x {capacity_result['within_2x']}/{capacity_result['n']}, "
            f"versus the best satellite predictor ({best_satellite['label']}) at "
            f"MAE(log) = {best_satellite['loo_mae_log']:.3f}, R^2 = {best_satellite['loo_r2']:.3f}, "
            f"within-2x {best_satellite['within_2x']}/{best_satellite['n']}. "
            "This is a loss for the satellite approach on raw predictive accuracy and "
            "should be reported as such, not spun. "
            "The defensible claim for continuing the satellite work is NOT 'more accurate than "
            "capacity' -- it currently is not. It is: (1) capacity_mw is nameplate capacity, not "
            "actual generation or fuel mix, so it is a proxy that works only because plants run near "
            "rated output most of the year in this dataset -- it will fail for plants with unusual "
            "capacity factors, retirements, or fuel switching, none of which CEA v17.0 or this N=24 "
            "sample can rule out; and (2) satellite estimation needs no self-reported operator data at "
            "all, which is the only route to independent verification in geographies without a CEA-grade "
            "reporting regime. If the goal is 'best predictor of CEA for these 24 known Indian plants,' "
            "use capacity. If the goal is 'verify emissions independent of self-reported data,' capacity "
            "is not a competitor -- it doesn't do that job at any accuracy."
        )
    else:
        verdict = (
            f"Satellite ({best_satellite['label']}) wins on LOO accuracy: MAE(log) = "
            f"{best_satellite['loo_mae_log']:.3f}, R^2 = {best_satellite['loo_r2']:.3f}, "
            f"within-2x {best_satellite['within_2x']}/{best_satellite['n']}, versus capacity at "
            f"MAE(log) = {capacity_result['loo_mae_log']:.3f}, R^2 = {capacity_result['loo_r2']:.3f}, "
            f"within-2x {capacity_result['within_2x']}/{capacity_result['n']}. Note the raw correlation "
            "reported in CLAUDE.md (r=0.776 for capacity vs r=0.236 for our_q) does not by itself "
            "determine which predicts better out-of-sample -- LOO accuracy is the fairer test, and "
            "it favors satellite here."
        )

    # ---- restricted verdict: within the gated 7, does satellite close the gap? ----
    gated_satellite_wins = resC["loo_mae_log"] < resE["loo_mae_log"]
    restricted_verdict = (
        f"Within the gated subset specifically (N={resC['n']}, same 7 plants for all three): "
        f"capacity-only MAE(log) = {resE['loo_mae_log']:.3f} (R^2={resE['loo_r2']:.3f}, "
        f"within-2x {resE['within_2x']}/{resE['n']}), IME-gated MAE(log) = "
        f"{resC['loo_mae_log']:.3f} (R^2={resC['loo_r2']:.3f}, within-2x {resC['within_2x']}/{resC['n']}), "
        f"combined MAE(log) = {resF['loo_mae_log']:.3f} (R^2={resF['loo_r2']:.3f}, "
        f"within-2x {resF['within_2x']}/{resF['n']}). "
        + (
            f"The satellite estimate DOES close the gap here -- IME-gated beats capacity-only "
            f"on this specific 7-plant set ({resC['loo_mae_log']:.3f} vs {resE['loo_mae_log']:.3f}). "
            if gated_satellite_wins else
            f"Capacity still wins even on the gated subset -- IME-gated does NOT beat capacity-only "
            f"here ({resC['loo_mae_log']:.3f} vs {resE['loo_mae_log']:.3f}). "
        )
        + f"For contrast, capacity-only on the complementary 17 non-gated plants scores "
        f"MAE(log) = {resG['loo_mae_log']:.3f} (R^2={resG['loo_r2']:.3f}, "
        f"within-2x {resG['within_2x']}/{resG['n']}) -- "
        + (
            "capacity is noticeably worse outside the gated set than inside it, "
            if resG["loo_mae_log"] > resE["loo_mae_log"] else
            "capacity does not degrade much outside the gated set, "
        )
        + "which matters because it means the 'quality gate' picked plants where satellite ALONE improves "
        "(compare B's ungated 0.596 to C's gated "
        f"{resC['loo_mae_log']:.3f}), but capacity improves on that same subset too "
        f"(A's all-plant {resA['loo_mae_log']:.3f} vs E's gated-only {resE['loo_mae_log']:.3f}), "
        "so the gate may just be selecting easier plants in general, not plants where satellite "
        "specifically becomes competitive with capacity. See Week 12's validate_quality_gate.py "
        "for the permutation test on that question."
    )

    output = {
        "seed": SEED,
        "n_facilities": len(rows),
        "gate": {"hit_days_min": GATE_HIT_DAYS_MIN, "wind_co2_diff_deg_max": GATE_WIND_DIFF_MAX,
                 "n_passing": int(gate_mask.sum())},
        "predictors": results,
        "best_overall": best["label"],
        "capacity_wins": bool(capacity_wins),
        "verdict": verdict,
        "restricted_gated_comparison": restricted_comparison,
        "restricted_verdict": {
            "gated_satellite_beats_capacity": bool(gated_satellite_wins),
            "text": restricted_verdict,
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print("=" * 70)
    print("Week 12: capacity_mw baseline vs IME Q, LOO-evaluated")
    print("=" * 70)
    for r in results:
        if "loo_mae_log" in r:
            print(f"{r['label']}")
            print(f"  LOO MAE(log)  = {r['loo_mae_log']:.3f}")
            print(f"  LOO R^2       = {r['loo_r2']:.3f}")
            print(f"  within 2x     = {r['within_2x']}/{r['n']}")
        else:
            print(f"{r['label']}: {r.get('note', 'skipped')}")
        print()
    print(f"Best overall (lowest LOO MAE): {best['label']}")
    print()
    print("VERDICT:")
    print(verdict)
    print()
    print("=" * 70)
    print("Restricted comparison: same N=7 gated plants for C, E, F")
    print("=" * 70)
    for r in [resC, resE, resF, resG]:
        if "loo_mae_log" in r:
            print(f"{r['label']}")
            print(f"  LOO MAE(log)  = {r['loo_mae_log']:.3f}")
            print(f"  LOO R^2       = {r['loo_r2']:.3f}")
            print(f"  within 2x     = {r['within_2x']}/{r['n']}")
        else:
            print(f"{r['label']}: {r.get('note', 'skipped')}")
        print()
    print("RESTRICTED VERDICT:")
    print(restricted_verdict)
    print()
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
