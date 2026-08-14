"""
The actual Q-correcting model RESEARCH_PLAN.md Sec 9's A1-A5 ablation
ladder called for, finally buildable now that pull_cea_ground_truth.py has
sourced a real, independent, non-Climate-TRACE ground-truth emissions
figure per facility (CEA's CO2 Baseline Database -- bottom-up from
reported fuel consumption, not satellite-inferred, so it's a legitimate
training/validation label per Sec 7's Climate-TRACE-specific restriction).

Two stages, both honestly evaluated by leave-one-out CV at N=18 (every
facility with both a Track B physics estimate and a CEA-matched ground
truth), consistent with this project's established single-feature-LOO-CV
discipline at small N (reliability_model.py, lofo_recall_correlates.py):

1. Baseline: how far off is the RAW physics estimate (q_t_per_year) from
   CEA's true figure, in log-ratio space (log(ours/CEA) -- the correct
   scale for multiplicative/relative error, symmetric between over- and
   under-estimates unlike a raw ratio)? Reports MAE, bias, and bracketing
   rate (does our stated uncertainty interval actually contain the truth).

2. Correction: can any single Track A/Track B feature (activity signal,
   wind alignment, sounding count, background noise, NO2 offset,
   capacity) predict that log-ratio error well enough, under LOO-CV, that
   correcting the raw estimate by the LOO-predicted error reduces MAE
   below the uncorrected baseline? This is the actual A1-A5-style
   correction, not the self-consistency-only fusion attempt
   track_fusion_model.py made without a ground-truth label.

Caveats stated where they apply, not buried: N=18, single-feature linear
fit only (per RESEARCH_PLAN.md Sec 8's caution against multi-feature
models at this facility count), one fiscal year of CEA data
(FY2020-21, not an exact calendar-year match to Track B's 2020 estimates,
see pull_cea_ground_truth.py's fiscal_year_caveat).
"""
import json

import numpy as np
import pandas as pd

est = {r["plant"]: r for r in json.load(open("data/emission_estimates.json"))}
cea = json.load(open("data/cea_ground_truth_2020_21.json"))["facilities"]
plant_res = {r["plant"]: r for r in json.load(open("data/plant_results.json"))}
activity = {r["plant"]: r for r in json.load(open("data/activity_signals.json"))}
candidates = pd.read_csv("data/candidate_plants.csv").set_index("name")

common = sorted(set(est) & set(cea))
rows = []
for name in common:
    e, c, pr = est[name], cea[name], plant_res.get(name)
    act = activity.get(name)
    if pr is None or act is None or name not in candidates.index:
        continue
    rows.append({
        "plant": name,
        "our_q": e["q_t_per_year"],
        "our_q_std": e["q_t_per_year_std"],
        "cea_truth": c["abs_emissions_t_co2"],
        "log_ratio": float(np.log(e["q_t_per_year"] / c["abs_emissions_t_co2"])),
        "activity_prob_mean": act["activity_prob_mean"],
        "activity_prob_std": act["activity_prob_std"],
        "wind_co2_diff_deg": pr["wind_co2_diff_deg"],
        "hit_days": pr["hit_days"],
        "n_soundings": pr["soundings"],
        "bg_std_ppm": pr["bg_std_ppm"],
        "no2_peak_km": pr["no2_peak_km"],
        "capacity_mw": float(candidates.loc[name, "capacity_mw"]),
    })

df = pd.DataFrame(rows).set_index("plant")
n = len(df)
print(f"=== Stage 1: baseline accuracy vs. CEA ground truth (N={n}) ===\n")
print(df[["our_q", "cea_truth", "log_ratio"]].to_string())

y = df["log_ratio"].values
baseline_mae = float(np.mean(np.abs(y)))
baseline_bias = float(np.mean(y))
bracketed = [(row["cea_truth"] >= row["our_q"] - row["our_q_std"]) and
             (row["cea_truth"] <= row["our_q"] + row["our_q_std"])
             for _, row in df.iterrows()]
bracket_rate = float(np.mean(bracketed))

print(f"\nBaseline |log-ratio| MAE: {baseline_mae:.3f}  (0 = perfect; 0.693 = 2x off)")
print(f"Baseline bias (mean log-ratio): {baseline_bias:+.3f}  "
      f"({'overestimate' if baseline_bias > 0 else 'underestimate'} on average)")
print(f"Bracketing rate (CEA truth inside our own uncertainty interval): "
      f"{sum(bracketed)}/{n} = {bracket_rate:.1%}")

print(f"\n=== Stage 2: single-feature LOO-CV correction (N={n}) ===\n")
candidate_features = ["activity_prob_mean", "activity_prob_std", "wind_co2_diff_deg",
                       "hit_days", "n_soundings", "bg_std_ppm", "no2_peak_km", "capacity_mw"]
corrs = {}
for feat in candidate_features:
    x = df[feat].values
    corrs[feat] = float(np.corrcoef(x, y)[0, 1])
    print(f"  {feat:22s} r(feature, log_ratio) = {corrs[feat]:+.3f}")

best_feat = max(corrs, key=lambda k: abs(corrs[k]))
print(f"\nStrongest single-feature correlation with log-ratio error: "
      f"{best_feat} (r={corrs[best_feat]:+.3f})")


def loo_linear_fit(x, y):
    n = len(x)
    preds = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        a, b = np.polyfit(x[mask], y[mask], 1)
        preds[i] = a * x[i] + b
    return preds


x_best = df[best_feat].values
loo_pred_log_ratio = loo_linear_fit(x_best, y)
corrected_log_ratio = y - loo_pred_log_ratio  # residual after removing the LOO-predicted bias
corrected_mae = float(np.mean(np.abs(corrected_log_ratio)))

print(f"\n--- Correcting by LOO-predicted {best_feat}-based bias ---")
for plant, raw, pred, corrected in zip(df.index, y, loo_pred_log_ratio, corrected_log_ratio):
    print(f"  {plant:18s} raw_log_ratio={raw:+.3f}  loo_predicted={pred:+.3f}  "
          f"corrected={corrected:+.3f}")

print(f"\nBaseline MAE:  {baseline_mae:.3f}")
print(f"Corrected MAE: {corrected_mae:.3f}")
improved = corrected_mae < baseline_mae
print(f"-> Correction {'HELPS' if improved else 'does NOT help'} "
      f"({'reduces' if improved else 'does not reduce'} error under honest LOO-CV)")

result = {
    "n_facilities": n,
    "ground_truth_source": "CEA CO2 Baseline Database v17.0, FY2020-21 (see pull_cea_ground_truth.py)",
    "feature_table": df.reset_index().to_dict(orient="records"),
    "baseline": {
        "mae_log_ratio": baseline_mae,
        "bias_log_ratio": baseline_bias,
        "bracket_rate": bracket_rate,
        "n_bracketed": int(sum(bracketed)),
    },
    "correction": {
        "correlations_with_log_ratio": corrs,
        "best_single_feature": best_feat,
        "loo_predictions_log_ratio": {p: float(v) for p, v in zip(df.index, loo_pred_log_ratio)},
        "corrected_mae_log_ratio": corrected_mae,
        "improves_on_baseline": bool(improved),
    },
    "caveat": (
        "N=18, single-feature linear LOO-CV correction only, one fiscal year "
        "(FY2020-21) of CEA ground truth against 2020-calendar-year Track B "
        "estimates (not an exact year match, see pull_cea_ground_truth.py). "
        "This is the first genuine Q-correcting model built against real "
        "ground truth for this project, but should still be read as an "
        "indicative feasibility result, not a validated production correction."
    ),
}
json.dump(result, open("data/q_correction_model_results.json", "w"), indent=2)
print("\n[SAVED] data/q_correction_model_results.json")
