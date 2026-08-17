"""
Phase 5 of NOVEL_METHODOLOGY_PROPOSAL.md: strengthens q_correction_model.py's
single-feature LOO-CV correction (baseline MAE 1.012 -> corrected 0.902,
N=17) with the statistical rigor a paper needs but the original script
didn't yet have: is the improvement itself statistically distinguishable
from noise, is it driven by one influential facility, and does a second
feature add anything beyond bg_std_ppm.

Rebuilds the same feature table q_correction_model.py builds (duplicated,
not imported -- q_correction_model.py runs at module level with no
main() guard, matching this project's reliability_model.py/track_fusion_
model.py style, so importing it would re-execute its whole pipeline).

Three additions, each answered honestly regardless of outcome:

1. Bootstrap CI on the MAE improvement (baseline - corrected), resampling
   facilities with replacement (1000 resamples) -- does the confidence
   interval exclude zero, i.e. is the improvement distinguishable from
   sampling noise at this facility count?
2. Leave-one-facility-out sensitivity: recompute the improvement N times,
   each time excluding one facility -- does any single facility's removal
   flip the sign of the conclusion (correction helps vs. doesn't)?
3. Two-feature LOO-CV: does adding a second feature to bg_std_ppm (the
   original single best feature) reduce MAE further, or is one feature
   already all the signal available at this facility count -- same
   question track_fusion_model.py asked for a different target, answered
   here for the real ground-truth-corrected target.
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
y = df["log_ratio"].values
# Picked the same way q_correction_model.py picks its best single feature --
# strongest |Pearson r| with log_ratio -- rather than a hardcoded name, since
# which feature wins can change as the facility set grows (it did: bg_std_ppm
# at N=17 -> hit_days at N=24, see RESEARCH_PAPER.md's revision log).
_candidate_features = ["activity_prob_mean", "activity_prob_std", "wind_co2_diff_deg",
                        "hit_days", "n_soundings", "bg_std_ppm", "no2_peak_km", "capacity_mw"]
_corrs = {f: float(np.corrcoef(df[f].values, y)[0, 1]) for f in _candidate_features}
BEST_FEATURE = max(_corrs, key=lambda k: abs(_corrs[k]))
print(f"N={n} facilities (matches q_correction_model.py's current run)")
print(f"Best single feature (by |r| with log_ratio): {BEST_FEATURE} (r={_corrs[BEST_FEATURE]:+.3f})\n")


def loo_linear_fit(X, y):
    """LOO-CV for an OLS fit, X can be 1 or more columns."""
    X = np.atleast_2d(X.T).T if X.ndim == 1 else X
    n = len(y)
    Xb = np.column_stack([X, np.ones(n)])
    preds = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        coef, *_ = np.linalg.lstsq(Xb[mask], y[mask], rcond=None)
        preds[i] = Xb[i] @ coef
    mae = float(np.mean(np.abs(preds - y)))
    return preds, mae


x_best = df[BEST_FEATURE].values
baseline_mae = float(np.mean(np.abs(y)))
loo_preds_1feat, corrected_mae_1feat = loo_linear_fit(x_best, y)
corrected_1feat = y - loo_preds_1feat

print(f"=== Baseline replication ===\nbaseline MAE = {baseline_mae:.4f}   "
      f"1-feature ({BEST_FEATURE}) corrected MAE = {corrected_mae_1feat:.4f}\n")

# --- 1. Bootstrap CI on the improvement ---
print("=== 1. Bootstrap CI on MAE improvement (1000 resamples) ===")
rng = np.random.default_rng(0)
abs_baseline = np.abs(y)
abs_corrected = np.abs(corrected_1feat)
improvements = []
for _ in range(1000):
    idx = rng.integers(0, n, size=n)
    improvements.append(float(abs_baseline[idx].mean() - abs_corrected[idx].mean()))
improvements = np.array(improvements)
ci_low, ci_high = np.percentile(improvements, [2.5, 97.5])
print(f"  mean improvement = {improvements.mean():+.4f}  95% CI = [{ci_low:+.4f}, {ci_high:+.4f}]")
excludes_zero = ci_low > 0
print(f"  {'CI excludes zero -- improvement is distinguishable from noise' if excludes_zero else 'CI includes zero -- improvement is NOT statistically distinguishable from noise at this N'}")

# --- 2. Leave-one-facility-out sensitivity on the improvement itself ---
print("\n=== 2. Leave-one-facility-out sensitivity (does one facility drive the result?) ===")
lofo_improvements = {}
for held_out in df.index:
    mask = df.index != held_out
    y_sub = y[mask]
    x_sub = x_best[mask]
    _, corrected_mae_sub = loo_linear_fit(x_sub, y_sub)
    baseline_mae_sub = float(np.mean(np.abs(y_sub)))
    lofo_improvements[held_out] = baseline_mae_sub - corrected_mae_sub
    print(f"  excluding {held_out:18s}: improvement = {lofo_improvements[held_out]:+.4f}")
still_positive = sum(1 for v in lofo_improvements.values() if v > 0)
print(f"  improvement stays positive in {still_positive}/{n} leave-one-out re-fits "
      f"{'(robust, not driven by a single facility)' if still_positive == n else '(NOT robust -- at least one facility flips the conclusion)'}")

# --- 3. Two-feature LOO-CV ---
print(f"\n=== 3. Does a second feature beyond {BEST_FEATURE} help? ===")
candidate_features = [f for f in _candidate_features if f != BEST_FEATURE]
two_feat_results = {}
for feat in candidate_features:
    X2 = df[[BEST_FEATURE, feat]].values
    _, mae2 = loo_linear_fit(X2, y)
    two_feat_results[feat] = mae2
    print(f"  {BEST_FEATURE} + {feat:22s} LOO MAE = {mae2:.4f}  "
          f"({'better' if mae2 < corrected_mae_1feat else 'worse'} than 1-feature's {corrected_mae_1feat:.4f})")
best_second_feat = min(two_feat_results, key=two_feat_results.get)
two_feature_helps = two_feat_results[best_second_feat] < corrected_mae_1feat

result = {
    "n_facilities": n, "baseline_mae": baseline_mae,
    "one_feature_corrected_mae": corrected_mae_1feat, "best_feature": BEST_FEATURE,
    "bootstrap": {"n_resamples": 1000, "mean_improvement": float(improvements.mean()),
                  "ci_95_low": float(ci_low), "ci_95_high": float(ci_high),
                  "distinguishable_from_noise": bool(excludes_zero)},
    "lofo_sensitivity": {"per_facility_improvement": lofo_improvements,
                          "n_still_positive": still_positive, "n_total": n,
                          "robust_to_single_facility_removal": bool(still_positive == n)},
    "two_feature_test": {"results_by_second_feature": two_feat_results,
                          "best_second_feature": best_second_feat,
                          "best_two_feature_mae": two_feat_results[best_second_feat],
                          "two_feature_helps": bool(two_feature_helps)},
    "caveat": (f"N={n}, same facility set and ground-truth source as q_correction_model.py. "
               "Bootstrap CI resamples facilities, not the underlying physics -- it quantifies "
               "sampling uncertainty in the improvement estimate given this specific facility "
               "set, not a claim the model would generalize to unseen facilities (that would "
               "need its own LOFO-style test, not attempted here)."),
}
json.dump(result, open("data/q_correction_model_strengthened_results.json", "w"), indent=2)
print("\n[SAVED] data/q_correction_model_strengthened_results.json")
