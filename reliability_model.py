"""
Small, interpretable reliability model for Track B's IME emission-rate
estimates -- the first step toward RESEARCH_PLAN.md's roadmap step 4
("correction model"), scoped down from a Q-correcting regression (which
would need ground-truth emissions we don't have yet -- Climate TRACE isn't
pulled until roadmap step 6, and RESEARCH_PLAN.md Sec 7 says it must not be
used as a training label anyway) to a question we CAN answer with data
already on hand: does the activity signal (extract_activity_signal.py) or
the wind/CO2 alignment check (process_plant.py) predict how large physics'
own self-reported relative uncertainty (q_rel_std, from physics_gaussian.py)
turns out to be?

This is not circular: q_rel_std = hypot(wind_rel_std, ime_rel_std) is
derived purely from ERA5 wind-speed variance and IME bootstrap sampling
noise. wind_co2_diff_deg (NO2-plume-offset vs. wind-direction consistency)
and the CNN activity signal are independent of that computation, so asking
whether they predict q_rel_std is a real, testable question -- e.g. it
would validate whether wind/CO2 misalignment (already flagged qualitatively
for ChandrapurCoal/Rihand) is quantitatively informative beyond what
wind_rel_std alone already captures.

N=17 as of the 20-facility expansion (only plants with a physics_gaussian.py
estimate; Mundra, Sipat, and Simhadri are excluded, all three skipped for
having 0 near-plant soundings after the near/bg split -- a genuine OCO-3
coverage gap, not something re-running fixes). Originally N=7 when this
script was first written. Still a single-feature linear fit evaluated by
leave-one-out CV, reported with explicit sample-size caveats -- per the
project's priority order (scientific validity > novelty > ...), this script
deliberately does NOT fit a multi-feature gradient-boosted-trees model,
which per RESEARCH_PLAN.md Sec 8 would be "underpowered and uninterpretable"
even at N=17.
"""
import json
import numpy as np
import pandas as pd

emis = {r["plant"]: r for r in json.load(open("data/emission_estimates.json"))}
plant_res = {r["plant"]: r for r in json.load(open("data/plant_results.json"))}
activity = {r["plant"]: r for r in json.load(open("data/activity_signals.json"))}
candidates = pd.read_csv("data/candidate_plants.csv").set_index("name")

rows = []
for name, e in emis.items():
    pr = plant_res[name]
    act = activity.get(name)
    if act is None or name not in candidates.index:
        continue
    rows.append({
        "plant": name,
        "q_rel_std": e["q_rel_std"],
        "activity_prob_mean": act["activity_prob_mean"],
        "activity_prob_std": act["activity_prob_std"],
        "wind_co2_diff_deg": pr["wind_co2_diff_deg"],
        "hit_days": pr["hit_days"],
        "n_soundings": pr["soundings"],
        "capacity_mw": float(candidates.loc[name, "capacity_mw"]),
    })

df = pd.DataFrame(rows).set_index("plant")
print(f"Feature table (N={len(df)}):")
print(df.to_string())

y = df["q_rel_std"].values
candidate_features = ["activity_prob_mean", "activity_prob_std", "wind_co2_diff_deg",
                       "hit_days", "n_soundings", "capacity_mw"]

print("\n--- Pearson correlation with q_rel_std (N={}) ---".format(len(df)))
corrs = {}
for feat in candidate_features:
    x = df[feat].values
    r = float(np.corrcoef(x, y)[0, 1])
    corrs[feat] = r
    print(f"  {feat:22s} r = {r:+.3f}")

best_feat = max(corrs, key=lambda k: abs(corrs[k]))
print(f"\nStrongest single-feature correlation: {best_feat} (r={corrs[best_feat]:+.3f})")


def loo_linear_fit(x, y):
    """Leave-one-out CV for a single-feature linear fit (y = a*x + b).
    N=7 is too small to hold out a separate test set, so LOO is the only
    honest option here; still very noisy with this few points."""
    n = len(x)
    preds = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        a, b = np.polyfit(x[mask], y[mask], 1)
        preds[i] = a * x[i] + b
    mae = float(np.mean(np.abs(preds - y)))
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return preds, mae, r2


x_best = df[best_feat].values
preds, mae, r2 = loo_linear_fit(x_best, y)

print(f"\n--- Leave-one-out CV, single feature ({best_feat}) ---")
for plant, actual, pred in zip(df.index, y, preds):
    print(f"  {plant:16s} actual={actual:.3f}  loo_pred={pred:.3f}  "
          f"err={pred - actual:+.3f}")
print(f"\n  LOO MAE = {mae:.3f}   LOO R^2 = {r2:.3f}   (N={len(df)} -- "
      f"treat as indicative only, not a validated model)")

result = {
    "n_facilities": len(df),
    "feature_table": df.reset_index().to_dict(orient="records"),
    "correlations_with_q_rel_std": corrs,
    "best_single_feature": best_feat,
    "loo_cv": {
        "feature": best_feat,
        "predictions": {p: float(v) for p, v in zip(df.index, preds)},
        "mae": mae,
        "r2": r2,
    },
    "caveat": ("N=7 facilities. This is a reliability/uncertainty-prediction "
               "exercise, not a Q-correcting model -- no ground-truth emissions "
               "are available yet to fit or validate a true correction against. "
               "Results are indicative, not a validated predictive model."),
}
json.dump(result, open("data/reliability_model_results.json", "w"), indent=2)
print("\n[SAVED] data/reliability_model_results.json")
