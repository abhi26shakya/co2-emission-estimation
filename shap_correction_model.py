"""
Adds real feature-attribution (SHAP) on top of q_correction_model.py /
strengthen_q_correction_model.py's existing single-feature Pearson
correlation + 2-feature LOO-MAE comparison -- neither of those is a proper
attribution method, and RESEARCH_PLAN.md/NOVEL_METHODOLOGY_PROPOSAL.md
never got further than "correlation with the target."

Rebuilds the identical feature table strengthen_q_correction_model.py
builds (same duplicated-not-imported pattern this project already uses,
since q_correction_model.py runs its whole pipeline at module import time
with no main() guard).

Model choice, deliberately conservative at this N: a single OLS linear
model (sklearn.linear_model.LinearRegression) fit on standardized features
over all N facilities (not leave-one-out here -- SHAP explains a fixed
fitted model's own predictions, not a cross-validated generalization
estimate; the existing LOO-CV numbers in q_correction_model.py /
strengthen_q_correction_model.py remain the project's generalization
claim, this script is a complementary attribution layer on top of the
full-data fit), explained with shap.LinearExplainer. A tree ensemble was
deliberately NOT used -- RESEARCH_PLAN.md Sec 8's caution against complex
multi-feature models at N=17-18 applies here too; SHAP on a linear model
answers "how does each feature move this simple model's prediction" while
staying within that established discipline, rather than introducing a
new black-box model to explain.

Cross-check performed explicitly: bg_std_ppm was the strongest single-
feature Pearson correlate in q_correction_model.py (r=+0.58) and the best
individual feature in strengthen_q_correction_model.py's two-feature test.
If SHAP's global ranking disagrees, that discrepancy is reported as a
finding, not smoothed over, consistent with this project's practice
throughout NOVEL_METHODOLOGY_PROPOSAL.md.
"""
import json

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

FEATURES = ["activity_prob_mean", "activity_prob_std", "wind_co2_diff_deg",
            "hit_days", "n_soundings", "bg_std_ppm", "no2_peak_km", "capacity_mw"]

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
print(f"N={n} facilities (same feature table as strengthen_q_correction_model.py)\n")

X = df[FEATURES].values
y = df["log_ratio"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression().fit(X_scaled, y)
r2_full_data = model.score(X_scaled, y)
print(f"Full-data (not LOO -- this is an attribution fit, not a "
      f"generalization claim) linear R^2 = {r2_full_data:.3f}")
print("Coefficients (standardized features):")
for feat, coef in zip(FEATURES, model.coef_):
    print(f"  {feat:22s} {coef:+.4f}")

explainer = shap.LinearExplainer(model, X_scaled)
shap_values = explainer.shap_values(X_scaled)  # (N, n_features)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
ranking = sorted(zip(FEATURES, mean_abs_shap), key=lambda t: -t[1])
print("\n=== Global SHAP feature ranking (mean |SHAP value|) ===")
for feat, val in ranking:
    print(f"  {feat:22s} {val:.4f}")

# --- Cross-check against q_correction_model.py's Pearson-correlation ranking ---
corr_ranking = sorted(
    ((feat, abs(float(np.corrcoef(df[feat].values, y)[0, 1]))) for feat in FEATURES),
    key=lambda t: -t[1])
shap_top = ranking[0][0]
corr_top = corr_ranking[0][0]
agrees = shap_top == corr_top
print(f"\nSHAP top feature: {shap_top}  |  Pearson-correlation top feature: {corr_top}  "
      f"({'AGREE' if agrees else 'DISAGREE -- reported honestly, not smoothed over'})")

# --- Per-facility SHAP values (for the data export / platform integration) ---
per_facility = {}
for i, plant in enumerate(df.index):
    facility_shap = {feat: float(shap_values[i, j]) for j, feat in enumerate(FEATURES)}
    top3 = sorted(facility_shap.items(), key=lambda kv: -abs(kv[1]))[:3]
    per_facility[plant] = {
        "log_ratio": float(y[i]),
        "predicted_log_ratio": float(model.predict(X_scaled[i:i + 1])[0]),
        "base_value": float(explainer.expected_value),
        "shap_values": facility_shap,
        "top_3_features": [{"feature": f, "shap_value": v} for f, v in top3],
    }

# --- Summary figure: global bar chart + one example waterfall ---
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
order = np.argsort(mean_abs_shap)
axes[0].barh(np.array(FEATURES)[order], mean_abs_shap[order], color="#4C72B0")
axes[0].set_xlabel("mean |SHAP value|  (contribution to predicted log-ratio error)")
axes[0].set_title(f"Global SHAP feature importance (N={n}, linear model)")

example_plant = df["log_ratio"].abs().idxmax()  # facility with the largest error, most informative example
ex_idx = list(df.index).index(example_plant)
ex_shap = shap_values[ex_idx]
ex_order = np.argsort(ex_shap)
colors = ["#C44E52" if v > 0 else "#4C72B0" for v in ex_shap[ex_order]]
axes[1].barh(np.array(FEATURES)[ex_order], ex_shap[ex_order], color=colors)
axes[1].axvline(0, color="black", linewidth=0.8)
axes[1].set_xlabel("SHAP value (this facility's log-ratio prediction)")
axes[1].set_title(f"Per-facility example: {example_plant}\n(largest |log-ratio| error)")

plt.tight_layout()
plt.savefig("data/shap_correction_model.png", dpi=120, bbox_inches="tight")
print("\nSaved data/shap_correction_model.png")

result = {
    "n_facilities": n,
    "features": FEATURES,
    "model": "sklearn LinearRegression on standardized features, full-data fit "
             "(not LOO -- attribution only, see module docstring)",
    "full_data_r2": float(r2_full_data),
    "global_shap_ranking": [{"feature": f, "mean_abs_shap": float(v)} for f, v in ranking],
    "pearson_correlation_ranking": [{"feature": f, "abs_corr": v} for f, v in corr_ranking],
    "shap_vs_pearson_top_feature_agree": bool(agrees),
    "per_facility": per_facility,
    "caveat": (
        "SHAP explains a full-data linear fit's own predictions, not a "
        "cross-validated generalization estimate -- the project's LOO-CV MAE "
        "figures in q_correction_model.py / strengthen_q_correction_model.py "
        "remain the generalization claim. N=17-18, deliberately a linear "
        "(not tree-ensemble) model per RESEARCH_PLAN.md Sec 8's caution "
        "against complex multi-feature models at this facility count."
    ),
}
json.dump(result, open("data/shap_correction_model_results.json", "w"), indent=2)
print("Saved data/shap_correction_model_results.json")
