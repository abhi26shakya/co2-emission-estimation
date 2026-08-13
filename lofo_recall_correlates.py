"""
Follow-up to lofo_track_a.py: what distinguishes the facilities Track A's
detector generalizes to (recall~1.0 when held out) from the ones it
doesn't (recall~0.0)? Same single-feature-LOO-CV approach as
reliability_model.py, applied to a different target: per-facility LOFO
recall instead of physics_gaussian.py's q_rel_std.

Mundra has two units in data/top5_plants.csv (MUNDRA_TPP, MUNDRA_UMPP) and
therefore two rows in lofo_track_a.py's per-facility results, but only one
row in data/plant_results.json / data/candidate_plants.csv (features are
per physical site, not per unit) -- the two units' recalls are averaged
into a single "Mundra" row before joining features, tile-count-weighted.

N=20 (every Track B facility has a LOFO recall, plant_results.json
features, and an activity signal -- unlike reliability_model.py's N=17,
nothing here depends on physics_gaussian.py's OCO-3-derived estimate).
"""
import json
import numpy as np
import pandas as pd

ALIAS_TO_PLANT = {"VINDH_CHAL_STPS": "Vindhyachal", "SASAN_UMPP": "Sasan",
                   "MUNDRA_TPP": "Mundra", "MUNDRA_UMPP": "Mundra",
                   "TIRORA_TPP": "Tirora"}

lofo = json.load(open("data/lofo_track_a_results.json"))["per_facility"]
plant_res = {r["plant"]: r for r in json.load(open("data/plant_results.json"))}
activity = {r["plant"]: r for r in json.load(open("data/activity_signals.json"))}
candidates = pd.read_csv("data/candidate_plants.csv").set_index("name")

# collapse Mundra's two units into one tile-weighted recall
by_plant = {}
for row in lofo:
    name = ALIAS_TO_PLANT.get(row["plant"], row["plant"])
    by_plant.setdefault(name, []).append(row)

recall_by_plant = {}
for name, rows in by_plant.items():
    total_tiles = sum(r["n_tiles"] for r in rows)
    recall_by_plant[name] = sum(r["recall"] * r["n_tiles"] for r in rows) / total_tiles

rows = []
for name, recall in recall_by_plant.items():
    pr = plant_res.get(name)
    act = activity.get(name)
    if pr is None or act is None or name not in candidates.index:
        continue
    rows.append({
        "plant": name,
        "lofo_recall": recall,
        "activity_prob_mean": act["activity_prob_mean"],
        "activity_prob_std": act["activity_prob_std"],
        "wind_co2_diff_deg": pr["wind_co2_diff_deg"],
        "hit_days": pr["hit_days"],
        "n_soundings": pr["soundings"],
        "co2_enhancement_ppm": pr["co2_enhancement_ppm"],
        "bg_std_ppm": pr["bg_std_ppm"],
        "no2_peak_km": pr["no2_peak_km"],
        "capacity_mw": float(candidates.loc[name, "capacity_mw"]),
    })

df = pd.DataFrame(rows).set_index("plant")
print(f"Feature table (N={len(df)}):")
print(df.to_string())

y = df["lofo_recall"].values
candidate_features = ["activity_prob_mean", "activity_prob_std", "wind_co2_diff_deg",
                       "hit_days", "n_soundings", "co2_enhancement_ppm", "bg_std_ppm",
                       "no2_peak_km", "capacity_mw"]

print(f"\n--- Pearson correlation with lofo_recall (N={len(df)}) ---")
corrs = {}
for feat in candidate_features:
    x = df[feat].values
    # pairwise deletion: co2_enhancement_ppm is null for 3 facilities
    # physics_gaussian.py excluded (Mundra, Simhadri, Sipat -- 0 near-plant
    # soundings), which would otherwise propagate to a silent NaN correlation
    valid = ~np.isnan(x)
    n_valid = int(valid.sum())
    r = float(np.corrcoef(x[valid], y[valid])[0, 1])
    corrs[feat] = r
    note = f"  (N={n_valid}, {len(x) - n_valid} missing)" if n_valid < len(x) else ""
    print(f"  {feat:22s} r = {r:+.3f}{note}")

best_feat = max(corrs, key=lambda k: abs(corrs[k]))
print(f"\nStrongest single-feature correlation: {best_feat} (r={corrs[best_feat]:+.3f})")


def loo_linear_fit(x, y):
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
    print(f"  {plant:16s} actual={actual:.3f}  loo_pred={pred:.3f}  err={pred - actual:+.3f}")
print(f"\n  LOO MAE = {mae:.3f}   LOO R^2 = {r2:.3f}   (N={len(df)} -- indicative only)")

result = {
    "n_facilities": len(df),
    "feature_table": df.reset_index().to_dict(orient="records"),
    "correlations_with_lofo_recall": corrs,
    "best_single_feature": best_feat,
    "loo_cv": {
        "feature": best_feat,
        "predictions": {p: float(v) for p, v in zip(df.index, preds)},
        "mae": mae,
        "r2": r2,
    },
    "caveat": ("N=20 facilities, single-feature linear fit only. Explores what predicts "
               "lofo_track_a.py's per-facility recall -- indicative, not a validated model."),
}
json.dump(result, open("data/lofo_recall_correlates.json", "w"), indent=2)
print("\n[SAVED] data/lofo_recall_correlates.json")
