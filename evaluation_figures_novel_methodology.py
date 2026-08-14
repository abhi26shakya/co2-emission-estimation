"""
Evaluation figures for NOVEL_METHODOLOGY_PROPOSAL.md's plume/hotspot model
and strengthened CEA correction (Phases 1-5) -- distinct from the existing
evaluation_figures.py, which covers Track A's ablation history and the
Climate TRACE comparison. All panels here plot already-computed results
from this session's own saved JSON/NPZ files; nothing is recomputed or
invented.

Honesty caveats carried into the figures themselves, not just this
docstring, since figures get quoted out of context more easily than prose:
- The plume hotspot maps (Fig 1) are physics-consistent VISUALIZATIONS
  calibrated to a validated total mass flux, not validated spatial
  predictions -- Fig 2/3 show directly why that distinction matters.
- Fig 2/3 show a mostly-negative validation result: only 1 of 14
  facilities' spatial alignment survives a random-sector null baseline,
  and day-matching (Fig 3) makes this worse, not better.
- Fig 4's correction-model improvement is real and directionally
  consistent (Fig 4c) but not statistically significant at N=17 (Fig 4b's
  CI includes zero) -- both facts are shown together, not just the point
  estimate.
"""
import json

import numpy as np
import matplotlib.pyplot as plt

FACILITIES_FOR_MAPS = ["Rihand", "Talcher", "Anpara"]


def fig1_plume_hotspot_maps():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, name in zip(axes, FACILITIES_FOR_MAPS):
        d = np.load(f"data/plume_maps/{name}_default.npz")
        grid, ex, nx = d["grid"], d["east_km"], d["north_km"]
        im = ax.pcolormesh(ex, nx, np.log10(grid + 1e-12), shading="auto", cmap="inferno")
        ax.set_title(f"{name}\nQ={float(d['Q_t_per_year']):,.0f} t/yr, "
                      f"wind from {float(d['wind_from_deg']):.0f}deg", fontsize=10)
        ax.set_xlabel("East (km)"); ax.set_ylabel("North (km)")
        ax.plot(0, 0, marker="*", color="cyan", markersize=14, markeredgecolor="black")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, label="log10(conc, kg/m^3)", shrink=0.8)
    fig.suptitle("Plume hotspot maps (Briggs/Pasquill-Gifford, class B, H=220m) --\n"
                  "physics-consistent visualizations calibrated to Track B's raw Q; "
                  "NOT validated spatial predictions (see Fig 2/3)", fontsize=11)
    fig.tight_layout()
    fig.savefig("data/eval_plume_hotspot_maps.png", dpi=150)
    plt.close(fig)
    print("[SAVED] data/eval_plume_hotspot_maps.png")


def fig2_spatial_robustness():
    d = json.load(open("data/plume_maps/random_sector_baseline_results.json"))
    results = {k: v for k, v in d["results"].items() if not v.get("skipped")}
    names = sorted(results, key=lambda k: results[k]["true_z"])
    true_z = [results[n]["true_z"] for n in names]
    null_lo = [results[n]["null_mean_z"] - results[n]["null_std_z"] for n in names]
    null_hi = [results[n]["null_mean_z"] + results[n]["null_std_z"] for n in names]
    survives = [results[n]["distinguishable_from_random_geometry"] for n in names]

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(len(names))
    ax.hlines(y_pos, null_lo, null_hi, color="lightgray", linewidth=8, label="null distribution (+/-1 std)")
    colors = ["#2ca02c" if s else "#7f7f7f" for s in survives]
    ax.scatter(true_z, y_pos, c=colors, s=80, zorder=3, label="true (wind-predicted) sector z")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos); ax.set_yticklabels(names)
    ax.set_xlabel("Sector z-score (soundings in predicted downwind sector vs. outside)")
    ax.set_title("Spatial-consistency robustness check: only Rihand (green) survives\n"
                  "the random-sector null baseline at p<0.05 -- 13/14 facilities' apparent\n"
                  "effects are indistinguishable from random-direction sampling geometry")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig("data/eval_spatial_robustness.png", dpi=150)
    plt.close(fig)
    print("[SAVED] data/eval_spatial_robustness.png")


def fig3_day_matching_comparison():
    agg = json.load(open("data/plume_maps/random_sector_baseline_results.json"))["results"]
    day = json.load(open("data/plume_maps/day_matched_results.json"))["results"]
    common = sorted(n for n in agg if n in day and not agg[n].get("skipped") and not day[n].get("skipped"))

    agg_z = [agg[n]["true_z"] for n in common]
    day_z = [day[n]["true_z"] for n in common]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(common))
    width = 0.35
    ax.bar(x - width / 2, agg_z, width, label="Annual/overpass-averaged wind", color="#1f77b4")
    ax.bar(x + width / 2, day_z, width, label="Per-overpass day-matched wind", color="#ff7f0e")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(common, rotation=45, ha="right")
    ax.set_ylabel("Sector z-score")
    ax.set_title("Day-matching does not rescue the spatial-consistency result --\n"
                  "note Rihand flips from +11.4 (aggregate wind) to -4.9 (day-matched)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("data/eval_day_matching_comparison.png", dpi=150)
    plt.close(fig)
    print("[SAVED] data/eval_day_matching_comparison.png")


def fig4_cea_correction_model():
    q = json.load(open("data/q_correction_model_results.json"))
    s = json.load(open("data/q_correction_model_strengthened_results.json"))
    ft = {r["plant"]: r for r in q["feature_table"]}
    names = sorted(ft)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) predicted vs actual, log-log
    ax = axes[0, 0]
    our_q = np.array([ft[n]["our_q"] for n in names])
    cea = np.array([ft[n]["cea_truth"] for n in names])
    ax.scatter(cea, our_q, s=50)
    lims = [min(cea.min(), our_q.min()) * 0.5, max(cea.max(), our_q.max()) * 2]
    ax.plot(lims, lims, "k--", linewidth=1, label="perfect agreement")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("CEA ground truth (t CO2/yr)"); ax.set_ylabel("Track B raw estimate (t CO2/yr)")
    ax.set_title(f"(a) Predicted vs. actual, N={len(names)}")
    ax.legend(fontsize=8)

    # (b) bootstrap CI on improvement
    ax = axes[0, 1]
    boot = s["bootstrap"]
    ax.errorbar([0], [boot["mean_improvement"]],
                yerr=[[boot["mean_improvement"] - boot["ci_95_low"]],
                      [boot["ci_95_high"] - boot["mean_improvement"]]],
                fmt="o", markersize=10, capsize=8, color="#2ca02c")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlim(-1, 1); ax.set_xticks([])
    ax.set_ylabel("MAE improvement (baseline - corrected)")
    ax.set_title(f"(b) Bootstrap 95% CI: [{boot['ci_95_low']:+.3f}, {boot['ci_95_high']:+.3f}]\n"
                  f"includes zero -- not significant at N={s['n_facilities']}")

    # (c) LOFO sensitivity
    ax = axes[1, 0]
    lofo = s["lofo_sensitivity"]["per_facility_improvement"]
    lofo_names = sorted(lofo, key=lambda k: lofo[k])
    vals = [lofo[n] for n in lofo_names]
    ax.barh(lofo_names, vals, color=["#2ca02c" if v > 0 else "#d62728" for v in vals])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Improvement when this facility is excluded")
    ax.set_title(f"(c) LOFO sensitivity: {s['lofo_sensitivity']['n_still_positive']}/"
                  f"{s['lofo_sensitivity']['n_total']} stay positive -- robust, not outlier-driven")
    ax.tick_params(axis="y", labelsize=7)

    # (d) two-feature comparison
    ax = axes[1, 1]
    two_feat = s["two_feature_test"]["results_by_second_feature"]
    feat_names = sorted(two_feat, key=lambda k: two_feat[k])
    mae_vals = [two_feat[n] for n in feat_names]
    baseline_line = s["one_feature_corrected_mae"]
    ax.barh(feat_names, mae_vals, color="#1f77b4")
    ax.axvline(baseline_line, color="red", linestyle="--", linewidth=1,
               label=f"1-feature MAE ({baseline_line:.3f})")
    ax.set_xlabel(f"LOO MAE with bg_std_ppm + [feature]")
    ax.set_title("(d) 2-feature LOO-CV (indicative, not a recommended\nfinal model at this N)")
    ax.legend(fontsize=8)

    fig.suptitle("CEA ground-truth correction model: point estimate, uncertainty, and robustness", fontsize=13)
    fig.tight_layout()
    fig.savefig("data/eval_cea_correction_model.png", dpi=150)
    plt.close(fig)
    print("[SAVED] data/eval_cea_correction_model.png")


def fig5_temporal_q_seasonal():
    d = json.load(open("data/temporal_q_model_results.json"))
    results = {k: v for k, v in d["results"].items() if not v.get("skipped") and v["n_months_usable"] >= 5}
    names = sorted(results, key=lambda k: -results[k]["n_months_usable"])[:6]

    fig, ax = plt.subplots(figsize=(10, 6))
    for name in names:
        series = results[name]["monthly_series"]
        months = sorted(int(m) for m in series)
        q_norm = np.array([series[str(m)]["q_t_per_year"] for m in months])
        q_norm = q_norm / q_norm.max()  # normalize to each facility's own peak month -- shape only, see caveat
        ax.plot(months, q_norm, marker="o", label=name)
    ax.set_xlabel("Month"); ax.set_ylabel("Monthly Q, normalized to facility's own peak month")
    ax.set_xticks(range(1, 13))
    ax.set_title("Seasonal shape of monthly CO2 signal (relative within each facility only --\n"
                  "absolute monthly Q is NOT comparable to the annual estimate, see "
                  "temporal_q_model.py docstring)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig("data/eval_temporal_q_seasonal.png", dpi=150)
    plt.close(fig)
    print("[SAVED] data/eval_temporal_q_seasonal.png")


if __name__ == "__main__":
    fig1_plume_hotspot_maps()
    fig2_spatial_robustness()
    fig3_day_matching_comparison()
    fig4_cea_correction_model()
    fig5_temporal_q_seasonal()
