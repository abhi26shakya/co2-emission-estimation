"""
Full evaluation figure set -- NEXT_STEPS.md roadmap step 7 / RESEARCH_PLAN.md
Sec 14 item 7: "predicted-vs-actual, residuals, error distribution,
facility-level comparison, uncertainty calibration plot, ablation table,
feature importance."

Honesty caveats carried over from the source data, restated here since
figures get quoted out of context more easily than prose:
- "predicted-vs-actual" here means Track B's Q estimate vs. Climate TRACE,
  which is benchmark-vs-benchmark, NOT benchmark-vs-ground-truth (no
  CEMS-equivalent measured Indian plant data exists). See
  data/climate_trace_comparison.json's own caveats.
- The A1->A5 Q-correcting ablation ladder from RESEARCH_PLAN.md Sec 9 was
  never built (blocked on ground-truth emissions per Sec 7). The "ablation
  table" panel here is Track A's channel/split ablation history instead
  (NO2-only -> +SO2 -> +VIIRS -> facility-split -> exhaustive LOFO), which
  IS complete data, not a placeholder for the unbuilt Track B ablation.
- "feature importance" here is Pearson correlation from two small
  single-feature-LOO-CV studies (reliability_model.py, N=17;
  lofo_recall_correlates.py, N=20) -- indicative, not SHAP/permutation
  importance from a trained multi-feature model, since RESEARCH_PLAN.md
  Sec 8 explicitly warns against fitting one at this facility count.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

ct = json.load(open("data/climate_trace_comparison.json"))
lofo = json.load(open("data/lofo_track_a_results.json"))
reliab = json.load(open("data/reliability_model_results.json"))
lofo_corr = json.load(open("data/lofo_recall_correlates.json"))

matched = [f for f in ct["facilities"] if "our_q_t_per_year" in f]
matched.sort(key=lambda f: f["climate_trace_co2_t"])
names = [f["plant"] for f in matched]
ours = np.array([f["our_q_t_per_year"] for f in matched])
ct_vals = np.array([f["climate_trace_co2_t"] for f in matched])
ratios = np.array([f["ratio_ours_over_ct"] for f in matched])
bracketed = np.array([f["bracketed_by_our_interval"] for f in matched])

emis = {r["plant"]: r for r in json.load(open("data/emission_estimates.json"))}
lo = np.array([emis[n]["q_t_per_year_low"] for n in names])
hi = np.array([emis[n]["q_t_per_year_high"] for n in names])

# ============================== Figure 1 ==============================
# Predicted-vs-actual + residuals + uncertainty calibration + facility bars
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Track B Evaluation vs. Climate TRACE (benchmark-vs-benchmark, N=17, 2020 vs 2021)",
             fontsize=14, fontweight="bold")

# --- Panel 1: predicted vs actual scatter ---
ax = axs[0, 0]
colors = ["#27ae60" if b else "#c0392b" for b in bracketed]
ax.errorbar(ct_vals, ours, yerr=[ours - lo, hi - ours], fmt="none", ecolor="gray", alpha=0.5, capsize=3)
ax.scatter(ct_vals, ours, c=colors, s=60, zorder=3, edgecolors="black", linewidths=0.5)
lims = [0, max(ct_vals.max(), hi.max()) * 1.05]
ax.plot(lims, lims, "k--", lw=1, label="y = x (perfect agreement)")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Climate TRACE CO2 (t/yr, 2021)")
ax.set_ylabel("Track B Q estimate (t/yr, 2020)")
ax.set_title("Predicted (Track B) vs. Climate TRACE\n(green = bracketed by our uncertainty interval)")
ax.legend(loc="upper left", fontsize=8)
for n, x, y in zip(names, ct_vals, ours):
    ax.annotate(n, (x, y), fontsize=7, alpha=0.7, xytext=(3, 3), textcoords="offset points")

# --- Panel 2: residual ratio distribution ---
ax = axs[0, 1]
order = np.argsort(ratios)
bar_colors = [colors[i] for i in order]
ax.barh(np.arange(len(names)), np.log2(ratios[order]), color=bar_colors)
ax.set_yticks(np.arange(len(names))); ax.set_yticklabels([names[i] for i in order], fontsize=8)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("log2(our Q / Climate TRACE)  [0 = exact match]")
ax.set_title("Residual ratio per facility\n(green = bracketed, red = not)")

# --- Panel 3: uncertainty calibration ---
ax = axs[1, 0]
n_bracketed = int(bracketed.sum()); n_total = len(bracketed)
ax.bar(["Bracketed", "Not bracketed"], [n_bracketed, n_total - n_bracketed],
       color=["#27ae60", "#c0392b"])
ax.set_ylabel("# facilities")
ax.set_title(f"Uncertainty calibration: does our interval bracket Climate TRACE?\n"
             f"{n_bracketed}/{n_total} = {100*n_bracketed/n_total:.0f}% (N=17, up from Week 10's 5/7=71%)")
for i, v in enumerate([n_bracketed, n_total - n_bracketed]):
    ax.text(i, v + 0.1, str(v), ha="center", fontweight="bold")

# --- Panel 4: facility-level side-by-side bars ---
ax = axs[1, 1]
x = np.arange(len(names)); w = 0.38
ax.bar(x - w/2, ct_vals / 1e6, w, label="Climate TRACE (2021)", color="#2980b9")
ax.bar(x + w/2, ours / 1e6, w, yerr=[(ours-lo)/1e6, (hi-ours)/1e6], capsize=3,
       label="Track B (2020)", color="#e67e22")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
ax.set_ylabel("CO2 (Mt/yr)")
ax.set_title("Facility-level comparison")
ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("data/eval_climate_trace_comparison.png", dpi=130, bbox_inches="tight")
print("Saved data/eval_climate_trace_comparison.png")
plt.close(fig)

# ============================== Figure 2 ==============================
# Track A ablation history table + LOFO recall distribution + feature importance
fig, axs = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("Track A Ablation History, Generalization, and Feature Importance",
             fontsize=14, fontweight="bold")

# --- Panel 1: ablation table ---
ax = axs[0, 0]; ax.axis("off")
ax.set_title("Track A ablation / methodology history", fontweight="bold", fontsize=11)
rows = [
    ["Week 3", "NO2-only", "4 fac.", "tile", "77.1%", "—"],
    ["Week 4", "NO2+SO2", "4 fac.", "tile", "79.2%", "—"],
    ["Week 5", "NO2+SO2+VIIRS", "4 fac.", "tile", "79.0%", "—"],
    ["Week 10", "NO2+SO2+VIIRS", "4 fac.", "facility (1 split)", "82.8%", "n/a"],
    ["Week 11", "NO2+SO2+VIIRS", "4 fac.", "facility (1 split)", "—", "8%"],
    ["This session", "NO2+SO2+VIIRS", "20 fac.", "facility (1 split)", "82.8/95.0%", "88%"],
    ["This session", "NO2+SO2+VIIRS", "20 fac.", "LOFO (21 folds)", "—", "47.2%"],
]
col_labels = ["When", "Channels", "Facilities", "Split method", "Accuracy", "Plant recall"]
col_widths = [0.14, 0.20, 0.12, 0.22, 0.16, 0.16]
tbl = ax.table(cellText=rows, colLabels=col_labels, colWidths=col_widths,
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 2.0)
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#0b3d5c"); tbl[0, j].set_text_props(color="white", fontweight="bold")
# highlight the two "this session" rows (indices 6, 7; row 0 is the header)
for i in (6, 7):
    for j in range(len(col_labels)):
        tbl[i, j].set_facecolor("#fdebd0")

# --- Panel 2: LOFO recall distribution ---
ax = axs[0, 1]
lofo_rows = sorted(lofo["per_facility"], key=lambda r: r["recall"])
lofo_names = [r["plant"] for r in lofo_rows]
lofo_recalls = [r["recall"] for r in lofo_rows]
bar_colors = ["#c0392b" if r < 0.5 else "#27ae60" for r in lofo_recalls]
ax.barh(np.arange(len(lofo_names)), lofo_recalls, color=bar_colors)
ax.set_yticks(np.arange(len(lofo_names))); ax.set_yticklabels(lofo_names, fontsize=7)
ax.axvline(lofo["mean_recall"], color="black", ls="--", lw=1,
           label=f"mean={lofo['mean_recall']:.2f}")
ax.set_xlabel("LOFO recall (held-out facility)")
ax.set_title(f"Exhaustive LOFO recall per facility (N={lofo['n_facilities']} folds)")
ax.legend(fontsize=8)

# --- Panel 3: feature importance -- q_rel_std predictors ---
ax = axs[1, 0]
corrs = reliab["correlations_with_q_rel_std"]
feats = sorted(corrs, key=lambda k: abs(corrs[k]))
vals = [corrs[f] for f in feats]
bar_colors = ["#2980b9" if v >= 0 else "#c0392b" for v in vals]
ax.barh(feats, vals, color=bar_colors)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("Pearson r")
ax.set_title(f"What predicts physics uncertainty (q_rel_std)?\n"
             f"(N={reliab['n_facilities']}, best={reliab['best_single_feature']}, "
             f"LOO R2={reliab['loo_cv']['r2']:.2f})")

# --- Panel 4: feature importance -- LOFO recall predictors ---
ax = axs[1, 1]
corrs2 = lofo_corr["correlations_with_lofo_recall"]
feats2 = sorted(corrs2, key=lambda k: abs(corrs2[k]))
vals2 = [corrs2[f] for f in feats2]
bar_colors2 = ["#2980b9" if v >= 0 else "#c0392b" for v in vals2]
ax.barh(feats2, vals2, color=bar_colors2)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("Pearson r")
ax.set_title(f"What predicts LOFO recall (generalization)?\n"
             f"(N={lofo_corr['n_facilities']}, best={lofo_corr['best_single_feature']}, "
             f"LOO R2={lofo_corr['loo_cv']['r2']:.2f})")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("data/eval_track_a_ablation_and_generalization.png", dpi=130, bbox_inches="tight")
print("Saved data/eval_track_a_ablation_and_generalization.png")
plt.close(fig)

print("\n[DONE] 2 evaluation figures saved to data/")
