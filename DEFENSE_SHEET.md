# Defense Sheet

Answers sourced directly from `RESEARCH_PAPER.md`. No new claims.

---

**1. "Why should I trust a method that loses to a simple baseline?"**

Because the paper doesn't claim it wins on accuracy — it reports, explicitly, that it doesn't. Nameplate capacity alone beats the IME satellite estimate at predicting CEA ground truth (LOO R²=0.527 vs. −0.152, N=24, §5.2.1b), and combining the two doesn't recover the gap. The stated justification for the satellite approach is not superior accuracy but independent verifiability: capacity only works because these particular plants happen to run near rated output most of the year, and it offers no path to verification in a geography without CEA-grade self-reported data — which is the one thing satellite estimation can offer regardless of today's accuracy (§5.2.1b, §8).

---

**2. "Why didn't you fix the quality gate instead of rejecting it?"**

The gate (`hit_days≥10`, `wind_co2_diff_deg≤60`, N=7) wasn't rejected for looking bad — LOO cross-validation showed it really did improve out-of-fold accuracy (sd(log) 0.55 vs. 1.49 ungated). It was rejected because a permutation test found label-shuffled groups did as well roughly 1 in 4 times — meaning the improvement wasn't distinguishable from selection bias at that sample size (CLAUDE.md, Week 12). "Fixing" it would mean either accepting a result that fails its own significance test, or manufacturing more gated facilities to test on, which wasn't available data. The honest path was to report the effect as real-but-unproven and move on, not to force a fix onto an already-small subset.

---

**3. "What actually explains Rihand's error, if you tested four things and found nothing?"**

We don't know, and here's what's ruled out: overpass density, signal-to-noise, background-annulus sensitivity, and wind-match quality were each tested as both a general N=24 predictor and a Rihand-specific check, and Rihand scores average-or-better than the 30-plant median on all four despite a +134% Climate TRACE error (§5.2.7). None of the four explains more than ~11% of variance in `|log_ratio|` across the dataset — this isn't one signal buried under noise, all four are individually weak. The hunt is explicitly paused, not resolved; the one remaining concrete, checkable hypothesis (plume geometry/plant layout and nearby confounding CO2 sources) is scoped but deliberately not attempted (§8 item 16), alongside a possible FY2020-21-CEA-vs-2020-satellite emissions-year mismatch (§7).

---

**4. "Why build a second physics method if you knew it might fail?"**

Because IME had never been benchmarked against an alternative — it was the default choice, not a validated best choice. The Gaussian cross-sectional method was built to close that gap directly, and its negative result is itself the finding: it fit only 10 of 30 facilities and underperformed IME on CEA ground truth wherever comparable (LOO R² −0.966 vs. −0.111, N=10, §5.2.8). A follow-up closed the one named limitation (annual-mean vs. per-overpass wind direction) and found no improvement either — fit rate churned rather than improved, and LOO R² got worse on the shared subset. The result: IME's status as this project's estimator now rests on direct comparative evidence rather than default-choice justification alone (§5.2.8, §8 item 13).

---

**5. "What would it take to make this method actually work?"**

Per the paper's own limitations and discussion sections, the two biggest documented levers are: (a) better wind data — wind measurement error is the dominant uncertainty term (45–59% relative sd vs. single-digit % for IME sampling), so improving wind quality or matching methodology likely matters more than refining the CO2 mass-balance math itself (§6, finding 2); and (b) more OCO-3 coverage — the regional context map shows only 8.8% of the study region (130/1,470 cells) has enough sounding density to report a value at all (§3.1), and per-overpass wind-direction rotation degraded results by fragmenting an already-sparse sample down to 3–5 usable days per facility (§5.2.8). The paper does not claim a fix exists yet; it identifies the sparsity of the underlying satellite data itself as the likely fundamental constraint, not a solvable engineering gap.

---

**6. "What's the single most defensible claim in this paper?"**

The CEA ground-truth correction result, stated with its full honesty caveats: Track B's raw estimate has an MAE of 1.02 in log-ratio space against real, non-satellite ground truth (CEA's fuel-consumption-based CO2 Baseline Database, matched to all 30 candidate facilities), and a single-feature correction (`hit_days`) reduces this to 0.933 — directionally consistent in all 24 leave-one-out re-fits, but with a bootstrap 95% CI on the improvement that still includes zero ([−0.20, +0.38]), so it is not yet statistically significant. The paper explicitly frames this as its strongest and safest empirical claim precisely because both halves — the real, robust direction and the not-yet-significant magnitude — are reported together (§8, Conclusion).

---

**7. "Why does the paper include a deep-learning model that doesn't work on real data?"**

Because §9's segmentation extension is reported as a confirmed negative result, not offered as a working capability. The U-Net reaches only moderate quality even on its own simulated data (positive-tile median Dice 0.29, collapsing to 0.038 on weak-signal tiles), and across three independently-targeted, mechanistically distinct fixes — matching real OCO-3 coverage density (§9.5), then giving the model an explicit valid-pixel channel built identically in both pipelines (§9.6) — its predictions on real tiles for two facilities consistently track the satellite coverage pattern rather than the true plume, unchanged by either fix. The paper includes it because the investigation itself is the finding: it closes off two specific, plausible causes (density, then a NaN-fill shortcut) rather than leaving the blueprint's original architecture as an untested assumption, and names the one remaining untried path (a fundamentally different label-generation methodology) explicitly instead of implying the door is still open on the paths already tested (§9.6, §1.3).
