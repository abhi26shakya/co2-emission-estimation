# CO₂ Emission Estimation — Project Context

## What this is
Satellite-based CO₂ emission estimation for Indian coal power plants.
Two tracks:
- Track A: 3-channel CNN detector (NO₂/SO₂/VIIRS tiles) — plume vs no-plume
- Track B: physics IME mass-balance Q estimator from OCO-3 XCO₂ + ERA5 wind

Ground truth: CEA CO₂ Baseline Database v17.0 (FY2020-21), 30/30 plants matched.
NOT Climate TRACE — that is satellite-derived and circular. Benchmark only.

## Current state (Week 15 complete)
- 30 plants processed: data/*_soundings.npz
- Q estimates: data/emission_estimates.json
- CEA truth: data/cea_ground_truth_2020_21.json
- Feature table (N=24 with both Q and CEA): data/q_correction_model_results.json
- IME math lives in physics_ime.py, split into estimate_emission_rate()
  (loads a plant's npz) and estimate_emission_rate_from_arrays() (takes
  sounding arrays directly — reuse this, don't reimplement IME, when a
  script needs to rerun the estimate on modified/subsampled soundings).

## The headline numbers (verify, don't trust)
All 24 plants, IME Q vs CEA:
  median bias 0.69x | sd(log ratio) 1.33 | within 2x: 10/24 | r(log) 0.236
  r(capacity_mw, CEA) = 0.776  <-- capacity beats our satellite estimate

Filtered to hit_days >= 10 AND wind_co2_diff_deg <= 60 ("quality gate"):
  N=7 | bias 1.07x | sd(log) 0.55 | within 2x: 6/7
  Week 12 verdict: SELECTION BIAS. LOO cross-validation showed the gated
  subset really does improve out-of-fold (sd 0.595 vs 1.486 ungated), but
  a permutation test failed (label-shuffled groups do as well ~1/4 of the
  time) — not solid enough to build a correction model on.
  baseline_capacity.py (Week 12): capacity_mw alone beats the IME Q
  estimate on every comparison tried (LOO R² 0.527 capacity-only vs
  -0.152 IME-ungated vs -0.242 IME-gated-N=7); combining the two doesn't
  beat capacity alone either.

Week 13 — four experiments, each testing one candidate explanation for
why some facilities (esp. Rihand) fail badly despite good inputs. All
four were tested as both (a) a predictor of |log_ratio| across the N=24
feature table, and (b) a specific check of whether it explains Rihand:

  exp 1 overpass_density_experiment.py: subsampled Vindhyachal/Rihand/
    Sasan (16-19 days) down to 3 days, 200 repeats/step. Vindhyachal and
    Sasan degrade smoothly as days drop (density is real for them), but
    Rihand is already +134% off at its BEST coverage (15 days) — day
    count is NOT its cause. n_days>=6 clears 17/24 plants.
  exp 2 snr_all_plants.py: signal-to-noise (near-plant mean minus
    background, over background std) for all 30. Rihand's SNR=1.27 is
    ABOVE the 30-plant median (0.36) — not a weak-signal problem either.
  exp 3 bg_sensitivity_all_plants.py: IME's swing across 5
    background-annulus definitions, for all 30. Rihand's swing = 4.3%,
    confirmed identical to the original 2-plant Week 10 check, far below
    Talcher's known-bad 20.2% reference — not a background-definition
    problem.
  exp 4 wind_match_quality_all_plants.py: real (unthresholded)
    per-overpass wind-match rate, for all 30 — required a small additive
    fix to physics_ime.py (n_wind_days_raw_matched,
    wind_days_matched_speeds; behavior-preserving, verified). Rihand's
    rate = 25% (4/16 days), 71st percentile — ABOVE median, not a
    wind-matching problem either.

  Predictor strength, ranked by |r| vs |log_ratio| (N=24): wind_match_rate
  -0.328 > hit_days -0.310 (Week 12) > SNR +0.269 > bg swing% +0.161.
  wind_match_rate is nominally the best of the four, but the margin over
  hit_days is small and none of the four explains >11% of the variance.

## The current research question
Week 13 synthesis ("four explanations tested, none sufficient",
WEEK13_LOG.txt): overpass density, signal-to-noise, background
definition, and wind-matching quality have ALL been tested and
REJECTED as Rihand's specific problem — it scores average-or-better on
every one despite a +134% Q error, and none of the four explains more
than ~11% of |log_ratio| variance across the 30-plant dataset. This is
not one signal buried under noise from the others; all four are
individually weak.

The single-cause hunt on Rihand is PAUSED here, by explicit direction —
not resolved. Future work, not yet started: plume geometry/plant layout,
nearby confounding CO2 sources near Rihand's near-plant/background
zones, and a genuine FY2020-21-CEA-vs-2020-satellite emissions-year
mismatch specific to Rihand. See WEEK13_LOG.txt for full per-experiment
detail and the synthesis table, including a self-caught methodology bug
in experiment 1 (an ascending threshold scan was fooled by a noisy lucky
draw at low n_days; fixed to require an unbroken stable run down from
full coverage).

## Week 15: Gaussian cross-section vs IME (negative result)
Tested an alternate Q estimator (physics_gaussian_crosssection.py) against
IME across all 30 plants, per direct request rather than the Rihand hunt
above. Fit succeeded for only 10/30 plants (20 skipped: too few downwind
soundings, or a degenerate curve fit — first run produced sigma in the
tens of thousands of km for 7 plants before a physical cap was added).
On the 10 common plants, IME beats Gaussian cross-section on LOO R² vs
CEA (-0.111 vs -0.966) — worse, not better. Rihand's Gaussian estimate
landed closer to CEA than IME's (42.9M vs 48.3M vs CEA's 20.6M) but is
still ~2x off, not a resolution. CAVEAT: this run used each plant's
ANNUAL-MEAN wind direction (available for all 30) rather than
per-overpass direction (cached for only 18/30) — a coarser choice than
physics_ime.py's own per-overpass wind SPEED matching, and not
disentangled from the method's other weaknesses this week. VERDICT: IME
remains the better-justified physics method for this dataset, now with
direct comparative evidence rather than default choice. See
WEEK15_LOG.txt. Future work, not started: rerun the 18 plants with
per-overpass wind direction available to check whether that specifically
improves fit quality, independent of OCO-3's sparse sampling.

## Hard rules
- Never use Climate TRACE as a training label. Benchmark only.
- Never split train/test at tile level. Split by facility (see train_3channel.py).
- Report negative results. This repo's value is its honesty:
  VIIRS tied exactly, activity signal didn't help, correction model CI crossed zero,
  reliability model LOO R² = -1.22. Keep all of it.
- Machine is a MacBook Air M1. No CUDA. Use device = 'mps' if available else 'cpu'.
- Every new script writes results to data/<name>_results.json and prints a summary.
- Follow the existing WEEKn_LOG.txt convention: state the hypothesis, then
  whether it held, including when it didn't.

## Conventions
- Scripts live at repo root, flat. Tests in tests/.
- Random seed 42 everywhere.
- Don't overwrite existing result JSONs — version them (_v2, or _before).