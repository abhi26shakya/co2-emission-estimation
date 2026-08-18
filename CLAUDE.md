# CO₂ Emission Estimation — Project Context

## What this is
Satellite-based CO₂ emission estimation for Indian coal power plants.
Two tracks:
- Track A: 3-channel CNN detector (NO₂/SO₂/VIIRS tiles) — plume vs no-plume
- Track B: physics IME mass-balance Q estimator from OCO-3 XCO₂ + ERA5 wind

Ground truth: CEA CO₂ Baseline Database v17.0 (FY2020-21), 30/30 plants matched.
NOT Climate TRACE — that is satellite-derived and circular. Benchmark only.

## Current state (Week 13 complete)
- 30 plants processed: data/*_soundings.npz
- Q estimates: data/emission_estimates.json
- CEA truth: data/cea_ground_truth_2020_21.json
- Feature table (N=24 with both Q and CEA): data/q_correction_model_results.json
- IME math lives in physics_gaussian.py, split into estimate_emission_rate()
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

Week 13 overpass-density experiment (overpass_density_experiment.py):
  subsampled Vindhyachal/Rihand/Sasan (16-19 overpass days each) down to
  as few as 3 days, 200 repeats/step, to test whether OCO-3 revisit
  frequency explains the plant-to-plant spread.
  Vindhyachal and Sasan: error and spread degrade smoothly and
  monotonically as days drop — density is a real factor for them.
  Rihand: already +134% off at its BEST available coverage (15 usable
  days) — more days never fixed it, ruling out day count as its cause.
  VERDICT: density matters but does not fully explain the failure
  pattern — some facilities (Rihand) have a bad underlying signal
  independent of coverage. A conservative n_days>=6 threshold clears
  17/24 plants, a looser filter than the Week 11 quality gate. See
  WEEK13_LOG.txt for the full breakdown, including a self-caught bug in
  the first threshold definition (ascending scan was fooled by a noisy
  lucky draw at low n_days; fixed to require an unbroken stable run down
  from full coverage).

## The current research question
Density alone doesn't explain the spread — what does, for facilities like
Rihand that fail even at their best achievable coverage? Candidates not
yet tested: background-annulus definition, wind-day matching quality,
plume geometry/plant layout, real emission variability year-to-year vs.
the FY2020-21 CEA baseline.

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