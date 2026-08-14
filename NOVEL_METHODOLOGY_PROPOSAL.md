# Novel Methodology Proposal: Spatially-Resolved, Ground-Truth-Validated CO2 Emission Estimation for Indian Coal Power Plants

**Status: PROPOSAL — not yet implemented.** This document analyzes the current project, identifies the strongest defensible research gap, and proposes a phased methodology. Per explicit instruction, nothing here is implemented until the direction is confirmed. No results, accuracies, or equations below are invented — equations are standard, cited physics; all "current state" numbers are pulled from this project's actual, already-verified output files.

---

## 1. What already exists (verified, not re-derived)

| Component | What it does | Genuine strength | Genuine limitation |
|---|---|---|---|
| **Track A** (`train_3channel.py`, `lofo_track_a.py`) | 3-channel CNN (NO2+SO2+VIIRS) classifies 64×64 tiles as plant/not-plant | Exhaustive LOFO evaluation (69.1% recall) — rare rigor; most tile-classifier papers report only random-split accuracy | **Binary per-tile only.** No spatial localization within the tile, no plume shape, no intensity gradient |
| **Track B** (`physics_gaussian.py`) | IME mass-balance: `Q = U_eff · IME / L_eff` from OCO-3 XCO2 soundings, near-plant circle vs. background annulus, 3-term uncertainty | Physically grounded, per-overpass wind-conditioned, month-stratified | **Not spatial.** Near/background split is a single scalar circle/annulus test, not a directional plume model. Despite the filename, there is **no actual 2D Gaussian dispersion equation anywhere in this codebase** — this is the single clearest, most honest gap |
| **CEA ground-truth validation** (`q_correction_model.py`) | Validates Track B against India's real, fuel-consumption-based CO2 database; LOO-CV correction (MAE 1.01→0.902, N=17) | **The project's strongest asset.** Neither Climate TRACE nor (as far as this project's literature scan found) prior India-specific work validates against non-proxy, government-reported ground truth | N=17, single feature, one fiscal year — explicitly flagged as indicative, not production-grade |
| Grad-CAM (`gradcam.py`) | Spatial attention map for the 1-channel detector | Already gives *some* within-tile spatial signal | Never extended to the 3-channel model, never used downstream |

**The user's requested capabilities — "spatial area affected," "density/hotspot map," "spatial and temporal variation" — do not exist anywhere in this codebase today.** This is the real gap, not a documentation gap.

## 2. How this differs from Climate TRACE and Deb & Das (2025)

- **Climate TRACE**: facility-level point estimates only; explicitly does not publish per-pixel spatial data (confirmed in this project's own prior research pass, §7.3 of `PROJECT_RESEARCH_DOCUMENTATION.md`). Itself a thermal-proxy ML pipeline, unvalidated for India, per this project's own literature scan.
- **Deb & Das (2025)**, this project's cited baseline: NO2+XCO2 fusion, U-Net-style, ~80% hotspot accuracy on real+simulated data across 71 plants — larger scale, but (per the citation record already in this repo) not benchmarked against real government ground truth, and this project has not verified it publishes calibrated spatial density maps either.
- **Neither source, per what this project has actually verified, combines**: (a) a physically-derived spatial plume field, (b) calibration to a mass-balance-estimated total emission rate, (c) validation against real fuel-consumption ground truth. That combination is the proposed novel contribution.

## 3. Proposed novel contribution (the paper's headline claim)

> **A wind-conditioned Gaussian plume dispersion model, calibrated to a satellite-derived (IME) total emission rate that is itself validated against real government fuel-consumption ground truth, producing a per-plant spatial CO2 density/hotspot map — cross-validated against the actual spatial pattern of OCO-3 soundings, not just point-estimate accuracy.**

This is defensible because every piece of it is either (a) standard, citable physics, (b) already built and verified in this repo, or (c) a stated, bounded extension with an honest validation plan — not an invented capability.

## 4. Mathematical formulation (standard physics, newly applied here)

### 4.1 What's already used (Track B, unchanged)
```
Q = U_eff · IME / L_eff          (IME mass balance, physics_gaussian.py, unchanged)
```

### 4.2 New: ground-level Gaussian plume equation (Pasquill-Gifford)

Standard continuous point-source Gaussian plume, ground-level concentration at downwind distance `x`, crosswind offset `y`:

```
C(x, y, 0) = Q / (π · U · σy(x) · σz(x)) · exp(-y² / (2σy(x)²)) · exp(-H² / (2σz(x)²))
```

- `Q` = emission rate — **taken directly from this project's existing, ground-truth-corrected Track B output** (`data/q_correction_model_results.json`), not re-estimated
- `U` = wind speed at plant location, from ERA5 (already pulled per-overpass in `physics_gaussian.py`)
- `H` = effective stack height (plant-specific; not currently in `candidate_plants.csv` — **a genuine new data requirement**, flagged in §6)
- `σy(x), σz(x)` = Pasquill-Gifford horizontal/vertical dispersion coefficients, standard lookup tables parameterized by downwind distance and atmospheric stability class (A–F)
- Stability class estimated from ERA5 wind speed + boundary-layer height (`blh`) via a standard Golder (1972) or Pasquill-Gifford-Turner parameterization — **both already-available ERA5 fields, no new data source needed**

**Honesty constraint to state explicitly in the paper**: OCO-3's sparse per-sounding footprint (~1.6 km, non-daily revisit) cannot directly validate a fine-grained ground-level plume map. The map is a **physically-derived visualization calibrated to a validated total mass flux**, not itself a directly-observed spatial measurement — validation (§7) is a spatial *consistency* check (does the plume's predicted near-plant excess pattern agree with where OCO-3 actually observed excess XCO2?), not a pixel-accuracy claim. This distinction must be stated plainly in any paper, not blurred.

### 4.3 New: temporal model
Per-month `Q_t` computed by re-running the existing IME method (§4.1) restricted to each calendar month's soundings (tile data + sounding dates already exist for this, no new pulls needed) — a time series, not a single annual scalar.

## 5. Proposed phased plan (implement one phase at a time, verify before continuing)

| Phase | Deliverable | New data needed? | Reuses |
|---|---|---|---|
| **1. Plume module — DONE** | `plume_model.py` (Briggs/Pasquill-Gifford plume equation) + `build_plume_maps.py` (prototype on Rihand/Talcher/Anpara, stability-class × stack-height ablation). 13 new unit tests verify the physics against known analytic properties (monotonic dispersion, linear Q-scaling, wind-dilution, downwind-only support, correct plume orientation under rotation) — all pass, full 35-test suite still green. See §8 for results and one bug caught during verification. | Effective stack height `H` — resolved via documented CPCB/MoEFCC regulatory default (220m), not invented; sensitivity ablation at 150/220/275m built in | `data/emission_estimates.json` (raw `q_t_per_year`), `data/plant_results.json` (`wind_deg`, `lat`/`lon`) |
| **2. Spatial self-consistency validation** | Compare plume-predicted excess pattern against actual OCO-3 sounding excess-by-location for each plant | None — soundings already saved per facility | `data/<Plant>_soundings.npz` |
| **3. Grad-CAM spatial fusion** | Extend Grad-CAM to the 3-channel model; compare CNN attention centroid vs. wind-predicted plume axis as an independent cross-modal agreement metric | None | `gradcam.py`, retrained 3-channel checkpoint |
| **4. Temporal model** | Monthly `Q_t` series per plant, seasonal variation plots | None | existing monthly tiles + sounding dates |
| **5. Enhanced correction model** | Pull 2-3 more CEA fiscal years (already know the URL pattern), refit correction with more data points across years×facilities | CEA historical versions (same public source, already proven accessible) | `pull_cea_ground_truth.py`, `q_correction_model.py` |
| **6. Evaluation suite** | Ablations (stability-class vs. fixed σ; plume-calibrated vs. raw Q), bootstrap CIs, calibration plots, plume-map figures for the paper | None | existing `evaluation_figures.py` pattern |
| **7. API schema** | Documented JSON/GeoJSON export: per-plant `{emission_rate, uncertainty, hotspot_raster_ref, provenance}` for downstream platform integration | None (design task) | — |

## 6. Open questions requiring a decision before Phase 1 starts

- **Stack height `H`**: not in current data. Options: (a) a documented literature-typical default (~150-275m for large coal units) with an explicit sensitivity ablation, (b) source real per-plant values if findable, (c) treat `H` as a free parameter fit against the OCO-3 spatial-consistency check itself. Needs a decision — this materially affects plume shape.
- **Scope of Phase 1**: build for all 21 processed plants, or prototype on 2-3 well-bracketed ones (e.g. Rihand, Talcher — already known strong-signal facilities) first?
- **Grad-CAM 3-channel retrain (Phase 3)**: worth the retraining cost, or defer past the paper's core claim?

## 7. Decisions confirmed (2026-08-14)

- Stack height: documented literature/regulatory default (220m, India's CPCB/MoEFCC coal-plant stack-height norm) + explicit sensitivity ablation. **Implemented.**
- Phase 1 scope: prototype on Rihand, Talcher, Anpara (2-3 strong-signal, well-bracketed facilities) before scaling to all 21. **Implemented.**

## 8. Phase 1 results

Ran `build_plume_maps.py` for all three prototype facilities. Full numeric output in `data/plume_maps/prototype_summary.json` and per-facility grids in `data/plume_maps/<Plant>_default.npz`.

- **Peak ground-level concentration** (default assumptions, class B, H=220m): Rihand 4.3×10⁻³ kg/m³ at 1.6km downwind; Anpara 2.9×10⁻³ kg/m³ at 1.6km; Talcher 4.5×10⁻⁴ kg/m³ at 1.4km (an order of magnitude lower, consistent with Talcher's much smaller raw Track B `Q`).
- **Sensitivity ablation** (stability class A/B/C × stack height 150/220/275m): peak concentration varies by roughly 2-3× across the grid for every facility — this range is reported in the summary JSON alongside the default-assumption number, not hidden, per the "documented default + honest ablation" decision.
- **One real bug caught by verification, before it propagated anywhere**: `data/plant_results.json`'s `wind_deg` field turned out to be the direction the wind blows *toward* (confirmed from `process_plant.py`'s own `"toward {wind_deg} deg"` print statement and its `atan2(u,v)` computation on the raw wind vector), not the standard meteorological "blows *from*" convention `plume_grid()` expects. Checking this before running the prototype (rather than after) avoided every plume silently pointing 180° in the wrong direction — `build_plume_maps.py` now converts explicitly (`wind_from_deg = wind_deg + 180`), documented in its own module docstring.
- **Known limitation carried forward honestly, not new**: Track B's own pre-existing wind/CO2-offset alignment check (`wind_co2_diff_deg`, already in `plant_results.json`) shows only moderate-to-poor alignment for Rihand (109°) and Anpara (80°) — good for Talcher (25°). Since the plume model uses the same wind direction, its predicted plume axis inherits this same limitation for Rihand/Anpara. This is not a new problem introduced by the plume model; it's an existing, already-documented data-quality caveat that now has a second, spatially-visible consequence worth stating plainly in any write-up.

## 9. Phase 2 — spatial self-consistency validation (DONE)

`validate_plume_spatial_consistency.py`, plus 2 more unit tests in `plume_model.py`'s new `concentration_at_locations()` (point evaluation, factored to share `_rotate_to_plume_frame()` with `plume_grid()` so grid and point evaluation cannot drift apart — full suite now 37 tests, all pass).

**Method**: for each prototype facility, project every real OCO-3 sounding within the plume's 30km extent into the same local (east_km, north_km) frame the plume grid uses, compute each sounding's XCO2 excess using the *same* background definition `physics_gaussian.py` already uses (imported, not reimplemented), evaluate the plume model's predicted concentration at each sounding's exact location, and test two things: (a) Pearson correlation between predicted concentration and observed excess, (b) whether soundings inside the plume's predicted downwind sector (±45°) show significantly higher excess than soundings outside it.

**Results** (`data/plume_maps/spatial_consistency_results.json`):

| Facility | N soundings | Pearson r | Sector test (in vs. out, ppm) | z-score |
|---|---|---|---|---|
| Rihand | 2028 | +0.144 | +2.74 vs. +1.25 | 11.43 |
| Talcher | 305 | +0.380 | +0.83 vs. +0.17 | 1.97 |
| Anpara | 1710 | +0.078 | +2.88 vs. +1.03 | 10.60 |

**All three show a positive, mostly statistically significant spatial-consistency signal** — including Rihand and Anpara, whose existing single-centroid wind/CO2 alignment check (`wind_co2_diff_deg` = 109° and 80°, computed independently in `process_plant.py`) suggested poor alignment. This isn't necessarily a contradiction: the sector test uses thousands of individual soundings, not one aggregate centroid bearing dominated by whichever soundings happen to be in the top-20%-CO2 subset — it's a statistically richer test that may be detecting a real but diffuse effect the coarser single-number metric missed.

**Honest caveat, not yet resolved**: the sector test as built does not fully rule out a confound where soundings simply closer to the source read higher regardless of true direction (a pure distance effect masquerading as a directional one), since near-source soundings could disproportionately land in any given 90°-wide sector by chance. A stronger version of this test — comparing against a *random*-direction sector as a null baseline, repeated many times, to establish what correlation/z-score would arise from distance alone — is the natural next robustness check before this spatial-consistency finding is presented as strong evidence in a paper. Flagged here explicitly rather than presented as fully conclusive.

## 10. Phase 2 robustness follow-up (DONE) — the naive result was substantially inflated

`validate_plume_random_sector_baseline.py`: for each facility, repeats Phase 2's exact sector test (same soundings, same excess values, same ±45° half-width) but centered on 2000 random bearings instead of the plume-predicted one, building an empirical null distribution of z-scores. Controls for two confounds at once: (a) a pure distance-from-source effect independent of true direction, and (b) OCO-3 swath/orbital-geometry azimuthal sampling bias — a real, already-documented phenomenon in this project (`diagnose_shrisingajimalwa.py` found exactly this kind of non-uniform sampling for a different reason, seasonally).

**Results** (`data/plume_maps/random_sector_baseline_results.json`):

| Facility | True (wind-predicted) z | Null distribution (mean ± std, from 2000 random sectors) | Percentile of true z | Empirical p (one-sided) | Survives at p<0.05? |
|---|---|---|---|---|---|
| Rihand | 11.43 | +0.52 ± 6.18 | 95.2th | 0.0480 | **Marginally yes** |
| Talcher | 1.97 | −0.22 ± 2.23 | 84.4th | 0.1560 | No |
| Anpara | 10.60 | −1.09 ± 12.54 | 73.8th | 0.2625 | **No** |

**Only 1 of 3 facilities (Rihand) survives this robustness check, and only marginally (p=0.048, right at the conventional threshold).** Talcher and Anpara's apparent "significant" sector effects from Phase 2 turn out to be statistically indistinguishable from what a randomly-oriented sector produces on the same sounding set — the null distributions themselves have large spread (Anpara's null std, 12.54, is even larger than its own observed z-score of 10.60), meaning OCO-3's azimuthal sampling around these plants is substantially non-uniform for reasons unrelated to the plume. **The Phase 2 write-up's headline claim ("all three facilities show a positive, mostly statistically significant spatial-consistency signal") is revised down by this robustness check and should not be quoted without this context.**

**Methodological lesson for the paper, stated as a first-class finding, not a footnote**: a naive in-sector-vs-out-of-sector significance test on satellite sounding data is not reliable on its own — a permutation/randomization-based null baseline is *necessary*, not optional, whenever claiming spatial consistency against this kind of data, because satellite revisit/swath geometry alone can produce apparent directional structure. This mirrors this project's own prior discipline (the Week 11 leakage catch, the LOFO single-split-vs-exhaustive lesson): a headline number without the right control can look far more convincing than it should.

**Revised honest conclusion**: the plume model's spatial predictions are weakly, marginally supported by real sounding data for one of three prototype facilities (Rihand), not confirmed for the other two. This is a genuine, if modest, result — not a failure of the overall approach, since it demonstrates the validation methodology itself works as intended (it correctly identified that 2 of 3 apparent effects were not robust, rather than accepting all three at face value). Scaling to more facilities (Phase 1's original 21-facility scope) would materially strengthen or weaken this conclusion with more statistical power — worth prioritizing over further methodology-building (e.g. Phase 3) until this core spatial claim is on firmer footing.

## 11. Scaled to all eligible facilities (DONE) — the finding is decisive, and it's a negative one

`build_plume_maps.py`, `validate_plume_spatial_consistency.py`, and `validate_plume_random_sector_baseline.py` all now run over every facility with the required inputs (`eligible_facilities()`: has a wind direction, a Track B estimate, and a soundings file) — **18 facilities** for the plume maps themselves, **14** with enough soundings for the random-sector robustness test (4 smaller facilities — ShriSingajiMalwa, TalwandiSabo, Tirora, and one more — skipped for having too few soundings on one side of even the true wind-predicted sector, an honest skip, not a forced result). Caught and fixed one real bug during this scale-up: the robustness script crashed (`TypeError`) on facilities where the true sector itself was too small, instead of skipping gracefully — fixed before the run was trusted.

**Full-scale result**: only **1 of 14 facilities (Rihand, p=0.045, still only marginal) survives the random-sector null-baseline test at p<0.05.** The other 13 facilities' apparent plume-direction spatial-consistency effects are statistically indistinguishable from what a randomly-oriented sector produces on the same sounding data. This is not a fluke of the small 3-facility prototype — the full-scale run confirms and sharpens it.

**Honest conclusion, stated plainly**: as currently built — using the existing annual/overpass-mean wind direction and a fixed default stability class, with no per-overpass wind-to-sounding date matching for the spatial test itself — **the plume model's predicted spatial orientation does not reliably align with real OCO-3 spatial patterns for the large majority of these facilities.** This is a genuine, scientifically meaningful negative result, not a failure of the validation methodology (which is working exactly as intended: it is correctly distinguishing real signal from geometry-driven noise, rather than accepting every plausible-looking effect at face value). It should be reported as such in any paper, not minimized.

**What this does and doesn't mean for the overall contribution**:
- It does **not** undermine Track B's scalar emission-rate estimate or its CEA ground-truth validation (§1's strongest asset) — those don't depend on plume direction being spatially correct.
- It **does** mean the spatial hotspot map, as currently calibrated, should be presented as a physics-consistent *visualization* calibrated to a validated total mass flux (exactly the honesty constraint stated from the start in §4.2) — not as a validated directional prediction. The one weak exception (Rihand) is too marginal to claim as a validated case either.
- The most likely fixable cause, not yet tested: the wind direction used here is an annual or per-overpass-*aggregated* mean, not matched to the specific date of each individual sounding being tested against it. A day-matched wind-to-sounding comparison (available in principle, since soundings already carry a `day` field and ERA5 wind can be pulled per-day) is the natural next methodological improvement before concluding the plume-direction hypothesis itself is wrong, rather than just poorly matched in time.

Full results: `data/plume_maps/spatial_consistency_results.json` (18 facilities) and `data/plume_maps/random_sector_baseline_results.json` (14 testable facilities).

## 12. Per-overpass wind-to-sounding date matching (DONE) — the proposed fix does not help, and clarifies the earlier result

`fetch_daily_wind_direction.py` pulled genuinely new data (per-day ERA5 wind direction, not just speed, for all 18 facilities, 555 days each, 2019–2020) since nothing in this codebase previously computed wind direction at daily granularity — only a single annual mean (`plant_results.json`'s `wind_deg`) existed. `validate_plume_day_matched.py` then tested each sounding against its *own overpass day's* actual wind direction instead of one facility-wide average, with a matched robustness check (a rotated-null baseline that preserves real day-to-day directional variability while testing whether the untouched alignment beats an arbitrary rigid rotation of the same pattern — a stronger, more appropriate null than the earlier uniform-random-bearing one, since it can't be satisfied by day-to-day variability alone).

**Result: 0 of 11 testable facilities show a day-matched effect distinguishable from the rotated-null baseline at p<0.05** — worse than the non-day-matched result (1/14). Most tellingly: **Rihand, the one facility that marginally survived the earlier (non-day-matched) test, now shows a *negative*, clearly non-significant result** (z=−4.89, only the 12.2th percentile of its own null distribution) when tested against its actual day-specific wind. This strongly suggests Rihand's earlier marginal "survival" was a coincidence of using the aggregate annual wind direction, not evidence of a real day-level directional signal — the day-matched test, being the more rigorous and more direct test, supersedes it.

**Honest conclusion, revised and sharpened again**: the per-overpass day-matching fix, the most likely candidate identified in §11 for the earlier negative result, does not rescue the spatial-consistency hypothesis — if anything it clarifies that the plume model's predicted orientation shows **no facility with a robust, reproducible spatial alignment to real OCO-3 data**, under any wind-matching granularity tested so far. Plausible remaining explanations, none yet tested: (a) ERA5's `ECMWF/ERA5/DAILY` collection is a daily-*mean* composite, not the wind at the specific time of each OCO-3 overpass (which can occur at any local time given the ISS's precessing orbit) — still a coarse temporal match, not a true one; (b) real atmospheric transport at these facilities may not be well-approximated by a single-source Gaussian plume at all (multiple stacks, complex terrain, boundary-layer mixing over tens of km); (c) OCO-3's per-sounding footprint and sparse revisit may simply be too coarse to resolve plume-scale spatial structure regardless of wind accuracy.

**Recommendation, given two independent negative results now**: further iteration on wind-matching precision is unlikely to be productive without first testing (a) — true per-overpass-time (not daily-mean) wind, which would require sub-daily ERA5 data (`ECMWF/ERA5_LAND/HOURLY` or similar) and each sounding's exact UTC overpass time (derivable from the OCO-3 `sounding_id`, not currently extracted) — a real but nontrivial next step, not attempted in this pass. Absent that, **the spatial hotspot map should be presented in any paper strictly as a physics-consistent visualization calibrated to a validated total mass flux, with an honest, quantified statement that its predicted orientation has not been validated against real spatial data at any wind-matching granularity tested.** The CEA ground-truth-validated Q-correction (§1) remains the project's strongest, safest empirical claim and should carry the paper's headline result.

Full results: `data/plume_maps/day_matched_results.json`; per-day wind cache: `data/daily_wind/`.

## 12.5 Strengthening the CEA correction model (Phase 5, DONE)

`strengthen_q_correction_model.py` adds the statistical rigor `q_correction_model.py`'s original single-feature result (MAE 1.012→0.902, N=17) didn't yet have — a paper cannot claim this improvement is meaningful without answering: is it distinguishable from noise, is it driven by one facility, and does a second feature add anything.

**1. Bootstrap CI on the improvement (1000 resamples, resampling facilities with replacement)**: mean improvement +0.111, **95% CI = [−0.287, +0.509] — includes zero.** At N=17, the improvement is **not statistically distinguishable from noise**, and this must be stated plainly rather than only quoting the point estimate.

**2. Leave-one-facility-out sensitivity**: the improvement stays **positive in all 17/17** leave-one-out re-fits — no single facility is driving the result; it is a small but directionally consistent effect across every subset, just underpowered at this N rather than an outlier artifact. Reporting both facts together (not statistically significant, but not an artifact either) is the honest characterization.

**3. Does a second feature help?** Yes, and by more than the first: `bg_std_ppm` + `n_soundings` reduces LOO MAE further to **0.846** (from 0.902); `+hit_days` (0.859) and `+activity_prob_mean` (0.894) also help; `+wind_co2_diff_deg` and `+capacity_mw` do not. This is a real lead — but per this project's own established caution against multi-feature models at small N (`reliability_model.py`'s docstring, `RESEARCH_PLAN.md` §8), it is reported here as an **indicative direction for more data to confirm, not a new recommended production model.**

**Revised honest framing for the paper**: the CEA-ground-truth correction is a real, directionally consistent, physically interpretable effect (background noise predicts estimation error, which makes physical sense) — but at N=17 it should be presented with its confidence interval, not just its point estimate, and the 2-feature lead should be flagged as needing more facilities/years before being adopted, not claimed as a finished model. This is more defensible for a paper than the original unqualified "11% improvement" framing, and is consistent with how this project has handled every other small-N result (§7.4's original N=7 reliability-model caution, LOFO's exhaustive-vs-single-split lesson).

Full results: `data/q_correction_model_strengthened_results.json`.

## 13. Next steps (not started)

- **True per-overpass-time wind matching** (not daily-mean) — the one remaining untested hypothesis before concluding the plume-direction claim is unsupported by available data, not just by the two matching approaches tried so far. Requires sub-daily ERA5/reanalysis wind and extracting each sounding's exact overpass time from its `sounding_id`.
- **Phase 3** (Grad-CAM spatial fusion) and beyond, per §5's table — deprioritized; building spatial-fusion methodology on top of an unvalidated spatial claim isn't productive yet.
- **Paper framing decision, now firmer**: lead with the CEA ground-truth Q-correction (§1). Present the plume/hotspot work as a rigorous, honestly-negative methodological exploration — a real contribution in its own right (a permutation-test discipline for validating satellite-derived spatial claims, demonstrated to catch two rounds of overstated naive results), not the positive spatial-accuracy result originally hoped for.
