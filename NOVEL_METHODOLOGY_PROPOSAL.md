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

## 9. Next: Phase 2 (not started)

Phase 2 (spatial self-consistency validation against actual OCO-3 sounding locations) is the natural next step and was explicitly out of scope for this turn — it requires its own verification pass before being trusted, consistent with the step-by-step approach. Recommend checking in before starting it, given Phase 1's own wind-convention bug is a reminder that each phase needs its own independent verification, not just physics-level unit tests.
