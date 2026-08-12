# Research Plan — Satellite/ML CO2 Emission Estimation (post-Week-7)

Status: research/design stage. No code changed as part of this document. Priority order per
project brief: **scientific validity > novelty > reproducibility > generalization > accuracy >
compute efficiency.**

---

## 1. Current project architecture (as of Week 7)

Two tracks, currently **not fused**:

**Track A — plant detector** (`export_monthly.py`/`export_hard_negatives.py`/`export_so2.py`/
`export_viirs.py` → `build_2channel.py`/`build_3channel.py` → `train_detector.py`/
`train_2channel.py`/`train_3channel.py`): small CNN, binary plant-vs-not-plant classifier.
Inputs: Sentinel-5P NO2 (`COPERNICUS/S5P/OFFL/L3_NO2`), SO2 (`L3_SO2`), VIIRS VNP14A1 MaxFRP,
stacked as 1/2/3-channel tiles, monthly, 2019–2020, 5 plants + 5 rural + 20 hard negatives
(`data/hard_negatives.csv`). Current best: **81.2% hard-negative-only accuracy** (Week 7),
68% plant recall, 92% negative recall. Dataset is small (~720 tiles for the 3-channel variant)
and the train/test split is **not facility-level** — the same physical location's tiles across
different months can land in both train and test, which is a leakage risk relative to how the
literature validates this kind of model (see §3).

**Track B — CO2 emission-rate estimate** (`process_plant.py` → `co2_enhancement.py` /
`co2_no2_colocation.py` / `wind_check.py` → `physics_gaussian.py`): physics-only Integrated
Mass Enhancement (IME) mass balance, `Q = U_eff · IME / L_eff`, on OCO-3 XCO2 soundings
(`OCO3_L2_Lite_FP` v11r via `earthaccess`), cross-checked against NO2 co-location and
ERA5 wind direction. 4 plants processed (`data/plant_results.json`,
`data/emission_estimates.json`): Vindhyachal 44.6 Mt/yr and Sasan 37.2 Mt/yr are physically
plausible; Tirora (3.2 Mt/yr) looks too low (thin OCO-3 coverage — 5 hit-days/671 soundings);
Mundra is skipped (57 soundings). **No ML component. Uses a single annual-mean wind speed and
a fixed approximate footprint per sounding — a fragility the project's own WEEK6_LOG.txt
already flagged after finding and fixing a wind-averaging bug. No uncertainty output.**

Both tracks are honestly documented (weekly logs read like real lab notebooks, with negative
and null results kept in, e.g. Mundra being skipped rather than papered over) — this is a good
foundation to build rigor on top of, not something to throw out.

## 2. Current baseline, restated for comparison purposes

| Track | Metric | Value |
|---|---|---|
| A (detector) | hard-negative accuracy | 81.2% (Week 7) |
| A (detector) | train/test split | random, tile-level (not facility-level) |
| B (IME) | facilities with a plausible estimate | 2 / 4 |
| B (IME) | uncertainty quantification | none |
| B (IME) | wind resolution | annual mean, not per-overpass |
| Cited baseline | Deb & Das 2025, arXiv:2502.02083 | U-Net, real OCO-2/3 + TROPOMI NO2, 71 plants, ~80%+ hotspot accuracy |

The cited baseline already does more (scale: 71 plants; fusion: NO2+XCO2 in one learned model)
than either of this repo's tracks individually. That reframes the task: the opportunity is not
"beat Deb & Das on accuracy" but to add dimensions their work and the rest of the literature do
not cover — see §5.

## 3. Literature findings (condensed; full agent reports available on request)

**Already done — do not claim as novel:**
- NO2 + OCO-2/3 XCO2 fusion via a learned model (U-Net) at multi-plant scale — Deb & Das 2025,
  71 power plants.
- CNN/U-Net direct regression from an XCO2 image + wind → Mt/yr, beating cross-sectional flux
  (~20–25% vs ~40% median error) — GMD 2024 (Copernicus), but **simulated SMARTCARB data only,
  Europe only, never run on real satellite imagery** — this is the paper's own stated
  limitation.
- Facility-level (leave-one-facility-out) train/test splits as the scientifically correct
  validation strategy for this problem class — GMD 2024 explicitly withholds a plant's
  inventory from its own model during training. Established methodology, not itself novel, but
  **never applied to real multi-sensor data.**
- VIIRS thermal anomalies as a *standalone, tabular* quantitative CO2/carbon-emissions
  predictor — Li et al. 2021 (GRL, VIIRS FRP vs. China industrial inventory, R²=0.86–0.90);
  MDPI Remote Sensing 14(12):2901 (boosting regression tree, VIIRS heat sources + auxiliary
  features, index of agreement 0.83). VIIRS-as-activity-proxy is real and well-supported, but
  **claiming "we discovered VIIRS correlates with industrial activity" would not be novel.**
- Wind-speed error as the dominant term in point-source flux uncertainty (~85% of total flux
  uncertainty in one CH4 study; IME estimates can flip >2x depending on wind-vector-averaging
  choice) — multiple 2024–2025 papers. This *confirms* rather than discovers the fragility
  already visible in this repo's own WEEK6_LOG.txt bug writeup.
- NO2-to-NOx / emission-ratio inversion as an alternative CO2-from-NO2 pathway (active
  2024–2026 ACP literature) — a citable alternative/complement to IME, not a gap to fill.
- Climate TRACE's own methodology is itself a satellite-thermal-proxy → generation →
  emission-factor ML pipeline for the power sector, trained mainly on US/EU CEMS-equivalent
  ground truth, not on Indian plants.

**Genuinely open, and reachable from this repo's existing code:**
1. **Per-overpass, wind-conditioned IME (or a learned correction to it) validated on real OCO-3
   soundings**, as opposed to annual-mean wind and as opposed to GMD 2024's simulation-only
   setting. Nobody has done the "real data, non-European coal plants" version of GMD 2024's
   idea.
2. **Calibrated per-plant uncertainty** (`CO2 = X ± σ`) for OCO-3-IME-based coal-plant
   estimates — propagating wind-variance and IME sampling noise into an explicit interval. No
   paper reviewed reports this for real point-source CO2 (only cites wind error as a term, not
   a calibrated output).
3. **Fusing this repo's own two tracks**: using the activity-classifier's learned signal
   (NO2+SO2+VIIRS combustion-activity confidence/embedding) as a physics-informed covariate or
   correction to the IME emission-rate estimate, rather than as an isolated classifier output.
   This specific combination does not appear in any paper found.
4. **Facility-level generalization + source-wise ablation on real multi-sensor data** — GMD
   2024 does facility-level splits on simulated single-sensor (XCO2+wind) data; this repo's own
   weekly logs already do an informal, sequential form of ablation (NO2 → +SO2 → +VIIRS →
   +more hard negatives) for the *classifier* — extending that same discipline, formally, to
   the *emission-rate* estimate, with proper held-out facilities, is unclaimed.
5. **Climate TRACE as an independent benchmark specifically for Indian coal plants** — most
   published comparisons are US/EU/China-centric; no Indian-focused independent cross-check of
   OCO-3-IME vs. Climate TRACE was found.

## 4. Research-gap table

| Paper/System | Data | Method | Target | Resolution | Validation | Limitation | Our difference |
|---|---|---|---|---|---|---|---|
| Deb & Das 2025 (arXiv:2502.02083) | TROPOMI NO2 + real OCO-2/3 XCO2, 71 plants | U-Net, NO2→continuous XCO2 map → emission rate | Emission rate | TROPOMI ~3.5×5.5 km | Not detailed in abstract | No stated uncertainty output; no VIIRS; split strategy unclear | Add per-overpass wind + uncertainty + VIIRS-activity fusion + facility-level ablation |
| GMD 2024 (Copernicus, SMARTCARB CNN) | Simulated XCO2+NO2+wind, CO2M-noise, Germany | Small CNN, image→Mt/yr | Emission rate | 2 km sim | Facility-level holdout | **Simulation only**, Europe only, no real satellite noise, no VIIRS | Real OCO-3 data, Indian plants, add activity fusion |
| Li et al. 2021 (GRL) | VIIRS FRP vs. China inventory | Correlation/regression | Sector-level carbon | 375 m VIIRS | Regional inventory comparison | Not facility-specific; not fused with atmospheric gases | Facility-specific, fused with NO2/SO2/OCO-3 |
| MDPI RS 14(12):2901 | VIIRS heat sources + auxiliary tabular features | Boosting regression tree | Industrial carbon emissions | 375 m VIIRS | Inventory comparison (IoA 0.83) | Tabular-only, no imagery/atmospheric fusion, no per-plant OCO CO2 | Fuse with real OCO-3 IME as physics-informed correction |
| Climate TRACE (power sector) | Thermal imagery → ML → generation → emission factor | Bottom-up ML + activity inference | Facility-level CO2 | Product-level, not published per-pixel | Ground truth: mostly US/EU CEMS-equivalent | Unvalidated on Indian plants; itself a thermal-proxy black box | Use only as independent benchmark, never as training label |
| This repo (Track A, pre-plan) | NO2+SO2+VIIRS tiles, 5 plants + 20 hard negatives | CNN classifier | Plant vs. not-plant (binary) | GEE-native | Random split (not facility-level) | Small N; classification only, no emission rate; leakage risk | Facility-level split; extend to regression/correction signal |
| This repo (Track B, pre-plan) | OCO-3 XCO2 soundings, 4 plants | IME mass balance | Emission rate (Mt/yr) | OCO-3 ~1.3×2.6 km footprint | 2/4 plausible vs. literature range | Annual-mean wind, fixed footprint, no uncertainty, no ML | Per-overpass wind, uncertainty, activity-conditioned correction |

## 5. Candidate novel contributions (ranked)

Ranked by scientific novelty, feasibility on a local machine with existing Earthdata/GEE
access, and conference-paper suitability.

1. **[Recommended] Physics-informed, uncertainty-aware, activity-conditioned emission-rate
   estimation with facility-level validation and Climate TRACE benchmarking** — fuses this
   repo's two existing tracks, fixes the wind/uncertainty weak point the project already
   documented, validates properly, benchmarks honestly. High novelty (nothing in the
   literature does this specific combination), high feasibility (all components already exist
   in some form in this repo), high reproducibility, directly extends the project's own
   documented history rather than discarding it.
2. **Real-data, non-European replication + extension of GMD 2024's CNN-regression idea**
   (image→Mt/yr, but on real OCO-3 over India instead of SMARTCARB simulation). Good novelty
   (closes GMD 2024's own stated gap) but data-hungry — a CNN trained end-to-end needs more
   than ~10–20 real facility-overpass samples to be credible, which this project's OCO-3
   coverage may not support without a large facility-expansion effort. Medium feasibility.
3. **Multimodal fusion at Deb & Das 2025's scale** (NO2+SO2+VIIRS+OCO-3+ERA5, U-Net-style,
   many plants) — largest potential accuracy gain, but overlaps heavily with the cited
   baseline and needs facility counts (dozens+) this project doesn't currently have, and large
   downloads the brief explicitly says to avoid. Lower near-term feasibility.
4. **Cross-region generalization (train India, test elsewhere or vice versa)** — scientifically
   interesting but needs a second region's facility set and ground-truth reference, which adds
   scope without a corresponding literature gap as clearly identified as #1. Medium novelty,
   lower feasibility right now.
5. **XAI-focused contribution** (Grad-CAM/SHAP for the fused model) — genuinely thin in the
   literature for this exact application, but is a modest addition on its own, better folded
   into #1 as a component (the repo already has `gradcam.py`) than positioned as the primary
   contribution.

## 6. Recommended primary direction

**"Activity-Conditioned, Uncertainty-Aware CO2 Emission-Rate Estimation from Real OCO-3
Soundings, with Facility-Level Generalization and Independent Benchmarking"**

Concretely, in priority order:

1. **Fix Track B's physics baseline**: replace the annual-mean ERA5 wind speed with
   per-overpass wind (matched to each OCO-3 sounding's date), and propagate wind-variance +
   IME sampling noise into a calibrated interval, producing `Q = X ± σ` per plant instead of a
   point estimate. This directly answers the literature's identified dominant error source and
   the repo's own documented fragility (WEEK6_LOG.txt).
2. **Fuse Track A into Track B**: use the already-trained NO2/SO2/VIIRS CNN's activity signal
   (calibrated probability or an intermediate embedding) as a covariate in a small, physically
   interpretable correction model (e.g. gradient-boosted trees or linear correction on the IME
   residual) — not an end-to-end deep net, since facility count is currently too small for
   that to be credible. This is the specific fusion the literature review found nowhere else.
3. **Expand the facility set modestly** (target: 10–20 Indian coal plants with usable OCO-3
   coverage, reusing `pick_plants.py`/`process_plant.py`), enough to support leave-one-facility-
   out cross-validation without triggering large, unnecessary downloads.
4. **Validate with facility-level (leave-one-facility-out) cross-validation**, not random
   splits, for both the corrected emission-rate estimate and (as a secondary check) the
   existing detector.
5. **Run a source-wise ablation**: physics-only (IME) → +NO2 activity → +SO2 → +VIIRS →
   +per-overpass wind correction → +uncertainty, formalizing the pattern the weekly logs
   already do informally for the classifier, but for the emission-rate task.
6. **Benchmark final per-plant estimates against Climate TRACE**, explicitly as an independent
   reference (not ground truth, not a training label) — report correlation, bias, MAE, and
   discuss the shared-thermal-proxy caveat from §3.

### Why this over the alternatives
It's the only option that (a) is clearly not already published, (b) reuses essentially all
existing code and data rather than requiring a rebuild, (c) fits a local-machine, small-N
setting honestly instead of pretending to have deep-learning-scale data, and (d) fixes a
problem the project already found and flagged in its own logs — which is a strong, legible
narrative for a paper ("we identified this weakness ourselves, the literature confirms it's
the dominant error source, here's how we fixed it and quantified the fix").

## 7. Proposed datasets

| Source | Use | Notes |
|---|---|---|
| Sentinel-5P NO2/SO2 (existing) | activity covariate | already exported for 5 plants + hard negatives; expand to new facility set |
| VIIRS VNP14A1 MaxFRP (existing) | activity covariate | same |
| OCO-3 `L2_Lite_FP` v11r (existing, via `earthaccess`) | XCO2 soundings for IME | expand facility list; check coverage before large pulls |
| ERA5 (`ECMWF/ERA5/DAILY` via GEE, or ERA5 hourly via CDS if per-overpass granularity is needed) | per-overpass wind speed/direction | current pipeline only pulls annual/daily; need overpass-time matching |
| Climate TRACE bulk download (CC BY 4.0, `climatetrace.org/data`) | independent benchmark only | do not use as label; note US/EU-centric training may not generalize to India |
| Facility metadata (plant capacity, fuel type, coordinates) | context features / stratifying the ablation and error analysis | reuse/extend `data/top5_plants.csv` |

No new large NASA/Copernicus collections are strictly required for the recommended direction —
it's primarily a re-analysis and extension of data already being pulled, plus per-overpass wind
matching and a modest facility-count expansion. This keeps compute/download load small, per
the project's stated constraints.

## 8. Proposed model architecture

```
OCO-3 soundings ─────────────┐
                              ├─► IME mass balance (existing physics_gaussian.py logic)
Per-overpass ERA5 wind ──────┘        │
                                       ▼
                              Q_physics ± σ_wind  (NEW: per-overpass wind, propagated uncertainty)

NO2 + SO2 + VIIRS tiles ─► existing CNN (Track A) ─► activity probability / embedding
                                       │
                                       ▼
                    Small correction model (gradient-boosted trees / linear,
                    NOT end-to-end deep learning given small N)
                    input: [Q_physics, activity signal, facility metadata]
                    output: corrected Q_final ± σ_final
```

Rationale for keeping the correction model simple: with ~10–20 facilities, a deep multimodal
fusion network (U-Net/Transformer) would be underpowered and uninterpretable relative to a
physics-anchored correction term — matches the brief's instruction not to reach for
architecture complexity without justification.

## 9. Proposed experiments / ablation

| Model | Inputs | Purpose |
|---|---|---|
| A0 | IME, annual-mean wind (current Track B) | baseline, already built |
| A1 | IME, per-overpass wind | isolate wind-conditioning effect |
| A2 | A1 + uncertainty propagation | calibration check |
| A3 | A1 + NO2/SO2 activity covariate | does activity signal correct physics residual? |
| A4 | A3 + VIIRS | does VIIRS add anything beyond NO2/SO2 (test the literature's proxy claim in *this* fused context)? |
| A5 (full) | A4 + facility metadata | final model |

Each ablation evaluated under leave-one-facility-out CV, not random split.

## 10. Validation strategy

- **Facility-level (leave-one-facility-out)** cross-validation as the primary strategy — never
  split by tile/sounding when the same facility appears in multiple folds.
- **Temporal holdout** as a secondary check (train on 2019, test on 2020 for facilities with
  both years of coverage).
- Report MAE, RMSE, R², relative error, bias, and uncertainty calibration (does the stated σ
  actually bracket the Climate TRACE reference at the claimed confidence level?).
- Explicitly separate "vs. Climate TRACE" (independent benchmark) from any internal
  cross-validation metric — never conflate the two.

## 11. Expected risks

- OCO-3 coverage may be too thin for several candidate plants (as already seen with
  Tirora/Mundra) — facility-set expansion may yield fewer usable plants than hoped.
- Per-overpass ERA5 wind matching adds pipeline complexity (need overpass timestamps, not just
  daily aggregates) — GEE's `ECMWF/ERA5/DAILY` may not be granular enough; may need hourly ERA5
  via Copernicus CDS.
- Small facility count (~10–20) limits statistical power of the ablation — results should be
  reported with appropriate caution (confidence intervals on the ablation deltas themselves).
- Climate TRACE may not correlate well with our physics-based estimate even if our method is
  correct, since Climate TRACE's own India-specific accuracy is unvalidated — a low correlation
  should be discussed as "two independent, imperfect estimators disagreeing," not treated as
  proof either is wrong.

## 12. Expected limitations (to state explicitly in the paper)

- Column XCO2 → surface emission-rate inversion (IME) carries irreducible uncertainty from
  boundary-layer height and background-subtraction choices, not just wind.
- Facility count is small relative to deep-learning-scale studies (Deb & Das 2025's 71 plants);
  this paper's contribution is methodological rigor (uncertainty, generalization, fusion), not
  raw sample size.
- Climate TRACE comparison is benchmark-vs-benchmark, not benchmark-vs-ground-truth, since no
  independently measured (CEMS-equivalent) Indian plant-level CO2 data is assumed available.

## 13. Conference-paper contribution statement (draft)

"We present a physics-informed, uncertainty-aware pipeline for facility-level CO2 emission-rate
estimation from real (not simulated) OCO-3 soundings, that (1) replaces the common annual-mean
wind assumption with per-overpass wind conditioning and propagates wind/footprint uncertainty
into calibrated per-plant confidence intervals, (2) fuses a multimodal NO2/SO2/VIIRS
combustion-activity signal as a physics-informed correction to the mass-balance estimate rather
than as an isolated classifier, (3) validates with facility-level (leave-one-facility-out)
cross-validation and a source-wise ablation study to determine which data modalities actually
help, and (4) benchmarks the result against Climate TRACE as an independent reference for a set
of Indian coal power plants — a validation rigor and facility population not covered by prior
OCO-2/3 point-source CO2 studies."

Suggested title: **"Wind-Conditioned, Activity-Corrected CO2 Emission Estimation from OCO-3:
An Uncertainty-Aware, Facility-Generalizing Study of Indian Coal Power Plants"**

## 14. Implementation roadmap (high level; sequencing only, not started)

1. Per-overpass wind matching + uncertainty propagation into `physics_gaussian.py` (extends
   existing file; addresses the project's own documented bug/limitation).
2. Facility-set expansion: identify 10–20 candidate Indian coal plants with usable OCO-3
   coverage (reuse `pick_plants.py`), check coverage before large pulls.
3. Extract activity signal from the existing trained 3-channel CNN (probability or penultimate
   embedding) for each facility.
4. Build the small correction model (A1→A5 ablation ladder in §9).
5. Implement leave-one-facility-out CV harness (new, since current split is random).
6. Pull Climate TRACE bulk data for the facility set; compute benchmark comparison metrics.
7. Produce the full evaluation figure set (§12 of the original brief): predicted-vs-actual,
   residuals, error distribution, facility-level comparison, uncertainty calibration plot,
   ablation table, feature importance.

## 15. Exact next steps for this repository

- Do not modify `train_detector.py`/`train_2channel.py`/`train_3channel.py` yet — the existing
  CNN is reused as-is for its activity signal (step 3 above), not retrained from scratch as
  step 1.
- First code change, when approved: extend `physics_gaussian.py` (or a new sibling script) to
  accept per-overpass wind and emit `Q ± σ` instead of a point estimate, tested first on the
  existing 4 plants before any facility-set expansion — this is the smallest, most
  self-contained change that directly targets the literature-confirmed dominant error source
  and requires no new data collection.
- Only after that is validated: proceed to facility-set expansion and the fusion/ablation
  pipeline.

---

*This document synthesizes: direct repository inspection (README.md, WEEK2–7 logs, all
scripts), and three independent literature-review passes covering (a) OCO-2/3 point-source
physics/ML methods and the Deb & Das 2025 baseline, (b) NO2/VIIRS multimodal fusion and
facility-level generalization literature, and (c) Climate TRACE methodology and appropriate-use
analysis.*
