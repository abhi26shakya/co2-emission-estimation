# CO2 Emission Estimation — Master Research Documentation

**Repository:** `co2-emission-estimation`
**Author (commits):** `devashishpandey044-code` (2026-07-01 – 2026-07-10), then `Abhishek Shakya` (2026-08-10 onward; several commits from 2026-08-12/13 co-authored with Claude Sonnet 5)
**Branch:** `main` (single branch, no tags)
**Timeline:** 2026-07-01 (initial commit) → 2026-08-13 (latest commit + uncommitted work-in-progress)
**Document compiled:** 2026-08-13, from git history, commit diffs, weekly logs, README, RESEARCH_PLAN, NEXT_STEPS, source code, and data-file schemas. Everything below is sourced from these artifacts; nothing is invented. Anything not found in the repo is explicitly marked **"Not documented / needs verification."**

---

## 1. Project Overview

### 1.1 Objective

Estimate CO2 emissions from individual Indian coal-fired power plants using publicly available satellite data, and separately detect the presence/activity of a power plant from satellite imagery — as two related but currently **unfused** research tracks:

- **Track A — Plant Detector:** a small CNN that classifies 64×64 pixel satellite tiles as "coal power plant" vs. "not a power plant," built incrementally by fusing NO2 → SO2 → VIIRS thermal channels, and hardened against confounding "hard negative" sources (cities, steel plants, highways).
- **Track B — CO2 Emission-Rate Estimator:** a physics-based Integrated Mass Enhancement (IME) mass-balance pipeline that turns OCO-3 satellite XCO2 (column-averaged dry-air CO2 mole fraction) soundings into a per-plant emission rate estimate (tons CO2/year), with wind, NO2 co-location, and uncertainty quantification, cross-checked against the independent Climate TRACE dataset.

### 1.2 Motivation / Research Problem

The project's own `RESEARCH_PLAN.md` (added Week 7, framed explicitly as a literature-grounded gap analysis before further work) states the guiding question directly: existing satellite-based CO2 point-source estimation work (see §3 below) has largely been demonstrated in simulation or on a handful of well-instrumented facilities elsewhere in the world. The gaps this project targets, per `RESEARCH_PLAN.md` §3–5, are:
- Applying **per-overpass wind-conditioned IME** to **real** (not simulated) OCO-3 data.
- Producing **calibrated, per-plant uncertainty** bands rather than point estimates.
- **Fusing** an independent activity/detection signal (Track A) with the physics-based estimate (Track B).
- **Facility-level ablation** validation (i.e., does the detector generalize to power plants it never saw during training, not just to held-out tiles of the same plants).
- An **India-specific** benchmark against Climate TRACE, which has not — per the project's own literature scan — been done for Indian coal plants specifically.

### 1.3 Citation Basis

The project cites **Deb & Das 2025** (arXiv:2502.02083) as related prior work combining NO2 and XCO2 fusion for power-plant emission estimation (per `README.md`'s citation section and `RESEARCH_PLAN.md`'s literature comparison). `RESEARCH_PLAN.md` explicitly flags this and several other works so as not to re-claim their findings as novel — see §3 below.

### 1.4 Two-Track, Unfused Structure

As of the latest committed and uncommitted state, Track A (detector) and Track B (emission estimator) remain **separate pipelines**. A partial bridge exists: Week 9's `extract_activity_signal.py` reuses Track A's trained CNN (inference only, not retrained) to produce a per-facility "activity probability" for Track B's facilities, and Week 10's `reliability_model.py` tested (and found largely inconclusive, at current sample size) whether this activity signal predicts Track B's own uncertainty. A full correction/fusion model combining the two tracks is listed as future work (see §11).

---

## 2. Timeline Summary — Full Commit History

All 29 commits, chronological (oldest → newest). Dates as recorded in git (local/IST as committed).

| # | Hash | Date | Author | Message |
|---|---|---|---|---|
| 1 | `5ac9bc1` | 2026-07-01 09:28 | devashishpandey044-code | Initial commit |
| 2 | `acda84b` | 2026-07-01 10:00 | devashishpandey044-code | Add first_no2.py |
| 3 | `12eec10` | 2026-07-02 11:30 | devashishpandey044-code | Week 2: dataset pipeline + CNN detector (91.2% test acc) |
| 4 | `c5e3d04` | 2026-07-03 12:52 | devashishpandey044-code | Week 3: hard negatives, honest retraining, failure analysis, Grad-CAM |
| 5 | `0591451` | 2026-07-03 12:52 | devashishpandey044-code | Add Week 3 figures: comparison, false alarms, Grad-CAM |
| 6 | `7b2b792` | 2026-07-03 14:30 | devashishpandey044-code | Add week logs |
| 7 | `ebb5059` | 2026-07-03 14:33 | devashishpandey044-code | Update gitignore for data and models |
| 8 | `3c4095` | 2026-07-10 10:58 | devashishpandey044-code | Add OCO-3 CO2 enhancement results |
| 9 | `d704b7a` | 2026-07-10 11:13 | devashishpandey044-code | Add post-Week-4 CO2 estimation: OCO-3 pipeline and 4-plant results |
| 10 | `0404273` | 2026-08-10 23:27 | Abhishek Shakya | Add README for CO2 Emission Estimation project |
| 11 | `f0bd9b0` | 2026-08-11 02:27 | Abhishek Shakya | Add Week 5: VIIRS thermal fusion for the plant detector |
| 12 | `8357017` | 2026-08-11 03:44 | Abhishek Shakya | Add Week 6-7: CO2 emission-rate estimate and highway hard-negative expansion |
| 13 | `2d807ea` | 2026-08-12 14:45 | Abhishek Shakya | Add Week 8: per-overpass wind conditioning and uncertainty for CO2 IME estimates |
| 14 | `4efc02a` | 2026-08-12 21:10 | Abhishek Shakya | Add Week 9: facility-set expansion for CO2 IME estimates |
| 15 | `76236ac` | 2026-08-12 21:28 | Abhishek Shakya | Add activity-signal extraction from Track A CNN for Track B facilities |
| 16 | `9ed65ad` | 2026-08-12 21:37 | Abhishek Shakya | Add reliability-model exercise for Track B uncertainty (negative result) |
| 17 | `a16e2af` | 2026-08-12 21:51 | Abhishek Shakya | Add Climate TRACE benchmark comparison for Track B facilities |
| 18 | `34375c1` | 2026-08-12 22:09 | Abhishek Shakya | Add Week 10 log and Talcher diagnosis (tight interval, wrong estimate) |
| 19 | `a0d70d0` | 2026-08-12 22:16 | Abhishek Shakya | Add background-definition sensitivity as a third uncertainty term |
| 20 | `fc3655b` | 2026-08-12 22:30 | Abhishek Shakya | Add facility-level train/test split for Track A, fixing Week-7-flagged leakage |
| 21 | `5baa198` | 2026-08-12 22:37 | Abhishek Shakya | Confirm activity-signal memorization/instability via checkpoint comparison |
| 22 | `43c5399` | 2026-08-12 23:42 | Abhishek Shakya (co-authored: Claude Sonnet 5) | Add NEXT_STEPS.md: consolidated Week 2-11 status and roadmap tracking |
| 23 | `0cebaa2` | 2026-08-13 01:35 | Abhishek Shakya (co-authored: Claude Sonnet 5) | Process RGundem and Korba: next 2 plants in facility-set expansion |
| 24 | `03e1f52` | 2026-08-13 04:44 | Abhishek Shakya (co-authored: Claude Sonnet 5) | Process ShriSingajiMalwa and Koradi; Tamnar incomplete (resource pressure) |
| 25 | `f88cff6` | 2026-08-13 09:16 | Abhishek Shakya (co-authored: Claude Sonnet 5) | Process Tamnar, completing the ShriSingajiMalwa/Koradi/Tamnar round |
| 26 | `0d2a823` | 2026-08-13 10:49 | Abhishek Shakya (co-authored: Claude Sonnet 5) | Process Kudgi, final plant of this session's facility-set expansion |
| 27 | `5ca1c0f` | 2026-08-14 00:05 | Abhishek Shakya (co-authored: Claude Sonnet 5) | Complete facility-set expansion to 20/20 plants; add reliability fixes and master research documentation |
| 28 | `5bcbb34` | 2026-08-14 03:02 | Abhishek Shakya (co-authored: Claude Sonnet 5) | Expand Track A positive class to 20 facilities; add exhaustive LOFO harness and root-cause negative-enhancement anomaly |
| 29 | `cadb194` | 2026-08-14 03:06 | Abhishek Shakya (co-authored: Claude Sonnet 5) | Fully resolve the negative-CO2-enhancement anomaly: ShriSingajiMalwa was a seasonal sampling artifact |

**Timeline gaps and pacing:**
- ~1 week gap: Jul 3 → Jul 10 (Week 3 → Week 4 handoff).
- **~4 week gap: Jul 10 → Aug 10** — unexplained in the repo (no commits, no log entries); marked **Not documented / needs verification**.
- From Aug 10 onward, authorship switches to the current user (Abhishek Shakya), starting with a catch-up README commit.
- 16 of the first 26 commits (Week 5 through the post-Week-11 facility-processing round) land within a single ~33-hour span (Aug 11 02:27 → Aug 13 10:49), consistent with an intensive, session-based working style. This pattern is also reflected in the language of the logs themselves ("continues the same session").
- Commits 27–29 (Aug 14, 00:05 → 03:06) form a same-day follow-up session, covered narratively in §12 below.

**No branches or tags exist** other than `main`.

---

## 3. Research Problem & Literature Context

Summarized from `RESEARCH_PLAN.md` §3–6 (a literature-grounded gap-analysis document, explicitly marked as a planning artifact — "no code changed as part of this document").

### 3.1 What is already established in the literature (not claimed as novel by this project)

| Reference | Contribution already established |
|---|---|
| Deb & Das 2025 (arXiv:2502.02083) | NO2 + XCO2 fusion for power-plant emission signal |
| GMD 2024 (SMARTCARB-related) | CNN-based emission regression, but demonstrated in **simulation only**, not on real satellite data |
| (VIIRS literature, unspecified citation — see note) | VIIRS as an activity/operational-status proxy for combustion facilities |
| (Wind-error literature, unspecified citation — see note) | Wind-speed measurement error is the dominant uncertainty term in mass-balance plume emission estimates |

*Note: `RESEARCH_PLAN.md` §3–4 references these findings in a comparison table against 6 total references; the exploration pass captured the substance of the comparisons but not a verbatim bibliographic list for every entry. Full citation text should be read directly from `RESEARCH_PLAN.md` §3–4 if exact reference formatting is needed — marked **Not documented / needs verification** for anything beyond what's summarized here.*

### 3.2 Literature explicitly used for methodology (cited directly in code docstrings)

- **Varon et al. 2018** — source of the Integrated Mass Enhancement (IME) method and the effective-wind-speed scaling factor α = 0.5 used in `physics_gaussian.py` (originally developed for CH4 plumes).
- **Nassar et al. 2017** — CO2 point-source mass-balance estimation from OCO-2/3, underlying the same mass-balance logic applied here.
- **Reuter et al. 2019** — same lineage, OCO-2/3 CO2 point-source estimation.
- **Li et al. 2021, GRL** — cited in `RESEARCH_PLAN.md`'s comparison table (specific contribution not captured in detail during this pass — **Not documented / needs verification**, refer to `RESEARCH_PLAN.md` directly).
- **Climate TRACE** — used as an independent benchmark dataset (not training/ground-truth), explicitly caveated as itself being a satellite-thermal-proxy ML pipeline, unvalidated for India (see §7.3 and §10).

### 3.3 Genuinely open gaps this project targets (per `RESEARCH_PLAN.md` §5)

Five ranked candidate contributions are listed; the project explicitly adopts **Recommendation #1**: *"physics-informed, uncertainty-aware, activity-conditioned, facility-validated, benchmarked estimation."* The proposed paper title recorded in `RESEARCH_PLAN.md` §6 is:

> *"Wind-Conditioned, Activity-Corrected CO2 Emission Estimation from OCO-3: An Uncertainty-Aware, Facility-Generalizing Study of Indian Coal Power Plants."*

### 3.4 Roadmap (RESEARCH_PLAN.md §14)

A 7-step implementation roadmap was set after Week 7, and subsequent weeks (8–11+) executed directly against it:

1. Per-overpass wind-conditioned IME (→ Week 8)
2. Facility-set expansion (→ Week 9, completed 20/20 through the Aug 13 session)
3. Activity-signal extraction (→ Week 9/10; re-run and low-confidence flag resolved in the 2026-08-14 follow-up, §12.2)
4. Correction/fusion model combining Track A + Track B (→ the uncertainty-*prediction* feasibility check flipped from negative (N=7) to positive (N=17, §12.3); the actual Q-correcting A1→A5 ladder is still **not yet done**, see §11/§12.6)
5. Leave-one-facility-out (LOFO) cross-validation (→ **done for Track A** in the 2026-08-14 follow-up, §12.4; Track B's own correction model will need its own LOFO harness once built)
6. Climate TRACE benchmark (→ done, Week 10; re-run at N=17 in the 2026-08-14 follow-up, §12.5)
7. Full evaluation figures (→ **done** in the 2026-08-14 follow-up, §12.5, for everything not blocked on the unbuilt step-4 model)

§15 of `RESEARCH_PLAN.md` also records an explicit methodological constraint: **"do not retrain the CNN yet"** — honored through at least Week 10 (Track A's only retrain event is the Week 11 facility-level-split experiment, which was a deliberate, flagged exception to diagnose the leakage bug, not a general retrain).

---

## 4. Data Sources

| Source | Product | Used for | Access method |
|---|---|---|---|
| Sentinel-5P (Copernicus) | `COPERNICUS/S5P/OFFL/L3_NO2` — tropospheric NO2 column density | Track A primary channel; Track B NO2 co-location check | Google Earth Engine (`ee` Python API), hardcoded GCP project `opportune-lore-415218` |
| Sentinel-5P (Copernicus) | `COPERNICUS/S5P/OFFL/L3_SO2` — SO2 column density | Track A 2nd channel (Week 4) | Google Earth Engine |
| NASA VIIRS | `VNP14A1` — MaxFRP (Maximum Fire Radiative Power, MW) | Track A 3rd channel (Week 5), thermal/activity signal | Google Earth Engine |
| NASA OCO-3 | `OCO3_L2_Lite_FP`, version `11r` — XCO2 (column-averaged dry-air CO2 mole fraction) soundings | Track B core input | `earthaccess` (NASA Earthdata login), parsed with `xarray` |
| ECMWF ERA5 | Daily mean (u, v) 10m wind components | Track B wind-conditioning and uncertainty | Not documented / needs verification — access library/method for ERA5 pull not confirmed in this pass (likely via Earth Engine's ERA5 collection or CDS API; re-check `process_plant.py` imports directly if needed) |
| WRI | Global Power Plant Database | Plant selection (`pick_plants.py`), site deduplication | Static CSV (`data/powerplants.csv`, 34,937 lines) |
| Climate TRACE | India power-sector CO2 estimates (2021) | Independent benchmark for Track B (not ground truth) | REST API, endpoint discovered empirically at `api.climatetrace.org/v6/swagger/openapi.json` (per `pull_climate_trace.py` docstring) |

---

## 5. Track A: Plant Detector — Chronological Development

### 5.1 Model Architecture (constant across all channel-count variants)

A small CNN, per the codebase exploration of `train_detector.py` / `train_2channel.py` / `train_3channel.py`:

```
Conv(C_in → 16) → BatchNorm → SiLU → MaxPool
Conv(16 → 32)   → BatchNorm → SiLU → MaxPool
Conv(32 → 64)   → BatchNorm → SiLU
GlobalAvgPool
Dropout(0.3)
Linear(64 → 2)
```
- Optimizer: AdamW, learning rate 3e-4
- Epochs: 30
- Loss: CrossEntropyLoss
- `C_in` = 1 (NO2 only, Weeks 2–3), 2 (NO2+SO2, Week 4), or 3 (NO2+SO2+VIIRS, Weeks 5–11)

### 5.2 Week-by-Week Narrative

**Pre-Week-2 (2026-07-01, commits `5ac9bc1`, `acda84b`):** Initial commit adds mostly-empty stub files, including `physics_gaussian.py` (which stays empty until Week 6) and `first_no2.py`, which is then fleshed out (+21 lines) as the earliest NO2 data-exploration script.

**Week 2 (2026-07-02, `12eec10`):** First real milestone. Scripts added: `check_tiles.py`, `export_monthly.py`, `export_negatives.py`, `export_tiles.py`, `pick_plants.py`, `train_detector.py`. Selected the top 5 Indian coal plants by capacity from the WRI Global Power Plant Database. Exported 64×64 px NO2 tiles (60 km box) via Earth Engine for these 5 plants and 5 rural "negative" locations, monthly for 2019–2020. Trained a 1-channel CNN on an RTX 3050 GPU. **Result: 91.2% test accuracy** on plant-vs-rural classification. The week's log flagged two possible next directions: move toward real CO2 estimation (OCO-2/3), or harden the detector with harder negatives — both were subsequently pursued.

**Week 3 (2026-07-03, `c5e3d04`, `0591451`):** Added `analyze_failures.py`, `compare_tiles.py`, `export_hard_negatives.py`, `gradcam.py`, `plan_hard_negatives.py`, `train_hard.py`. Introduced 20 "hard negative" locations — cities, steel plants, highways — chosen via `plan_hard_negatives.py` with a haversine-distance sanity check (≥80 km from any known plant). **Key finding:** accuracy on the hard-negative-only test set dropped to **77.1%** (mixed-negative accuracy: 87.5%). Grad-CAM analysis (`gradcam.py`, hooking the last conv layer) showed the model was attending to any compact NO2 hotspot, not a plant-specific signature — the week's log reframes the Week 2 model as a "concentrated combustion detector," not a power-plant detector specifically. Worst false alarms: Jamshedpur, Bhilai (both steel), Bangalore, Mumbai. Figures committed: `comparison_types.png`, `gradcam.png`, `worst_false_alarms.png`.

**Same day (`7b2b792`):** `WEEK2_LOG.txt` and `WEEK3_LOG.txt` committed — establishes the weekly lab-notebook-log convention that continues through Week 11.

**Week 4 (2026-07-10, part of `d704b7a`, "post-Week-4"):** Added `build_2channel.py`, `train_2channel.py`, `analyze_2channel.py`, `compare_so2.py`, plus SO2 tile exports (`export_so2.py`). Fused SO2 as a 2nd channel. **Result: hard-only accuracy 79.2%** — fixes the city false alarms but *not* the steel-plant false alarms (steel manufacturing also emits SO2, so the added channel doesn't discriminate steel from coal-plant combustion).

**Week 5 (2026-08-11, `f0bd9b0`):** VIIRS thermal fusion. Added `export_viirs.py`, `build_3channel.py`, `train_3channel.py` (initial version). VIIRS `MaxFRP` band captures thermal anomalies; the module docstring notes that "power-plant stack heat is a weak, sub-pixel signal at 1km, so sparse coverage here is expected, not a bug." **Result: hard-only accuracy tied at 79.2%** (no net change), but VIIRS measurably reduced steel-plant false-alarm *confidence* specifically (Bhilai 0.59→0.52, Jamshedpur 0.54→0.50), offset by highways becoming relatively more confusable in the fused feature space.

**Week 6-7 (2026-08-11, `8357017` — this single commit covers both a Track B milestone, Week 6, and a Track A milestone, Week 7):** Track A portion (Week 7): expanded highway hard negatives from 5 to 10, spread more broadly geographically. **Result: first accuracy improvement since Week 4 — hard-only 79.2% → 81.2%.** The Week 7 log explicitly flags a **methodological risk not yet fixed**: the train/test split is at the tile level (random), meaning the same physical facility can appear in both train and test across different months — a potential leakage source. This flag is not acted on until Week 11.

**Week 11 (2026-08-12, `fc3655b`):** Fixes the Week-7-flagged leakage. Added `facility_level_split()` to `train_3channel.py` — groups tiles by facility (regex-strips the `_YYYY_MM.npy` month suffix from filenames) so all months of a given physical site land on the same side of the split, stratified by class (since only 5 positive-class facilities exist in Track A's training set). The old `tile_level_split()` is kept for direct comparison rather than deleted. **Result — a real, substantial drop once the leak is closed:**
- Hard-only accuracy: **81.2% → 67.3%**
- "Mixed" model's plant recall: **53% → 8%**

This is the single most consequential finding in Track A's development: the apparent Week 7 improvement was partly (or largely) an artifact of memorized facility identity, not a genuinely more general detector. Separate checkpoints (`detector3_2ch_hard_only_facility_split.pt`, etc.) are saved so the earlier tile-level checkpoints remain available/reproducible for downstream use (notably `extract_activity_signal.py`, see §6.6).

Same day, `5baa198` directly follow-up-confirms this concern from the activity-signal-extraction angle (see §6.3).

### 5.3 Full Accuracy Progression Table

| Week | Channels | Split type | Hard-negative-only accuracy | Note |
|---|---|---|---|---|
| 2 | NO2 | tile-level, easy negatives only | 91.2% | Baseline; later shown to be a generic "combustion detector" |
| 3 | NO2 | tile-level, hard negatives added | 77.1% | Exposes the confound |
| 4 | NO2+SO2 | tile-level | 79.2% | Fixes city false alarms, not steel |
| 5 | NO2+SO2+VIIRS | tile-level | 79.2% (tie) | Reduces steel false-alarm confidence specifically |
| 7 | NO2+SO2+VIIRS, +5 highways | tile-level | 81.2% | First improvement since Week 4 |
| 11 | NO2+SO2+VIIRS | **facility-level** (leakage-corrected) | **67.3%** | True generalization is substantially weaker than tile-level numbers suggested |
| 12.1 | NO2+SO2+VIIRS, 20 facilities (4→20 positive class) | facility-level, single random split | 82.8% (95.0% mixed, 88% recall) | Superseded as a generalization estimate by the row below — single split, not representative |
| 12.4 | NO2+SO2+VIIRS, 20 facilities, 12-tile depth | **exhaustive LOFO** (21 folds) | — (mean recall 47.2%, tile-weighted 48.7%) | Superseded by the row below; recall ranged 0% (Kahalgaon, Kudgi, Mouda) to 100% (Anpara, Korba, Rihand, Talcher), predicted by detector confidence (r=+0.90) |
| 12.9 | NO2+SO2+VIIRS, 21 facilities, **24-tile depth** (2019+2020) | **exhaustive LOFO** (22 folds) | — (mean recall **69.1%**, tile-weighted **70.5%**) | The current trustworthy generalization number. Matching the original top-5's temporal depth (not augmentation, which was tried and made no difference) closed most of the gap; only Kudgi and ShriSingajiMalwa remain at 0% recall |

Full detail on rows 12.1/12.4/12.9 is in §12.1, §12.4, and §12.9.

### 5.4 Explainability / Failure-Analysis Tooling

- `gradcam.py` — Grad-CAM on the 1-channel detector's last conv layer.
- `analyze_failures.py`, `analyze_2channel.py`, `analyze_failures_3ch.py` — rank hard-negative sources by mean P(plant) score; `analyze_failures_3ch.py` also hardcodes earlier weeks' scores inline for a cross-week steel-plant-focus comparison table.
- `compare_tiles.py`, `compare_so2.py` — visual/statistical comparisons of representative tiles by source type (plant, city, industry, highway, rural).
- `check_tiles.py` — sanity-check visualization tool (gap %, mean value per tile).
- `summary_figure.py` — bar chart of the Week 2→5 accuracy progression and the city-vs-steel false-alarm confidence shift; some values are hardcoded from earlier logs rather than recomputed live (flagged here for transparency, not as an error — this is a plotting convenience script).

---

## 6. Track B: CO2 Emission-Rate Estimation — Chronological Development

### 6.1 Pipeline Overview

Orchestrated end-to-end by `process_plant.py <PlantName>`, which for a single plant:

1. **OCO-3 scan** — `earthaccess.search_data(short_name="OCO3_L2_Lite_FP", version="11r", ...)` over a ±1° bounding box for calendar year 2020; downloads each matching granule and opens it with `xarray`.
2. Filters soundings to `quality_flag == 0` and a tighter ±0.5° box around the plant; extracts `lat`, `lon`, `xco2`, and (since Week 8) a `day` field derived from `sounding_id` (which encodes `YYYYMMDDHHMMSSmm`) via integer division by `10**8`.
3. **Checkpointing** — writes `data/<Plant>_scan_checkpoint.npz` after each granule using a write-then-rename pattern for atomicity, so an interrupted scan resumes rather than restarting from scratch.
4. Computes **near-plant** mean XCO2 (soundings within 0.25° of the plant) vs. **background** mean XCO2 (soundings in a 0.4°–0.9° ring around the plant), and the enhancement `near_mean − bg_mean`.
5. **NO2 co-location check** — pulls a Sentinel-5P NO2 grid via Earth Engine, finds the peak-NO2 pixel, and computes its distance from the plant's registered coordinates ("NO2 peak km") as an independent plausibility signal.
6. **Wind alignment check** — pulls ERA5 daily mean (u, v) wind over the plant for the year, computes the downwind direction, and compares it against the centroid direction of the top-20%-CO2 soundings — i.e., does the CO2 enhancement actually sit downwind of the plant, as physically expected?
7. Appends a results row to `data/plant_results.json`, using `fcntl.flock` (POSIX file locking) to guard the read-modify-write cycle against concurrent-process corruption — this locking was added later in the project after a race condition was found (see §8).

### 6.2 The IME (Integrated Mass Enhancement) Method

Implemented in `physics_gaussian.py`. Despite the filename, this is explicitly **not** a textbook ground-level Gaussian plume model — the module's own docstring explains why: OCO-3 XCO2 is a **column-averaged** quantity, not a surface concentration, so ground-level Gaussian dispersion physics doesn't directly apply. IME is presented as the literature-standard column-consistent alternative, citing Varon et al. 2018 (originally for CH4 plumes) and the same mass-balance logic underlying Nassar et al. 2017 and Reuter et al. 2019 for OCO-2/3 CO2.

Core formula:

```
Q = U_eff · IME / L_eff
```

Where:
- **`IME`** (kg) — total excess column-mass CO2 over near-plant soundings, computed by `column_mass_enhancement()`: converts the ppm excess to kg/m² using standard surface pressure (101,325 Pa) and the CO2/air molar-mass ratio (0.04401 / 0.02897 kg/mol), then multiplies by an assumed per-sounding footprint area of **2.25 km²** (an OCO-3 sounding-footprint-area assumption — see Limitations, §10).
- **`L_eff`** (m) — effective plume length, computed as √(plume area covered).
- **`U_eff`** — effective wind speed, `U_eff = α · wind_speed`, with **α = 0.5** (the Varon et al. 2018 default, correcting for the fact that a 10 m surface wind measurement doesn't equal the deeper-layer mixing wind relevant to plume transport).

### 6.3 Week-by-Week Narrative

**Pre-commit exploration (dates not separately logged — Not documented / needs verification for exact dates):** `oco_search.py`, `oco_open.py`, `oco_vindhyachal.py`, `oco_scan_vindhyachal.py` are early, single-plant, largely hardcoded scripts that established the OCO-3 search/download/parse workflow, later generalized into `process_plant.py`.

**Week 4 / "post-Week-4" (2026-07-10, `3c4095` and `d704b7a`):** Track B is born. `data/plant_results.json` first appears (54 lines). `co2_enhancement.py` (earliest, single-plant/Vindhyachal-hardcoded near-vs-background XCO2 comparison), `co2_no2_colocation.py` (NO2 heatmap + CO2 sounding scatter overlay), `build_2channel.py`, `analyze_2channel.py`, `compare_so2.py` all land in this large catch-up commit, alongside saved sounding data for Mundra/Sasan/Tirora and the WRI power-plant CSV. The commit message frames this explicitly as a "post-Week-4" catch-up — i.e., work had progressed past what was previously committed.

**Week 6 (2026-08-11, part of `8357017`):** `physics_gaussian.py` — empty since the initial commit — is implemented for the first time with the IME method described in §6.2. **A significant bug is found and fixed in the same week:** the initial wind implementation averaged the ERA5 (u, v) vector components over the *entire year* before taking the magnitude. Because wind direction rotates seasonally, vector-averaging largely cancels out the speed component — this was correct behavior for `wind_check.py` (which only needs a direction), but wrong for an emission-rate estimate, which needs a genuine mean wind *speed*. After the fix (averaging daily speed magnitudes instead of vector-averaging first):
- Vindhyachal's estimate changed from 14.0 → **44.6 Mt/yr**
- Sasan's estimate changed from 14.5 → **37.2 Mt/yr**
(roughly a 3x change in both cases). Tirora's estimate remained implausibly low post-fix, attributed to thin sounding coverage. Mundra was skipped entirely (only 57 soundings, judged insufficient).

**Week 7 (same commit, `8357017`):** (Track A work — see §5.2.) Also flags the Track A leakage risk not fixed until Week 11.

**Week 8 (2026-08-12, `2d807ea`):** Two structural improvements to `physics_gaussian.py`, per `RESEARCH_PLAN.md` §14's top-priority recommendation:
1. **Per-overpass wind conditioning** — replaces the single annual-mean wind scalar with wind matched to the actual sounding dates, where sufficient match data exists.
2. **Uncertainty propagation** — output changes from a bare point estimate to `Q ± σ` with explicit low/high bands.

Also adds checkpointing to `process_plant.py`'s OCO-3 scan loop (see §6.1, point 3). **Two further bugs found and fixed during self-review** (an `ecc:python-review` agent pass was run before committing, per the commit's own narrative):
- A wind-day deduplication bug that deflated `wind_speed_std` by treating multiple soundings on the same day as independent wind samples.
- A broken atomic-checkpoint rename, caused by `np.savez` silently auto-appending a `.npz` extension to the target path, breaking the intended write-then-rename atomicity.

Example post-fix output: Sasan → `39.96M ± 17.90M t/yr`.

**Week 9 (2026-08-12, `4efc02a`):** Facility-set expansion from 5 candidate plants to **20**. `pick_plants.py` and `data/candidate_plants.csv` generalize what was previously a hardcoded 4-plant registry in `process_plant.py` into a CSV-driven registry. `check_coverage.py` is added as a cheap, metadata-only OCO-3 granule-count pre-filter meant to triage which candidates are worth the expensive full download-and-process step — **explicitly logged as a negative result**: granule-count pre-filtering did not, in practice, discriminate well between eventually-good and eventually-poor candidates. 5 new plants processed this week (Talcher, Rihand, Sipat, ChandrapurCoal, Anpara): total processed count 4 → 9, plausible-estimate count 2/4 → 7/9. The week's log also records a measured real-world cost constraint: **~50–65 minutes per plant**, network-bound (OCO-3 granule downloads), documented explicitly as a hard constraint on how fast facility expansion can proceed.

**Same day (`76236ac`):** Activity-signal extraction. Reuses Track A's already-trained `detector3_2ch_mixed.pt` checkpoint (the "mixed" variant chosen for presumed better generalization) purely for **inference** — not retrained — to compute a per-facility activity probability and a 64-dimensional embedding for all 9 Track B facilities at that point (`export_facility_tiles.py`, `extract_activity_signal.py`). To do this correctly, `extract_activity_signal.py` recomputes Track A's exact training-time per-channel normalization (mean/std) from the original training tile set, since `train_3channel.py` never saved those statistics. The activity signal broadly tracks the physics-based Q estimate; ChandrapurCoal is flagged as an outlier/suspicious case.

**Same day (`9ed65ad`):** Reliability-model exercise — `reliability_model.py`. Tests whether the Track A activity signal, or the wind/CO2-offset alignment metric, predicts `physics_gaussian.py`'s own self-reported relative uncertainty (`q_rel_std`), via single-feature Pearson correlation and leave-one-out cross-validation on N=7 facilities. **Explicit negative result, honestly reported:** the best single-feature correlation found was r = −0.62, but this does not survive leave-one-out CV at N=7 (LOO R² = −1.22). The script's own docstring explains the restraint: N=7 is judged far too small for a multi-feature model, so a gradient-boosted-trees or similar model was deliberately *not* fit, in favor of a simpler, honestly-caveated single-feature result.

**Same day (`a16e2af`):** Climate TRACE benchmark. `pull_climate_trace.py` pulls Climate TRACE's India power-sector CO2 estimates (2021) for all 9 facilities at that point, matched to Track B facilities by lat/lon centroid distance. Result: **5 of 7** facilities with a Track B estimate fall within their own stated uncertainty interval when compared to the Climate TRACE figure. The two misses: Sasan (Track B estimate 2.13x over Climate TRACE) and Talcher (Track B estimate 0.50x under Climate TRACE, despite having the *tightest* uncertainty interval of any plant to that point — this contradiction is what motivates the Talcher-specific diagnosis below).

**Week 10 (2026-08-12, `34375c1`):** Talcher diagnosis. `diagnose_talcher.py` investigates why Talcher's tightest, most-confident interval was also its most-wrong estimate. Root cause identified: a thin signal-to-noise ratio in Talcher's raw CO2 enhancement (0.18, vs. 1.27 for the well-bracketed comparison plant Rihand) combined with high sensitivity to how the background annulus is defined — Talcher's IME swings ~20% across 5 reasonable background-ring definitions, vs. only ~4% for Rihand. This identifies a **structural uncertainty source the model wasn't previously accounting for at all**.

**Same day (`a0d70d0`):** The background-definition sensitivity discovered while diagnosing Talcher is folded directly into `physics_gaussian.py` as a formal **third uncertainty term** (`_bg_definition_rel_std()`), computed for all facilities alongside the existing wind and IME-bootstrap-sampling terms, and combined in quadrature into the final `q_rel_std`.

**Week 11 (2026-08-12):** Track A leakage fix (`fc3655b`, see §5.2), followed same day by `5baa198` — a direct confirmation exercise re-running `extract_activity_signal.py` against the new facility-split checkpoint and adding `compare_activity_signal_checkpoints.py` to diff the two checkpoints' activity-signal outputs per facility. Two distinct problems are found: **memorization** (Tirora's activity signal shifts by −0.117 between checkpoints, consistent with the old checkpoint having partly memorized it) and **instability on genuinely novel facilities** (Sipat and ChandrapurCoal swing by ~0.22 between checkpoints; Rihand, by contrast, remains stable) — a caveat for anyone using the activity signal downstream.

**Same day (`43c5399`):** `NEXT_STEPS.md` is added — a consolidated status document synthesizing all of Weeks 2–11 against the `RESEARCH_PLAN.md` §14 roadmap. This is the project's first "as of Week 11" running status snapshot.

### 6.4 Facility-Processing Round (2026-08-13, commits 23–26, all co-authored with Claude Sonnet 5)

A continuation session processing the remaining candidate plants toward the full 20:

- **`0cebaa2` (01:35):** Processes RGundem and Korba → 11/20 total. Both fall back to annual-mean wind (insufficient per-overpass matches); both estimates land on the low end (1.5–2.7M t/yr) relative to prior plants (range 3.2–47M t/yr).
- **`03e1f52` (04:44):** Processes ShriSingajiMalwa and Koradi → 13/20 candidates touched. **First-ever negative CO2 enhancement observed in this project:** ShriSingajiMalwa −1.05 ppm; Koradi −0.13 ppm — the latter despite having the *best* sounding coverage of any plant processed to date (3,672 soundings). Tamnar was attempted but left incomplete due to resource pressure: repeated `resource_tracker` semaphore interruptions, worsening as free memory fell over the course of the long session; its checkpoint was left intact at ~86/360 granules.
- **`f88cff6` (09:16):** Completes Tamnar → third consecutive negative-enhancement plant (−0.12 ppm), despite having the *best* wind/CO2-offset alignment (16°) of any plant processed to date — directly complicating the working heuristic (from the Talcher diagnosis) that good wind/CO2 alignment implies a trustworthy signal.
- **`0d2a823` (10:49):** Processes Kudgi → breaks the 3-for-3 negative-enhancement streak (+0.34 ppm), bounding the negative-enhancement issue to those specific 3 plants rather than a systemic pipeline problem. Facility-set expansion now at 13/20 candidates processed; 7 remain (Kahalgaon, Mouda, Chhabra, Farakka, Simhadri, plus 2 known non-starters: Mundra and Sipat).

### 6.5 Uncertainty Model — Summary

Three independent relative-uncertainty terms, combined in quadrature into `q_rel_std`:

1. **Wind term** — per-overpass ERA5 wind matching (Week 8), replacing the Week 6 annual-mean-scalar approach; relative std of matched daily wind speeds, deduplicated to unique days (Week 8 bug fix) to avoid deflating the std.
2. **IME bootstrap sampling term** — 500 bootstrap resamples of the near-plant/background sounding populations.
3. **Background-annulus-definition term** (Week 10) — recomputes IME under 5 alternate background-ring definitions, holding the near-plant zone fixed; captures structural sensitivity that matters most for plants with a weak plume signal (e.g., Talcher).

Per `WEEK8_LOG.txt`, the wind term dominates overall (45–59% relative std in observed cases) vs. 4–6% for the IME sampling term — consistent with, and empirically reproducing, the literature finding that wind measurement error is the dominant uncertainty source in mass-balance plume estimates (see §3.2).

### 6.6 Supporting Scripts (reference)

| Script | Purpose |
|---|---|
| `wind_check.py` | Original, single-plant-hardcoded wind-vs-CO2-offset alignment check; predecessor of the wind-alignment step later folded into `process_plant.py` |
| `co2_enhancement.py` | Earliest near-vs-background XCO2 histogram/enhancement test (Vindhyachal, hardcoded) |
| `co2_no2_colocation.py` | NO2 heatmap + CO2 sounding scatter overlay for visual co-location |
| `check_coverage.py` | Cheap OCO-3 metadata-only granule-count pre-check (Week 9; logged negative result) |
| `extract_activity_signal.py` | Track A CNN inference (not retrain) → per-facility activity probability + embedding for Track B facilities |
| `reliability_model.py` | Tests whether activity signal / alignment predicts Track B's own uncertainty (negative result, N=7) |
| `pull_climate_trace.py` | Pulls and matches Climate TRACE India power-sector CO2 estimates |
| `diagnose_talcher.py` | Root-cause investigation of Talcher's tight-but-wrong estimate |
| `compare_activity_signal_checkpoints.py` | Diffs activity signal across Track A checkpoints (tile-level vs. facility-level split) |
| `export_facility_tiles.py` | Exports NO2/SO2/VIIRS tiles for the 5 new Track B-only facilities (kept in a separate `data/activity_tiles/` directory, deliberately not merged into Track A's main training tile set) |

---

## 7. Datasets & Results Reference

### 7.1 `data/plant_results.json`

20 entries as of the latest committed/working-tree state (one per plant candidate). Fields observed:

```
plant                 — plant name (string)
lat, lon               — plant coordinates (float, degrees)
hit_days                — number of distinct days with usable OCO-3 soundings (int)
soundings               — total usable sounding count after quality filtering (int)
co2_enhancement_ppm     — near-plant minus background mean XCO2 (float, or null if insufficient soundings)
bg_std_ppm              — standard deviation of background XCO2 (float)
no2_peak_km             — distance from plant coordinate to peak-NO2 pixel (float, km)
wind_deg                — mean wind direction (float, degrees)
co2_offset_deg          — direction of top-20%-CO2 sounding centroid from plant (float, degrees)
wind_co2_diff_deg       — angular difference between wind_deg and co2_offset_deg (float, degrees)
```

Example entries (redacted/representative, from the exploration pass):
```json
{"plant": "Mundra", "lat": 22.82, "lon": 69.55, "hit_days": 5, "soundings": 57,
 "co2_enhancement_ppm": null, "bg_std_ppm": 0.825, "no2_peak_km": 64.7,
 "wind_deg": 98.0, "co2_offset_deg": 312.0, "wind_co2_diff_deg": 146.0}

{"plant": "Sasan", "lat": 23.98, "lon": 82.62, "hit_days": 19, "soundings": 7379,
 "co2_enhancement_ppm": 0.739, "bg_std_ppm": 1.835, "no2_peak_km": 15.5,
 "wind_deg": 106.0, "co2_offset_deg": 52.0, "wind_co2_diff_deg": 54.0}
```
`soundings` ranges from 12 (Kudgi — essentially unusable) to 7,379 (Sasan). `co2_enhancement_ppm` is `null` for plants below a usable-soundings threshold (e.g., Mundra, with only 57 soundings).

### 7.2 `data/emission_estimates.json`

13 entries (fewer than 20 total plants, since some candidates have insufficient soundings for a stable IME computation). Fields:

```
plant, n_soundings_used, ime_kg, l_eff_m, wind_speed_ms, wind_speed_std_ms,
wind_mode, n_wind_days_matched, u_eff_ms, wind_rel_std, ime_rel_std,
bg_rel_std, n_bg_definitions, q_rel_std, q_kg_s, q_t_per_year,
q_t_per_year_std, q_t_per_year_low, q_t_per_year_high
```

`wind_mode` is either `"per-overpass"` (≥3 unique wind-matched days available) or `"annual-mean (fallback)"`. Example: Sasan → `q_t_per_year = 39,957,381 ± 18,089,602`, with the wind relative std dominating the uncertainty budget at 44.6% (vs. 4.1% IME sampling, 6.5% background-definition).

An `emission_estimates_before.json` also exists in `data/`, presumably an earlier snapshot — **exact provenance/diff from the current file not verified in this pass; marked Not documented / needs verification.**

### 7.3 `data/climate_trace_comparison.json`

Structure: `{year_caveat, benchmark_caveat, facilities: [...]}`, each facility entry containing `climate_trace_co2_t` (2021 figure), `our_q_t_per_year` (2020 figure — note the year mismatch, explicitly caveated in the file itself per the `year_caveat` field), `ratio_ours_over_ct`, and `bracketed_by_our_interval` (boolean). Notable results: Sasan overestimated relative to Climate TRACE by 2.13x; Tirora underestimated (ratio 0.21).

### 7.4 `data/reliability_model_results.json`

N=7 feature table with Pearson correlation and leave-one-out CV fit results, explicitly caveated in the file/script as "indicative only" given the small sample size.

### 7.5 `data/talcher_diagnosis.json`

Structured root-cause diagnosis output from `diagnose_talcher.py` (signal-to-noise ratio, background-definition sensitivity, comparison against Rihand).

### 7.6 Other Data Assets

- `data/candidate_plants.csv` — 20 rows (name, capacity_mw, lat, lon), the current Track B candidate registry.
- `data/top5_plants.csv` — 5 rows, original Track A plant set (raw WRI names with STPS/TPP/UMPP suffixes).
- `data/hard_negatives.csv` — 20 rows: 5 cities, 5 industry/steel sites, 10 highway corridors.
- `data/powerplants.csv` — WRI Global Power Plant Database, 34,937 lines (not read in full during this pass).
- `data/<Plant>_soundings.npz` — per-plant OCO-3 sounding archives (arrays: `lat`, `lon`, `xco2`, `day` as integer `YYYYMMDD`); 20 files, one per candidate plant.
- `.npy` tile files — thousands, under `data/monthly/`, `data/so2/`, `data/viirs/`, `data/twoch/`, `data/threech/`, `data/activity_tiles/` (each a `(H,W)` or `(C,H,W)` float32 array, 64×64 px), organized in `positive/negative/hard_negative` subfolders for Track A.
- PNG figures (11 total) — Grad-CAM overlays, false-alarm panels, tile comparisons, per-plant CO2 summary dashboards, wind/CO2 alignment scatter maps — all matplotlib output from the analysis scripts listed above.
- `data/scan_tmp/` — 8 leftover temporary files from OCO-3 granule scanning (working artifacts, not final results).

---

## 8. Errors, Bugs, and Fixes Log

A consolidated table of every explicitly documented bug-find-and-fix cycle in this project. All of these were narrated with before/after numbers in the weekly logs or commit messages, not silently corrected — the project has a consistent practice of logging its own mistakes.

| # | When found | Bug | Root cause | Fix | Measured impact |
|---|---|---|---|---|---|
| 1 | Week 6 | Wind speed drastically underestimated in emission-rate calc | ERA5 (u,v) vectors were averaged over the full year *before* taking magnitude; seasonal direction rotation cancels most of the speed | Average daily wind-speed *magnitude* instead of vector-averaging first | Vindhyachal 14.0→44.6 Mt/yr; Sasan 14.5→37.2 Mt/yr (~3x change) |
| 2 | Week 7 (flagged) / Week 11 (fixed) | Track A train/test leakage | Random tile-level split let the same physical facility appear in both train and test across different months | Added `facility_level_split()`, grouping by facility, stratified by class | Hard-only accuracy 81.2%→67.3%; mixed-model plant recall 53%→8% |
| 3 | Week 8 | `wind_speed_std` deflated | Multiple same-day soundings treated as independent wind samples, artificially shrinking the computed std | Deduplicate wind-matching to unique days | Corrected uncertainty term (no single before/after figure captured in this pass — see `WEEK8_LOG.txt` directly) |
| 4 | Week 8 | Checkpoint atomicity broken | `np.savez` silently auto-appends `.npz` to the target filename, breaking the intended write-then-rename atomic-save pattern | Adjusted checkpoint save logic to account for the auto-appended extension | Checkpointing now genuinely atomic/resumable |
| 5 | Week 9 (logged, not "fixed" — a negative result) | Granule-count pre-filter didn't discriminate candidates | Cheap OCO-3 metadata-only granule counts don't correlate well with eventual estimate quality | None applied — filter abandoned/deprioritized as a triage signal | Documented as a negative result, not carried forward as a gate |
| 6 | Week 10 | Talcher: tight uncertainty interval but wrong estimate (vs. Climate TRACE) | Thin signal-to-noise ratio (0.18 vs. 1.27 for Rihand) and high sensitivity to background-annulus definition (20% IME swing vs. 4%) — a real uncertainty source not previously modeled | Added a third, formal background-definition-sensitivity uncertainty term to `physics_gaussian.py` | Uncertainty budget now includes this term for all facilities |
| 7 | 2026-08-13 session (uncommitted) | `process_plant.py` OCO-3 downloads hung indefinitely | Twice observed (Kahalgaon, then Mouda) — each stuck 1.5h+ in `CLOSE_WAIT` state to `urs.earthdata.nasa.gov` | Added a 90-second per-granule `SIGALRM` timeout around `earthaccess.download()` | Prevents indefinite hangs; not yet committed to git as of this writing |
| 8 | 2026-08-13 session (uncommitted) | `data/plant_results.json` race condition / corruption risk | Two duplicate-process incidents: (a) an ad hoc Kahalgaon restart accidentally running two processing loops simultaneously, (b) a deliberate parallel Farakka/Simhadri run racing against the original sequential loop. Only the final JSON write was previously lock-protected — per-granule checkpoint writes were not | Added `fcntl.flock`-protected read-modify-write around the `plant_results.json` update | Both incidents were caught and the duplicate process killed within seconds each time; no data loss occurred, but the structural fix (checkpoint-level locking specifically, as opposed to results-file locking) is still only partially addressed — see §9 |

---

## 9. Current Implementation State

### 9.1 Committed State (as of `cadb194`, 2026-08-14 03:06 — see §12 for the narrative of commits 27–29)

- **Track A:** 3-channel (NO2+SO2+VIIRS) detector, retrained on the expanded 20-facility positive class (§12.1). Single-split facility-level accuracy reached **82.8%** hard-only / **95.0%** mixed (88% recall, 4 held-out); the more rigorous exhaustive LOFO evaluation (§12.4, 21 folds, 12-tile depth) found this optimistic — true mean recall was only 47.2% (tile-weighted 48.7%). A same-day follow-up (§12.9) traced most of this gap to a real data-quantity limitation (16 facilities had half the temporal depth of the original top-5) rather than a pure signal-strength ceiling: exporting a matching 2nd data year and re-running LOFO (22 folds, 24-tile depth) raised **true mean recall to 69.1%** (tile-weighted 70.5%) — data augmentation was tried on both dataset sizes and helped neither. Only Kudgi and ShriSingajiMalwa still show 0% LOFO recall, narrowing the remaining generalization weakness to two specific facilities rather than a broad ~50% failure rate.
- **Track B:** all 20/20 candidate plants processed. 17/20 produce a `physics_gaussian.py` estimate (Mundra, Sipat, Simhadri excluded for a genuine 0-near-sounding coverage gap, not left unverified). The three previously-flagged negative-CO2-enhancement facilities (ShriSingajiMalwa, Koradi, Tamnar) are now fully root-caused (§12.6–§12.7): Koradi and Tamnar are statistically consistent with zero signal; ShriSingajiMalwa was a seasonal near/background sampling artifact whose sign flips positive under a same-month comparison. None represent a pipeline bug or genuine negative signal.
- Climate TRACE uncertainty-interval calibration: 71% (5/7, Week 10) → 53% (9/17, §12.5, larger sample) → **35.3% (6/17, §12.12a)** after month-stratifying `physics_gaussian.py`'s near/background comparison — each drop is a more honest number from a larger sample or a real bug fix, not a regression in methodology, but the trend means current calibration is meaningfully worse than early small-sample figures suggested.
- `reliability_model.py`, re-run at N=17 (up from N=7): now a **positive result** — `hit_days` predicts `q_rel_std` at r=−0.617, LOO R²=0.212 (§12.3).
- `NEXT_STEPS.md`, `RESEARCH_PAPER.md`, and this document were all updated in commits 27–29 and are current as of `cadb194`.

### 9.2 Uncommitted Working-Tree State (as of 2026-08-14, this session)

Per `git status`, the working tree is again ahead of the last commit (`cadb194`) — this is separate, newer work not covered by §12:

- **Modified (uncommitted):** `data/candidate_plants.csv`, `data/plant_results.json`, `pick_plants.py`.
- **New/untracked:** `data/TalwandiSabo_soundings.npz`, and a `logs/` directory.

This reflects a paused facility-set-expansion batch (a further 10 candidate coal plants — TalwandiSabo, Lalitpur, Pryagraj(Bara), Dadri(Nctpp), Akaltara, KGudemNew, Raichur, Bellary, RayalSeema, Sagardighi — queued via `run_queue.sh`, then explicitly stopped mid-run per user instruction). Only TalwandiSabo appears to have produced a soundings artifact before the stop. **Not yet documented in prose** (no week/session log exists for this batch) and **not committed** — flagged here for completeness, not analyzed further in this pass since the work was intentionally halted rather than completed.

### 9.3 What Is Fused vs. Not

Track A and Track B remain architecturally separate pipelines. The only bridge is the one-directional, inference-only activity-signal extraction (§6.3, Week 9) and its accompanying reliability test (Week 9, negative result) — there is no trained joint or correction model combining the two tracks' outputs as of this writing.

---

## 10. Limitations & Assumptions

Compiled from `README.md`, `RESEARCH_PLAN.md`, and script docstrings:

- **Climate TRACE is a benchmark, not ground truth.** Explicitly argued in three places (`RESEARCH_PLAN.md`, `pull_climate_trace.py`'s docstring, `diagnose_talcher.py`): Climate TRACE is itself a satellite-thermal-proxy ML pipeline, unvalidated specifically for India. Disagreement between Track B and Climate TRACE is framed as "two independent, imperfect estimators disagreeing," not as Track B being definitively wrong.
- **N=7 is too small for a multi-feature reliability model.** `reliability_model.py`'s docstring explicitly reasons that a gradient-boosted or other multi-feature model would overfit at this sample size, and deliberately restricts itself to a single-feature, honestly-caveated LOO-CV fit.
- **Activity-signal instability on non-training facilities.** `compare_activity_signal_checkpoints.py` quantifies that facilities never included in Track A's training set show activity-probability swings of up to 0.22 across reasonable training variation (i.e., between the tile-level and facility-level split checkpoints) — a reliability caveat for anyone using this signal downstream.
- **IME footprint-area assumption.** The 2.25 km² per-OCO-3-sounding footprint area used in `column_mass_enhancement()` is a fixed assumption, not measured per-sounding — **the sensitivity of results to this specific assumed value has not been separately tested in this project** (Not documented / needs verification: no ablation of this constant was found in the exploration pass).
- **Wind uncertainty dominates and is not fully resolved by per-overpass matching.** Even with per-overpass wind conditioning (Week 8) and the added background-definition term (Week 10), the wind relative-std term remains the largest single uncertainty contributor observed (45–59% in cases reviewed).
- **Track A's true generalization accuracy improved substantially but is not yet resolved.** The single-split facility-level number (82.8% hard-only / 95.0% mixed, §12.1) was not representative — exhaustive LOFO (§12.4, 12-tile depth) found true mean recall of only 47.2%. A same-day follow-up (§12.9) traced most of this to a real data-quantity gap and closed it by exporting matching 2019 tiles, raising true LOFO recall to **69.1%** (§12.9). Two facilities (Kudgi, ShriSingajiMalwa) still show 0% recall — the remaining, narrower open weakness.
- **Negative CO2 enhancement cases (ShriSingajiMalwa, Koradi, Tamnar) are now fully root-caused (§12.6–§12.7).** Koradi and Tamnar are statistically consistent with zero signal (not a real anomaly); ShriSingajiMalwa was a seasonal near/background sampling artifact that flips positive under a same-month comparison. A general follow-up remains open: `physics_gaussian.py` does not yet month-stratify its near/background comparison, so this exact failure mode could recur elsewhere.
- **No `pyproject.toml`/environment file, `CLAUDE.md`, or `docs/` folder exists anywhere in this repository.** `requirements.txt` was added (§12.12c), pinned to `/opt/miniconda3/envs/co2`; that pass also found scripts have inconsistently run under a second Python (the macOS framework build) missing `earthaccess`/`xarray`/`geemap`. A basic test suite was added (§12.13, `tests/`, 22 stdlib-`unittest` tests) but has no CI wired up and doesn't cover integration paths (Earth Engine, model training). This is stated here as a factual gap, not a criticism.
- **The ~4-week gap in the commit history (2026-07-10 → 2026-08-10) is unexplained** in any file found during this pass — marked **Not documented / needs verification**.
- **ERA5 wind-data access method is not confirmed** in this pass (likely Earth Engine's ERA5 collection or the CDS API, based on the pattern used for other Earth Engine data, but not directly verified against `process_plant.py`'s import statements during this documentation pass) — **Not documented / needs verification**.

---

## 11. Pending Work / Future Scope

Directly from the unfinished portions of `RESEARCH_PLAN.md` §14's roadmap and `NEXT_STEPS.md`'s own tracking, plus items surfaced organically during the research passes for this document:

1. ~~Complete the facility-set expansion~~ — **Done**: all 20/20 facilities processed; 17/20 produce a `physics_gaussian.py` estimate (the remaining 3 excluded for a genuine 0-near-sounding coverage gap, not left unverified). See §12.3.
2. ~~Root-cause the negative-CO2-enhancement plants~~ — **Done** (§12.6–§12.7): Koradi and Tamnar's negative values are statistically consistent with zero (no detectable signal, not a real anomaly); ShriSingajiMalwa's was a seasonal-sampling artifact (near-plant data all from January, background blended in April/May's ~4 ppm-higher seasonal baseline) — same-month comparison flips the sign to a small positive difference. All three resolved; none a real negative signal or pipeline bug. New follow-up item: month-stratify `physics_gaussian.py`'s near/background comparison generally.
3. **Correction / fusion model combining Track A and Track B** (`RESEARCH_PLAN.md` §14 step 4) — the uncertainty-*prediction* feasibility check is now a positive result at N=17 (§12.3), but the actual Q-correcting fusion model is still not built; it needs an independent ground-truth emissions source, since Climate TRACE is explicitly barred from use as a training label. A first fusion attempt was made without that source (§12.11): does Track A's activity signal improve Track B's own uncertainty prediction, or track actual Climate-TRACE-bracketing success? **Both tested negative** — this specific self-consistency framing doesn't work, though it doesn't rule out fusion in general. *(Still open — the key remaining blocker; one concrete approach has now been ruled out.)*
4. ~~Leave-one-facility-out (LOFO) cross-validation~~ — **Done for Track A** (`lofo_track_a.py`, 21 exhaustive folds, §12.4); revealed the single-split 88% recall was not representative (true mean recall 47.2%). Track B's own correction model will need its own LOFO harness once item 3 is built.
5. ~~Full evaluation figures~~ — **Done** (`evaluation_figures.py`, §12.5): `data/eval_climate_trace_comparison.png` and `data/eval_track_a_ablation_and_generalization.png`, covering everything not blocked on the unbuilt item-3 model.
6. ~~Improve Track A's facility-level generalization~~ — **Done** (§12.9–§12.10): exhaustive LOFO recall rose from 47.2% to **69.1%** by matching all facilities to 24-tile temporal depth (data augmentation tried separately and found not to help). The two remaining 0%-recall cases, **Kudgi and ShriSingajiMalwa**, were diagnosed (§12.10): their raw NO2/SO2 signal sits at or below the rural-negative-class noise floor in every month, not a data or training artifact, and no coordinate/tile-placement bug explains it (both are better-centered than several 100%-recall facilities). This is a genuine satellite-observability limit, not a fixable modeling gap — closed as diagnosed, not as "improved further."
7. ~~Structural checkpoint-file locking~~ — **Done** (§12.12b): an exclusive, non-blocking per-plant `fcntl.flock` on `data/{NAME}.lock`, acquired before `process_plant.py`'s scan loop begins, prevents the actual 2026-08-13 bug (two independent scan loops for the same plant) rather than just guarding individual checkpoint writes.
8. **Explain the 2026-07-10 → 2026-08-10 gap**, or otherwise confirm there is no missing/undocumented work from that period, for the record's completeness. *(Still open.)*
9. ~~Formalize a dependency manifest~~ — **Done** (§12.12c): `requirements.txt`, pinned to the working conda env. Surfaced an unrelated finding: a second Python (the macOS framework build) lacking `earthaccess`/`xarray`/`geemap` has also been used to run scripts, undetected because this session's numpy/torch/pandas/ee-only scripts happened to work under either environment — flagged in `requirements.txt` itself. ~~A basic test suite~~ — **Done** (§12.13): 22 stdlib-`unittest` tests covering `physics_gaussian.py`'s math, `build_3channel.py`'s tile pairing, and `lofo_track_a.py`'s fold-splitting logic. No CI configured to run them automatically, and no coverage of integration paths (Earth Engine, model training) — *(that gap still open, but the core item is done)*.

---

## 12. Same-Day Follow-Up Session (2026-08-14, commits 27–29)

This section covers a follow-up session run after §§1–11 above were originally compiled (which described state as of `5ca1c0f`, 2026-08-14 00:05). The work below is now committed — split across `5ca1c0f` (master documentation + reliability fixes), `5bcbb34` (Track A expansion, LOFO harness, negative-enhancement root-cause), and `cadb194` (ShriSingajiMalwa full resolution) — and is reflected in the timeline table (§2, rows 27–29).

**12.1 Track A positive-class expansion (4→20 facilities).** Wrote `export_new_positive_tiles.py`, exporting NO2/SO2/VIIRS 2020 tiles directly into `data/{monthly,so2,viirs}/positive` for the 16 Track B facilities Track A had never trained on (it had only ever used the original `top5_plants.csv` 4-site set). Reused the 5 facilities' tiles already downloaded to `data/activity_tiles/` (Week 10) where present; fetched the remaining 11 fresh (one transient `HTTP 503` on Koradi SO2 2020-01, backfilled manually). Rebuilt `data/threech/positive` (312 tiles, up from ~120) and reran `train_3channel.py`. **Result: the Week 11 leakage-driven recall collapse (53%→8%) is resolved** — hard_only facility-split accuracy rose 67.3%→82.8%, mixed facility-split accuracy reached 95.0% with 88% plant recall (single random split, 4 held-out facilities).

**12.2 Activity-signal re-extraction.** Re-ran `extract_activity_signal.py` against both retrained checkpoints, now covering all 20 facilities (up from 9). Fixed a bug this surfaced: `monthly_paths()` still special-cased non-original facilities into `data/activity_tiles/`, which would have silently skipped the 11 newly-exported facilities; now all 20 read from the `positive` dirs uniformly. Also rewrote `compare_activity_signal_checkpoints.py`, whose `FACILITY_GROUPS` and prose conclusion were hardcoded to the old 9-facility/4-site world; it now derives groups from `train_3channel.py`'s actual printed test_groups and generates its conclusion from computed deltas. **Result: the memorization signature is gone** — held-out facilities (Anpara, ChandrapurCoal, Sasan, Talcher) now show mean|delta|=0.036, *more* stable than in-training facilities (0.061), the reverse of what memorization would predict.

**12.3 Reliability model re-run at N=17 (roadmap step 4).** `physics_gaussian.py` had only been run against 13 of 20 facilities; re-ran it against the full set, adding 4 more estimates (Kahalgaon, Mouda, Chhabra, Farakka — soundings existed but hadn't gone through the physics step). 17/20 facilities now have an estimate; only Mundra, Sipat, Simhadri remain excluded (genuine 0-near-sounding coverage gap). Re-ran `reliability_model.py` (§7.4's N=7 negative result) at N=17: **now a positive result** — `hit_days` predicts `q_rel_std` at r=−0.617, LOO R²=0.212 (MAE=0.095), versus no surviving correlation at N=7.

**12.4 General leave-one-facility-out CV harness (roadmap step 5).** Wrote `lofo_track_a.py`: previously step 5 was only satisfied ad hoc by Week 11's single random facility-level split. This harness holds out each plant facility in turn (21 folds — Mundra's two units count separately), training a fresh model each time. **Result: the single-split 88% recall (§12.1) was not representative.** True exhaustive LOFO mean recall is 47.2% (tile-weighted 48.7%), ranging from 0% (Kahalgaon, Kudgi, Mouda) to 100% (Anpara, Korba, Rihand, Talcher). Follow-up `lofo_recall_correlates.py` (N=20) found `activity_prob_mean` — the standard model's own confidence — strongly predicts which facilities generalize (r=+0.903, LOO R²=0.783): weak/ambiguous combustion signature predicts LOFO failure, suggesting intrinsic signal clarity rather than memorization drives the gap. A silent `NaN`-correlation bug (from `co2_enhancement_ppm` being null for the 3 excluded facilities) was fixed via pairwise deletion.

**12.5 Climate TRACE re-run and evaluation figure set (roadmap steps 6–7).** Re-ran `pull_climate_trace.py` (no code change needed — it already read the full facility list dynamically) against all 17 estimate-bearing facilities. **Uncertainty calibration dropped from 71% (5/7, Week 10) to 53% (9/17)** — a more honest number from the larger sample, not a regression. Wrote `evaluation_figures.py`, producing `data/eval_climate_trace_comparison.png` (predicted-vs-actual, residuals, uncertainty calibration, facility-level comparison) and `data/eval_track_a_ablation_and_generalization.png` (ablation history table, LOFO recall distribution, both feature-importance panels) — the roadmap step 7 deliverable, scoped honestly to what current data supports (Climate TRACE benchmark-vs-benchmark, Track A's channel/split history in place of the unbuilt Track B A1→A5 ladder, correlation-based rather than trained-model feature importance).

**12.6 Root-caused the negative-CO2-enhancement anomaly (§11 item 2).** Wrote `diagnose_negative_enhancement.py`, reusing `diagnose_talcher.py`'s near/background and background-definition-sensitivity functions plus a new statistical-significance check (is the near-minus-background difference distinguishable from zero given its standard error?). **Result: 2 of 3 facilities (Koradi z=−1.58, Tamnar z=−1.33) are statistically consistent with zero** — the honest characterization is "no detectable enhancement given available signal-to-noise," not a real negative signal, which also resolves the apparent Tamnar wind-alignment contradiction (wind alignment and CO2 signal-to-noise are computed independently; a facility can have plausible NO2 plume geometry and simultaneously too weak a CO2 signal to measure). ShriSingajiMalwa's negative enhancement (−1.052 ppm, z=−5.32) was initially statistically significant and looked like a genuine anomaly.

**12.7 Fully resolved ShriSingajiMalwa.** Wrote a targeted follow-up, `diagnose_shrisingajimalwa.py`, testing seasonal and directional sampling confounds against Koradi as a comparison case. **Found the cause: all 59 near-plant soundings are from a single month (January), while the background ring also draws from April/May (mean XCO2 ~415.6–416.5 ppm, ~4 ppm above January's ~412 ppm baseline) and October** — a real seasonal atmospheric-CO2 cycle unrelated to the plant, likely surfaced by OCO-3 swath geometry only crossing the tight near-plant circle on some overpasses. Restricting the comparison to the one month both zones share (January) **flips the sign**: near=412.321 ppm vs bg=412.029 ppm, a small positive +0.292 ppm difference (z=2.13) — consistent with a real, weak plant signal, not a negative anomaly. All three originally-flagged facilities are now resolved; none represent a pipeline bug or a genuinely negative CO2 signal. **`physics_gaussian.py` does not currently stratify near/background comparisons by month**, so this exact failure mode could recur for any facility with uneven near/background month coverage — a worthwhile general hardening item, not just a one-off fix for this facility.

**12.8 What remains open after this session.** The A1→A5 Q-correcting ablation ladder (`RESEARCH_PLAN.md` §9) is still not built — it needs an independent ground-truth emissions source, since Climate TRACE is explicitly barred from use as a training label (§7). Track A's exhaustive LOFO recall (47.2% at the time of writing) reframes "improve facility-level generalization" (§11 item 6) as the primary open weakness, now with a concrete lead (facilities with ambiguous NO2+SO2+VIIRS signal generalize worst) — **§12.9 below reports a follow-up that acts on this lead and substantially closes the gap.** The negative-CO2-enhancement anomaly (§11 item 2) is now **fully closed** — all three flagged facilities resolved to either noise or a seasonal-sampling artifact, none a real negative signal. §11 item 1 (facility-set completion) is otherwise unaffected by this session's work. A new, general item surfaces from §12.7: month-stratify `physics_gaussian.py`'s near/background comparison to prevent this failure mode from recurring elsewhere.

**12.9 Closing the LOFO generalization gap: augmentation vs. more training data (later same-day follow-up).** §12.4's LOFO recall (47.2%) and §12.4's own follow-up correlation both pointed at weak/ambiguous per-facility signal as the driver. Two competing explanations were tested: (a) the 16 newly-expanded facilities had only 12 tiles each (2020 only) vs. 24 for the original top-5 (2019+2020) — a training-diversity artifact fixable without new data; or (b) a genuine signal-strength limit, not fixable without new data. **Experiment 1 (`lofo_track_a_aug.py`): light spatial augmentation (random flip/90°-rotation per training epoch), no new data.** Result: no improvement (mean recall 47.2%→46.8%, tile-weighted 48.7%→48.7% — unchanged) — rules out (a) as the primary driver. **Experiment 2 (`export_new_positive_tiles_2019.py`): a real 2nd data year.** Exported 2019 NO2/SO2/VIIRS tiles for the 16 facilities (612/612 downloaded, zero failures; incidentally also covered `TalwandiSabo`, an unrelated in-progress candidate that had entered `plant_results.json`), bringing every facility to 24-tile depth. Rebuilt `data/threech/positive` (312→516 tiles) and reran the LOFO harness. **Result: mean recall rose 47.2%→69.1% (tile-weighted 48.7%→70.5%)** — 17 of 21 comparable facilities improved, several sharply (Mouda 0%→71%, Sipat 33%→96%, TIRORA 4%→67%). **Experiment 3: augmentation reapplied on top of the expanded dataset** — result: slightly worse, not better (69.1%→67.2%, 70.5%→68.4%), a second negative result for augmentation. **Conclusion: the LOFO gap was substantially a real data-quantity limitation, not primarily a signal-strength ceiling** — closing it required actual additional satellite observations, not synthetic augmentation. Two facilities still show 0% LOFO recall after the fix — **Kudgi and ShriSingajiMalwa** — narrowing "improve facility-level generalization" (§11 item 6) from a broad ~50%-of-facilities problem to these two specific cases, which do look like genuine weak-signal facilities per the original correlation lead. One unexplained operational anomaly: one fold (SASAN_UMPP) took ~21 minutes vs. ~45s typical during the 24-tile LOFO re-run — likely transient system contention (this session had concurrent background processes running), not a code defect, since the fold's own result (92% recall) was unremarkable. Prior 12-tile-depth results preserved as `data/lofo_track_a_results_v1_12tile.json` / `lofo_track_a_aug_results_v1_12tile.json` before being superseded.

**12.10 Diagnosing the two remaining 0%-recall facilities (Kudgi, ShriSingajiMalwa).** Wrote `diagnose_lofo_weak_facilities.py` to test whether these two facilities' raw NO2/SO2/VIIRS signal is genuinely weak in absolute terms, or a detector/training artifact. Compared per-channel mean tile intensity against the rest of the positive class, hard_negative, and rural_negative. **Result: both facilities' raw NO2/SO2 signal sits at or below rural_negative levels** (Kudgi NO2=2.66e-05 vs. rural_negative's 2.49e-05; ShriSingajiMalwa SO2=1.19e-04, essentially identical to rural_negative's 1.19e-04) and roughly 3–4x below the rest-of-positive-class average (NO2=8.13e-05, SO2=2.95e-04) — checked every individual month, not just the average, with no month showing a signal spike for either facility. `no2_peak_km` (distance from registered plant coordinates to the tile's peak NO2 pixel) rules out a coordinate/tile-placement bug: both (3.6km, 1.0km) are *smaller* than several 100%-LOFO-recall facilities' own offsets (Talcher 34.7km, Anpara 16.7km, Rihand 8.1km) — if anything, these two are better-centered than the facilities the detector *does* recognize. **Conclusion: this is a genuine satellite-observability limit, not a fixable bug** — the raw signal is not present in the tile data for a classifier to learn from, regardless of architecture or training regime. Both are large plants (2,400–2,520 MW per `data/candidate_plants.csv`, comparable to or larger than the well-detected comparison facilities), so capacity alone doesn't explain the gap. A plausible but **unverified** physical explanation: modern flue-gas desulfurization (FGD) / selective catalytic reduction (SCR) emissions-control equipment suppressing stack NO2/SO2 well below older, less-controlled plants of similar capacity — `candidate_plants.csv` has no emissions-control-equipment field, so this is Not documented / needs verification against each plant's actual equipment, not established fact. Full comparison in `data/lofo_weak_facility_diagnosis.json`. **This closes item 6's remaining scope**: the two facilities aren't a training or data problem to solve further, they're an honest limit of what NO2/SO2/VIIRS satellite tiles can currently discriminate.

**12.11 First Track A/Track B fusion attempt (`track_fusion_model.py`, §11 item 3).** RESEARCH_PLAN.md §9's A1→A5 Q-correcting ablation ladder needs an independent ground-truth emissions source that doesn't exist for this project (Climate TRACE is explicitly barred as a training label, §7). Instead of building the literal ladder, this script builds what's legitimately buildable without that source: a Track A + Track B *reliability/trust* fusion, evaluated against — never fit to — Climate TRACE's `bracketed_by_our_interval` outcome. **Question 1: does Track A's `activity_prob_mean` add to Track B's own best uncertainty predictor (`hit_days`, LOO R²=0.212 per `reliability_model.py`) in a 2-feature LOO fit?** No — R² got worse, not better (0.212→0.173): Track A's activity signal carries no incremental information about Track B's own uncertainty beyond what `hit_days` already captures. **Question 2: does a composite fusion trust score (low predicted uncertainty + high activity confidence) track which facilities actually get bracketed by their own uncertainty interval?** Essentially no signal — mean trust for bracketed facilities (0.477) vs. missed facilities (0.470) is a 0.007 gap on a 0–1 scale, noise-level at N=17, though nominally in the expected direction. **Honest conclusion: this specific fusion approach (activity signal as an uncertainty co-predictor) doesn't work** — Track A's detector confidence and Track B's physics uncertainty appear to be largely independent signals that don't reinforce each other in this framing. This doesn't mean fusion is impossible in general, only that this particular self-consistency approach found nothing; the genuine Q-correcting model (RESEARCH_PLAN.md §9) remains unbuilt and still needs an independent ground-truth source. Full results in `data/track_fusion_model_results.json`.

**12.12 Three smaller hardening items (§11 items 7–9).** (a) **Month-stratified `physics_gaussian.py`** (§12.7's flagged general gap): the background population (main annulus and each of the 5 alternate sensitivity-term definitions) is now restricted to only the months the near-plant zone was actually sampled in, when per-sounding dates exist — falling back to the unrestricted background when day data is missing or stratifying would leave too few soundings. Re-ran for all 20 facilities: several point estimates changed substantially (ShriSingajiMalwa +320%, Chhabra −93%, Tirora −90%), confirming the seasonal-imbalance bug was real and not isolated to ShriSingajiMalwa. Re-ran the Climate TRACE comparison at the same N=17: **calibration dropped from 53% (9/17) to 35.3% (6/17)** — a materially worse-looking number, reported honestly rather than hidden. This is not evidence the fix was wrong: several facilities' month-restricted background sample sizes dropped sharply (e.g. Tirora 524→127 soundings, RGundem 80→61), trading a systematic seasonal bias for increased sampling noise — a new caveat, not a regression in methodology. Old (pre-fix) estimates preserved as `data/emission_estimates_v1_no_month_stratification.json` / `data/climate_trace_comparison_v1_no_month_stratification.json`. (b) **Per-plant process locking** in `process_plant.py`: an exclusive, non-blocking `fcntl.flock` on `data/{NAME}.lock`, acquired before the OCO-3 scan begins, so a second process for the same plant fails fast instead of silently racing the first (the actual 2026-08-13 bug — two independent scan loops, not just an unprotected checkpoint write). Verified in isolation: a second lock attempt on the same file correctly raises `OSError`. (c) **Added `requirements.txt`**, pinned to the working conda env (`/opt/miniconda3/envs/co2`, Python 3.11.15). Writing it surfaced an unrelated, previously-undocumented finding: scripts have also been run under a second, separate Python (the macOS python.org 3.14 framework build on `/usr/local/bin/python`) missing `earthaccess`/`xarray`/`geemap` — anything importing those (`process_plant.py`, OCO-3 scripts) would fail under it, though this session's numpy/torch/pandas/ee-only scripts happened to work under either, which is how it went unnoticed. Flagged in `requirements.txt` itself.

**12.13 First test suite (`tests/`, closing item 9's remaining scope).** Added 22 stdlib-`unittest` tests (no new dependency) covering the deterministic building blocks of the pipeline most likely to silently produce a wrong number, per this session's own bug history: `test_physics_gaussian.py` (IME math, the new month-stratification logic), `test_build_3channel.py` (NO2/SO2/VIIRS tile pairing, gap-filling, SO2/VIIRS negative-value clamping), and `test_lofo_track_a.py` (`facility_fold_indices()`, directly guarding against a regression of Week 11's tile-level-leakage bug). Required two small, behavior-preserving refactors to make modules safely importable: `build_3channel.py` gained an `if __name__ == "__main__":` guard (it previously ran its full pairing pipeline at import time), and `lofo_track_a.py`'s inline fold-splitting logic was extracted into a top-level `facility_fold_indices()` function. All 22 tests pass (`python -m unittest discover -s tests -v`, ~0.1s, no network/GPU required). Deliberately excludes integration paths (Earth Engine, OCO-3 downloads, end-to-end model training) — those are exercised by running the actual pipeline scripts, not unit tests. No CI configured to run these automatically.

---

## 13. References

As cited directly in `README.md` and `RESEARCH_PLAN.md` (full bibliographic detail for entries not fully captured during the exploration pass should be verified against `RESEARCH_PLAN.md` §3–4 directly):

- Deb & Das, 2025. arXiv:2502.02083. (NO2 + XCO2 fusion for power-plant emission signal — cited in `README.md`'s citation section.)
- Varon, D. J., et al., 2018. (Source of the IME method and the α=0.5 effective-wind-speed scaling factor used in `physics_gaussian.py`; originally developed for CH4 plume quantification.)
- Nassar, R., et al., 2017. (CO2 point-source mass-balance estimation from OCO-2, cited as the CO2-specific lineage of the IME approach used here.)
- Reuter, M., et al., 2019. (OCO-2/3 CO2 point-source estimation, same mass-balance lineage.)
- Li et al., 2021, GRL. (Cited in `RESEARCH_PLAN.md`'s literature comparison table — specific contribution **Not documented / needs verification** in this pass; consult `RESEARCH_PLAN.md` directly.)
- GMD 2024 (SMARTCARB-related). (CNN-based emission regression, demonstrated in simulation only — cited in `RESEARCH_PLAN.md` as a "don't re-claim as novel" reference.)
- Climate TRACE (climatetrace.org / `api.climatetrace.org`). (Independent India power-sector CO2 benchmark, explicitly not used as ground truth.)
- WRI Global Power Plant Database. (Source of `data/powerplants.csv`, used for plant selection and site deduplication.)

---

*End of master research log. §§1–11 were compiled by reading the full git commit history (`git log --all`), all commit diffs for major milestones, every Markdown/text documentation file in the repository (`README.md`, `RESEARCH_PLAN.md`, `NEXT_STEPS.md`, `WEEK2_LOG.txt`–`WEEK11_LOG.txt`), all 39 Python source files, and the schemas/example contents of the JSON and CSV data files, as of 2026-08-13. §12 was added 2026-08-14 covering a same-day follow-up session (uncommitted at the time of writing — see §12's own note). Sections or claims marked "Not documented / needs verification" indicate information that was not found in the repository during compilation and should not be treated as fact without further checking.*
