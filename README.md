# CO2 Emission Estimation (Now Validated Against Real Ground Truth)

## Live Demo

**https://co2-estimation-project.streamlit.app/**

Interactive Streamlit app: the 30-plant Q-vs-CEA comparison map and the four Week 13 Rihand-style diagnostic checks per facility (§5.2.7/§5.2.9 of `RESEARCH_PAPER.md`). Deployed via Streamlit Community Cloud directly from this repo's `main` branch — everything shown is read from this repo's own committed result files at runtime, nothing is recomputed live. (Hugging Face Spaces was attempted first, per the project blueprint's recommendation, but its free tier only supports Static Spaces, which cannot run a Python backend; see `WEEK25_LOG.txt` for the full deployment story.)

Research and implementation work on detecting and estimating CO2 emissions from coal power plants using satellite data and deep learning. The project has two parallel tracks:

1. **Plant detector (Track A)** — a small CNN that classifies satellite tiles as "power plant" vs. "not a power plant," built up incrementally by fusing more pollutant/thermal channels (Weeks 2–5, described below). Since expanded to 20 positive-class facilities; exhaustive leave-one-facility-out recall is currently **69.1%**.
2. **CO2 enhancement estimation (Track B)** — a physics-driven track that estimates near-plant XCO2 enhancement and a per-plant emission rate directly from OCO-3 soundings, cross-checked against NO2 co-location and wind direction. Now benchmarked against **two** independent references: Climate TRACE (satellite-inferred) and, more importantly, India's Central Electricity Authority CO2 Baseline Database — a real, non-satellite ground-truth source computed from each plant's reported fuel consumption. A working, cross-validated correction model exists against this real ground truth (MAE 1.01→0.902 in log-ratio space).

For the full week-by-week narrative through Week 7 (superseded by everything below, kept only as historical record), see the "Original Week 2–7 walkthrough" section near the bottom of this file, or `PROJECT_RESEARCH_DOCUMENTATION.md` §5.2/§6.3 for the exhaustive version. Everything from here on describes the current state.

## Track A: plant detector — current state

Expanded from the original 4-facility positive class to 20 (`data/candidate_plants.csv`), each with 24 months of NO2+SO2+VIIRS tile depth (2019+2020). Tile-level accuracy numbers (Week 2–7, up to 81.2%) turned out to substantially overstate real generalization — a facility-level split dropped this to 67.3%, and even that was an artifact of an easy single random split. The number that matters is **exhaustive leave-one-facility-out (LOFO) recall**, which holds out each of the 20 facilities in turn and trains a fresh model each time:

| Stage | LOFO mean recall | Note |
|---|---|---|
| Original (12-tile depth per facility) | 47.2% | Roughly a coin flip on average, 0–100% by facility |
| Light data augmentation (no new data) | 46.8% | No improvement — rules out training-diversity as the driver |
| 24-tile depth (2nd data year exported) | **69.1%** | Real additional satellite observations closed most of the gap |

Two facilities (Kudgi, ShriSingajiMalwa) still show 0% LOFO recall after the fix — diagnosed as a genuine satellite-observability limit (their raw NO2/SO2 signal sits at or below background-noise levels in every month checked), not a fixable modeling gap.

**Run order:**
```
export_monthly.py                 # NO2 tiles: data/monthly/{positive,negative}
export_hard_negatives.py          # NO2 tiles: data/monthly/hard_negative
export_so2.py / export_viirs.py   # SO2 / VIIRS MaxFRP tiles
export_new_positive_tiles.py      # 2020 tiles for the 16 facility-expansion plants
export_new_positive_tiles_2019.py # 2019 tiles for the same, closing a data-quantity LOFO gap

build_2channel.py / build_3channel.py   # channel-stack tiles -> data/{twoch,threech}/*

train_detector.py / train_2channel.py / train_3channel.py   # 1/2/3-channel detectors

lofo_track_a.py                # exhaustive leave-one-facility-out harness (21+ folds)
lofo_track_a_aug.py            # same, with light spatial augmentation
lofo_recall_correlates.py      # what predicts per-facility LOFO recall
diagnose_lofo_weak_facilities.py   # why Kudgi/ShriSingajiMalwa still fail (signal-strength diagnosis)

gradcam.py / analyze_failures.py / analyze_failures_3ch.py   # explainability, false-alarm ranking
extract_activity_signal.py / compare_activity_signal_checkpoints.py   # Track A -> Track B activity signal
summary_figure.py / evaluation_figures.py   # summary figures
```

## Track B: CO2 emission estimation — current state

All 20 committed candidate plants processed; 17–18 produce a usable `physics_ime.py` estimate (a few excluded for a genuine 0-near-sounding OCO-3 coverage gap). The IME (Integrated Mass Enhancement) pipeline now uses per-overpass wind conditioning, a three-term uncertainty budget (wind, IME-sampling, background-definition sensitivity), and month-stratifies its near/background comparison (fixing a real seasonal-sampling-bias bug found via a facility with spuriously negative CO2 enhancement).

**Benchmarked against two independent references**, which measure different things:
- **Climate TRACE** (satellite-thermal-proxy inferred, itself unvalidated for India) — 35.3% of estimates fall within their own stated uncertainty interval.
- **CEA's CO2 Baseline Database** (Government of India, bottom-up from each plant's reported fuel consumption — genuine ground truth, not another satellite proxy) — 47.1% bracketing, MAE 1.01 in log-ratio space. A single-feature, leave-one-out-cross-validated correction (using background XCO2 noise) reduces this to **MAE 0.902**, a real ~11% improvement — the project's first correction model validated against actual reported emissions.

**Run order:**
```
process_plant.py <PlantName>   # end-to-end: OCO-3 scan, NO2 co-location, wind check for one plant
co2_enhancement.py             # near-plant vs background XCO2 enhancement (single plant, from saved soundings)
co2_no2_colocation.py          # NO2 heatmap + CO2 soundings overlay
wind_check.py                  # wind-direction vs high-CO2-offset alignment check
physics_ime.py            # IME mass-balance emission-rate estimate (t CO2/yr), per-overpass wind + 3-term uncertainty + month-stratified background
reliability_model.py           # does activity signal / wind alignment predict physics's own uncertainty?
track_fusion_model.py          # self-consistency Track A/B fusion attempt (tested negative, kept for the record)

pull_climate_trace.py          # benchmark vs. Climate TRACE (independent, not ground truth)
pull_cea_ground_truth.py       # pull real ground-truth emissions (CEA CO2 Baseline Database)
q_correction_model.py          # the actual Q-correcting model, validated against CEA ground truth

diagnose_talcher.py / diagnose_negative_enhancement.py / diagnose_shrisingajimalwa.py   # targeted uncertainty/anomaly diagnoses
summary_co2_figure.py          # cross-plant summary figure

baseline_capacity.py                     # headline comparison: capacity_mw alone vs IME Q vs both, LOO against CEA (reads data/q_correction_model_results.json only, no network access needed)
physics_gaussian_crosssection.py         # alternate Q estimator (Gaussian cross-sectional flux), benchmarked against IME on the same soundings (reads data/*_soundings.npz + plant_results.json/cea_ground_truth_2020_21.json/emission_estimates.json, no network access needed)
```

Full facility-level results are in `data/plant_results.json`, `data/emission_estimates.json`, `data/climate_trace_comparison.json`, `data/cea_ground_truth_2020_21.json`, `data/q_correction_model_results.json`, `data/baseline_capacity_results.json`, and `data/gaussian_crosssection_results.json`.

## Track B, DL extension: segmentation-only U-Net (confirmed negative real-tile result)

A segmentation-only deep-learning extension (Weeks 20–23, `RESEARCH_PAPER.md` §9) toward the project's original blueprint architecture. Trains a compact U-Net on simulated (XCO2 tile, plume mask) pairs from a physics-based simulator (reusing `plume_model.py`'s validated Gaussian plume physics and Week 20's real-OCO-3-orbital-geometry sampler) since no real labeled dataset exists at this scale. Reaches moderate quality on simulated data (positive-tile median Dice 0.29) but its predictions on real single-overpass OCO-3 tiles consistently track satellite coverage patterns rather than true plume shape — confirmed across three independent fixes, a closed negative result, not an open question.

**Run order:**
```
simulate_training_pairs.py      # generates data/simulated_train/simulated_tiles.npz (already committed; no need to regenerate to reproduce below)
train_unet_segmentation.py      # trains UNetSmall (488K params) on the simulated pairs -> unet_segmentation.pt, data/unet_segmentation_results.json
sanity_check_real_tiles.py      # illustrative-only: runs the trained checkpoint on real Vindhyachal/Rihand OCO-3 tiles (no accuracy metric -- no real ground-truth mask exists)
```

`unet_segmentation.pt` (the trained checkpoint) is **not** committed — `*.pt` is gitignored for this file, unlike the Track A detector checkpoints, which are force-added exceptions. `sanity_check_real_tiles.py` therefore requires running `train_unet_segmentation.py` first on a fresh clone. This is deterministic (seed 42, `torch`/`numpy` fixed) and fast (60 epochs, ~2 min on an M1 MacBook Air's `mps` backend; `torch` — already in `requirements.txt` for Track A's CNN, no new dependency): a fresh-clone run reproduces the paper's exact positive-tile median Dice (0.286) and the real-tile coverage figures (13.3%/11.2% for Vindhyachal/Rihand) reported in §9.5–§9.6. No other new package was needed for this extension.

## Requirements

Python 3, with:

* `torch` (CNN training/inference)
* `numpy`, `pandas`, `matplotlib`
* `earthengine-api` (`ee`) — Sentinel-5P NO2/SO2 and VIIRS tile export
* `earthaccess` — OCO-3 sounding download (CO2-enhancement track only)
* `xarray` — reading OCO-3 NetCDF granules

`requirements.txt` now exists (added later than this README), pinned to the project's working conda environment — `pip install -r requirements.txt`.

## Earth Engine setup

Every script that pulls satellite tiles calls `ee.Initialize(project="<your-project-id>")`. You need:

1. A Google Earth Engine account with a registered Cloud project (register at `https://code.earthengine.google.com/register` if you don't have one).
2. Local auth: `earthengine authenticate`
3. Update the hardcoded project ID in the scripts (`ee.Initialize(project="...")`) to your own project if you don't have access to the one currently checked in.

OCO-3 sounding download additionally requires a NASA Earthdata login (`earthaccess.login(persist=True)`, used only by `process_plant.py`).

## Testing

A basic test suite exists (`tests/`, 46 stdlib-`unittest` tests) covering the pipeline's most bug-prone deterministic logic — `physics_ime.py`'s IME math and month-stratification, `build_3channel.py`'s tile pairing, `lofo_track_a.py`'s facility fold-splitting, `plume_model.py`'s Gaussian-plume physics, and `validate_quality_gate.py`'s LOO-split/permutation-shuffle mechanics. Run with:

```
python -m unittest discover -s tests -v
```

Doesn't cover Earth Engine, OCO-3 downloads, or model training end-to-end. Runs automatically on every push/PR via `.github/workflows/tests.yml`. See `tests/README.md`.

## Known limitations

* Track A's exhaustive LOFO recall (69.1%) is still meaningfully below tile-level or single-split numbers you might see quoted elsewhere for similar work — treat any single-split facility-level recall figure with suspicion unless it comes from a full leave-one-facility-out evaluation, since this project's own experience shows a single split can differ from the true rate by up to 2×. Two facilities (Kudgi, ShriSingajiMalwa) are diagnosed as a genuine satellite-observability limit, not a training gap.
* Track B's Q-correcting model (against real CEA ground truth) is still indicative, not production-validated: N=17, a single-feature linear correction, one fiscal year (FY2020-21) of ground-truth data compared against 2020-calendar-year Track B estimates (not an exact year match), and one CEA database version.
* Climate TRACE and CEA ground-truth bracketing rates (35.3% and 47.1% respectively) measure different things and should not be conflated — Climate TRACE is itself a satellite-inferred estimator (independent benchmark, not ground truth); CEA is bottom-up from reported fuel consumption (genuine ground truth). Neither number supersedes the other.
* CI (`.github/workflows/tests.yml`) only runs the fast unit-test suite on push/PR; it does not cover Earth Engine, OCO-3 downloads, or model training end-to-end — those still require running the pipeline scripts manually before trusting a change.
* Scripts have historically been run under two different Python installations (a conda env at `/opt/miniconda3/envs/co2` with everything needed, and a separate macOS framework Python missing `earthaccess`/`xarray`/`geemap`) — use `requirements.txt` and the conda env to avoid import errors on `process_plant.py` or other OCO-3-touching scripts.
* Track B's segmentation-only U-Net extension (§9, Weeks 20–23) is a **confirmed negative result for real-tile transfer**: the model does not transfer from simulated to real single-overpass OCO-3 tiles, across three independently-targeted fixes. `unet_segmentation.pt` is not committed (unlike Track A's detector checkpoints) — verified (Week 24 fresh-clone audit) that `train_unet_segmentation.py` reproduces it deterministically in ~2 minutes, so this is a documented gap, not a broken reproduction path.

## Original Week 2–7 walkthrough (historical record, superseded)

<details>
<summary>Expand for the project's original state as of Week 7 — kept for the record, not current. See the sections above for current numbers.</summary>

### Weekly progress (plant detector track)

Full details and numbers are in `WEEK2_LOG.txt` through `WEEK7_LOG.txt` (`WEEK6_LOG.txt` covers the CO2-enhancement track, not the detector).

| Week | Input channels | Hard-negative test accuracy | Key finding |
|---|---|---|---|
| 2 | NO2 (Sentinel-5P) | 91.2% (easy negatives) | Baseline plant-vs-forest detector |
| 3 | NO2 | 77.1% | Adding hard negatives (cities/steel/highways) reveals the detector is really a "concentrated combustion detector," not plant-specific |
| 4 | NO2 + SO2 | 79.2% | SO2 fixes city false alarms (Delhi 0.43→0.11), but steel plants (Bhilai, Jamshedpur) remain confused since they also emit SO2 |
| 5 | NO2 + SO2 + VIIRS (MaxFRP) | 79.2% (tied) | VIIRS thermal data measurably reduces false-alarm confidence on the two steel plants specifically (Bhilai 0.59→0.52, Jamshedpur 0.54→0.50), but highways become relatively more confusable, so overall accuracy doesn't move |
| 7 | NO2 + SO2 + VIIRS, +5 more highway hard negatives | 81.2% | Expanding the highway hard-negative set from 5 to 10 (geographically spread, not clustered) drops both of Week 5's worst false alarms (hwy_AhmedabadVadodara 0.46→0.38, hwy_GTRoad_UP 0.44→0.34) — first accuracy improvement since Week 4 |

This tile-level accuracy turned out to substantially overstate real generalization — see the current-state Track A section above.

### Data sources originally used

* **Sentinel-5P NO2** — `COPERNICUS/S5P/OFFL/L3_NO2` (via Google Earth Engine)
* **Sentinel-5P SO2** — `COPERNICUS/S5P/OFFL/L3_SO2` (via Earth Engine)
* **NASA VIIRS active-fire/thermal** — `NASA/VIIRS/002/VNP14A1`, `MaxFRP` band (via Earth Engine)
* **OCO-3 XCO2 soundings** — `OCO3_L2_Lite_FP` v11r (via NASA `earthaccess`, CO2-enhancement track only)
* **ERA5 wind** — `ECMWF/ERA5/DAILY` (wind-alignment sanity check, CO2-enhancement track only)

All tiles originally covered the 2019–2020 window, for 5 Indian coal plants (`data/top5_plants.csv`), 5 rural/background points, and 20 hard-negative locations (cities, steel mills, highway corridors — `data/hard_negatives.csv`). Since expanded to 20 positive-class facilities at full 24-month depth — see the current-state sections above.

### Original results (4 plants, Week 6)

Results across the 4 originally-processed plants (Vindhyachal, Sasan, Mundra, Tirora) are in `data/plant_results.json` (now holds 20). Emission-rate estimates (Week 6, `physics_ime.py`) originally in `data/emission_estimates.json`: Vindhyachal 44.6 Mt/yr and Sasan 37.2 Mt/yr landed in the physically expected range for large baseload coal plants, a sanity check on the method's magnitude. Tirora's estimate (3.2 Mt/yr) looked too low relative to its capacity, most likely due to thin OCO-3 coverage (5 hit-days / 671 soundings). Mundra was skipped entirely (only 57 soundings total). See `WEEK6_LOG.txt` for the full original writeup, including a wind-speed bug found and fixed during that work — since further corrected (per-overpass wind, month-stratified background) and validated against real ground truth, per the current-state Track B section above.

</details>

## Citation

If you use this repository in your research, please cite the associated paper:

```bibtex
@misc{deb2025improvingpowerplantco2,
  title={Improving Power Plant CO2 Emission Estimation with Deep Learning and Satellite/Simulated Data},
  author={Dibyabha Deb and Kamal Das},
  year={2025},
  eprint={2502.02083},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2502.02083}
}
```

## License

Add a license file if you want to specify how others may use this code.
