# CO2 Emission Estimation

Research and implementation work on detecting and estimating CO2 emissions from coal power plants using satellite data and deep learning. The project has two parallel tracks:

1. **Plant detector** — a small CNN that classifies satellite tiles as "power plant" vs. "not a power plant," built up incrementally by fusing more pollutant/thermal channels (Weeks 2–5, described below).
2. **CO2 enhancement estimation** — a separate, physics-driven track that estimates near-plant XCO2 enhancement directly from OCO-3 soundings, cross-checked against NO2 co-location and wind direction.

## Weekly progress (plant detector track)

Full details and numbers are in `WEEK2_LOG.txt` through `WEEK5_LOG.txt`.

| Week | Input channels | Hard-negative test accuracy | Key finding |
|---|---|---|---|
| 2 | NO2 (Sentinel-5P) | 91.2% (easy negatives) | Baseline plant-vs-forest detector |
| 3 | NO2 | 77.1% | Adding hard negatives (cities/steel/highways) reveals the detector is really a "concentrated combustion detector," not plant-specific |
| 4 | NO2 + SO2 | 79.2% | SO2 fixes city false alarms (Delhi 0.43→0.11), but steel plants (Bhilai, Jamshedpur) remain confused since they also emit SO2 |
| 5 | NO2 + SO2 + VIIRS (MaxFRP) | 79.2% (tied) | VIIRS thermal data measurably reduces false-alarm confidence on the two steel plants specifically (Bhilai 0.59→0.52, Jamshedpur 0.54→0.50), but highways become relatively more confusable, so overall accuracy doesn't move |

## Data sources actually used

* **Sentinel-5P NO2** — `COPERNICUS/S5P/OFFL/L3_NO2` (via Google Earth Engine)
* **Sentinel-5P SO2** — `COPERNICUS/S5P/OFFL/L3_SO2` (via Earth Engine)
* **NASA VIIRS active-fire/thermal** — `NASA/VIIRS/002/VNP14A1`, `MaxFRP` band (via Earth Engine)
* **OCO-3 XCO2 soundings** — `OCO3_L2_Lite_FP` v11r (via NASA `earthaccess`, CO2-enhancement track only)
* **ERA5 wind** — `ECMWF/ERA5/DAILY` (wind-alignment sanity check, CO2-enhancement track only)

All tiles cover the 2019–2020 window, for 5 Indian coal plants (`data/top5_plants.csv`), 5 rural/background points, and 15 hard-negative locations (cities, steel mills, highway corridors — `data/hard_negatives.csv`).

## Plant detector: run order

```
export_monthly.py            # NO2 tiles: data/monthly/{positive,negative}
export_hard_negatives.py     # NO2 tiles: data/monthly/hard_negative
export_so2.py                # SO2 tiles: data/so2/*
export_viirs.py              # VIIRS MaxFRP tiles: data/viirs/*

build_2channel.py            # NO2+SO2 -> data/twoch/*
build_3channel.py            # NO2+SO2+VIIRS -> data/threech/*

train_detector.py            # Week 2/3: 1-channel (NO2) detector
train_2channel.py            # Week 4: 2-channel (NO2+SO2) detector
train_3channel.py            # Week 5: 3-channel (NO2+SO2+VIIRS) detector

gradcam.py                   # Grad-CAM explainability (1-channel model)
analyze_failures.py          # Per-source false-alarm ranking (1-channel model)
analyze_failures_3ch.py      # Per-source false-alarm ranking (3-channel model)
summary_figure.py            # Week 2-5 accuracy comparison chart
```

## CO2 enhancement track: run order

```
process_plant.py <PlantName>   # end-to-end: OCO-3 scan, NO2 co-location, wind check for one plant
co2_enhancement.py             # near-plant vs background XCO2 enhancement (single plant, from saved soundings)
co2_no2_colocation.py          # NO2 heatmap + CO2 soundings overlay
wind_check.py                  # wind-direction vs high-CO2-offset alignment check
summary_co2_figure.py          # cross-plant summary figure
```

Results across the 4 processed plants (Vindhyachal, Sasan, Mundra, Tirora) are in `data/plant_results.json`. Note: 3 of 4 plants show a large mismatch between wind direction and the CO2-enhancement offset, so the enhancement signal should be treated as preliminary, not validated against ground truth.

## Requirements

Python 3, with:

* `torch` (CNN training/inference)
* `numpy`, `pandas`, `matplotlib`
* `earthengine-api` (`ee`) — Sentinel-5P NO2/SO2 and VIIRS tile export
* `earthaccess` — OCO-3 sounding download (CO2-enhancement track only)
* `xarray` — reading OCO-3 NetCDF granules

There is no `requirements.txt` yet; install the above with `pip install torch numpy pandas matplotlib earthengine-api earthaccess xarray`.

## Earth Engine setup

Every script that pulls satellite tiles calls `ee.Initialize(project="<your-project-id>")`. You need:

1. A Google Earth Engine account with a registered Cloud project (register at `https://code.earthengine.google.com/register` if you don't have one).
2. Local auth: `earthengine authenticate`
3. Update the hardcoded project ID in the scripts (`ee.Initialize(project="...")`) to your own project if you don't have access to the one currently checked in.

OCO-3 sounding download additionally requires a NASA Earthdata login (`earthaccess.login(persist=True)`, used only by `process_plant.py`).

## Known limitations

* `physics_gaussian.py` is a planned Gaussian-plume dispersion model to convert CO2 ppm enhancement into an emission-rate estimate — currently unimplemented (empty file).
* The plant detector's "mixed" accuracy (plants vs. all negatives including rural) is not directly comparable across weeks — only the "hard-only" number (plants vs. hard negatives, balanced) is tracked consistently and shown in `summary_figure.py`.
* Dataset is small (600 tiles total per detector variant, ~240 in the balanced hard-only split), so accuracy differences within a few points should be treated as noisy.

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
