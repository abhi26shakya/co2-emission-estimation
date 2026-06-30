# CO₂ Emission Estimation from Satellite Imagery

Detecting and estimating CO₂ emissions from power plants using
satellite data (Sentinel-5P, OCO-2/3) with a physics baseline and
a deep-learning pipeline.

## Status (Week 1)
- Physics baseline (Gaussian-plume inversion) reproducing Nassar 2017 — ~12% error
- Deep-learning spine: U-Net segmentation → CNN emission regressor (PyTorch)
- MC-dropout uncertainty estimation
- First real Sentinel-5P NO₂ map over the Singrauli coal complex

## Files
- `physics_gaussian.py` — Gaussian-plume physics baseline
- `dl_spine.py` — U-Net + CNN deep-learning pipeline
- `first_no2.py` — pulls real NO₂ data via Google Earth Engine
