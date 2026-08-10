# CO2 Emission Estimation

This repository contains research and implementation work for **CO2 emission estimation using satellite imagery, simulated data, and deep learning**. The project focuses on improving the accuracy of emission-rate estimation for power plants by combining multiple data sources and training an encoder-decoder style model.

## Overview

Power plants are major point sources of CO2 emissions. Estimating emissions accurately from space is challenging because of noisy observations, atmospheric variation, and limited labeled data. This project explores a data-driven approach that integrates:

* **Sentinel-5P NO2 data**
* **OCO-2 / OCO-3 satellite observations**
* **Simulated data**
* **A customized U-Net / encoder-decoder model**

The goal is to improve estimation accuracy and generalization across different spatial and temporal conditions.

## Key Ideas

* Expand the available training data using satellite-derived information.
* Combine simulated and real satellite observations.
* Train deep learning models for emission-rate prediction.
* Evaluate results on simulated, satellite, and combined settings.

## Repository Structure

The repository is organized around three main workflows:

* **Simulated** experiments
* **Satellite** experiments
* **Combined** experiments

An additional **EDA** script is included for exploratory analysis.

## Files and Workflow

To reproduce the main experiments, run the scripts in this order:

1. `org_model_train_eval.py`
2. `shuf_model_train_eval.py`
3. `sim_eval.py`
4. `error_calc.py`
5. `sat_cur.py`
6. `sat_eval.py`
7. `error_calc.py`
8. `comb.py`
9. `model.py`
10. `error_calc.py`
11. `eda.py`

## Requirements

The code is written in **Python** and mainly uses **TensorFlow**.

You may also need common scientific Python packages such as:

* NumPy
* Pandas
* Matplotlib
* SciPy
* scikit-learn

Install the project dependencies with:

```bash
pip install -r requirements.txt
```

## How to Run

Clone the repository:

```bash
git clone https://github.com/devashishpandey044-code/co2-emission-estimation.git
cd co2-emission-estimation
```

Then run the scripts according to the experiment you want to reproduce.

## Results

This project aims to improve CO2 emission estimation accuracy by leveraging both simulated and real satellite data. The methodology is designed to support more reliable estimation of emissions from large point sources such as power plants.

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

## Acknowledgements

This repository is part of ongoing work on CO2 emission estimation using deep learning and satellite data.
