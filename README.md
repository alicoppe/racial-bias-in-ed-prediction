# Assessing the Impact of Race on Machine Learning Models Predicting Hospital Admissions from the Emergency Department

This repository accompanies the paper “Assessing the Impact of Race on Machine Learning Models Predicting Hospital Admissions from the Emergency Department” by Aidan Licoppe, Lucy An, David L. Buckeridge, Antony Robert, and James M.G. Tsui (McGill University & McGill University Health Centre).

## Overview
We analyse how explicitly encoding race influences multilayer perceptron (MLP) models that predict emergency-department (ED) admissions. Using the MIMIC-IV v2.2 ED subset, we train paired networks with identical architectures:
- **Baseline model** – structured and text features without race.
- **Race-aware model** – the same feature set plus one-hot encoded race indicators.

We compare calibration, probability shifts, and statistical significance across racial groups, and run robustness checks with balanced and randomised cohorts.

## Repository Layout
```
helper_functions/               Shared Python package (model builders, training utilities, callbacks, plotting helpers)
00_data_exploration.ipynb       Clean triage data, standardise race labels, export `combined_data.csv`
01_basic_model_training.ipynb   Benchmark classical models and train the first text+numeric MLP
02_class_imbalance.ipynb        Explore resampling, cost-sensitive training, uncertainty, and calibration metrics
03_baseline_metrics_race.ipynb  Generate paired MLP predictions (without vs with race) and store outputs under `experiments_data/`
04_hyperparameter_tuning.ipynb  Tune architecture / loss choices for the race-aware models
05_multiple_rerun_generation.ipynb  Automate repeated training runs and archive prediction sets
06_effect_size_calc.ipynb       Aggregate prediction archives into summary tables and effect sizes
A1_confounding_variables_analysis.ipynb  Merge insurance/age attributes; export `combined_w_age_and_insurance.csv`
assessing_equity.ipynb          Comprehensive fairness analysis with calibration curves, distributional comparisons, and statistical tests
A2_extra_statistical_analysis.r Additional R-based mixed-effects and non-parametric tests on exported CSVs
experiments_data/               Generated masks, prediction arrays, and summary CSVs (ignored by git; recreate via notebooks)
pyproject.toml, requirements.txt  Dependency manifests for uv/pip installs
```

Raw MIMIC-IV files should reside beneath `mimic-iv-2.2/` (mirroring PhysioNet’s structure). The directory is untracked and must be supplied locally by credentialed users.

## Data Requirements
1. Request access to [MIMIC-IV v2.2](https://physionet.org/content/mimiciv/2.2/).
2. Place the ED CSVs under `mimic-iv-2.2/` (e.g., `mimic-iv-2.2/ed/triage.csv`, `mimic-iv-2.2/data/admissions.csv`).
3. Produce project-specific intermediates by running:
   - `00_data_exploration.ipynb` → `combined_data.csv`
   - `A1_confounding_variables_analysis.ipynb` → `combined_w_age_and_insurance.csv`

## Environment Setup
Target Python 3.10+. Dependencies are declared in both `pyproject.toml` (PEP 621, uv) and `requirements.txt` (pip). Notable libraries: TensorFlow 2.15, scikit-learn ≥1.4, imbalanced-learn, SHAP, matplotlib-inline, ipykernel.

### Using uv (recommended)
```
uv sync
source .venv/bin/activate
```
`uv sync` creates `.venv/` and installs the project package (`helper_functions`) in editable mode alongside notebook dependencies.

### Using pip
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

JupyterLab and ipykernel are included, so VS Code / Jupyter should detect the `.venv` kernel automatically once the environment is activated.

## Reproducing the Analyses
Follow the notebooks in the numbered order:
1. `00_data_exploration.ipynb` – clean and explore triage data; export the baseline combined dataset.
2. `A1_confounding_variables_analysis.ipynb` – merge external covariates (insurance, age) and save `combined_w_age_and_insurance.csv`.
3. `01_basic_model_training.ipynb` – establish classical baselines and the initial MLP architecture.
4. `02_class_imbalance.ipynb` – assess imbalance mitigation strategies and model uncertainty.
5. `03_baseline_metrics_race.ipynb` – retrain paired models without/with race, recording predictions to `experiments_data/`.
6. `04_hyperparameter_tuning.ipynb` – sweep architecture and loss settings for the race-aware models.
7. `05_multiple_rerun_generation.ipynb` – orchestrate repeated training runs for robustness experiments.
8. `06_effect_size_calc.ipynb` – aggregate outputs into effect-size and summary tables for the manuscript.
9. `assessing_equity.ipynb` – perform the comprehensive fairness analysis used in the paper.
10. Optionally run `A2_extra_statistical_analysis.r` for additional mixed-effects and non-parametric testing in R.

Generated artifacts (prediction archives, masks, summary CSVs) live under `experiments_data/`. Because the folder is ignored by git, rerun the notebooks to regenerate its contents when cloning the repository.

## Helper Package
The `helper_functions` package contains reusable logic used across notebooks:
- `helper_functions.nn.create_nn` – build text + structured MLP architectures.
- `helper_functions.notebook_utils` – calibration loss, callbacks, retraining loops, balancing utilities, plotting helpers, effect-size routines.
- `helper_functions.testing.cross_validate` – cross-validation wrapper for neural models.

Import from this package rather than duplicating code cells when creating new experiments.

## Citation
A BibTeX entry will be provided post-publication. Please acknowledge “Assessing the Impact of Race on Machine Learning Models Predicting Hospital Admissions from the Emergency Department” when referencing this work.

## License and Data Use
Code is released for academic use. The MIMIC-IV dataset remains under the PhysioNet Credentialed Health Data License—do not redistribute raw or patient-level derivative data. Ensure compliance with MIMIC-IV data-use agreements and institutional review policies.
