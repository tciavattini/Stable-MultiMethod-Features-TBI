# Machine Learning-driven biomarker discovery for stratifying treatment response in tick-borne illness

> Reproducible machine learning pipeline for identifying stable and clinically relevant predictors of treatment response in heterogeneous patient cohorts.

# Abstract
Lyme disease and related tick-borne infections can cause persistent symptoms even after antibiotic treatment. Currently, clinicians cannot reliably predict which patients will respond well to therapy. We analysed clinical and laboratory data from 301 patients collected before treatment. By systematically evaluating 149 patient characteristics across 100 repeated analyses using multiple methods (over 700,000 evaluations), we identified 22 features that consistently distinguished responders from non-responders. Patients who responded well tended to report greater physical symptom burden at baseline, whereas non-responders reported better mood and overall well-being. Five features were consistently identified across all analytical approaches. These findings suggest that baseline symptom and immune profiles may help identify patients more likely to benefit from treatment and support more informed clinical decision-making.

---

## Overview

Machine learning models applied to clinical datasets often suffer from instability and poor reproducibility, particularly in small and heterogeneous cohorts such as tick-borne illness.

This repository implements a stability-aware machine learning framework designed to:

- identify robust baseline predictors of treatment response
- reduce dependence on specific algorithms or random data splits
- improve reproducibility of feature selection in clinical ML

The approach combines multi-method feature selection, consensus ranking, and rigorous cross-validation to extract stable biomarkers.

---

## Pipeline Overview

The full workflow includes:

1. Data preprocessing (imputation, scaling)
2. Patient stratification (high vs non-responders)
3. Multi-method feature selection
4. Stability-based feature aggregation
5. Model training (RF, SVM, Logistic Regression, KNN)
6. Performance evaluation with cross-validation

![Pipeline overview](figures/main/pipeline_workflow.jpg)

## Main Result

![Stable features](figures/main/stable_features.jpg)


---

## Repository Structure

```
.
├── dataset/            # Raw dataset
├── src/                # Core pipeline scripts
├── figures/            # Figures used in README and paper
├── requirements.txt    # Python dependencies
├── LICENSE
└── README.md
```


---

## Installation

Clone the repository:

```bash
git clone https://github.com/tciavattini/Stable-MultiMethod-Features-TBI.git
cd Stable-MultiMethod-Features-TBI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

The analysis pipeline is organized into sequential scripts.

Suggested execution order:

```bash
src/01_feature_engineering_pipeline.ipynb
python src/02_outcome_classification.py
python src/03_prepare_data.py
python src/sanity_check.py
python src/04_stability_top30.py
# nohup python stability_top30.py > logs/stability_run_$(date +%Y%m%d_%H%M).log 2>&1 &
# echo $! > logs/stability.pid
# to run cross_validation you need the files data_loading.py and stability_top30

python src/05_multifeature.py
python src/06_alt_feature_selection.py
python src/07_alt_fs_classification.py
python src/08_phenotype_analysis.py
python src/09_sensitivity_analyses.py
```

All experiments are executed with fixed random seeds to ensure reproducibility.

---

## Data Availability

Due to privacy and ethical restrictions associated with clinical data, the full original dataset is not publicly released.

The repository includes the code used for preprocessing, feature selection, modeling, and figure generation. Any shared data files are provided only in accordance with anonymization and data-sharing constraints.

---

## Results

### Stable Feature Identification

The proposed framework identifies features that are consistently selected across multiple methods and data splits.

### Model Performance

Predictive performance was evaluated using cross-validation across multiple classifiers.

---

## Key Contributions

- Stability-aware feature selection framework for clinical ML
- Multi-method consensus ranking strategy
- Robust identification of reproducible biomarkers
- Fully reproducible pipeline aligned with TRIPOD+AI principles

---

## Reproducibility

- Python version: 3.10
- Fixed random seeds for all stochastic processes
- Deterministic pipeline execution
- Full code and environment specification provided

---

## Citation

If you use this repository, please cite the associated manuscript:

> Ciavattini T, et al. *Stability-aware machine learning identifies reproducible baseline predictors of treatment response in tick-borne illness.* Manuscript under submission.

---

## License

- Code is licensed under the MIT License (see `MIT license`)
- Dataset is licensed under CC BY-SA 4.0 (see `CC-BY-SA-4.0 license`)

---

## Contact

**Teresa Ciavattini**
Sorbonne Université / SCAI – Sorbonne Center for Artificial Intelligence
Université de Technologie de Compiègne

# Dataset Description

This dataset contains anonymized clinical variables used in the study.

## Content
- Baseline biomarkers
- Clinical symptoms
- Outcome variables

## Preprocessing
- Missing values handled via imputation
- Standardization applied

## Privacy
All data have been anonymized and do not contain personally identifiable information.
