Stability-aware Machine Learning for Robust Biomarker Discovery in Tick-borne Illness

Reproducible machine learning pipeline for identifying stable and clinically relevant predictors of treatment response in heterogeneous patient cohorts.

⸻

Overview

Machine learning models applied to clinical datasets often suffer from instability and poor reproducibility, particularly in small and heterogeneous cohorts such as tick-borne illness.

This repository implements a stability-aware machine learning framework designed to:
	•	identify robust baseline predictors of treatment response
	•	reduce dependence on specific algorithms or random data splits
	•	improve reproducibility of feature selection in clinical ML

The approach combines multi-method feature selection, consensus ranking, and rigorous cross-validation to extract stable biomarkers.

⸻

Pipeline Overview

The full workflow includes:
	1.	Data preprocessing (imputation, scaling)
	2.	Patient stratification (high vs non-responders)
	3.	Multi-method feature selection
	4.	Stability-based feature aggregation
	5.	Model training (RF, SVM, Logistic Regression, KNN)
	6.	Performance evaluation with cross-validation


⸻

Repository Structure

.
├── data/               # Raw and processed datasets
├── src/                # Core pipeline scripts
├── notebooks/          # Exploratory analyses (optional)
├── figures/            # Figures used in README and paper
├── results/            # Outputs (tables, metrics)
├── requirements.txt    # Python dependencies
├── LICENSE
└── README.md


⸻

Installation

Clone the repository:

git clone https://github.com/tciavattini/Stable-MultiMethod-Features-TBI.git
cd Stable-MultiMethod-Features-TBI

Install dependencies:

pip install -r requirements.txt


⸻

Usage

To reproduce the full pipeline:

python src/run_pipeline.py

Alternatively, run step-by-step:

python src/preprocessing.py
python src/feature_selection.py
python src/train_models.py
python src/evaluation.py

All experiments are executed with fixed random seeds to ensure reproducibility.

⸻

Data Availability

Due to privacy and ethical constraints related to clinical data:
	•	The original dataset is not publicly available
	•	A processed / anonymized version (if applicable) is provided in data/
	•	Access to the full dataset may be granted upon reasonable request

⸻

Results

Stable Feature Identification

The proposed framework identifies features that are consistently selected across multiple methods and data splits.


⸻

Model Performance

Predictive performance was evaluated using cross-validation across multiple classifiers.


⸻

Key Contributions
	•	Stability-aware feature selection framework for clinical ML
	•	Multi-method consensus ranking strategy
	•	Robust identification of reproducible biomarkers
	•	Fully reproducible pipeline aligned with TRIPOD+AI principles

⸻

Reproducibility
	•	Python version: 3.10
	•	Fixed random seeds for all stochastic processes
	•	Deterministic pipeline execution
	•	Full code and environment specification provided

⸻

Citation

If you use this work, please cite:

@article{ciavattini2026,
  title={Stability-aware machine learning identifies reproducible baseline predictors of treatment response in tick-borne illness},
  author={Ciavattini, Teresa et al.},
  journal={Communications Medicine},
  year={2026}
}


⸻

License

This project is licensed under the MIT License. See the LICENSE file for details.

⸻

Contact

Teresa Ciavattini
Sorbonne Université / SCAI – Sorbonne Center for Artificial Intelligence
Université de Technologie de Compiègne

