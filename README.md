# Machine Learning-driven biomarker discovery for stratifying treatment response in tick-borne illness

**Teresa Ciavattini, Marc Shawky, Séverine Padiolleau-Lefèvre, Florian De Vuyst, Gordana Avramovic, John Shearer Lambert**

> Ciavattini T et al. *Machine Learning-driven biomarker discovery for stratifying treatment response in tick-borne illness* Manuscript under submission.

---

## Abstract

**Background.** Lyme disease and tick-borne co-infections affect hundreds of thousands of individuals annually, with 10–36% reporting persistent symptoms after antibiotic treatment. Predicting which patients will respond to treatment before therapy begins remains an unsolved problem, as no established baseline predictors exist. A key methodological barrier is that in small clinical datasets, feature selection results are highly sensitive to the choice of algorithm and data partitioning, often producing findings that fail to replicate.

**Methods.** We analysed baseline clinical and laboratory data from 301 patients with tick-borne infections evaluated at a tertiary referral centre. Treatment response was defined using a composite score integrating longitudinal changes across four symptom domains. Patients were stratified into high responders (n=71) and non-responders (n=70) based on extreme tertiles. We applied a resampling-based feature evaluation framework that assessed the stability of discriminative features across five scoring criteria, 100 independent data partitions, and four methodologically distinct selection approaches.

**Results.** 22 baseline features were reproducibly selected across resampling iterations, with five confirmed by all four methods: CD8+ T cells, severe fatigue severity, overall symptom severity, muscle pain severity, and mood severity. Models trained on the stable feature set outperformed those using all 149 features or random subsets of equal size (best AUC: 0.710). High responders were characterised by greater somatic symptom burden at baseline, while non-responders exhibited better mood and self-rated wellbeing.

**Conclusions.** Stability-aware feature evaluation can identify reproducible baseline characteristics that distinguish treatment responders from non-responders in tick-borne illness. If validated in independent cohorts, these findings could inform pre-treatment patient stratification.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Pipeline Overview](#pipeline-overview)
3. [Data](#data)
4. [Repository Structure](#repository-structure)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Results](#results)
8. [Key Contributions](#key-contributions)
9. [Reproducibility](#reproducibility)
10. [Licenses](#licenses)
11. [Feedback](#feedback)
12. [References](#references)
13. [Citation](#citation)

---

## Introduction

Lyme disease and associated tick-borne co-infections represent a growing public health burden, with an estimated 500,000 new cases per year in the United States and comparable incidence across endemic European regions. While antibiotic therapy is effective for most patients, 10% to 36% report persistent or recurring symptoms following treatment — including severe fatigue, musculoskeletal pain, neurological dysfunction, and impaired mood — a condition formally classified as post-treatment Lyme disease syndrome (PTLDS).

The inability to identify at baseline which patients are likely to respond to treatment remains an unsolved clinical problem. Patients unlikely to respond might benefit from earlier escalation; likely responders could be spared unnecessary intensification. Machine learning offers advantages over conventional univariate or stepwise approaches for identifying such discriminators, as it can evaluate multiple candidate features simultaneously while accounting for their interactions.

However, a critical methodological barrier limits clinical translation: in small datasets where candidate features approach or exceed the number of patients, feature selection results are highly sensitive to the choice of algorithm and data partitioning, often producing findings that fail to replicate. Standard sparsity-promoting methods frequently yield unstable feature sets across independent cohorts.

This repository implements a **stability-aware machine learning framework** that addresses these limitations by aggregating evidence across 100 independent resampling iterations and multiple selection methods, formally quantifying reproducibility using the Nogueira stability index, Jaccard similarity, and Spearman rank concordance, and retaining only features that are consistently discriminative regardless of algorithmic choice or sampling variation.

The full analytical pipeline is publicly available to support independent replication, in alignment with TRIPOD+AI reporting principles.

---

## Pipeline Overview

The complete analytical workflow is organised into five stages:

1. **Feature engineering** — Raw survey and laboratory data preprocessing, recoding of categorical variables, aggregation into domain-specific symptom counts; yields a final matrix of 149 baseline variables
2. **Outcome definition and stratification** — Composite treatment response score derived from longitudinal symptom changes across four domains; patients stratified by extreme tertiles into high responders (n=71) and non-responders (n=70); middle tertile excluded (n=70)
3. **Resampling-based single-feature evaluation** — Each of the 149 features assessed across five scoring criteria (Linear SVM, k-NN, neural network, Linear Regression R², information gain), 100 independent iterations of 10 × 80/20 stratified splits; consensus ranking; stable feature identification at ≥70% selection frequency
4. **Multi-feature classification** — Stable feature set compared against full-feature and random baselines using three classifiers (Linear SVM, k-NN, Decision Tree) under 10×10 repeated stratified cross-validation
5. **Phenotypic characterisation** — Univariate between-group differences assessed via Mann–Whitney U tests with Benjamini–Hochberg FDR correction

![Pipeline overview](figures/main/pipeline_workflow.jpg)

---

## Data

Due to privacy and ethical restrictions associated with clinical data, the full original dataset is not publicly released. The repository includes all preprocessing, feature selection, modelling, and figure generation code. Anonymised data files are provided only in accordance with applicable data-sharing constraints.

### Study Cohort

The dataset comprises 301 patients with tick-borne infections evaluated at the Mater Misericordiae University Hospital (Dublin, Ireland). Diagnostic evaluation included TICKPLEX® ELISA serological testing for *Borrelia burgdorferi* sensu lato, *Babesia microti*, *Bartonella henselae*, *Ehrlichia chaffeensis*, and *Rickettsia akari*. Of 301 patients, 140 (46.5%) were seropositive; the remainder were clinically diagnosed based on compatible presentation and exposure history. The final analytic cohort comprised 141 patients (high responders n=71, non-responders n=70) after exclusion of 90 patients with insufficient outcome data.

**Ethical approval:** IRB reference 1/378/1946, Mater Misericordiae University Hospital. All participants provided written informed consent. The study was conducted in accordance with the Declaration of Helsinki.

### Variable Categories

Patients completed anonymised self-reported surveys at baseline (T0, pre-treatment) and follow-up (T2, approximately six months post-treatment). The final baseline feature matrix contains **149 variables** across the following categories:

**Demographics and disease history**
- `Age`: Patient age at baseline
- `Gender`: Patient sex (F / M)
- `Tick bite history`: Binary
- `Time since tick bite (months)`
- `Previous chronic symptoms`: Binary
- `Prior antibiotic treatment`: Binary; prior antibiotic duration (weeks)
- `Employment status`

**Self-reported symptom severity (1–10 numeric rating scales)**
- `Severe fatigue severity` — burden scale (lower = better)
- `Muscle pain severity` — burden scale (lower = better)
- `Overall symptom severity` — well-being scale (higher = better)
- `Mood severity` — well-being scale (higher = better)
- `Total number of symptoms`
- Domain-specific symptom counts: neurological, musculoskeletal, autonomic, and other symptom domains (individual items and aggregated counts)

**Immunological markers**
- `CD3%`: T lymphocyte percentage — reference range 61–84%
- `CD3 Total`: Total T lymphocyte count — reference range 960–2600 cells/µL
- `CD4%`: Helper T cell percentage — reference range 32–60%
- `CD4-Helper`: CD4+ Helper T cell count — reference range 540–1600 cells/µL
- `CD8%`: Cytotoxic T cell percentage — reference range 13–40%
- `CD8-Suppr`: CD8 suppressor count — reference range 270–930 cells/µL
- `H/S ratio`: Helper/suppressor ratio — reference range 0.9–4.5
- NK cells, B cells (CD19+), IgM

**Routine haematology and biochemistry**
- `HgB`: Haemoglobin — reference range 11.5–16.5 g/dL
- `Platelets`: Platelet count — reference range 150–400 ×10⁹/L
- `Neutrophils`: Neutrophil count — reference range 2–8 ×10⁹/L
- `Lymphocytes`: Lymphocyte count — reference range 1–4 ×10⁹/L
- `WCC`: White cell count — reference range 3.5–11 ×10⁹/L
- `CRP`: C-reactive protein — reference range <7 mg/L
- `Iron`: Serum iron — reference range 6–33 µmol/L
- `Transf`: Transferrin — reference range 1.88–3.02 g/dL
- `%Trans sat`: Transferrin saturation — reference range 19–55%
- `Ferritin`, `Folate`, `TSH`, `Rheumatoid factor`

**Serological profile**
- Number of positive markers; number of serological tests performed
- Pathogen-specific seropositivity: *B. burgdorferi*, *Babesia*, *Bartonella*, *Ehrlichia*, *Rickettsia*

### Preprocessing

All preprocessing was performed within each resampling split using training-fold statistics only, to prevent information leakage:
- Missing values imputed using the median (continuous) or mode (categorical) computed from the training partition
- Continuous features standardised to zero mean and unit variance; applied identically to the test fold
- Categorical and binary features were not scaled
- Overall missingness across 149 features: 7.13% (median 3.5% per feature)

---

## Repository Structure

```
.
├── dataset/            # Anonymised dataset (where applicable)
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

Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Python 3.10 is required.

---

## Usage

The analysis pipeline is organised into sequential scripts. Run them in the following order:

**Step 1 — Feature engineering**
```bash
src/01_feature_engineering_pipeline.ipynb
```
Handles raw data ingestion, recoding of categorical variables, aggregation into domain-specific symptom counts, and exclusion of post-baseline measurements. Produces the final 149-variable baseline feature matrix.

**Step 2 — Outcome classification**
```bash
python src/02_outcome_classification.py
```
Derives the composite treatment response score from longitudinal changes across four symptom domains (severe fatigue, muscle pain, overall symptoms, mood) and stratifies patients into high responders, middle responders, and non-responders by tertiles.

**Step 3 — Data preparation**
```bash
python src/03_prepare_data.py
```
Prepares formatted input arrays and stratified train/test splits for downstream modelling.

**Step 4 — Sanity check**
```bash
python src/sanity_check.py
```
Validates data integrity, class balance, and preprocessing correctness before running the main analysis.

**Step 5 — Stability analysis (main)**
```bash
python src/04_stability_top30.py
```
Core of the pipeline. Runs 100 independent resampling iterations, each comprising 10 independent 80/20 stratified train/test splits. Assesses each of the 149 features across five scoring criteria (Linear SVM, k-NN, neural network, Linear Regression R², information gain), computes consensus rankings, and identifies stable features at ≥70% selection frequency (~700,000 evaluations total). Note: requires `data_loading.py`.

To run in background with logging:
```bash
nohup python src/04_stability_top30.py > logs/stability_run_$(date +%Y%m%d_%H%M).log 2>&1 &
echo $! > logs/stability.pid
```

**Step 6 — Multi-feature consensus**
```bash
python src/05_multifeature.py
```
Aggregates feature rankings across methods into a consensus score. Evaluates classification performance across feature set configurations (full set, stable set, random baselines, cumulative top-k sets) using Linear SVM, k-NN, and Decision Tree under 10×10 repeated stratified cross-validation.

**Step 7 — Alternative feature selection methods**
```bash
python src/06_alt_feature_selection.py
python src/07_alt_fs_classification.py
```
Independently applies three embedded methods (LASSO, Elastic Net, Random Forest) under identical resampling conditions. Computes a three-method combined consensus and identifies the robust core feature set selected by ≥3 of 4 approaches.

**Step 8 — Phenotype analysis**
```bash
python src/08_phenotype_analysis.py
```
Characterises clinical phenotypes associated with stable feature subsets. Computes two-sided Mann–Whitney U tests with Benjamini–Hochberg FDR correction (α=0.05) and effect sizes (Cohen's *d*, Common Language Effect Size) for the 22 stable features.

**Step 9 — Sensitivity analyses**
```bash
python src/09_sensitivity_analyses.py
```
Evaluates robustness of findings under alternative stability thresholds (50–90%), alternative hyperparameter specifications, and alternative modelling assumptions.

All experiments are executed with fixed random seeds to ensure full reproducibility.

---

## Results

### Stable Feature Identification

Across 100 resampling iterations, **22 baseline features** demonstrated stable selection frequency (≥70%), corresponding to 14.8% of the original feature space. The highest-ranked features were predominantly somatic symptom measures: severe fatigue severity, neurological symptom count, swollen glands, muscle pain severity, and mood severity. Immunological markers (CD8+, CD3+ T cells), demographic variables (age, employment status), and treatment-related variables were also retained.

Stability metrics confirmed moderate-to-good reproducibility: Nogueira stability index = 0.70, mean pairwise Jaccard similarity = 0.61, Spearman rank concordance *r* = 0.87 (*p* < 0.001).

![Stable features](figures/main/stable_features.jpg)

### Multi-Feature Classification Performance

Models trained on the 22 stable features consistently outperformed those trained on the full 149-feature set and random subsets of equal size across all three classifiers:

| Feature set | Classifier | Accuracy | AUC |
|---|---|---|---|
| Stable (22) | Linear SVM | 0.641 [0.619, 0.663] | 0.708 [0.681, 0.734] |
| Stable (22) | k-NN | 0.646 [0.620, 0.671] | 0.710 [0.681, 0.737] |
| Stable (22) | Decision Tree | 0.635 [0.612, 0.658] | 0.651 [0.626, 0.675] |
| Full (149) | Linear SVM | 0.529 [0.506, 0.553] | 0.534 [0.507, 0.562] |
| Random (22) | Linear SVM | 0.531 [0.521, 0.541] | 0.539 [0.526, 0.552] |

The stable feature set significantly outperformed both baselines across all classifiers (permutation *p* < 0.001; Cohen's *d* = 0.83–1.14). Classification performance plateaued at approximately five top-ranked features and declined gradually beyond 15.

### Robust Core Features

Five features were selected by all four methodologies (Vendrow consensus, LASSO, Elastic Net, Random Forest): **CD8+ T cells (%), severe fatigue severity, overall symptom severity, muscle pain severity, and mood severity**.

### Phenotypic Characterisation

High responders were characterised by elevated somatic symptom burden (higher fatigue, muscle pain, neurological symptoms) coupled with relatively preserved mood at baseline. Non-responders exhibited the inverse pattern: better mood and self-rated wellbeing but lower somatic burden. Seven of the 22 stable features reached individual significance after FDR correction (*q* < 0.05; effect sizes |*d*| = 0.44–0.69).

---

## Key Contributions

- Stability-aware feature selection framework for clinical ML in heterogeneous, small-sample cohorts
- Multi-method consensus ranking that reduces dependence on any single algorithm or sampling choice
- Formal quantification of feature reproducibility using Nogueira stability index, Jaccard similarity, and Spearman rank concordance
- Demonstration that apparent performance advantages can arise from shared regularisation assumptions between selection and classification steps, rather than genuine generalisation
- Fully reproducible pipeline aligned with TRIPOD+AI reporting principles

---

## Reproducibility

- Python 3.10.13
- scikit-learn 1.5.1 · NumPy 1.26.4 · pandas 2.2.2 · SciPy 1.14.1
- Fixed random seeds for all stochastic processes
- Deterministic pipeline execution order
- Full environment specification provided via `requirements.txt`

---

## Licenses

- **Code**: MIT License — see `MIT license` file for details
- **Dataset**: Creative Commons BY-SA 4.0 — see `CC-BY-SA-4.0 license` file for details

---

## Feedback

We welcome questions, suggestions, and contributions. Please open an issue on GitHub if you encounter any problems with the repository or would like to discuss the methodology.

---

## References

1. Xi, D. et al. A Longitudinal Study of a Large Clinical Cohort of Patients with Lyme Disease and Tick-Borne Co-Infections Treated with Combination Antibiotics. *Microorganisms* 2023, 11, 2152. https://doi.org/10.3390/microorganisms11092152

2. Xi, D. et al. Scrutinizing Clinical Biomarkers in a Large Cohort of Patients with Lyme Disease and Other Tick-Borne Infections. *Microorganisms* 2024, 12, 380. https://doi.org/10.3390/microorganisms12020380

3. Garg, K. et al. Biomarker-Based Analysis of Pain in Patients with Tick-Borne Infections before and after Antibiotic Treatment. *Antibiotics* 2024, 13, 693. https://doi.org/10.3390/antibiotics13080693

4. Vendrow, J. et al. Feature Selection from Lyme Disease Patient Survey Using Machine Learning. *Algorithms* 2020, 13, 334. https://doi.org/10.3390/a13120334

5. Nogueira, S., Sechidis, K. & Brown, G. On the stability of feature selection algorithms. *J. Mach. Learn. Res.* 2017, 18, 6345–6398.

6. Hédou, J. et al. Discovery of sparse, reliable omic biomarkers with Stabl. *Nat Biotechnol* 2024, 42, 1581–1593. https://doi.org/10.1038/s41587-023-02033-3

---

## Citation

If you use this repository, please cite the associated manuscript:

> Ciavattini T, Shawky M, Padiolleau-Lefèvre S, De Vuyst F, Avramovic G, Lambert JS. *Stability-aware machine learning identifies reproducible baseline predictors of treatment response in tick-borne illness.* Manuscript under submission.

```bibtex
@misc{Stable-MultiMethod-Features-TBI,
  author    = {Teresa Ciavattini and Marc Shawky and S{\'e}verine Padiolleau-Lef{\`e}vre
               and Florian {De Vuyst} and Gordana Avramovic and John Shearer Lambert},
  title     = {Machine Learning-Driven Biomarker Discovery for Stratifying Treatment
               Response in Tick-Borne Illness},
  year      = {2025},
  publisher = {GitHub},
  journal   = {GitHub repository},
  howpublished = {\url{https://github.com/tciavattini/Stable-MultiMethod-Features-TBI}},
}
```

---

**Funding.** This project is co-funded by the European Union's Horizon Europe research and innovation programme Cofund SOUND.AI under the Marie Skłodowska-Curie Grant Agreement No 101081674, by Hauts-de-France Region (STIMulE, STIP, DiaLyme), and by the Institut des Sciences du Calcul et des Données (ISCD) of Sorbonne University via the Num4Lyme project.

---

**Teresa Ciavattini**  
Université de Technologie de Compiègne / Sorbonne Université – SCAI
